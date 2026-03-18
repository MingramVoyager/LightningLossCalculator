"""
Shutdown state-machine for lightning-driven operational downtime.

Rules implemented
-----------------
  • Shutdown begins when ANY strike occurs within warn_miles (default 20 mi).
  • Shutdown is considered "full" when ANY strike occurs within shutdown_miles (default 15 mi).
    Both phases count as lost production time per user requirements.
  • Operations may resume only after clear_minutes (default 30) have elapsed
    with NO strike within warn_miles.
  • Loss hours are counted only within the configured operating window
    (start_hour to end_hour, local Mountain Time).

Algorithm
---------
Rather than iterating minute-by-minute, this engine:
  1. Selects strikes within warn_miles from the full dataset.
  2. Groups consecutive strikes into "storm events" where the gap between
     any two successive strikes is ≤ clear_minutes.
  3. Each storm event spans [first_strike, last_strike + clear_minutes].
  4. Intersects each event interval with the operating window for that day.
  5. Sums the overlapping minutes to produce loss hours.

This is O(n log n) on the number of qualifying strikes and runs in
milliseconds even for 15+ years of data.
"""

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytz

MOUNTAIN = pytz.timezone("America/Denver")


# ── Public entry point ────────────────────────────────────────────────────────

def compute_loss(
    strikes_utc: pd.DataFrame,
    warn_miles: float = 20.0,
    shutdown_miles: float = 15.0,
    clear_minutes: int = 30,
    start_hour: int = 0,
    end_hour: int = 24,
) -> pd.DataFrame:
    """
    Compute daily loss hours from a DataFrame of lightning strikes.

    Parameters
    ----------
    strikes_utc : pd.DataFrame
        Must contain columns: timestamp_utc (tz-aware UTC), distance_miles.
    warn_miles : float
        Distance threshold that initiates (and must clear) shutdown.
    shutdown_miles : float
        Distance threshold for full operational shutdown (informational —
        both phases count as lost time per site rules).
    clear_minutes : int
        Minutes after the last warn-zone strike before operations resume.
    start_hour : int
        Start of operating window in local Mountain Time (0-23).
    end_hour : int
        End of operating window in local Mountain Time (1-24, exclusive).
        Use 24 for midnight end-of-day (full 24-hour operation).

    Returns
    -------
    pd.DataFrame with columns:
        date            (datetime.date)
        loss_hours      (float)
        shutdown_events (int)     — number of discrete storm events that day
        max_event_min   (int)     — longest single event in minutes (within window)
        year            (int)
        month           (int)
    """
    if strikes_utc.empty:
        return _empty_result()

    # Convert all timestamps to Mountain local time for window logic
    df = strikes_utc.copy()
    df["ts_local"] = df["timestamp_utc"].dt.tz_convert(MOUNTAIN)

    # Keep only strikes within the warn radius (these are the ones that
    # trigger and reset the shutdown clock)
    warn_df = df[df["distance_miles"] <= warn_miles].copy()
    warn_df = warn_df.sort_values("ts_local").reset_index(drop=True)

    if warn_df.empty:
        return _empty_result()

    # ── Group strikes into storm events ──────────────────────────────────────
    # Two consecutive strikes belong to the same event if the gap is ≤ clear_minutes.
    clear_td = timedelta(minutes=clear_minutes)
    events: list[tuple[datetime, datetime]] = []  # (event_start, event_end_local)

    evt_start = warn_df.loc[0, "ts_local"]
    evt_last  = warn_df.loc[0, "ts_local"]

    for _, row in warn_df.iloc[1:].iterrows():
        t = row["ts_local"]
        if t - evt_last > clear_td:
            # Close current event
            events.append((evt_start, evt_last + clear_td))
            evt_start = t
        evt_last = t

    events.append((evt_start, evt_last + clear_td))

    # ── Intersect events with operating windows, per day ─────────────────────
    results: dict[date, dict] = {}

    for evt_start, evt_end in events:
        # An event may span midnight — check each calendar day it touches
        day_cursor = evt_start.date()
        while day_cursor <= evt_end.date():
            window_start, window_end = _operating_window(day_cursor, start_hour, end_hour)
            overlap_min = _overlap_minutes(evt_start, evt_end, window_start, window_end)

            if overlap_min > 0:
                rec = results.setdefault(day_cursor, {"loss_min": 0, "events": 0, "max_min": 0})
                rec["loss_min"] += overlap_min
                rec["events"]   += 1
                rec["max_min"]   = max(rec["max_min"], overlap_min)

            day_cursor += timedelta(days=1)

    # ── Build output DataFrame ────────────────────────────────────────────────
    if not results:
        return _empty_result()

    rows = []
    for d, rec in sorted(results.items()):
        rows.append(
            {
                "date":            d,
                "loss_hours":      round(rec["loss_min"] / 60, 4),
                "shutdown_events": rec["events"],
                "max_event_min":   rec["max_min"],
                "year":            d.year,
                "month":           d.month,
            }
        )

    return pd.DataFrame(rows)


# ── Aggregation helpers ───────────────────────────────────────────────────────

def yearly_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """Group daily loss into annual totals."""
    if daily.empty:
        return pd.DataFrame(columns=["year", "loss_hours", "shutdown_events", "days_affected"])

    return (
        daily.groupby("year")
        .agg(
            loss_hours=("loss_hours", "sum"),
            shutdown_events=("shutdown_events", "sum"),
            days_affected=("date", "count"),
        )
        .reset_index()
    )


def monthly_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """Group daily loss into year × month totals."""
    if daily.empty:
        return pd.DataFrame(columns=["year", "month", "loss_hours", "shutdown_events"])

    return (
        daily.groupby(["year", "month"])
        .agg(
            loss_hours=("loss_hours", "sum"),
            shutdown_events=("shutdown_events", "sum"),
        )
        .reset_index()
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _operating_window(d: date, start_hour: int, end_hour: int):
    """
    Return (window_start, window_end) as tz-aware Mountain Time datetimes
    for calendar day d.  end_hour=24 means midnight at the end of the day.
    """
    naive_start = datetime(d.year, d.month, d.day, start_hour % 24, 0, 0)
    if end_hour == 24:
        naive_end = datetime(d.year, d.month, d.day, 23, 59, 59) + timedelta(seconds=1)
    else:
        naive_end = datetime(d.year, d.month, d.day, end_hour % 24, 0, 0)

    # Localise to Mountain Time (handles DST automatically)
    w_start = MOUNTAIN.localize(naive_start, is_dst=None) if False else MOUNTAIN.localize(naive_start)
    w_end   = MOUNTAIN.localize(naive_end)
    return w_start, w_end


def _overlap_minutes(a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> int:
    """Minutes of overlap between interval [a_start, a_end) and [b_start, b_end)."""
    overlap_start = max(a_start, b_start)
    overlap_end   = min(a_end, b_end)
    delta = overlap_end - overlap_start
    if delta.total_seconds() <= 0:
        return 0
    return int(delta.total_seconds() / 60)


def _empty_result() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["date", "loss_hours", "shutdown_events", "max_event_min", "year", "month"]
    )
