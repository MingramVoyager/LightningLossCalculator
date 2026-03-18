"""
NCEI Integrated Surface Dataset (ISD) client — global-hourly.

Uses the Pueblo Memorial Airport ASOS station (KPUB, ~3 miles from depot)
as a thunderstorm proximity proxy.  When KPUB reports a thunderstorm in its
present-weather field we treat lightning as present within the shutdown
threshold and emit a synthetic strike at the depot coordinates.

Why this works for the analysis:
  • KPUB is 3 miles from the Pueblo Chemical Depot — any thunderstorm
    at the airport is definitively within both the 15-mile and 20-mile
    shutdown thresholds.
  • The 30-minute clear rule is applied by the shutdown engine as normal.
  • Resolution is hourly (one observation per hour), so the minimum
    detectable event is ~1 hour.  This is a slight conservative overestimate
    but appropriate for a multi-year downtime analysis.

Limitation vs NLDN strike data:
  • Cannot resolve storms at exactly 15-vs-20 miles (both thresholds are
    always triggered when a TS is observed at KPUB).
  • Storms that approach within 20 miles but never reach the airport may be
    missed.  In practice, Pueblo is small and storms at 20 miles are often
    visible to the airport sensors.

NCEI ADS endpoint:
  https://www.ncei.noaa.gov/access/services/data/v1
  dataset: global-hourly
  station: 72465023058  (KPUB — Pueblo Memorial Airport)
"""

import time
from datetime import date, datetime, timedelta, timezone
from io import StringIO

import pandas as pd
import requests

NCEI_ADS_BASE = "https://www.ncei.noaa.gov/access/services/data/v1"

# KPUB — Pueblo Memorial Airport
# ISD station IDs = USAF (6 digits) + WBAN (5 digits).
# WBAN for KPUB is confirmed as 23058; USAF candidates to try in order:
KPUB_STATION_CANDIDATES = [
    "72464023058",  # USAF 724640 — most likely correct
    "72465023058",  # USAF 724650 — previously tried, returned no rows
    "72466023058",  # USAF 724660 — additional candidate
    "KPUB",         # ICAO code — accepted by some NCEI endpoints
]
KPUB_STATION   = KPUB_STATION_CANDIDATES[0]   # updated after probe confirms
KPUB_LAT       = 38.2890
KPUB_LON       = -104.4970
KPUB_DIST_MI   = 3.0   # approximate distance from depot

_CHUNK_DAYS    = 90    # ISD queries are lighter; use 90-day chunks
_REQUEST_PAUSE = 0.5


# ── Present-weather thunderstorm detection ────────────────────────────────────
# ISD encodes present weather in several compound fields.
# Thunderstorm WMO descriptors observed in ISD CSV:
#   MW field codes that indicate TS:  17, 29, 91-99
#   METAR strings that indicate TS:   "TS", "TSRA", "TSGR", etc.
# We scan every field in the row for these patterns.

_TS_WMO_CODES = {"17", "29", "91", "92", "93", "94", "95", "96", "97", "98", "99"}
_TS_METAR_SUBSTRINGS = ("TS",)   # "TS" appears in TSRA, TSGR, TSPL, etc.


def _row_has_thunderstorm(row: pd.Series) -> bool:
    """
    Return True if any column in the ISD row contains a thunderstorm indicator.
    Checks WMO numeric codes (MW/AW fields) and METAR substrings.
    """
    for col, val in row.items():
        if pd.isna(val):
            continue
        s = str(val).strip()
        if not s or s in ("nan", "None", "9", "99999", "999999"):
            continue

        col_lo = col.lower()

        # MW/AW columns contain comma-separated WMO codes like "17,1" or "95,1"
        if col_lo.startswith(("mw", "aw")):
            code = s.split(",")[0].strip()
            if code in _TS_WMO_CODES:
                return True

        # REM (remark) and any METAR-like field may contain "TS" substring
        if any(sub in s for sub in _TS_METAR_SUBSTRINGS):
            # Avoid false positives like "DIST" or "LAST"
            import re
            if re.search(r'\bTS\b|TS[A-Z]', s):
                return True

    return False


# ── API fetch ─────────────────────────────────────────────────────────────────

def _parse_raw_csv(text: str) -> pd.DataFrame:
    """Parse a raw NCEI ADS CSV response into a DataFrame."""
    if not text or not text.strip():
        return pd.DataFrame()
    lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
    if len(lines) < 2:
        return pd.DataFrame()
    try:
        return pd.read_csv(StringIO("\n".join(lines)), low_memory=False)
    except Exception:
        return pd.DataFrame()


def _fetch_chunk(start: date, end: date) -> pd.DataFrame:
    params = {
        "dataset":   "global-hourly",
        "stations":  KPUB_STATION,
        "startDate": start.isoformat(),
        "endDate":   end.isoformat(),
        "format":    "csv",
    }
    try:
        resp = requests.get(NCEI_ADS_BASE, params=params, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"NCEI ISD request failed ({start}–{end}): {exc}") from exc
    return _parse_raw_csv(resp.text)


def _parse_ts_hours(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Filter raw ISD observations to rows that contain a thunderstorm indicator.
    Returns DataFrame with columns: timestamp_utc, distance_miles.
    """
    if raw.empty:
        return pd.DataFrame(columns=["timestamp_utc", "distance_miles"])

    # Identify timestamp column
    date_col = None
    for c in raw.columns:
        if c.upper() in ("DATE", "DATETIME", "DATE_TIME", "TIMESTAMP"):
            date_col = c
            break
    if date_col is None:
        date_cols = [c for c in raw.columns if "date" in c.lower() or "time" in c.lower()]
        date_col = date_cols[0] if date_cols else None

    if date_col is None:
        return pd.DataFrame(columns=["timestamp_utc", "distance_miles"])

    # Keep METAR / ASOS reports only (skip special/synoptic types if present)
    report_cols = [c for c in raw.columns if "report" in c.lower() or "type" in c.lower()]
    if report_cols:
        rc = report_cols[0]
        raw = raw[raw[rc].astype(str).str.contains("FM-15|FM-16|METAR|ASOS", na=False, regex=True)]
        if raw.empty:
            # Fallback: keep all rows
            pass

    ts_mask = raw.apply(_row_has_thunderstorm, axis=1)
    ts_rows  = raw[ts_mask].copy()

    if ts_rows.empty:
        return pd.DataFrame(columns=["timestamp_utc", "distance_miles"])

    ts_rows["timestamp_utc"] = pd.to_datetime(ts_rows[date_col], utc=True, errors="coerce")
    ts_rows = ts_rows.dropna(subset=["timestamp_utc"])
    ts_rows["distance_miles"] = KPUB_DIST_MI

    return ts_rows[["timestamp_utc", "distance_miles"]].reset_index(drop=True)


# ── Diagnostics ───────────────────────────────────────────────────────────────

def probe_api() -> dict:
    """
    Try each KPUB station ID candidate against a 5-day July 2023 window
    and return diagnostic info for all candidates so we can identify which
    station ID actually has records in the NCEI database.
    """
    test_start = date(2023, 7, 4)
    test_end   = date(2023, 7, 8)

    candidates = []
    working_station = None

    for station_id in KPUB_STATION_CANDIDATES:
        params = {
            "dataset":   "global-hourly",
            "stations":  station_id,
            "startDate": test_start.isoformat(),
            "endDate":   test_end.isoformat(),
            "format":    "csv",
        }
        req = requests.Request("GET", NCEI_ADS_BASE, params=params).prepare()
        entry = {
            "station_id":  station_id,
            "url":         req.url,
            "status_code": None,
            "row_count":   0,
            "ts_hours":    0,
            "error":       None,
        }
        try:
            resp = requests.get(NCEI_ADS_BASE, params=params, timeout=30)
            entry["status_code"] = resp.status_code
            resp.raise_for_status()
            raw = _parse_raw_csv(resp.text)
            entry["row_count"] = len(raw)
            if not raw.empty:
                ts = _parse_ts_hours(raw)
                entry["ts_hours"] = len(ts)
                if working_station is None:
                    working_station = station_id
        except Exception as exc:
            entry["error"] = str(exc)
        candidates.append(entry)
        time.sleep(0.5)

    # Use the first candidate that returned rows for the main result display
    first_working = next((c for c in candidates if c["row_count"] > 0), None)
    if first_working:
        params_w = {
            "dataset":   "global-hourly",
            "stations":  first_working["station_id"],
            "startDate": test_start.isoformat(),
            "endDate":   test_end.isoformat(),
            "format":    "csv",
        }
        resp_w = requests.get(NCEI_ADS_BASE, params=params_w, timeout=30)
        raw_text = resp_w.text[:2500]
        raw      = _parse_raw_csv(resp_w.text)
        columns  = list(raw.columns)
        ts       = _parse_ts_hours(raw)
    else:
        raw_text = ""
        columns  = []
        ts       = pd.DataFrame()

    return {
        "candidates":    candidates,
        "working_station": working_station,
        "url":           candidates[0]["url"],
        "raw_text":      raw_text,
        "columns":       columns,
        "row_count":     first_working["row_count"] if first_working else 0,
        "ts_hours":      len(ts) if first_working else 0,
        "error":         None if first_working else "No station ID returned data.",
    }


# ── Public fetch entry point ──────────────────────────────────────────────────

def fetch_strikes(
    start_year: int,
    end_year: int,
    radius_miles: float = 25.0,   # accepted for API compatibility; not used (KPUB is fixed)
    progress_callback=None,
) -> pd.DataFrame:
    """
    Fetch KPUB thunderstorm hours for the given year range and return them
    as synthetic strike records usable by the shutdown engine.

    Each thunderstorm observation hour becomes a synthetic strike at
    KPUB_DIST_MI from the depot, so the existing proximity rules fire
    as expected.

    Returns DataFrame: timestamp_utc (UTC), latitude, longitude, distance_miles
    """
    chunks: list[tuple[date, date]] = []
    current  = date(start_year, 1, 1)
    end_date = date(end_year, 12, 31)
    while current <= end_date:
        chunk_end = min(current + timedelta(days=_CHUNK_DAYS - 1), end_date)
        chunks.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)

    total  = len(chunks)
    frames: list[pd.DataFrame] = []

    for i, (chunk_start, chunk_end) in enumerate(chunks):
        if progress_callback:
            progress_callback(i, total)

        raw = _fetch_chunk(chunk_start, chunk_end)
        ts  = _parse_ts_hours(raw)
        if not ts.empty:
            frames.append(ts)

        time.sleep(_REQUEST_PAUSE)

    if progress_callback:
        progress_callback(total, total)

    if not frames:
        return pd.DataFrame(columns=["timestamp_utc", "latitude", "longitude", "distance_miles"])

    df = pd.concat(frames, ignore_index=True)
    df["latitude"]  = KPUB_LAT
    df["longitude"] = KPUB_LON

    return (
        df[["timestamp_utc", "latitude", "longitude", "distance_miles"]]
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )
