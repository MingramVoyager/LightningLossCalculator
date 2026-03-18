"""
Iowa Environmental Mesonet (IEM) ASOS client.

Fetches METAR present-weather observations from the IEM ASOS archive for
Pueblo Memorial Airport (station PUB / KPUB, ~3 miles from the depot).
When KPUB reports a thunderstorm (TS present in the present-weather field),
lightning is treated as occurring within the shutdown threshold.

Why IEM instead of NCEI:
  NCEI ISD global-hourly files for KPUB (USAF 724640) are either absent
  (HTTP 404) or return zero rows via the ADS query API.  IEM archives
  the same ASOS/METAR observations and exposes a reliable, well-documented
  CSV endpoint used widely in academic meteorology.

API reference:
  https://mesonet.agron.iastate.edu/request/download.phtml

Why this works for the analysis:
  • PUB is 3 miles from the Pueblo Chemical Depot — any thunderstorm
    at the airport is definitively within both the 15-mile and 20-mile
    shutdown thresholds.
  • The 30-minute clear rule is applied by the shutdown engine as normal.
  • Resolution is per-METAR (roughly hourly for ASOS).
"""

import time
from datetime import date, timedelta
from io import StringIO

import pandas as pd
import requests

IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
IEM_STATION  = "PUB"    # Pueblo Memorial Airport — IEM uses 3-letter without K

KPUB_LAT     = 38.2890
KPUB_LON     = -104.4970
KPUB_DIST_MI = 3.0      # approximate distance to depot

_REQUEST_PAUSE = 1.0    # seconds between year fetches


# ── IEM fetch ─────────────────────────────────────────────────────────────────

def _iem_params(year: int) -> dict:
    return {
        "station":     IEM_STATION,
        "data":        "presentwx",
        "year1":       year, "month1":  1,  "day1":  1,  "hour1":  0, "minute1": 0,
        "year2":       year, "month2": 12,  "day2": 31,  "hour2": 23, "minute2": 59,
        "tz":          "UTC",
        "format":      "comma",
        "latlon":      "no",
        "elev":        "no",
        "missing":     "M",
        "trace":       "T",
        "direct":      "yes",
        "report_type": ["1", "2"],   # 1=ASOS routine, 2=ASOS special
    }


def _fetch_year_iem(year: int) -> pd.DataFrame:
    params = _iem_params(year)
    try:
        resp = requests.get(IEM_ASOS_URL, params=params, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"IEM ASOS request failed ({year}): {exc}") from exc
    return _parse_iem_csv(resp.text)


def _parse_iem_csv(text: str) -> pd.DataFrame:
    """
    Parse IEM ASOS response.  IEM prepends comment lines starting with '#'.
    Expected columns: station, valid, presentwx
    """
    if not text or not text.strip():
        return pd.DataFrame()
    lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
    if len(lines) < 2:
        return pd.DataFrame()
    try:
        df = pd.read_csv(StringIO("\n".join(lines)), low_memory=False)
        return df
    except Exception:
        return pd.DataFrame()


# ── Thunderstorm detection ────────────────────────────────────────────────────

def _has_thunderstorm(presentwx: object) -> bool:
    """Return True if the present-weather string contains a thunderstorm code."""
    if pd.isna(presentwx):
        return False
    s = str(presentwx).strip().upper()
    return "TS" in s and s not in ("M", "")


def _parse_ts_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter an IEM ASOS DataFrame to rows with thunderstorm present weather.
    Returns a DataFrame with columns: timestamp_utc, distance_miles.
    """
    if df.empty:
        return pd.DataFrame(columns=["timestamp_utc", "distance_miles"])

    # Locate timestamp column — IEM uses 'valid'
    time_col = next(
        (c for c in df.columns if c.strip().lower() in ("valid", "date", "datetime", "timestamp")),
        None,
    )
    if time_col is None:
        return pd.DataFrame(columns=["timestamp_utc", "distance_miles"])

    # Locate present-weather column
    wx_col = next(
        (c for c in df.columns if "presentwx" in c.lower() or "wxcodes" in c.lower() or "wxcode" in c.lower()),
        None,
    )
    if wx_col is None:
        return pd.DataFrame(columns=["timestamp_utc", "distance_miles"])

    ts_mask = df[wx_col].apply(_has_thunderstorm)
    ts_rows  = df[ts_mask].copy()
    if ts_rows.empty:
        return pd.DataFrame(columns=["timestamp_utc", "distance_miles"])

    ts_rows["timestamp_utc"]  = pd.to_datetime(ts_rows[time_col], utc=True, errors="coerce")
    ts_rows                   = ts_rows.dropna(subset=["timestamp_utc"])
    ts_rows["distance_miles"] = KPUB_DIST_MI
    return ts_rows[["timestamp_utc", "distance_miles"]].reset_index(drop=True)


# ── Diagnostics ───────────────────────────────────────────────────────────────

def probe_api() -> dict:
    """
    Fetch one month of IEM ASOS data (July 2023) for PUB to confirm the
    endpoint is reachable and thunderstorm detection is working.
    """
    result = {
        "station_id":  IEM_STATION,
        "station_msg": f"IEM ASOS station {IEM_STATION} (Pueblo Memorial Airport, ~{KPUB_DIST_MI} mi from depot)",
        "url":         "",
        "status_code": None,
        "raw_text":    "",
        "columns":     [],
        "row_count":   0,
        "ts_hours":    0,
        "error":       None,
    }

    # Probe: fetch July 2023 only (peak CO storm month)
    params = {
        "station":     IEM_STATION,
        "data":        "presentwx",
        "year1":  2023, "month1":  7, "day1":  1, "hour1":  0, "minute1": 0,
        "year2":  2023, "month2":  7, "day2": 31, "hour2": 23, "minute2": 59,
        "tz":          "UTC",
        "format":      "comma",
        "latlon":      "no",
        "elev":        "no",
        "missing":     "M",
        "trace":       "T",
        "direct":      "yes",
        "report_type": ["1", "2"],
    }
    req = requests.Request("GET", IEM_ASOS_URL, params=params).prepare()
    result["url"] = req.url

    try:
        resp = requests.get(IEM_ASOS_URL, params=params, timeout=60)
        result["status_code"] = resp.status_code
        result["raw_text"]    = resp.text[:2500]
        resp.raise_for_status()

        df = _parse_iem_csv(resp.text)
        result["columns"]   = list(df.columns)
        result["row_count"] = len(df)

        ts = _parse_ts_hours(df)
        result["ts_hours"] = len(ts)

    except Exception as exc:
        result["error"] = str(exc)

    return result


# ── Public fetch entry point ──────────────────────────────────────────────────

def fetch_strikes(
    start_year: int,
    end_year:   int,
    radius_miles: float = 25.0,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Fetch KPUB thunderstorm observation hours for the given year range
    via the IEM ASOS archive.

    Returns DataFrame: timestamp_utc, latitude, longitude, distance_miles
    """
    years  = list(range(start_year, end_year + 1))
    total  = len(years)
    frames: list[pd.DataFrame] = []

    for i, year in enumerate(years):
        if progress_callback:
            progress_callback(i, total)

        raw = _fetch_year_iem(year)
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
