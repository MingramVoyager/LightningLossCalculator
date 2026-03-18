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
  • Resolution is hourly; minimum detectable event is ~1 hour.

Limitation vs NLDN strike data:
  • Cannot resolve storms at exactly 15-vs-20 miles.
  • Conservative estimate: any thunderstorm at KPUB = shutdown.

Station lookup:
  Station IDs are resolved from the NCEI ISD history file at startup so
  we always use the correct USAF+WBAN regardless of format changes.

Station ID format for NCEI ADS:
  The global-hourly dataset requires USAF-WBAN with a hyphen, e.g.
  "724640-23058".  The 11-digit concatenated form (no hyphen) returns
  HTTP 200 but zero rows.
"""

import time
from datetime import date, timedelta
from io import StringIO

import pandas as pd
import requests

NCEI_ADS_BASE   = "https://www.ncei.noaa.gov/access/services/data/v1"
ISD_HISTORY_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"

KPUB_ICAO    = "KPUB"
KPUB_LAT     = 38.2890
KPUB_LON     = -104.4970
KPUB_DIST_MI = 3.0

_CHUNK_DAYS    = 90
_REQUEST_PAUSE = 0.5


# ── Station ID lookup ─────────────────────────────────────────────────────────

def lookup_station_id(icao: str) -> tuple[str | None, str | None, str]:
    """
    Look up the NCEI ISD station for an ICAO call sign by fetching
    isd-history.csv.  Returns (usaf, wban, status_message).

    The returned usaf and wban are zero-padded strings (6 and 5 chars).
    If multiple rows match, the one with the most recent END date is used.
    If lookup fails, usaf and wban are None.
    """
    try:
        resp = requests.get(ISD_HISTORY_URL, timeout=60)
        resp.raise_for_status()
    except Exception as exc:
        return None, None, f"Failed to fetch ISD history: {exc}"

    try:
        df = pd.read_csv(StringIO(resp.text), dtype=str)
    except Exception as exc:
        return None, None, f"Failed to parse ISD history: {exc}"

    # Locate the call-sign column
    call_col = next(
        (c for c in df.columns if c.strip().upper() in ("CALL", "CALL_SIGN", "ICAO")),
        None,
    )
    if call_col is None:
        return None, None, f"No CALL column found. Columns: {list(df.columns)}"

    df[call_col] = df[call_col].astype(str).str.strip().str.upper()
    match = df[df[call_col] == icao.upper()].copy()

    if match.empty:
        return None, None, f"No ISD station found for ICAO '{icao}'."

    # Most-recent END date first
    if "END" in match.columns:
        match["END"] = pd.to_numeric(match["END"], errors="coerce")
        match = match.sort_values("END", ascending=False)

    row = match.iloc[0]
    try:
        usaf = str(int(float(row["USAF"]))).zfill(6)
        wban = str(int(float(row["WBAN"]))).zfill(5)
    except Exception as exc:
        return None, None, f"Could not parse USAF/WBAN: {dict(row)} — {exc}"

    name = str(row.get("STATION NAME", row.get("STATION_NAME", ""))).strip()
    lat  = row.get("LAT", "")
    lon  = row.get("LON", "")
    end  = row.get("END", "")
    msg  = (f"{name} | USAF {usaf}  WBAN {wban} | "
            f"Lat {lat}  Lon {lon} | Active through {end}")
    return usaf, wban, msg


def _station_param(usaf: str, wban: str) -> str:
    """Format the station ID as required by NCEI ADS: 'USAF-WBAN'."""
    return f"{usaf}-{wban}"


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _parse_raw_csv(text: str) -> pd.DataFrame:
    if not text or not text.strip():
        return pd.DataFrame()
    lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
    if len(lines) < 2:
        return pd.DataFrame()
    try:
        return pd.read_csv(StringIO("\n".join(lines)), low_memory=False)
    except Exception:
        return pd.DataFrame()


def _fetch_chunk(station_param: str, start: date, end: date) -> pd.DataFrame:
    params = {
        "dataset":   "global-hourly",
        "stations":  station_param,
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


# ── Present-weather thunderstorm detection ────────────────────────────────────

_TS_WMO_CODES        = {"17","29","91","92","93","94","95","96","97","98","99"}
_TS_METAR_SUBSTRINGS = ("TS",)


def _row_has_thunderstorm(row: pd.Series) -> bool:
    import re
    for col, val in row.items():
        if pd.isna(val):
            continue
        s = str(val).strip()
        if not s or s in ("nan", "None", "9", "99999", "999999"):
            continue
        col_lo = col.lower()
        if col_lo.startswith(("mw", "aw", "au")):
            code = s.split(",")[0].strip()
            if code in _TS_WMO_CODES:
                return True
        if any(sub in s for sub in _TS_METAR_SUBSTRINGS):
            if re.search(r'\bTS\b|TS[A-Z]', s):
                return True
    return False


def _parse_ts_hours(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["timestamp_utc", "distance_miles"])

    date_col = next(
        (c for c in raw.columns if c.strip().upper() in ("DATE", "DATETIME", "DATE_TIME", "TIMESTAMP")),
        None,
    )
    if date_col is None:
        candidates = [c for c in raw.columns if "date" in c.lower() or "time" in c.lower()]
        date_col = candidates[0] if candidates else None
    if date_col is None:
        return pd.DataFrame(columns=["timestamp_utc", "distance_miles"])

    ts_mask = raw.apply(_row_has_thunderstorm, axis=1)
    ts_rows  = raw[ts_mask].copy()
    if ts_rows.empty:
        return pd.DataFrame(columns=["timestamp_utc", "distance_miles"])

    ts_rows["timestamp_utc"]  = pd.to_datetime(ts_rows[date_col], utc=True, errors="coerce")
    ts_rows                   = ts_rows.dropna(subset=["timestamp_utc"])
    ts_rows["distance_miles"] = KPUB_DIST_MI
    return ts_rows[["timestamp_utc", "distance_miles"]].reset_index(drop=True)


# ── Diagnostics ───────────────────────────────────────────────────────────────

def probe_api() -> dict:
    """
    Step 1: resolve KPUB station ID from isd-history.csv.
    Step 2: fetch a 5-day July 2023 test window with both ID formats.
    Returns full diagnostic info.
    """
    result = {
        "usaf":         None,
        "wban":         None,
        "station_id":   None,   # hyphenated USAF-WBAN
        "station_msg":  "",
        "url":          "",
        "status_code":  None,
        "raw_text":     "",
        "columns":      [],
        "row_count":    0,
        "ts_hours":     0,
        "error":        None,
    }

    # Step 1 — resolve station ID
    usaf, wban, msg = lookup_station_id(KPUB_ICAO)
    result["station_msg"] = msg

    if usaf is None:
        result["error"] = msg
        return result

    result["usaf"]       = usaf
    result["wban"]       = wban
    result["station_id"] = f"{usaf}-{wban}"

    # Step 2 — test fetch with the hyphenated station ID
    test_start  = date(2023, 7, 4)
    test_end    = date(2023, 7, 8)
    station_param = _station_param(usaf, wban)
    params = {
        "dataset":   "global-hourly",
        "stations":  station_param,
        "startDate": test_start.isoformat(),
        "endDate":   test_end.isoformat(),
        "format":    "csv",
    }
    req = requests.Request("GET", NCEI_ADS_BASE, params=params).prepare()
    result["url"] = req.url

    try:
        resp = requests.get(NCEI_ADS_BASE, params=params, timeout=30)
        result["status_code"] = resp.status_code
        result["raw_text"]    = resp.text[:2500]
        resp.raise_for_status()

        raw = _parse_raw_csv(resp.text)
        result["columns"]   = list(raw.columns)
        result["row_count"] = len(raw)

        ts = _parse_ts_hours(raw)
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
    Fetch KPUB thunderstorm hours for the given year range.
    Station ID is resolved from isd-history.csv on every call.
    """
    usaf, wban, msg = lookup_station_id(KPUB_ICAO)
    if usaf is None:
        raise RuntimeError(f"Could not resolve KPUB station ID: {msg}")

    station_param = _station_param(usaf, wban)

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
        raw = _fetch_chunk(station_param, chunk_start, chunk_end)
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
