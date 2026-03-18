"""
NOAA NCEI lightning data client.

Primary:  NCEI Access Data Service (v1) — newer endpoint, replaces broken SWDI webservice v2
Fallback: NCEI SWDI webservice v2 (currently broken as of 2026 — DNS failure on their end)

API reference:
  https://www.ncei.noaa.gov/access/services/data/v1  (primary)
  https://www.ncei.noaa.gov/swdi/#Webservices        (legacy, broken)
"""

import math
import time
from datetime import date, timedelta
from io import StringIO

import pandas as pd
import requests

# ── Site constants ────────────────────────────────────────────────────────────
DEPOT_LAT = 38.2710
DEPOT_LON = -104.3390

# NCEI Access Data Service — current working endpoint
NCEI_ADS_BASE  = "https://www.ncei.noaa.gov/access/services/data/v1"
# SWDI webservice v2 — kept for reference; returns 500 (broken infrastructure)
SWDI_BASE_LEGACY = "https://www.ncei.noaa.gov/swdi/webservice/v2/nldn.csv"

_CHUNK_DAYS    = 30    # days per request chunk
_REQUEST_PAUSE = 0.75  # seconds between requests


# ── Geometry ──────────────────────────────────────────────────────────────────

def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 3958.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _bbox_for_radius(lat: float, lon: float, radius_miles: float) -> dict:
    margin = radius_miles * 1.10
    dlat   = margin / 69.0
    dlon   = margin / (69.0 * math.cos(math.radians(lat)))
    return {
        "lat_min": lat - dlat, "lat_max": lat + dlat,
        "lon_min": lon - dlon, "lon_max": lon + dlon,
    }


# ── NCEI Access Data Service (primary) ───────────────────────────────────────

def _ads_params(start: date, end: date, bbox: dict) -> dict:
    """Build query parameters for the NCEI Access Data Service."""
    return {
        "dataset":     "nldn",
        "startDate":   start.isoformat(),
        "endDate":     end.isoformat(),
        "boundingBox": (f"{bbox['lat_min']:.4f},{bbox['lon_min']:.4f},"
                        f"{bbox['lat_max']:.4f},{bbox['lon_max']:.4f}"),
        "format":      "csv",
    }


def _parse_ads_response(text: str) -> pd.DataFrame:
    """
    Parse NCEI ADS CSV response. ADS responses may include comment lines
    starting with '#' before the header and data rows.
    """
    if not text or not text.strip():
        return pd.DataFrame()

    lines = text.splitlines()
    data_lines = [ln for ln in lines if not ln.startswith("#")]

    # Check for no-data signals in comments
    comments = " ".join(ln for ln in lines if ln.startswith("#")).lower()
    if any(phrase in comments for phrase in ("no data", "no records", "0 records")):
        return pd.DataFrame()

    if len(data_lines) < 2:   # header only
        return pd.DataFrame()

    try:
        return pd.read_csv(StringIO("\n".join(data_lines)))
    except Exception:
        return pd.DataFrame()


def _fetch_chunk_ads(start: date, end: date, bbox: dict) -> pd.DataFrame:
    params = _ads_params(start, end, bbox)
    try:
        resp = requests.get(NCEI_ADS_BASE, params=params, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"NCEI ADS request failed ({start}–{end}): {exc}") from exc
    return _parse_ads_response(resp.text)


# ── Column normalisation ──────────────────────────────────────────────────────

def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map raw API column names → internal standard names.

    SWDI webservice v2 used:  ZTIME, LAT, LON
    NCEI ADS may use:         DATE, LATITUDE, LONGITUDE  (TBD — check probe output)
    We try all known variants.
    """
    df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]

    # Timestamp
    ts_map = {"ztime": "timestamp_utc", "date": "timestamp_utc",
               "datetime": "timestamp_utc", "date_time": "timestamp_utc",
               "time": "timestamp_utc", "begindate": "timestamp_utc",
               "begin_date_time": "timestamp_utc"}
    for src, dst in ts_map.items():
        if src in df.columns and "timestamp_utc" not in df.columns:
            df = df.rename(columns={src: dst})
    if "timestamp_utc" not in df.columns:
        candidates = [c for c in df.columns if "time" in c or "date" in c]
        if candidates:
            df = df.rename(columns={candidates[0]: "timestamp_utc"})

    # Latitude
    for src in ("lat", "latitude", "begin_lat", "begin_latitude"):
        if src in df.columns and "latitude" not in df.columns:
            df = df.rename(columns={src: "latitude"})

    # Longitude
    for src in ("lon", "long", "longitude", "begin_lon", "begin_longitude"):
        if src in df.columns and "longitude" not in df.columns:
            df = df.rename(columns={src: "longitude"})

    return df


# ── Diagnostics ───────────────────────────────────────────────────────────────

def probe_api() -> dict:
    """
    Test the NCEI Access Data Service with a small 3-day request.
    Returns a dict with all diagnostic info for display in the UI.
    """
    bbox  = _bbox_for_radius(DEPOT_LAT, DEPOT_LON, 25.0)

    # Use a fixed historical window known to have summer thunderstorms in CO
    test_start = date(2023, 7, 1)
    test_end   = date(2023, 7, 4)

    params = _ads_params(test_start, test_end, bbox)
    req    = requests.Request("GET", NCEI_ADS_BASE, params=params).prepare()

    result = {
        "url":         req.url,
        "status_code": None,
        "raw_text":    "",
        "columns":     [],
        "row_count":   0,
        "error":       None,
    }

    try:
        resp = requests.get(NCEI_ADS_BASE, params=params, timeout=30)
        result["status_code"] = resp.status_code
        result["raw_text"]    = resp.text[:2000]
        resp.raise_for_status()

        df = _parse_ads_response(resp.text)
        result["columns"]   = list(df.columns)
        result["row_count"] = len(df)

    except Exception as exc:
        result["error"] = str(exc)

    return result


# ── Public fetch entry point ──────────────────────────────────────────────────

def fetch_strikes(
    start_year: int,
    end_year: int,
    radius_miles: float = 25.0,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Fetch all cloud-to-ground lightning strikes within radius_miles of the
    Pueblo Chemical Depot for the given year range via NCEI Access Data Service.

    Returns DataFrame: timestamp_utc, latitude, longitude, distance_miles
    """
    bbox = _bbox_for_radius(DEPOT_LAT, DEPOT_LON, radius_miles)

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

        raw = _fetch_chunk_ads(chunk_start, chunk_end, bbox)
        if not raw.empty:
            frames.append(raw)

        time.sleep(_REQUEST_PAUSE)

    if progress_callback:
        progress_callback(total, total)

    if not frames:
        return pd.DataFrame(columns=["timestamp_utc", "latitude", "longitude", "distance_miles"])

    df = pd.concat(frames, ignore_index=True)
    df = _normalise_columns(df)

    required = {"timestamp_utc", "latitude", "longitude"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise RuntimeError(
            f"NCEI response missing expected columns: {missing}. "
            f"Columns returned: {list(df.columns)}"
        )

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"])
    df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])

    df["distance_miles"] = df.apply(
        lambda r: haversine_miles(DEPOT_LAT, DEPOT_LON, r["latitude"], r["longitude"]),
        axis=1,
    )
    df = df[df["distance_miles"] <= radius_miles].copy()

    return (
        df[["timestamp_utc", "latitude", "longitude", "distance_miles"]]
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )
