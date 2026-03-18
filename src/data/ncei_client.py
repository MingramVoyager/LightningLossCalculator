"""
NOAA NCEI Severe Weather Data Inventory (SWDI) client.

Fetches cloud-to-ground lightning strike records for a bounding box around
the Pueblo Chemical Depot from the NLDN dataset.

API reference: https://www.ncei.noaa.gov/swdi/#Webservices
Endpoint:      https://www.ncei.noaa.gov/swdi/webservice/v2/nldn.csv
"""

import math
import time
from datetime import date, timedelta
from io import StringIO

import pandas as pd
import requests

# ── Site constants ───────────────────────────────────────────────────────────
DEPOT_LAT = 38.2710
DEPOT_LON = -104.3390

SWDI_BASE = "https://www.ncei.noaa.gov/swdi/webservice/v2/nldn.csv"

# Fetch in monthly chunks to stay within SWDI record limits per request
_CHUNK_DAYS = 30
# Pause between requests (seconds) — be a good API citizen
_REQUEST_PAUSE = 0.5


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in miles between two lat/lon points."""
    R = 3958.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _bbox_for_radius(lat: float, lon: float, radius_miles: float) -> dict:
    """
    Compute a bounding box that fully contains a circle of radius_miles.
    Adds 10% margin so edge storms are never clipped before the haversine filter.
    """
    margin = radius_miles * 1.10
    dlat = margin / 69.0
    dlon = margin / (69.0 * math.cos(math.radians(lat)))
    return {
        "lat_min": lat - dlat,
        "lat_max": lat + dlat,
        "lon_min": lon - dlon,
        "lon_max": lon + dlon,
    }


def probe_api(days: int = 3) -> dict:
    """
    Make a small test request and return diagnostic info:
      status_code, url, raw_text (first 1000 chars), columns (if parsed), error
    Used by the UI to verify the API is reachable and returning expected data.
    """
    bbox = _bbox_for_radius(DEPOT_LAT, DEPOT_LON, 25.0)
    test_end = date.today().replace(day=1) - timedelta(days=1)          # last day of prev month
    test_start = test_end - timedelta(days=days - 1)

    params = {
        "begin": f"{test_start.isoformat()}T00:00:00",
        "end":   f"{test_end.isoformat()}T23:59:59",
        "bbox":  (f"{bbox['lat_min']:.4f},{bbox['lon_min']:.4f},"
                  f"{bbox['lat_max']:.4f},{bbox['lon_max']:.4f}"),
    }

    result = {
        "url":         requests.Request("GET", SWDI_BASE, params=params).prepare().url,
        "status_code": None,
        "raw_text":    "",
        "columns":     [],
        "row_count":   0,
        "error":       None,
    }

    try:
        resp = requests.get(SWDI_BASE, params=params, timeout=30)
        result["status_code"] = resp.status_code
        result["raw_text"] = resp.text[:1500]
        resp.raise_for_status()

        df = _parse_response(resp.text)
        result["columns"]   = list(df.columns)
        result["row_count"] = len(df)

    except Exception as exc:
        result["error"] = str(exc)

    return result


def _parse_response(text: str) -> pd.DataFrame:
    """
    Parse a raw SWDI CSV response, stripping leading comment/header lines
    that begin with '#'.  Returns an empty DataFrame if no data rows exist.
    """
    if not text or not text.strip():
        return pd.DataFrame()

    lines = text.splitlines()

    # Separate comment lines from data lines
    data_lines = [ln for ln in lines if not ln.startswith("#")]

    if len(data_lines) < 2:
        # Only a header row (or nothing) — no actual strike records
        return pd.DataFrame()

    # Check for a "no data" indicator in comment lines
    comment_text = " ".join(ln for ln in lines if ln.startswith("#")).lower()
    if "no data" in comment_text or "no records" in comment_text:
        return pd.DataFrame()

    try:
        return pd.read_csv(StringIO("\n".join(data_lines)))
    except Exception:
        return pd.DataFrame()


def _fetch_chunk(start: date, end: date, bbox: dict) -> pd.DataFrame:
    """
    Fetch one date-range chunk from SWDI and return a raw DataFrame.
    Raises RuntimeError on HTTP failure; returns empty DataFrame for no-data responses.
    """
    params = {
        "begin": f"{start.isoformat()}T00:00:00",
        "end":   f"{end.isoformat()}T23:59:59",
        "bbox":  (f"{bbox['lat_min']:.4f},{bbox['lon_min']:.4f},"
                  f"{bbox['lat_max']:.4f},{bbox['lon_max']:.4f}"),
    }

    try:
        resp = requests.get(SWDI_BASE, params=params, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"NCEI SWDI request failed ({start} – {end}): {exc}") from exc

    return _parse_response(resp.text)


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise SWDI column names to lowercase snake_case and map to
    standard internal names: timestamp_utc, latitude, longitude.

    Known SWDI column names:
      ZTIME  → timestamp_utc
      LAT    → latitude
      LON    → longitude
    """
    # Lowercase + strip
    df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]

    # Timestamp: SWDI uses 'ztime' (UTC)
    ts_map = {"ztime": "timestamp_utc", "datetime": "timestamp_utc",
               "date_time": "timestamp_utc", "time": "timestamp_utc"}
    for src, dst in ts_map.items():
        if src in df.columns and "timestamp_utc" not in df.columns:
            df = df.rename(columns={src: dst})
    # Fallback: any column containing 'time' or 'date'
    if "timestamp_utc" not in df.columns:
        candidates = [c for c in df.columns if "time" in c or "date" in c]
        if candidates:
            df = df.rename(columns={candidates[0]: "timestamp_utc"})

    # Latitude
    lat_map = {"lat": "latitude", "latitude": "latitude"}
    for src, dst in lat_map.items():
        if src in df.columns and "latitude" not in df.columns:
            df = df.rename(columns={src: dst})

    # Longitude
    lon_map = {"lon": "longitude", "long": "longitude", "longitude": "longitude"}
    for src, dst in lon_map.items():
        if src in df.columns and "longitude" not in df.columns:
            df = df.rename(columns={src: dst})

    return df


def fetch_strikes(
    start_year: int,
    end_year: int,
    radius_miles: float = 25.0,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Fetch all cloud-to-ground lightning strikes within radius_miles of the
    Pueblo Chemical Depot for the given year range.

    Returns a DataFrame with columns:
        timestamp_utc  (datetime64[ns, UTC])
        latitude       (float)
        longitude      (float)
        distance_miles (float)
    """
    bbox = _bbox_for_radius(DEPOT_LAT, DEPOT_LON, radius_miles)

    # Build monthly chunks
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

        raw = _fetch_chunk(chunk_start, chunk_end, bbox)
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
            f"SWDI response missing expected columns: {missing}. "
            f"Columns found: {list(df.columns)}"
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
