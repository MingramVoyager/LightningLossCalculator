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
    Returns dict with min/max lat and lon.  We add a 10% margin so edge
    storms are never clipped before the haversine filter.
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


def _fetch_chunk(start: date, end: date, bbox: dict) -> pd.DataFrame:
    """
    Fetch one date-range chunk from SWDI and return a raw DataFrame.
    Returns an empty DataFrame on any error.
    """
    params = {
        "begin": f"{start.isoformat()}T00:00:00",
        "end": f"{end.isoformat()}T23:59:59",
        "bbox": f"{bbox['lat_min']:.4f},{bbox['lon_min']:.4f},{bbox['lat_max']:.4f},{bbox['lon_max']:.4f}",
    }

    try:
        resp = requests.get(SWDI_BASE, params=params, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"NCEI SWDI request failed ({start} – {end}): {exc}") from exc

    text = resp.text.strip()
    if not text or text.startswith("#") or "no data" in text.lower():
        return pd.DataFrame()

    # Strip any comment lines that SWDI prepends
    lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
    if len(lines) < 2:
        return pd.DataFrame()

    try:
        df = pd.read_csv(StringIO("\n".join(lines)))
    except Exception:
        return pd.DataFrame()

    return df


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise SWDI column names — NCEI occasionally changes capitalisation
    or spacing. Map everything to lowercase snake_case.
    """
    rename = {}
    for col in df.columns:
        lo = col.strip().lower().replace(" ", "_").replace("-", "_")
        rename[col] = lo
    df = df.rename(columns=rename)

    # Identify timestamp column (could be 'ztime', 'datetime', 'date_time', ...)
    ts_candidates = [c for c in df.columns if "time" in c or "date" in c]
    if ts_candidates and "timestamp_utc" not in df.columns:
        df = df.rename(columns={ts_candidates[0]: "timestamp_utc"})

    # Identify lat / lon
    for candidate, target in [
        (["lat", "latitude"], "latitude"),
        (["lon", "long", "longitude"], "longitude"),
    ]:
        matches = [c for c in df.columns if c in candidate]
        if matches and target not in df.columns:
            df = df.rename(columns={matches[0]: target})

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

    Parameters
    ----------
    start_year : int
    end_year   : int  (inclusive)
    radius_miles : float  — fetch bounding box sized for this radius; haversine
                            filter applied afterwards.  Default 25 mi gives a
                            comfortable buffer beyond the 20-mi shutdown trigger.
    progress_callback : callable(current_chunk, total_chunks) | None

    Returns
    -------
    pd.DataFrame with columns:
        timestamp_utc  (datetime64[ns, UTC])
        latitude       (float)
        longitude      (float)
        distance_miles (float)   — great-circle distance from depot
    """
    bbox = _bbox_for_radius(DEPOT_LAT, DEPOT_LON, radius_miles)

    # Build list of monthly chunks
    chunks: list[tuple[date, date]] = []
    current = date(start_year, 1, 1)
    end_date = date(end_year, 12, 31)
    while current <= end_date:
        chunk_end = min(current + timedelta(days=_CHUNK_DAYS - 1), end_date)
        chunks.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)

    total = len(chunks)
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

    # Require usable columns
    required = {"timestamp_utc", "latitude", "longitude"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise RuntimeError(
            f"SWDI response missing expected columns: {missing}. "
            f"Columns found: {list(df.columns)}"
        )

    # Parse and localise timestamp
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc", "latitude", "longitude"])

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])

    # Haversine filter — keep only strikes within the requested radius
    df["distance_miles"] = df.apply(
        lambda r: haversine_miles(DEPOT_LAT, DEPOT_LON, r["latitude"], r["longitude"]),
        axis=1,
    )
    df = df[df["distance_miles"] <= radius_miles].copy()

    df = df[["timestamp_utc", "latitude", "longitude", "distance_miles"]].sort_values("timestamp_utc")
    df = df.reset_index(drop=True)

    return df
