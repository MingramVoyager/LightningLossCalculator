"""
Local parquet cache for downloaded SWDI lightning data.

Cache files are keyed by year and stored in the system temp directory so
writes succeed on both local machines and read-only cloud environments
(Streamlit Community Cloud mounts the repo as read-only).
"""

import tempfile
from pathlib import Path

import pandas as pd

# Use the OS temp directory — writable on all platforms including Streamlit Cloud
_CACHE_DIR = Path(tempfile.gettempdir()) / "lightning_loss_cache"


def cache_path(year: int) -> Path:
    _CACHE_DIR.mkdir(exist_ok=True)
    return _CACHE_DIR / f"strikes_{year}.parquet"


def is_cached(year: int) -> bool:
    return cache_path(year).exists()


def load(year: int) -> pd.DataFrame:
    p = cache_path(year)
    if not p.exists():
        raise FileNotFoundError(f"No cache for {year}")
    df = pd.read_parquet(p)
    # Re-attach UTC timezone if stripped during serialisation
    if "timestamp_utc" in df.columns and df["timestamp_utc"].dt.tz is None:
        df["timestamp_utc"] = df["timestamp_utc"].dt.tz_localize("UTC")
    return df


def save(year: int, df: pd.DataFrame) -> None:
    _CACHE_DIR.mkdir(exist_ok=True)
    df.to_parquet(cache_path(year), index=False)


def delete(year: int) -> None:
    p = cache_path(year)
    if p.exists():
        p.unlink()


def cached_years() -> list[int]:
    _CACHE_DIR.mkdir(exist_ok=True)
    years = []
    for p in _CACHE_DIR.glob("strikes_*.parquet"):
        try:
            years.append(int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            pass
    return sorted(years)


def cache_dir() -> Path:
    """Return the cache directory path (for display purposes)."""
    return _CACHE_DIR
