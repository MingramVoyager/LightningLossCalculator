# CLAUDE.md — LightningLossCalculator

## Project Overview

A Streamlit web app that estimates operational downtime and labor costs caused by lightning proximity shutdown rules at the **Pueblo Chemical Depot** (Pueblo, CO). It analyzes historical thunderstorm observations from Pueblo Memorial Airport (KPUB) via the Iowa Environmental Mesonet (IEM) ASOS archive, and supports ROI analysis for lightning protection system investments.

## Running the App

```bash
pip install -r requirements.txt
streamlit run app.py
```

App runs at `http://localhost:8501` by default.

## Architecture

```
app.py                       # Streamlit entry point, 4 tabs: Data / Loss Analysis / Cost Calculator / ROI
src/
  data/
    isd_client.py            # Fetches thunderstorm hours from IEM ASOS (primary source)
    ncei_client.py           # NOAA NCEI ADS client (fallback, currently unused)
    cache.py                 # Parquet caching layer in system temp dir
  analysis/
    shutdown_engine.py       # Core O(n log n) state machine: groups strikes → shutdown events → loss hours
  costs/
    calculator.py            # Labor cost breakdown and ROI/NPV analysis
.streamlit/config.toml       # UI theme (blue, white, sans-serif)
requirements.txt
```

## Key Design Decisions

- **Data source:** IEM ASOS METAR observations for KPUB. Thunderstorm codes (`TS*` in present-weather field) are treated as lightning within all configured distance thresholds, since the airport is ~3 miles from the depot.
- **Shutdown logic:** Consecutive thunderstorm observations within `clear_minutes` of each other are merged into a single event. Each event spans `[first_obs, last_obs + clear_minutes]`. Loss hours are the intersection of events with the configured operating window (Mountain Time).
- **Caching:** Fetched data is stored as per-year Parquet files in the system temp directory so Streamlit reruns don't re-fetch.
- **Timezone:** All raw timestamps are UTC internally; conversion to Mountain Time happens only in the shutdown engine's window logic.

## Configuration (Sidebar)

| Parameter | Default | Description |
|---|---|---|
| Year range | 2006–present | Years of data to fetch/analyze |
| Operating window | 6 AM – 6 PM MT | Hours when the depot operates |
| Warning threshold | 20 miles | Distance that triggers shutdown |
| All-clear time | 30 minutes | Minutes after last storm obs before resuming |

## Default Labor Categories (editable in UI)

| Category | Headcount | Rate |
|---|---|---|
| Operators | 12 | $28/hr |
| Technicians | 6 | $35/hr |
| Supervisors | 3 | $52/hr |
| Floor Engineers | 2 | $68/hr |

## Dependencies

Python 3.9+, see `requirements.txt`:
- `streamlit >= 1.35.0`
- `pandas >= 2.1.0`
- `plotly >= 5.20.0`
- `numpy >= 1.26.0`
- `pyarrow >= 14.0.0`
- `pytz >= 2024.1`
- `requests >= 2.31.0`

## Notes

- NCEI KPUB ISD files were found to be unavailable; the app was switched to IEM ASOS in commit `30b1a5f`.
- `ncei_client.py` is retained for potential future use but is not wired into the active data pipeline.
- No secrets or environment variables are required to run the app.
