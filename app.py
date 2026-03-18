"""
Lightning Loss Calculator
=========================
Estimates operational downtime and labor costs driven by lightning proximity
rules at the Pueblo Chemical Depot energetics site.

Data source : NOAA NCEI ISD global-hourly — KPUB (Pueblo Airport ASOS, ~3 mi from depot)
Site        : Pueblo Chemical Depot, Pueblo CO  (38.2710°N, 104.3390°W)
"""

from datetime import date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analysis.shutdown_engine import (
    compute_loss,
    monthly_summary,
    yearly_summary,
)
from src.costs.calculator import DEFAULT_CATEGORIES, cost_breakdown, roi_analysis
from src.data import cache, isd_client

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lightning Loss Calculator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

CURRENT_YEAR = date.today().year
MONTH_NAMES  = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


# ── Sidebar — global configuration ───────────────────────────────────────────
with st.sidebar:
    st.title("⚡ Lightning Loss Calculator")
    st.caption("Pueblo Chemical Depot · Pueblo, CO")
    st.divider()

    st.subheader("Analysis Period")
    year_range = st.slider(
        "Year range",
        min_value=2006,
        max_value=CURRENT_YEAR - 1,
        value=(max(2006, CURRENT_YEAR - 10), CURRENT_YEAR - 1),
        help="NOAA NCEI SWDI data is available from ~2006 onward.",
    )
    start_year, end_year = year_range

    st.subheader("Operating Window")
    col_s, col_e = st.columns(2)
    with col_s:
        start_hour = st.number_input("Start (hr)", min_value=0, max_value=23, value=0, step=1,
                                      help="Local Mountain Time (0 = midnight)")
    with col_e:
        end_hour = st.number_input("End (hr)", min_value=1, max_value=24, value=24, step=1,
                                    help="24 = end of day (full 24-hr operation)")

    if start_hour >= end_hour and end_hour != 24:
        st.warning("End hour must be greater than start hour.")

    st.subheader("Shutdown Thresholds")
    warn_miles     = st.number_input("Warning / shutdown start (mi)", value=20, min_value=1, max_value=100, step=1)
    shutdown_miles = st.number_input("Full shutdown (mi)",            value=15, min_value=1, max_value=100, step=1)
    clear_minutes  = st.number_input("All-clear time (min)",          value=30, min_value=5, max_value=120, step=5,
                                      help="Minutes after last strike within warning distance before resuming.")

    if shutdown_miles >= warn_miles:
        st.warning("Full shutdown distance should be less than warning distance.")

    st.subheader("Site")
    st.caption("Lat 38.2710°N  ·  Lon 104.3390°W")

    st.divider()
    st.caption("Data: NOAA NCEI ISD / KPUB ASOS")


# ── Session state defaults ────────────────────────────────────────────────────
if "daily_loss" not in st.session_state:
    st.session_state.daily_loss = None
if "labor_categories" not in st.session_state:
    st.session_state.labor_categories = [dict(c) for c in DEFAULT_CATEGORIES]


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_data, tab_analysis, tab_cost, tab_roi = st.tabs(
    ["📡 Data", "📊 Loss Analysis", "💰 Cost Calculator", "📈 ROI / Break-Even"]
)


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA
# ════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.header("Lightning Data")
    st.markdown(
        "Data source: **NOAA NCEI ISD global-hourly** — Pueblo Memorial Airport ASOS "
        "station (KPUB, ~3 miles from the depot).  When KPUB reports a thunderstorm "
        "in its present-weather field, lightning is treated as present within both "
        "shutdown thresholds.  The 30-minute all-clear rule is applied normally.\n\n"
        "> **Note on precision:** This approach cannot distinguish a storm at 3 miles "
        "from one at 19 miles — any thunderstorm at the airport triggers a shutdown. "
        "This is a conservative estimate appropriate for a multi-year trend analysis. "
        "For exact 15/20-mile radius analysis, Xweather (aerisweather.com) NLDN data "
        "can be integrated — contact us for setup."
    )
    st.caption(f"Cache location: `{cache.cache_dir()}`")

    # ── API Diagnostic ────────────────────────────────────────────────────────
    with st.expander("🔬 API Diagnostic — test connection before fetching"):
        st.caption(
            "Fetches 5 days of KPUB ISD data (July 4–8, 2023 — peak CO storm season) "
            "to confirm the NCEI endpoint is reachable and thunderstorm parsing is working."
        )
        if st.button("Run API Test"):
            with st.spinner("Probing NCEI ISD — trying all station ID candidates…"):
                probe = isd_client.probe_api()

            # Per-candidate results
            st.subheader("Station ID Candidates")
            cand_rows = []
            for c in probe["candidates"]:
                cand_rows.append({
                    "Station ID":   c["station_id"],
                    "HTTP Status":  c["status_code"],
                    "Obs Rows":     c["row_count"],
                    "TS Hours":     c["ts_hours"],
                    "Error":        c["error"] or "",
                })
            st.dataframe(pd.DataFrame(cand_rows), use_container_width=True, hide_index=True)

            if probe["working_station"]:
                st.success(
                    f"✅ Working station ID: **{probe['working_station']}** — "
                    f"{probe['row_count']} observations, {probe['ts_hours']} thunderstorm hours "
                    f"in the July 4–8 2023 test window."
                )
                st.write(f"Columns: `{probe['columns']}`")
            else:
                st.error(
                    "No station ID returned data. Check the raw response below and "
                    "look up the correct USAF+WBAN at "
                    "ncei.noaa.gov/pub/data/noaa/isd-history.csv (search KPUB)."
                )

            st.text_area("Raw response from first working candidate (first 2500 chars)",
                         probe["raw_text"], height=220)

    st.divider()

    # ── Fetch / cache ─────────────────────────────────────────────────────────
    cached = cache.cached_years()
    needed = list(range(start_year, end_year + 1))
    missing = [y for y in needed if y not in cached]

    col_status, col_fetch = st.columns([3, 1])
    with col_status:
        if missing:
            st.info(f"Years not yet cached: **{', '.join(str(y) for y in missing)}**")
        else:
            st.success(f"All {len(needed)} years cached.")
    with col_fetch:
        fetch_clicked = st.button("⬇ Fetch / Refresh", type="primary", use_container_width=True)

    if fetch_clicked:
        fetch_years = missing if missing else needed
        progress_bar = st.progress(0.0, text="Starting…")
        status_box   = st.empty()
        errors = []

        for yi, year in enumerate(fetch_years):
            def _progress(chunk, total, _year=year, _yi=yi, _fetch_years=fetch_years):
                frac = (_yi + (chunk / max(total, 1))) / len(_fetch_years)
                progress_bar.progress(
                    min(frac, 1.0),
                    text=f"Fetching {_year} — chunk {chunk}/{total}…"
                )

            try:
                status_box.info(f"Fetching {year}…")
                df_year = isd_client.fetch_strikes(
                    year, year,
                    progress_callback=_progress,
                )
                cache.save(year, df_year)
                status_box.success(f"{year}: {len(df_year):,} strike records cached.")
            except Exception as exc:
                errors.append(f"{year}: {exc}")
                status_box.error(f"{year} failed: {exc}")

        progress_bar.progress(1.0, text="Done.")
        if errors:
            st.error("Errors during fetch:\n" + "\n".join(errors))
        else:
            st.success("All years fetched. Cached: " + ", ".join(str(y) for y in cache.cached_years()))
        st.rerun()

    # ── Cache summary table ───────────────────────────────────────────────────
    cached = cache.cached_years()   # re-read after possible fetch
    if cached:
        st.subheader("Cached Data Summary")
        rows = []
        for y in sorted(cached):
            try:
                df_y = cache.load(y)
                rows.append({
                    "Year": y,
                    "Strike records": f"{len(df_y):,}",
                    "Min distance (mi)": f"{df_y['distance_miles'].min():.1f}" if not df_y.empty else "—",
                })
            except Exception as e:
                rows.append({"Year": y, "Strike records": f"load error: {e}", "Min distance (mi)": "—"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No data cached yet. Click **Fetch / Refresh** above.")

    # ── Run analysis ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Run Analysis")
    st.caption("Loads cached data for the selected year range and applies the shutdown rules from the sidebar.")
    run_clicked = st.button("▶ Run Analysis", type="primary")

    if run_clicked:
        with st.spinner("Loading cached data and computing shutdown periods…"):
            frames = []
            for year in needed:
                if year in cached:
                    try:
                        frames.append(cache.load(year))
                    except Exception as e:
                        st.warning(f"Could not load {year}: {e}")
                else:
                    st.warning(f"{year} not cached — skipping. Fetch it above first.")

            if frames:
                all_strikes = pd.concat(frames, ignore_index=True)
                daily = compute_loss(
                    all_strikes,
                    warn_miles=float(warn_miles),
                    shutdown_miles=float(shutdown_miles),
                    clear_minutes=int(clear_minutes),
                    start_hour=int(start_hour),
                    end_hour=int(end_hour),
                )
                st.session_state.daily_loss = daily
                st.success(f"Analysis complete — {len(daily):,} days with lightning-related downtime.")
            else:
                st.error("No cached data available for the selected years. Fetch data first.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — LOSS ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
with tab_analysis:
    st.header("Loss Analysis")
    daily: pd.DataFrame = st.session_state.daily_loss

    if daily is None or daily.empty:
        st.info("Run the analysis from the **📡 Data** tab first.")
    else:
        yearly = yearly_summary(daily)
        monthly = monthly_summary(daily)

        # ── Key metrics ──────────────────────────────────────────────────────
        avg_annual    = yearly["loss_hours"].mean()
        total_hours   = yearly["loss_hours"].sum()
        worst_year    = yearly.loc[yearly["loss_hours"].idxmax()]
        num_years     = len(yearly)
        avg_days      = yearly["days_affected"].mean()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Avg Annual Loss Hours", f"{avg_annual:.1f} hr")
        m2.metric("Total Hours (all years)", f"{total_hours:.1f} hr")
        m3.metric(f"Worst Year ({int(worst_year['year'])})", f"{worst_year['loss_hours']:.1f} hr")
        m4.metric("Avg Days Affected / Year", f"{avg_days:.1f}")
        st.divider()

        # ── Annual bar chart ─────────────────────────────────────────────────
        st.subheader("Loss Hours by Year")
        fig_year = px.bar(
            yearly,
            x="year",
            y="loss_hours",
            text="loss_hours",
            labels={"year": "Year", "loss_hours": "Loss Hours"},
            color="loss_hours",
            color_continuous_scale="Reds",
        )
        fig_year.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_year.update_layout(
            xaxis=dict(tickmode="linear", dtick=1),
            coloraxis_showscale=False,
            margin=dict(t=20, b=20),
        )
        fig_year.add_hline(y=avg_annual, line_dash="dot", line_color="gray",
                           annotation_text=f"Avg {avg_annual:.1f} hr", annotation_position="top right")
        st.plotly_chart(fig_year, use_container_width=True)

        # ── Monthly heat map ──────────────────────────────────────────────────
        st.subheader("Loss Hours — Year × Month Heat Map")
        pivot = monthly.pivot(index="year", columns="month", values="loss_hours").fillna(0)
        pivot.columns = [MONTH_NAMES[m - 1] for m in pivot.columns]

        fig_heat = px.imshow(
            pivot,
            aspect="auto",
            color_continuous_scale="YlOrRd",
            labels={"color": "Loss Hours"},
            title="",
        )
        fig_heat.update_layout(margin=dict(t=10, b=20), xaxis_title="Month", yaxis_title="Year")
        st.plotly_chart(fig_heat, use_container_width=True)

        # ── Average loss by month (across all years) ──────────────────────────
        st.subheader("Average Loss Hours by Month (all years)")
        avg_month = monthly.groupby("month")["loss_hours"].mean().reset_index()
        avg_month["month_name"] = avg_month["month"].apply(lambda m: MONTH_NAMES[m - 1])
        fig_mon = px.bar(
            avg_month, x="month_name", y="loss_hours",
            text="loss_hours",
            labels={"month_name": "Month", "loss_hours": "Avg Loss Hours"},
            color="loss_hours", color_continuous_scale="Oranges",
        )
        fig_mon.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_mon.update_layout(coloraxis_showscale=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig_mon, use_container_width=True)

        # ── Detailed event table ──────────────────────────────────────────────
        with st.expander("Daily Detail Table"):
            display = daily[["date","year","month","loss_hours","shutdown_events","max_event_min"]].copy()
            display.columns = ["Date","Year","Month","Loss Hours","Events","Longest Event (min)"]
            st.dataframe(display, use_container_width=True, hide_index=True)
            csv = display.to_csv(index=False).encode()
            st.download_button("Download CSV", csv, "lightning_loss_daily.csv", "text/csv")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — COST CALCULATOR
# ════════════════════════════════════════════════════════════════════════════
with tab_cost:
    st.header("Cost Calculator")

    daily = st.session_state.daily_loss
    if daily is None or daily.empty:
        st.info("Run the analysis from the **📡 Data** tab first.")
    else:
        yearly = yearly_summary(daily)
        avg_annual_hours = float(yearly["loss_hours"].mean())
        num_years = len(yearly)

        st.markdown(
            f"Average annual loss from the analysis period: **{avg_annual_hours:.1f} hours/year** "
            f"({start_year}–{end_year}, {num_years} years)"
        )

        # ── Labor categories editor ───────────────────────────────────────────
        st.subheader("Labor Categories")
        st.caption("Edit headcount and hourly rate for each category. Add or remove rows as needed.")

        cats = st.session_state.labor_categories
        edited = st.data_editor(
            pd.DataFrame(cats),
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "category":    st.column_config.TextColumn("Category",     required=True),
                "headcount":   st.column_config.NumberColumn("Headcount",  min_value=0, step=1, format="%d"),
                "hourly_rate": st.column_config.NumberColumn("Hourly Rate ($)", min_value=0.0, format="$%.2f"),
            },
            key="labor_editor",
        )
        st.session_state.labor_categories = edited.to_dict("records")

        projection_years = st.slider("Project over N years", 1, 20, 5,
                                      help="For multi-year cost totals in the table below.")

        # ── Cost breakdown ────────────────────────────────────────────────────
        st.subheader("Cost Breakdown")
        breakdown = cost_breakdown(
            st.session_state.labor_categories,
            annual_loss_hours=avg_annual_hours,
            years=projection_years,
        )

        if not breakdown.empty:
            display_df = breakdown.copy()
            display_df["annual_loss_cost"] = display_df["annual_loss_cost"].apply(
                lambda v: f"${v:,.0f}" if pd.notna(v) else "—"
            )
            display_df["total_loss_cost"] = display_df["total_loss_cost"].apply(
                lambda v: f"${v:,.0f}" if pd.notna(v) else "—"
            )
            display_df["hourly_rate"] = display_df["hourly_rate"].apply(
                lambda v: f"${v:.2f}" if pd.notna(v) else "—"
            )
            display_df.columns = [
                "Category", "Headcount", "Hourly Rate",
                f"Annual Cost", f"{projection_years}-Year Cost"
            ]

            # Highlight totals row
            def _highlight_total(row):
                if row["Category"] == "TOTAL":
                    return ["background-color: #f0f0f0; font-weight: bold"] * len(row)
                return [""] * len(row)

            st.dataframe(
                display_df.style.apply(_highlight_total, axis=1),
                use_container_width=True,
                hide_index=True,
            )

            # Summary callout
            total_row = breakdown[breakdown["category"] == "TOTAL"]
            if not total_row.empty:
                annual_total = float(total_row["annual_loss_cost"].iloc[0])
                multi_total  = float(total_row["total_loss_cost"].iloc[0])

                c1, c2 = st.columns(2)
                c1.metric("Annual Labor Cost of Downtime", f"${annual_total:,.0f}")
                c2.metric(f"{projection_years}-Year Labor Cost of Downtime", f"${multi_total:,.0f}")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — ROI / BREAK-EVEN
# ════════════════════════════════════════════════════════════════════════════
with tab_roi:
    st.header("ROI / Break-Even Analysis")

    daily = st.session_state.daily_loss
    if daily is None or daily.empty:
        st.info("Run the analysis from the **📡 Data** tab first.")
    else:
        yearly = yearly_summary(daily)
        avg_annual_hours = float(yearly["loss_hours"].mean())

        # Compute current annual labor cost from current categories
        cats = st.session_state.labor_categories
        breakdown_now = cost_breakdown(cats, avg_annual_hours, years=1)
        total_row = breakdown_now[breakdown_now["category"] == "TOTAL"]
        current_annual_cost = float(total_row["annual_loss_cost"].iloc[0]) if not total_row.empty else 0.0

        st.markdown(
            f"Baseline annual labor cost of downtime: **${current_annual_cost:,.0f}** "
            f"(from Cost Calculator tab — update labor categories there to recalculate here)."
        )
        st.divider()

        # ── Lightning protection system inputs ────────────────────────────────
        st.subheader("Lightning Protection System")
        c1, c2 = st.columns(2)
        with c1:
            system_cost = st.number_input(
                "Capital cost of protection system ($)",
                min_value=0, value=500_000, step=10_000,
                help="One-time installed cost of the lightning protection system.",
            )
            annual_maintenance = st.number_input(
                "Annual maintenance cost ($)",
                min_value=0, value=10_000, step=1_000,
            )
        with c2:
            reduction_pct = st.slider(
                "Expected downtime reduction (%)",
                min_value=0, max_value=100, value=80,
                help="What % of current lightning downtime the system is expected to eliminate.",
            )
            projection_years = st.slider(
                "Projection period (years)",
                min_value=1, max_value=30, value=10,
            )

        # ── Compute ROI ───────────────────────────────────────────────────────
        roi = roi_analysis(
            annual_loss_cost=current_annual_cost,
            protection_system_cost=float(system_cost),
            annual_maintenance_cost=float(annual_maintenance),
            reduction_pct=float(reduction_pct),
            projection_years=projection_years,
        )

        st.divider()
        st.subheader("Results")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Annual Savings",       f"${roi['annual_savings']:,.0f}")
        m2.metric("Net Annual Savings",   f"${roi['net_annual_savings']:,.0f}",
                  help="Savings minus annual maintenance cost")
        be = roi["break_even_years"]
        m3.metric("Break-Even",
                  f"{be:.1f} yrs" if be != float("inf") else "Never",
                  delta=f"vs {projection_years}-yr projection" if be < projection_years else None)
        net_key = f"net_{projection_years}yr"
        net_val = roi[net_key]
        m4.metric(f"{projection_years}-Year Net Benefit",
                  f"${net_val:,.0f}",
                  delta="positive" if net_val > 0 else "negative")

        # ── Cumulative savings chart ──────────────────────────────────────────
        st.subheader(f"Cumulative Net Benefit Over {projection_years} Years")
        years_list  = list(range(1, projection_years + 1))
        cumulative  = [roi["net_annual_savings"] * y - system_cost for y in years_list]
        baseline    = [-current_annual_cost * y for y in years_list]  # cost with no system

        fig_roi = go.Figure()
        fig_roi.add_trace(go.Scatter(
            x=years_list, y=cumulative,
            mode="lines+markers", name="With Protection System",
            line=dict(color="#2196F3", width=3),
        ))
        fig_roi.add_trace(go.Scatter(
            x=years_list, y=baseline,
            mode="lines", name="Without Protection (cumulative loss)",
            line=dict(color="#F44336", width=2, dash="dot"),
        ))
        fig_roi.add_hline(y=0, line_color="gray", line_dash="dash")
        fig_roi.update_layout(
            yaxis_title="Cumulative Net Benefit ($)",
            xaxis_title="Years from Installation",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            margin=dict(t=10, b=20),
        )
        fig_roi.update_yaxes(tickformat="$,.0f")
        st.plotly_chart(fig_roi, use_container_width=True)

        # ── Scenario summary table ────────────────────────────────────────────
        st.subheader("Scenario Summary")
        summary = pd.DataFrame([
            {"Scenario": "No protection system",
             "Annual Downtime Cost": f"${current_annual_cost:,.0f}",
             f"{projection_years}-yr Total Cost": f"${current_annual_cost * projection_years:,.0f}"},
            {"Scenario": f"With protection ({reduction_pct}% reduction)",
             "Annual Downtime Cost": f"${roi['protected_annual_cost']:,.0f}",
             f"{projection_years}-yr Total Cost":
                 f"${roi['protected_annual_cost'] * projection_years + system_cost + annual_maintenance * projection_years:,.0f}"},
        ])
        st.dataframe(summary, use_container_width=True, hide_index=True)
