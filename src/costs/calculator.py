"""
Labor cost calculator.

Takes annual loss-hour totals and a list of labor categories and returns
a cost breakdown per category plus totals.
"""

from dataclasses import dataclass, field

import pandas as pd

# Default labor categories for the Pueblo Chemical Depot energetics line.
DEFAULT_CATEGORIES = [
    {"category": "Operators",       "headcount": 12, "hourly_rate": 28.00},
    {"category": "Technicians",     "headcount":  6, "hourly_rate": 35.00},
    {"category": "Supervisors",     "headcount":  3, "hourly_rate": 52.00},
    {"category": "Floor Engineers", "headcount":  2, "hourly_rate": 68.00},
]


def cost_breakdown(
    categories: list[dict],
    annual_loss_hours: float,
    years: int = 1,
) -> pd.DataFrame:
    """
    Compute cost per labor category.

    Parameters
    ----------
    categories : list of dict with keys: category, headcount, hourly_rate
    annual_loss_hours : average annual loss hours (from shutdown analysis)
    years : number of years to project (for multi-year totals)

    Returns
    -------
    pd.DataFrame with columns:
        category, headcount, hourly_rate,
        annual_loss_cost, total_loss_cost (over `years`)
    Includes a TOTAL summary row.
    """
    rows = []
    for cat in categories:
        name        = cat.get("category", "")
        headcount   = float(cat.get("headcount", 0))
        hourly_rate = float(cat.get("hourly_rate", 0))

        annual_cost = headcount * hourly_rate * annual_loss_hours
        total_cost  = annual_cost * years

        rows.append(
            {
                "category":         name,
                "headcount":        int(headcount),
                "hourly_rate":      hourly_rate,
                "annual_loss_cost": annual_cost,
                "total_loss_cost":  total_cost,
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Append totals row
    totals = pd.DataFrame(
        [
            {
                "category":         "TOTAL",
                "headcount":        df["headcount"].sum(),
                "hourly_rate":      None,
                "annual_loss_cost": df["annual_loss_cost"].sum(),
                "total_loss_cost":  df["total_loss_cost"].sum(),
            }
        ]
    )
    return pd.concat([df, totals], ignore_index=True)


def roi_analysis(
    annual_loss_cost: float,
    protection_system_cost: float,
    annual_maintenance_cost: float,
    reduction_pct: float,
    projection_years: int = 10,
) -> dict:
    """
    Return ROI metrics for a lightning protection investment.

    Parameters
    ----------
    annual_loss_cost : current annual labour cost of downtime
    protection_system_cost : one-time capital cost of the system
    annual_maintenance_cost : recurring yearly cost to maintain the system
    reduction_pct : expected % reduction in downtime (0–100)
    projection_years : number of years to project

    Returns
    -------
    dict with keys:
        annual_savings, net_annual_savings, break_even_years,
        npv_{n}yr (cumulative net savings over projection_years),
        protected_annual_cost
    """
    reduction_frac       = max(0.0, min(1.0, reduction_pct / 100.0))
    annual_savings       = annual_loss_cost * reduction_frac
    net_annual_savings   = annual_savings - annual_maintenance_cost
    protected_annual_cost = annual_loss_cost - annual_savings

    if net_annual_savings <= 0:
        break_even_years = float("inf")
    else:
        break_even_years = protection_system_cost / net_annual_savings

    cumulative_net = (net_annual_savings * projection_years) - protection_system_cost

    return {
        "annual_savings":         annual_savings,
        "net_annual_savings":     net_annual_savings,
        "protected_annual_cost":  protected_annual_cost,
        "break_even_years":       break_even_years,
        f"net_{projection_years}yr": cumulative_net,
        "projection_years":       projection_years,
    }
