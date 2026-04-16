"""
DigiJazz Demo Data Generator
Generates weekly financial data for DigiJazz (a fictional webshop client).

Revenue grows over time. Marketing, IT, and shipping costs are correlated
with revenue. Rental costs are fixed. All other costs are plausible but
uncorrelated noise.

Run:
    python generate_demo_data.py
Outputs:
    digijazz_data.csv
"""

import numpy as np
import pandas as pd

SEED = 42
N_WEEKS = 104  # 2 years of weekly data
START_DATE = "2022-01-03"  # First Monday


def generate():
    rng = np.random.default_rng(SEED)
    dates = pd.date_range(START_DATE, periods=N_WEEKS, freq="W-MON")

    # --- Revenue: starts ~€50k, grows to ~€200k with some weekly noise ---
    trend = np.linspace(50_000, 200_000, N_WEEKS)
    seasonal = 3_000 * np.sin(2 * np.pi * np.arange(N_WEEKS) / 52)  # mild seasonality
    noise = rng.normal(0, 4_000, N_WEEKS)
    revenue = np.maximum(trend + seasonal + noise, 10_000)

    # --- Correlated costs (driven by revenue) ---
    marketing_pct = rng.uniform(0.08, 0.10, N_WEEKS)   # 6–10% of revenue
    it_pct        = rng.uniform(0.015, 0.02, N_WEEKS)  # 1.5–3% of revenue
    shipping_pct  = rng.uniform(0.03, 0.04, N_WEEKS)   # 3–6% of revenue

    marketing_expenses = revenue * marketing_pct + rng.normal(0, 300, N_WEEKS)
    it_costs           = revenue * it_pct          + rng.normal(0, 100, N_WEEKS)
    shipping_costs     = revenue * shipping_pct    + rng.normal(0, 200, N_WEEKS)

    # --- Fixed cost ---
    rental_costs = np.full(N_WEEKS, 2_200.0)  # fixed rent per week

    # --- Plausible uncorrelated costs ---
    employee_expenses = (
        rng.normal(12_000, 1_500, N_WEEKS)
        + np.linspace(0, 4_000, N_WEEKS)     # slight headcount growth
    )
    legal_costs    = np.maximum(rng.exponential(600, N_WEEKS), 100)  # sporadic
    lease_car_costs = rng.normal(950, 80, N_WEEKS)                    # fleet leases
    grocery_costs   = rng.normal(350, 60, N_WEEKS)                    # office groceries

    df = pd.DataFrame({
        "week":               dates,
        "revenue":            revenue.round(2),
        "marketing_expenses": marketing_expenses.clip(0).round(2),
        "it_costs":           it_costs.clip(0).round(2),
        "shipping_costs":     shipping_costs.clip(0).round(2),
        "employee_expenses":  employee_expenses.clip(0).round(2),
        "rental_costs":       rental_costs.round(2),
        "legal_costs":        legal_costs.round(2),
        "lease_car_costs":    lease_car_costs.clip(0).round(2),
        "grocery_costs":      grocery_costs.clip(0).round(2),
    })

    return df


if __name__ == "__main__":
    df = generate()
    out = "digijazz_data.csv"
    df.to_csv(out, index=False)
    print(f"Generated {len(df)} rows → {out}")
    print(df.describe().to_string())
