# ============================================================
# Assignment 4 (pt.2): Interpolation Using Numerical Methods
# Methods: Newton, Lagrange, Cubic Spline
# Tasks: estimates 1990–2015, predict 2018 & 2025, scaling study
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


# -------------------------
# Data (given)
# -------------------------
t = np.array([1990, 1995, 2000, 2005, 2010, 2015], dtype=float)
y = np.array([2450800, 2710500, 2890200, 3150700, 3420300, 3810600], dtype=float)


# ============================================================
# NEWTON (Divided Differences)
# ============================================================

def divided_differences(x, f):
    # Create a table for divided differences
    n = len(x)
    table = np.zeros((n, n), dtype=float)
    table[:, 0] = f

    # Fill the divided difference table
    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / (x[i + j] - x[i])

    return table


def newton_eval(x_data, dd_table, x):
    # Evaluate Newton polynomial at x using divided differences
    n = len(x_data)
    result = dd_table[0, 0]
    term = 1.0

    for k in range(1, n):
        term *= (x - x_data[k - 1])
        result += dd_table[0, k] * term

    return result


# ============================================================
# LAGRANGE
# ============================================================

def lagrange_eval(x_data, y_data, x):
    # Evaluate Lagrange polynomial at x
    n = len(x_data)
    total = 0.0

    for i in range(n):
        Li = 1.0
        for j in range(n):
            if j != i:
                Li *= (x - x_data[j]) / (x_data[i] - x_data[j])
        total += y_data[i] * Li

    return total


# ============================================================
# MAIN CALCULATIONS
# ============================================================

# --- (a) Divided difference table (print) ---
dd = divided_differences(t, y)
print("Divided Difference Table (top-left triangle):")
for i in range(len(t)):
    row = []
    for j in range(len(t) - i):
        row.append(f"{dd[i, j]:.6g}")
    print("  ".join(row))

# --- (b) Newton polynomial numerical evaluation ---
# We'll evaluate on a grid later (years 1990–2015 each year)

# --- Cubic Spline ---
spline = CubicSpline(t, y)  # natural cubic spline by default in SciPy


# ============================================================
# (d) Estimate population for each year 1990–2015
# ============================================================
years = np.arange(1990, 2016, 1, dtype=float)

newton_vals = np.array([newton_eval(t, dd, yr) for yr in years], dtype=float)
lagrange_vals = np.array([lagrange_eval(t, y, yr) for yr in years], dtype=float)
spline_vals = spline(years)

print("\nYear   Newton         Lagrange       CubicSpline")
for yr, nv, lv, sv in zip(years.astype(int), newton_vals, lagrange_vals, spline_vals):
    print(f"{yr}  {nv:12.2f}  {lv:12.2f}  {sv:12.2f}")

# ============================================================
# (f) Predict for 2018 and 2025
# ============================================================
for target in [2018.0, 2025.0]:
    n_pred = newton_eval(t, dd, target)
    l_pred = lagrange_eval(t, y, target)
    s_pred = float(spline(target))
    print(f"\nPrediction for {int(target)}:")
    print(f"  Newton   = {n_pred:.2f}")
    print(f"  Lagrange = {l_pred:.2f}")
    print(f"  Spline   = {s_pred:.2f}")

# ============================================================
# Plots (Newton vs Lagrange vs Spline)
# ============================================================
plt.scatter(t, y, label="Data points")

plt.plot(years, newton_vals, label="Newton")
plt.plot(years, lagrange_vals, label="Lagrange")
plt.plot(years, spline_vals, label="Cubic Spline")

plt.xlabel("Year")
plt.ylabel("Population")
plt.title("Interpolation (Newton, Lagrange, Cubic Spline)")
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# (g) Scaling study:
# ti = years since 1990, yi = population in millions
# ============================================================
t2 = t - 1990.0
y2 = y / 1_000_000.0

dd2 = divided_differences(t2, y2)
spline2 = CubicSpline(t2, y2)

years2 = years - 1990.0

newton_vals2 = np.array([newton_eval(t2, dd2, u) for u in years2], dtype=float)
lagrange_vals2 = np.array([lagrange_eval(t2, y2, u) for u in years2], dtype=float)
spline_vals2 = spline2(years2)

print("\n--- Scaling study (years since 1990, population in millions) ---")
print("Year   Newton(m)     Lagrange(m)   Spline(m)")
for yr, nv, lv, sv in zip(years.astype(int), newton_vals2, lagrange_vals2, spline_vals2):
    print(f"{yr}  {nv:10.4f}    {lv:10.4f}    {sv:10.4f}")

for target in [2018.0, 2025.0]:
    u = target - 1990.0
    n_pred = newton_eval(t2, dd2, u)
    l_pred = lagrange_eval(t2, y2, u)
    s_pred = float(spline2(u))
    print(f"\nScaled prediction for {int(target)} (millions):")
    print(f"  Newton   = {n_pred:.4f} million")
    print(f"  Lagrange = {l_pred:.4f} million")
    print(f"  Spline   = {s_pred:.4f} million")