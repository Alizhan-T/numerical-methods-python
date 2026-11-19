"""Polynomial and spline interpolation: Newton Divided Differences, Lagrange, and Cubic Spline."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline


def divided_differences(x: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Build the Newton divided-difference table."""
    n = len(x)
    table = np.zeros((n, n), dtype=float)
    table[:, 0] = f
    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / (x[i + j] - x[i])
    return table


def newton_eval(x_data: np.ndarray, dd_table: np.ndarray, x: float) -> float:
    """Evaluate the Newton interpolating polynomial at point x."""
    result = float(dd_table[0, 0])
    term = 1.0
    for k in range(1, len(x_data)):
        term *= x - x_data[k - 1]
        result += dd_table[0, k] * term
    return float(result)


def lagrange_eval(x_data: np.ndarray, y_data: np.ndarray, x: float) -> float:
    """Evaluate the Lagrange interpolating polynomial at point x."""
    n = len(x_data)
    total = 0.0
    for i in range(n):
        basis = 1.0
        for j in range(n):
            if j != i:
                basis *= (x - x_data[j]) / (x_data[i] - x_data[j])
        total += y_data[i] * basis
    return float(total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Newton, Lagrange, and Cubic Spline interpolation.")
    parser.add_argument("--no-plot", action="store_true", help="Run calculations without opening plot windows.")
    args = parser.parse_args()

    t = np.array([1990, 1995, 2000, 2005, 2010, 2015], dtype=float)
    y = np.array([2450800, 2710500, 2890200, 3150700, 3420300, 3810600], dtype=float)

    dd = divided_differences(t, y)
    print("Divided Difference Table (upper triangle):")
    for i in range(len(t)):
        row = [f"{dd[i, j]:.6g}" for j in range(len(t) - i)]
        print("  " + "  ".join(row))

    spline = CubicSpline(t, y)
    years = np.arange(1990, 2016, 1, dtype=float)
    newton_vals = np.array([newton_eval(t, dd, yr) for yr in years])
    lagrange_vals = np.array([lagrange_eval(t, y, yr) for yr in years])
    spline_vals = spline(years)

    print("\nYear   Newton         Lagrange       CubicSpline")
    for yr, nv, lv, sv in zip(years.astype(int), newton_vals, lagrange_vals, spline_vals):
        print(f"{yr}  {nv:12.2f}  {lv:12.2f}  {sv:12.2f}")

    for target in (2018.0, 2025.0):
        print(f"\nExtrapolation for {int(target)}:")
        print(f"  Newton   = {newton_eval(t, dd, target):.2f}")
        print(f"  Lagrange = {lagrange_eval(t, y, target):.2f}")
        print(f"  Spline   = {float(spline(target)):.2f}")

    # Scaling study: shift years to offset from 1990 and scale population to millions
    t_scaled = t - 1990.0
    y_scaled = y / 1_000_000.0
    dd_scaled = divided_differences(t_scaled, y_scaled)
    spline_scaled = CubicSpline(t_scaled, y_scaled)

    print("\n--- Scaling Study (years since 1990, population in millions) ---")
    for target in (2018.0, 2025.0):
        u = target - 1990.0
        print(
            f"  {int(target)}: Newton={newton_eval(t_scaled, dd_scaled, u):.4f}M, "
            f"Lagrange={lagrange_eval(t_scaled, y_scaled, u):.4f}M, "
            f"Spline={float(spline_scaled(u)):.4f}M"
        )

    if not args.no_plot:
        plt.figure(figsize=(9, 5))
        plt.scatter(t, y, color="black", zorder=5, label="Data points")
        plt.plot(years, newton_vals, label="Newton")
        plt.plot(years, lagrange_vals, linestyle="--", label="Lagrange")
        plt.plot(years, spline_vals, label="Cubic Spline")
        plt.xlabel("Year")
        plt.ylabel("Population")
        plt.title("Interpolation Comparison (Newton, Lagrange, Cubic Spline)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
