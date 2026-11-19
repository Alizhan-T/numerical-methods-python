"""ODE solver comparison: Euler, Modified Euler (Heun's), and 4th-order Runge-Kutta (RK4)."""

from __future__ import annotations

import argparse
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

OdeFn = Callable[[float, float], float]


def ode_rhs(x: float, y: float) -> float:
    """Differential equation: dy/dx = e^x - y^2."""
    return float(np.exp(x) - y**2)


def euler(f: OdeFn, x: np.ndarray, y0: float, h: float) -> np.ndarray:
    """Explicit Euler method."""
    y = np.zeros(len(x), dtype=float)
    y[0] = y0
    for i in range(len(x) - 1):
        y[i + 1] = y[i] + h * f(x[i], y[i])
    return y


def modified_euler(f: OdeFn, x: np.ndarray, y0: float, h: float) -> np.ndarray:
    """Modified Euler (Heun's predictor-corrector) method."""
    y = np.zeros(len(x), dtype=float)
    y[0] = y0
    for i in range(len(x) - 1):
        k1 = f(x[i], y[i])
        y_pred = y[i] + h * k1
        y[i + 1] = y[i] + (h / 2.0) * (k1 + f(x[i + 1], y_pred))
    return y


def rk4(f: OdeFn, x: np.ndarray, y0: float, h: float) -> np.ndarray:
    """Classical 4th-order Runge-Kutta method."""
    y = np.zeros(len(x), dtype=float)
    y[0] = y0
    for i in range(len(x) - 1):
        k1 = h * f(x[i], y[i])
        k2 = h * f(x[i] + h / 2.0, y[i] + k1 / 2.0)
        k3 = h * f(x[i] + h / 2.0, y[i] + k2 / 2.0)
        k4 = h * f(x[i] + h, y[i] + k3)
        y[i + 1] = y[i] + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return y


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Euler, Modified Euler, and RK4 ODE solvers.")
    parser.add_argument("--no-plot", action="store_true", help="Run calculations without opening plot windows.")
    args = parser.parse_args()

    x0, y0, h, x_end = 0.0, 1.0, 0.1, 2.0
    steps = int(round((x_end - x0) / h))
    x_vals = np.linspace(x0, x_end, steps + 1)

    y_euler = euler(ode_rhs, x_vals, y0, h)
    y_mod_euler = modified_euler(ode_rhs, x_vals, y0, h)
    y_rk4 = rk4(ode_rhs, x_vals, y0, h)

    print(f"{'Step':<5} {'x_i':<6} {'Euler':<12} {'Mod Euler':<12} {'RK4':<12}")
    print("-" * 50)
    for i in range(len(x_vals)):
        print(f"{i:<5} {x_vals[i]:<6.1f} {y_euler[i]:<12.4f} {y_mod_euler[i]:<12.4f} {y_rk4[i]:<12.4f}")

    if not args.no_plot:
        plt.figure(figsize=(9, 5))
        plt.plot(x_vals, y_euler, label="Euler", marker="o")
        plt.plot(x_vals, y_mod_euler, label="Modified Euler", marker="s")
        plt.plot(x_vals, y_rk4, label="RK4", linestyle="--", linewidth=2)
        plt.title("ODE Solver Comparison: dy/dx = e^x - y^2")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
