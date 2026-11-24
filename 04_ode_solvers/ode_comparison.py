"""ODE solver comparison: Euler, Modified Euler (Heun's), and 4th-order Runge-Kutta (RK4)."""

from __future__ import annotations

import argparse
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

OdeFn = Callable[[float, float], float]


def ode_rhs(x: float, y: float) -> float:
    """Differential equation: dy/dx = e^x - y^2."""
    return float(np.exp(x) - y**2)


def _validate_grid(x: np.ndarray, h: float) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim != 1 or len(x_arr) < 2:
        raise ValueError("Grid x must be a 1D array with at least 2 points.")
    if h <= 0:
        raise ValueError("Step size h must be strictly positive.")
    return x_arr


def euler(f: OdeFn, x: np.ndarray, y0: float, h: float) -> np.ndarray:
    """Explicit Euler method."""
    x_arr = _validate_grid(x, h)
    y = np.zeros(len(x_arr), dtype=float)
    y[0] = y0
    for i in range(len(x_arr) - 1):
        y[i + 1] = y[i] + h * f(float(x_arr[i]), float(y[i]))
    return y


def modified_euler(f: OdeFn, x: np.ndarray, y0: float, h: float) -> np.ndarray:
    """Modified Euler (Heun's predictor-corrector) method."""
    x_arr = _validate_grid(x, h)
    y = np.zeros(len(x_arr), dtype=float)
    y[0] = y0
    for i in range(len(x_arr) - 1):
        k1 = f(float(x_arr[i]), float(y[i]))
        y_pred = y[i] + h * k1
        y[i + 1] = y[i] + (h / 2.0) * (k1 + f(float(x_arr[i + 1]), float(y_pred)))
    return y


def rk4(f: OdeFn, x: np.ndarray, y0: float, h: float) -> np.ndarray:
    """Classical 4th-order Runge-Kutta method."""
    x_arr = _validate_grid(x, h)
    y = np.zeros(len(x_arr), dtype=float)
    y[0] = y0
    for i in range(len(x_arr) - 1):
        xi, yi = float(x_arr[i]), float(y[i])
        k1 = h * f(xi, yi)
        k2 = h * f(xi + h / 2.0, yi + k1 / 2.0)
        k3 = h * f(xi + h / 2.0, yi + k2 / 2.0)
        k4 = h * f(xi + h, yi + k3)
        y[i + 1] = yi + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return y


def reference_solution(f: OdeFn, x: np.ndarray, y0: float) -> np.ndarray:
    """Compute a high-accuracy DOP853 reference solution via SciPy."""
    sol = solve_ivp(
        lambda t, y: [f(float(t), float(y[0]))],
        (float(x[0]), float(x[-1])),
        [float(y0)],
        method="DOP853",
        t_eval=x,
        rtol=1e-12,
        atol=1e-14,
    )
    return sol.y[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Euler, Modified Euler, and RK4 ODE solvers.")
    parser.add_argument("--no-plot", action="store_true", help="Run calculations without opening plot windows.")
    parser.add_argument("--save-plot", type=str, default=None, help="Save plot to a PNG/JPG path.")
    args = parser.parse_args()

    x0, y0, h, x_end = 0.0, 1.0, 0.1, 2.0
    steps = int(round((x_end - x0) / h))
    x_vals = np.linspace(x0, x_end, steps + 1)

    y_euler = euler(ode_rhs, x_vals, y0, h)
    y_mod_euler = modified_euler(ode_rhs, x_vals, y0, h)
    y_rk4 = rk4(ode_rhs, x_vals, y0, h)
    y_ref = reference_solution(ode_rhs, x_vals, y0)

    print(f"{'Step':<5} {'x_i':<6} {'Euler':<11} {'Mod Euler':<11} {'RK4':<11} {'Reference':<11}")
    print("-" * 60)
    for i in range(len(x_vals)):
        print(
            f"{i:<5} {x_vals[i]:<6.1f} {y_euler[i]:<11.5f} "
            f"{y_mod_euler[i]:<11.5f} {y_rk4[i]:<11.5f} {y_ref[i]:<11.5f}"
        )

    print("\nMax Absolute Error vs SciPy DOP853 Reference:")
    print(f"  Euler          : {np.max(np.abs(y_euler - y_ref)):.6e}")
    print(f"  Modified Euler : {np.max(np.abs(y_mod_euler - y_ref)):.6e}")
    print(f"  RK4            : {np.max(np.abs(y_rk4 - y_ref)):.6e}")

    if not args.no_plot or args.save_plot:
        import matplotlib

        if args.save_plot and args.no_plot:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(9, 5), dpi=140)
        plt.plot(x_vals, y_euler, label="Euler (O(h))", marker="o", markersize=4, color="#EF4444")
        plt.plot(x_vals, y_mod_euler, label="Modified Euler / Heun (O(h²))", marker="s", markersize=4, color="#F59E0B")
        plt.plot(x_vals, y_rk4, label="RK4 (O(h⁴))", linewidth=2.4, color="#2563EB")
        plt.plot(x_vals, y_ref, label="SciPy DOP853 Ref", linestyle="--", linewidth=1.8, color="#10B981")
        plt.title("ODE Solver Comparison: dy/dx = eˣ − y² (h = 0.1)")
        plt.xlabel("x")
        plt.ylabel("y(x)")
        plt.legend(frameon=True)
        plt.grid(True, alpha=0.35)
        plt.tight_layout()
        if args.save_plot:
            plt.savefig(args.save_plot, dpi=160)
        if not args.no_plot:
            plt.show()
        plt.close()


if __name__ == "__main__":
    main()
