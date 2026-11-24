"""Least-squares curve fitting across 5 regression models with SSE and R^2 metrics."""

from __future__ import annotations

import argparse
from typing import Callable, Tuple

import numpy as np

ModelFn = Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, str]]


def _validate_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.ndim != 1 or y_arr.ndim != 1 or len(x_arr) != len(y_arr):
        raise ValueError("x and y must be 1D arrays of equal length.")
    if len(x_arr) < 2:
        raise ValueError("At least 2 data points are required for curve fitting.")
    return x_arr, y_arr


def sse_and_r2(y: np.ndarray, y_hat: np.ndarray) -> Tuple[float, float]:
    """Compute Sum of Squared Errors (SSE) and coefficient of determination (R^2)."""
    y_arr = np.asarray(y, dtype=float)
    y_hat_arr = np.asarray(y_hat, dtype=float)
    if len(y_arr) != len(y_hat_arr) or len(y_arr) == 0:
        raise ValueError("y and y_hat must be non-empty arrays of equal length.")
    sse = float(np.sum((y_arr - y_hat_arr) ** 2))
    sst = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    if sst == 0.0:
        return sse, 1.0 if np.isclose(sse, 0.0) else 0.0
    r2 = 1.0 - sse / sst
    return sse, r2


def linear_model(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, str]:
    """Fit y = a + b*x."""
    x_arr, y_arr = _validate_xy(x, y)
    design = np.column_stack((np.ones(len(x_arr)), x_arr))
    a, b = np.linalg.lstsq(design, y_arr, rcond=None)[0]
    return a + b * x_arr, f"y = {a:.4f} + {b:.4f}x"


def parabolic_model(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, str]:
    """Fit y = a + b*x + c*x^2."""
    x_arr, y_arr = _validate_xy(x, y)
    if len(x_arr) < 3:
        raise ValueError("Parabolic model requires at least 3 data points.")
    design = np.column_stack((np.ones(len(x_arr)), x_arr, x_arr**2))
    a, b, c = np.linalg.lstsq(design, y_arr, rcond=None)[0]
    return a + b * x_arr + c * x_arr**2, f"y = {a:.4f} + {b:.4f}x + {c:.4f}x^2"


def power_model(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, str]:
    """Fit y = a * x^b via log-linearization (requires x > 0 and y > 0)."""
    x_arr, y_arr = _validate_xy(x, y)
    if np.any(x_arr <= 0) or np.any(y_arr <= 0):
        raise ValueError("Power model requires strictly positive x and y values.")
    design = np.column_stack((np.ones(len(x_arr)), np.log(x_arr)))
    ln_a, b = np.linalg.lstsq(design, np.log(y_arr), rcond=None)[0]
    a = float(np.exp(ln_a))
    return a * x_arr**b, f"y = {a:.4f} * x^{b:.4f}"


def exponential_model(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, str]:
    """Fit y = a * e^(b*x) via log-linearization (requires y > 0)."""
    x_arr, y_arr = _validate_xy(x, y)
    if np.any(y_arr <= 0):
        raise ValueError("Exponential model requires strictly positive y values.")
    design = np.column_stack((np.ones(len(x_arr)), x_arr))
    ln_a, b = np.linalg.lstsq(design, np.log(y_arr), rcond=None)[0]
    a = float(np.exp(ln_a))
    return a * np.exp(b * x_arr), f"y = {a:.4f} * e^({b:.4f}x)"


def logarithmic_model(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, str]:
    """Fit y = a + b * ln(x) (requires x > 0)."""
    x_arr, y_arr = _validate_xy(x, y)
    if np.any(x_arr <= 0):
        raise ValueError("Logarithmic model requires strictly positive x values.")
    ln_x = np.log(x_arr)
    design = np.column_stack((np.ones(len(x_arr)), ln_x))
    a, b = np.linalg.lstsq(design, y_arr, rcond=None)[0]
    return a + b * ln_x, f"y = {a:.4f} + {b:.4f} ln(x)"


MODELS: list[tuple[str, ModelFn]] = [
    ("Linear", linear_model),
    ("Parabolic", parabolic_model),
    ("Power", power_model),
    ("Exponential", exponential_model),
    ("Logarithmic", logarithmic_model),
]


def evaluate_dataset(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    show_plot: bool,
    save_plot: str | None = None,
) -> tuple[str, float]:
    """Fit all regression models to a dataset, print error metrics, and optionally plot."""
    print(f"=== {name} ===")
    best_name, best_r2 = "", float("-inf")
    should_draw = show_plot or save_plot is not None
    plt = None

    if should_draw:
        import matplotlib

        if save_plot and not show_plot:
            matplotlib.use("Agg")
        import matplotlib.pyplot as _plt

        plt = _plt
        plt.figure(figsize=(9, 5), dpi=140)
        plt.scatter(x, y, color="#0F172A", s=46, zorder=5, label="Observed data")

    for label, model in MODELS:
        y_hat, equation = model(x, y)
        sse, r2 = sse_and_r2(y, y_hat)
        print(f"  {label:<12} | {equation:<32} | SSE = {sse:10.4f} | R^2 = {r2:.6f}")
        if r2 > best_r2:
            best_name, best_r2 = label, r2
        if should_draw and plt is not None:
            plt.plot(x, y_hat, linewidth=2.0, label=f"{label} (R²={r2:.4f})")

    print(f"  -> Best fit: {best_name} (R^2 = {best_r2:.6f})\n")

    if should_draw and plt is not None:
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Least-Squares Curve Fitting — {name} (Best: {best_name})")
        plt.legend(frameon=True)
        plt.grid(True, alpha=0.35)
        plt.tight_layout()
        if save_plot:
            plt.savefig(save_plot, dpi=160)
        if show_plot:
            plt.show()
        plt.close()

    return best_name, best_r2


def main() -> None:
    parser = argparse.ArgumentParser(description="Least-squares curve fitting comparison.")
    parser.add_argument("--no-plot", action="store_true", help="Run calculations without opening plot windows.")
    parser.add_argument("--save-plot", type=str, default=None, help="Save Data Set A plot to a PNG/JPG path.")
    args = parser.parse_args()

    datasets = [
        (
            "Data Set A",
            np.array([2.5, 3.5, 5, 6, 7.5, 10, 12.5, 15, 17.5, 20], dtype=float),
            np.array([5, 3.4, 2, 1.6, 1.2, 0.8, 0.6, 0.4, 0.3, 0.3], dtype=float),
        ),
        (
            "Data Set B",
            np.array([10, 15, 22.5, 33.7, 50.6, 75.9], dtype=float),
            np.array([0.3, 0.675, 1.519, 3.417, 7.639, 11.3], dtype=float),
        ),
        (
            "Data Set C",
            np.array([0.05, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4], dtype=float),
            np.array([550, 750, 1000, 1400, 2000, 2700, 3750], dtype=float),
        ),
    ]

    for idx, (name, x, y) in enumerate(datasets):
        target_save = args.save_plot if idx == 0 else None
        evaluate_dataset(name, x, y, show_plot=not args.no_plot, save_plot=target_save)


if __name__ == "__main__":
    main()
