"""Least-squares curve fitting across 5 regression models with SSE and R^2 metrics."""

from __future__ import annotations

import argparse
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np

ModelFn = Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, str]]


def sse_and_r2(y: np.ndarray, y_hat: np.ndarray) -> Tuple[float, float]:
    """Compute Sum of Squared Errors (SSE) and coefficient of determination (R^2)."""
    y_arr = np.asarray(y, dtype=float)
    y_hat_arr = np.asarray(y_hat, dtype=float)
    sse = float(np.sum((y_arr - y_hat_arr) ** 2))
    sst = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    r2 = 1.0 - sse / sst
    return sse, r2


def linear_model(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, str]:
    """Fit y = a + b*x."""
    design = np.column_stack((np.ones(len(x)), x))
    a, b = np.linalg.lstsq(design, y, rcond=None)[0]
    return a + b * x, f"y = {a:.4f} + {b:.4f}x"


def parabolic_model(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, str]:
    """Fit y = a + b*x + c*x^2."""
    design = np.column_stack((np.ones(len(x)), x, x**2))
    a, b, c = np.linalg.lstsq(design, y, rcond=None)[0]
    return a + b * x + c * x**2, f"y = {a:.4f} + {b:.4f}x + {c:.4f}x^2"


def power_model(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, str]:
    """Fit y = a * x^b via log-linearization."""
    design = np.column_stack((np.ones(len(x)), np.log(x)))
    ln_a, b = np.linalg.lstsq(design, np.log(y), rcond=None)[0]
    a = float(np.exp(ln_a))
    return a * x**b, f"y = {a:.4f} * x^{b:.4f}"


def exponential_model(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, str]:
    """Fit y = a * e^(b*x) via log-linearization."""
    design = np.column_stack((np.ones(len(x)), x))
    ln_a, b = np.linalg.lstsq(design, np.log(y), rcond=None)[0]
    a = float(np.exp(ln_a))
    return a * np.exp(b * x), f"y = {a:.4f} * e^({b:.4f}x)"


def logarithmic_model(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, str]:
    """Fit y = a + b * ln(x)."""
    ln_x = np.log(x)
    design = np.column_stack((np.ones(len(x)), ln_x))
    a, b = np.linalg.lstsq(design, y, rcond=None)[0]
    return a + b * ln_x, f"y = {a:.4f} + {b:.4f} ln(x)"


MODELS: list[tuple[str, ModelFn]] = [
    ("Linear", linear_model),
    ("Parabolic", parabolic_model),
    ("Power", power_model),
    ("Exponential", exponential_model),
    ("Logarithmic", logarithmic_model),
]


def evaluate_dataset(name: str, x: np.ndarray, y: np.ndarray, show_plot: bool) -> None:
    """Fit all regression models to a dataset, print error metrics, and optionally plot."""
    print(f"=== {name} ===")
    best_name, best_r2 = "", float("-inf")

    if show_plot:
        plt.figure(figsize=(9, 5))
        plt.scatter(x, y, color="black", zorder=5, label="Data points")

    for label, model in MODELS:
        y_hat, equation = model(x, y)
        sse, r2 = sse_and_r2(y, y_hat)
        print(f"  {label:<12} | {equation:<32} | SSE = {sse:10.4f} | R^2 = {r2:.6f}")
        if r2 > best_r2:
            best_name, best_r2 = label, r2
        if show_plot:
            plt.plot(x, y_hat, label=f"{label}: {equation}")

    print(f"  -> Best fit: {best_name} (R^2 = {best_r2:.6f})\n")

    if show_plot:
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Curve Fitting - {name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Least-squares curve fitting comparison.")
    parser.add_argument("--no-plot", action="store_true", help="Run calculations without opening plot windows.")
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

    for name, x, y in datasets:
        evaluate_dataset(name, x, y, show_plot=not args.no_plot)


if __name__ == "__main__":
    main()
