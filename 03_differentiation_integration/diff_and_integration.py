"""Numerical differentiation (Newton finite differences, Lagrange) and integration (Trapezoidal, Simpson's 1/3 & 3/8)."""

from __future__ import annotations

import numpy as np


def newton_diff_table(y: np.ndarray) -> np.ndarray:
    """Construct the forward finite-difference table for equally spaced samples."""
    n = len(y)
    table = np.zeros((n, n), dtype=float)
    table[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = table[i + 1, j - 1] - table[i, j - 1]
    return table


def lagrange_derivative_at_point(x_nodes: np.ndarray, y_nodes: np.ndarray, xi: float) -> float:
    """Estimate f'(xi) by differentiating the Lagrange interpolating polynomial."""
    n = len(x_nodes)
    deriv = 0.0
    for i in range(n):
        li_prime = 0.0
        for j in range(n):
            if i == j:
                continue
            prod_term = 1.0 / (x_nodes[i] - x_nodes[j])
            for k in range(n):
                if k == i or k == j:
                    continue
                prod_term *= (xi - x_nodes[k]) / (x_nodes[i] - x_nodes[k])
            li_prime += prod_term
        deriv += y_nodes[i] * li_prime
    return float(deriv)


def trapezoidal_rule(y: np.ndarray, h: float) -> float:
    """Composite Trapezoidal Rule: (h / 2) * (y0 + 2 * sum(y_i) + y_n)."""
    return float((h / 2.0) * (y[0] + 2.0 * np.sum(y[1:-1]) + y[-1]))


def simpsons_13(y: np.ndarray, h: float) -> float:
    """Composite Simpson's 1/3 Rule (requires an even number of intervals)."""
    n = len(y) - 1
    if n % 2 != 0:
        raise ValueError("Simpson's 1/3 rule requires an even number of subintervals.")
    return float((h / 3.0) * (y[0] + 4.0 * np.sum(y[1:n:2]) + 2.0 * np.sum(y[2:n:2]) + y[-1]))


def simpsons_38(y: np.ndarray, h: float) -> float:
    """Composite Simpson's 3/8 Rule (requires subinterval count divisible by 3)."""
    n = len(y) - 1
    if n % 3 != 0:
        raise ValueError("Simpson's 3/8 rule requires the number of subintervals to be divisible by 3.")
    total = float(y[0] + y[-1])
    for i in range(1, n):
        total += (2.0 if i % 3 == 0 else 3.0) * float(y[i])
    return float((3.0 * h / 8.0) * total)


def main() -> None:
    print("=== Problem A: Newton Forward & Backward Differentiation ===")
    x_a = np.array([0, 0.7, 1.8, 3.4, 5.1, 6.3, 7.3, 8.0, 8.4], dtype=float)
    h = 2.0
    diff_table = newton_diff_table(x_a)

    # Forward differences at t = 2s (index 1)
    dy_2, d2y_2, d3y_2 = diff_table[1, 1], diff_table[1, 2], diff_table[1, 3]
    v_2 = (1.0 / h) * (dy_2 - 0.5 * d2y_2 + (1.0 / 3.0) * d3y_2)
    a_2 = (1.0 / h**2) * (d2y_2 - d3y_2)
    print(f"At t = 2s  (Forward):  v = {v_2:.4f} m/s, a = {a_2:.4f} m/s^2")

    # Backward differences at t = 14s (index 7)
    dy_14, d2y_14, d3y_14 = diff_table[6, 1], diff_table[5, 2], diff_table[4, 3]
    v_14 = (1.0 / h) * (dy_14 + 0.5 * d2y_14 + (1.0 / 3.0) * d3y_14)
    a_14 = (1.0 / h**2) * (d2y_14 + d3y_14)
    print(f"At t = 14s (Backward): v = {v_14:.4f} m/s, a = {a_14:.4f} m/s^2\n")

    print("=== Problem B: Unequally Spaced Data (Lagrange Derivative) ===")
    x_b = np.array([0.6, 1.5, 1.6, 2.5, 3.5], dtype=float)
    y_b = 5.0 * np.exp(-2.0 * x_b) * x_b
    print(f"{'x':<6} | {'Exact f\'(x)':<12} | {'Lagrange f\'(x)':<15} | {'Abs Error':<10}")
    print("-" * 52)
    for val in x_b:
        exact_d = float(5.0 * np.exp(-2.0 * val) * (1.0 - 2.0 * val))
        num_d = lagrange_derivative_at_point(x_b, y_b, float(val))
        print(f"{val:<6.1f} | {exact_d:<12.5f} | {num_d:<15.5f} | {abs(exact_d - num_d):<10.5f}")

    print("\n=== Problem C & D: Numerical Integration ===")
    x_test = np.linspace(0.0, np.pi, 13)
    y_test = np.sin(x_test)
    h_test = float(x_test[1] - x_test[0])
    print("Benchmark integral sin(x) on [0, pi] (exact = 2.00000):")
    print(f"  Trapezoidal: {trapezoidal_rule(y_test, h_test):.5f}")
    print(f"  Simpson 1/3: {simpsons_13(y_test, h_test):.5f}")
    print(f"  Simpson 3/8: {simpsons_38(y_test, h_test):.5f}")

    p_watts = np.array([500, 480, 450, 600, 800, 950, 1000, 980, 920, 850, 700, 550, 500], dtype=float)
    h_energy = 2.0
    print("\n24-Hour City Energy Consumption (MWh):")
    print(f"  Trapezoidal   : {trapezoidal_rule(p_watts, h_energy):.2f} MWh")
    print(f"  Simpson's 1/3 : {simpsons_13(p_watts, h_energy):.2f} MWh")
    print(f"  Simpson's 3/8 : {simpsons_38(p_watts, h_energy):.2f} MWh")


if __name__ == "__main__":
    main()
