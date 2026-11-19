"""SIR epidemic compartmental model solved with 4th-order Runge-Kutta (RK4)."""

from __future__ import annotations

import argparse
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


def sir_system(y: np.ndarray, population: float, beta: float, gamma: float) -> np.ndarray:
    """Compute [dS/dt, dI/dt, dR/dt] for the normalized SIR system."""
    s, i, _ = y
    infection = beta * s * i / population
    recovery = gamma * i
    return np.array([-infection, infection - recovery, recovery], dtype=float)


def rk4_solve_sir(
    population: float,
    s0: float,
    i0: float,
    r0: float,
    beta: float,
    gamma: float,
    days: int = 100,
    h: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate the SIR ODE system using classical RK4."""
    steps = int(round(days / h))
    t = np.linspace(0.0, float(days), steps + 1)
    y = np.zeros((steps + 1, 3), dtype=float)
    y[0] = [s0, i0, r0]

    for n in range(steps):
        k1 = h * sir_system(y[n], population, beta, gamma)
        k2 = h * sir_system(y[n] + k1 / 2.0, population, beta, gamma)
        k3 = h * sir_system(y[n] + k2 / 2.0, population, beta, gamma)
        k4 = h * sir_system(y[n] + k3, population, beta, gamma)
        y[n + 1] = y[n] + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    return t, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate SIR epidemic intervention scenarios with RK4.")
    parser.add_argument("--no-plot", action="store_true", help="Run calculations without opening plot windows.")
    args = parser.parse_args()

    population = 1_000_000.0
    s0_base, i0, r0 = 999_000.0, 1_000.0, 0.0
    beta_base = 0.35
    gamma = 0.10

    scenarios = [
        ("Baseline", s0_base, beta_base, "-"),
        ("50% Vaccination", s0_base * 0.5, beta_base, "--"),
        ("50% Social Distancing", s0_base, beta_base * 0.5, ":"),
    ]

    if not args.no_plot:
        plt.figure(figsize=(10, 6))

    for label, s_start, beta_val, style in scenarios:
        t, result = rk4_solve_sir(population, s_start, i0, r0, beta_val, gamma)
        infected = result[:, 1]
        recovered = result[:, 2]
        peak = float(np.max(infected))
        peak_day = float(t[int(np.argmax(infected))])
        total_infected = float(recovered[-1] + infected[-1])
        print(f"{label:<22} | Peak = {peak:9.0f} (Day {peak_day:4.1f}) | Total Infected = {total_infected:9.0f}")

        if not args.no_plot:
            plt.plot(t, infected, label=label, linestyle=style, linewidth=2)

    if not args.no_plot:
        plt.title("Impact of Interventions on Active Infections I(t)")
        plt.xlabel("Days")
        plt.ylabel("Infected Population")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
