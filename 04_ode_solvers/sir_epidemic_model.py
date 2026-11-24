"""SIR epidemic compartmental model solved with 4th-order Runge-Kutta (RK4)."""

from __future__ import annotations

import argparse
from typing import Tuple

import numpy as np


def sir_system(y: np.ndarray, population: float, beta: float, gamma: float) -> np.ndarray:
    """Compute [dS/dt, dI/dt, dR/dt] for the normalized SIR system."""
    if population <= 0:
        raise ValueError("Total population must be strictly positive.")
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
    """Integrate the SIR ODE system using classical RK4 with mass conservation check."""
    if population <= 0 or days <= 0 or h <= 0:
        raise ValueError("population, days, and h must be strictly positive.")
    if beta < 0 or gamma < 0:
        raise ValueError("beta and gamma must be non-negative.")
    if s0 < 0 or i0 < 0 or r0 < 0:
        raise ValueError("Initial compartments S0, I0, R0 must be non-negative.")
    if not np.isclose(s0 + i0 + r0, population, rtol=1e-6):
        raise ValueError("Initial compartments must conserve total population: S0 + I0 + R0 == population.")

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
    parser.add_argument("--save-plot", type=str, default=None, help="Save plot to a PNG/JPG path.")
    args = parser.parse_args()

    population = 1_000_000.0
    s0_base, i0, r0_base = 999_000.0, 1_000.0, 0.0
    s0_vac = s0_base * 0.5
    r0_vac = population - s0_vac - i0  # Vaccinated individuals start immune in R0
    beta_base = 0.35
    gamma = 0.10

    scenarios = [
        ("Baseline (R₀ = 3.5)", s0_base, r0_base, beta_base, "-", "#EF4444"),
        ("50% Vaccination (50% immune in R₀)", s0_vac, r0_vac, beta_base, "--", "#2563EB"),
        ("50% Social Distancing (R₀ = 1.75)", s0_base, r0_base, beta_base * 0.5, ":", "#10B981"),
    ]

    should_draw = not args.no_plot or args.save_plot is not None
    plt = None
    if should_draw:
        import matplotlib

        if args.save_plot and args.no_plot:
            matplotlib.use("Agg")
        import matplotlib.pyplot as _plt

        plt = _plt
        plt.figure(figsize=(10, 5.5), dpi=140)

    for label, s_start, r_start, beta_val, style, color in scenarios:
        t, result = rk4_solve_sir(population, s_start, i0, r_start, beta_val, gamma)
        infected = result[:, 1]
        recovered = result[:, 2]
        peak = float(np.max(infected))
        peak_day = float(t[int(np.argmax(infected))])
        cumulative_infected = float((recovered[-1] - r_start) + infected[-1])
        print(
            f"{label:<36} | Peak = {peak:9.0f} (Day {peak_day:4.1f}) | "
            f"Cumulative Infected = {cumulative_infected:9.0f}"
        )

        if should_draw and plt is not None:
            plt.plot(t, infected / 1_000.0, label=label, linestyle=style, linewidth=2.4, color=color)

    if should_draw and plt is not None:
        plt.title("SIR Epidemic Model (RK4) — Impact of Interventions on Active Infections")
        plt.xlabel("Days")
        plt.ylabel("Active Infections (Thousands)")
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
