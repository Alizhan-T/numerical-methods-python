# Numerical Methods in Python

A structured collection of numerical analysis algorithms implemented from scratch in Python with **NumPy**, **SciPy**, and **Matplotlib**.

This repository consolidates curve fitting, interpolation, numerical differentiation & integration, and ODE solvers (including an SIR epidemic simulation) into a single modular toolkit.

## Modules

| Directory | Script | Methods Implemented |
| :--- | :--- | :--- |
| `01_curve_fitting/` | `least_squares.py` | Linear, Parabolic, Power, Exponential, and Logarithmic Least-Squares regression + SSE & $R^2$ |
| `02_interpolation/` | `interpolation.py` | Newton Divided Differences, Lagrange Polynomial, Cubic Spline, extrapolation & scaling study |
| `03_differentiation_integration/` | `diff_and_integration.py` | Newton Forward/Backward differentiation, Lagrange derivative, Trapezoidal, Simpson's 1/3 & 3/8 rules |
| `04_ode_solvers/` | `ode_comparison.py` | Explicit Euler, Modified Euler (Heun's), and 4th-order Runge-Kutta (RK4) comparison |
| `04_ode_solvers/` | `sir_epidemic_model.py` | Coupled SIR epidemic ODE system solved with RK4 across baseline, vaccination, and distancing scenarios |

## Setup & Usage

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run any module (pass `--no-plot` for headless CLI output):

```bash
python3 01_curve_fitting/least_squares.py --no-plot
python3 02_interpolation/interpolation.py --no-plot
python3 03_differentiation_integration/diff_and_integration.py
python3 04_ode_solvers/ode_comparison.py --no-plot
python3 04_ode_solvers/sir_epidemic_model.py --no-plot
```
