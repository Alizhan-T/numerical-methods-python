"""Automated unit tests for all 4 numerical methods modules."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_module(rel_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / rel_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


least_squares = load_module("01_curve_fitting/least_squares.py", "least_squares")
interpolation = load_module("02_interpolation/interpolation.py", "interpolation")
diff_int = load_module("03_differentiation_integration/diff_and_integration.py", "diff_and_integration")
ode_cmp = load_module("04_ode_solvers/ode_comparison.py", "ode_comparison")
sir_mod = load_module("04_ode_solvers/sir_epidemic_model.py", "sir_epidemic_model")


class TestCurveFitting(unittest.TestCase):
    def test_linear_and_parabolic_exact_fits(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_lin = 3.0 + 2.5 * x
        y_hat, _ = least_squares.linear_model(x, y_lin)
        sse, r2 = least_squares.sse_and_r2(y_lin, y_hat)
        self.assertAlmostEqual(sse, 0.0, places=10)
        self.assertAlmostEqual(r2, 1.0, places=10)

        y_quad = 1.0 - 2.0 * x + 0.5 * x**2
        y_qhat, _ = least_squares.parabolic_model(x, y_quad)
        sse_q, r2_q = least_squares.sse_and_r2(y_quad, y_qhat)
        self.assertAlmostEqual(sse_q, 0.0, places=10)
        self.assertAlmostEqual(r2_q, 1.0, places=10)

    def test_constant_y_and_non_positive_domain_guards(self) -> None:
        y_const = np.array([5.0, 5.0, 5.0])
        sse, r2 = least_squares.sse_and_r2(y_const, y_const)
        self.assertEqual((sse, r2), (0.0, 1.0))

        with self.assertRaises(ValueError):
            least_squares.power_model(np.array([-1.0, 2.0]), np.array([1.0, 4.0]))
        with self.assertRaises(ValueError):
            least_squares.exponential_model(np.array([1.0, 2.0]), np.array([0.0, 4.0]))
        with self.assertRaises(ValueError):
            least_squares.logarithmic_model(np.array([0.0, 2.0]), np.array([1.0, 4.0]))


class TestInterpolation(unittest.TestCase):
    def test_newton_and_lagrange_match_exact_cubic(self) -> None:
        x = np.array([-1.0, 0.0, 2.0, 4.0])
        f = lambda v: 2.0 * v**3 - 3.0 * v + 1.0
        y = f(x)
        dd = interpolation.divided_differences(x, y)

        for probe in (-0.5, 1.25, 3.0):
            expected = float(f(probe))
            self.assertAlmostEqual(interpolation.newton_eval(x, dd, probe), expected, places=9)
            self.assertAlmostEqual(interpolation.lagrange_eval(x, y, probe), expected, places=9)

    def test_duplicate_nodes_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            interpolation.divided_differences(np.array([1.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0]))
        with self.assertRaises(ValueError):
            interpolation.lagrange_eval(np.array([1.0, 1.0]), np.array([1.0, 2.0]), 1.5)


class TestDifferentiationAndIntegration(unittest.TestCase):
    def test_quadrature_accuracy_on_sin_x(self) -> None:
        x = np.linspace(0.0, np.pi, 13)
        y = np.sin(x)
        h = float(x[1] - x[0])

        trap = diff_int.trapezoidal_rule(y, h)
        s13 = diff_int.simpsons_13(y, h)
        s38 = diff_int.simpsons_38(y, h)

        self.assertLess(abs(trap - 2.0), 0.02)
        self.assertLess(abs(s13 - 2.0), 5e-4)
        self.assertLess(abs(s38 - 2.0), 1e-3)

    def test_lagrange_derivative_exact_for_quadratic_and_quadrature_guards(self) -> None:
        x = np.array([0.0, 1.5, 4.0])
        y = 3.0 * x**2 - 2.0 * x + 7.0
        self.assertAlmostEqual(diff_int.lagrange_derivative_at_point(x, y, 2.0), 10.0, places=9)

        with self.assertRaises(ValueError):
            diff_int.simpsons_13(np.array([1.0]), 0.5)
        with self.assertRaises(ValueError):
            diff_int.simpsons_38(np.array([1.0, 2.0, 3.0]), 0.5)
        with self.assertRaises(ValueError):
            diff_int.trapezoidal_rule(np.array([1.0, 2.0]), -0.1)


class TestODESolversAndSIR(unittest.TestCase):
    def test_ode_accuracy_hierarchy_rk4_beats_heun_beats_euler(self) -> None:
        x = np.linspace(0.0, 2.0, 21)
        h = 0.1
        y_euler = ode_cmp.euler(ode_cmp.ode_rhs, x, 1.0, h)
        y_heun = ode_cmp.modified_euler(ode_cmp.ode_rhs, x, 1.0, h)
        y_rk4 = ode_cmp.rk4(ode_cmp.ode_rhs, x, 1.0, h)
        y_ref = ode_cmp.reference_solution(ode_cmp.ode_rhs, x, 1.0)

        err_euler = float(np.max(np.abs(y_euler - y_ref)))
        err_heun = float(np.max(np.abs(y_heun - y_ref)))
        err_rk4 = float(np.max(np.abs(y_rk4 - y_ref)))

        self.assertLess(err_rk4, 1e-4)
        self.assertLess(err_rk4, err_heun)
        self.assertLess(err_heun, err_euler)

    def test_sir_conserves_population_and_validates_initial_conditions(self) -> None:
        pop = 1_000_000.0
        s0_vac = 499_500.0
        i0 = 1_000.0
        r0_vac = pop - s0_vac - i0
        t, y = sir_mod.rk4_solve_sir(pop, s0_vac, i0, r0_vac, beta=0.35, gamma=0.10, days=100, h=0.1)

        totals = np.sum(y, axis=1)
        self.assertTrue(np.allclose(totals, pop, rtol=1e-9))
        self.assertEqual(len(t), 1001)

        # Reject unbalanced S0 + I0 + R0 != population
        with self.assertRaises(ValueError):
            sir_mod.rk4_solve_sir(pop, s0_vac, i0, 0.0, beta=0.35, gamma=0.10)


if __name__ == "__main__":
    unittest.main()
