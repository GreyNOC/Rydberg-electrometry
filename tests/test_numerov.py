"""Validation: Numerov stepper against exactly solvable problems."""

import numpy as np
import pytest

from rydsim.numerov import numerov_inward, numerov_outward


def test_harmonic_oscillator_ground_state():
    """y'' = (x^2 - 1) y has solution exp(-x^2/2) (HO ground state, E=1/2)."""
    x = np.linspace(-6, 6, 2001)
    g = x**2 - 1.0
    exact = np.exp(-x**2 / 2)
    y = numerov_outward(x, g, exact[0], exact[1])
    # compare in the well region where the solution is not exponentially tiny
    mask = np.abs(x) < 3
    y_norm = y * exact[len(x) // 2] / y[len(x) // 2]
    assert np.max(np.abs(y_norm[mask] - exact[mask])) < 1e-8


def test_inward_equals_outward_for_symmetric_problem():
    x = np.linspace(-6, 6, 2001)
    g = x**2 - 1.0
    exact = np.exp(-x**2 / 2)
    y_in = numerov_inward(x, g, exact[-1], exact[-2])
    mid = len(x) // 2
    y_in_norm = y_in * exact[mid] / y_in[mid]
    mask = np.abs(x) < 3
    assert np.max(np.abs(y_in_norm[mask] - exact[mask])) < 1e-8


def test_hydrogen_2p_radial():
    """u'' = [l(l+1)/r^2 - 2/r - 2E] u with u = r R_21, E = -1/8 (a.u.).

    Exact: R_21 = r exp(-r/2) / (2 sqrt(6)), so u = r^2 exp(-r/2)/(2 sqrt(6)).
    """
    r = np.linspace(1e-6, 60.0, 60000)
    l, E = 1, -0.125
    g = l * (l + 1) / r**2 - 2.0 / r - 2.0 * E
    exact = r**2 * np.exp(-r / 2) / (2 * np.sqrt(6))
    y = numerov_inward(r, g, exact[-1], exact[-2])
    # normalize to exact at the peak (r = 4 for 2p u(r))
    ipk = np.argmin(np.abs(r - 4.0))
    y_norm = y * exact[ipk] / y[ipk]
    mask = (r > 0.5) & (r < 30)
    rel_err = np.max(np.abs(y_norm[mask] - exact[mask]) / np.max(exact))
    assert rel_err < 1e-6


def test_convergence_order():
    """Global error should scale ~ h^4 (halving h -> ~16x error reduction)."""
    errs = []
    for n in (500, 1000, 2000):
        x = np.linspace(-6, 6, n + 1)
        g = x**2 - 1.0
        exact = np.exp(-x**2 / 2)
        y = numerov_outward(x, g, exact[0], exact[1])
        mid = len(x) // 2
        y_norm = y * exact[mid] / y[mid]
        mask = np.abs(x) < 3
        errs.append(np.max(np.abs(y_norm[mask] - exact[mask])))
    order1 = np.log2(errs[0] / errs[1])
    order2 = np.log2(errs[1] / errs[2])
    assert order1 > 3.5, f"convergence order {order1:.2f}, expected ~4"
    assert order2 > 3.5, f"convergence order {order2:.2f}, expected ~4"


def test_nonuniform_grid_rejected():
    x = np.array([0.0, 0.1, 0.3, 0.6])
    with pytest.raises(ValueError):
        numerov_outward(x, np.zeros_like(x), 1.0, 1.0)
