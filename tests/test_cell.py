"""Spec 05 vapor-cell benchmarks (B1-B2 + worked-example table).

Includes the spec-00 enforcement test for ruling R-3 (transit prefactor).
"""

import numpy as np
import pytest

from rydsim.cell import (
    NATURAL_ABUNDANCE,
    mean_transverse_speed,
    number_density_m3,
    transit_rate,
    vapor_pressure_torr,
)

M_RB87 = 86.909 * 1.66053906892e-27  # kg


def test_b1_rb_vapor_pressure_steck_25c():
    """Steck quotes 3.92e-7 torr at T = 298 K (not 298.15)."""
    assert vapor_pressure_torr("Rb", 298.0) == pytest.approx(3.92e-7, rel=0.01)


def test_b2_cs_vapor_pressure_steck_25c():
    assert vapor_pressure_torr("Cs", 298.0) == pytest.approx(1.488e-6, rel=0.01)


def test_worked_example_rb_densities():
    """Spec 05 §2.a worked table (values reproducible from the formulas)."""
    n25 = number_density_m3("Rb", 298.15)
    assert n25 == pytest.approx(1.292e10 * 1e6, rel=0.01)   # cm^-3 -> m^-3
    n50 = number_density_m3("Rb", 323.15)
    assert n50 == pytest.approx(1.467e11 * 1e6, rel=0.01)
    n100 = number_density_m3("Rb", 373.15)
    assert n100 == pytest.approx(6.014e12 * 1e6, rel=0.01)


def test_isotope_partial_density():
    n87 = number_density_m3("Rb", 298.15, NATURAL_ABUNDANCE["Rb87"])
    assert n87 == pytest.approx(3.595e9 * 1e6, rel=0.01)


def test_cs_density_25c():
    n = number_density_m3("Cs", 298.15)
    assert n == pytest.approx(4.894e10 * 1e6, rel=0.01)


def test_r3_transit_prefactor_lock():
    """Spec 00 enforcement (ruling R-3): gamma_t * w0 / <v_perp> = sqrt(2 ln 2).

    Guards against reverting to the demoted spec-06 estimator (~u/2w0), a
    factor-2.35 error in every transit-limited linewidth.
    """
    w0 = 1e-3
    gt = transit_rate(M_RB87, 300.0, w0)
    vperp = mean_transverse_speed(M_RB87, 300.0)
    assert gt * w0 / vperp == pytest.approx(np.sqrt(2 * np.log(2)), rel=1e-12)


def test_transit_fwhm_rb87_1mm():
    """Spec 05 benchmark: gamma_t/2pi = 39.8 kHz (FWHM 79.6 kHz), Rb-87,
    300 K, w0 = 1 mm."""
    gt = transit_rate(M_RB87, 300.0, 1e-3)
    assert gt / (2 * np.pi) == pytest.approx(39.8e3, rel=0.01)


def test_solid_liquid_branch_continuity():
    """The two branches meet near the melting point within the model's
    accuracy (not exactly — different fits)."""
    p_below = vapor_pressure_torr("Rb", 312.44)
    p_above = vapor_pressure_torr("Rb", 312.46)
    assert abs(p_above - p_below) / p_below < 0.05
