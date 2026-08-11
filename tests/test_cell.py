"""Spec 05 vapor-cell benchmarks (B1-B2 + worked-example table).

Includes the spec-00 enforcement test for ruling R-3 (transit prefactor).
"""

import warnings

import numpy as np
import pytest

from rydsim.atom import SPECIES
from rydsim.cell import (
    DENSITY_MODEL_UNC_REL,
    NATURAL_ABUNDANCE,
    VALID_T_RANGE_K,
    density_sensitivity_per_k,
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


def test_natural_abundance_is_not_a_second_transcription():
    """Audit R10 ('duplicated constants are how normative values fork'):
    cell.NATURAL_ABUNDANCE is DERIVED from the spec-01 Species rows, so the
    two tables cannot drift apart. Checked against an independent physical
    fact as well: natural rubidium is exactly two primordial isotopes, so the
    two fractions must sum to 1, and caesium is mononuclidic.
    """
    for name, sp in SPECIES.items():
        assert NATURAL_ABUNDANCE[name] == sp.abundance
    assert NATURAL_ABUNDANCE.keys() == SPECIES.keys()
    assert NATURAL_ABUNDANCE["Rb85"] + NATURAL_ABUNDANCE["Rb87"] == \
        pytest.approx(1.0, abs=1e-12)
    assert NATURAL_ABUNDANCE["Cs133"] == 1.0


def test_elemental_vs_isotope_partial_density_semantics():
    """The 04<->05 interface, pinned. isotope_fraction=1.0 is the TOTAL
    elemental density (the convention Weller's self-broadening beta*N and the
    Fermi pseudopotential shift are defined against — every ground-state
    perturber counts); NATURAL_ABUNDANCE gives the isotope-partial density
    the single-isotope probe susceptibility needs. Feeding the partial value
    to a collisional coefficient understates it by 1/eta = 3.593 (Rb-87) /
    1.386 (Rb-85), the factors the lifetimes audit entry quotes.
    """
    T = 403.15
    elemental = number_density_m3("Rb", T, 1.0)
    for iso, ratio in (("Rb87", 3.5932), ("Rb85", 1.3856)):
        partial = number_density_m3("Rb", T, NATURAL_ABUNDANCE[iso])
        assert elemental / partial == pytest.approx(ratio, rel=1e-4)
    # Cs is mononuclidic: the two conventions coincide, which is why a
    # Cs-only regression cannot detect the confusion.
    assert number_density_m3("Cs", T, NATURAL_ABUNDANCE["Cs133"]) == \
        number_density_m3("Cs", T, 1.0)


def test_extrapolation_outside_the_alcock_window_warns():
    """Audit §3 item 17: 'warn + flag extrapolation in the result; never
    bare.' The module docstring has always promised this; the code did not
    do it."""
    lo, hi = VALID_T_RANGE_K
    with pytest.warns(UserWarning, match="extrapolated"):
        number_density_m3("Rb", lo - 20.0)
    with pytest.warns(UserWarning, match="extrapolated"):
        vapor_pressure_torr("Cs", hi + 20.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # in-band must stay silent
        number_density_m3("Rb", 300.0)
        number_density_m3("Rb", 403.15)
        vapor_pressure_torr("Cs", 373.15)
    assert DENSITY_MODEL_UNC_REL == 0.05        # Alcock's stated fit accuracy


@pytest.mark.parametrize("element, T, expected", [
    ("Rb", 298.15, 0.1058), ("Rb", 373.15, 0.0641), ("Cs", 300.0, 0.0990)])
def test_density_temperature_sensitivity_is_derived_not_transcribed(
        element, T, expected):
    """Audit R22 requires the density's temperature sensitivity to ship with
    every density-dependent output. It is DERIVED from the fit coefficients
    (d ln n/dT = ln10*b/T^2 - 1/T), and here checked against a central
    finite difference of the shipped density function — an independent
    numerical route to the same derivative, agreeing to <1e-8 relative.

    The 0.1058/K at room temperature reproduces the audit's '11 %/K' figure,
    so a 1 K cell-temperature error dominates the +-5 % model band.
    """
    ana = density_sensitivity_per_k(element, T)
    h = 1e-4
    fd = (np.log(number_density_m3(element, T + h))
          - np.log(number_density_m3(element, T - h))) / (2 * h)
    assert ana == pytest.approx(fd, rel=1e-8)
    assert ana == pytest.approx(expected, abs=5e-4)
    assert ana > DENSITY_MODEL_UNC_REL / 5.0    # 1 K swamps the model band


def test_solid_liquid_branch_continuity():
    """The two branches meet near the melting point within the model's
    accuracy (not exactly — different fits)."""
    p_below = vapor_pressure_torr("Rb", 312.44)
    p_above = vapor_pressure_torr("Rb", 312.46)
    assert abs(p_above - p_below) / p_below < 0.05
