"""Absolute susceptibility chain (spec 06 §2.4): sum rule + suppression.

B-2 is the strongest kind of test in the suite: an exact analytic identity
(sigma_0 = 3 lambda^2 / 2 pi) that ties together every SI prefactor in the
chain N -> dipole -> chi -> absorption. If any factor (2, pi, hbar, eps0,
degeneracy) is wrong anywhere, this fails.
"""

import numpy as np
import pytest

from rydsim.constants import wavelength_to_angular_freq
from rydsim.eit import (
    LadderChiParams,
    chi_ladder,
    chi_si,
    dipole_from_linewidth,
    transmission,
)

LAMBDA_P = 780.241e-9
GAMMA_E = 2 * np.pi * 6.0666e6
OMEGA_ATOM = wavelength_to_angular_freq(LAMBDA_P)


def test_b2_resonant_cross_section_sum_rule():
    """alpha/N on resonance (two-level) = 3 lambda^2 / (2 pi), rel 1e-8."""
    d_ge = dipole_from_linewidth(GAMMA_E, OMEGA_ATOM)
    # two-level: no coupling; normalized response S = 1/gamma_ge on resonance
    p = LadderChiParams(
        omegas=np.array([2 * np.pi * 1.0, 0.0]),
        deltas=np.zeros(2),
        k_signed=np.zeros(2),
        coherence_decays=np.array([GAMMA_E / 2, 1.0]),  # 2nd level inert
    )
    s = chi_ladder(p, np.array([0.0]), 0.0)[0]
    n_density = 1.0e16
    chi = chi_si(np.array([s]), n_density, d_ge)[0]
    k_p = 2 * np.pi / LAMBDA_P
    sigma_0 = k_p * chi.imag / n_density
    expected = 3 * LAMBDA_P**2 / (2 * np.pi)
    assert sigma_0 == pytest.approx(expected, rel=1e-8)
    assert expected == pytest.approx(2.9070e-13, rel=1e-3)  # spec 06 B-2 number


def test_eit_residual_absorption_suppression():
    """On two-photon resonance the residual absorption relative to the
    EIT-free peak is 1/(1 + Om_c^2/(4 g_ge g_gr)) (spec 06 §2.5, VERIFIED
    vs NJP 25, 035001 (2023))."""
    g_ge = GAMMA_E / 2
    g_gr = 2 * np.pi * 20e3
    om_c = 2 * np.pi * 3e6
    p_eit = LadderChiParams(
        omegas=np.array([2 * np.pi * 1.0, om_c]),
        deltas=np.zeros(2), k_signed=np.zeros(2),
        coherence_decays=np.array([g_ge, g_gr]),
    )
    p_free = LadderChiParams(
        omegas=np.array([2 * np.pi * 1.0, 0.0]),
        deltas=np.zeros(2), k_signed=np.zeros(2),
        coherence_decays=np.array([g_ge, g_gr]),
    )
    a_eit = chi_ladder(p_eit, np.array([0.0]), 0.0)[0].real
    a_free = chi_ladder(p_free, np.array([0.0]), 0.0)[0].real
    suppression = a_eit / a_free
    expected = 1.0 / (1.0 + om_c**2 / (4 * g_ge * g_gr))
    assert suppression == pytest.approx(expected, rel=1e-10)


def test_transmission_beer_lambert():
    """T = exp(-k L Im chi); optically-thick sanity on the Doppler-free line.

    A 5 cm room-T Rb cell on bare resonance is strongly absorbing for the
    homogeneous line; the function itself is exact bookkeeping.
    """
    chi = np.array([0.001j])
    t = transmission(chi, 0.05, LAMBDA_P)[0]
    k_p = 2 * np.pi / LAMBDA_P
    assert t == pytest.approx(np.exp(-k_p * 0.05 * 0.001), rel=1e-12)


def test_dipole_from_linewidth_rb_d2_magnitude():
    """Rb D2 closed-cycle dipole ~ 2.5-5 e*a0 (Steck: reduced 4.228 ea0,
    cycling ~2.99 ea0 x sqrt factors) — order-anchor, exact values land
    with the matrix-element module."""
    from rydsim.constants import AU_DIPOLE
    d = dipole_from_linewidth(GAMMA_E, OMEGA_ATOM)
    assert 2.0 < d / AU_DIPOLE < 6.0
