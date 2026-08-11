"""Validation: analytic weak-probe engine vs Lindblad, and convergence.

This file encodes the lesson of the quadrature-aliasing incident: the
analytic continued-fraction path and the exact Lindblad path must agree
class-by-class and after averaging, and the average must be provably
converged (grid doubling) before any number leaves the engine.
"""

import numpy as np
import pytest

from rydsim.doppler import DopplerLadder
from rydsim.eit import (
    LadderChiParams,
    chi_ladder,
    doppler_average,
    resonance_refined_vgrid,
)
from rydsim.lindblad import LadderSystem
from rydsim.provenance import IntegrityError

M_RB87 = 86.909 * 1.66053906892e-27
KP, KC = 2 * np.pi / 780.241e-9, 2 * np.pi / 480.0e-9
GAMMA_E = 2 * np.pi * 6.07e6
G_R = 2 * np.pi * 2e3
G_RP = 2 * np.pi * 1e3
DEPH = 2 * np.pi * 100e3


def _params(omega_rf=0.0):
    omegas = [2 * np.pi * 10e3, 2 * np.pi * 5e6]
    deltas = [0.0, 0.0]
    ks = [KP, -KC]
    gs = [GAMMA_E / 2, G_R / 2 + DEPH]
    if omega_rf > 0:
        omegas.append(omega_rf)
        deltas.append(0.0)
        ks.append(0.0)
        gs.append(G_RP / 2 + DEPH)
    return LadderChiParams(omegas=np.array(omegas), deltas=np.array(deltas),
                           k_signed=np.array(ks),
                           coherence_decays=np.array(gs))


def _lindblad_norm_coherence(omega_rf, dp, v):
    """Exact Lindblad steady state for one velocity class."""
    omegas = [2 * np.pi * 10e3, 2 * np.pi * 5e6]
    deltas = [dp - KP * v, 0.0 + KC * v]
    decays = [0.0, GAMMA_E, G_R]
    dephs = [0.0, 0.0, DEPH]
    if omega_rf > 0:
        omegas.append(omega_rf)
        deltas.append(0.0)
        decays.append(G_RP)
        dephs.append(DEPH)
    sys = LadderSystem(omegas=omegas, deltas=deltas, decays=decays,
                       dephasings=dephs)
    rho = sys.steady_state()
    return rho[1, 0] / (1j * omegas[0] / 2)


@pytest.mark.parametrize("omega_rf", [0.0, 2 * np.pi * 20e6])
def test_analytic_matches_lindblad_per_velocity_class(omega_rf):
    """Continued fraction == Lindblad for every sampled (detuning, velocity)."""
    p = _params(omega_rf)
    for dp in 2 * np.pi * np.array([-15e6, -5e6, 0.0, 3e6, 12e6]):
        for v in [-200.0, -50.0, 0.0, 1.7, 120.0]:
            ana = chi_ladder(p, np.array([dp]), v)[0]
            num = _lindblad_norm_coherence(omega_rf, dp, v)
            assert num.real == pytest.approx(ana.real, rel=3e-3,
                                             abs=1e-4 * abs(ana))
            assert num.imag == pytest.approx(ana.imag, rel=3e-3,
                                             abs=1e-4 * abs(ana))


def test_averaged_analytic_matches_averaged_lindblad_spot():
    """Full average agreement at spot detunings on a shared dense grid."""
    omega_rf = 2 * np.pi * 20e6
    p = _params(omega_rf)
    sigma = np.sqrt(1.380649e-23 * 300.0 / M_RB87)
    dps = 2 * np.pi * np.array([0.0, 6.15e6])  # center + near an AT peak
    vg = resonance_refined_vgrid(p, sigma, (dps.min() - 1, dps.max() + 1))
    # thin the grid for Lindblad cost, identically for both paths
    vg = vg[::12]

    ana = np.array([
        doppler_average(p, np.array([dp]), M_RB87, 300.0, v_grid=vg)[0]
        for dp in dps
    ])

    dl = DopplerLadder(
        omegas=np.array([2 * np.pi * 10e3, 2 * np.pi * 5e6, omega_rf]),
        deltas=np.array([0.0, 0.0, 0.0]),
        decays=np.array([0.0, GAMMA_E, G_R, G_RP]),
        k_signed=np.array([KP, -KC, 0.0]),
        mass=M_RB87, temperature=300.0,
        dephasings=np.array([0.0, 0.0, DEPH, DEPH]),
    )
    num = dl.spectrum(dps, vg)
    for a, n in zip(ana, num):
        assert n.real == pytest.approx(a.real, rel=5e-3)
        assert n.imag == pytest.approx(a.imag, rel=5e-3, abs=5e-3 * abs(a))


def test_convergence_check_passes_on_refined_grid():
    p = _params(2 * np.pi * 25e6)
    dps = 2 * np.pi * np.linspace(-20e6, 20e6, 41)
    out = doppler_average(p, dps, M_RB87, 300.0, check_convergence=True)
    assert out.shape == (41,)
    assert np.all(np.isfinite(out))


def test_convergence_check_fails_on_coarse_grid():
    """A deliberately aliasing-coarse grid must raise, not return numbers."""
    p = _params(2 * np.pi * 25e6)
    dps = 2 * np.pi * np.linspace(-20e6, 20e6, 21)
    sigma = np.sqrt(1.380649e-23 * 300.0 / M_RB87)
    coarse = np.linspace(-5 * sigma, 5 * sigma, 129)  # ~13 m/s spacing: aliases
    with pytest.raises(IntegrityError):
        doppler_average(p, dps, M_RB87, 300.0, v_grid=coarse,
                        check_convergence=True)


def test_weak_probe_3level_analytic_limit():
    """No RF, no Doppler: continued fraction reduces to the textbook chi."""
    p = _params(0.0)
    dps = 2 * np.pi * np.linspace(-20e6, 20e6, 81)
    s = chi_ladder(p, dps, 0.0)
    ge = GAMMA_E / 2
    gr = G_R / 2 + DEPH
    oc = 2 * np.pi * 5e6
    ana = 1.0 / ((ge - 1j * dps) + (oc**2 / 4) / (gr - 1j * dps))
    assert np.allclose(s, ana, rtol=1e-12)
