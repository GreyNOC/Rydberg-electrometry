"""Spec 06 benchmark suite (docs/spec/06-optical-bloch-eit.md §6).

Implements the normative convention locks: B-4/B-5 (dressed-state
splittings), B-7 (invariants + transpose canary), B-13 (transit channel),
B-14 (decay routing insensitivity), and the transit-on analytic/Lindblad
agreement that retroactively catches the missing |g><g| transit operator.
"""

import numpy as np
import pytest
from scipy.signal import find_peaks

from rydsim.eit import LadderChiParams, chi_ladder
from rydsim.lindblad import LadderSystem

GAMMA_E = 2 * np.pi * 6.0666e6   # Rb D2, Steck (spec 06 §3, VERIFIED)
G_R = 2 * np.pi * 1e3
G_RP = 2 * np.pi * 1e3


def _cold_at_spectrum(omega_rf, delta_rf, dps, omega_c=2 * np.pi * 3e6):
    p = LadderChiParams(
        omegas=np.array([2 * np.pi * 1.0, omega_c, omega_rf]),
        deltas=np.array([0.0, 0.0, delta_rf]),
        k_signed=np.zeros(3),
        coherence_decays=np.array([GAMMA_E / 2, G_R / 2, G_RP / 2]),
    )
    return chi_ladder(p, dps, 0.0)


def _extract_splitting(dps, absorption):
    trans = -absorption
    peaks, props = find_peaks(trans, prominence=0.02 * np.ptp(trans))
    assert len(peaks) >= 2
    order = np.argsort(props["prominences"])[::-1]
    top2 = sorted(peaks[order[:2]])
    return dps[top2[1]] - dps[top2[0]]


def test_b4_at_splitting_linearity():
    """B-4: splitting = Omega_RF at {10,20,40,80} MHz; slope 1.000 +- 0.5%."""
    rabis = 2 * np.pi * np.array([10e6, 20e6, 40e6, 80e6])
    measured = []
    for om in rabis:
        dps = np.linspace(-om, om, 4001)
        s = _cold_at_spectrum(om, 0.0, dps)
        split = _extract_splitting(dps, s.real)
        assert split == pytest.approx(om, rel=0.01), f"at {om/(2*np.pi)/1e6} MHz"
        measured.append(split)
    slope = np.polyfit(rabis, measured, 1)[0]
    assert slope == pytest.approx(1.0, abs=0.005)


def test_b5_detuned_rf_splitting():
    """B-5: splitting = sqrt(Omega^2 + Delta_RF^2) = 25 MHz for 20/15."""
    om, drf = 2 * np.pi * 20e6, 2 * np.pi * 15e6
    dps = 2 * np.pi * np.linspace(-30e6, 30e6, 6001)
    s = _cold_at_spectrum(om, drf, dps)
    split = _extract_splitting(dps, s.real)
    assert split == pytest.approx(np.hypot(om, drf), rel=0.01)


def test_b5_detuned_doublet_asymmetric_centroid():
    """Under RF detuning the doublet centroid shifts by -Delta_RF/2 (§2.4)."""
    om, drf = 2 * np.pi * 20e6, 2 * np.pi * 15e6
    dps = 2 * np.pi * np.linspace(-35e6, 35e6, 7001)
    s = _cold_at_spectrum(om, drf, dps)
    trans = -s.real
    peaks, props = find_peaks(trans, prominence=0.02 * np.ptp(trans))
    order = np.argsort(props["prominences"])[::-1]
    top2 = sorted(peaks[order[:2]])
    centroid = 0.5 * (dps[top2[0]] + dps[top2[1]])
    assert centroid == pytest.approx(-drf / 2, abs=2 * np.pi * 0.5e6)


def test_b7_invariants_and_transpose_canary():
    """B-7: trace/hermiticity/positivity + absorptive sign with asymmetric
    parameters (catches a silent row/column-major vec bug)."""
    sys = LadderSystem(
        omegas=[2 * np.pi * 0.2e6, 2 * np.pi * 5e6],
        deltas=[2 * np.pi * 3.7e6, 2 * np.pi * (-1.3e6)],
        decays=[0.0, GAMMA_E, G_R],
        dephasings=[0.0, 0.0, 2 * np.pi * 50e3],
        transit=2 * np.pi * 20e3,
    )
    rho = sys.steady_state()
    assert abs(np.trace(rho) - 1) < 1e-12
    assert np.allclose(rho, rho.conj().T, atol=1e-10)
    assert np.linalg.eigvalsh(rho).min() > -1e-10
    # absorptive sign: normalized response Re[rho_10/(i O_p/2)] > 0
    assert (rho[1, 0] / (1j * 2 * np.pi * 0.2e6 / 2)).real > 0


def test_b13_transit_channel_definition():
    """B-13: with all Omega = 0 and transit on, every coherence decays at
    gamma_t (NOT gamma_t/2) and populations refill g — the full
    measure-and-replace channel including the i=g Kraus operator."""
    gt = 2 * np.pi * 50e3
    sys = LadderSystem(
        omegas=[0.0, 0.0],
        deltas=[0.0, 0.0],
        decays=[0.0, 0.0, 0.0],   # isolate the transit channel
        transit=gt,
    )
    lv = sys.liouvillian()
    n = 3
    # populations: d rho_ii/dt = -gt rho_ii + (refill into rho_00)
    for i in range(1, n):
        idx = i * n + i
        assert lv[idx, idx].real == pytest.approx(-gt, rel=1e-12)
    # ground coherences rho_0j must decay at gt (full), not gt/2
    for j in range(1, n):
        idx = 0 * n + j
        assert lv[idx, idx].real == pytest.approx(-gt, rel=1e-12), (
            "ground-coherence transit decay must be gamma_t "
            "(measure-and-replace with i=g operator, spec 06 §2.2)")


def test_b14_decay_routing_insensitivity():
    """B-14: routing Rydberg decay to g vs cascade via e changes weak-probe
    chi by < 1e-3 relative (spec 06 §2.2 argument)."""
    dps = 2 * np.pi * np.linspace(-10e6, 10e6, 41)
    common = dict(
        omegas=[2 * np.pi * 1e3, 2 * np.pi * 5e6],
        deltas=[0.0, 0.0],
        decays=[0.0, GAMMA_E, 2 * np.pi * 2e3],
        dephasings=[0.0, 0.0, 2 * np.pi * 100e3],
    )
    for dp in dps[:: 8]:
        d = dict(common)
        d["deltas"] = [dp, 0.0]
        to_ground = LadderSystem(**d).steady_state()
        cascade = LadderSystem(**d, decay_to={2: {1: 2 * np.pi * 2e3}})
        casc = cascade.steady_state()
        a = to_ground[1, 0]
        b = casc[1, 0]
        assert abs(a - b) / abs(a) < 1e-3


def test_transit_on_analytic_lindblad_agreement():
    """The check that would have caught the transit bug: with transit != 0
    the analytic coherence-decay bookkeeping (G += gamma_t) must match the
    Lindblad steady state in the weak-probe limit."""
    gt = 2 * np.pi * 60e3
    p = LadderChiParams(
        omegas=np.array([2 * np.pi * 1e3, 2 * np.pi * 5e6]),
        deltas=np.array([0.0, 0.0]),
        k_signed=np.zeros(2),
        coherence_decays=np.array([GAMMA_E / 2 + gt,
                                   G_R / 2 + 2 * np.pi * 100e3 + gt]),
    )
    for dp in 2 * np.pi * np.array([-8e6, -2e6, 0.0, 1.5e6, 6e6]):
        ana = chi_ladder(p, np.array([dp]), 0.0)[0]
        sys = LadderSystem(
            omegas=[2 * np.pi * 1e3, 2 * np.pi * 5e6],
            deltas=[dp, 0.0],
            decays=[0.0, GAMMA_E, G_R],
            dephasings=[0.0, 0.0, 2 * np.pi * 100e3],
            transit=gt,
        )
        rho = sys.steady_state()
        num = rho[1, 0] / (1j * 2 * np.pi * 1e3 / 2)
        assert num.real == pytest.approx(ana.real, rel=2e-3)
        assert num.imag == pytest.approx(ana.imag, rel=2e-3,
                                         abs=1e-3 * abs(ana))
