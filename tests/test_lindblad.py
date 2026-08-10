"""Validation: Lindblad engine vs analytic quantum-optics results."""

import numpy as np
import pytest

from rydsim.lindblad import LadderSystem, weak_probe_chi3


def test_trace_and_hermiticity_and_positivity():
    sys = LadderSystem(
        omegas=[2e6, 8e6, 3e6],
        deltas=[1e6, -2e6, 0.5e6],
        decays=[0.0, 38.1e6, 1e4, 5e3],
        dephasings=[0.0, 0.0, 2e5, 2e5],
        transit=1e5,
    )
    rho = sys.steady_state()
    assert np.trace(rho).real == pytest.approx(1.0, abs=1e-12)
    assert abs(np.trace(rho).imag) < 1e-12
    assert np.allclose(rho, rho.conj().T, atol=1e-12)
    evals = np.linalg.eigvalsh(rho)
    assert evals.min() > -1e-10  # positivity within numerical precision


def test_two_level_analytic_steady_state():
    """Two-level atom: rho_ee = (O^2/4) / (D^2 + G^2/4 + O^2/2) exactly."""
    omega, delta, gamma = 5e6, 3e6, 6e6
    sys = LadderSystem(omegas=[omega], deltas=[delta], decays=[0.0, gamma])
    rho = sys.steady_state()
    s = (omega**2 / 2) / (delta**2 + gamma**2 / 4)
    rho_ee_exact = 0.5 * s / (1 + s)
    assert rho[1, 1].real == pytest.approx(rho_ee_exact, rel=1e-12)
    # coherence: rho_eg = (i O / 2)(G/2 + i D)/(D^2 + G^2/4 + O^2/2) ... check |.|
    rho_eg_mag_exact = (omega / 2) * np.sqrt(delta**2 + gamma**2 / 4) / (
        delta**2 + gamma**2 / 4 + omega**2 / 2
    )
    assert abs(rho[1, 0]) == pytest.approx(rho_eg_mag_exact, rel=1e-12)


def test_three_level_weak_probe_matches_analytic():
    """Numerical Lindblad -> analytic weak-probe chi to high precision."""
    gamma_e = 2 * np.pi * 6.07e6       # intermediate decay [rad/s]
    gamma_r_pop = 2 * np.pi * 1e3      # Rydberg population decay
    gamma_r_deph = 2 * np.pi * 50e3    # extra Rydberg dephasing
    omega_c = 2 * np.pi * 10e6
    omega_p = 2 * np.pi * 1e3          # very weak probe
    delta_c = 0.0

    detunings = 2 * np.pi * np.linspace(-30e6, 30e6, 61)
    for dp in detunings:
        sys = LadderSystem(
            omegas=[omega_p, omega_c],
            deltas=[dp, delta_c],
            decays=[0.0, gamma_e, gamma_r_pop],
            dephasings=[0.0, 0.0, gamma_r_deph],
        )
        rho = sys.steady_state()
        num = rho[1, 0] / (1j * omega_p / 2)  # normalized response

        # analytic: ground-Rydberg coherence decay = pop/2 + pure dephasing
        gamma_r_coh = gamma_r_pop / 2 + gamma_r_deph
        ana = weak_probe_chi3(np.array([dp]), omega_c, delta_c,
                              gamma_e, gamma_r_coh)[0]
        # sign/conjugation convention: compare magnitudes and real/imag parts
        assert num.real == pytest.approx(ana.real, rel=2e-4, abs=1e-12 * abs(ana))
        assert abs(num.imag) == pytest.approx(abs(ana.imag), rel=2e-4, abs=1e-12 * abs(ana))


def test_eit_transparency_dip():
    """On two-photon resonance the absorption shows the EIT dip."""
    gamma_e = 2 * np.pi * 6e6
    omega_c = 2 * np.pi * 12e6
    omega_p = 2 * np.pi * 10e3

    def absorption(dp):
        sys = LadderSystem(
            omegas=[omega_p, omega_c],
            deltas=[dp, 0.0],
            decays=[0.0, gamma_e, 2 * np.pi * 1e3],
        )
        rho = sys.steady_state()
        return (rho[1, 0] / (1j * omega_p / 2)).real  # absorptive part

    a0 = absorption(0.0)
    a_side = absorption(omega_c / 2)   # near the AT peak
    a_far = absorption(2 * np.pi * 100e6)
    assert a0 < 0.1 * a_side          # deep transparency on resonance
    assert a_far < a_side             # far wings absorb little


def test_autler_townes_splitting_4level():
    """With a resonant RF field on the Rydberg link, EIT peak splits by ~Omega_RF."""
    gamma_e = 2 * np.pi * 6e6
    omega_c = 2 * np.pi * 4e6
    omega_p = 2 * np.pi * 10e3
    omega_rf = 2 * np.pi * 20e6

    dps = 2 * np.pi * np.linspace(-25e6, 25e6, 501)
    trans = []
    for dp in dps:
        sys = LadderSystem(
            omegas=[omega_p, omega_c, omega_rf],
            deltas=[dp, 0.0, 0.0],
            decays=[0.0, gamma_e, 2 * np.pi * 1e3, 2 * np.pi * 1e3],
        )
        rho = sys.steady_state()
        trans.append(-(rho[1, 0] / (1j * omega_p / 2)).real)  # ~ transmission
    trans = np.array(trans)

    # find the two transmission maxima (EIT peaks split by AT)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(trans)
    assert len(peaks) >= 2
    # take the two highest
    top2 = peaks[np.argsort(trans[peaks])[-2:]]
    splitting = abs(dps[top2[0]] - dps[top2[1]])
    assert splitting == pytest.approx(omega_rf, rel=0.05)


def test_input_validation():
    with pytest.raises(ValueError):
        LadderSystem(omegas=[1.0], deltas=[1.0, 2.0], decays=[0.0, 1.0])
    with pytest.raises(ValueError):
        LadderSystem(omegas=[1.0], deltas=[1.0], decays=[0.0])
