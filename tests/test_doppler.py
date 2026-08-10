"""Validation: Doppler-averaged EIT — emergent hot-vapor physics.

Runs on the converged analytic engine (rydsim.eit via the experiment
facade). The key test: the wavelength-mismatch AT-splitting factor
k_p/k_c = lambda_c/lambda_p must EMERGE from the velocity average — it is
not coded anywhere in the physics chain.
"""

import numpy as np
import pytest
from scipy.signal import find_peaks

from rydsim.doppler import doppler_fwhm
from rydsim.experiment import LadderConfig, spectrum

M_RB87 = 86.909 * 1.66053906892e-27  # kg
L_PROBE = 780.241e-9
L_COUPL = 480.0e-9
KP = 2 * np.pi / L_PROBE
KC = 2 * np.pi / L_COUPL


def _cfg(omega_rf=0.0, T=300.0):
    return LadderConfig(
        omega_probe=2 * np.pi * 50e3,
        omega_coupling=2 * np.pi * 5e6,
        omega_rf=omega_rf,
        gamma_e=2 * np.pi * 6.07e6,
        gamma_r=2 * np.pi * 2e3,
        gamma_rp=2 * np.pi * 1e3,
        deph_r=2 * np.pi * 100e3,
        temperature=T,
        mass=M_RB87,
        lambda_probe=L_PROBE,
        lambda_coupling=L_COUPL,
    )


def test_doppler_fwhm_rb_d2():
    """Rb D2 Doppler FWHM at 300 K ~ 0.5 GHz (standard textbook value)."""
    fwhm = doppler_fwhm(L_PROBE, M_RB87, 300.0)
    assert 0.45e9 < fwhm < 0.55e9


def test_eit_feature_is_sub_doppler():
    """Averaged EIT transparency structure is MHz-class, orders below the
    ~500 MHz Doppler width."""
    cfg = _cfg()
    dps = 2 * np.pi * np.linspace(-40e6, 40e6, 321)
    spec = spectrum(cfg, dps, check_convergence=True)
    absorption = spec.real
    i0 = len(dps) // 2
    # transparency dip at line center relative to nearby background
    bg = np.max(absorption)
    dip_depth = bg - absorption[i0]
    assert dip_depth > 0.15 * bg
    # width of the transparency feature
    half = absorption[i0] + dip_depth / 2
    above = np.where(absorption[i0:] > half)[0]
    hw = dps[i0 + above[0]] - dps[i0] if len(above) else np.inf
    doppler_w = 2 * np.pi * doppler_fwhm(L_PROBE, M_RB87, 300.0)
    assert hw < 0.05 * doppler_w, f"half-width {hw/doppler_w:.3f} of Doppler FWHM"


def test_wavelength_mismatch_at_scaling_emerges():
    """Probe-scanned AT splitting = (k_p/k_c) * Omega_RF in a hot cell.

    k_p/k_c = lambda_c/lambda_p = 480/780.24 = 0.6152 — the NIST correction
    factor for Doppler-mismatched ladders, emerging from the average alone.
    With the converged engine this must hold to ~1%.
    """
    omega_rf = 2 * np.pi * 25e6
    cfg = _cfg(omega_rf=omega_rf)
    dps = 2 * np.pi * np.linspace(-15e6, 15e6, 601)
    spec = spectrum(cfg, dps, check_convergence=True)
    transmission = -spec.real
    peaks, props = find_peaks(transmission,
                              prominence=0.05 * np.ptp(transmission))
    # the two dominant transparency peaks of the AT doublet
    assert len(peaks) >= 2, "AT doublet not resolved"
    order = np.argsort(props["prominences"])[::-1]
    top2 = sorted(peaks[order[:2]])
    splitting = dps[top2[1]] - dps[top2[0]]
    expected = (KP / KC) * omega_rf
    assert splitting == pytest.approx(expected, rel=0.02), (
        f"emergent mismatch factor {splitting/omega_rf:.4f}, "
        f"expected {KP/KC:.4f}"
    )


def test_cold_atom_limit_recovers_unscaled_splitting():
    """With Doppler off the mismatch factor disappears: splitting = Omega_RF."""
    omega_rf = 2 * np.pi * 25e6
    cfg = _cfg(omega_rf=omega_rf)
    cfg.doppler = False
    dps = 2 * np.pi * np.linspace(-25e6, 25e6, 501)
    spec = spectrum(cfg, dps)
    transmission = -spec.real
    peaks, _ = find_peaks(transmission, prominence=0.02 * np.ptp(transmission))
    assert len(peaks) >= 2
    top2 = sorted(peaks[np.argsort(transmission[peaks])[-2:]])
    splitting = dps[top2[1]] - dps[top2[0]]
    # finite-Omega_c systematic pulls peaks inward (~sqrt(O^2 - O_c^2/2) plus
    # overlap attraction): tolerate 5%, and the bias must be inward only
    assert splitting == pytest.approx(omega_rf, rel=0.05)
    assert splitting <= omega_rf * 1.005


def test_co_propagating_beams_larger_mismatch():
    """Co-propagating probe/coupling: two-photon Doppler ADDS (k_p + k_c),
    EIT contrast collapses relative to counter-propagating — the reason
    every experiment counter-propagates."""
    cfg_counter = _cfg()
    cfg_co = _cfg()
    cfg_co.counter_propagating = False
    dps = 2 * np.pi * np.linspace(-10e6, 10e6, 161)
    s_counter = spectrum(cfg_counter, dps)
    s_co = spectrum(cfg_co, dps)

    def dip_contrast(a):
        i0 = len(a) // 2
        return (np.max(a.real) - a.real[i0]) / np.max(a.real)

    assert dip_contrast(s_counter) > 2 * dip_contrast(s_co)
