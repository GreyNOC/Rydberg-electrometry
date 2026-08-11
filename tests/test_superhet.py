"""Validation: spectroscopy extraction + superhet transduction/noise chain."""

import numpy as np
import pytest

from rydsim.constants import HBAR
from rydsim.lindblad import LadderSystem
from rydsim.spectroscopy import eit_fwhm, measure_at_splitting
from rydsim.superhet import (
    compression_point,
    min_detectable_field,
    optimize_lo,
    shot_noise_psd,
    sql_field,
    transfer_slope,
)

GAMMA_E = 2 * np.pi * 6.07e6


def _at_spectrum(omega_rf, dps, deph=2 * np.pi * 150e3):
    """4-level AT spectrum with realistic Rydberg dephasing (transit etc.)."""
    out = []
    for dp in dps:
        omegas = [2 * np.pi * 10e3, 2 * np.pi * 4e6]
        deltas = [dp, 0.0]
        decays = [0.0, GAMMA_E, 2 * np.pi * 1e3]
        dephs = [0.0, 0.0, deph]
        if omega_rf > 0:
            omegas.append(omega_rf)
            deltas.append(0.0)
            decays.append(2 * np.pi * 1e3)
            dephs.append(deph)
        sys = LadderSystem(omegas=omegas, deltas=deltas, decays=decays,
                           dephasings=dephs)
        rho = sys.steady_state()
        out.append(-(rho[1, 0] / (1j * 2 * np.pi * 10e3 / 2)).real)
    return np.array(out)


def test_at_extraction_recovers_rabi():
    """Measured splitting tracks Omega_RF, with the known finite-Omega_c
    inward pull (~sqrt(O_RF^2 - O_c^2/2) plus peak-overlap attraction) —
    the systematic NIST corrects for in metrology mode. Assert recovery
    within 6% and that the bias direction is inward (measured <= O_RF)."""
    omega_rf = 2 * np.pi * 18e6
    dps = 2 * np.pi * np.linspace(-22e6, 22e6, 441)
    m = measure_at_splitting(dps, _at_spectrum(omega_rf, dps))
    assert m is not None and m.resolved
    assert m.splitting == pytest.approx(omega_rf, rel=0.06)
    assert m.splitting <= omega_rf * 1.005  # inward systematic, never outward


def test_unresolved_returns_none_or_unresolved():
    """Splitting below the (dephasing-set) EIT linewidth: no resolved claim.

    With ~150 kHz Rydberg dephasing the EIT window is several hundred kHz
    wide, so a 100 kHz AT splitting is genuinely unresolvable.
    """
    omega_rf = 2 * np.pi * 0.1e6
    dps = 2 * np.pi * np.linspace(-10e6, 10e6, 801)
    m = measure_at_splitting(dps, _at_spectrum(omega_rf, dps))
    assert m is None or not m.resolved


def test_eit_fwhm_positive_and_sane():
    dps = 2 * np.pi * np.linspace(-10e6, 10e6, 2001)
    absorption = -_at_spectrum(0.0, dps)  # helper returns -absorption
    w = eit_fwhm(dps, absorption)
    assert w is not None
    # dephasing-limited floor ~2*deph; power-broadening ceiling well below 10 MHz
    assert 2 * np.pi * 0.05e6 < w < 2 * np.pi * 10e6


# ---- transduction / noise ----

def test_transfer_slope_exact_on_polynomial():
    f = lambda e: 3.0 + 2.0 * e - 0.5 * e**2
    assert transfer_slope(f, 1.0) == pytest.approx(2.0 - 1.0, rel=1e-8)


def test_shot_noise_value():
    # 100 uW at 780 nm: S = 2 h nu P
    s = shot_noise_psd(100e-6, 780e-9)
    nu = 299792458.0 / 780e-9
    assert s == pytest.approx(2 * 6.62607015e-34 * nu * 100e-6, rel=1e-12)


def test_optimize_lo_on_synthetic_curve():
    """Shot-noise NEF optimum: minimizes sqrt(2 h nu P(E))/|dP/dE|.

    NOT the max-slope point: shot noise grows with sqrt(P), pulling the
    optimum toward lower transmission. Verify against brute-force minimum
    of the analytic NEF over a dense grid.
    """
    p0, w = 200e-6, 1.0  # W, (V/m)
    transfer = lambda e: p0 * (1 - 0.5 * np.exp(-(e / w) ** 2))
    grid = np.linspace(0.05, 3, 120)
    op = optimize_lo(transfer, grid, 780e-9,
                     rin_db_per_hz=-300, detector_nep_w_per_rthz=0.0)

    # brute-force analytic NEF on a dense grid
    dense = np.linspace(0.02, 3, 3000)
    slope_a = p0 * (dense / w**2) * np.exp(-(dense / w) ** 2)  # exact dP/dE
    nef_a = np.sqrt(shot_noise_psd(transfer(dense), 780e-9)) / slope_a
    e_star = dense[np.argmin(nef_a)]

    assert op.e_lo == pytest.approx(e_star, abs=grid[1] - grid[0])
    # max-slope point (w/sqrt(2)) must NOT beat the returned optimum
    slope_ms = abs(transfer_slope(transfer, w / np.sqrt(2)))
    nef_ms = np.sqrt(shot_noise_psd(transfer(w / np.sqrt(2)), 780e-9)) / slope_ms
    assert op.nef <= nef_ms * (1 + 1e-9)
    # NEF formula self-consistency at the operating point
    slope = abs(transfer_slope(transfer, op.e_lo))
    assert op.nef == pytest.approx(
        np.sqrt(shot_noise_psd(transfer(op.e_lo), 780e-9)) / slope, rel=1e-6)


def test_min_detectable_field_scaling():
    assert min_detectable_field(1e-6, 4.0) == pytest.approx(0.5e-6)
    assert min_detectable_field(1e-6, 1.0, snr=3) == pytest.approx(3e-6)


def test_compression_point_detects_saturation():
    p0, w = 200e-6, 1.0
    transfer = lambda e: p0 * (1 - 0.5 * np.exp(-(e / w) ** 2))
    e_lo = w / np.sqrt(2)
    c = compression_point(transfer, e_lo, np.linspace(0.01, 2.0, 100))
    assert c is not None
    assert 0.05 * w < c < 1.5 * w  # compresses within the curve's scale


def test_sql_scaling_laws():
    """SQL: 1/sqrt(N), 1/sqrt(t), 1/d scaling and plausible magnitude."""
    d = 1000 * 8.478353619788951e-30  # 1000 ea0, typical Rydberg-Rydberg
    e1 = sql_field(d, 2 * np.pi * 100e3, 1e6, 1.0)
    assert sql_field(d, 2 * np.pi * 100e3, 4e6, 1.0) == pytest.approx(e1 / 2)
    assert sql_field(d, 2 * np.pi * 100e3, 1e6, 4.0) == pytest.approx(e1 / 2)
    assert sql_field(2 * d, 2 * np.pi * 100e3, 1e6, 1.0) == pytest.approx(e1 / 2)
    # magnitude: with N=1e6, gamma=2pi*100kHz, t=1s -> sub-uV/m class
    assert 1e-9 < e1 < 1e-5
