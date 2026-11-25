"""Validation of the co-site hidden-signal model (campaign T3).

Anchors, in order of independence from the code under test:
  1. exact identities on synthetic transfer curves (quadratic: zero
     desensitization and a single first-harmonic spur of known size);
  2. the E_I -> 0 limit recovering the single-tone superhet slope;
  3. brute-force two-tone DFT demodulation agreeing with the phase-average
     gain on the real engine's transfer curve;
  4. refusal paths: out-of-band offsets, spur collisions, extrapolation,
     unconverged quadrature, malformed inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from rydsim import cosite as cs
from rydsim.constants import AU_DIPOLE, HBAR
from rydsim.experiment import LadderConfig, converged_superhet_operating_point
from rydsim.provenance import IntegrityError

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

DOMAIN = (0.0, 10.0)


def quad(c: float = 3.0, a: float = 0.5):
    return (lambda e: a + c * np.asarray(e) ** 2,
            lambda e: 2 * c * np.asarray(e))


def linear(b: float = 2.0):
    return (lambda e: b * np.asarray(e, float),
            lambda e: np.full_like(np.asarray(e, float), b))


# --- 1. exact identities ---------------------------------------------------

@pytest.mark.parametrize("e_i", [0.1, 0.5, 0.99, 1.5, 3.0])
def test_quadratic_transfer_has_zero_desensitization(e_i):
    """T = a + cE^2: T'(A) e^{-i phi} = 2c conj(Z), whose average is 2c E_L
    for ANY interferer amplitude, including E_I > E_L."""
    t, dt = quad()
    g = cs.small_signal_gain(t, dt, 1.0, e_i, DOMAIN)
    assert g.gain == pytest.approx(2 * 3.0 * 1.0, rel=1e-12)


def test_quadratic_transfer_mean_power_is_exact():
    t, dt = quad(c=3.0, a=0.5)
    g = cs.small_signal_gain(t, dt, 1.0, 0.7, DOMAIN)
    assert g.mean_power == pytest.approx(0.5 + 3.0 * (1.0 + 0.49), rel=1e-12)


def test_quadratic_transfer_spurs_are_exact():
    """|Z|^2 = E_L^2 + E_I^2 + 2 E_L E_I cos(theta): one harmonic only."""
    t, _ = quad(c=3.0)
    spurs = cs.interferer_spurs(t, 1.0, 0.4, DOMAIN, n_harmonics=4)
    assert spurs[0] == pytest.approx(2 * 3.0 * 1.0 * 0.4, rel=1e-12)
    assert np.all(spurs[1:] < 1e-12)


def test_linear_envelope_detector_desensitizes():
    """A linear-in-|E| detector is NOT immune: the phase average of
    e^{-i phi} shrinks as the interferer grows — the mechanism exists even
    without curvature, so desensitization is not just compression."""
    t, dt = linear()
    gains = [cs.small_signal_gain(t, dt, 1.0, e, DOMAIN).gain
             for e in (0.0, 0.3, 0.6, 0.9)]
    assert gains[0] == pytest.approx(2.0)
    assert all(a > b for a, b in zip(gains, gains[1:]))


# --- 2./3. limits and cross-method agreement on the real engine -------------

@pytest.fixture(scope="module")
def engine():
    cfg = LadderConfig(rf_dipole=1000 * AU_DIPOLE)
    conv = converged_superhet_operating_point(
        cfg, cfg.rf_dipole / HBAR, 4.0, 100e-6)
    spl = conv.spline()
    return cfg, conv, spl, spl.derivative(), (0.0, float(conv.e_grid[-1]))


def test_zero_interferer_recovers_superhet_slope(engine):
    _, conv, spl, d, dom = engine
    g = cs.small_signal_gain(spl, d, conv.op.e_lo, 0.0, dom)
    assert g.gain == pytest.approx(abs(conv.op.slope), rel=1e-6)


def test_weak_interferer_is_continuous_with_zero(engine):
    _, conv, spl, d, dom = engine
    g0 = cs.small_signal_gain(spl, d, conv.op.e_lo, 0.0, dom).gain
    g = cs.small_signal_gain(spl, d, conv.op.e_lo, 1e-4 * conv.op.e_lo, dom).gain
    assert g == pytest.approx(g0, rel=1e-6)


@pytest.mark.parametrize("frac", [0.05, 0.3, 0.6, 0.85])
def test_phase_average_matches_brute_force_dft(engine, frac):
    _, conv, spl, d, dom = engine
    e_lo = conv.op.e_lo
    e_i = frac * e_lo
    analytic = cs.small_signal_gain(spl, d, e_lo, e_i, dom).gain
    brute = cs.two_tone_fft_gain(spl, e_lo, e_i, 1e-5 * e_lo, dom)
    assert brute == pytest.approx(analytic, rel=1e-6)


def test_desensitization_grows_with_interferer(engine):
    cfg, conv, spl, d, dom = engine
    e_lo = conv.op.e_lo
    pts = cs.cosite_scan(spl, d, e_lo, np.geomspace(1e-3, 0.9, 12) * e_lo,
                         dom, cfg.lambda_probe)
    ds = [p.desense_db for p in pts]
    assert ds[0] == pytest.approx(0.0, abs=1e-3)
    assert ds[-1] > 1.0
    assert all(b >= a - 1e-9 for a, b in zip(ds, ds[1:]))


def test_scan_noise_tracks_mean_power(engine):
    cfg, conv, spl, d, dom = engine
    e_lo = conv.op.e_lo
    (p,) = cs.cosite_scan(spl, d, e_lo, [0.5 * e_lo], dom, cfg.lambda_probe)
    from rydsim.superhet import shot_noise_psd
    assert p.budget.shot == pytest.approx(
        shot_noise_psd(p.mean_power, cfg.lambda_probe), rel=1e-12)
    assert p.nef == pytest.approx(np.sqrt(p.budget.total) / p.gain, rel=1e-12)


def test_ibw_is_positive_and_finite(engine):
    cfg, conv, *_ = engine
    ibw = cs.instantaneous_bandwidth_hz(cfg, conv.op.e_lo,
                                        cfg.rf_dipole / HBAR)
    assert np.isfinite(ibw) and 1e3 < ibw < 1e9


# --- blocking-level interpolation -----------------------------------------

def _pts(es, ds):
    from rydsim.superhet import NoiseBudget
    nb = NoiseBudget(0.0, 0.0, 0.0)
    return [cs.CositePoint(e, 1.0, dd, 1.0, nb, 1.0, 1.0, 0.0)
            for e, dd in zip(es, ds)]


def test_blocking_level_log_interpolates():
    pts = _pts([0.01, 0.1], [0.0, 2.0])
    assert cs.blocking_level(pts, 1.0) == pytest.approx(np.sqrt(0.01 * 0.1))


def test_blocking_level_none_when_not_reached():
    assert cs.blocking_level(_pts([0.01, 0.1], [0.1, 0.5]), 1.0) is None


def test_blocking_level_rejects_unsorted_grid():
    with pytest.raises(ValueError):
        cs.blocking_level(_pts([0.1, 0.01], [0.0, 2.0]))


# --- 4. refusals -------------------------------------------------------------

def test_out_of_band_offset_refused():
    with pytest.raises(IntegrityError, match="Floquet"):
        cs.require_quasi_static([1e3, 2e6], ibw_hz=1e6)


def test_in_band_offsets_accepted():
    cs.require_quasi_static([1e3, -5e4], ibw_hz=1e6)


@pytest.mark.parametrize("ibw", [0.0, -1.0, float("nan"), float("inf")])
def test_unusable_ibw_refused(ibw):
    with pytest.raises(IntegrityError):
        cs.require_quasi_static([1.0], ibw_hz=ibw)


@pytest.mark.parametrize("sig,intf", [(50e3, 50e3), (100e3, 50e3),
                                      (-150e3, 50e3)])
def test_signal_on_interferer_harmonic_refused(sig, intf):
    with pytest.raises(IntegrityError, match="harmonic"):
        cs.require_no_spur_collision(sig, intf, n_harmonics=5)


def test_signal_between_harmonics_accepted():
    cs.require_no_spur_collision(13e3, 50e3, n_harmonics=5)


def test_zero_offset_rejected():
    with pytest.raises(ValueError):
        cs.require_no_spur_collision(0.0, 50e3, n_harmonics=3)


def test_envelope_outside_transfer_curve_refused():
    t, dt = quad()
    with pytest.raises(IntegrityError, match="extrapolat"):
        cs.small_signal_gain(t, dt, 1.0, 0.5, (0.6, 10.0))


def test_envelope_through_zero_fails_convergence_gate():
    """E_I = E_L drives the envelope through zero: the phase jumps by pi and
    the trapezoid rule only converges algebraically. Must refuse."""
    t, dt = linear()
    with pytest.raises(IntegrityError, match="converge"):
        cs.small_signal_gain(t, dt, 1.0, 1.0, DOMAIN, n_max=1 << 12)


@pytest.mark.parametrize("e_lo,e_i", [(0.0, 0.1), (1.0, -0.1)])
def test_malformed_amplitudes_rejected(e_lo, e_i):
    t, dt = quad()
    with pytest.raises(ValueError):
        cs.small_signal_gain(t, dt, e_lo, e_i, DOMAIN)


def test_fft_rejects_colliding_tone_counts():
    t, _ = quad()
    with pytest.raises(ValueError):
        cs.two_tone_fft_gain(t, 1.0, 0.1, 1e-4, DOMAIN, m_i=6, m_s=3)


def test_scan_rejects_nonpositive_integration():
    t, dt = quad()
    with pytest.raises(ValueError):
        cs.cosite_scan(t, dt, 1.0, [0.1], DOMAIN, 780e-9, integration_time=0)


# --- weak-interferer law -----------------------------------------------------

def test_weak_interferer_desensitization_is_quadratic(engine):
    """The O(E_I) term averages to zero over the interferer phase, so a
    smooth transfer curve must desensitize as (E_I/E_LO)^2."""
    cfg, conv, spl, d, dom = engine
    e_lo = conv.op.e_lo
    pts = cs.cosite_scan(spl, d, e_lo, np.geomspace(3e-3, 0.05, 10) * e_lo,
                         dom, cfg.lambda_probe)
    law = cs.small_interferer_law(pts, e_lo)
    assert law is not None
    assert law[0] == pytest.approx(2.0, abs=0.05)


def test_weak_interferer_law_needs_three_points():
    assert cs.small_interferer_law(_pts([0.01, 0.02], [1e-3, 4e-3]), 1.0) is None


# --- CLI ----------------------------------------------------------------------

def _run(argv):
    import warnings

    from rydsim.cli import main
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return main(argv)


def test_cli_cosite_writes_finding(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RYDSIM_FINDINGS_DIR", str(tmp_path))
    rc = _run(["cosite", "--rf-dipole-ea0", "1000", "--ei-points", "8",
               "--finding"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "1 dB blocking" in out and "cross-check" in out
    (md,) = tmp_path.glob("t3-cosite-*.md")
    (js,) = tmp_path.glob("t3-cosite-*.json")
    import json
    data = json.loads(js.read_text(encoding="utf-8"))
    assert data["cross_checks"][0]["spread_rel"] <= 1e-4
    (scan,) = [v for k, v in data["result"].items() if k.startswith("scan")]
    assert len(scan) == 8
    assert "Floquet" in md.read_text(encoding="utf-8")


@pytest.mark.parametrize("extra,msg", [
    (["--interferer-offset-khz", "5000"], "Floquet"),
    (["--signal-offset-khz", "100", "--interferer-offset-khz", "50"],
     "harmonic"),
])
def test_cli_cosite_refuses_invalid_physics(capsys, extra, msg):
    rc = _run(["cosite", "--rf-dipole-ea0", "1000", "--ei-points", "4",
               *extra])
    assert rc == 3
    assert msg in capsys.readouterr().err


@pytest.mark.parametrize("extra", [
    ["--ei-min-frac", "0.5", "--ei-max-frac", "0.1"],
    ["--ei-max-frac", "1.2"],
    ["--ei-points", "1"],
    ["--integration-s", "0"],
])
def test_cli_cosite_rejects_bad_arguments(capsys, extra):
    assert _run(["cosite", "--rf-dipole-ea0", "1000", *extra]) == 2


def test_cli_cosite_requires_dipole(capsys):
    assert _run(["cosite"]) == 2
