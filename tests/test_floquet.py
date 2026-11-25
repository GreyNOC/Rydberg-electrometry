"""Validation of the two-tone Floquet (harmonic-balance) solver.

Independent anchors:
  1. direct time integration of the OBE (commensurate tones, DFT) at an
     offset far outside the quasi-static regime;
  2. the quasi-static phase-average model (rydsim.cosite) on the same thin,
     single-class transfer curve, approached as (Delta/IBW)^2;
  3. the second-order AC-Stark law at large offsets;
  4. structural identities of the periodic state (trace, hermiticity);
plus every refusal path.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from rydsim import floquet as fq
from rydsim.experiment import LadderConfig
from rydsim.lindblad import LadderSystem, ibw_3db
from rydsim.provenance import IntegrityError

TAU = 2 * np.pi


def _base(w_l: float, dc: float = 0.0, drf: float = 0.0) -> LadderSystem:
    c = LadderConfig()
    return LadderSystem(
        omegas=[c.omega_probe, c.omega_coupling, w_l], deltas=[0.0, dc, drf],
        decays=[0.0, c.gamma_e, c.gamma_r, c.gamma_rp],
        dephasings=[0.0, 0.0, c.deph_r, c.deph_r])


@pytest.fixture(scope="module")
def thin_lo() -> float:
    return fq.optimal_thin_lo(_base(TAU * 4e6), 4 * LadderConfig().omega_coupling)


@pytest.fixture(scope="module")
def solver(thin_lo) -> fq.TwoToneFloquet:
    return fq.TwoToneFloquet(_base(thin_lo))


@pytest.fixture(scope="module")
def grid(thin_lo) -> np.ndarray:
    return np.geomspace(0.01, 1000, 121) * thin_lo


# --- 1. direct OBE time integration --------------------------------------

def test_matches_time_domain_obe_far_out_of_band():
    w_l = TAU * 4e6
    s = fq.TwoToneFloquet(_base(w_l))
    w0 = TAU * 0.4e6
    d_i, d_s = 5 * w0, 2 * w0           # 2 MHz interferer, 0.8 MHz signal
    w_i, w_s = 0.5 * w_l, 1e-4 * w_l
    l0, cu, cd = s.l0, s.c_up, s.c_down

    def rhs(t, y):
        lt = (l0 + w_i * (cu * np.exp(1j * d_i * t) + cd * np.exp(-1j * d_i * t))
              + w_s * (cu * np.exp(1j * d_s * t) + cd * np.exp(-1j * d_s * t)))
        return lt @ y

    ev = np.linalg.eigvals(l0)
    slow = np.min(-ev.real[ev.real < -1e-6 * s.scale])
    t0 = TAU / w0
    n_set = int(np.ceil(14 / slow / t0))
    y0 = _base(w_l).steady_state().reshape(-1).astype(complex)
    y1 = solve_ivp(rhs, (0, n_set * t0), y0, method="DOP853",
                   rtol=1e-11, atol=1e-14).y[:, -1]
    ns = 2048
    ts = n_set * t0 + np.arange(ns) * t0 / ns
    ys = solve_ivp(rhs, (n_set * t0, (n_set + 1) * t0), y1, method="DOP853",
                   rtol=1e-11, atol=1e-14, t_eval=ts).y
    x = np.imag(ys.reshape(4, 4, -1)[1, 0, :])
    ph = np.exp(-1j * np.arange(ns) * TAU / ns)

    def tone(m: int) -> float:
        return 2 * abs(np.mean(x * ph**m))

    r = s.response(w_i, d_i, d_s, n_harmonics=3)
    assert tone(2) / w_s == pytest.approx(r.gain, rel=1e-5)
    for k in (1, 2, 3):
        assert tone(5 * k) == pytest.approx(r.harmonics_im_rho10[k - 1], rel=1e-6)


# --- 2. quasi-static limit -------------------------------------------------

def test_converges_to_quasi_static_as_offset_squared(solver, thin_lo, grid):
    """Blocking error vs the phase-average model scales as (Delta/IBW)^2 and
    is ~7% at 0.1 IBW — the measurement behind cosite's 0.02 x IBW gate."""
    ibw = ibw_3db(solver.base, 3, TAU * 100e6) / TAU
    qs = fq.quasi_static_thin_blocking(solver.base, grid)
    errs = {}
    for frac in (0.005, 0.02, 0.1):
        b, _ = fq.blocking_rabi(solver, TAU * frac * ibw, TAU * 10.0, grid)
        errs[frac] = abs(b - qs) / qs
    assert errs[0.005] < 5e-4
    assert errs[0.02] < 5e-3
    assert 0.03 < errs[0.1] < 0.12
    assert errs[0.02] / errs[0.005] == pytest.approx(16.0, rel=0.25)


def test_no_interferer_gain_is_the_static_slope(solver, thin_lo):
    r = solver.response(0.0, TAU * 1e6, TAU * 10.0)
    t = fq.thin_transfer(solver.base, np.array([thin_lo * (1 - 1e-4),
                                                thin_lo * (1 + 1e-4)]))
    slope = abs(t[1] - t[0]) / (2e-4 * thin_lo)
    assert r.gain == pytest.approx(slope, rel=1e-5)


# --- 3. AC-Stark asymptote ---------------------------------------------------

@pytest.mark.parametrize("f_hz", [500e6, 2e9])
def test_far_offset_blocking_follows_ac_stark_law(solver, thin_lo, grid, f_hz):
    s_crit = fq.ac_stark_shift_crit(solver.base)
    b, _ = fq.blocking_rabi(solver, TAU * f_hz, TAU * 10.0, grid)
    assert b == pytest.approx(np.sqrt(4 * TAU * f_hz * s_crit), rel=5e-3)


def test_ac_stark_law_explains_the_mean_state(solver, thin_lo):
    """The light-shift mapping (-s, +2s) reproduces the interferer's effect on
    the mean probe absorption, not just the blocking level."""
    d = TAU * 2e9
    w_i = 40 * thin_lo
    s = w_i**2 / (4 * d)
    r = solver.response(w_i, d, TAU * 10.0)
    static = np.imag(_base(thin_lo, -s, 2 * s).steady_state()[1, 0])
    assert r.mean_im_rho10 == pytest.approx(static, rel=1e-3)


def test_far_offset_rejection_is_10_db_per_decade(solver, grid):
    offs = TAU * np.array([200e6, 2e9])
    b = [fq.blocking_rabi(solver, o, TAU * 10.0, grid)[0] for o in offs]
    assert 20 * np.log10(b[1] / b[0]) == pytest.approx(10.0, abs=0.1)


def test_slow_mode_blocks_offset_signals_earlier(solver, grid):
    """The light shift opens a slow response mode: a 13 kHz hidden signal is
    blocked at a lower interferer level than a near-DC one."""
    d = TAU * 50e6
    b_dc, _ = fq.blocking_rabi(solver, d, TAU * 10.0, grid)
    b_13k, _ = fq.blocking_rabi(solver, d, TAU * 13e3, grid)
    assert b_13k < 0.97 * b_dc


# --- 4. structure of the periodic state --------------------------------------

def test_periodic_state_is_normalized_and_hermitian(solver, thin_lo):
    n_sb = 24
    rho = solver.periodic_state(0.8 * thin_lo, TAU * 3e6, n_sb)
    assert np.trace(rho[n_sb]) == pytest.approx(1.0, abs=1e-12)
    for k in range(1, n_sb + 1):
        assert abs(np.trace(rho[n_sb + k])) < 1e-12
        np.testing.assert_allclose(rho[n_sb - k], rho[n_sb + k].conj().T,
                                   atol=1e-12)


# --- refusals and input validation -------------------------------------------

def test_signal_far_inside_first_harmonic_is_not_a_collision(solver, thin_lo):
    """Regression: delta_s / Delta ~ 1e-9 once rounded to 'harmonic 0'."""
    r = solver.response(0.5 * thin_lo, TAU * 1e9, TAU * 1.0)
    assert r.gain > 0


@pytest.mark.parametrize("k", [1, 2, 3])
def test_signal_on_interferer_harmonic_refused(solver, thin_lo, k):
    with pytest.raises(IntegrityError, match="harmonic"):
        solver.response(0.5 * thin_lo, TAU * 1e6, k * TAU * 1e6)


@pytest.mark.parametrize("w_i,d_i,d_s", [
    (-1.0, 1e6, 1e3), (1.0, 0.0, 1e3), (1.0, 1e6, 0.0), (0.0, 1e6, 0.0)])
def test_malformed_tones_rejected(solver, w_i, d_i, d_s):
    with pytest.raises(ValueError):
        solver.response(w_i, d_i, d_s)


def test_unconverged_truncation_refused(solver, thin_lo):
    with pytest.raises(IntegrityError, match="truncation"):
        solver.response(thin_lo, TAU * 100.0, TAU * 37.0, n_max=64)


@pytest.mark.parametrize("bad", [
    np.array([1.0]), np.array([2.0, 1.0]), np.array([0.0, 1.0])])
def test_blocking_grid_validated(solver, bad):
    with pytest.raises(ValueError):
        fq.blocking_rabi(solver, TAU * 1e6, TAU * 10.0, bad)


def test_blocking_not_reached_returns_none(solver, thin_lo):
    b, scan = fq.blocking_rabi(solver, TAU * 1e9, TAU * 10.0,
                               np.array([0.01, 0.02]) * thin_lo)
    assert b is None and len(scan) == 2


def test_thin_lo_boundary_refused():
    with pytest.raises(IntegrityError, match="boundary"):
        fq.optimal_thin_lo(_base(TAU * 1e6), TAU * 0.1e6)


def test_link_validation():
    with pytest.raises(ValueError):
        fq.TwoToneFloquet(_base(TAU * 1e6), rf_link=4)


def test_ac_stark_needs_link_above_probe():
    with pytest.raises(ValueError):
        fq.ac_stark_shift_crit(_base(TAU * 1e6), rf_link=1)


# --- CLI ---------------------------------------------------------------------

def _run(argv):
    import warnings

    from rydsim.cli import main
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return main(argv)


def test_cli_selectivity_requires_no_doppler(capsys):
    assert _run(["selectivity"]) == 2
    assert "--no-doppler" in capsys.readouterr().err


@pytest.mark.parametrize("extra", [
    ["--offsets", "1"], ["--offset-min-khz", "0"],
    ["--offset-min-khz", "5000", "--offset-max-mhz", "1"],
    ["--signal-offset-khz", "0"], ["--ei-max-ratio", "0.001"],
    ["--ei-points", "1"]])
def test_cli_selectivity_rejects_bad_arguments(extra):
    assert _run(["selectivity", "--no-doppler", *extra]) == 2


def test_cli_selectivity_writes_checked_finding(tmp_path, monkeypatch, capsys):
    import json

    monkeypatch.setenv("RYDSIM_FINDINGS_DIR", str(tmp_path))
    rc = _run(["selectivity", "--no-doppler", "--rf-dipole-ea0", "1000",
               "--offsets", "4", "--offset-min-khz", "100",
               "--offset-max-mhz", "200", "--ei-max-ratio", "300",
               "--ei-points", "61", "--finding"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "far-offset rejection" in out
    (js,) = tmp_path.glob("t3-selectivity-*.json")
    data = json.loads(js.read_text(encoding="utf-8"))
    assert len(data["cross_checks"]) == 2
    assert all(c["spread_rel"] <= c["tolerance_rel"] for c in data["cross_checks"])
