"""LO-grid convergence of the superhet operating point.

Regression for a defect in the CLI/GUI superhet path: a linear 60-point LO
grid on [0.05, 20] V/m (0.34 V/m spacing) skipped the low-field optimum and
reported 8.9 nV/cm/rtHz for the Rb-87 / 1000 ea0 default against a
converged 1.12 at E_LO ~ 0.31 V/m. The design layer (objective.evaluate)
was not affected: it already scans a log grid scaled to Omega_c.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from rydsim.cli import main as cli_main
from rydsim.constants import AU_DIPOLE, HBAR
from rydsim.experiment import (
    LadderConfig,
    converged_superhet_operating_point,
    superhet_transfer,
)
from rydsim.provenance import IntegrityError
from rydsim.superhet import optimize_lo

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture(scope="module")
def cfg() -> LadderConfig:
    return LadderConfig(rf_dipole=1000 * AU_DIPOLE)


@pytest.fixture(scope="module")
def direct_optimum(cfg: LadderConfig):
    """Spline-free reference: exact transfer curve, finite-difference slope,
    dense LO scan bracketing the optimum."""
    e2w = cfg.rf_dipole / HBAR

    def direct(e: float) -> float:
        return float(superhet_transfer(cfg, e2w, np.array([e]), 100e-6)[0])

    return optimize_lo(direct, np.linspace(0.1, 0.7, 61), cfg.lambda_probe)


def test_converged_nef_matches_spline_free_reference(cfg, direct_optimum):
    conv = converged_superhet_operating_point(
        cfg, cfg.rf_dipole / HBAR, 20.0, 100e-6, e_min=0.002)
    assert conv.nef_rel_change <= 1e-2
    assert conv.op.nef == pytest.approx(direct_optimum.nef, rel=5e-3)
    assert conv.op.e_lo == pytest.approx(direct_optimum.e_lo, rel=0.1)


def test_old_linear_grid_overstated_nef(cfg, direct_optimum):
    """Document the magnitude of the defect being fixed: the former default
    grid is >5x pessimistic. If this ever stops failing the old way, the
    regression above is no longer guarding anything meaningful."""
    from scipy.interpolate import CubicSpline

    e2w = cfg.rf_dipole / HBAR
    grid = np.linspace(0.05, 20.0, 60)
    spl = CubicSpline(grid, superhet_transfer(cfg, e2w, grid, 100e-6))
    old = optimize_lo(lambda e: float(spl(e)), grid[2:-2], cfg.lambda_probe)
    assert old.nef / direct_optimum.nef > 5.0


def test_grid_starts_at_zero_field(cfg):
    conv = converged_superhet_operating_point(
        cfg, cfg.rf_dipole / HBAR, 4.0, 100e-6)
    assert conv.e_grid[0] == 0.0
    assert np.all(np.diff(conv.e_grid) > 0)


def test_optimum_on_boundary_is_refused(cfg):
    # optimum is near 0.31 V/m; a scan capped at 0.1 V/m cannot contain it
    with pytest.raises(IntegrityError, match="boundary"):
        converged_superhet_operating_point(
            cfg, cfg.rf_dipole / HBAR, 0.1, 100e-6, e_min=0.001)


def test_unconverged_refinement_is_refused(cfg):
    with pytest.raises(IntegrityError, match="did not converge"):
        converged_superhet_operating_point(
            cfg, cfg.rf_dipole / HBAR, 20.0, 100e-6, max_doublings=0)


@pytest.mark.parametrize("kw", [
    {"e_max": 0.0}, {"e_max": 1.0, "n_points": 4},
    {"e_max": 1.0, "e_min": 2.0}, {"e_max": 1.0, "e_min": 0.0},
])
def test_rejects_malformed_grid_requests(cfg, kw):
    e_max = kw.pop("e_max")
    with pytest.raises(ValueError):
        converged_superhet_operating_point(
            cfg, cfg.rf_dipole / HBAR, e_max, 100e-6, **kw)


def test_cli_superhet_reports_converged_nef(capsys, direct_optimum):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rc = cli_main(["superhet", "--rf-dipole-ea0", "1000"])
    assert rc == 0
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if ln.startswith("NEF:"))
    nv = float(line.split("=")[1].split()[0])
    assert nv == pytest.approx(direct_optimum.nef / 100 * 1e9, rel=5e-3)


def test_cli_superhet_refuses_boundary_optimum(capsys):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rc = cli_main(["superhet", "--rf-dipole-ea0", "1000",
                       "--e-min", "0.001", "--e-max", "0.1"])
    assert rc == 3
    assert "refused" in capsys.readouterr().err
