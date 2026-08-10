"""DESIGNER search layer (D1-D3): sampler + campaign + frontier contracts."""

import numpy as np
import pytest

from rydsim.objective import SensorDesign
from rydsim.search import (
    frontier_table,
    grid_designs,
    random_designs,
    run_campaign,
)

BASE = SensorDesign(rf_dipole_ea0=1000.0)


def test_grid_designs_cartesian_product():
    ds = grid_designs(BASE, coupling_rabi_hz=[2e6, 5e6],
                      temperature_k=[300.0, 330.0])
    assert len(ds) == 4
    assert {d.coupling_rabi_hz for d in ds} == {2e6, 5e6}
    assert {d.temperature_k for d in ds} == {300.0, 330.0}
    assert all(d.rf_dipole_ea0 == 1000.0 for d in ds)  # base preserved


def test_unknown_axis_rejected():
    with pytest.raises(ValueError):
        grid_designs(BASE, not_a_field=[1, 2])
    with pytest.raises(ValueError):
        random_designs(BASE, 4, {"nope": (1, 2)})


def test_random_designs_deterministic_and_in_range():
    kw = dict(ranges={"coupling_rabi_hz": (1e6, 10e6)},
              log_scale={"coupling_rabi_hz"})
    a = random_designs(BASE, 8, seed=7, **kw)
    b = random_designs(BASE, 8, seed=7, **kw)
    assert [d.coupling_rabi_hz for d in a] == [d.coupling_rabi_hz for d in b]
    assert all(1e6 <= d.coupling_rabi_hz <= 10e6 for d in a)
    # different seed gives a different draw
    c = random_designs(BASE, 8, seed=8, **kw)
    assert [d.coupling_rabi_hz for d in a] != [d.coupling_rabi_hz for d in c]


def test_latin_hypercube_stratification():
    """Each of n strata is hit exactly once (the point of LHS)."""
    n = 16
    ds = random_designs(BASE, n, {"temperature_k": (300.0, 340.0)}, seed=3)
    vals = np.array([d.temperature_k for d in ds])
    bins = ((vals - 300.0) / 40.0 * n).astype(int)
    assert sorted(bins) == list(range(n))


def test_campaign_reports_infeasible_reasons():
    designs = [BASE, SensorDesign()]  # second has no dipole -> refused
    res = run_campaign(designs, lo_points=24)
    assert res.n_total == 2 and res.n_feasible == 1
    assert sum(res.infeasible_reasons.values()) == 1
    assert "campaign: 1/2 designs feasible" in res.summary()


def test_campaign_frontier_and_table():
    designs = grid_designs(BASE, coupling_rabi_hz=[2e6, 5e6, 9e6])
    res = run_campaign(designs, lo_points=24)
    assert res.n_feasible == 3
    front = res.frontier()
    assert 1 <= len(front) <= 3
    # every frontier point is non-dominated in (NEF, -IBW)
    for a in front:
        for b in res.feasible:
            if b is a:
                continue
            assert not (b.nef <= a.nef and b.ibw_hz >= a.ibw_hz
                        and (b.nef < a.nef or b.ibw_hz > a.ibw_hz))
    table = frontier_table(front)
    assert "NEF [nV/cm/rtHz]" in table and "IBW [MHz]" in table


def test_sensitivity_bandwidth_tradeoff_is_visible():
    """The T2/DESIGNER premise: sweeping coupling Rabi trades NEF against
    IBW, so the frontier has more than one point (if it collapses to one,
    the objective is not resolving the trade)."""
    designs = grid_designs(BASE, coupling_rabi_hz=list(np.geomspace(1e6, 15e6, 8)))
    res = run_campaign(designs, lo_points=24)
    assert res.n_feasible >= 6
    nefs = np.array([e.nef for e in res.feasible])
    ibws = np.array([e.ibw_hz for e in res.feasible])
    # a genuine trade-off: the best-NEF point is not also the best-IBW point
    assert np.argmin(nefs) != np.argmax(ibws)
    assert len(res.frontier()) >= 2
