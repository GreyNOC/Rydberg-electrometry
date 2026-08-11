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


def test_nef_is_u_shaped_in_coupling_rabi_and_the_trade_is_weak():
    """The T2/DESIGNER premise, stated as the physics actually observed.

    The original form of this test swept Omega_c over 1-15 MHz and asserted
    only `argmin(NEF) != argmax(IBW)` — a weak proxy for "a trade exists".
    It passed for the wrong reason: the probe-absorption chain used the
    closed-cycling dipole and omitted the ground hyperfine fraction, making
    the medium 2.40x too absorbing (audit MED-22 / spec 10 R10-10) and
    pushing the NEF optimum DOWN into the swept range. Correcting the
    absorption moved the optimum to ~13 MHz, i.e. to the top edge of the old
    sweep, so both optima landed on the same design and the proxy failed —
    correctly, because the sweep no longer bracketed the optimum.

    What is actually true, and what this now asserts:
      1. NEF is U-SHAPED in Omega_c — an interior optimum exists. Below it,
         NEF and IBW improve TOGETHER (there is no trade at all); above it
         they oppose.
      2. On the trade branch the coupling is WEAK: NEF ~ IBW^alpha with
         alpha well below the folklore value of 1 that a constant
         NEF x IBW product would require.
    Both are physics claims, not proxies, and either failing is informative.
    """
    designs = grid_designs(BASE,
                           coupling_rabi_hz=list(np.geomspace(1e6, 120e6, 16)))
    res = run_campaign(designs, lo_points=20)
    assert res.n_feasible >= 10

    order = np.argsort([e.design.coupling_rabi_hz for e in res.feasible])
    ev = [res.feasible[i] for i in order]
    nefs = np.array([e.nef for e in ev])
    ibws = np.array([e.ibw_hz for e in ev])

    # 1. interior optimum (U-shape), not a boundary minimum
    i = int(np.argmin(nefs))
    assert 0 < i < len(nefs) - 1, (
        f"NEF minimum at index {i} of {len(nefs)} — the sweep does not "
        "bracket the optimum, so no statement about the trade is supported")

    # below the optimum the two objectives do NOT oppose
    assert nefs[0] > nefs[i] and ibws[i] > ibws[0]

    # 2. above the optimum they do oppose, but weakly
    br = slice(i, None)
    assert np.all(np.diff(ibws[br]) > 0), "IBW must grow with Omega_c"
    alpha = np.polyfit(np.log10(ibws[br]), np.log10(nefs[br]), 1)[0]
    assert 0.0 < alpha < 0.5, (
        f"trade exponent {alpha:.3f} outside the measured weak-trade regime; "
        "alpha ~ 1 would mean the folklore constant-product law holds")

    assert len(res.frontier()) >= 2
