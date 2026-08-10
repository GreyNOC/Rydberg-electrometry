"""DESIGNER objective API (D0): contract + firewall tests.

These lock the properties any search layer depends on: infeasibility is
reported not raised, refusal beats guessing, objectives are minimization-
form, and the Pareto filter is exact bookkeeping over oracle numbers.
"""

import numpy as np
import pytest

from rydsim.objective import (
    Evaluation,
    SensorDesign,
    evaluate,
    evaluate_many,
    pareto_front,
)
from rydsim.provenance import IntegrityError


def test_evaluate_baseline_design_is_feasible():
    d = SensorDesign(rf_dipole_ea0=1000.0, temperature_k=313.0)
    ev = evaluate(d, lo_points=32)
    assert ev.feasible, ev.reason
    assert ev.nef > 0 and ev.ibw_hz > 0
    assert 0 < ev.e_lo < 20
    # sanity band: idealized-noise NEF in the nV/cm/rtHz class
    assert 0.1 < ev.nef_nv_per_cm_rthz < 1e4
    # IBW is MHz-class for these EIT rates
    assert 1e4 < ev.ibw_hz < 5e7
    assert ev.confidence_taint.startswith("LITERATURE-RECALL")


PUBLISHED_CONFIGS = [
    # label, species, n, state, partner, published RF [GHz], tol [GHz]
    ("Sedlacek 2012", "Rb85", 53, "D5/2", "P3/2", 14.233, 0.005),
    ("Holloway 2014", "Rb85", 28, "D5/2", "P3/2", 104.77, 0.05),
    ("Jing 2020", "Cs133", 47, "D5/2", "P3/2", 6.9452, 0.010),
    ("Sci. Adv. 2024", "Rb87", 39, "D5/2", "P3/2", 36.9, 0.010),
]


@pytest.mark.parametrize("label,sp,n,st,pt,f_pub,tol", PUBLISHED_CONFIGS)
def test_species_chain_reproduces_published_rf_frequencies(
        label, sp, n, st, pt, f_pub, tol):
    """End-to-end: a design named by species/state resolves through
    rydsim.atom -> rydsim.dipoles and reproduces the published RF carrier."""
    ev = evaluate(SensorDesign(species=sp, n=n, state=st, rf_partner=pt),
                  lo_points=24)
    assert ev.feasible, ev.reason
    assert ev.rf_frequency_hz / 1e9 == pytest.approx(f_pub, abs=tol), label
    assert ev.dipole_cm > 0
    assert "nist_pi" in ev.confidence_taint          # convention is stamped


def test_dipole_convention_is_explicit_not_implicit():
    """Same transition, two published conventions, ~29% apart. Sci. Adv.
    2024 quotes 1218 e*a0 for Rb 39D5/2->40P3/2 under 'stretched'; the
    normative NIST-pi value is lower. Silently picking one would corrupt
    every field inversion, so the convention is a design axis."""
    from rydsim.constants import AU_DIPOLE

    common = dict(species="Rb87", n=39, state="D5/2", rf_partner="P3/2")
    nist = evaluate(SensorDesign(**common, dipole_convention="nist_pi"),
                    lo_points=24)
    stretched = evaluate(SensorDesign(**common, dipole_convention="stretched"),
                         lo_points=24)
    assert stretched.dipole_cm / AU_DIPOLE == pytest.approx(1218.0, rel=0.02)
    assert stretched.dipole_cm > nist.dipole_cm
    assert "stretched" in stretched.confidence_taint


def test_rf_sign_is_carried():
    """nD5/2 -> (n+1)P3/2 is a DOWNWARD transition; the sign drives AT
    sideband asymmetry downstream and must not be discarded."""
    ev = evaluate(SensorDesign(species="Rb85", n=53, state="D5/2",
                               rf_partner="P3/2"), lo_points=24)
    assert ev.rf_sign in (-1, 1)


def test_forbidden_transition_refused():
    """D5/2 -> F7/2 with |dl|=1 is allowed; D5/2 -> D5/2 is not an E1 pair
    and must be refused, never returned as a zero dipole (which would
    invert to an infinite field)."""
    ev = evaluate(SensorDesign(species="Rb87", n=50, state="D5/2",
                               rf_partner="D5/2"), lo_points=16)
    assert not ev.feasible


def test_low_n_refused_through_the_objective():
    """The spec-01 validity floor propagates all the way to the search API."""
    ev = evaluate(SensorDesign(species="Rb87", n=5, state="S1/2",
                               rf_partner="P3/2"), lo_points=16)
    assert not ev.feasible


def test_missing_dipole_refuses_rather_than_guessing():
    """The firewall's first line: no dipole -> infeasible with a reason,
    never a fabricated number."""
    ev = evaluate(SensorDesign())
    assert not ev.feasible
    assert "will not guess" in ev.reason
    assert ev.nef is None


def test_infeasible_evaluation_has_no_objectives():
    ev = evaluate(SensorDesign())
    with pytest.raises(IntegrityError):
        ev.objectives()


def test_objectives_are_minimization_form():
    ev = evaluate(SensorDesign(rf_dipole_ea0=1000.0), lo_points=32)
    obj = ev.objectives()
    assert obj["nef"] == ev.nef
    assert obj["neg_ibw"] == -ev.ibw_hz   # bandwidth maximized => negated


def test_evaluate_many_keeps_infeasible_points_visible():
    """Search layers must be able to report their invalid-proposal rate."""
    designs = [SensorDesign(rf_dipole_ea0=1000.0), SensorDesign()]
    evs = evaluate_many(designs, lo_points=24)
    assert len(evs) == 2
    assert sum(e.feasible for e in evs) == 1


def test_stronger_dipole_improves_sensitivity():
    """Physics monotonicity: larger RF dipole transduces more field per volt,
    so NEF must improve ~1/d. This also guards the scale-invariance of the
    LO grid: with a grid fixed in absolute V/m the optimum falls outside the
    searchable range as d grows and stronger atoms look WORSE — the artifact
    this test originally exposed."""
    weak = evaluate(SensorDesign(rf_dipole_ea0=500.0), lo_points=32)
    strong = evaluate(SensorDesign(rf_dipole_ea0=2000.0), lo_points=32)
    assert weak.feasible and strong.feasible
    assert strong.nef < weak.nef
    # 4x dipole should buy close to 4x NEF (transduction slope ~ d)
    assert weak.nef / strong.nef == pytest.approx(4.0, rel=0.25)


def test_lo_optimum_is_scale_invariant_in_rabi_units():
    """The optimal LO Rabi frequency is a property of the ATOMIC dynamics,
    not of the dipole: E_LO* * d / hbar must be ~invariant across d."""
    from rydsim.constants import AU_DIPOLE, HBAR

    rabis = []
    for d_ea0 in (500.0, 1000.0, 2000.0):
        ev = evaluate(SensorDesign(rf_dipole_ea0=d_ea0), lo_points=32)
        assert ev.feasible
        rabis.append(ev.e_lo * (d_ea0 * AU_DIPOLE) / HBAR)
    assert max(rabis) / min(rabis) < 1.15


def test_sql_floor_is_reported_and_not_violated():
    """Every evaluation carries the atom-projection-noise floor with the
    (N_eff, f_vel, tau) that produced it — spec 08 §2.4.5 requires the
    bookkeeping to be computed and printed, never folklore."""
    ev = evaluate(SensorDesign(rf_dipole_ea0=1000.0, temperature_k=313.0),
                  lo_points=32)
    assert ev.feasible
    assert ev.nef_sql > 0 and ev.n_eff > 0
    assert 0 < ev.f_vel <= 1.0            # participation ratio is bounded
    assert ev.tau_coh > 0
    # total is the quadrature sum of technical and projection noise
    assert ev.nef == pytest.approx(np.hypot(ev.nef_technical, ev.nef_sql))
    assert ev.nef >= ev.nef_sql           # can never beat the SQL


def test_frequency_noise_vanishes_at_line_centre_by_symmetry():
    """At exactly zero probe detuning the transmission is at a stationary
    point, so dP/dnu = 0 and first-order laser-frequency noise does not
    couple. This is a real symmetry of the model — and an idealization: it
    is why the objective must expose a lock offset (next test)."""
    ev = evaluate(SensorDesign(rf_dipole_ea0=1000.0, probe_detuning_hz=0.0),
                  lo_points=24)
    assert ev.feasible
    assert ev.noise_budget["frequency"] == pytest.approx(0.0, abs=1e-40)


def test_frequency_noise_appears_off_line_centre():
    """With a realistic lock offset the frequency-noise term switches on and
    degrades NEF — the term published vapor-cell work attributes its floor
    to (Jing 2020: transit + laser frequency noise)."""
    centre = evaluate(SensorDesign(rf_dipole_ea0=1000.0,
                                   probe_detuning_hz=0.0), lo_points=24)
    offset = evaluate(SensorDesign(rf_dipole_ea0=1000.0,
                                   probe_detuning_hz=2e6), lo_points=24)
    assert centre.feasible and offset.feasible
    assert offset.noise_budget["frequency"] > 0
    assert offset.nef > centre.nef


def test_broader_linewidth_worsens_nef_off_centre():
    narrow = evaluate(SensorDesign(rf_dipole_ea0=1000.0, probe_detuning_hz=2e6,
                                   probe_linewidth_hz=10e3,
                                   coupling_linewidth_hz=10e3), lo_points=24)
    broad = evaluate(SensorDesign(rf_dipole_ea0=1000.0, probe_detuning_hz=2e6,
                                  probe_linewidth_hz=1e6,
                                  coupling_linewidth_hz=1e6), lo_points=24)
    assert broad.nef > narrow.nef


def _mk(nef, ibw):
    d = SensorDesign(rf_dipole_ea0=1000.0)
    return Evaluation(design=d, feasible=True, nef=nef, ibw_hz=ibw,
                      dynamic_range_db=100.0)


def test_pareto_front_exactness():
    a = _mk(1e-6, 1e6)     # best NEF
    b = _mk(2e-6, 5e6)     # best IBW
    c = _mk(3e-6, 0.5e6)   # dominated by a
    front = pareto_front([a, b, c])
    assert a in front and b in front
    assert c not in front


def test_pareto_front_ignores_infeasible():
    good = _mk(1e-6, 1e6)
    bad = Evaluation(design=SensorDesign(), feasible=False, reason="x")
    assert pareto_front([good, bad]) == [good]
