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
    rydsim.atom -> rydsim.dipoles and reproduces the published RF carrier.

    Thin cell (2 mm): the RF carrier is cell-independent, and Cs at 300 K
    over 5 cm is legitimately refused by the optical-depth gate (the
    published experiments operate thick via EIT bleaching our chain does
    not yet model — spec 06 §7.2)."""
    ev = evaluate(SensorDesign(species=sp, n=n, state=st, rf_partner=pt,
                               cell_length_m=0.002),
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


def test_rydberg_decay_comes_from_lifetimes_not_a_default():
    """Integration gap regression: the objective must take the Rydberg decay
    rate from rydsim.lifetimes (Beterov radiative + BBR), not a fixed
    default. Holding it fixed across an n-sweep suppresses a term that
    OPPOSES the n^2 dipole gain, and silently makes IBW n-independent.
    """
    from rydsim.objective import rydberg_decay_hz

    rates = {}
    for n in (40, 50, 60, 70):
        d = SensorDesign(species="Rb87", n=n, state="D5/2", rf_partner="P3/2",
                         temperature_k=300.0)
        g, prov = rydberg_decay_hz(d)
        rates[n] = g
        assert "Beterov" in prov
    # decay falls with n (roughly n*^-3): strictly monotone
    assert rates[40] > rates[50] > rates[60] > rates[70]
    # anchor: Beterov Table VII gives tau_eff(Rb 50D5/2, 300 K) = 65.352 us
    tau_50 = 1.0 / (2 * np.pi * rates[50])
    assert tau_50 == pytest.approx(65.352e-6, rel=0.05)


def test_no_state_falls_back_to_explicit_decay():
    """Without a named state there is nothing to compute from, so the
    explicit value is used — and says so."""
    from rydsim.objective import rydberg_decay_hz

    d = SensorDesign(rf_dipole_ea0=1000.0, rydberg_decay_hz=3.3e3)
    g, prov = rydberg_decay_hz(d)
    assert g == pytest.approx(3.3e3)
    assert "default" in prov


def test_species_cell_parameters_are_not_rb87_for_everything():
    """AUDIT CRIT-1 regression: _to_ladder_config used to discard every
    species-dependent parameter, so Cs and Rb-85 designs were simulated
    inside a Rb-87 vapor cell (wrong mass, wavelengths, linewidth, density).
    """
    from rydsim.objective import species_cell_parameters

    cs = species_cell_parameters(SensorDesign(species="Cs133", n=47,
                                              state="D5/2", rf_partner="P3/2"))
    rb87 = species_cell_parameters(SensorDesign(species="Rb87", n=53,
                                                state="D5/2", rf_partner="P3/2"))
    rb85 = species_cell_parameters(SensorDesign(species="Rb85", n=53,
                                                state="D5/2", rf_partner="P3/2"))
    # Cs must be Cs: 852.347 nm probe, 5.234 MHz linewidth, 132.905 u, frac 1.0
    assert cs["element"] == "Cs"
    assert cs["lambda_probe"] == pytest.approx(852.347e-9, rel=1e-6)
    assert cs["gamma_e"] / (2 * np.pi) == pytest.approx(5.234e6, rel=1e-3)
    assert cs["mass"] / 1.66053906892e-27 == pytest.approx(132.905, rel=1e-4)
    assert cs["isotope_fraction"] == pytest.approx(1.0)
    # the two Rb isotopes differ in mass and natural abundance
    assert rb87["mass"] != rb85["mass"]
    assert rb87["isotope_fraction"] == pytest.approx(0.2783, rel=1e-3)
    assert rb85["isotope_fraction"] == pytest.approx(0.7217, rel=1e-3)
    # ruling R-15: lambda_c is state-dependent, never hard-coded 480 nm
    assert cs["lambda_coupling"] != pytest.approx(480e-9, rel=1e-3)
    assert 500e-9 < cs["lambda_coupling"] < 520e-9      # Cs 6P3/2 -> nD


def test_optically_thick_cell_is_refused_not_reported():
    """AUDIT CRIT-2 regression: beyond the thin-medium validity limit the
    transmission collapses and the transduction slope -> 0, so NEF diverges
    to a number that LOOKS like a result (a Cs 313 K / 5 cm design returned
    5.4e9 nV/cm/rtHz). The engine must refuse instead.
    """
    thick = evaluate(SensorDesign(species="Cs133", n=47, state="D5/2",
                                  rf_partner="P3/2", temperature_k=313.0,
                                  cell_length_m=0.05), lo_points=16)
    assert not thick.feasible
    assert "optical depth" in thick.reason

    # the same physics in a thin cell is computable and sane
    thin = evaluate(SensorDesign(species="Cs133", n=47, state="D5/2",
                                 rf_partner="P3/2", temperature_k=300.0,
                                 cell_length_m=0.001), lo_points=16)
    if thin.feasible:
        assert 0 < thin.nef_nv_per_cm_rthz < 1e4

    # a room-temperature 5 cm Rb-87 cell (OD ~ 1.7) is ORDINARY lab physics
    # and must NOT be refused — an over-strict gate is itself an audit
    # defect class (weak-probe Beer-Lambert is exact at moderate OD)
    ordinary = evaluate(SensorDesign(species="Rb87", n=50, state="D5/2",
                                     rf_partner="P3/2", temperature_k=300.0,
                                     cell_length_m=0.05), lo_points=16)
    assert ordinary.feasible, ordinary.reason


def test_species_element_mapping_has_one_source_of_truth():
    """Audit R10 follow-up: `sp.name.startswith(...)` was duplicated across
    four modules and silently classified any unknown species as Cs (or Rb).
    All of them must route through atom.element_symbol, which refuses."""
    import ast
    import pathlib

    from rydsim.atom import CS133, RB85, RB87, element_symbol

    assert element_symbol(RB87) == element_symbol(RB85) == "Rb"
    assert element_symbol(CS133) == "Cs"

    # Parse the AST rather than grepping text: a docstring that DESCRIBES the
    # old form is not an occurrence of it, and a text scan cannot tell the
    # difference (this test's first version failed on its own explanation).
    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "rydsim"
    offenders = []
    for path in src.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "startswith"):
                continue
            inner = node.func.value
            if not (isinstance(inner, ast.Attribute) and inner.attr == "name"):
                continue
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) \
                        and arg.value[:2] in ("Rb", "Cs"):
                    offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"species-name slicing survives at {offenders}"


def test_ladder_config_reports_unoverridden_species_defaults():
    """Audit CRIT-1 follow-up: LadderConfig's six species-dependent fields
    are conveniences, not physics. A config must be able to say which are
    still at their Rb-87 values so a mixed-species run is detectable."""
    from rydsim.experiment import LadderConfig
    from rydsim.objective import species_cell_parameters

    assert set(LadderConfig().species_defaults_in_use()) == {
        "gamma_e", "mass", "lambda_probe", "lambda_coupling",
        "element", "isotope_fraction"}
    cs = species_cell_parameters(SensorDesign(species="Cs133", n=47,
                                              state="D5/2", rf_partner="P3/2"))
    assert LadderConfig(**cs).species_defaults_in_use() == []


def test_confidence_enum_has_verified_arc_rank():
    """The audit ranks VERIFIED > VERIFIED-ARC > LITERATURE-RECALL and
    dipoles._CONF_ORDER already uses the string; the enum lacked the member,
    so a taint could not be expressed in typed form."""
    from rydsim.dipoles import _CONF_ORDER
    from rydsim.provenance import Confidence

    assert Confidence.VERIFIED_ARC.value == "VERIFIED-ARC"
    assert _CONF_ORDER.index("VERIFIED") < _CONF_ORDER.index("VERIFIED-ARC")
    assert _CONF_ORDER.index("VERIFIED-ARC") < _CONF_ORDER.index("LITERATURE-RECALL")


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
