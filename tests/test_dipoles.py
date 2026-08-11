"""Validation of rydsim.dipoles — the specs 02+03 integration layer.

Benchmark plan (assignment per docs/spec/00-conventions.md SS4/SS7 step 5 and
docs/spec/00-integrity-audit.md R5): every literature anchor is RECOMPUTED
from scratch (consensus radial x angular) and compared against the recall
value, which is transcribed below with its source + confidence tag and never
used as an input. Rows implemented:

  * audit R5 / spec 08 SS3.2  — Jing mu_RF(Cs 47D5/2->48P3/2) = 1443.450
    e a0 (and the radial 2946.512 a0), 2 %. PASSES.
  * spec 09 C5e — Sedlacek d(Rb 53D5/2->54P3/2) = 1.37e-26 C m, 2 % (task
    tolerance; corpus row allows 5 %). **FIXTURE FLAGGED, not passing**:
    neither published convention reproduces the printed number, and the
    residual under the paper's own stated ("stretched hyperfine states")
    convention is a clean sqrt(2). Per audit R5 the fixture — not the code —
    is flagged, so the row is asserted as a DOCUMENTED LITERATURE TENSION
    with a code-independent corroboration, never as agreement. See
    ``rydsim.dipoles.C5E_CONVENTION_TENSION``.
  * spec 09 C5f / spec 08 SS3.3 — Tu d(Rb 39D5/2->40P3/2) = 1218 e a0, 2 %.
    PASSES under 'stretched', the convention the paper states.
  * spec 02 B15 / audit R5 pattern — Rb-87 D2 closure: +5..10 % bias band,
    radial in [5.45, 5.70] a0.
  * task row — Cs D2 closure: same +5..10 % band (band membership only; no
    Cs bias anchor exists in specs 02/03/09, so no numeric pin is asserted).
"""

from __future__ import annotations

import math

import pytest

from rydsim.angular import angular_factor, effective_rf_dipole, steck_to_racah
from rydsim.atom import CS133, RB87
from rydsim.constants import AU_DIPOLE
from rydsim.dipoles import (C5E_CONVENTION_TENSION, DLINE_BIAS_BAND,
                            MU_RF_CONVENTIONS, DipoleResult, dline_closure,
                            full_dipole_matrix_element, mu_rf)
from rydsim.provenance import IntegrityError

# ---------------------------------------------------------------------------
# Literature recall/anchor values — transcribed with source + confidence tags
# (no-fabrication rule). None of these is an input to the computation.
# ---------------------------------------------------------------------------

# Jing et al., Nat. Phys. 16, 911 (2020) fixture — spec 08 SS3.2 / audit R5.
# LITERATURE-RECALL (secondary survey arXiv:2412.05554-family citing Jing);
# audit R5 mandates recomputation from module 02+03 with 2 % agreement.
JING_MU_RF_EA0 = 1443.450          # [e a0]  LITERATURE-RECALL
JING_RADIAL_A0 = 2946.512          # [a0]    LITERATURE-RECALL (same row)

# Sedlacek et al., Nat. Phys. 8, 819 (2012), params from arXiv:1205.4461 v1.
# The PRINTED number is VERIFIED (v1 full text); the CONVENTION behind it is
# UNVERIFIED — the paper says "4-level model, stretched hyperfine states",
# which recomputes 41.8 % high (see the tension tests below).
SEDLACEK_MU_CM = 1.37e-26          # [C m]   VERIFIED (number, v1 full text)

# Tu et al., Sci. Adv. 10, eads0683 (2024), PMC full text — spec 08 SS3.3 /
# spec 09 C5f [VERIFIED]. Ground 5S1/2 F=2 mF=2 (stretched sigma+ ladder).
TU_MU_EA0 = 1218.0                 # [e a0]  VERIFIED (PMC)

# Published Rb Ritz quantum defects (spec 01 SS3 tables), transcribed here so
# the Sedlacek/Tu cross-ratio below can be evaluated WITHOUT the dipole chain
# under test — pencil-and-paper reproducible from the two papers plus these.
RB_DEFECT_D52 = (1.3464622, -0.594)    # (d0, d2) Mack 2011 Table I [VERIFIED]
RB_DEFECT_P32 = (2.6416737, 0.295)     # (d0, d2) Li 2003 [VERIFIED-ARC digits]

# Steck rev 2.3.4 D2 reduced dipoles (Steck convention) — spec 01/03
# [VERIFIED, primary PDFs]; the closure targets, reached only through the
# documented +5..10 % model-potential bias band (spec 02 SS7.1/B15).
STECK_D2_RB87_EA0 = 4.22752        # [e a0]  VERIFIED
STECK_D2_CS_EA0 = 4.4837           # [e a0]  VERIFIED

# Spec 02 B15 regression band on the Rb 5S1/2->5P3/2 model-potential radial
# integral (computed 5.569 a0 with measured-energy nu) [computed VERIFIED].
B15_RADIAL_BAND_A0 = (5.45, 5.70)


def _ritz_nu(n: int, defect: tuple[float, float]) -> float:
    """nu = n - delta(n) from PUBLISHED Ritz coefficients (spec 01 Eq. 1.4),
    by fixed-point iteration. Deliberately re-implemented in three lines of
    arithmetic so the Sedlacek/Tu cross-ratio below is independent of every
    module under test."""
    d0, d2 = defect
    d = d0
    for _ in range(64):
        d = d0 + d2 / (n - d) ** 2
    return n - d


def _published_radial_ratio() -> float:
    """R(53D5/2->54P3/2) / R(39D5/2->40P3/2) from published quantum defects
    alone, via the semiclassical nu1*nu2 scaling of a fixed Delta-n = 1,
    fixed-(l, l') channel. No rydsim.radial, no rydsim.dipoles."""
    return ((_ritz_nu(53, RB_DEFECT_D52) * _ritz_nu(54, RB_DEFECT_P32))
            / (_ritz_nu(39, RB_DEFECT_D52) * _ritz_nu(40, RB_DEFECT_P32)))


# ---------------------------------------------------------------------------
# Shared fixtures (consensus radial runs are the expensive part)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def jing() -> DipoleResult:
    return mu_rf(CS133, (47, 2, 2.5), (48, 1, 1.5))  # default nist_pi


@pytest.fixture(scope="module")
def sedlacek() -> DipoleResult:
    """Spec 09 C5e pair under the convention the PAPER states (stretched
    hyperfine states) — audit R5 computes the fixture's own reading."""
    return mu_rf(RB87, (53, 2, 2.5), (54, 1, 1.5), convention="stretched")


@pytest.fixture(scope="module")
def sedlacek_nist() -> DipoleResult:
    """Same pair under lock #11 (normative), the other candidate reading."""
    return mu_rf(RB87, (53, 2, 2.5), (54, 1, 1.5), convention="nist_pi")


@pytest.fixture(scope="module")
def tu() -> DipoleResult:
    return mu_rf(RB87, (39, 2, 2.5), (40, 1, 1.5), convention="stretched")


# ---------------------------------------------------------------------------
# Money benchmarks (audit R5 and the spec 09 C5 rows)
# ---------------------------------------------------------------------------

def test_jing_mu_rf_recomputed_2pct(jing: DipoleResult) -> None:
    """Audit R5 / spec 08 SS3.2: mu_RF(Cs 47D5/2->48P3/2) recomputed from
    scratch (consensus radial x NIST-pi angular) agrees with the
    LITERATURE-RECALL fixture value 1443.450 e a0 to 2 % — else the FIXTURE,
    not the code, is flagged."""
    assert jing.convention == "nist_pi"
    # Audit R5 sets this tolerance at 2 %. It is NOT tightened code-side: a
    # tighter gate would treat an unnamed-secondary-survey recall value as
    # more certain than the recomputation, inverting R5 (spec 09 SS7 rule 5 /
    # audit SS3 item 38 — re-toleranced benchmarks require a spec edit).
    assert jing.value_ea0 == pytest.approx(JING_MU_RF_EA0, rel=0.02)


#: Values the CODE currently computes for the Jing and D-line-closure
#: anchors. These are NOT fixtures, published values, or gates on the
#: literature — they pin the implementation against silent drift, which the
#: audit-R5 2% fixture gates are deliberately too loose to catch. Regenerate
#: deliberately (and say why in the commit) if the physics legitimately
#: changes; a surprise failure here means a code path moved.
COMPUTED_JING_MU_EA0 = 1443.4364014305193
COMPUTED_JING_RADIAL_A0 = 2946.4022163866016
COMPUTED_DLINE_BIAS_REL = {"Rb87": 0.07558391219210603,
                           "Cs133": 0.0960743281263996}


def test_computed_anchors_do_not_drift(jing: DipoleResult) -> None:
    """Code-vs-code stability pin, distinct from the R5 fixture gates.

    The remediation replaced a 1e-3 assertion on the Jing dipole with the
    R5-mandated 2% gate, and a bias_rel point value with band membership.
    Both were correct as FIXTURE gates — R5 forbids treating a
    LITERATURE-RECALL secondary-survey number as more certain than our own
    recomputation. But dropping them also dropped drift detection: the
    recomputation could move 1.9% and every gate would still pass.

    This test restores that sensitivity WITHOUT re-asserting the fixture:
    it pins what the code computes against itself. Fixture agreement and
    implementation stability are different questions and now have different
    tests.
    """
    assert jing.value_ea0 == pytest.approx(COMPUTED_JING_MU_EA0, rel=1e-9)
    assert abs(jing.radial_a0) == pytest.approx(COMPUTED_JING_RADIAL_A0, rel=1e-9)
    for name, sp in (("Rb87", RB87), ("Cs133", CS133)):
        res = dline_closure(sp, "D2")
        assert res.bias_rel == pytest.approx(
            COMPUTED_DLINE_BIAS_REL[name], rel=1e-9), name


def test_jing_radial_recall_2pct(jing: DipoleResult) -> None:
    """Audit R5 (same row): the radial ME behind the Jing fixture,
    2946.512 a0 [LITERATURE-RECALL], reproduced by the spec 02 consensus."""
    assert abs(jing.radial_a0) == pytest.approx(JING_RADIAL_A0, rel=0.02)
    # angular factor is exactly sqrt(6)/5 (NIST 0.4899, spec 03 B14)
    assert jing.angular_factor == pytest.approx(math.sqrt(6.0) / 5.0, rel=1e-12)


def test_sedlacek_c5e_fixture_is_flagged_not_reproduced(
        sedlacek: DipoleResult, sedlacek_nist: DipoleResult) -> None:
    """Spec 09 C5e, audit R5: BOTH candidate readings of Sedlacek's printed
    mu_RF(Rb 53D5/2->54P3/2) = 1.37e-26 C m miss it, so R5 flags the
    FIXTURE. Recorded as a tension, never forced into agreement — the code
    must not acquire a convention that lands on this number.

      * 'stretched' — the paper's own words ("4-level model, stretched
        hyperfine states"): 1.9426e-26 C m, +41.8 %.
      * 'nist_pi'   — lock #11, normative: 1.5047e-26 C m, +9.8 %.

    Both exceed audit R5's 2 % AND spec 09 C5e's 5 %."""
    for d in (sedlacek, sedlacek_nist):
        rel = d.value_Cm / SEDLACEK_MU_CM - 1.0
        assert rel > 0.05, (
            f"{d.convention} now lands inside the C5e tolerance (rel="
            f"{rel:+.4f}); the tension is resolved — update spec 09 C5e "
            "and this test together, never the code alone")
    assert sedlacek.value_Cm > sedlacek_nist.value_Cm > SEDLACEK_MU_CM
    # The two readings differ by an EXACT rational angular ratio (spec 03
    # Eq. 2.9), independent of the radial layer: (sqrt(6)/5)/sqrt(2/5)
    # = sqrt(3/5). Checked here so the "41 % apart" claim is an identity.
    assert (sedlacek_nist.value_Cm / sedlacek.value_Cm
            == pytest.approx(math.sqrt(0.6), rel=1e-12))
    assert sedlacek.angular_factor == pytest.approx(math.sqrt(0.4), rel=1e-12)
    assert sedlacek_nist.angular_factor == pytest.approx(math.sqrt(6.0) / 5.0,
                                                        rel=1e-12)


def test_sedlacek_residual_is_exactly_sqrt2(sedlacek: DipoleResult) -> None:
    """The C5e residual is not scatter — it is a clean sqrt(2), the
    amplitude-vs-RMS field-convention artifact ruling R-22 already records
    for Jing's printed prefactor. Under the paper's own stated convention,
    computed/printed = 1.41796, which is sqrt(2) to 0.26 %.

    Asserted at 1 % so it stays a statement about sqrt(2) and not about the
    third significant figure of a two-digit printed number."""
    ratio = sedlacek.value_Cm / SEDLACEK_MU_CM
    assert ratio == pytest.approx(math.sqrt(2.0), rel=0.01)
    # ... and the radial layer is not the suspect: three methods agree to
    # better than 1e-4, so no radial error of size sqrt(2) is available.
    assert sedlacek.radial_spread_rel < 1e-4


def test_sedlacek_tu_ratio_shows_sqrt2_without_rydsim() -> None:
    """Code-independent corroboration of the C5e tension (audit R5 requires
    the tension be reproducible without RydSim).

    Sedlacek (Rb 53D5/2->54P3/2, 1615.88 e a0) and Tu (Rb 39D5/2->40P3/2,
    1218 e a0) print dipoles for the SAME D5/2->P3/2 angular channel and
    both describe stretched states, so under one convention their ratio is
    the radial ratio alone. For a fixed Delta-n = 1 channel that scales as
    nu1*nu2 (semiclassical), evaluated here from PUBLISHED Ritz defects with
    plain arithmetic — no rydsim.radial, no rydsim.dipoles:

        nu(53D)nu(54P) / nu(39D)nu(40P) = 1.8859
        printed 1615.88 / 1218          = 1.3267
        ratio of ratios                 = 1.4215  ->  sqrt(2) to 0.5 %

    So one of the two printed numbers carries an extra 1/sqrt(2), and it is
    the one RydSim's stretched convention misses by that factor."""
    radial_ratio = _published_radial_ratio()
    printed_ratio = (SEDLACEK_MU_CM / AU_DIPOLE) / TU_MU_EA0
    assert radial_ratio == pytest.approx(1.8859, rel=1e-3)
    assert printed_ratio == pytest.approx(1.3267, rel=1e-3)
    assert radial_ratio / printed_ratio == pytest.approx(math.sqrt(2.0),
                                                         rel=0.01)
    # The same ratio through the spec-02 consensus radial agrees, so the
    # semiclassical nu1*nu2 shortcut above is not what produces the sqrt(2).
    engine_ratio = (abs(mu_rf(RB87, (53, 2, 2.5), (54, 1, 1.5),
                              convention="stretched").radial_a0)
                    / abs(mu_rf(RB87, (39, 2, 2.5), (40, 1, 1.5),
                                convention="stretched").radial_a0))
    assert engine_ratio == pytest.approx(radial_ratio, rel=5e-3)


def test_c5e_tension_note_digits_track_live_computation(
        sedlacek: DipoleResult, sedlacek_nist: DipoleResult) -> None:
    """The shipped ``C5E_CONVENTION_TENSION`` provenance string is a
    machine-readable integrity claim, so every digit in it must be the digit
    a live run produces — the A4_L1_NOTE failure mode (a stale bound inside
    a provenance string) must not repeat.

    This does not grep the module source: each token is FORMATTED from a
    freshly computed quantity, so if the radial engine or the angular chain
    moves, the note stops matching and this fails."""
    printed_ea0 = SEDLACEK_MU_CM / AU_DIPOLE
    radial_ratio = _published_radial_ratio()
    printed_ratio = printed_ea0 / TU_MU_EA0
    sqrt2_residual = sedlacek.value_Cm / SEDLACEK_MU_CM

    expected = {
        "printed anchor [e a0]": f"{printed_ea0:.2f}",              # 1615.88
        "stretched [e a0]": f"{sedlacek.value_ea0:.1f}",            # 2291.2
        "nist_pi [e a0]": f"{sedlacek_nist.value_ea0:.1f}",         # 1774.8
        "stretched excess [%]": f"{(sqrt2_residual - 1) * 100:.1f}",     # 41.8
        "nist_pi excess [%]": f"{(sedlacek_nist.value_Cm / SEDLACEK_MU_CM - 1) * 100:.1f}",
        "consensus radial [a0]": f"{abs(sedlacek.radial_a0):.2f}",  # 3622.78
        "residual/printed": f"{sqrt2_residual:.5f}",                # 1.41796
        "sqrt2 deviation [%]": f"{(sqrt2_residual / math.sqrt(2) - 1) * 100:.2f}",
        "published nu ratio": f"{radial_ratio:.4f}",                # 1.8859
        "printed dipole ratio": f"{printed_ratio:.4f}",             # 1.3267
        "ratio of ratios": f"{radial_ratio / printed_ratio:.4f}",   # 1.4215
        "cross-ratio sqrt2 deviation [%]":
            f"{(radial_ratio / printed_ratio / math.sqrt(2) - 1) * 100:.1f}",
    }
    for label, token in expected.items():
        assert token in C5E_CONVENTION_TENSION, (
            f"{label}: live value formats to {token!r}, which no longer "
            f"appears in C5E_CONVENTION_TENSION — the shipped note is stale")
    # The note must say what it is: a flagged fixture, not an agreement.
    assert "FIXTURE FLAGGED" in C5E_CONVENTION_TENSION
    assert "audit R5" in C5E_CONVENTION_TENSION


def test_tu_mu_rf_2pct(tu: DipoleResult) -> None:
    """Spec 09 C5f / spec 08 SS3.3 (task tol 2 %): Tu 2024
    d(Rb 39D5/2->40P3/2) = 1218 e a0 [VERIFIED, PMC]. Their stretched
    sigma+-ladder config (5S1/2 F=2 mF=2 ground) -> 'stretched' convention;
    measured agreement 3e-4."""
    assert tu.convention == "stretched"
    assert tu.value_ea0 == pytest.approx(TU_MU_EA0, rel=0.02)
    # stretched A for D5/2->P3/2 is exactly sqrt(2/5)
    assert tu.angular_factor == pytest.approx(math.sqrt(0.4), rel=1e-12)
    assert tu.q == 1 and tu.mj == 2.5 and tu.mjp == 1.5


# ---------------------------------------------------------------------------
# D-line closure (audit R5 self-check pattern; spec 02 B15 + task row)
# ---------------------------------------------------------------------------

def test_dline_closure_rb87_d2_bias_band() -> None:
    """Spec 02 B15 / audit R5 pattern: Rb-87 D2 reduced dipole recomputed
    from model-potential radial x angular lands +5..10 % ABOVE Steck's
    4.22752 e a0 [VERIFIED] — the documented low-n model bias (measured
    +7.6 %), asserted as a band, never as agreement. Both numbers exposed."""
    res = dline_closure(RB87, "D2")
    assert res.in_band
    assert DLINE_BIAS_BAND[0] <= res.bias_rel <= DLINE_BIAS_BAND[1]
    assert res.d_steck_measured_ea0 == STECK_D2_RB87_EA0
    assert res.d_steck_computed_ea0 > res.d_steck_measured_ea0
    # spec 02 B15 regression band on the radial integral itself
    assert B15_RADIAL_BAND_A0[0] <= abs(res.radial_a0) <= B15_RADIAL_BAND_A0[1]
    # Racah/Steck conversion handled once, here: sqrt(2) for j_lower = 1/2
    assert res.d_racah_computed_ea0 == pytest.approx(
        steck_to_racah(res.d_steck_computed_ea0, 0.5), rel=1e-12)


def test_dline_closure_cs_d2_bias_band() -> None:
    """Task row (audit R5 pattern): Cs D2 closure vs Steck 4.4837 e a0
    [VERIFIED] inside the same +5..10 % band, asserted exactly as the Rb row
    asserts it — band membership plus the sign of the model-potential bias.

    No numeric bias pin: specs 02/03/09 carry a bias anchor for Rb
    5S->5P3/2 (+7.6 %) and none for Cs, so a Cs number here could only be a
    transcript of this code's own output. The previous
    `bias_rel == approx(0.096, abs=0.02)` was exactly that, and half its
    window (0.10..0.116) was unreachable anyway: dline_closure raises above
    DLINE_BIAS_BAND[1] = 0.10 before the assertion is ever evaluated."""
    res = dline_closure(CS133, "D2")
    assert res.in_band
    assert DLINE_BIAS_BAND[0] <= res.bias_rel <= DLINE_BIAS_BAND[1]
    assert res.d_steck_measured_ea0 == STECK_D2_CS_EA0
    assert res.d_steck_computed_ea0 > res.d_steck_measured_ea0
    # Racah/Steck conversion handled once, in the module (spec 03 Eq. 2.3)
    assert res.d_racah_computed_ea0 == pytest.approx(
        steck_to_racah(res.d_steck_computed_ea0, 0.5), rel=1e-12)


def test_dline_closure_band_edges_are_reachable() -> None:
    """Guards the defect the Cs pin above had: an assertion window wider
    than the function's own enforced band can never be observed, because
    require_band=True raises first. Any bias assertion must therefore either
    sit inside DLINE_BIAS_BAND or pass require_band=False."""
    with pytest.raises(IntegrityError, match="out of band"):
        dline_closure(CS133, "D2", band=(0.0, DLINE_BIAS_BAND[0] / 2))
    loose = dline_closure(CS133, "D2", band=(0.0, DLINE_BIAS_BAND[0] / 2),
                          require_band=False)
    strict = dline_closure(CS133, "D2")
    assert loose.bias_rel == pytest.approx(strict.bias_rel, rel=1e-12)
    assert not loose.in_band and strict.in_band


def test_dline_closure_out_of_band_raises() -> None:
    """Audit R5 pattern: an out-of-band closure is an IntegrityError (bias
    STABILITY regression), not a warning. Forced here with a narrow band."""
    with pytest.raises(IntegrityError, match="out of band"):
        dline_closure(RB87, "D2", band=(0.0, 0.01))
    # require_band=False exposes the numbers without raising
    res = dline_closure(RB87, "D2", band=(0.0, 0.01), require_band=False)
    assert not res.in_band and res.bias_rel > 0.01


# ---------------------------------------------------------------------------
# Cross-layer consistency
# ---------------------------------------------------------------------------

def test_mu_rf_matches_angular_effective_rf_dipole(jing: DipoleResult) -> None:
    """Lock #11 single-implementation check: mu_rf (radial from spec 02)
    equals rydsim.angular.effective_rf_dipole fed the same radial integral."""
    expected = effective_rf_dipole(2, 2.5, 1, 1.5, jing.radial_a0)
    assert jing.value_Cm == pytest.approx(expected, rel=1e-12)


def test_full_dipole_consistency_and_zero(jing: DipoleResult) -> None:
    """full_dipole_matrix_element = e a0 R A for the same pair; (m, q)
    selection-rule zeros return exact 0.0 with provenance intact
    (coupling-matrix entries), spec 03 SS2.4."""
    d = full_dipole_matrix_element(CS133, (47, 2, 2.5), (48, 1, 1.5),
                                   q=0, mj=0.5)
    A = angular_factor(2, 2.5, 0.5, 1, 1.5, 0.5, 0)
    assert d.value_Cm == pytest.approx(
        AU_DIPOLE * d.radial_a0 * A, rel=1e-12)
    assert abs(d.value_Cm) == pytest.approx(jing.value_Cm, rel=1e-12)
    assert d.mjp == 0.5 and d.convention == "racah_mj_resolved"
    # q = 0 with mjp = mj - 1 violates the 3j m-condition -> exact 0.0
    z = full_dipole_matrix_element(CS133, (47, 2, 2.5), (48, 1, 1.5),
                                   q=0, mj=0.5, mjp=-0.5)
    assert z.value_Cm == 0.0 and z.unc_Cm == 0.0
    assert z.confidence_floor and z.sources


def test_mu_rf_direction_symmetric(jing: DipoleResult) -> None:
    """|mu_RF| is direction-symmetric (spec 03 SS2.1: magnitudes are);
    D->P and P->D give the same effective dipole."""
    rev = mu_rf(CS133, (48, 1, 1.5), (47, 2, 2.5))
    assert rev.value_Cm == pytest.approx(jing.value_Cm, rel=1e-9)


def test_spread_based_uncertainty(jing: DipoleResult) -> None:
    """Audit SS4 item 5: the shipped uncertainty IS the cross-method radial
    spread; >= 2 methods present (consensus rule, spec 02 SS1)."""
    assert jing.unc_Cm == pytest.approx(
        jing.value_Cm * jing.radial_spread_rel, rel=1e-12)
    assert jing.radial_spread_rel > 0.0
    assert set(jing.radial_methods) >= {"model_potential", "coulomb"}
    assert jing.unc_ea0 == pytest.approx(jing.unc_Cm / AU_DIPOLE, rel=1e-12)


# ---------------------------------------------------------------------------
# Provenance taint (audit SS4 item 4)
# ---------------------------------------------------------------------------

def test_confidence_floor_cs_verified(jing: DipoleResult) -> None:
    """Cs 47D5/2 and 48P3/2 defects are both Deiglmayr-2016 VERIFIED, MSD94
    VERIFIED, angular exact -> floor VERIFIED."""
    assert jing.confidence_floor == "VERIFIED"
    assert any("Deiglmayr" in s for s in jing.sources)


def test_confidence_floor_rb_p_series_taints(tu: DipoleResult) -> None:
    """Rb nP3/2 defects are VERIFIED-ARC (digits) only (audit R6) -> the
    minimum confidence on the Rb D->P path is VERIFIED-ARC, and the Rb
    a4(l=1) transcription-discrepancy note rides along (spec 02 SS3.3)."""
    assert tu.confidence_floor == "VERIFIED-ARC"
    assert any("a4(l=1)" in s for s in tu.sources)


# ---------------------------------------------------------------------------
# Refusals (audit SS3; no silent zeros / no guessing)
# ---------------------------------------------------------------------------

def test_forbidden_pair_raises() -> None:
    """Audit SS3 / spec 03 SS2.4-2.5: E1-forbidden pairs raise
    IntegrityError instead of returning a zero effective dipole.
    Delta-j = 2 (D5/2->P1/2) kills the (j 1 j') triangle; Delta-l = 2
    (D->S) is refused before any radial work."""
    with pytest.raises(IntegrityError):
        mu_rf(CS133, (47, 2, 2.5), (48, 1, 0.5))       # Delta j = 2
    with pytest.raises(IntegrityError):
        mu_rf(CS133, (47, 2, 2.5), (48, 0, 0.5))       # Delta l = 2
    with pytest.raises(IntegrityError):
        full_dipole_matrix_element(CS133, (47, 2, 2.5), (48, 0, 0.5),
                                   q=0, mj=0.5)        # Delta l = 2


def test_unknown_convention_raises() -> None:
    """Convention stamps are a closed set of PUBLISHED conventions (audit
    SS4 item 7). 'pi_manifold_rms' is named explicitly: it was an invented,
    unsourced averaging rule whose only property was landing on the C5e
    fixture, and it must not come back (audit R5 / no-fabrication rule).
    A mixed-m_j ensemble is spec 03 SS2.3's sum over individual splittings,
    not a scalar rms dipole."""
    assert MU_RF_CONVENTIONS == ("nist_pi", "stretched")
    for bad in ("racah", "pi_manifold_rms"):
        with pytest.raises(ValueError, match="unknown mu_RF convention"):
            mu_rf(CS133, (47, 2, 2.5), (48, 1, 1.5), convention=bad)


def test_low_n_consensus_refused() -> None:
    """The consensus entry refuses below the Ritz hard floor (spec 01/02):
    D-line dipoles must come from experiment or the explicit closure path."""
    with pytest.raises(IntegrityError):
        mu_rf(RB87, (5, 0, 0.5), (5, 1, 1.5))
