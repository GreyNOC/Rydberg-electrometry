"""Validation of rydsim.dipoles — the specs 02+03 integration layer.

Benchmark plan (assignment per docs/spec/00-conventions.md SS4/SS7 step 5 and
docs/spec/00-integrity-audit.md R5): every literature anchor is RECOMPUTED
from scratch (consensus radial x angular) and compared against the recall
value, which is transcribed below with its source + confidence tag and never
used as an input. Rows implemented:

  * audit R5 / spec 08 SS3.2  — Jing mu_RF(Cs 47D5/2->48P3/2) = 1443.450
    e a0 (and the radial 2946.512 a0), 2 %.
  * spec 09 C5e — Sedlacek d(Rb 53D5/2->54P3/2) = 1.37e-26 C m, 2 %
    (task tolerance; corpus row allows 5 % with convention risk flagged).
  * spec 09 C5f / spec 08 SS3.3 — Tu d(Rb 39D5/2->40P3/2) = 1218 e a0, 2 %.
  * spec 02 B15 / audit R5 pattern — Rb-87 D2 closure: +5..10 % bias band,
    radial in [5.45, 5.70] a0.
  * task row — Cs D2 closure: same +5..10 % band.
"""

from __future__ import annotations

import math

import pytest

from rydsim.angular import angular_factor, effective_rf_dipole, steck_to_racah
from rydsim.atom import CS133, RB87
from rydsim.constants import AU_DIPOLE
from rydsim.dipoles import (DLINE_BIAS_BAND, MU_RF_CONVENTIONS, DipoleResult,
                            dline_closure, full_dipole_matrix_element, mu_rf)
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

# Sedlacek et al., Nat. Phys. 8, 819 (2012), params from arXiv:1205.4461 v1
# [VERIFIED number; convention risk flagged, spec 09 C5e/SS4.9]. Their
# "4-level model, stretched hyperfine states" effective RF dipole.
SEDLACEK_MU_CM = 1.37e-26          # [C m]   VERIFIED (number, v1 full text)

# Tu et al., Sci. Adv. 10, eads0683 (2024), PMC full text — spec 08 SS3.3 /
# spec 09 C5f [VERIFIED]. Ground 5S1/2 F=2 mF=2 (stretched sigma+ ladder).
TU_MU_EA0 = 1218.0                 # [e a0]  VERIFIED (PMC)

# Steck rev 2.3.4 D2 reduced dipoles (Steck convention) — spec 01/03
# [VERIFIED, primary PDFs]; the closure targets, reached only through the
# documented +5..10 % model-potential bias band (spec 02 SS7.1/B15).
STECK_D2_RB87_EA0 = 4.22752        # [e a0]  VERIFIED
STECK_D2_CS_EA0 = 4.4837           # [e a0]  VERIFIED

# Spec 02 B15 regression band on the Rb 5S1/2->5P3/2 model-potential radial
# integral (computed 5.569 a0 with measured-energy nu) [computed VERIFIED].
B15_RADIAL_BAND_A0 = (5.45, 5.70)


# ---------------------------------------------------------------------------
# Shared fixtures (consensus radial runs are the expensive part)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def jing() -> DipoleResult:
    return mu_rf(CS133, (47, 2, 2.5), (48, 1, 1.5))  # default nist_pi


@pytest.fixture(scope="module")
def sedlacek() -> DipoleResult:
    return mu_rf(RB87, (53, 2, 2.5), (54, 1, 1.5),
                 convention="pi_manifold_rms")


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
    assert jing.value_ea0 == pytest.approx(JING_MU_RF_EA0, rel=0.02)
    # measured agreement this session: 9.7e-6 relative
    assert jing.value_ea0 == pytest.approx(JING_MU_RF_EA0, rel=1e-3)


def test_jing_radial_recall_2pct(jing: DipoleResult) -> None:
    """Audit R5 (same row): the radial ME behind the Jing fixture,
    2946.512 a0 [LITERATURE-RECALL], reproduced by the spec 02 consensus."""
    assert abs(jing.radial_a0) == pytest.approx(JING_RADIAL_A0, rel=0.02)
    # angular factor is exactly sqrt(6)/5 (NIST 0.4899, spec 03 B14)
    assert jing.angular_factor == pytest.approx(math.sqrt(6.0) / 5.0, rel=1e-12)


def test_sedlacek_mu_rf_2pct(sedlacek: DipoleResult) -> None:
    """Spec 09 C5e (task tol 2 %): Sedlacek 2012 d(Rb 53D5/2->54P3/2) =
    1.37e-26 C m. Reproduced by the pi-manifold-rms convention
    (R/sqrt(5) for D5/2->P3/2); measured agreement 0.3 %. The paper's own
    'stretched hyperfine states' description does NOT reproduce the printed
    number (see test_sedlacek_stretched_convention_tension)."""
    assert sedlacek.value_Cm == pytest.approx(SEDLACEK_MU_CM, rel=0.02)
    # rms pi-manifold A for D5/2->P3/2 is exactly 1/sqrt(5)
    assert sedlacek.angular_factor == pytest.approx(1.0 / math.sqrt(5.0),
                                                    rel=1e-12)


def test_sedlacek_stretched_convention_tension() -> None:
    """Spec 09 C5e convention-risk documentation (SS4.9): the stretched
    fine-structure reading of Sedlacek's dipole misses their printed value
    by ~+42 % (computed 2291 e a0 vs 1616 e a0). Kept as a regression so
    nobody 'fixes' the convention silently."""
    d = mu_rf(RB87, (53, 2, 2.5), (54, 1, 1.5), convention="stretched")
    assert d.value_Cm > 1.3 * SEDLACEK_MU_CM


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
    [VERIFIED] inside the same +5..10 % band (measured +9.6 % this
    session — near the top edge; a breach flags the radial engine)."""
    res = dline_closure(CS133, "D2")
    assert res.in_band
    assert res.d_steck_measured_ea0 == STECK_D2_CS_EA0
    assert res.bias_rel == pytest.approx(0.096, abs=0.02)


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
    """Convention stamps are a closed set (audit SS4 item 7)."""
    with pytest.raises(ValueError, match="unknown mu_RF convention"):
        mu_rf(CS133, (47, 2, 2.5), (48, 1, 1.5), convention="racah")
    assert MU_RF_CONVENTIONS == ("nist_pi", "stretched", "pi_manifold_rms")


def test_low_n_consensus_refused() -> None:
    """The consensus entry refuses below the Ritz hard floor (spec 01/02):
    D-line dipoles must come from experiment or the explicit closure path."""
    with pytest.raises(IntegrityError):
        mu_rf(RB87, (5, 0, 0.5), (5, 1, 1.5))
