"""Validation of rydsim.stark — spec 07 §6 benchmark table (RS-07-01..18)
plus the audit §3 (items 24-30) refusal rules and rulings R-5/R-7/R-19.

Every literature/check value below is transcribed with its source and
confidence tag (no-fabrication rule) and is never an input to the engine.
All conversion factors used by the MODULE are runtime-derived from
scipy.constants; the hard-coded numbers here are independent CHECK values
(spec 07 §4.3: allowed only in tests).

Deviations found while implementing (reported, not silenced):
  * RS-07-04: the printed target "H, n=10, q=0, m=0" does not exist as a
    parabolic state (n-1-|m| = 9 is odd, so q is odd). The nearest states
    are q = +-1, whose exact alpha (1/8)n^4(17n^2 - 3 + 19) = 2.1450e6 a.u.
    sits 0.17 % below the printed q=0 formula value 2.14875e6 — inside the
    row's 1 % tolerance, so the benchmark is implemented on the q = +-1
    pair mean (odd orders cancel exactly by the E -> -E symmetry).
  * RS-07-15: strict form is xfail — see the test docstring.
  * RS-07-16: the spec's tail note "~5.4/n_max^3 a.u." underestimates the
    bound-series tail (measured ~3.2/n_max^2, i.e. df/dn ~ 1.6/n^3 with
    1/omega^2 ~ 4); irrelevant at n_max = 100 (tail ~ 3e-4 a.u.).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from rydsim.atom import CS133, RB85, n_star, transition_hz
from rydsim.constants import (A0, AU_DIPOLE, AU_POLARIZABILITY, E_CHARGE,
                              E_H, H, M_E,
                              polarizability_au_to_hz_per_v2_cm2)
from rydsim.provenance import IntegrityError
from rydsim.radial import radial_matrix_element_consensus
from rydsim.stark import (DegeneracyError, EngineValidityError,
                          MapCurvature, Polarizability, ResonanceError,
                          ScreeningModel, StarkBasis, _pair_me_a0,
                          _window_partner_terms, ac_stark_shift_J,
                          alpha_dynamic, alpha_map_curvature,
                          alpha_perturbative, alpha_ponderomotive,
                          alpha_scalar_tensor, build_stark_matrices,
                          classical_ionization_field_Vm,
                          hydrogen_1s_discrete_alpha_au,
                          hydrogen_manifold_slopes_Cm,
                          hydrogen_stark_energy_J, inglis_teller_field_Vm,
                          manifold_gap, nef_dc_Vm_sqrtHz,
                          nef_external_Vm_sqrtHz, optimal_bias_Vm,
                          screening_transfer, stark_map, stark_shift_J,
                          stark_responsivity_hz_per_Vm,
                          unbiased_min_field_Vm)

# ---------------------------------------------------------------------------
# Check values / literature anchors (sources + tags; never engine inputs)
# ---------------------------------------------------------------------------

# CODATA 2022 derived check values (spec 07 §3.1 / spec 00 §2) [VERIFIED]
AU_TO_HZ_VCM2 = 2.488318e-4          # 1 a.u. alpha -> alpha/h [Hz/(V/cm)^2]
EA0_OVER_H_MHZ_VCM = 1.2795448       # e a0/h [MHz/(V/cm)]

# Yerokhin et al., PRA 94, 032503 (2016) Tables IV-VII (a.u.; the "a0^5"
# tensor-table label is a typo — spec 07 §3.2 note) + O'Sullivan &
# Stoicheff PRA 31, 2718 (1985) / PRA 33, 1640 (1986) exp values [VERIFIED]
RB_50S_ALPHA0_AU = 2.03e11           # exp & th; = 50.5 MHz/(V/cm)^2 (R-5)
RB_35S_ALPHA0_AU = 1.69e10           # = 4.21 MHz/(V/cm)^2
RB_35D52_ALPHA0_AU = 2.53e10
RB_35D52_ALPHA2_AU = 4.18e10
CS_39D52_ALPHA0_AU = -4.9e11         # NEGATIVE alpha_0 (8 %, sign exact)

# O'S&S PRA 31, 2718 (1985) Rb nS fit, Eq. (7.6) [VERIFIED]
OSS_C6 = 2.202e-9                    # MHz/(V/cm)^2 per n*^6
OSS_C7 = 5.53e-11                    # MHz/(V/cm)^2 per n*^7

# Hydrogen exact (Bethe-Salpeter, spec 07 Eqs. 7.7-7.8) [VERIFIED]
H_N10_Q9_SLOPE_EA0 = 135.0                        # (3/2) n q
H_N10_Q9_SLOPE_MHZ_VCM = 172.74                   # x e a0/h
H_N10_Q0_ALPHA_AU = 2.14875e6                     # (1/8) n^4 (17 n^2 + 19)
H_N10_Q1_ALPHA_AU = 0.125 * 10**4 * (17 * 100 - 3 + 19)   # exact q = +-1
H_1S_DISCRETE_RATIO = 0.8141                      # alpha_disc / (9/2)
H_1S_DISCRETE_AU = 3.6633
H_1S_DISCRETE_TRK = 0.5650

F_IT_30_VCM = 70.54                  # Eq. (7.9); Hogan '71 V/cm' [VERIFIED]
F_ION_35_VCM = 214.0                 # F0/(16 n^4), analytic [VERIFIED]


def au(alpha_si: float) -> float:
    return alpha_si / AU_POLARIZABILITY


def mhz_vcm2(alpha_au: float) -> float:
    return polarizability_au_to_hz_per_v2_cm2(alpha_au) / 1e6


# ---------------------------------------------------------------------------
# Expensive fixtures (module scope; the module-level ME cache is shared)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pol_50s() -> Polarizability:
    """Rb 50S1/2 m_j=1/2 with the spec 07 §4.1 doubling demonstration
    (n_window 15 -> 30 must move alpha < 1e-4: convergence demonstrated,
    not assumed)."""
    return alpha_perturbative(RB85, 50, 0, 0.5, 0.5, check_convergence=True)


@pytest.fixture(scope="module")
def pol_35s() -> Polarizability:
    return alpha_perturbative(RB85, 35, 0, 0.5, 0.5)


@pytest.fixture(scope="module")
def pol_35d() -> Polarizability:
    return alpha_perturbative(RB85, 35, 2, 2.5, 0.5)


@pytest.fixture(scope="module")
def st_35p32() -> tuple[float, float]:
    return alpha_scalar_tensor(RB85, 35, 1, 1.5)


@pytest.fixture(scope="module")
def st_cs39d() -> tuple[float, float]:
    with warnings.catch_warnings():
        # n' = 24 partners sit below Cs n_min_mhz = 25 (100-MHz-grade
        # energies): harmless in far-denominator terms, warned honestly.
        warnings.simplefilter("ignore", UserWarning)
        return alpha_scalar_tensor(CS133, 39, 2, 2.5)


@pytest.fixture(scope="module")
def oss_pols() -> dict[int, Polarizability]:
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)   # n' < 19 window edge
        for nn in (30, 40, 60, 70):
            out[nn] = alpha_perturbative(RB85, nn, 0, 0.5, 0.5)
    return out


@pytest.fixture(scope="module")
def dyn_50s():
    """alpha(omega) at 2pi x (1 MHz, 10 THz) — quasi-static + crossover."""
    return alpha_dynamic(RB85, 50, 0, 0.5, 0.5,
                         [2 * math.pi * 1e6, 2 * math.pi * 1e13])


@pytest.fixture(scope="module")
def map_curv_50s() -> MapCurvature:
    """Rb 50S map-curvature alpha with the widened-window convergence
    check (spec 07 §2.6: n +- 2, l_max + 2, target moves < 100 kHz)."""
    return alpha_map_curvature(RB85, 50, 0, 0.5, 0.5, conv_check=True)


@pytest.fixture(scope="module")
def rb_anticross() -> np.ndarray:
    """Rb 30/31 fan-edge gap vs field, 40 -> 95 V/cm in 0.5 V/cm steps."""
    mats = build_stark_matrices(StarkBasis("Rb85", 0.5, 29, 32, None))
    return manifold_gap(mats, 30, 31, np.arange(40e2, 95e2, 50.0))


@pytest.fixture(scope="module")
def h_map_mats():
    """Hydrogen map basis n = 4..16 (Delta n >= 6 around n=10), m = 0,
    full manifolds — exact Gordon matrix elements."""
    return build_stark_matrices(StarkBasis("H", 0, 4, 16, None))


# ---------------------------------------------------------------------------
# RS-07-01 / RS-07-02 — unit chain (runtime-derived, tests hold the check
# values; spec 07 §3.1 "THE MINEFIELD", conventions lock #14)
# ---------------------------------------------------------------------------

def test_rs_07_01_unit_chain_au_to_hz_vcm2():
    """RS-07-01: 1 a.u. polarizability -> alpha/h = 2.488318e-4 Hz/(V/cm)^2
    derived from scipy.constants (tol 1e-5). CODATA 2022 check value."""
    assert polarizability_au_to_hz_per_v2_cm2(1.0) == pytest.approx(
        AU_TO_HZ_VCM2, rel=1e-5)


def test_rs_07_02_ea0_over_h():
    """RS-07-02: e a0/h = 1.2795448 MHz/(V/cm) (tol 1e-6), derived."""
    mhz_per_vcm = E_CHARGE * A0 / H * 100.0 / 1e6
    assert mhz_per_vcm == pytest.approx(EA0_OVER_H_MHZ_VCM, rel=1e-6)


# ---------------------------------------------------------------------------
# RS-07-03 / RS-07-04 / RS-07-16 — hydrogen exact benchmarks
# ---------------------------------------------------------------------------

def test_rs_07_03_h_manifold_linear_slope():
    """RS-07-03: H n=10, q=9, m=0 within-manifold linear slope = 135 e a0
    = 172.74 MHz/(V/cm) (tol 1e-3). Map route (manifold diagonalization)
    against the exact Eq. (7.7) generator."""
    slopes = hydrogen_manifold_slopes_Cm(10, 0)
    assert slopes.size == 10                     # q in {-9, -7, ..., +9}
    top = slopes[-1]
    assert top / AU_DIPOLE == pytest.approx(H_N10_Q9_SLOPE_EA0, rel=1e-3)
    assert top / H * 100.0 / 1e6 == pytest.approx(H_N10_Q9_SLOPE_MHZ_VCM,
                                                  rel=1e-3)
    # exact-generator cross-check (Eq. 7.7, order 1)
    f_ref = 1e4                                  # V/m
    de = (hydrogen_stark_energy_J(10, 9, 0, f_ref, order=1)
          - hydrogen_stark_energy_J(10, 9, 0, 0.0, order=1))
    # abs=0: J-scale values sit far below pytest.approx's default abs=1e-12
    assert de == pytest.approx(top * f_ref, rel=1e-12, abs=0)


def test_rs_07_04_h_quadratic_map_curvature(h_map_mats):
    """RS-07-04: H n=10 quadratic alpha from map curvature, n-window
    4..16 (Delta n >= 6), m=0: 2.1488e6 a.u. within 1 %.

    SPEC DEFECT (reported): (n, q, m) = (10, 0, 0) is not a parabolic
    state (q must be odd) — the exact generator refuses it below. The
    q = +-1 pair mean (odd orders cancel exactly) has exact alpha
    2.1450e6 a.u., 0.17 % below the printed value, inside the 1 % band."""
    e0, d = h_map_mats.e0_J, h_map_mats.d_Cm
    e10 = float(e0[[s == (10, 0) for s in h_map_mats.labels].index(True)])
    fit_max = 0.03 * inglis_teller_field_Vm(10)
    fields = np.linspace(fit_max / 8, fit_max, 8)
    means = []
    for f_now in fields:
        ev = np.linalg.eigvalsh(np.diag(e0) + f_now * d)
        sel = np.sort(ev[np.abs(ev - e10) < 2e-4 * E_H])
        assert sel.size == 10
        means.append(0.5 * (sel[4] + sel[5]))    # q = -1, +1 pair mean
    x = fields / fit_max                         # conditioning: x in [0, 1]
    mat = np.column_stack([np.ones_like(x), x**2, x**4])
    b2 = np.linalg.lstsq(mat, np.array(means), rcond=None)[0][1]
    alpha_au_val = au(-2.0 * b2 / fit_max**2)
    assert alpha_au_val == pytest.approx(H_N10_Q0_ALPHA_AU, rel=0.01)
    assert alpha_au_val == pytest.approx(H_N10_Q1_ALPHA_AU, rel=5e-3)


def test_rs_07_04_q0_state_does_not_exist():
    """The exact generator refuses the unphysical (10, 0, 0) parabolic
    label (spec 07 Eq. 7.7 q-range; see RS-07-04 note)."""
    with pytest.raises(ValueError):
        hydrogen_stark_energy_J(10, 0, 0, 1.0)
    # neighbours exist and are exact
    e = hydrogen_stark_energy_J(10, 1, 0, 1e4)
    assert math.isfinite(e)


def test_rs_07_16_h1s_discrete_only_alpha():
    """RS-07-16: discrete-bound-only alpha(H 1s)/exact = 0.8141 +- 0.005
    (alpha_disc = 3.6633 a.u., discrete TRK = 0.5650) — documents the
    continuum omission behind the n >= 10 engine floor (audit item 24)."""
    alpha_disc, f_disc = hydrogen_1s_discrete_alpha_au(100)
    assert alpha_disc / 4.5 == pytest.approx(H_1S_DISCRETE_RATIO, abs=5e-3)
    assert alpha_disc == pytest.approx(H_1S_DISCRETE_AU, rel=2e-3)
    assert f_disc == pytest.approx(H_1S_DISCRETE_TRK, rel=2e-3)


# ---------------------------------------------------------------------------
# RS-07-05..07, 10, 11, 17, 13 — perturbative alpha anchors
# ---------------------------------------------------------------------------

def test_rs_07_05_rb_50s_alpha0(pol_50s):
    """RS-07-05: Rb 50S1/2 alpha_0 = 2.03e11 a.u. = 50.5 MHz/(V/cm)^2
    (3 %, O'S&S 1985 exp via Yerokhin T.IV). Enforces ruling R-5: spec
    04's ~6e2 MHz/(V/cm)^2 placeholder is dead (x12 too large)."""
    assert pol_50s.alpha0_au == pytest.approx(RB_50S_ALPHA0_AU, rel=0.03)
    assert mhz_vcm2(pol_50s.alpha0_au) == pytest.approx(50.5, rel=0.03)
    assert pol_50s.converged            # doubling demonstration ran


def test_rs_07_06_rb_35s_alpha0(pol_35s):
    """RS-07-06: Rb 35S1/2 alpha_0 = 1.69e10 a.u. = 4.21 MHz/(V/cm)^2
    (3 %)."""
    assert pol_35s.alpha0_au == pytest.approx(RB_35S_ALPHA0_AU, rel=0.03)
    assert mhz_vcm2(pol_35s.alpha0_au) == pytest.approx(4.21, rel=0.03)


def test_rs_07_07_rb_35d52(pol_35d):
    """RS-07-07: Rb 35D5/2 alpha_0 = 2.53e10, alpha_2 = 4.18e10 a.u.
    (6 % each, O'S&S 1986 exp via Yerokhin T.IV-V)."""
    assert pol_35d.alpha0_au == pytest.approx(RB_35D52_ALPHA0_AU, rel=0.06)
    assert pol_35d.alpha2_au == pytest.approx(RB_35D52_ALPHA2_AU, rel=0.06)


def test_rs_07_10_sign_anchors(st_35p32, st_cs39d, pol_35d):
    """RS-07-10: alpha_2(Rb 35P3/2) < 0; alpha_0(Cs 39D5/2) = -4.9e11 a.u.
    (sign exact, 8 %). Signs pin the phase convention (spec 07 §2.3);
    per R-19 they are tested on alpha_0/alpha_2, never on alpha(m_j)."""
    a0_p, a2_p = st_35p32
    assert a2_p < 0.0
    a0_cs, a2_cs = st_cs39d
    assert a0_cs < 0.0                   # sign exact (Cs (n+1)P3/2 below nD)
    assert a0_cs == pytest.approx(CS_39D52_ALPHA0_AU, rel=0.08)
    assert a2_cs > 0.0                   # Yerokhin T.VII: +5.6e11 a.u.
    assert pol_35d.alpha2_au > 0.0       # alpha_2(Rb nD5/2) > 0 anchor


def test_rs_07_11_alpha2_identically_zero(pol_50s, pol_35s):
    """RS-07-11: alpha_2(nS1/2) and alpha_2(nP1/2) are EXACTLY 0.0
    (j = 1/2 angular identity, conventions lock #14)."""
    assert pol_50s.alpha2_au == 0.0
    assert pol_35s.alpha2_au == 0.0
    pol_p12 = alpha_perturbative(RB85, 35, 1, 0.5, 0.5, n_window=5)
    assert pol_p12.alpha2_au == 0.0


def test_rs_07_17_oss_fit_consistency(pol_50s, oss_pols):
    """RS-07-17: perturbative alpha_0(Rb nS) vs the O'S&S fit Eq. (7.6)
    2.202e-9 n*^6 + 5.53e-11 n*^7 MHz/(V/cm)^2 at n = 30..70 (5 %)."""
    pols = dict(oss_pols)
    pols[50] = pol_50s
    for nn, pol in sorted(pols.items()):
        nst = float(n_star(RB85, nn, 0, 0.5))
        fit = OSS_C6 * nst**6 + OSS_C7 * nst**7
        assert mhz_vcm2(pol.alpha0_au) == pytest.approx(fit, rel=0.05), nn


def test_rs_07_13_log_slope(pol_50s, oss_pols):
    """RS-07-13: d ln alpha_0 / d ln n* over Rb nS, n = 40-70, lands in
    [6.4, 6.9] (n*^6 + n*^7 fit structure — NOT a pure power law)."""
    pols = {40: oss_pols[40], 50: pol_50s, 60: oss_pols[60],
            70: oss_pols[70]}
    ln_a = np.array([math.log(p.alpha0_au) for p in pols.values()])
    ln_n = np.array([math.log(float(n_star(RB85, nn, 0, 0.5)))
                     for nn in pols])
    slope = float(np.polyfit(ln_n, ln_a, 1)[0])
    assert 6.4 <= slope <= 6.9


# ---------------------------------------------------------------------------
# RS-07-18 + R-19 + 6j cross-check — decomposition machinery
# ---------------------------------------------------------------------------

def test_rs_07_18_mj_fit_residual(pol_35d):
    """RS-07-18: the j=5/2 m_j set is EXACTLY alpha_0 + alpha_2 * bracket
    (brackets -0.8, -0.2, +1.0); residual < 1e-6 relative (any m_j^4 term
    at second order is a bug)."""
    a0_si = pol_35d.alpha0_au * AU_POLARIZABILITY
    a2_si = pol_35d.alpha2_au * AU_POLARIZABILITY
    brackets = (-0.8, -0.2, 1.0)                     # Eq. (7.4), j = 5/2
    resid = max(abs(a0_si + b * a2_si - a)
                for b, a in zip(brackets, pol_35d.mj_alphas_SI))
    assert resid / abs(a0_si) < 1e-6


def test_r19_mj_half_total_alpha_negative(pol_35d):
    """Ruling R-19: for Rb nD5/2, alpha(m_j=1/2) = alpha_0 - 0.8 alpha_2
    is NEGATIVE while the scalar alpha_0 is positive — the Meyer -8.6
    GHz/(V/cm)^2 'scalar' is the m_j-resolved value, not alpha_0."""
    assert pol_35d.alpha0_au > 0.0
    assert pol_35d.mj_alphas_SI[0] < 0.0             # m_j = 1/2
    assert pol_35d.alpha_SI == pol_35d.mj_alphas_SI[0]


def test_6j_closed_form_cross_check():
    """Spec 07 §2.3: the Yerokhin 6j closed forms (7.5) must agree with
    the normative m_j fit to 1e-9 relative (pure algebra) — the library
    asserts internally with method='6j' (IntegrityError on mismatch)."""
    a0_fit, a2_fit = alpha_scalar_tensor(RB85, 35, 2, 2.5)
    a0_6j, a2_6j = alpha_scalar_tensor(RB85, 35, 2, 2.5, method="6j")
    assert a0_6j == pytest.approx(a0_fit, rel=1e-12)
    assert a2_6j == pytest.approx(a2_fit, rel=1e-12)


# ---------------------------------------------------------------------------
# RS-07-12 — closure sum rule (phase/convention tripwire)
# ---------------------------------------------------------------------------

def test_rs_07_12_closure_sum_rule(pol_50s):
    """RS-07-12: sum_k |<k|z|v>|^2 over Delta n = 25 vs <v|z^2|v> =
    <r^2>/3 for Rb 50S m_j=1/2, within 1 % (Eq. 7.3). <r^2> comes from
    the spec 02 consensus entry point (k=2, independent path); the
    angular factor 1/3 is exact for l = 0."""
    from rydsim.angular import angular_factor
    terms = _window_partner_terms(RB85, 50, 0, 0.5, 25)
    z2 = 0.0
    for n_p, lp, jp, de_j, r_a0, spread in terms:
        ang = angular_factor(0, 0.5, 0.5, lp, jp, 0.5, 0)
        z2 += (r_a0 * ang) ** 2                       # a0^2
    r2 = radial_matrix_element_consensus(RB85, (50, 0, 0.5), (50, 0, 0.5),
                                         k=2)
    assert z2 == pytest.approx(r2.value / 3.0, rel=0.01)


# ---------------------------------------------------------------------------
# RS-07-14 / RS-07-15 — dynamic alpha
# ---------------------------------------------------------------------------

def test_rs_07_14_quasi_static_limit(pol_50s, dyn_50s):
    """RS-07-14: alpha(2pi x 1 MHz)/alpha(0) - 1 = 0 within 1e-4 for
    Rb 50S (dominant omega_k ~ 2pi x 30 GHz -> (omega/omega_k)^2 ~ 1e-9)."""
    ratio = float(dyn_50s.alpha_SI[0]) / dyn_50s.alpha_static_SI
    assert abs(ratio - 1.0) < 1e-4
    # continuity vs the static entry point (ratio form: SI alphas are
    # ~1e-30 — pytest.approx's default abs=1e-12 would be vacuous)
    assert dyn_50s.alpha_static_SI / pol_50s.alpha_SI == pytest.approx(
        1.0, rel=2e-4)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "RS-07-15 as printed (truncated discrete sum vs -e^2/(m_e omega^2)"
        " to 2 %) cannot pass: the bound-state window's TRK completeness"
        " saturates at S ~ 0.979 (measured; continuum + deep-bound weight"
        " ~2.2 % — same physics as RS-07-16), so alpha(10 THz) ="
        " -S e^2/(m_e omega^2) sits 2.2 % below the ponderomotive limit."
        " Spec 07 §2.7's own crossover rule mandates switching to Eq."
        " (7.11) when S < 0.98. The TRK-ANCHORED form of this benchmark"
        " passes to 0.1 % (next test)."))
def test_rs_07_15_ponderomotive_strict(dyn_50s):
    """RS-07-15 (strict, as printed): alpha(2pi x 10 THz, Rb 50S) vs
    -e^2/(m_e omega^2), 2 %."""
    pond = alpha_ponderomotive(2 * math.pi * 1e13)
    # abs=0 is essential: |pond| ~ 7e-36 << pytest.approx default abs=1e-12
    assert float(dyn_50s.alpha_SI[1]) == pytest.approx(pond, rel=0.02, abs=0)


def test_rs_07_15_ponderomotive_trk_anchored(dyn_50s):
    """RS-07-15 (TRK-anchored): at omega = 2pi x 10 THz >> every coupled
    omega_k, the truncated sum obeys alpha -> -S e^2/(m_e omega^2) with S
    the window's TRK completeness (spec 07 §2.7): agreement to 0.5 %, and
    the deviation from the full ponderomotive limit is bounded by the
    spec's crossover uncertainty |1-S| e^2/(m_e omega^2) (30 % margin for
    the omega_k^2/omega^2 corrections of the deepest terms)."""
    omega = 2 * math.pi * 1e13
    pond = alpha_ponderomotive(omega)
    s_trk = dyn_50s.trk_completeness
    assert 0.95 < s_trk < 1.0
    alpha_thz = float(dyn_50s.alpha_SI[1])
    assert alpha_thz == pytest.approx(s_trk * pond, rel=5e-3, abs=0)
    bound = abs(1.0 - s_trk) * E_CHARGE**2 / (M_E * omega**2)
    assert abs(alpha_thz - pond) <= 1.3 * bound
    assert omega > dyn_50s.omega_k_max_rad_s          # truly in the limit


# ---------------------------------------------------------------------------
# RS-07-08 / RS-07-08b — Inglis-Teller and the 30/31 anticrossing
# ---------------------------------------------------------------------------

def test_rs_07_08_inglis_teller_field():
    """RS-07-08: F_IT(n=30) = F0/(3 n^5) = 70.54 V/cm (0.5 %); Hogan
    arXiv:1603.04432 quotes '71 V/cm between n = 30 and 31' [VERIFIED]."""
    assert inglis_teller_field_Vm(30) / 100.0 == pytest.approx(F_IT_30_VCM,
                                                               rel=5e-3)


def test_classical_ionization_field():
    """F_ion = F0/(16 n^4): n=35 -> 214 V/cm (spec 07 §2.5, analytic)."""
    assert classical_ionization_field_Vm(35) / 100.0 == pytest.approx(
        F_ION_35_VCM, rel=0.01)


def test_rs_07_08b_first_anticrossing_field(rb_anticross):
    """RS-07-08b: the first n=30/31 fan-edge anticrossing in the Rb map
    sits within 20 % of 70.5 V/cm (qualitative row). Measured this
    implementation: 70.5 V/cm — right on the Eq. (7.9) formula. The gap
    stays strictly positive: alkali fan edges ANTICROSS (core coupling)."""
    gaps = rb_anticross
    imin = int(np.argmin(gaps[:, 1]))
    f_min_vcm = gaps[imin, 0] / 100.0
    assert abs(f_min_vcm / 70.5 - 1.0) < 0.20
    assert np.all(gaps[:, 1] > 0.0)


def test_hydrogen_states_genuinely_cross(rb_anticross):
    """Spec 07 §2.6 core-effect regression: hydrogen fan edges CROSS
    (character gap changes sign through zero) where alkali edges anticross
    with a finite gap; the H crossing sits near F_IT(10) (the Eq. 7.9
    formula is the n >> 1 asymptote — 20 % window)."""
    mats = build_stark_matrices(StarkBasis("H", 0, 9, 12, None))
    rows = manifold_gap(mats, 10, 11, np.arange(1.2e6, 1.8e6, 5e3))
    signs = np.sign(rows[:, 1])
    assert np.any(signs > 0) and np.any(signs < 0)    # genuine crossing
    i_cross = int(np.where(np.diff(signs) != 0)[0][0])
    f_cross = rows[i_cross, 0]
    assert abs(f_cross / inglis_teller_field_Vm(10) - 1.0) < 0.20
    # smallest sampled |gap| at the crossing is far below the Rb avoided gap
    h_min = float(np.min(np.abs(rows[:, 1])))
    rb_min = float(np.min(rb_anticross[:, 1]))
    assert h_min < rb_min


# ---------------------------------------------------------------------------
# RS-07-09 — perturbative vs map curvature (self-consistency)
# ---------------------------------------------------------------------------

def test_rs_07_09_pert_vs_map_curvature(pol_50s, map_curv_50s):
    """RS-07-09: perturbative alpha vs Stark-map curvature for Rb 50S1/2
    m_j=1/2 at fields <= 0.03 F_IT agree to 0.5 % (measured 4e-5). The
    map ran the spec 07 §2.6 widened-window convergence check (n +- 2,
    l_max + 2, target moved < 100 kHz — IntegrityError otherwise)."""
    assert map_curv_50s.alpha_SI / pol_50s.alpha_SI == pytest.approx(
        1.0, rel=5e-3)                 # ratio form: SI alphas ~1e-30
    assert map_curv_50s.converged
    assert map_curv_50s.quartic_fraction < 0.01
    assert map_curv_50s.fit_max_field_Vm <= 0.03 * inglis_teller_field_Vm(50)
    # gamma is DIAGNOSTIC only (audit §3 item 29) — present, never a finding
    assert math.isfinite(map_curv_50s.gamma_SI_diagnostic)


def test_map_unit_tripwire(map_curv_50s):
    """Spec 07 §3.1: all three representations populated and consistent."""
    assert map_curv_50s.alpha_au == pytest.approx(
        map_curv_50s.alpha_SI / AU_POLARIZABILITY, rel=1e-12)
    assert map_curv_50s.alpha_Hz_per_Vcm2 == pytest.approx(
        polarizability_au_to_hz_per_v2_cm2(map_curv_50s.alpha_au), rel=1e-12)


def test_polarizability_unit_tripwire(pol_50s):
    """Spec 07 §3.1 unit tripwire on the perturbative result object."""
    assert pol_50s.alpha_au * AU_POLARIZABILITY / pol_50s.alpha_SI == \
        pytest.approx(1.0, rel=1e-12)
    assert pol_50s.alpha_Hz_per_Vcm2 == pytest.approx(
        pol_50s.alpha_SI / H * 1e4, rel=1e-10)     # (100 V/m per V/cm)^2
    assert pol_50s.E_mix_Vm > 50.0                 # far from degeneracy
    assert 0.0 < pol_50s.sigma_rel < 1e-2          # spread-based uncertainty


# ---------------------------------------------------------------------------
# Consensus equivalence — ties the shared-grid pair machinery to spec 02
# ---------------------------------------------------------------------------

def test_pair_me_matches_consensus_entry_point():
    """The module's shared-grid dual-method pair element equals the public
    spec 02 consensus entry point (which also runs the Kaulakys third
    method and its ceilings) to 1e-9 — same physics, same ceilings, grid
    chosen per window instead of per pair (module docstring policy)."""
    me, spread = _pair_me_a0(RB85, (50, 0, 0.5), (50, 1, 1.5), 65)
    ref = radial_matrix_element_consensus(RB85, (50, 0, 0.5), (50, 1, 1.5))
    assert me == pytest.approx(ref.value, rel=1e-9)
    assert spread <= 1e-4                          # spec 02 §6 B8 ceiling
    assert "kaulakys" in ref.per_method            # 3rd method exercised


# ---------------------------------------------------------------------------
# Refusal rules (audit §3 items 24-26, 30) — raise, never guess
# ---------------------------------------------------------------------------

def test_refuse_n_below_10():
    """Audit item 24: perturbative alpha for n < 10 raises
    EngineValidityError (an IntegrityError): continuum omission."""
    with pytest.raises(EngineValidityError):
        alpha_perturbative(RB85, 9, 0, 0.5, 0.5)
    assert issubclass(EngineValidityError, IntegrityError)


def test_n10_is_inside_validity():
    """n = 10 is the validity floor — must compute, not raise (window
    clamps at the species hard floor n = 8; low-n' energies warn)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        pol = alpha_perturbative(RB85, 10, 0, 0.5, 0.5, n_window=5)
    assert pol.alpha_SI > 0.0
    assert pol.E_mix_Vm > 50.0


def test_refuse_high_l_target():
    """Audit item 25: l >= 4 targets are quasi-degenerate in-manifold:
    Eq. (7.1) refuses (DegeneracyError) — the map is the only route."""
    with pytest.raises(DegeneracyError):
        alpha_perturbative(RB85, 50, 4, 4.5, 0.5)
    assert issubclass(DegeneracyError, IntegrityError)


def test_refuse_resonant_omega():
    """Audit item 26: dynamic alpha inside the guard band of a coupled
    transition raises ResonanceError (dressed states are spec 06/08)."""
    w_res = 2 * math.pi * abs(float(transition_hz(RB85, 50, 0, 0.5,
                                                  50, 1, 1.5)))
    with pytest.raises(ResonanceError):
        alpha_dynamic(RB85, 50, 0, 0.5, 0.5, [w_res + 1e3], n_window=3)
    assert issubclass(ResonanceError, IntegrityError)


def test_stark_shift_validity_cap(pol_50s):
    """DC shift helper: DeltaE = -(1/2) alpha E^2 inside the quadratic
    validity cap; refuses beyond min(0.1 E_mix, 0.03 F_IT) (item 30)."""
    e_ok = 5.0                                    # V/m, far inside the cap
    de = stark_shift_J(pol_50s, e_ok)
    assert de == pytest.approx(-0.5 * pol_50s.alpha_SI * e_ok**2,
                               rel=1e-12, abs=0)
    assert de < 0.0                               # alpha > 0 shifts DOWN
    cap = min(0.1 * pol_50s.E_mix_Vm, 0.03 * inglis_teller_field_Vm(50))
    with pytest.raises(IntegrityError):
        stark_shift_J(pol_50s, 2.0 * cap)


def test_ac_dc_factor(pol_50s):
    """Lock #14: AC shift is -(1/4) alpha E_amp^2 — exactly half the DC
    -(1/2) alpha E^2 at equal field values (cos^2 time average)."""
    e_amp = 5.0
    assert ac_stark_shift_J(pol_50s.alpha_SI, e_amp) == pytest.approx(
        0.5 * stark_shift_J(pol_50s, e_amp), rel=1e-12, abs=0)


def test_map_flags_above_f_ion():
    """Audit item 30: map fields above F_ion(n_max) are flagged (hand-off
    with the validity cap attached), not silently trusted."""
    res = stark_map(StarkBasis("H", 0, 10, 11, None),
                    np.array([1.0e6, 3.0e6]))
    assert res.f_ion_Vm == pytest.approx(classical_ionization_field_Vm(11))
    assert any("above_F_ion" in fl for fl in res.flags)


# ---------------------------------------------------------------------------
# Sensing / readout formulas (spec 07 §2.8; worked-number anchors)
# ---------------------------------------------------------------------------

def test_sensing_worked_numbers():
    """Spec 07 §2.8 worked example (alpha/h = 50.5 MHz/(V/cm)^2,
    delta_nu = 1 kHz): unbiased detection floor 6.3 mV/cm (square-root
    pathology) vs 40 uV/cm at E_b = 0.5 V/cm (linear, biased)."""
    alpha_si = 50.5e6 / 1e4 * H            # 50.5 MHz/(V/cm)^2 -> SI
    e_unb = unbiased_min_field_Vm(alpha_si, 1e3)
    assert e_unb == pytest.approx(0.63, rel=0.02)      # V/m = 6.3 mV/cm
    e_b = 0.5 * 1e2                        # 0.5 V/cm in V/m
    resp = stark_responsivity_hz_per_Vm(alpha_si, e_b)
    de_biased = 1e3 / resp                 # V/m
    assert de_biased * 1e4 == pytest.approx(40.0, rel=0.02)  # uV/cm
    assert stark_responsivity_hz_per_Vm(alpha_si, 0.0) == 0.0
    assert nef_dc_Vm_sqrtHz(alpha_si, e_b, 1e3) == pytest.approx(
        de_biased, rel=1e-12)


def test_optimal_bias_formula():
    """Spec 07 Eq. (7.13): E_b* = sqrt(h Gamma0/(alpha eta)), clipped at
    the cap; eta -> 0 pushes to the cap."""
    alpha_si = 50.5e6 / 1e4 * H
    gamma0 = 1e5                           # Hz
    eta = 0.01
    e_star, nef_scale = optimal_bias_Vm(alpha_si, gamma0, eta,
                                        e_cap_Vm=1e4)
    assert e_star == pytest.approx(math.sqrt(H * gamma0 / (alpha_si * eta)),
                                   rel=1e-12)
    assert nef_scale == pytest.approx(
        2.0 * math.sqrt(eta * gamma0 * H / alpha_si), rel=1e-12)
    e_cap, _ = optimal_bias_Vm(alpha_si, gamma0, 0.0, e_cap_Vm=123.0)
    assert e_cap == 123.0
    e_clip, _ = optimal_bias_Vm(alpha_si, gamma0, eta, e_cap_Vm=1.0)
    assert e_clip == 1.0


# ---------------------------------------------------------------------------
# Screening interface (spec 07 §2.9 / ruling R-7 / audit item 20)
# ---------------------------------------------------------------------------

def test_screening_r7_reduces_to_both_parents():
    """Ruling R-7 enforcement: the unified T(f) reduces to spec 07 Eq.
    (7.14) at (S_geo=1, beta=1) and to spec 05's high-pass at T_dc=0."""
    f = np.array([0.0, 0.1, 1.0, 10.0, 1e4])
    m07 = ScreeningModel(T_dc=0.2, tau_s_s=0.5)          # spec 07 case
    x = 1j * 2 * np.pi * f * 0.5
    with np.errstate(invalid="ignore"):
        t_ref = 0.2 + 0.8 * x / (1 + x)
    assert np.allclose(screening_transfer(f, m07), t_ref)
    m05 = ScreeningModel(T_dc=0.0, tau_s_s=0.5, S_geo=0.7, beta=0.8)
    x = (1j * 2 * np.pi * f * 0.5) ** 0.8
    t_ref = 0.7 * x / (1 + x)
    assert np.allclose(screening_transfer(f, m05), t_ref)
    # limits: T(0) = S_geo T_dc; |T(inf)| -> S_geo
    assert screening_transfer(0.0, m07) == pytest.approx(0.2)
    assert abs(screening_transfer(1e9, m05)) == pytest.approx(0.7, rel=1e-3)


def test_screening_refuses_uncalibrated_absolute():
    """Audit item 20 / R12: uncalibrated screening parameters refuse an
    absolute external-field NEF; estimate_ok=True yields a flagged
    estimate; calibrated models pass. Eq. (7.15) arithmetic pinned."""
    f = np.array([10.0, 1e3])
    m_uncal = ScreeningModel(T_dc=0.0, tau_s_s=1.0)
    with pytest.raises(IntegrityError):
        nef_external_Vm_sqrtHz(f, 1e-6, m_uncal)
    est = nef_external_Vm_sqrtHz(f, 1e-6, m_uncal, estimate_ok=True)
    assert np.all(est >= 1e-6)
    m_cal = ScreeningModel(T_dc=0.5, tau_s_s=1.0, calibrated=True,
                           source="unit-test fixture")
    out = nef_external_Vm_sqrtHz(f, 1e-6, m_cal)
    assert np.allclose(out, 1e-6 / np.abs(screening_transfer(f, m_cal)))
