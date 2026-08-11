"""rydsim.zeeman tests — ruling R-17 test plan + spec 09 benchmark C9.

Test plan (00-conventions §5 R-17 / §8 enforcement item iv):
  * g_J exact values derived independently (projection theorem, exact
    rationals) — never copied from the implementation's formula;
  * mu_B/h = 1.399625 MHz/G check value;
  * tuning-law sign/symmetry checks;
  * the R-17 validity fence raises IntegrityError.
Spec 09 §6 row C9 (TIGHT, 1% rel) is implemented directly. E7.1/E7.2 are
validation-corpus benchmarks (spec 09 owner) needing the full sensing
pipeline and beyond-linear-Zeeman physics; they are out of this module's
scope (see test_e7_conditions_exceed_linear_fence for the documented fence
behaviour at the E7 field scale).
"""

import math
from fractions import Fraction

import numpy as np
import pytest
import scipy.constants as sc
from scipy.integrate import quad
from scipy.special import genlaguerre, lpmv

from rydsim.constants import H, MU_B
from rydsim.provenance import IntegrityError
from rydsim.zeeman import (
    DIAMAGNETIC_FENCE_FRACTION,
    DIAMAGNETIC_HZ_PER_T2_A0SQ,
    GAUSS_TO_TESLA,
    LINEAR_FENCE_FRACTION,
    MU_B_OVER_H_HZ_PER_T,
    ZeemanState,
    diamagnetic_shift_hz,
    gauss_to_tesla,
    hz_per_t_to_mhz_per_gauss,
    lande_g_j,
    mean_cos2_theta,
    mean_rho2_a0sq,
    mean_sin2_theta,
    require_linear_dominates,
    require_linear_regime,
    state_shift_hz,
    stretched_pair,
    stretched_tuning_rate_hz_per_t,
    transition_shift_hz,
    tuning_rate_hz_per_t,
)


def _g_j_projection_theorem(l: int, j: Fraction, s: Fraction = Fraction(1, 2)) -> Fraction:
    """Independent oracle: g_J from the projection theorem, exact rationals.

    g_J = g_L [j(j+1)+l(l+1)-s(s+1)]/(2j(j+1))
        + g_S [j(j+1)+s(s+1)-l(l+1)]/(2j(j+1)),  g_L = 1, g_S = 2.
    Algebraically distinct route from the module's 3/2 + [...] form, so the
    comparison is a derivation, not a copy (R-17 test-plan requirement).
    """
    jj = j * (j + 1)
    ll = Fraction(l * (l + 1))
    ss = s * (s + 1)
    return (jj + ll - ss) / (2 * jj) + 2 * (jj + ss - ll) / (2 * jj)


# (term, l, j, exact g_J) — expected column from the R-17 test plan; the
# oracle above re-derives each value so nothing is copied blind.
_GJ_CASES = [
    ("S1/2", 0, Fraction(1, 2), Fraction(2)),
    ("P1/2", 1, Fraction(1, 2), Fraction(2, 3)),
    ("P3/2", 1, Fraction(3, 2), Fraction(4, 3)),
    ("D3/2", 2, Fraction(3, 2), Fraction(4, 5)),
    ("D5/2", 2, Fraction(5, 2), Fraction(6, 5)),
    ("F5/2", 3, Fraction(5, 2), Fraction(6, 7)),
    ("F7/2", 3, Fraction(7, 2), Fraction(8, 7)),
]


@pytest.mark.parametrize("term,l,j,expected", _GJ_CASES, ids=[c[0] for c in _GJ_CASES])
def test_lande_g_j_exact_values(term, l, j, expected):
    """R-17 test plan: g_J exact values (S1/2: 2, P1/2: 2/3, P3/2: 4/3,
    D3/2: 4/5, D5/2: 6/5), derived — projection-theorem oracle in exact
    rational arithmetic must reproduce the expected fraction, and the
    module must match it to float precision."""
    assert _g_j_projection_theorem(l, j) == expected
    assert lande_g_j(l, float(j)) == pytest.approx(float(expected), rel=1e-14)


def test_mu_b_over_h_check_value():
    """R-17 check value: mu_B/h = 1.399625 MHz/G (CODATA arithmetic; also
    spec 09 C9's printed 1.39962 MHz/G within its 1% tolerance)."""
    mhz_per_g = hz_per_t_to_mhz_per_gauss(MU_B_OVER_H_HZ_PER_T)
    assert mhz_per_g == pytest.approx(1.399625, abs=1e-6)
    assert mhz_per_g == pytest.approx(1.39962, rel=0.01)
    # never typed by hand: the module constant is exactly scipy's ratio
    assert MU_B_OVER_H_HZ_PER_T == MU_B / H


def test_c9_cs_45d52_46p32_stretched_tuning_rate():
    """Spec 09 §6 benchmark C9 (TIGHT 1%): Zeeman tuning rate for Cs
    45D5/2 -> 46P3/2, stretched m_J, equals (mu_B/h)(g_J4 m_J4 - g_J3 m_J3)
    = (mu_B/h)(4/3*3/2 - 6/5*5/2) = -(mu_B/h), |rate| = 1.39962 MHz/G.
    n-independent (linear Zeeman), so the 45/46 labels carry no extra data."""
    rate = stretched_tuning_rate_hz_per_t(2, 2.5, 1, 1.5)  # D5/2 -> P3/2
    assert rate == pytest.approx(-MU_B_OVER_H_HZ_PER_T, rel=1e-12)
    assert abs(hz_per_t_to_mhz_per_gauss(rate)) == pytest.approx(1.39962, rel=0.01)


def test_e7_49s12_49p32_stretched_rate():
    """E7 fixture transition Cs 49S1/2 -> 49P3/2 stretched:
    (mu_B/h)(4/3*3/2 - 2*1/2) = +1 * mu_B/h (sign-reversed vs C9)."""
    rate = stretched_tuning_rate_hz_per_t(0, 0.5, 1, 1.5)
    assert rate == pytest.approx(+MU_B_OVER_H_HZ_PER_T, rel=1e-12)


def test_e7_46d52_44f72_stretched_rate():
    """E7 fixture transition Cs 46D5/2 -> 44F7/2 stretched:
    (mu_B/h)(8/7*7/2 - 6/5*5/2) = (4 - 3) mu_B/h = +mu_B/h."""
    rate = stretched_tuning_rate_hz_per_t(2, 2.5, 3, 3.5)
    assert rate == pytest.approx(+MU_B_OVER_H_HZ_PER_T, rel=1e-12)


# ---------------------------------------------------------------------------
# Tuning-law sign / symmetry checks (R-17 test plan)
# ---------------------------------------------------------------------------

def test_tuning_rate_antisymmetric_under_state_exchange():
    """df/dB(a->b) = -df/dB(b->a): the signed-frequency convention."""
    a = ZeemanState(2, 2.5, 2.5)
    b = ZeemanState(1, 1.5, 1.5)
    assert tuning_rate_hz_per_t(a, b) == pytest.approx(
        -tuning_rate_hz_per_t(b, a), rel=1e-14
    )


def test_tuning_rate_odd_under_m_reversal():
    """Zeeman shifts are odd in m_J: flipping both stretched manifolds
    (sign=+1 -> -1) flips the tuning-rate sign exactly."""
    up = stretched_tuning_rate_hz_per_t(2, 2.5, 1, 1.5, sign=+1)
    dn = stretched_tuning_rate_hz_per_t(2, 2.5, 1, 1.5, sign=-1)
    assert up == pytest.approx(-dn, rel=1e-14)


def test_shift_odd_and_linear_in_b():
    """Delta_f is strictly linear and odd in B (linear law by construction);
    slope equals tuning_rate_hz_per_t exactly."""
    a = ZeemanState(2, 2.5, 2.5)
    b = ZeemanState(1, 1.5, 1.5)
    b1 = 1e-3  # 10 G
    s1 = transition_shift_hz(a, b, b1)
    assert transition_shift_hz(a, b, 2 * b1) == pytest.approx(2 * s1, rel=1e-14)
    assert transition_shift_hz(a, b, -b1) == pytest.approx(-s1, rel=1e-14)
    assert s1 / b1 == pytest.approx(tuning_rate_hz_per_t(a, b), rel=1e-14)


def test_transition_shift_is_state_shift_difference():
    """Delta_f(transition) = Delta_f(to) - Delta_f(from), R-17 form."""
    a = ZeemanState(2, 2.5, 1.5)
    b = ZeemanState(3, 3.5, 2.5)
    bt = 5e-3
    assert transition_shift_hz(a, b, bt) == pytest.approx(
        state_shift_hz(b, bt) - state_shift_hz(a, bt), rel=1e-14
    )


def test_state_shift_vectorized_matches_scalar():
    st = ZeemanState(0, 0.5, 0.5)
    bt = np.array([0.0, 1e-4, 5e-4, -2e-3])
    vec = state_shift_hz(st, bt)
    assert vec.shape == bt.shape
    for b, v in zip(bt, vec):
        assert v == pytest.approx(state_shift_hz(st, float(b)), rel=1e-14, abs=1e-30)


def test_state_shift_worked_value_60g():
    """Cs nD5/2 stretched at 60 G: g_J m_J = 3 -> shift = 3 * 1.399625 MHz/G
    * 60 G = 251.93 MHz (reproducible arithmetic, no literature import)."""
    st = ZeemanState(2, 2.5, 2.5)
    shift = state_shift_hz(st, gauss_to_tesla(60.0))
    assert shift == pytest.approx(3 * 1.399625e6 * 60.0, rel=1e-5)


# ---------------------------------------------------------------------------
# Validity fence (R-17 scope guard) — the fence raises
# ---------------------------------------------------------------------------

def test_fence_raises_above_5pct_of_fs_interval():
    """R-17: |shift| > 5% of the fine-structure interval raises
    IntegrityError (quadratic Zeeman / j-mixing out of scope)."""
    st = ZeemanState(2, 2.5, 2.5)  # g_J m_J = 3
    fs = 1e9  # synthetic 1 GHz interval fixture
    # 5% of 1 GHz = 50 MHz; 3 * 1.3996 MHz/G * 12 G = 50.4 MHz > fence
    with pytest.raises(IntegrityError):
        state_shift_hz(st, gauss_to_tesla(12.0), fs_interval_hz=fs)


def test_fence_passes_just_below_threshold():
    st = ZeemanState(2, 2.5, 2.5)
    fs = 1e9
    # 3 * 1.3996 MHz/G * 11 G = 46.2 MHz < 50 MHz: allowed
    shift = state_shift_hz(st, gauss_to_tesla(11.0), fs_interval_hz=fs)
    assert abs(shift) < LINEAR_FENCE_FRACTION * fs


def test_fence_applies_to_transition_states_individually():
    """transition_shift_hz fences each state against its own interval, so a
    small *differential* shift cannot smuggle a large state shift through."""
    a = ZeemanState(2, 2.5, 2.5)  # g m = 3
    b = ZeemanState(3, 3.5, 3.5)  # g m = 4; difference only 1 * mu_B B / h
    bt = gauss_to_tesla(100.0)  # state shifts 420/560 MHz, differential 140 MHz
    with pytest.raises(IntegrityError):
        transition_shift_hz(a, b, bt, fs_interval_from_hz=1e9, fs_interval_to_hz=1e9)


def test_fence_vector_b_uses_worst_case():
    st = ZeemanState(2, 2.5, 2.5)
    bt = gauss_to_tesla(np.array([1.0, 5.0, 40.0]))  # 40 G entry breaks 1 GHz fence
    with pytest.raises(IntegrityError):
        state_shift_hz(st, bt, fs_interval_hz=1e9)


def test_fence_refuses_nonsense_interval():
    """Refuse-to-guess: a non-finite or non-positive FS interval is an
    IntegrityError, never a silently skipped check."""
    for bad in (0.0, -1e9, float("nan"), float("inf")):
        with pytest.raises(IntegrityError):
            require_linear_regime(1e3, bad)


def test_e7_conditions_exceed_linear_fence():
    """Documented scope boundary: at the E7 field scale (60 G, spec 09
    §3.5) a stretched Cs nD5/2 state shifts by ~252 MHz — beyond 5% of any
    plausible Rydberg nD fine-structure interval (< ~5 GHz), so the linear
    module refuses. E7.1's 1.17 GHz@60 G tuning therefore requires
    beyond-linear physics and stays with the validation corpus, not here."""
    st = ZeemanState(2, 2.5, 2.5)
    with pytest.raises(IntegrityError):
        state_shift_hz(st, gauss_to_tesla(60.0), fs_interval_hz=5e9)


# ---------------------------------------------------------------------------
# Diamagnetic channel — the term that actually breaks the linear law for
# Rydberg states (audit 2026-08-10: the fine-structure fence is not the
# binding condition, and for l = 0 it cannot be armed at all).
#
# Every ingredient below is checked against an INDEPENDENT computation:
# the angular factor against numerical integration of |Y_lm|^2, the radial
# factor against numerical integration of the exact hydrogen radial function,
# and the SI prefactor against the atomic-unit route through scipy's
# "atomic unit of mag. flux density".
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("l", [0, 1, 2, 3, 4])
def test_mean_cos2_theta_vs_numerical_integration(l):
    """<l m|cos^2 theta|l m> from the closed form equals the integral of
    |Y_lm(theta,phi)|^2 cos^2(theta) over the sphere, for every m.

    |Y_lm|^2 is built from the associated Legendre function directly
    (scipy.special.lpmv) and normalised by its own integral, so the check
    depends on nothing but the definition of Y_lm.
    """
    for m in range(-l, l + 1):
        def weight(th, l=l, m=m):
            return lpmv(abs(m), l, math.cos(th)) ** 2 * math.sin(th)
        norm, _ = quad(weight, 0.0, math.pi, epsabs=1e-14)
        num, _ = quad(lambda th: weight(th) * math.cos(th) ** 2,
                      0.0, math.pi, epsabs=1e-14)
        assert mean_cos2_theta(l, m) == pytest.approx(num / norm, rel=1e-10)
    # exact sum rule: sum_m <cos^2> = (2l+1)/3 (isotropy of a filled shell)
    total = sum(mean_cos2_theta(l, m) for m in range(-l, l + 1))
    assert total == pytest.approx((2 * l + 1) / 3, rel=1e-13)


def test_mean_sin2_theta_fine_structure_values():
    """<sin^2 theta> in |l j m_j>: l = 0 is isotropic (2/3); a stretched
    m_j = j = l+1/2 state is pure |m_l = l, m_s = +1/2>, so it must equal the
    pure-m_l value 1 - 1/(2l+3); and the (2j+1) states of a j-manifold must
    average to the isotropic 2/3 (closure over m_j)."""
    assert mean_sin2_theta(0, 0.5, 0.5) == pytest.approx(2 / 3, rel=1e-13)
    for l in (1, 2, 3):
        j = l + 0.5
        assert mean_sin2_theta(l, j, j) == pytest.approx(
            1.0 - 1.0 / (2 * l + 3), rel=1e-12)
    for (l, j) in [(1, 0.5), (1, 1.5), (2, 1.5), (2, 2.5), (3, 3.5)]:
        tj = round(2 * j)
        avg = sum(mean_sin2_theta(l, j, tm / 2)
                  for tm in range(-tj, tj + 1, 2)) / (tj + 1)
        assert avg == pytest.approx(2 / 3, rel=1e-12), (l, j)


@pytest.mark.parametrize("n,l", [(1, 0), (2, 1), (5, 2), (10, 3), (20, 0)])
def test_hydrogenic_r2_vs_numerical_integration(n, l):
    """<r^2> = (n^2/2)[5n^2 + 1 - 3l(l+1)] a0^2 checked by integrating the
    exact hydrogen radial function R_nl (associated Laguerre form).

    This is the check that matters: the n^2 prefactor (not n^4) is the
    difference between a 1.5 % and a 27x diamagnetic/linear ratio at 60 G.
    """
    r = np.linspace(1e-9, 60.0 * n * n, 2_000_001)
    rho = 2.0 * r / n
    R = np.exp(-rho / 2.0) * rho**l * genlaguerre(n - l - 1, 2 * l + 1)(rho)
    w = R * R * r * r
    numeric = float(np.trapezoid(w * r * r, r) / np.trapezoid(w, r))
    analytic = 0.5 * n**2 * (5 * n**2 + 1 - 3 * l * (l + 1))
    assert numeric == pytest.approx(analytic, rel=1e-9)
    # and mean_rho2_a0sq must be that radial value times the angular one
    j = l + 0.5
    assert mean_rho2_a0sq(n, l, j, j) == pytest.approx(
        analytic * mean_sin2_theta(l, j, j), rel=1e-12)


def test_diamagnetic_prefactor_vs_atomic_units():
    """e^2 a0^2/(8 m_e h) assembled from CODATA equals the atomic-unit route
    (1/8)(B/B_au)^2 <rho^2> E_h/h with B_au = scipy's atomic unit of magnetic
    flux density — a different constant chain reaching the same number."""
    b_au = sc.physical_constants["atomic unit of mag. flux density"][0]
    e_h_over_h = sc.physical_constants["Hartree energy"][0] / sc.h
    rho2 = mean_rho2_a0sq(42.5, 2, 2.5, 2.5)
    for b in (1e-4, 6e-3, 4.12e-2):
        si = DIAMAGNETIC_HZ_PER_T2_A0SQ * rho2 * b * b
        au = 0.125 * (b / b_au) ** 2 * rho2 * e_h_over_h
        assert si == pytest.approx(au, rel=1e-11)
        assert diamagnetic_shift_hz(42.5, 2, 2.5, 2.5, b) == pytest.approx(si, rel=1e-14)


def test_diamagnetic_scaling_laws():
    """The reason it is the binding term: quartic in n* (asymptotically) and
    quadratic in B, against a linear, n-independent Zeeman shift."""
    b = gauss_to_tesla(50.0)
    d1 = diamagnetic_shift_hz(40.0, 2, 2.5, 2.5, b)
    d2 = diamagnetic_shift_hz(80.0, 2, 2.5, 2.5, b)
    assert d2 / d1 == pytest.approx(16.0, rel=2e-3)     # ~n*^4
    assert diamagnetic_shift_hz(40.0, 2, 2.5, 2.5, 2 * b) / d1 == pytest.approx(
        4.0, rel=1e-12)                                  # exactly B^2
    assert diamagnetic_shift_hz(40.0, 2, 2.5, 2.5, -b) == pytest.approx(d1, rel=1e-14)


@pytest.mark.parametrize("n_star,l,j,gauss,expected_ratio", [
    (42.5, 2, 2.5, 60.0, 0.014843529356),   # Cs-like nD5/2 stretched, E7 field
    (42.5, 2, 2.5, 412.0, 0.101925568242),  # top of the E7 field range
    (44.9, 0, 0.5, 60.0, 0.043231962746),   # l = 0: FS fence unarmable here
    (44.9, 0, 0.5, 412.0, 0.296859477521),
])
def test_neglected_over_returned_ratio_is_10_to_30_percent(
        n_star, l, j, gauss, expected_ratio):
    """Reproducible statement of the defect: over the module's own E7 field
    range the NEGLECTED diamagnetic term is 1.5-30 % of the shift the module
    RETURNS, while a 5 GHz fine-structure interval trips the j-mixing fence
    only above ~60 G and not at all for l = 0."""
    st = ZeemanState(l, j, j)
    b = gauss_to_tesla(gauss)
    lin = abs(MU_B_OVER_H_HZ_PER_T * st.g_j * st.m_j * b)
    dia = diamagnetic_shift_hz(n_star, l, j, j, b)
    assert dia / lin == pytest.approx(expected_ratio, rel=1e-8)


def test_diamagnetic_fence_raises_where_the_fs_fence_passes():
    """The point of the fix: a state that sails through the fine-structure
    fence is refused by the diamagnetic one, because that is the term that
    actually breaks the linear law."""
    st = ZeemanState(2, 2.5, 2.5)
    b = gauss_to_tesla(300.0)
    # generous 100 GHz interval -> j-mixing fence passes comfortably
    state_shift_hz(st, b, fs_interval_hz=1e11)
    # ... but the neglected diamagnetic term is 7.4 % of the returned shift
    with pytest.raises(IntegrityError, match="diamagnetic"):
        state_shift_hz(st, b, n_star=42.5)


def test_diamagnetic_fence_is_armable_for_l_zero():
    """The gap the fine-structure fence structurally cannot cover: an S1/2
    state has no same-l j-partner, so channel 1 has no interval to test. The
    diamagnetic channel arms fine and refuses at 412 G (29.7 % neglected)."""
    st = ZeemanState(0, 0.5, 0.5)
    state_shift_hz(st, gauss_to_tesla(412.0))                    # unfenced: allowed
    with pytest.raises(IntegrityError, match="diamagnetic"):
        state_shift_hz(st, gauss_to_tesla(412.0), n_star=44.9)


def test_diamagnetic_fence_passes_in_the_regime_it_should():
    """Not an over-strict gate: at low field / low n the linear law is
    genuinely dominant and the fence must stay out of the way."""
    st = ZeemanState(2, 2.5, 2.5)
    for gauss in (0.0, 1.0, 10.0, 100.0):
        state_shift_hz(st, gauss_to_tesla(gauss), n_star=42.5)
    # a low-n (non-Rydberg) state is unaffected even at 412 G
    state_shift_hz(st, gauss_to_tesla(412.0), n_star=6.0)
    # ... and the crossing sits where the arithmetic says it does (202 G)
    st_ok = state_shift_hz(st, gauss_to_tesla(200.0), n_star=42.5)
    assert abs(st_ok) > 0.0
    with pytest.raises(IntegrityError):
        state_shift_hz(st, gauss_to_tesla(205.0), n_star=42.5)


def test_diamagnetic_fence_vector_b_uses_worst_case():
    st = ZeemanState(2, 2.5, 2.5)
    bt = gauss_to_tesla(np.array([1.0, 10.0, 400.0]))
    with pytest.raises(IntegrityError, match="diamagnetic"):
        state_shift_hz(st, bt, n_star=42.5)
    ok = state_shift_hz(st, gauss_to_tesla(np.array([1.0, 10.0, 100.0])), n_star=42.5)
    assert ok.shape == (3,)


def test_diamagnetic_fence_on_transitions_is_per_state():
    """The two levels have different l and n*, so their diamagnetic shifts do
    not cancel in the difference; each state is fenced against its own n*."""
    a = ZeemanState(2, 2.5, 2.5)   # nD5/2
    b = ZeemanState(1, 1.5, 1.5)   # n'P3/2
    bt = gauss_to_tesla(300.0)
    transition_shift_hz(a, b, bt)  # unfenced: allowed
    with pytest.raises(IntegrityError, match="diamagnetic"):
        transition_shift_hz(a, b, bt, n_star_from=42.5)
    with pytest.raises(IntegrityError, match="diamagnetic"):
        transition_shift_hz(a, b, bt, n_star_to=43.7)


def test_diamagnetic_refuses_unbound_n_star():
    """Refuse-to-guess: n* <= l is not a bound orbital; never return a number
    for a state that does not exist."""
    for bad in (2.0, 1.5, 0.0, -3.0, float("nan"), float("inf")):
        with pytest.raises(IntegrityError):
            mean_rho2_a0sq(bad, 2, 2.5, 2.5)
    with pytest.raises(IntegrityError):
        diamagnetic_shift_hz(1.9, 2, 2.5, 2.5, 1e-3)


def test_require_linear_dominates_direct():
    st = ZeemanState(2, 2.5, 2.5)
    require_linear_dominates(st, 42.5, gauss_to_tesla(100.0))
    with pytest.raises(IntegrityError):
        require_linear_dominates(st, 42.5, gauss_to_tesla(400.0))
    # tolerance is a parameter, and tightening it tightens the fence
    with pytest.raises(IntegrityError):
        require_linear_dominates(st, 42.5, gauss_to_tesla(100.0), max_fraction=0.01)
    assert DIAMAGNETIC_FENCE_FRACTION == LINEAR_FENCE_FRACTION == 0.05


# ---------------------------------------------------------------------------
# Input validation (no rounding, no forbidden transitions)
# ---------------------------------------------------------------------------

def test_invalid_quantum_numbers_raise():
    with pytest.raises(ValueError):
        lande_g_j(-1, 0.5)  # negative l
    with pytest.raises(ValueError):
        lande_g_j(1, 0.7)  # j not half-integer (no rounding, audit rule 12)
    with pytest.raises(ValueError):
        lande_g_j(2, 0.5)  # j violates triangle with l=2, s=1/2
    with pytest.raises(ValueError):
        lande_g_j(1, 1.0)  # wrong half-integer character for s=1/2
    with pytest.raises(ValueError):
        ZeemanState(2, 2.5, 3.5)  # |m_j| > j
    with pytest.raises(ValueError):
        ZeemanState(2, 2.5, 1.0)  # m_j parity mismatch


def test_dipole_forbidden_transitions_raise():
    s_half = ZeemanState(0, 0.5, 0.5)
    d52 = ZeemanState(2, 2.5, 0.5)
    with pytest.raises(ValueError):
        tuning_rate_hz_per_t(s_half, d52)  # Delta l = 2
    a = ZeemanState(1, 1.5, -1.5)
    b = ZeemanState(2, 2.5, 0.5)
    with pytest.raises(ValueError):
        tuning_rate_hz_per_t(a, b)  # Delta m_J = 2
    with pytest.raises(ValueError):
        stretched_pair(0, 0.5, 1, 1.5, sign=2)


def test_gauss_conversion_exact():
    assert GAUSS_TO_TESLA == 1e-4
    assert gauss_to_tesla(412.0) == pytest.approx(0.0412, rel=1e-15)
