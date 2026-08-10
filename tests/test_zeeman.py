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

from fractions import Fraction

import numpy as np
import pytest

from rydsim.constants import H, MU_B
from rydsim.provenance import IntegrityError
from rydsim.zeeman import (
    GAUSS_TO_TESLA,
    LINEAR_FENCE_FRACTION,
    MU_B_OVER_H_HZ_PER_T,
    ZeemanState,
    gauss_to_tesla,
    hz_per_t_to_mhz_per_gauss,
    lande_g_j,
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
