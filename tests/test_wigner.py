"""Validation: rydsim.wigner vs exact symbolic values (sympy) + sum rules.

Coverage note (audit 2026-08-10, MEDIUM "float path returns mathematically
impossible values"): the previous version of this file exercised only rank-1
(j2 = 1) symbols, generic j <= 4, and one degenerate (j, j, 0) case whose
Racah sum has a single term. That blind spot is exactly where the log-factorial
path failed catastrophically. The generic large-j section below is the
independent check that was missing: an exact-rational oracle (Fraction
arithmetic, cross-validated against sympy) over generic j up to 300.
"""

import itertools
import math
from fractions import Fraction

import pytest
from sympy import Rational, S
from sympy.physics.wigner import wigner_3j as sy3j, wigner_6j as sy6j
from sympy.physics.quantum.cg import CG

from rydsim.provenance import IntegrityError
from rydsim.wigner import (
    FLOAT_PATH_TARGET_REL,
    clebsch_gordan,
    exact_to_float,
    six_j_bound,
    three_j_bound,
    wigner_3j,
    wigner_3j_exact,
    wigner_3j_float,
    wigner_6j,
    wigner_6j_exact,
    wigner_6j_float,
)


def _r(x: float) -> Rational:
    return Rational(round(2 * x), 2)


def _exact3(*args) -> float:
    return exact_to_float(*wigner_3j_exact(*args))


def _exact6(*args) -> float:
    return exact_to_float(*wigner_6j_exact(*args))


def test_3j_grid_vs_sympy():
    """Exhaustive check over all valid (j,m) with j up to 4 (integer + half)."""
    js = [x / 2 for x in range(0, 9)]  # 0 .. 4 in half steps
    n_checked = 0
    for j1, j2 in itertools.product(js, repeat=2):
        j3min, j3max = abs(j1 - j2), j1 + j2
        j3 = j3min
        while j3 <= j3max + 1e-9:
            for m1 in [x / 2 for x in range(-round(2 * j1), round(2 * j1) + 1, 2)]:
                for m2 in [x / 2 for x in range(-round(2 * j2), round(2 * j2) + 1, 2)]:
                    m3 = -(m1 + m2)
                    if abs(m3) > j3 + 1e-9:
                        continue
                    ours = wigner_3j(j1, j2, j3, m1, m2, m3)
                    exact = float(sy3j(_r(j1), _r(j2), _r(j3), _r(m1), _r(m2), _r(m3)))
                    assert ours == pytest.approx(exact, abs=1e-12), (
                        f"3j({j1},{j2},{j3};{m1},{m2},{m3}) = {ours} != {exact}"
                    )
                    n_checked += 1
            j3 += 1.0
    assert n_checked > 500  # make sure the sweep actually ran


def test_3j_known_value():
    # (1 1 1; 1 -1 0) — a standard tabulated value: 1/sqrt(6)
    assert wigner_3j(1, 1, 1, 1, -1, 0) == pytest.approx(1 / math.sqrt(6), rel=1e-14)


def test_3j_large_j_stability():
    """Log-space evaluation must survive j ~ 100 (Rydberg-Rydberg regime)."""
    ours = wigner_3j(100, 100, 0, 50, -50, 0)
    # (j j 0; m -m 0) = (-1)^(j-m)/sqrt(2j+1) exactly
    exact = (-1) ** (100 - 50) / math.sqrt(201)
    assert ours == pytest.approx(exact, rel=1e-10)


def test_6j_grid_vs_sympy():
    """Check all valid 6j with entries up to 3 in half steps."""
    js = [x / 2 for x in range(0, 7)]  # 0 .. 3
    n_checked = 0
    for j1, j2, j3, j4, j5, j6 in itertools.product(js, repeat=6):
        ours = wigner_6j(j1, j2, j3, j4, j5, j6)
        try:
            exact = float(sy6j(_r(j1), _r(j2), _r(j3), _r(j4), _r(j5), _r(j6)))
        except ValueError:
            # sympy raises on violated triangle rules; ours returns 0
            assert ours == 0.0
            continue
        assert ours == pytest.approx(exact, abs=1e-12), (
            f"6j({j1},{j2},{j3};{j4},{j5},{j6}) = {ours} != {exact}"
        )
        n_checked += 1
    assert n_checked > 100


def test_6j_physical_case():
    """The 6j entering alkali fine-structure dipole reductions (l s j basis)."""
    # {1/2 1 3/2 / 2 5/2 1} appears in D2 nP3/2 -> nD5/2 reductions
    ours = wigner_6j(0.5, 1, 1.5, 2, 2.5, 1)
    exact = float(sy6j(S(1) / 2, 1, S(3) / 2, 2, S(5) / 2, 1))
    assert ours == pytest.approx(exact, abs=1e-14)


def test_cg_vs_sympy():
    for (j1, m1, j2, m2, j3, m3) in [
        (0.5, 0.5, 0.5, -0.5, 1, 0),
        (0.5, 0.5, 0.5, -0.5, 0, 0),
        (1, 0, 1, 0, 2, 0),
        (1.5, 0.5, 1, -1, 0.5, -0.5),
        (2, 1, 1.5, 0.5, 2.5, 1.5),
    ]:
        ours = clebsch_gordan(j1, m1, j2, m2, j3, m3)
        exact = float(CG(_r(j1), _r(m1), _r(j2), _r(m2), _r(j3), _r(m3)).doit())
        assert ours == pytest.approx(exact, abs=1e-13), (j1, m1, j2, m2, j3, m3)


def test_3j_orthogonality_sum_rule():
    """sum_{m1 m2} (2j3+1) 3j(...m3)^2 = 1 for each valid (j3, m3)."""
    j1, j2 = 2, 1.5
    j3 = 2.5
    for m3 in [-2.5, -0.5, 1.5]:
        total = 0.0
        for m1 in [-2, -1, 0, 1, 2]:
            for m2 in [-1.5, -0.5, 0.5, 1.5]:
                if abs(m1 + m2 + m3) < 1e-9:
                    total += (2 * j3 + 1) * wigner_3j(j1, j2, j3, m1, m2, m3) ** 2
        assert total == pytest.approx(1.0, rel=1e-12)


def test_selection_rules_zero():
    assert wigner_3j(1, 1, 3, 0, 0, 0) == 0.0          # triangle violated
    assert wigner_3j(1, 1, 1, 1, 1, 1) == 0.0          # m-sum nonzero
    assert wigner_3j(1, 1, 1, 0, 0, 0) == 0.0          # odd-perimeter zero (j1+j2+j3 odd rule)
    assert wigner_6j(1, 1, 3, 1, 1, 1) == 0.0          # triangle violated


def test_half_integer_validation():
    with pytest.raises(ValueError):
        wigner_3j(0.3, 1, 1, 0, 0, 0)


# ---------------------------------------------------------------------------
# The exact-rational oracle itself (audit refusal #13's "route"), validated
# against sympy — an independent exact implementation, not our own output.
# ---------------------------------------------------------------------------

_ORACLE_3J = [
    (0.5, 1, 1.5, 0.5, 0, -0.5),
    (2, 1, 1, 0, 0, 0),
    (3, 2, 4, 1, -2, 1),
    (2.5, 1.5, 2, -1.5, 0.5, 1),
    (10, 9, 7, 3, -2, -1),
    (30, 25, 20, 4, -4, 0),
    (100, 90, 80, 0, 0, 0),
    (100, 90, 80, 13, -7, -6),
    (150, 150, 150, 0, 0, 0),
    (75.5, 60.5, 40, -10.5, 12.5, -2),
]

_ORACLE_6J = [
    (1, 1, 1, 1, 1, 1),
    (2, 3, 1, 3.5, 2.5, 0.5),
    (1.5, 1.5, 1, 1.5, 1.5, 2),
    (10, 11, 12, 13, 14, 15),
    (40, 35, 30, 25, 20, 22),
    (100, 100, 100, 100, 100, 100),
    (60.5, 60.5, 40, 70.5, 70.5, 35),
]


@pytest.mark.parametrize("args", _ORACLE_3J, ids=[str(a[0]) for a in _ORACLE_3J])
def test_exact_3j_oracle_vs_sympy(args):
    """The Fraction-arithmetic 3j oracle reproduces sympy's exact value."""
    exact = float(sy3j(*[_r(x) for x in args]))
    assert _exact3(*args) == pytest.approx(exact, rel=1e-14, abs=1e-300)


@pytest.mark.parametrize("args", _ORACLE_6J, ids=[str(a[0]) for a in _ORACLE_6J])
def test_exact_6j_oracle_vs_sympy(args):
    """The Fraction-arithmetic 6j oracle reproduces sympy's exact value."""
    exact = float(sy6j(*[_r(x) for x in args]))
    assert _exact6(*args) == pytest.approx(exact, rel=1e-14, abs=1e-300)


def test_exact_oracle_returns_exact_rational_square():
    """The oracle's second element is an exact Fraction, not a float — that
    is what makes it usable as ground truth at any j."""
    sign, square = wigner_3j_exact(1, 1, 1, 1, -1, 0)
    assert isinstance(square, Fraction)
    assert square == Fraction(1, 6) and sign == 1


# ---------------------------------------------------------------------------
# The defect this file previously could not see: generic (non-rank-1) symbols
# at large j. Independent oracle, and the elementary |3j| bound.
# ---------------------------------------------------------------------------

def test_generic_large_j_3j_matches_exact_oracle():
    """Generic large-j 3j: the shipped symbol must equal the exact value.

    (100, 90, 80; 0, 0, 0) is the audit's headline case: the bare
    log-factorial sum returns +2.52 where the exact value is -6.783e-3 —
    wrong sign, 372x too large, and 36x outside the hard bound 1/sqrt(201).
    """
    args = (100, 90, 80, 0, 0, 0)
    exact = float(sy3j(*[_r(x) for x in args]))
    assert exact == pytest.approx(-6.783314088539e-03, rel=1e-10)
    assert wigner_3j(*args) == pytest.approx(exact, rel=FLOAT_PATH_TARGET_REL)
    # ... and the raw float path is still as broken as the audit measured,
    # so the fix is the routing, not a silent change of the kernel.
    raw, est = wigner_3j_float(*args)
    assert abs(raw) > 1.0                      # mathematically impossible
    assert est > 1.0                           # and the estimator says so


def test_generic_large_j_3j_beyond_the_j150_ceiling():
    """j = 300 generic: was -1.23e+40 (no refusal, twice the declared 150
    ceiling); must now be the exact +2.0175e-3."""
    args = (300, 300, 300, 0, 0, 0)
    exact = float(sy3j(*[_r(x) for x in args]))
    assert wigner_3j(*args) == pytest.approx(exact, rel=FLOAT_PATH_TARGET_REL)
    assert wigner_3j(*args) == pytest.approx(2.017505855571e-03, rel=1e-10)
    raw, est = wigner_3j_float(*args)
    assert abs(raw) > 1e30 and est > 1.0


def test_generic_large_j_6j_matches_exact_oracle():
    """{100 100 100; 100 100 100} was +8.4962e-2 vs exact -4.6984e-4."""
    args = (100, 100, 100, 100, 100, 100)
    exact = float(sy6j(*[_r(x) for x in args]))
    assert wigner_6j(*args) == pytest.approx(exact, rel=FLOAT_PATH_TARGET_REL)
    assert wigner_6j(*args) == pytest.approx(-4.698416232987e-04, rel=1e-10)
    raw, est = wigner_6j_float(*args)
    assert abs(raw) > 1e-2 and est > 1.0


def test_generic_3j_grid_vs_exact_oracle_to_j150():
    """Stratified generic-3j grid to j = 150 against the exact oracle.

    The old suite capped generic 3j at j <= 20 (test_angular) / j <= 4 (here)
    and never saw the cancellation. This grid deliberately includes the
    all-zero-m diagonal and near-equal (j1, j2, j3), where the alternating
    sum cancels hardest.
    """
    checked = 0
    worst = 0.0
    for j1 in (12, 33, 60, 101, 150):
        for j2 in (j1 // 3, j1 // 2, j1 - 1, j1):
            if j2 < 1:
                continue
            for j3 in (abs(j1 - j2), (j1 + j2) // 2, j1 + j2 - 1):
                if j3 < abs(j1 - j2) or j3 > j1 + j2 or (j1 + j2 + j3) % 2:
                    continue
                for (m1, m2) in ((0, 0), (1, -1), (j1 // 2, -j2 // 2), (j1, -j2)):
                    m3 = -(m1 + m2)
                    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
                        continue
                    args = (j1, j2, j3, m1, m2, m3)
                    ours = wigner_3j(*args)
                    exact = _exact3(*args)
                    if exact == 0.0:
                        assert ours == 0.0, args
                    else:
                        rel = abs(ours - exact) / abs(exact)
                        worst = max(worst, rel)
                        assert rel <= FLOAT_PATH_TARGET_REL, (args, ours, exact)
                    checked += 1
    assert checked >= 100, checked
    print(f"generic 3j grid: {checked} symbols, worst rel err {worst:.3e}")


def test_generic_6j_grid_vs_exact_oracle():
    """Stratified generic-6j grid against the exact oracle."""
    checked = 0
    worst = 0.0
    for a in (7, 20, 51, 100):
        for d in (a - 2, a, a + 3):
            for shift in (0, 5, a // 2):
                args = (a, a, a, d, d, d - shift if d - shift > 0 else d)
                ours = wigner_6j(*args)
                exact = _exact6(*args)
                if exact == 0.0:
                    assert ours == 0.0, args
                else:
                    rel = abs(ours - exact) / abs(exact)
                    worst = max(worst, rel)
                    assert rel <= FLOAT_PATH_TARGET_REL, (args, ours, exact)
                checked += 1
    assert checked >= 20, checked
    print(f"generic 6j grid: {checked} symbols, worst rel err {worst:.3e}")


def test_rank1_worst_case_from_the_audit_sweep():
    """The exact symbol the audit's exhaustive rank-1 sweep found worst.

    ``3j(141.5, 1, 141.5; -0.5, 0, 0.5)``: the bare float sum returns
    -0.00020894172804314623 against sympy's 30-digit
    -0.000208941728049253125, a 2.9e-11 relative error — 7x the 4.2e-12 the
    module docstring used to claim. The estimator flags it (1.2e-9), the
    exact route runs, and the shipped value is now exact to 1 ulp. The claim
    is thus a construction bound, not a transcribed measurement.
    """
    args = (141.5, 1, 141.5, -0.5, 0, 0.5)
    exact = _exact3(*args)
    assert exact == pytest.approx(float(sy3j(*[_r(x) for x in args])), rel=1e-15)
    raw, est = wigner_3j_float(*args)
    assert abs(raw - exact) / abs(exact) == pytest.approx(2.9e-11, rel=0.1)
    assert est > FLOAT_PATH_TARGET_REL          # the estimator catches it
    assert wigner_3j(*args) == exact            # exact route -> bit-identical


def test_rank1_sweep_vs_exact_oracle():
    """Stratified rank-1 (j2 = 1) sweep to j = 150, including the Delta j = 0,
    |m| = 1/2 corner where the audit measured the worst float error (2.9e-11
    at j1 = 141.5, against a docstring claim of 4.2e-12). The routed value
    must now be within 1e-12 of the exact oracle everywhere.
    """
    worst = 0.0
    worst_at = None
    checked = 0
    for t1 in list(range(0, 302, 7)) + [283, 300, 301]:
        for t3 in (t1 - 2, t1, t1 + 2):
            if t3 < 0 or not (abs(t1 - 2) <= t3 <= t1 + 2):
                continue
            for u1 in ({0, 1, t1} if t1 else {0}) | {-1 if t1 else 0}:
                if abs(u1) > t1 or (t1 - u1) % 2:
                    continue
                for q in (-1, 0, 1):
                    u3 = -u1 - 2 * q
                    if abs(u3) > t3 or (t3 - u3) % 2:
                        continue
                    args = (t1 / 2, 1.0, t3 / 2, u1 / 2, float(q), u3 / 2)
                    ours = wigner_3j(*args)
                    exact = _exact3(*args)
                    if exact == 0.0:
                        assert ours == 0.0, args
                    else:
                        rel = abs(ours - exact) / abs(exact)
                        if rel > worst:
                            worst, worst_at = rel, args
                    checked += 1
    assert checked >= 200, checked
    assert worst <= FLOAT_PATH_TARGET_REL, (worst, worst_at)
    print(f"rank-1 sweep: {checked} symbols, worst rel err {worst:.3e} at {worst_at}")


# ---------------------------------------------------------------------------
# The universal sanity gate: no value outside the mathematically possible range
# ---------------------------------------------------------------------------

def test_three_j_bound_holds_and_is_tight():
    """|3j| <= 1/sqrt(2 j_max + 1) on every symbol we compute, and the bound
    is attained (so it is not vacuously loose): (j j 0; m -m 0) saturates it."""
    for j in (0.5, 1, 2.5, 7, 50, 100):
        b = three_j_bound(j, j, 0)
        assert abs(wigner_3j(j, j, 0, j, -j, 0)) == pytest.approx(b, rel=1e-12)
    for args in _ORACLE_3J:
        assert abs(wigner_3j(*args)) <= three_j_bound(*args[:3]) * (1 + 1e-12)


def test_six_j_bound_holds():
    for args in _ORACLE_6J:
        assert abs(wigner_6j(*args)) <= six_j_bound(*args) * (1 + 1e-12)


def test_impossible_value_is_refused_not_returned():
    """A value outside the elementary bound must raise IntegrityError, never
    be returned. Simulated by forcing the broken float path back in."""
    import rydsim.wigner as W

    saved = W.FLOAT_PATH_TARGET_REL
    W._w3j.cache_clear()
    try:
        W.FLOAT_PATH_TARGET_REL = 1e300      # disable the exact-rational route
        with pytest.raises(IntegrityError, match="elementary bound"):
            W.wigner_3j(100, 90, 80, 0, 0, 0)
    finally:
        W.FLOAT_PATH_TARGET_REL = saved
        W._w3j.cache_clear()
    # with the route restored the same call returns the exact value
    assert W.wigner_3j(100, 90, 80, 0, 0, 0) == pytest.approx(
        -6.783314088539e-03, rel=1e-10)
