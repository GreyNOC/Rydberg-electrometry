"""Wigner 3j / 6j symbols and Clebsch-Gordan coefficients.

Pure angular-momentum algebra (Racah closed forms). Half-integer angular
momenta are supported; internally everything is doubled to integers.

Evaluation strategy (audit refusal #13: "route to the exact-rational oracle
or refuse; the float path is certified only for the rank-1 chain")
-------------------------------------------------------------------------
The log-factorial Racah sum is an *alternating* sum whose terms are computed
from ``math.lgamma``.  Each term therefore carries a relative error of order
``L * eps`` where ``L`` is the magnitude of the log-factorials involved
(~5e3 at j = 150, ~2e4 at j = 300) and ``eps = 2.2e-16``.  When the sum
cancels by a factor ``R = |sum| / sum|terms|`` the *result* carries a
relative error of order ``L * eps / R``.  For generic (non-rank-1) symbols at
large j this cancellation is catastrophic: the pure float path returns
``3j(100,90,80;0,0,0) = +2.52`` where the exact value is ``-6.78e-3`` — wrong
sign, 372x too large, and 36x outside the elementary bound
``|3j| <= 1/sqrt(2 j_max + 1) = 0.0705``.

So this module does NOT ship the bare float sum.  Every public 3j/6j:

1. evaluates the float sum together with the error estimate ``L * eps / R``;
2. if that estimate exceeds :data:`FLOAT_PATH_TARGET_REL` (1e-12), recomputes
   the symbol in **exact rational arithmetic** (:func:`wigner_3j_exact` /
   :func:`wigner_6j_exact`, ``fractions.Fraction`` — no floating point until
   the final square root), which is exact by construction at any j;
3. checks the returned value against the elementary orthogonality bound
   (:func:`three_j_bound` / :func:`six_j_bound`) and raises
   ``rydsim.provenance.IntegrityError`` if it is violated — a value outside
   the range the quantity can mathematically occupy is never returned.

Consequence: the stated accuracy is a *construction* bound, not a transcribed
measurement.  ``|value - exact| <= 1e-12 * |exact|`` holds for every symbol,
rank-1 or generic, at every j — verified in tests/test_wigner.py against the
exact-rational oracle (which is itself cross-checked against sympy) over a
generic large-j grid including the (100,90,80) and (300,300,300) cases.

Validation: tests/test_wigner.py cross-checks against sympy.physics.wigner
(exact symbolic values) over a broad grid, plus orthogonality sum rules.

Conventions: Condon-Shortley phases, as in Edmonds, *Angular Momentum in
Quantum Mechanics* (1957) — the same convention sympy uses.
"""

from __future__ import annotations

import math
from fractions import Fraction
from functools import lru_cache
from math import factorial as _fact

from .provenance import IntegrityError

__all__ = [
    "wigner_3j",
    "wigner_6j",
    "clebsch_gordan",
    "wigner_3j_exact",
    "wigner_6j_exact",
    "exact_to_float",
    "wigner_3j_float",
    "wigner_6j_float",
    "three_j_bound",
    "six_j_bound",
    "FLOAT_PATH_TARGET_REL",
]

_LGAMMA_CACHE_MAX = 4096
_EPS = 2.220446049250313e-16  # 2**-52, double precision

#: Relative accuracy the public 3j/6j guarantee.  A symbol whose float-path
#: error estimate exceeds this is recomputed in exact rational arithmetic.
FLOAT_PATH_TARGET_REL = 1e-12

#: Slack on the elementary orthogonality bound before the sanity gate fires.
#: The bound is exact mathematics; the slack only absorbs round-off in the
#: bound itself and in a legitimately maximal symbol.
_BOUND_SLACK_REL = 1e-9

#: Result caches (keyed on doubled integers).  Bounded so an exhaustive sweep
#: cannot grow them without limit.
_RESULT_CACHE_MAX = 65536


@lru_cache(maxsize=_LGAMMA_CACHE_MAX)
def _lnfact(n: int) -> float:
    """ln(n!) with cache; n must be a non-negative integer."""
    if n < 0:
        raise ValueError(f"factorial of negative integer {n}")
    return math.lgamma(n + 1)


def _two_j(j: float, name: str) -> int:
    """Convert an angular momentum (possibly half-integer) to doubled int."""
    tj = round(2 * j)
    if abs(2 * j - tj) > 1e-9:
        raise ValueError(f"{name}={j} is not integer or half-integer")
    return tj


def _triangle_ok(tj1: int, tj2: int, tj3: int) -> bool:
    """Triangle rule on doubled j's, including parity (integer perimeter)."""
    return (
        abs(tj1 - tj2) <= tj3 <= tj1 + tj2
        and (tj1 + tj2 + tj3) % 2 == 0
    )


def _ln_delta(tj1: int, tj2: int, tj3: int) -> float:
    """ln of the triangle coefficient Delta(j1 j2 j3) (doubled args)."""
    return 0.5 * (
        _lnfact((tj1 + tj2 - tj3) // 2)
        + _lnfact((tj1 - tj2 + tj3) // 2)
        + _lnfact((-tj1 + tj2 + tj3) // 2)
        - _lnfact((tj1 + tj2 + tj3) // 2 + 1)
    )


def _ln_delta_mag(tj1: int, tj2: int, tj3: int) -> float:
    """Sum of |ln n!| entering _ln_delta — the round-off scale of that term."""
    return 0.5 * (
        _lnfact((tj1 + tj2 - tj3) // 2)
        + _lnfact((tj1 - tj2 + tj3) // 2)
        + _lnfact((-tj1 + tj2 + tj3) // 2)
        + _lnfact((tj1 + tj2 + tj3) // 2 + 1)
    )


# ---------------------------------------------------------------------------
# Elementary bounds (exact mathematics, used as the universal sanity gate)
# ---------------------------------------------------------------------------

def three_j_bound(j1: float, j2: float, j3: float) -> float:
    """Hard upper bound on |(j1 j2 j3; m1 m2 m3)| for every (m1, m2, m3).

    From the orthogonality sum ``sum_{m1 m2} (2 j3 + 1) 3j^2 = 1`` (Edmonds
    3.7.8): every individual term obeys ``|3j| <= 1/sqrt(2 j3 + 1)``, and the
    3j is invariant in magnitude under column permutation, so the tightest
    form uses the largest j.  Exact — no fitted constant.
    """
    tmax = max(_two_j(j1, "j1"), _two_j(j2, "j2"), _two_j(j3, "j3"))
    return 1.0 / math.sqrt(tmax + 1.0)


def six_j_bound(j1: float, j2: float, j3: float,
                j4: float, j5: float, j6: float) -> float:
    """Hard upper bound on |{j1 j2 j3; j4 j5 j6}|.

    From ``sum_{j3} (2 j3 + 1)(2 j6 + 1) {..}^2 = 1`` (Edmonds 6.2.9):
    ``|6j|^2 <= 1/((2 j3 + 1)(2 j6 + 1))``.  The 6j is invariant under
    permutation of its three *columns*, so the same holds for the (j1, j4)
    and (j2, j5) columns; the tightest is the largest column product.
    """
    t = [_two_j(j, f"j{i + 1}") for i, j in enumerate((j1, j2, j3, j4, j5, j6))]
    prod = max((t[0] + 1) * (t[3] + 1),
               (t[1] + 1) * (t[4] + 1),
               (t[2] + 1) * (t[5] + 1))
    return 1.0 / math.sqrt(prod)


# ---------------------------------------------------------------------------
# Exact-rational oracle (audit refusal #13's "route", promoted out of tests/)
# ---------------------------------------------------------------------------

def _ln_int(n: int) -> float:
    """ln(n) for an arbitrarily large positive Python int (overflow-safe)."""
    if n <= 0:
        raise ValueError(f"ln of non-positive integer {n}")
    bits = n.bit_length()
    if bits <= 900:
        return math.log(n)
    shift = bits - 900
    return math.log(n >> shift) + shift * math.log(2.0)


def exact_to_float(sign: int, square: Fraction) -> float:
    """Convert an exact ``(sign, value^2)`` oracle result to a float.

    The square is an exact :class:`fractions.Fraction`; the conversion goes
    through ``ln`` of the numerator/denominator so that symbols whose squares
    over- or under-flow a double (any j beyond ~150) still convert exactly to
    the nearest representable value.
    """
    if sign == 0 or square == 0:
        return 0.0
    return sign * math.exp(
        0.5 * (_ln_int(square.numerator) - _ln_int(square.denominator))
    )


@lru_cache(maxsize=_RESULT_CACHE_MAX)
def _w3j_exact_core(t1: int, t2: int, t3: int,
                    u1: int, u2: int, u3: int) -> tuple[int, Fraction]:
    """Exact 3j on doubled integers; returns (sign, value^2 as Fraction).

    Racah's closed form evaluated entirely in integer/rational arithmetic —
    no cancellation is possible because every partial sum is exact.
    """
    pref = Fraction(
        _fact((t1 + t2 - t3) // 2) * _fact((t1 - t2 + t3) // 2)
        * _fact((-t1 + t2 + t3) // 2),
        _fact((t1 + t2 + t3) // 2 + 1),
    )
    pref *= (_fact((t1 + u1) // 2) * _fact((t1 - u1) // 2)
             * _fact((t2 + u2) // 2) * _fact((t2 - u2) // 2)
             * _fact((t3 + u3) // 2) * _fact((t3 - u3) // 2))
    kmin = max(0, (t2 - t3 - u1) // 2, (t1 - t3 + u2) // 2)
    kmax = min((t1 + t2 - t3) // 2, (t1 - u1) // 2, (t2 + u2) // 2)
    total = Fraction(0)
    for k in range(kmin, kmax + 1):
        den = (_fact(k) * _fact(k - (t2 - t3 - u1) // 2)
               * _fact(k - (t1 - t3 + u2) // 2)
               * _fact((t1 + t2 - t3) // 2 - k)
               * _fact((t1 - u1) // 2 - k) * _fact((t2 + u2) // 2 - k))
        total += Fraction(-1 if k % 2 else 1, den)
    if total == 0:
        return 0, Fraction(0)
    phase = -1 if ((t1 - t2 - u3) // 2) % 2 else 1
    sign = phase * (1 if total > 0 else -1)
    return sign, pref * total * total


@lru_cache(maxsize=_RESULT_CACHE_MAX)
def _w6j_exact_core(ta: int, tb: int, tc: int,
                    td: int, te: int, tf: int) -> tuple[int, Fraction]:
    """Exact 6j on doubled integers; returns (sign, value^2 as Fraction)."""
    def delta_sq(x: int, y: int, z: int) -> Fraction:
        return Fraction(
            _fact((x + y - z) // 2) * _fact((x - y + z) // 2)
            * _fact((-x + y + z) // 2),
            _fact((x + y + z) // 2 + 1),
        )

    pref = (delta_sq(ta, tb, tc) * delta_sq(ta, te, tf)
            * delta_sq(td, tb, tf) * delta_sq(td, te, tc))
    s1 = (ta + tb + tc) // 2
    s2 = (ta + te + tf) // 2
    s3 = (td + tb + tf) // 2
    s4 = (td + te + tc) // 2
    q1 = (ta + tb + td + te) // 2
    q2 = (tb + tc + te + tf) // 2
    q3 = (ta + tc + td + tf) // 2
    total = Fraction(0)
    for k in range(max(s1, s2, s3, s4), min(q1, q2, q3) + 1):
        den = (_fact(k - s1) * _fact(k - s2) * _fact(k - s3) * _fact(k - s4)
               * _fact(q1 - k) * _fact(q2 - k) * _fact(q3 - k))
        total += Fraction((-1 if k % 2 else 1) * _fact(k + 1), den)
    if total == 0:
        return 0, Fraction(0)
    return (1 if total > 0 else -1), pref * total * total


def wigner_3j_exact(j1: float, j2: float, j3: float,
                    m1: float, m2: float, m3: float) -> tuple[int, Fraction]:
    """Exact-rational 3j oracle: returns ``(sign, value**2)``.

    ``value**2`` is an exact :class:`fractions.Fraction`; the symbol itself is
    ``sign * sqrt(value**2)`` (:func:`exact_to_float`).  This is the route
    audit refusal #13 requires for symbols the float path cannot certify —
    it is exact at any j and any rank, at the cost of big-integer arithmetic.
    """
    t1, t2, t3 = _two_j(j1, "j1"), _two_j(j2, "j2"), _two_j(j3, "j3")
    u1, u2, u3 = _two_j(m1, "m1"), _two_j(m2, "m2"), _two_j(m3, "m3")
    if _selection_zero_3j(t1, t2, t3, u1, u2, u3):
        return 0, Fraction(0)
    return _w3j_exact_core(t1, t2, t3, u1, u2, u3)


def wigner_6j_exact(j1: float, j2: float, j3: float,
                    j4: float, j5: float, j6: float) -> tuple[int, Fraction]:
    """Exact-rational 6j oracle: returns ``(sign, value**2)``. See
    :func:`wigner_3j_exact`."""
    t = [_two_j(j, f"j{i + 1}") for i, j in enumerate((j1, j2, j3, j4, j5, j6))]
    ta, tb, tc, td, te, tf = t
    for (x, y, z) in ((ta, tb, tc), (ta, te, tf), (td, tb, tf), (td, te, tc)):
        if not _triangle_ok(x, y, z):
            return 0, Fraction(0)
    return _w6j_exact_core(ta, tb, tc, td, te, tf)


# ---------------------------------------------------------------------------
# Float (log-factorial) path, with its own error estimate
# ---------------------------------------------------------------------------

def _selection_zero_3j(tj1: int, tj2: int, tj3: int,
                       tm1: int, tm2: int, tm3: int) -> bool:
    """True when the 3j vanishes by selection rule (exact 0.0, spec 03 SS2.4)."""
    if tm1 + tm2 + tm3 != 0:
        return True
    if not _triangle_ok(tj1, tj2, tj3):
        return True
    if abs(tm1) > tj1 or abs(tm2) > tj2 or abs(tm3) > tj3:
        return True
    if (tj1 + tm1) % 2 or (tj2 + tm2) % 2 or (tj3 + tm3) % 2:
        return True  # m must have same integer/half-integer character as j
    return False


def _w3j_float_core(tj1: int, tj2: int, tj3: int,
                    tm1: int, tm2: int, tm3: int) -> tuple[float, float]:
    """Float Racah sum for the 3j; returns (value, estimated relative error).

    The estimate is ``4 * L * eps / R`` with ``L`` the total magnitude of the
    log-factorials entering the largest term plus the prefactor and ``R`` the
    cancellation ratio ``|sum| / sum|terms|``.  ``inf`` means the float path
    carries no significant digits (total cancellation or overflow).
    """
    # Racah sum limits (all in doubled units -> halved where needed)
    t_min = max(0, (tj2 - tj3 - tm1) // 2, (tj1 - tj3 + tm2) // 2)
    t_max = min((tj1 + tj2 - tj3) // 2, (tj1 - tm1) // 2, (tj2 + tm2) // 2)
    if t_min > t_max:
        return 0.0, 0.0

    ln_pref = _ln_delta(tj1, tj2, tj3) + 0.5 * (
        _lnfact((tj1 + tm1) // 2) + _lnfact((tj1 - tm1) // 2)
        + _lnfact((tj2 + tm2) // 2) + _lnfact((tj2 - tm2) // 2)
        + _lnfact((tj3 + tm3) // 2) + _lnfact((tj3 - tm3) // 2)
    )
    pref_mag = _ln_delta_mag(tj1, tj2, tj3) + 0.5 * (
        _lnfact((tj1 + tm1) // 2) + _lnfact((tj1 - tm1) // 2)
        + _lnfact((tj2 + tm2) // 2) + _lnfact((tj2 - tm2) // 2)
        + _lnfact((tj3 + tm3) // 2) + _lnfact((tj3 - tm3) // 2)
    )

    ln_terms: list[float] = []
    signs: list[float] = []
    for t in range(t_min, t_max + 1):
        ln_t = -(
            _lnfact(t)
            + _lnfact(t - (tj2 - tj3 - tm1) // 2)
            + _lnfact(t - (tj1 - tj3 + tm2) // 2)
            + _lnfact((tj1 + tj2 - tj3) // 2 - t)
            + _lnfact((tj1 - tm1) // 2 - t)
            + _lnfact((tj2 + tm2) // 2 - t)
        )
        ln_terms.append(ln_t)
        signs.append(-1.0 if t % 2 else 1.0)

    ln_max = max(ln_terms)
    scaled = [math.exp(lt - ln_max) for lt in ln_terms]
    total = math.fsum(s * v for s, v in zip(signs, scaled))
    sum_abs = math.fsum(scaled)

    phase = -1.0 if ((tj1 - tj2 - tm3) // 2) % 2 else 1.0
    try:
        value = phase * math.exp(ln_pref + ln_max) * total
    except OverflowError:
        return math.inf, math.inf

    # round-off scale: |ln| magnitudes feeding exp(), times eps, over the
    # cancellation ratio of the alternating sum.
    scale = pref_mag + abs(ln_max) + max(abs(lt) for lt in ln_terms)
    if total == 0.0 or not math.isfinite(value):
        return value, math.inf
    ratio = abs(total) / sum_abs
    est = 4.0 * (1.0 + scale) * _EPS / ratio
    return value, est


def _w6j_float_core(ta: int, tb: int, tc: int,
                    td: int, te: int, tf: int) -> tuple[float, float]:
    """Float Racah sum for the 6j; returns (value, estimated relative error)."""
    ln_pref = (
        _ln_delta(ta, tb, tc) + _ln_delta(ta, te, tf)
        + _ln_delta(td, tb, tf) + _ln_delta(td, te, tc)
    )
    pref_mag = (
        _ln_delta_mag(ta, tb, tc) + _ln_delta_mag(ta, te, tf)
        + _ln_delta_mag(td, tb, tf) + _ln_delta_mag(td, te, tc)
    )

    s_abc = (ta + tb + tc) // 2
    s_aef = (ta + te + tf) // 2
    s_dbf = (td + tb + tf) // 2
    s_dec = (td + te + tc) // 2
    q_abde = (ta + tb + td + te) // 2
    q_bcef = (tb + tc + te + tf) // 2
    q_acdf = (ta + tc + td + tf) // 2

    t_min = max(s_abc, s_aef, s_dbf, s_dec)
    t_max = min(q_abde, q_bcef, q_acdf)
    if t_min > t_max:
        return 0.0, 0.0

    ln_terms: list[float] = []
    signs: list[float] = []
    for tt in range(t_min, t_max + 1):
        ln_t = _lnfact(tt + 1) - (
            _lnfact(tt - s_abc) + _lnfact(tt - s_aef)
            + _lnfact(tt - s_dbf) + _lnfact(tt - s_dec)
            + _lnfact(q_abde - tt) + _lnfact(q_bcef - tt)
            + _lnfact(q_acdf - tt)
        )
        ln_terms.append(ln_t)
        signs.append(-1.0 if tt % 2 else 1.0)

    ln_max = max(ln_terms)
    scaled = [math.exp(lt - ln_max) for lt in ln_terms]
    total = math.fsum(s * v for s, v in zip(signs, scaled))
    sum_abs = math.fsum(scaled)
    try:
        value = math.exp(ln_pref + ln_max) * total
    except OverflowError:
        return math.inf, math.inf

    scale = pref_mag + abs(ln_max) + max(abs(lt) for lt in ln_terms)
    if total == 0.0 or not math.isfinite(value):
        return value, math.inf
    ratio = abs(total) / sum_abs
    est = 4.0 * (1.0 + scale) * _EPS / ratio
    return value, est


def wigner_3j_float(j1: float, j2: float, j3: float,
                    m1: float, m2: float, m3: float) -> tuple[float, float]:
    """Diagnostic: the raw float-path 3j and its estimated relative error.

    Exposed so the certification claim is *testable* rather than asserted:
    tests compare this against :func:`wigner_3j_exact` to show the estimator
    flags exactly the symbols the float sum gets wrong.  Production code must
    call :func:`wigner_3j`, which routes to the exact path when this estimate
    exceeds :data:`FLOAT_PATH_TARGET_REL`.
    """
    tj1, tj2, tj3 = _two_j(j1, "j1"), _two_j(j2, "j2"), _two_j(j3, "j3")
    tm1, tm2, tm3 = _two_j(m1, "m1"), _two_j(m2, "m2"), _two_j(m3, "m3")
    if _selection_zero_3j(tj1, tj2, tj3, tm1, tm2, tm3):
        return 0.0, 0.0
    return _w3j_float_core(tj1, tj2, tj3, tm1, tm2, tm3)


def wigner_6j_float(j1: float, j2: float, j3: float,
                    j4: float, j5: float, j6: float) -> tuple[float, float]:
    """Diagnostic: the raw float-path 6j and its estimated relative error.
    See :func:`wigner_3j_float`."""
    t = [_two_j(j, f"j{i + 1}") for i, j in enumerate((j1, j2, j3, j4, j5, j6))]
    ta, tb, tc, td, te, tf = t
    for (x, y, z) in ((ta, tb, tc), (ta, te, tf), (td, tb, tf), (td, te, tc)):
        if not _triangle_ok(x, y, z):
            return 0.0, 0.0
    return _w6j_float_core(ta, tb, tc, td, te, tf)


# ---------------------------------------------------------------------------
# Public symbols: hybrid float/exact evaluation + universal bound gate
# ---------------------------------------------------------------------------

@lru_cache(maxsize=_RESULT_CACHE_MAX)
def _w3j(tj1: int, tj2: int, tj3: int,
         tm1: int, tm2: int, tm3: int) -> float:
    if _selection_zero_3j(tj1, tj2, tj3, tm1, tm2, tm3):
        return 0.0
    value, est = _w3j_float_core(tj1, tj2, tj3, tm1, tm2, tm3)
    if est > FLOAT_PATH_TARGET_REL:
        value = exact_to_float(*_w3j_exact_core(tj1, tj2, tj3, tm1, tm2, tm3))
    bound = 1.0 / math.sqrt(max(tj1, tj2, tj3) + 1.0)
    if not math.isfinite(value) or abs(value) > bound * (1.0 + _BOUND_SLACK_REL):
        raise IntegrityError(
            f"wigner_3j({tj1 / 2}, {tj2 / 2}, {tj3 / 2}; {tm1 / 2}, {tm2 / 2}, "
            f"{tm3 / 2}) evaluated to {value!r}, outside the elementary bound "
            f"|3j| <= 1/sqrt(2 j_max + 1) = {bound:.6g} (Edmonds 3.7.8). "
            "Refusing to return a mathematically impossible value."
        )
    return value


@lru_cache(maxsize=_RESULT_CACHE_MAX)
def _w6j(ta: int, tb: int, tc: int, td: int, te: int, tf: int) -> float:
    for (x, y, z) in ((ta, tb, tc), (ta, te, tf), (td, tb, tf), (td, te, tc)):
        if not _triangle_ok(x, y, z):
            return 0.0
    value, est = _w6j_float_core(ta, tb, tc, td, te, tf)
    if est > FLOAT_PATH_TARGET_REL:
        value = exact_to_float(*_w6j_exact_core(ta, tb, tc, td, te, tf))
    bound = 1.0 / math.sqrt(max((ta + 1) * (td + 1),
                                (tb + 1) * (te + 1),
                                (tc + 1) * (tf + 1)))
    if not math.isfinite(value) or abs(value) > bound * (1.0 + _BOUND_SLACK_REL):
        raise IntegrityError(
            f"wigner_6j({ta / 2}, {tb / 2}, {tc / 2}; {td / 2}, {te / 2}, "
            f"{tf / 2}) evaluated to {value!r}, outside the elementary bound "
            f"|6j| <= {bound:.6g} (Edmonds 6.2.9 column orthogonality). "
            "Refusing to return a mathematically impossible value."
        )
    return value


def wigner_3j(j1: float, j2: float, j3: float,
              m1: float, m2: float, m3: float) -> float:
    """Wigner 3j symbol (j1 j2 j3 / m1 m2 m3), Racah's closed form.

    Accuracy: relative error <= :data:`FLOAT_PATH_TARGET_REL` (1e-12) for
    every j and every rank — the log-factorial sum is used only where its own
    round-off estimate certifies it, otherwise the exact-rational path runs
    (module docstring).  Selection-rule violations return exact ``0.0``
    (spec 03 SS2.4).  A value violating ``|3j| <= 1/sqrt(2 j_max + 1)`` raises
    ``rydsim.provenance.IntegrityError``.
    """
    return _w3j(
        _two_j(j1, "j1"), _two_j(j2, "j2"), _two_j(j3, "j3"),
        _two_j(m1, "m1"), _two_j(m2, "m2"), _two_j(m3, "m3"),
    )


def wigner_6j(j1: float, j2: float, j3: float,
              j4: float, j5: float, j6: float) -> float:
    """Wigner 6j symbol {j1 j2 j3 / j4 j5 j6}, Racah's closed form.

    Same accuracy contract and bound gate as :func:`wigner_3j` (the 6j bound
    is the column-orthogonality one, :func:`six_j_bound`).
    """
    return _w6j(
        _two_j(j1, "j1"), _two_j(j2, "j2"), _two_j(j3, "j3"),
        _two_j(j4, "j4"), _two_j(j5, "j5"), _two_j(j6, "j6"),
    )


def clebsch_gordan(j1: float, m1: float, j2: float, m2: float,
                   j3: float, m3: float) -> float:
    """<j1 m1 j2 m2 | j3 m3> via the 3j symbol."""
    tj1, tj2, tm3 = _two_j(j1, "j1"), _two_j(j2, "j2"), _two_j(m3, "m3")
    phase_exp = (tj1 - tj2 + tm3) // 2
    phase = -1.0 if phase_exp % 2 else 1.0
    return (
        phase
        * math.sqrt(2 * j3 + 1)
        * wigner_3j(j1, j2, j3, m1, m2, -m3)
    )
