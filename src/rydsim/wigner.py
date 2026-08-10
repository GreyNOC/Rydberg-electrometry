"""Wigner 3j / 6j symbols and Clebsch-Gordan coefficients.

Pure angular-momentum algebra (Racah closed forms) implemented with
log-factorials for speed and overflow safety. Half-integer angular momenta
are supported; internally everything is doubled to integers.

Validation: tests/test_wigner.py cross-checks against sympy.physics.wigner
(exact symbolic values) over a broad grid, plus orthogonality sum rules.

Conventions: Condon-Shortley phases, as in Edmonds, *Angular Momentum in
Quantum Mechanics* (1957) — the same convention sympy uses.
"""

from __future__ import annotations

import math
from functools import lru_cache

_LGAMMA_CACHE_MAX = 4096


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


def wigner_3j(j1: float, j2: float, j3: float,
              m1: float, m2: float, m3: float) -> float:
    """Wigner 3j symbol (j1 j2 j3 / m1 m2 m3). Racah's closed form."""
    tj1, tj2, tj3 = _two_j(j1, "j1"), _two_j(j2, "j2"), _two_j(j3, "j3")
    tm1, tm2, tm3 = _two_j(m1, "m1"), _two_j(m2, "m2"), _two_j(m3, "m3")

    # selection rules
    if tm1 + tm2 + tm3 != 0:
        return 0.0
    if not _triangle_ok(tj1, tj2, tj3):
        return 0.0
    if abs(tm1) > tj1 or abs(tm2) > tj2 or abs(tm3) > tj3:
        return 0.0
    if (tj1 + tm1) % 2 or (tj2 + tm2) % 2 or (tj3 + tm3) % 2:
        return 0.0  # m must have same integer/half-integer character as j

    # Racah sum limits (all in doubled units -> halved where needed)
    # t ranges so that all factorial arguments are non-negative:
    #   t >= 0, t >= (tj2 - tj3 - tm1)/2, t >= (tj1 - tj3 + tm2)/2
    #   t <= (tj1 + tj2 - tj3)/2, t <= (tj1 - tm1)/2, t <= (tj2 + tm2)/2
    t_min = max(0, (tj2 - tj3 - tm1) // 2, (tj1 - tj3 + tm2) // 2)
    t_max = min((tj1 + tj2 - tj3) // 2, (tj1 - tm1) // 2, (tj2 + tm2) // 2)
    if t_min > t_max:
        return 0.0

    ln_pref = _ln_delta(tj1, tj2, tj3) + 0.5 * (
        _lnfact((tj1 + tm1) // 2) + _lnfact((tj1 - tm1) // 2)
        + _lnfact((tj2 + tm2) // 2) + _lnfact((tj2 - tm2) // 2)
        + _lnfact((tj3 + tm3) // 2) + _lnfact((tj3 - tm3) // 2)
    )

    # sum terms in scaled linear space for stability
    ln_terms = []
    signs = []
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
    total = sum(s * math.exp(lt - ln_max) for s, lt in zip(signs, ln_terms))

    phase = -1.0 if ((tj1 - tj2 - tm3) // 2) % 2 else 1.0
    return phase * math.exp(ln_pref + ln_max) * total


def wigner_6j(j1: float, j2: float, j3: float,
              j4: float, j5: float, j6: float) -> float:
    """Wigner 6j symbol {j1 j2 j3 / j4 j5 j6}. Racah's closed form."""
    t = [_two_j(j, f"j{i+1}") for i, j in enumerate((j1, j2, j3, j4, j5, j6))]
    ta, tb, tc, td, te, tf = t

    for (x, y, z) in ((ta, tb, tc), (ta, te, tf), (td, tb, tf), (td, te, tc)):
        if not _triangle_ok(x, y, z):
            return 0.0

    ln_pref = (
        _ln_delta(ta, tb, tc) + _ln_delta(ta, te, tf)
        + _ln_delta(td, tb, tf) + _ln_delta(td, te, tc)
    )

    # doubled sums for the seven factorial arguments
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
        return 0.0

    ln_terms = []
    signs = []
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
    total = sum(s * math.exp(lt - ln_max) for s, lt in zip(signs, ln_terms))
    return math.exp(ln_pref + ln_max) * total


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
