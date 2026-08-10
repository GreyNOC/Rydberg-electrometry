"""Validation of rydsim.angular against spec 03 SS6 (benchmarks B1-B32).

Every benchmark row of docs/spec/03-angular-algebra-dipole-moments.md SS6 is
implemented here, each test docstring citing its row. Property-based tests use
the exact-rational Wigner oracle mandated by spec 03 SS4.3.6 (Fraction
arithmetic, ships in tests only) — the no-fabrication guarantee for the whole
angular layer — plus a sympy cross-check of the oracle itself.

Constant transcriptions below carry source + confidence tags per the house
rule (docs/spec/00-integrity-audit.md).
"""

from __future__ import annotations

import math
from fractions import Fraction
from math import factorial as fact

import numpy as np
import pytest

from rydsim.angular import (
    FineState,
    angular_factor,
    dipole_matrix_element,
    effective_probe_dipole_far,
    effective_rf_dipole,
    einstein_A,
    line_strength_S,
    oscillator_strength,
    oscillator_strength_l,
    racah_to_steck,
    reduced_C1,
    reduced_dipole_F,
    reduced_dipole_j,
    steck_to_racah,
    trk_sum,
)
from rydsim.constants import A0, AU_DIPOLE, C, E_CHARGE, E_H, EPS0, HBAR, M_E
from rydsim.provenance import IntegrityError
from rydsim.wigner import wigner_3j, wigner_6j

# ---------------------------------------------------------------------------
# D-line reference data — Steck alkali datasheets rev 2.3.4 (8 Aug 2025),
# spec 03 SS3.2 [VERIFIED — machine-extracted from the primary PDFs during
# spec preparation; audit 00 re-confirmed the Cs rows from the PDF itself].
# d_steck_ea0 = <J||er||J'> in the STECK convention, units e*a0.
# ---------------------------------------------------------------------------
D_LINES = {
    "Rb87_D2": dict(nu_hz=384.2304844685e12, tau_s=26.2348e-9,
                    d_steck_ea0=4.22752, j_lower=0.5, j_upper=1.5),
    "Rb87_D1": dict(nu_hz=377.107463380e12, tau_s=27.679e-9,
                    d_steck_ea0=2.9931, j_lower=0.5, j_upper=0.5),
    "Rb85_D2": dict(nu_hz=384.230406373e12, tau_s=26.2348e-9,
                    d_steck_ea0=4.22753, j_lower=0.5, j_upper=1.5),
    "Rb85_D1": dict(nu_hz=377.107385690e12, tau_s=27.679e-9,
                    d_steck_ea0=2.9931, j_lower=0.5, j_upper=0.5),
    "Cs_D2": dict(nu_hz=351.72571850e12, tau_s=30.405e-9,
                  d_steck_ea0=4.4837, j_lower=0.5, j_upper=1.5),
    "Cs_D1": dict(nu_hz=335.116048807e12, tau_s=34.791e-9,
                  d_steck_ea0=3.1869, j_lower=0.5, j_upper=0.5),
}

# Nuclear spins (exact integers/half-integers, standard) [VERIFIED, Steck 2.3.4]
NUCLEAR_SPIN = {"Rb87": 1.5, "Rb85": 2.5, "Cs": 3.5}


# ---------------------------------------------------------------------------
# Exact-rational Wigner oracle (spec 03 SS4.3.6; test-only ground truth).
# Returns (sign, exact square as Fraction); value = sign * sqrt(square).
# ---------------------------------------------------------------------------

def _tw(x: float) -> int:
    t = round(2 * x)
    assert abs(2 * x - t) < 1e-9, f"{x} not (half-)integer"
    return int(t)


def wigner_3j_exact(j1, j2, j3, m1, m2, m3) -> tuple[int, Fraction]:
    """Exact-rational 3j oracle (Racah closed sum, Fraction arithmetic)."""
    t1, t2, t3 = _tw(j1), _tw(j2), _tw(j3)
    u1, u2, u3 = _tw(m1), _tw(m2), _tw(m3)
    if u1 + u2 + u3 != 0:
        return 0, Fraction(0)
    if not (abs(t1 - t2) <= t3 <= t1 + t2) or (t1 + t2 + t3) % 2:
        return 0, Fraction(0)
    if abs(u1) > t1 or abs(u2) > t2 or abs(u3) > t3:
        return 0, Fraction(0)
    if (t1 + u1) % 2 or (t2 + u2) % 2 or (t3 + u3) % 2:
        return 0, Fraction(0)
    pref = Fraction(
        fact((t1 + t2 - t3) // 2) * fact((t1 - t2 + t3) // 2)
        * fact((-t1 + t2 + t3) // 2),
        fact((t1 + t2 + t3) // 2 + 1),
    )
    pref *= (fact((t1 + u1) // 2) * fact((t1 - u1) // 2)
             * fact((t2 + u2) // 2) * fact((t2 - u2) // 2)
             * fact((t3 + u3) // 2) * fact((t3 - u3) // 2))
    kmin = max(0, (t2 - t3 - u1) // 2, (t1 - t3 + u2) // 2)
    kmax = min((t1 + t2 - t3) // 2, (t1 - u1) // 2, (t2 + u2) // 2)
    S = Fraction(0)
    for k in range(kmin, kmax + 1):
        den = (fact(k) * fact(k - (t2 - t3 - u1) // 2)
               * fact(k - (t1 - t3 + u2) // 2)
               * fact((t1 + t2 - t3) // 2 - k)
               * fact((t1 - u1) // 2 - k) * fact((t2 + u2) // 2 - k))
        S += Fraction(-1 if k % 2 else 1, den)
    if S == 0:
        return 0, Fraction(0)
    phase = -1 if ((t1 - t2 - u3) // 2) % 2 else 1
    sign = phase * (1 if S > 0 else -1)
    return sign, pref * S * S


def wigner_6j_exact(a, b, c, d, e, f) -> tuple[int, Fraction]:
    """Exact-rational 6j oracle (Racah closed sum, Fraction arithmetic)."""
    ta, tb, tc, td, te, tf = (_tw(x) for x in (a, b, c, d, e, f))
    for (x, y, z) in ((ta, tb, tc), (ta, te, tf), (td, tb, tf), (td, te, tc)):
        if not (abs(x - y) <= z <= x + y) or (x + y + z) % 2:
            return 0, Fraction(0)

    def delta_sq(x, y, z):
        return Fraction(
            fact((x + y - z) // 2) * fact((x - y + z) // 2)
            * fact((-x + y + z) // 2),
            fact((x + y + z) // 2 + 1),
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
    S = Fraction(0)
    for k in range(max(s1, s2, s3, s4), min(q1, q2, q3) + 1):
        den = (fact(k - s1) * fact(k - s2) * fact(k - s3) * fact(k - s4)
               * fact(q1 - k) * fact(q2 - k) * fact(q3 - k))
        S += Fraction((-1 if k % 2 else 1) * fact(k + 1), den)
    if S == 0:
        return 0, Fraction(0)
    return (1 if S > 0 else -1), pref * S * S


def _oracle_val(sign: int, square: Fraction) -> float:
    return sign * math.sqrt(float(square))


def test_oracle_vs_sympy():
    """Validate the exact-rational oracle itself against sympy (independent
    exact implementation; sympy permitted as test oracle per project rules)."""
    from sympy import Rational
    from sympy.physics.wigner import wigner_3j as sy3j
    from sympy.physics.wigner import wigner_6j as sy6j

    def r(x):
        return Rational(round(2 * x), 2)

    grid3 = [(0.5, 1, 1.5, 0.5, 0, -0.5), (2, 1, 1, 0, 0, 0),
             (2.5, 1, 1.5, 0.5, 0, -0.5), (3, 2, 4, 1, -2, 1),
             (2.5, 1.5, 2, -1.5, 0.5, 1)]
    for args in grid3:
        s, sq = wigner_3j_exact(*args)
        exact = float(sy3j(*[r(x) for x in args]))
        assert _oracle_val(s, sq) == pytest.approx(exact, abs=1e-14)
    grid6 = [(1, 1, 1, 1, 1, 1), (0, 1, 1, 1.5, 0.5, 0.5),
             (2, 1, 1, 1.5, 2.5, 0.5), (2, 3, 1, 3.5, 2.5, 0.5),
             (1.5, 1.5, 1, 1.5, 1.5, 2)]
    for args in grid6:
        s, sq = wigner_6j_exact(*args)
        exact = float(sy6j(*[r(x) for x in args]))
        assert _oracle_val(s, sq) == pytest.approx(exact, abs=1e-14)


# ---------------------------------------------------------------------------
# B1-B9: individual 3j / 6j values (dual-impl verified radicals)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("args,expected", [
    ((0.5, 1, 1.5, 0.5, 0, -0.5), 1 / math.sqrt(6)),        # B1
    ((2, 1, 1, 0, 0, 0), math.sqrt(2 / 15)),                # B2
    ((2.5, 1, 1.5, 0.5, 0, -0.5), -1 / math.sqrt(10)),      # B3
    ((50, 1, 51, 0, 0, 0), -math.sqrt(51 / 10403)),         # B4
    ((60.5, 1, 61.5, 0.5, 0, -0.5), 1 / math.sqrt(246)),    # B5
])
def test_3j_benchmark_values(args, expected):
    """Spec 03 SS6 rows B1-B5: tabulated 3j radicals, tol 1e-12 (1e-10 large j)."""
    tol = 1e-12 if args[0] <= 10 else 1e-10
    assert wigner_3j(*args) == pytest.approx(expected, rel=tol)


@pytest.mark.parametrize("args,expected", [
    ((1, 1, 1, 1, 1, 1), 1 / 6),                             # B6
    ((0, 1, 1, 1.5, 0.5, 0.5), -1 / math.sqrt(6)),           # B7
    ((2, 1, 1, 1.5, 2.5, 0.5), -1 / math.sqrt(20)),          # B8
    ((2, 3, 1, 3.5, 2.5, 0.5), -1 / math.sqrt(42)),          # B9
])
def test_6j_benchmark_values(args, expected):
    """Spec 03 SS6 rows B6-B9: tabulated 6j values, tol 1e-12."""
    assert wigner_6j(*args) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("j1,j2,j3,m3", [
    (1, 1, 2, 0), (1, 1, 2, 1),
    (1.5, 1, 2.5, 0.5), (1.5, 1, 2.5, -1.5),
    (6, 4, 9, 0), (6, 4, 9, 3),
])
def test_3j_orthogonality(j1, j2, j3, m3):
    """Spec 03 SS6 row B10: sum_{m1,m2} (2j3+1) 3j^2 = 1 at fixed m3."""
    total = 0.0
    t1, t2 = _tw(j1), _tw(j2)
    for u1 in range(-t1, t1 + 1, 2):
        for u2 in range(-t2, t2 + 1, 2):
            if u1 + u2 + _tw(m3) != 0:
                continue
            total += (2 * j3 + 1) * wigner_3j(j1, j2, j3, u1 / 2, u2 / 2, m3) ** 2
    assert total == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("p,q,expected", [
    (1.5, 1.5, 1 / 4), (2.5, 2.5, 1 / 6), (1.5, 2.5, 0.0),
])
def test_6j_orthogonality(p, q, expected):
    """Spec 03 SS6 row B11: sum_x (2x+1){a b x; c d p}{a b x; c d q} for
    (a,b,c,d) = (3/2, 1, 5/2, 2) equals delta_pq/(2p+1): 1/4, 1/6, 0."""
    a, b, c, d = 1.5, 1, 2.5, 2
    total = 0.0
    for tx in range(1, 6, 2):  # x in {1/2, 3/2, 5/2}
        x = tx / 2
        total += (2 * x + 1) * wigner_6j(a, b, x, c, d, p) * wigner_6j(a, b, x, c, d, q)
    assert total == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# B12-B15: reduced C1, NIST angular factors
# ---------------------------------------------------------------------------

def test_reduced_C1_values():
    """Spec 03 SS6 row B12: <0||C1||1> = -1, <1||C1||2> = -sqrt(2),
    <2||C1||1> = +sqrt(2); closed forms of Eq. (2.8), tol 1e-12."""
    assert reduced_C1(0, 1) == pytest.approx(-1.0, rel=1e-12)
    assert reduced_C1(1, 2) == pytest.approx(-math.sqrt(2), rel=1e-12)
    assert reduced_C1(2, 1) == pytest.approx(math.sqrt(2), rel=1e-12)
    # closed-form contract over a range
    for l in range(0, 8):
        assert reduced_C1(l, l + 1) == pytest.approx(-math.sqrt(l + 1), rel=1e-12)
        if l > 0:
            assert reduced_C1(l, l - 1) == pytest.approx(math.sqrt(l), rel=1e-12)
        assert reduced_C1(l, l) == 0.0
        assert reduced_C1(l, l + 3) == 0.0


@pytest.mark.parametrize("qns,exact,nist", [
    ((0, 0.5, 0.5, 1, 1.5, 0.5, 0), math.sqrt(2) / 3, 0.4714),      # B13
    ((2, 2.5, 0.5, 1, 1.5, 0.5, 0), math.sqrt(6) / 5, 0.4899),      # B14
    ((2, 2.5, 0.5, 3, 3.5, 0.5, 0), 2 * math.sqrt(3) / 7, 0.4949),  # B15
])
def test_nist_angular_factors(qns, exact, nist):
    """Spec 03 SS6 rows B13-B15: |A| for the NIST RF pairs (pi, m_j = 1/2)
    equals sqrt(2)/3, sqrt(6)/5, 2 sqrt(3)/7 to 1e-12 and the published
    4-decimal values (Simons/Gordon/Holloway JAP 120, 123103 (2016)) to 2e-4."""
    A = abs(angular_factor(*qns))
    assert A == pytest.approx(exact, rel=1e-12)
    assert A == pytest.approx(nist, rel=2e-4)


@pytest.mark.parametrize("qns,exact", [
    ((2, 2.5, 1.5, 1, 1.5, 1.5, 0), 2 / 5),                  # D5/2->P3/2 m=3/2
    ((2, 2.5, 1.5, 3, 3.5, 1.5, 0), math.sqrt(10) / 7),      # D5/2->F7/2 m=3/2
    ((2, 2.5, 2.5, 3, 3.5, 2.5, 0), math.sqrt(6) / 7),       # D5/2->F7/2 m=5/2
])
def test_pi_manifold_factors(qns, exact):
    """Spec 03 SS2.5 full pi-manifold table (exact, dual-verified)."""
    assert abs(angular_factor(*qns)) == pytest.approx(exact, rel=1e-12)


def test_angular_factor_m_symmetry():
    """|A(+m)| = |A(-m)| (3j m -> -m symmetry) — the basis of the single-p
    NIST convention (spec 03 SS2.5)."""
    for (l, j, lp, jp) in [(0, 0.5, 1, 1.5), (2, 2.5, 1, 1.5), (2, 2.5, 3, 3.5)]:
        a_plus = angular_factor(l, j, 0.5, lp, jp, 0.5, 0)
        a_minus = angular_factor(l, j, -0.5, lp, jp, -0.5, 0)
        assert abs(a_plus) == pytest.approx(abs(a_minus), rel=1e-12)


def test_angular_factor_normalization():
    """Racah normalization (spec 03 SS2.1): sum_{m',q} |A|^2 =
    |reduced_dipole_j|^2 / (2j+1), independent of m."""
    for (l, j, lp, jp) in [(0, 0.5, 1, 1.5), (1, 1.5, 2, 2.5), (2, 2.5, 1, 1.5)]:
        red_sq = reduced_dipole_j(l, j, lp, jp) ** 2
        tj, tjp = _tw(j), _tw(jp)
        for tm in range(-tj, tj + 1, 2):
            total = sum(
                angular_factor(l, j, tm / 2, lp, jp, tmp / 2, q) ** 2
                for q in (-1, 0, 1)
                for tmp in range(-tjp, tjp + 1, 2)
            )
            assert total == pytest.approx(red_sq / (tj + 1), rel=1e-12)


def test_fine_structure_reduction_identity():
    """Assignment/spec 03 SS2.2 consistency identity to 1e-12: Steck-form
    Eq. (2.10) equals Racah-form Eq. (2.5) after applying Eq. (2.3) at both
    the L and J levels — closes the 6j fine-structure reduction convention."""
    checked = 0
    for l in range(0, 5):
        for j in ([0.5] if l == 0 else [l - 0.5, l + 0.5]):
            for lp in (l - 1, l + 1):
                if lp < 0:
                    continue
                for jp in ([0.5] if lp == 0 else [lp - 0.5, lp + 0.5]):
                    if abs(j - jp) > 1:
                        continue
                    # Steck L-level reduced element / (e R) = C1/sqrt(2l+1)
                    lhs = (
                        reduced_C1(l, lp) / math.sqrt(2 * l + 1)
                        * (-1) ** int(round(jp + l + 1 + 0.5))
                        * math.sqrt((2 * jp + 1) * (2 * l + 1))
                        * wigner_6j(l, lp, 1, jp, j, 0.5)
                    )
                    rhs = reduced_dipole_j(l, j, lp, jp) / math.sqrt(2 * j + 1)
                    assert lhs == pytest.approx(rhs, abs=1e-12)
                    checked += 1
    assert checked >= 12


def test_d_line_fine_structure_factors():
    """Spec 03 SS2.2: |<S1/2||er||P3/2>_R|/(eR) = 2/sqrt(3), D1 = sqrt(2/3);
    Steck equivalents sqrt(2/3) and sqrt(1/3)."""
    d2 = reduced_dipole_j(0, 0.5, 1, 1.5)
    d1 = reduced_dipole_j(0, 0.5, 1, 0.5)
    assert abs(d2) == pytest.approx(2 / math.sqrt(3), rel=1e-12)
    assert abs(d1) == pytest.approx(math.sqrt(2 / 3), rel=1e-12)
    assert abs(racah_to_steck(d2, 0.5)) == pytest.approx(math.sqrt(2 / 3), rel=1e-12)
    assert abs(racah_to_steck(d1, 0.5)) == pytest.approx(math.sqrt(1 / 3), rel=1e-12)


# ---------------------------------------------------------------------------
# B16-B17: hyperfine line strengths
# ---------------------------------------------------------------------------

def test_line_strength_rb87():
    """Spec 03 SS6 row B16: S_FF' Rb-87 D2, F=2 -> F'=1,2,3 = 1/20, 1/4, 7/10
    (Steck Eq. 41 / Table 8), tol 1e-12."""
    I = NUCLEAR_SPIN["Rb87"]
    assert line_strength_S(0.5, 1.5, 2, 1, I) == pytest.approx(1 / 20, rel=1e-12)
    assert line_strength_S(0.5, 1.5, 2, 2, I) == pytest.approx(1 / 4, rel=1e-12)
    assert line_strength_S(0.5, 1.5, 2, 3, I) == pytest.approx(7 / 10, rel=1e-12)


def test_line_strength_rb85_cs():
    """Spec 03 SS2.6 table: Rb-85 F=3 -> 5/63, 5/18, 9/14; Cs F=4 -> 7/72,
    7/24, 11/18 (exact fractions, dual-impl verified)."""
    I85 = NUCLEAR_SPIN["Rb85"]
    assert line_strength_S(0.5, 1.5, 3, 2, I85) == pytest.approx(5 / 63, rel=1e-12)
    assert line_strength_S(0.5, 1.5, 3, 3, I85) == pytest.approx(5 / 18, rel=1e-12)
    assert line_strength_S(0.5, 1.5, 3, 4, I85) == pytest.approx(9 / 14, rel=1e-12)
    Ics = NUCLEAR_SPIN["Cs"]
    assert line_strength_S(0.5, 1.5, 4, 3, Ics) == pytest.approx(7 / 72, rel=1e-12)
    assert line_strength_S(0.5, 1.5, 4, 4, Ics) == pytest.approx(7 / 24, rel=1e-12)
    assert line_strength_S(0.5, 1.5, 4, 5, Ics) == pytest.approx(11 / 18, rel=1e-12)


def test_line_strength_sum_rule():
    """Spec 03 SS6 row B17: sum_F' S_FF' = 1 for all 3 species, both ground F,
    tol 1e-12 (Steck Eq. 42)."""
    J, Jp = 0.5, 1.5
    for I in NUCLEAR_SPIN.values():
        for F in (I - 0.5, I + 0.5):
            tFp_min, tFp_max = _tw(abs(Jp - I)), _tw(Jp + I)
            total = sum(
                line_strength_S(J, Jp, F, tFp / 2, I)
                for tFp in range(tFp_min, tFp_max + 1, 2)
            )
            assert total == pytest.approx(1.0, abs=1e-12)


def test_line_strength_vs_reduced_dipole_F():
    """Wiring identity: S_FF' = (2J+1)/(2F+1) * reduced_dipole_F^2 (Eqs. 2.12
    vs 2.15, related by 6j column symmetry), tol 1e-12."""
    J, Jp = 0.5, 1.5
    for I in NUCLEAR_SPIN.values():
        for F in (I - 0.5, I + 0.5):
            for dF in (-1, 0, 1):
                Fp = F + dF
                if Fp < abs(Jp - I) - 1e-9 or Fp > Jp + I + 1e-9:
                    continue
                s = line_strength_S(J, Jp, F, Fp, I)
                r = reduced_dipole_F(J, Jp, F, Fp, I)
                assert s == pytest.approx((2 * J + 1) / (2 * F + 1) * r * r, abs=1e-12)


# ---------------------------------------------------------------------------
# B18-B22: lifetime <-> dipole round trips (Einstein A, both conventions)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("line,tol", [
    ("Rb87_D2", 3e-4),   # B18
    ("Rb87_D1", 5e-4),   # B19
    ("Cs_D2", 5e-4),     # B20
    ("Cs_D1", 5e-4),     # B21
    ("Rb85_D2", 3e-4),
    ("Rb85_D1", 5e-4),
])
def test_lifetime_from_dipole(line, tol):
    """Spec 03 SS6 rows B18-B21: tau from the Steck rev 2.3.4 dipole through
    Eq. (2.3) + Eq. (2.20) reproduces the Steck lifetime (closes the whole
    convention chain, including tau(Rb-87 5P3/2) = 26.2348 ns and
    tau(Cs 6P3/2) = 30.405 ns)."""
    d = D_LINES[line]
    omega0 = 2 * math.pi * d["nu_hz"]
    d_racah_Cm = steck_to_racah(d["d_steck_ea0"], d["j_lower"]) * AU_DIPOLE
    A = einstein_A(omega0, d_racah_Cm, d["j_upper"])
    assert 1.0 / A == pytest.approx(d["tau_s"], rel=tol)


def test_einstein_A_racah_form_direct():
    """Spec 03 SS6 row B22: Racah-form Eq. (2.20) with the printed
    d_R = sqrt(2)*4.22752 = 5.97862 e*a0, j_e = 3/2, gives the same
    tau = 26.2348 ns, tol 3e-4 (convention identity)."""
    omega0 = 2 * math.pi * D_LINES["Rb87_D2"]["nu_hz"]
    A = einstein_A(omega0, 5.97862 * AU_DIPOLE, 1.5)
    assert 1.0 / A == pytest.approx(26.2348e-9, rel=3e-4)


def test_einstein_A_steck_form_equivalence():
    """Eq. (2.20) == Eq. (2.21) under Eq. (2.3): the Steck-form rate
    (2J_g+1)/(2J_e+1) |d_S|^2 assembled inline equals einstein_A, 1e-12."""
    for line in ("Rb87_D2", "Cs_D1"):
        d = D_LINES[line]
        omega0 = 2 * math.pi * d["nu_hz"]
        jg, je = d["j_lower"], d["j_upper"]
        d_S_Cm = d["d_steck_ea0"] * AU_DIPOLE
        A_steck = (omega0**3 / (3 * math.pi * EPS0 * HBAR * C**3)
                   * (2 * jg + 1) / (2 * je + 1) * d_S_Cm**2)
        A_racah = einstein_A(omega0, steck_to_racah(d["d_steck_ea0"], jg) * AU_DIPOLE, je)
        assert A_racah == pytest.approx(A_steck, rel=1e-12)


def test_einstein_A_vectorized():
    """einstein_A broadcasts over omega0/d (spec 03 SS5 contract)."""
    om = 2 * math.pi * np.array([384.23e12, 351.73e12])
    out = einstein_A(om, 4.5 * AU_DIPOLE, 1.5)
    assert out.shape == (2,)
    assert out[0] == pytest.approx(einstein_A(float(om[0]), 4.5 * AU_DIPOLE, 1.5))


# ---------------------------------------------------------------------------
# B23-B25: inverted radial-integral targets and the D2/D1 sqrt(2) band
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("line,r_expected", [
    ("Rb87_D2", 5.17766), ("Rb87_D1", 5.18420),
    ("Rb85_D2", 5.17768), ("Rb85_D1", 5.18420),
    ("Cs_D2", 5.49142), ("Cs_D1", 5.51988),
])
def test_inverted_radial_targets(line, r_expected):
    """Spec 03 SS6 rows B23/B24 (angular/convention part): inverting the
    Steck dipole through the fine-structure chain gives the radial-integral
    targets of SS2.7 (|R(5S-5P3/2)|_Rb87 = 5.1777 a0, |R(6S-6P3/2)|_Cs =
    5.4914 a0). The doc-02 numerical comparison against these targets is
    owned by the radial/dipoles test suite (module outside this assignment)."""
    d = D_LINES[line]
    red = reduced_dipole_j(0, 0.5, 1, d["j_upper"])
    R_a0 = abs(steck_to_racah(d["d_steck_ea0"], d["j_lower"]) / red)
    assert R_a0 == pytest.approx(r_expected, rel=1e-4)


@pytest.mark.parametrize("d2,d1", [
    (4.22752, 2.9931),   # Rb-87
    (4.22753, 2.9931),   # Rb-85
    (4.4837, 3.1869),    # Cs
])
def test_d2_d1_ratio_band(d2, d1):
    """Spec 03 SS6 row B25: measured D2/D1 Steck-dipole ratio within
    [-0.7%, 0%] of sqrt(2) (spin-orbit radial difference, per species)."""
    ratio = d2 / d1
    dev = ratio / math.sqrt(2) - 1.0
    assert -0.007 <= dev <= 0.0


# ---------------------------------------------------------------------------
# B26-B31: hydrogen anchors, TRK
# ---------------------------------------------------------------------------

def _hydrogen_1s2p_radial_a0() -> float:
    """<1s|r|2p> numerically from the analytic wavefunctions (a.u.).

    P_1s = 2 r e^-r, P_2p = r^2 e^(-r/2) / (2 sqrt(6)) — analytic hydrogen,
    Bethe-Salpeter SS63 [VERIFIED: closed form 128 sqrt(6)/243]."""
    r = np.linspace(0.0, 80.0, 400_001)
    p1s = 2.0 * r * np.exp(-r)
    p2p = r**2 * np.exp(-r / 2.0) / (2.0 * math.sqrt(6.0))
    return float(np.trapezoid(p1s * r * p2p, r))


def test_hydrogen_radial_1s2p():
    """Spec 03 SS6 row B26: <1s|r|2p> = 128 sqrt(6)/243 = 1.2902662 a0,
    tol 1e-6 (analytic, reproduced numerically)."""
    assert _hydrogen_1s2p_radial_a0() == pytest.approx(128 * math.sqrt(6) / 243, rel=1e-6)


def test_hydrogen_f_1s2p():
    """Spec 03 SS6 row B27: f(1s->2p) = 2^13/3^9 = 0.4161967, tol 1e-6,
    via Eq. (2.18) with omega = (3/8) E_h / hbar."""
    omega = 0.375 * E_H / HBAR
    f = oscillator_strength_l(omega, 128 * math.sqrt(6) / 243, 0, 1)
    assert f == pytest.approx(2**13 / 3**9, rel=1e-6)
    assert f == pytest.approx(0.4161967, rel=1e-6)


def _hydrogen_discrete_f(n: int) -> float:
    """Closed-form f(1s -> np) = 2^8 n^5 (n-1)^(2n-4) / (3 (n+1)^(2n+4)).

    Bethe-Salpeter SS63 [VERIFIED in spec 03 SS2.7: reproduces 2^13/3^9 at
    n = 2 and sums to 0.565004]. Evaluated in log space for large n."""
    if n == 2:
        return 2**13 / 3**9
    ln_f = (8 * math.log(2) + 5 * math.log(n) + (2 * n - 4) * math.log(n - 1)
            - math.log(3) - (2 * n + 4) * math.log(n + 1))
    return math.exp(ln_f)


def test_hydrogen_discrete_sum():
    """Spec 03 SS6 row B29: discrete sum_n f(1s->np) = 0.565004 +- 0.0005
    (closed-form series, converged; tail beyond n=4000 is < 1e-7)."""
    total = trk_sum(np.array([_hydrogen_discrete_f(n) for n in range(2, 4001)]))
    assert total == pytest.approx(0.565004, abs=5e-4)


def _hydrogen_continuum_f() -> tuple[float, float]:
    """(continuum TRK contribution, threshold df/de) for hydrogen 1s.

    df/de = (2^8/3) exp(-4 arctan(k)/k) / ((1+k^2)^4 (1 - exp(-2 pi/k))),
    e = k^2/2 (a.u.). Source: Bethe-Salpeter SS71 photoionization
    [LITERATURE-RECALL — fenced by two in-test self-checks per the
    no-fabrication rule: (i) the k->0 limit must match the discrete-series
    limit n^3 f_n (continuity across threshold), and (ii) discrete +
    continuum must satisfy TRK = 1.000 +- 0.005 (B28, spec-designated
    implementation self-check)."""
    k = np.linspace(1e-6, 25.0, 500_001)
    dfde = (256.0 / 3.0) * np.exp(-4.0 * np.arctan(k) / k) / (
        (1.0 + k**2) ** 4 * (1.0 - np.exp(-2.0 * math.pi / k)))
    return float(np.trapezoid(dfde * k, k)), float(dfde[0])


def test_hydrogen_trk_total():
    """Spec 03 SS6 row B28: TRK sum f(1s -> all np, discrete + continuum)
    = 1.000 +- 0.5% (Eq. 2.19); plus threshold-continuity self-check on the
    recalled continuum formula."""
    discrete = trk_sum(np.array([_hydrogen_discrete_f(n) for n in range(2, 4001)]))
    continuum, threshold = _hydrogen_continuum_f()
    # self-check (i): df/de at threshold == lim n^3 f(1s->np)
    n = 3000
    series_limit = n**3 * _hydrogen_discrete_f(n)
    assert threshold == pytest.approx(series_limit, rel=1e-2)
    # spec-quoted continuum share
    assert continuum == pytest.approx(0.434996, abs=1e-3)
    # self-check (ii) == B28
    assert discrete + continuum == pytest.approx(1.000, abs=0.005)


def test_partial_sum_rule_identity():
    """Spec 03 SS6 row B30: [(l+1)(2l+3) - l(2l-1)] / [3(2l+1)] = 1 exactly
    for l = 0..5 (Bethe-Salpeter partial sum rules, self-verifying pair)."""
    for l in range(0, 6):
        num = Fraction((l + 1) * (2 * l + 3) - l * (2 * l - 1), 3 * (2 * l + 1))
        assert num == 1


def test_rb_trk_sanity():
    """Spec 03 SS6 row B31: f(D1) + f(D2) for Rb-87 = 1.038 +- 0.01 from the
    VERIFIED Steck dipoles (core-valence TRK violation, sanity band)."""
    f_total = 0.0
    for line in ("Rb87_D1", "Rb87_D2"):
        d = D_LINES[line]
        omega = 2 * math.pi * d["nu_hz"]
        d_racah = steck_to_racah(d["d_steck_ea0"], d["j_lower"]) * AU_DIPOLE
        f_total += oscillator_strength(omega, d_racah, d["j_lower"])
    assert f_total == pytest.approx(1.038, abs=0.01)


def test_oscillator_strength_j_vs_l_consistency():
    """Spec 03 SS2.7 internal consistency: sum_j' f from Eq. (2.17) with
    radial integrals held equal must reproduce the l-basis Eq. (2.18)."""
    omega = 2 * math.pi * 3.0e14
    R_a0 = 4.0
    for (l, lp) in [(0, 1), (1, 2), (2, 1)]:
        j_list = [0.5] if l == 0 else [l - 0.5, l + 0.5]
        jp_list = [0.5] if lp == 0 else [lp - 0.5, lp + 0.5]
        f_l = oscillator_strength_l(omega, R_a0, l, lp)
        for j in j_list:
            f_sum = sum(
                oscillator_strength(
                    omega,
                    E_CHARGE * A0 * R_a0 * reduced_dipole_j(l, j, lp, jp),
                    j,
                )
                for jp in jp_list
            )
            assert f_sum == pytest.approx(f_l, rel=1e-12)


# ---------------------------------------------------------------------------
# B32 + NIST effective dipole + full-element plumbing
# ---------------------------------------------------------------------------

def test_effective_probe_dipole_far():
    """Spec 03 SS6 row B32 + assignment: d_eff,far(Rb-87 D2) =
    4.22752/sqrt(3) = 2.44076 e*a0 (Eq. 2.16, Steck Eq. 44), tol 1e-3
    against the spec-quoted 2.4408 and 1e-5 against the exact quotient."""
    d_eff = effective_probe_dipole_far(D_LINES["Rb87_D2"]["d_steck_ea0"])
    assert d_eff == pytest.approx(2.44076, rel=1e-5)
    assert d_eff == pytest.approx(2.4408, rel=1e-3)


def test_effective_rf_dipole_nist_convention():
    """Spec 00 lock #11: p = e a0 R |A(l,j,1/2 -> l',j',1/2; q=0)| — the
    Simons/Gordon/Holloway A factors through the public API, in C*m."""
    R = 100.0  # a0, arbitrary scale for the linearity check
    p = effective_rf_dipole(0, 0.5, 1, 1.5, R)
    assert p == pytest.approx(E_CHARGE * A0 * R * math.sqrt(2) / 3, rel=1e-12)
    p = effective_rf_dipole(2, 2.5, 1, 1.5, R)
    assert p == pytest.approx(E_CHARGE * A0 * R * math.sqrt(6) / 5, rel=1e-12)
    p = effective_rf_dipole(2, 2.5, 3, 3.5, R)
    assert p == pytest.approx(E_CHARGE * A0 * R * 2 * math.sqrt(3) / 7, rel=1e-12)
    # sign of R must not matter (only |R| observable)
    assert effective_rf_dipole(0, 0.5, 1, 1.5, -R) == pytest.approx(
        effective_rf_dipole(0, 0.5, 1, 1.5, R), rel=1e-15)


def test_dipole_matrix_element_plumbing():
    """dipole_matrix_element = e a0 R A with FineState validation; magnitude
    matches effective_rf_dipole for the m_j = 1/2 pi element."""
    bra = FineState(50, 0, 0.5)
    ket = FineState(50, 1, 1.5)
    R = 42.0
    val = dipole_matrix_element(bra, ket, 0.5, 0.5, 0, R)
    assert abs(val) == pytest.approx(effective_rf_dipole(0, 0.5, 1, 1.5, R), rel=1e-12)
    assert val == pytest.approx(
        E_CHARGE * A0 * R * angular_factor(0, 0.5, 0.5, 1, 1.5, 0.5, 0), rel=1e-15)


def test_steck_racah_round_trip():
    """Eq. (2.3) and its inverse: Rb-87 D2 4.22752 (Steck) <-> 5.97862 (Racah)."""
    d_r = steck_to_racah(4.22752, 0.5)
    assert d_r == pytest.approx(5.97862, rel=1e-5)
    assert racah_to_steck(d_r, 0.5) == pytest.approx(4.22752, rel=1e-15)


# ---------------------------------------------------------------------------
# Selection rules: exact zeros (spec 03 SS2.4 — never special-cased upstream)
# ---------------------------------------------------------------------------

def test_selection_rule_exact_zeros():
    """All E1 selection rules emerge as exact 0.0 from the symbols."""
    assert wigner_3j(1, 1, 1, 0, 0, 0) == 0.0          # parity zero
    assert wigner_3j(1, 1, 4, 0, 0, 0) == 0.0          # triangle
    assert wigner_3j(1, 1, 2, 1, 1, -1) == 0.0         # m-sum
    assert wigner_6j(1, 1, 1, 1, 1, 3) == 0.0          # triad
    assert angular_factor(1, 1.5, 0.5, 1, 1.5, 0.5, 0) == 0.0   # Delta l = 0
    assert angular_factor(0, 0.5, 0.5, 3, 3.5, 0.5, 0) == 0.0   # Delta l = 3
    assert angular_factor(0, 0.5, 0.5, 1, 1.5, 1.5, 0) == 0.0   # Delta m != q
    assert angular_factor(2, 2.5, 2.5, 1, 1.5, 2.5, 0) == 0.0   # no m=5/2 in P3/2
    assert reduced_dipole_j(0, 0.5, 2, 2.5) == 0.0     # Delta l = 2
    assert line_strength_S(0.5, 1.5, 2, 4, 1.5) == 0.0  # Delta F = 2
    assert reduced_dipole_F(0.5, 1.5, 1, 3, 1.5) == 0.0


# ---------------------------------------------------------------------------
# Refusals — IntegrityError / ValueError, never a guessed number
# ---------------------------------------------------------------------------

def test_refuse_non_half_integer():
    """Audit refusal #12: non-(half-)integer j/m raise; no rounding."""
    with pytest.raises(ValueError):
        wigner_3j(0.3, 1, 1.3, 0.3, 0, -0.3)
    with pytest.raises(ValueError):
        angular_factor(0, 0.6, 0.5, 1, 1.5, 0.5, 0)
    with pytest.raises(ValueError):
        reduced_C1(-1, 0)
    with pytest.raises(ValueError):
        angular_factor(0, 0.5, 0.5, 1, 1.5, 0.5, 2)   # q outside {-1,0,1}
    with pytest.raises(ValueError):
        oscillator_strength_l(1e15, 1.0, 0, 2)        # |dl| != 1


def test_refuse_uncertified_large_j():
    """Audit refusal #13 / spec 03 SS4.3.4: float path certified only to
    j = 150 — beyond it the chain raises IntegrityError (oracle route)."""
    with pytest.raises(IntegrityError):
        angular_factor(151, 151.5, 0.5, 152, 152.5, 0.5, 0)
    with pytest.raises(IntegrityError):
        reduced_dipole_j(151, 151.5, 152, 152.5)
    with pytest.raises(IntegrityError):
        line_strength_S(0.5, 1.5, 200, 200, 199.5)


def test_refuse_forbidden_rf_dipole():
    """effective_rf_dipole raises IntegrityError for an E1-forbidden pair
    instead of returning 0 C*m (which would invert to an infinite field)."""
    with pytest.raises(IntegrityError):
        effective_rf_dipole(0, 0.5, 0, 0.5, 100.0)
    with pytest.raises(IntegrityError):
        effective_rf_dipole(0, 0.5, 2, 2.5, 100.0)


def test_finestate_validation():
    """FineState rejects unphysical (n, l, j) combinations."""
    FineState(50, 2, 2.5)  # valid
    with pytest.raises(ValueError):
        FineState(5, 5, 5.5)       # l >= n
    with pytest.raises(ValueError):
        FineState(5, 1, 2.5)       # j != l +- 1/2
    with pytest.raises(ValueError):
        FineState(5, 1, 1.0)       # integer j with s = 1/2
    with pytest.raises(ValueError):
        FineState(0, 0, 0.5)       # n < 1


# ---------------------------------------------------------------------------
# Property-based float-vs-oracle grids (spec 03 SS6, pinned seed)
# ---------------------------------------------------------------------------

_SEED = 20260810


def test_property_rank1_3j_vs_oracle():
    """Spec 03 SS6 property test (a): 2000 random rank-1 3j (j2 = 1,
    j <= 150) agree with the exact-rational oracle to 1e-9 relative
    (measured max err during spec prep: 4.2e-12)."""
    rng = np.random.default_rng(_SEED)
    checked = 0
    while checked < 2000:
        t1 = int(rng.integers(0, 301))
        t3 = t1 + 2 * int(rng.integers(-1, 2))
        if t3 < 0 or not (abs(t1 - 2) <= t3 <= t1 + 2):
            continue
        u1 = t1 - 2 * int(rng.integers(0, t1 + 1)) if t1 else 0
        q = int(rng.integers(-1, 2))
        u3 = -u1 - 2 * q
        if abs(u3) > t3:
            continue
        args = (t1 / 2, 1.0, t3 / 2, u1 / 2, float(q), u3 / 2)
        ours = wigner_3j(*args)
        sign, sq = wigner_3j_exact(*args)
        exact = _oracle_val(sign, sq)
        if exact == 0.0:
            assert abs(ours) < 1e-12
        else:
            assert ours == pytest.approx(exact, rel=1e-9)
        checked += 1


def test_property_rank1_6j_vs_oracle():
    """Spec 03 SS6 property test (b): 2000 random 6j of the dipole-chain type
    {l j 1/2; j' l' 1} (l <= 150) agree with the oracle to 1e-9 relative
    (measured max err 3.0e-13)."""
    rng = np.random.default_rng(_SEED + 1)
    checked = 0
    while checked < 2000:
        l = int(rng.integers(0, 151))
        j = l + 0.5 * int(rng.choice([-1, 1]))
        if j < 0:
            continue
        lp = l + int(rng.choice([-1, 1]))
        if lp < 0 or lp > 150:
            continue
        jp = lp + 0.5 * int(rng.choice([-1, 1]))
        if jp < 0 or abs(j - jp) > 1:
            continue
        args = (l, j, 0.5, jp, lp, 1.0)
        ours = wigner_6j(*args)
        sign, sq = wigner_6j_exact(*args)
        exact = _oracle_val(sign, sq)
        if exact == 0.0:
            assert abs(ours) < 1e-12
        else:
            assert ours == pytest.approx(exact, rel=1e-9)
        checked += 1


def test_property_generic_3j_vs_oracle():
    """Spec 03 SS6 property test (c): 3000 random generic 3j restricted to
    j <= 20 agree with the oracle to 1e-9 relative (measured 2.4e-11). Per
    SS4.3.4 this tolerance must NOT be extended to generic j <= 60."""
    rng = np.random.default_rng(_SEED + 2)
    checked = 0
    while checked < 3000:
        t1 = int(rng.integers(0, 41))
        t2 = int(rng.integers(0, 41))
        t3 = int(rng.integers(abs(t1 - t2), t1 + t2 + 1))
        if (t1 + t2 + t3) % 2:
            continue
        u1 = t1 - 2 * int(rng.integers(0, t1 + 1)) if t1 else 0
        u2 = t2 - 2 * int(rng.integers(0, t2 + 1)) if t2 else 0
        u3 = -u1 - u2
        if abs(u3) > t3:
            continue
        args = tuple(x / 2 for x in (t1, t2, t3, u1, u2, u3))
        ours = wigner_3j(*args)
        sign, sq = wigner_3j_exact(*args)
        exact = _oracle_val(sign, sq)
        if exact == 0.0:
            assert abs(ours) < 1e-12
        else:
            assert ours == pytest.approx(exact, rel=1e-9)
        checked += 1


def test_property_3j_symmetries():
    """Spec 03 SS6: m -> -m symmetry 3j(-m) = (-1)^(j1+j2+j3) 3j(m), even
    (cyclic) column permutation invariance, odd permutation phase."""
    rng = np.random.default_rng(_SEED + 3)
    checked = 0
    while checked < 300:
        t1 = int(rng.integers(0, 21))
        t2 = int(rng.integers(0, 21))
        t3 = int(rng.integers(abs(t1 - t2), t1 + t2 + 1))
        if (t1 + t2 + t3) % 2:
            continue
        u1 = t1 - 2 * int(rng.integers(0, t1 + 1)) if t1 else 0
        u2 = t2 - 2 * int(rng.integers(0, t2 + 1)) if t2 else 0
        u3 = -u1 - u2
        if abs(u3) > t3:
            continue
        j1, j2, j3 = t1 / 2, t2 / 2, t3 / 2
        m1, m2, m3 = u1 / 2, u2 / 2, u3 / 2
        base = wigner_3j(j1, j2, j3, m1, m2, m3)
        ph = (-1.0) ** ((t1 + t2 + t3) // 2)
        assert wigner_3j(j1, j2, j3, -m1, -m2, -m3) == pytest.approx(ph * base, abs=1e-13)
        assert wigner_3j(j2, j3, j1, m2, m3, m1) == pytest.approx(base, abs=1e-13)
        assert wigner_3j(j2, j1, j3, m2, m1, m3) == pytest.approx(ph * base, abs=1e-13)
        checked += 1


# ---------------------------------------------------------------------------
# Import smoke: the module's public API is complete (spec 03 SS5)
# ---------------------------------------------------------------------------

def test_public_api_surface():
    """Every function of the spec 03 SS5 API exists and is importable."""
    import rydsim.angular as ang
    for name in ("FineState", "wigner_3j", "wigner_6j", "reduced_C1",
                 "reduced_dipole_j", "angular_factor", "reduced_dipole_F",
                 "line_strength_S", "dipole_matrix_element",
                 "effective_rf_dipole", "steck_to_racah", "racah_to_steck",
                 "einstein_A", "oscillator_strength", "oscillator_strength_l",
                 "trk_sum", "effective_probe_dipole_far"):
        assert hasattr(ang, name), name
