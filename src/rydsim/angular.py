"""Angular algebra and dipole matrix elements (spec 03).

Wigner-Eckart chain in the fine-structure basis |n l j m_j>, reduced dipole
elements, m_j-resolved angular factors, hyperfine line strengths, oscillator
strengths and Einstein A coefficients — the single implementation of the
angular layer (spec 00 lock #9: no other module reimplements 3j/6j/phases).

Conventions (normative, docs/spec/00-conventions.md lock #9 + spec 03 SS2.1):

* Condon-Shortley phases throughout (Edmonds 1957; same as ``rydsim.wigner``).
* Internal Wigner-Eckart is the **Racah** convention, spec 03 Eq. (2.1):
  ``<j m|T^k_q|j' m'> = (-1)^(j-m) (j k j'; -m q m') <j||T^k||j'>``.
* Steck-datasheet reduced elements convert via spec 03 Eq. (2.3):
  ``<j||er||j'>_Racah = sqrt(2j+1) * <j||er||j'>_Steck`` (j = bra label).
* Signed reduced elements and radial-integral signs are convention-relative;
  only |element|^2 is comparable across codes (spec 03 SS4.3.8).
* SI units internally; radial integrals cross this API in units of a0
  (spec 00 lock #1 — the rydsim.radial exception ends at its boundary).
* All frequencies angular [rad/s] (spec 00 lock #2).

Float-path certification (spec 03 SS4.3.4, integrity audit R24 / refusal #13):
``rydsim.wigner`` no longer ships a bare log-factorial sum — it evaluates the
Racah sum together with its own round-off estimate and falls back to an
**exact-rational** evaluation whenever that estimate exceeds 1e-12, so every
3j/6j is accurate to 1e-12 relative by construction, at any j and any rank
(``rydsim.wigner`` module docstring). Measured against the exact-rational
oracle by tests/test_wigner.py, which prints the figures rather than
transcribing them: worst 6.6e-14 over the stratified rank-1 sweep to j = 150
(including the Delta j = 0, |m| = 1/2 corner where the bare float path is
2.9e-11), 5.4e-14 over the generic 3j grid and 2.9e-14 over the generic 6j
grid — grids on which the bare float path returned values up to 372x outside
the elementary |3j| bound. Refusal #13's "route to the exact-rational oracle"
is therefore taken automatically; :func:`wigner_3j_exact` /
:func:`wigner_6j_exact` expose that route directly.

Independently of accuracy, every public function here — including the
:func:`wigner_3j` / :func:`wigner_6j` wrappers — guards its angular momenta at
j <= 150 (:data:`_J_FLOAT_CERT_MAX`) and raises
``rydsim.provenance.IntegrityError`` beyond it: no physical state in this
simulator carries j > 150, so a request past the ceiling is a caller bug, and
the exact oracle is the explicit opt-in route for it.

Sources for the numeric contracts baked into docstrings/tests:
- D. A. Steck, Rb-87/Rb-85/Cs D-line data, rev 2.3.4 (8 Aug 2025)  [VERIFIED,
  spec 03 SS3.2-3.3; re-verified against the primary PDFs 2026-08-10].
- M. T. Simons, J. A. Gordon, C. L. Holloway, J. Appl. Phys. 120, 123103
  (2016) — NIST effective-dipole A factors (sqrt(2)/3, sqrt(6)/5, 2 sqrt(3)/7)
  [VERIFIED, primary PDF, spec 03 SS2.5].
- A. R. Edmonds, Angular Momentum in Quantum Mechanics (1957), Eq. 7.1.7
  (spectator-spin reduction) [VERIFIED against dual implementations].
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .constants import A0, C, E_CHARGE, EPS0, HBAR, M_E
from .provenance import IntegrityError
from .wigner import wigner_3j as _w3j_kernel
from .wigner import wigner_6j as _w6j_kernel
from .wigner import wigner_3j_exact, wigner_6j_exact, exact_to_float

__all__ = [
    "FineState",
    "wigner_3j",
    "wigner_6j",
    "wigner_3j_exact",
    "wigner_6j_exact",
    "exact_to_float",
    "reduced_C1",
    "reduced_dipole_j",
    "angular_factor",
    "reduced_dipole_F",
    "line_strength_S",
    "dipole_matrix_element",
    "effective_rf_dipole",
    "effective_probe_dipole_far",
    "steck_to_racah",
    "racah_to_steck",
    "einstein_A",
    "oscillator_strength",
    "oscillator_strength_l",
    "trk_sum",
]

# Float-path certification ceiling (spec 03 SS4.3.4; audit refusal rule #13).
_J_FLOAT_CERT_MAX = 150.0


def _twice(x: float, name: str) -> int:
    """Angular momentum -> doubled integer; raise on non-(half-)integer.

    Audit refusal rule #12: no rounding of invalid j/m inputs.
    """
    t = round(2 * x)
    if abs(2 * x - t) > 1e-9:
        raise ValueError(f"{name}={x} is not integer or half-integer")
    return int(t)


def _neg_one_pow(twice_exponent: int, context: str) -> float:
    """(-1)^k for an integer exponent supplied doubled (spec 03 SS4.3.1).

    Every phase exponent in the chain is provably integer when the selection
    rules hold; a half-integer exponent here means the caller leaked an
    unphysical (l, s, j) combination past the symbol checks.
    """
    if twice_exponent % 2:
        raise IntegrityError(
            f"non-integer phase exponent in {context}: unphysical quantum "
            "numbers reached a phase evaluation (spec 03 SS4.3.1)"
        )
    return -1.0 if (twice_exponent // 2) % 2 else 1.0


def _require_certified(context: str, **js: float) -> None:
    """Refuse angular momenta beyond the certified float range (j > 150)."""
    for name, j in js.items():
        if abs(j) > _J_FLOAT_CERT_MAX:
            raise IntegrityError(
                f"{context}: {name}={j} exceeds the certified float-path range "
                f"j <= {_J_FLOAT_CERT_MAX:g} (spec 03 SS4.3.4 / audit refusal "
                "#13) — route through the exact-rational oracle "
                "rydsim.angular.wigner_3j_exact / wigner_6j_exact "
                "(+ exact_to_float), which is exact at any j"
            )


def wigner_3j(j1: float, j2: float, j3: float,
              m1: float, m2: float, m3: float) -> float:
    """Certified 3j: :func:`rydsim.wigner.wigner_3j` behind the j <= 150 gate.

    The spec 03 SS5 public entry point. The kernel is accurate to 1e-12
    relative at any j (it routes to the exact-rational path when the float sum
    cannot certify itself, and refuses any value outside the elementary
    ``|3j| <= 1/sqrt(2 j_max + 1)`` bound), but this module's declared domain
    stops at j = 150 — beyond it callers must ask for
    :func:`wigner_3j_exact` explicitly rather than receive a number the
    module docstring does not cover. Audit refusal #13 / spec 03 SS4.3.4.
    """
    _require_certified("wigner_3j", j1=j1, j2=j2, j3=j3)
    return _w3j_kernel(j1, j2, j3, m1, m2, m3)


def wigner_6j(j1: float, j2: float, j3: float,
              j4: float, j5: float, j6: float) -> float:
    """Certified 6j: :func:`rydsim.wigner.wigner_6j` behind the j <= 150 gate.
    See :func:`wigner_3j`."""
    _require_certified("wigner_6j", j1=j1, j2=j2, j3=j3, j4=j4, j5=j5, j6=j6)
    return _w6j_kernel(j1, j2, j3, j4, j5, j6)


@dataclass(frozen=True, slots=True)
class FineState:
    """|n l j> fine-structure state (spec 03 SS5). s = 1/2 implicit.

    Validates 0 <= l < n and j = l +- 1/2 (j = 1/2 for l = 0); j half-integer.
    Dimensionless quantum numbers.
    """

    n: int
    l: int
    j: float

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError(f"n={self.n} must be a positive integer")
        if not 0 <= self.l < self.n:
            raise ValueError(f"l={self.l} outside 0 <= l < n={self.n}")
        tj = _twice(self.j, "j")
        if tj % 2 == 0:
            raise ValueError(f"j={self.j} must be half-integer for s=1/2")
        if abs(tj - 2 * self.l) != 1:
            raise ValueError(f"j={self.j} incompatible with l={self.l} (j = l +- 1/2)")


def reduced_C1(l: int, lp: int) -> float:
    """<l||C^(1)||l'> = (-1)^l sqrt((2l+1)(2l'+1)) (l 1 l'; 0 0 0). Eq. (2.7).

    Dimensionless. Exact closed forms (Eq. 2.8, verified to 3e-16):
    -sqrt(l+1) for l' = l+1, +sqrt(l) for l' = l-1, exact 0.0 otherwise.
    Raises ValueError for negative or non-integer orbital quantum numbers.
    """
    if l != int(l) or lp != int(lp) or l < 0 or lp < 0:
        raise ValueError(f"orbital quantum numbers must be integers >= 0, got l={l}, l'={lp}")
    _require_certified("reduced_C1", l=l, lp=lp)
    three = wigner_3j(l, 1, lp, 0, 0, 0)
    if three == 0.0:
        return 0.0
    phase = _neg_one_pow(2 * int(l), "Eq 2.7")
    return phase * math.sqrt((2 * l + 1) * (2 * lp + 1)) * three


def reduced_dipole_j(l: int, j: float, lp: int, jp: float, s: float = 0.5) -> float:
    """<l s j||er||l' s j'>_Racah / (e*R): Eq. (2.5) x (2.7), dimensionless.

    n-independent fine-structure reduction with the spin spectator (Edmonds
    7.1.7):  (-1)^(l+s+j'+1) sqrt((2j+1)(2j'+1)) {l j s; j' l' 1} <l||C1||l'>.
    D2 contract: reduced_dipole_j(0, 0.5, 1, 1.5) == +-2/sqrt(3).
    Exact 0.0 off the E1 selection rules (spec 03 SS2.4).
    """
    tl, tj = _twice(l, "l"), _twice(j, "j")
    tlp, tjp = _twice(lp, "lp"), _twice(jp, "jp")
    ts = _twice(s, "s")
    _require_certified("reduced_dipole_j", j=j, jp=jp)
    c1 = reduced_C1(l, lp)
    if c1 == 0.0:
        return 0.0
    six = wigner_6j(l, j, s, jp, lp, 1)
    if six == 0.0:
        return 0.0
    phase = _neg_one_pow(tl + ts + tjp + 2, "Eq 2.5")
    return phase * math.sqrt((tj + 1.0) * (tjp + 1.0)) * six * c1


@lru_cache(maxsize=None)
def _angular_factor_cached(tl: int, tj: int, tm: int, tlp: int, tjp: int,
                           tmp: int, q: int, ts: int) -> float:
    """Cached core of Eq. (2.9) on doubled-integer quantum numbers."""
    j, m = tj / 2.0, tm / 2.0
    jp, mp = tjp / 2.0, tmp / 2.0
    three = wigner_3j(j, 1, jp, -m, q, mp)
    if three == 0.0:
        return 0.0
    red = reduced_dipole_j(tl // 2, j, tlp // 2, jp, ts / 2.0)
    if red == 0.0:
        return 0.0
    phase = _neg_one_pow(tj - tm, "Eq 2.4")
    return phase * three * red


def angular_factor(l: int, j: float, mj: float, lp: int, jp: float, mjp: float,
                   q: int, s: float = 0.5) -> float:
    """Dimensionless angular factor A of Eq. (2.9), n-independent, cached.

    <n l j mj|e r_q|n' l' j' mj'> = e * R * A with R the radial integral
    (doc 02, per-(j,j') pair). q in {-1, 0, +1} = sigma-/pi/sigma+ (SS2.3);
    exact 0.0 off the E1 selection rules (SS2.4).
    Contract: abs(angular_factor(2, 2.5, 0.5, 1, 1.5, 0.5, 0)) == sqrt(6)/5
    (NIST 0.4899, Simons/Gordon/Holloway JAP 120, 123103 (2016)).
    """
    if q not in (-1, 0, 1):
        raise ValueError(f"polarization index q={q} must be -1, 0 or +1 (spec 03 SS2.3)")
    if l != int(l) or lp != int(lp) or l < 0 or lp < 0:
        raise ValueError(f"orbital quantum numbers must be integers >= 0, got l={l}, l'={lp}")
    _require_certified("angular_factor", j=j, jp=jp)
    return _angular_factor_cached(
        2 * int(l), _twice(j, "j"), _twice(mj, "mj"),
        2 * int(lp), _twice(jp, "jp"), _twice(mjp, "mjp"),
        int(q), _twice(s, "s"),
    )


def reduced_dipole_F(J: float, Jp: float, F: float, Fp: float, I: float) -> float:
    """<F||er||F'>_Racah / <J||er||J'>_Racah: Eq. (2.12), dimensionless.

    Hyperfine reduction with the nuclear spin I as spectator:
    (-1)^(J+I+F'+1) sqrt((2F+1)(2F'+1)) {J F I; F' J' 1}.
    Exact 0.0 off the F-selection rules (spec 03 SS2.4).
    """
    tJ, tI, tFp = _twice(J, "J"), _twice(I, "I"), _twice(Fp, "Fp")
    tF = _twice(F, "F")
    _twice(Jp, "Jp")
    _require_certified("reduced_dipole_F", J=J, Jp=Jp, F=F, Fp=Fp, I=I)
    six = wigner_6j(J, F, I, Fp, Jp, 1)
    if six == 0.0:
        return 0.0
    phase = _neg_one_pow(tJ + tI + tFp + 2, "Eq 2.12")
    return phase * math.sqrt((tF + 1.0) * (tFp + 1.0)) * six


def line_strength_S(J: float, Jp: float, F: float, Fp: float, I: float) -> float:
    """Hyperfine line-strength factor S_FF' of Eq. (2.15), dimensionless.

    S_FF' = (2F'+1)(2J+1) {J J' 1; F' F I}^2, with sum rule
    Sigma_F' S_FF' = 1 (contract: within 1e-12). Convention-free ratio —
    identical in Racah and Steck conventions. Steck Eq. (41) [VERIFIED,
    rev 2.3.4]; Rb-87 D2 F=2 -> F'=1,2,3 gives exactly 1/20, 1/4, 7/10.
    """
    tJ, tFp = _twice(J, "J"), _twice(Fp, "Fp")
    _twice(Jp, "Jp"), _twice(F, "F"), _twice(I, "I")
    _require_certified("line_strength_S", J=J, Jp=Jp, F=F, Fp=Fp, I=I)
    six = wigner_6j(J, Jp, 1, Fp, F, I)
    return (tFp + 1.0) * (tJ + 1.0) * six * six


def dipole_matrix_element(bra: FineState, ket: FineState, mj: float, mjp: float,
                          q: int, radial_integral_a0: float) -> float:
    """Full <bra mj|e r_q|ket mjp> in C*m (spec 03 SS5).

    radial_integral_a0 = R_{n l j}^{n' l' j'} in units of a0 (doc 02 boundary,
    spec 00 lock #1). Returns e * a0 * R * A(l,j,mj -> l',j',mj'; q); the sign
    is convention-relative (SS4.3.8) — only |value|^2 is comparable across codes.
    """
    A = angular_factor(bra.l, bra.j, mj, ket.l, ket.j, mjp, q)
    return E_CHARGE * A0 * radial_integral_a0 * A


def effective_rf_dipole(l: int, j: float, lp: int, jp: float,
                        radial_integral_a0: float,
                        mj: float = 0.5, q: int = 0) -> float:
    """NIST-convention effective RF dipole |p| = e*a0*|R|*|A| in C*m.

    Spec 03 SS2.5 / spec 00 lock #11: co-linear pi geometry, m_j = +-1/2,
    p = e a0 R |A| for the drive ``(l, j, mj) --q--> (l', j', mj + q)``.
    Defaults (mj = 1/2, q = 0) reproduce the Simons/Gordon/Holloway JAP 120,
    123103 (2016) A factors (sqrt(2)/3, sqrt(6)/5, 2 sqrt(3)/7).

    Polarization bookkeeping (this is the sign trap): :func:`angular_factor`
    evaluates ``<bra|e r_q|ket>`` through the 3j ``(j 1 j'; -m_bra q m_ket)``,
    which enforces ``m_bra = m_ket + q``. The state being *driven into* is
    therefore the BRA, so the element is
    ``angular_factor(l', j', mj + q, l, j, mj, q)`` — not
    ``angular_factor(l, j, mj, l', j', mj + q, q)``, which imposes the
    opposite sign and makes every q = +-1 element vanish identically. The
    magnitude is polarization-symmetric, |<a|e r_q|b>| = |<b|e r_-q|a>|, so
    this agrees with ``rydsim.dipoles._rf_angular``'s stretched-convention
    call ``angular_factor(l, j, j, l', j', j', j - j')`` on the same pair.

    Lock #11 fixes (mj, q) = (+-1/2, 0); any other (mj, q) is a documented
    deviation the caller owns — legitimate for the stretched convention
    (``mj = j``, ``q = j' - j``) and for assembling the coherent multi-q sum
    that SS2.3 requires for non-collinear/elliptical RF, but it is NOT the
    normative single-scalar wp and must be stamped as such downstream
    (``rydsim.dipoles.mu_rf`` carries the convention field for that).

    Raises IntegrityError for an E1-forbidden pair rather than returning a 0
    dipole that would silently invert to an infinite field.
    """
    A = angular_factor(lp, jp, mj + q, l, j, mj, q)
    if A == 0.0:
        raise IntegrityError(
            f"effective_rf_dipole: transition (l={l}, j={j}, mj={mj}) -> "
            f"(l'={lp}, j'={jp}, mj'={mj + q}) with q={q} is E1-forbidden; "
            "refusing to return a zero effective dipole (spec 03 SS2.4-2.5)"
        )
    return E_CHARGE * A0 * abs(radial_integral_a0) * abs(A)


def effective_probe_dipole_far(d_reduced_steck: float) -> float:
    """Far-detuned hot-vapor probe dipole d_eff = <J||er||J'>_Steck / sqrt(3).

    Spec 03 Eq. (2.16) / spec 00 interface-gap closure #7: pi-polarized light,
    Doppler-unresolved excited hyperfine structure; exactly 1/3 of the Steck
    line strength per ground |F m_F>, independent of m_F (Steck Eq. (44),
    rev 2.3.4, VERIFIED). Unit-transparent (input e*a0 -> output e*a0; input
    C*m -> output C*m). Rb-87 D2: 4.22752/sqrt(3) = 2.44076 e*a0. Used for
    BOTH Omega_p and the chi prefactor (spec 00 SS6.7).
    """
    return d_reduced_steck / math.sqrt(3.0)


def steck_to_racah(d_steck: float, j_lower: float) -> float:
    """<j||er||j'>_Racah = sqrt(2 j_lower + 1) * <j||er||j'>_Steck. Eq. (2.3).

    j_lower is the bra (first) label of the reduced element. Unit-transparent.
    The single most common factor-of-sqrt(2) bug in this field (spec 03 SS2.1).
    """
    tj = _twice(j_lower, "j_lower")
    return math.sqrt(tj + 1.0) * d_steck


def racah_to_steck(d_racah: float, j_lower: float) -> float:
    """Inverse of :func:`steck_to_racah`: divide by sqrt(2 j_lower + 1)."""
    tj = _twice(j_lower, "j_lower")
    return d_racah / math.sqrt(tj + 1.0)


def einstein_A(omega0, d_reduced_racah_Cm, j_upper: float):
    """Einstein A coefficient [1/s], Eq. (2.20), Racah convention.

    A_(e->g) = omega0^3 |<g||er||e>_Racah|^2 / (3 pi eps0 hbar c^3 (2 j_e + 1))
    with (2 j_e + 1) the UPPER-state degeneracy (spec 00 symbol table, owner
    03/04). omega0 in rad/s (angular, lock #2), d in C*m. Vectorized over
    omega0 / d (broadcasting); scalar in -> scalar out.
    Identity check (spec 03 B18/B22): reproduces tau(Rb-87 5P3/2) = 26.2348 ns
    from d_S = 4.22752 e*a0 through Eq. (2.3).
    """
    tj = _twice(j_upper, "j_upper")
    if tj < 0:
        raise ValueError(f"j_upper={j_upper} must be >= 0")
    omega = np.asarray(omega0, dtype=float)
    d = np.asarray(d_reduced_racah_Cm, dtype=float)
    out = omega**3 * d**2 / (3.0 * np.pi * EPS0 * HBAR * C**3 * (tj + 1.0))
    return out.item() if out.ndim == 0 else out


def oscillator_strength(omega, d_reduced_racah_Cm, j_initial: float):
    """Oscillator strength f_(i->f) (dimensionless), Eq. (2.17).

    ``f_(i->f) = 2 m_e omega_fi |<i||er||f>_Racah|^2 / (3 hbar e^2 (2 j_i + 1))``
    with ``omega_fi = (E_f - E_i)/hbar`` in rad/s and ``2 j_i + 1 = g_i`` the
    degeneracy of the **INITIAL** state. d in C*m. Vectorized over omega / d.

    Direction (the factor-of-g trap): omega > 0 is absorption, so ``j_initial``
    is the lower level; omega < 0 is emission, and then ``j_initial`` is the
    **upper** level. Passing the lower-state j with a negative omega overstates
    |f| by g_upper/g_lower — exactly 2x on any D2 line — and is the standard
    way this function is misused. The Racah reduced element's *magnitude* is
    direction-symmetric (``|<i||er||f>|^2 = S``, the line strength, by the
    3j normalization sum), so only the degeneracy label distinguishes the two
    directions; the pair satisfies ``g_i f_(i->f) = -g_f f_(f->i)``.

    Cross-check (spec 03 SS2.7, regression-tested against the same module's
    :func:`einstein_A`): ``f_(e->g) = -2 pi eps0 m_e c^3 A_(e->g)/(e^2 omega^2)``.
    """
    tj = _twice(j_initial, "j_initial")
    if tj < 0:
        raise ValueError(f"j_initial={j_initial} must be >= 0")
    omega_arr = np.asarray(omega, dtype=float)
    d = np.asarray(d_reduced_racah_Cm, dtype=float)
    out = 2.0 * M_E * omega_arr * d**2 / (3.0 * HBAR * (tj + 1.0) * E_CHARGE**2)
    return out.item() if out.ndim == 0 else out


def oscillator_strength_l(omega, radial_integral_a0, l: int, lp: int):
    """Spin-free l-basis oscillator strength, Eq. (2.18) (hydrogenic/TRK).

    f = (2 m_e omega / 3 hbar) * max(l, l')/(2l+1) * (R a0)^2 with the radial
    integral in a0 and omega in rad/s. Consistency: Sigma_j' f(j -> j') from
    Eq. (2.17) at equal radial integrals must equal this (spec 03 SS2.7).
    Vectorized over omega / radial_integral_a0.
    """
    if l != int(l) or lp != int(lp) or l < 0 or lp < 0:
        raise ValueError(f"orbital quantum numbers must be integers >= 0, got l={l}, l'={lp}")
    if abs(l - lp) != 1:
        raise ValueError(f"Eq. (2.18) requires |l - l'| = 1, got l={l}, l'={lp}")
    omega_arr = np.asarray(omega, dtype=float)
    R_m = np.asarray(radial_integral_a0, dtype=float) * A0
    out = 2.0 * M_E * omega_arr / (3.0 * HBAR) * (max(l, lp) / (2.0 * l + 1.0)) * R_m**2
    return out.item() if out.ndim == 0 else out


def trk_sum(f_values: np.ndarray) -> float:
    """Sum of oscillator strengths (TRK helper, Eq. (2.19)). Dimensionless.

    Caller supplies the discrete + continuum set; downward transitions enter
    with negative f. Every term must be an ``f_(i->f)`` computed from the SAME
    initial state i, i.e. :func:`oscillator_strength` called with that state's
    j (see its "direction" note) — a downward channel evaluated with the
    lower level's degeneracy is wrong by g_upper/g_lower and does not cancel
    across the sum. Exact TRK (= 1 to 0.5%) applies only to the hydrogenic
    mode; alkali valence sums land in the sanity band 0.95-1.10 (spec 03 SS2.7).
    """
    return float(np.sum(np.asarray(f_values, dtype=float)))
