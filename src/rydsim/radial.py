"""Rydberg radial wavefunctions and matrix elements (spec 02).

Owns (docs/spec/00-conventions.md §4): radial wavefunctions R_nlj(r) and
radial matrix elements <n l j| r^k |n' l' j'> for Rb-85/Rb-87/Cs-133 (and the
hydrogenic limit used by the benchmarks). THE sanctioned exception to unit
lock #1: this module computes in Hartree atomic units internally (length a0,
energy E_h, hbar = e = m_e = 1) and returns radial integrals in a0 at its API
boundary — the caller multiplies by ``rydsim.constants.A0`` for metres.

Three independent methods (spec 02 §2):
  A. ``model_potential`` — Numerov integration of the Marinescu-Sadeghpour-
     Dalgarno (1994) l-dependent model potential + spin-orbit term, in the
     x = sqrt(r) coordinate with X(x) = R(r) r^(3/4), integrated inward from
     r_o = 2n(n+15) a0 with the ARC-style divergence guard (spec 02 §4.2).
  B. ``coulomb`` — single-channel QDT / Coulomb approximation: the same
     Numerov machinery with V = -1/r exactly (model-parameter-free). The
     Whittaker/hyperu closed form (``whittaker_u``) is the pointwise
     cross-check for nu <= 20 only — scipy.special.hyperu loses precision
     catastrophically at large nu (audit R4; re-measured on the installed
     scipy at test time) and additionally returns NaN for non-integer
     nu >~ 15 beyond r ~ 0.2-0.5 nu^2 (measured, scipy 1.17.1).
  C. ``kaulakys`` — Kaulakys, J. Phys. B 28, 4963 (1995) quasiclassical
     Anger-function formula (eqs. 19, 21-24, 30-31; full text in-repo at
     ``kaulakys_text.txt``, spot-verified by audit item 14). k = 1 only.

House rule (spec 02 §1): no single-method number ships. The public entry
point for alkali matrix elements is ``radial_matrix_element_consensus``,
which runs every available method, reports the spread as the numerical
uncertainty, refuses (IntegrityError) when fewer than two methods are
available, and refuses when the spread exceeds the per-regime ceilings of
spec 02 §6 (audit §3 item 8: no consensus => raise, don't average).

Known limitations carried verbatim from spec 02 §7: low-n dipoles are
~5-10 % wrong (Rb 5S->5P3/2 computes 5.57 a0 vs ~5.18 a0 experiment-derived,
a documented +8 % bias — D-line dipoles must come from experiment, docs
03/04); wavefunctions are unreliable inside r_cut; contact-type observables
(k < 0) are refused (audit §3 item 11).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.integrate import quad
from scipy.special import eval_genlaguerre, hyperu

from .atom import Species, n_star
from .constants import ALPHA_FS, AMU, M_E
from .numerov import numerov_inward
from .provenance import IntegrityError

__all__ = [
    "ModelPotentialParams", "RB_MODEL_POTENTIAL", "CS_MODEL_POTENTIAL",
    "A4_L1_NOTE", "effective_charge", "model_potential",
    "RadialSolution", "radial_wavefunction", "coulomb_wavefunction",
    "radial_matrix_element", "radial_me_gordon", "radial_me_kaulakys",
    "whittaker_u", "hyperu_hydrogen_error", "RadialMEResult",
    "radial_matrix_element_consensus", "clear_radial_cache",
]

# ---------------------------------------------------------------------------
# MSD94 model-potential parameter tables (spec 02 §3.1-3.3)
# ---------------------------------------------------------------------------

#: Spec 02 §3.3 disclosure — the ONE transcription discrepancy in the MSD94
#: tables: Rb a4(l=1) is -0.81633314 in pairinteraction/ryd-numerov (8
#: significant digits, consistent formatting — ADOPTED) but -0.8163314 in ARC
#: master (7 digits). Confidence of the last digit: UNVERIFIED (two secondary
#: sources disagree; primary PRA 49, 982 Table I unfetched). Impact bound:
#: delta-a4 = 8e-8 perturbs Z_1(r) by < 4e-8 -> far below every §6 tolerance.
A4_L1_NOTE = (
    "Rb a4(l=1): adopted -0.81633314 (ryd-numerov); ARC reads -0.8163314; "
    "last digit UNVERIFIED, effect on MEs < 4e-8 (spec 02 §3.3 / audit R20)"
)

_MSD94_SOURCE = (
    "Marinescu, Sadeghpour & Dalgarno, PRA 49, 982 (1994) Table I, via two "
    "independent transcriptions (ARC master + pairinteraction/ryd-numerov, "
    "diffed 2026-08-10) [VERIFIED cross-transcription, except " + A4_L1_NOTE + "]"
)


@dataclass(frozen=True)
class ModelPotentialParams:
    """MSD94 parameters for one species (spec 02 §3). All atomic units.

    Tuple index = min(l, 3); l >= 4 states use the pure Coulomb potential
    (ARC convention, spec 02 §2.1) so the l = 3 row is never applied there.
    """

    z: int                                  # nuclear charge (Rb 37, Cs 55)
    alpha_c: float                          # core polarizability [a0^3]
    a1: tuple[float, float, float, float]
    a2: tuple[float, float, float, float]
    a3: tuple[float, float, float, float]
    a4: tuple[float, float, float, float]
    r_c: tuple[float, float, float, float]
    source: str

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("ModelPotentialParams requires a source tag")


RB_MODEL_POTENTIAL = ModelPotentialParams(
    z=37, alpha_c=9.0760,
    a1=(3.69628474, 4.44088978, 3.78717363, 2.39848933),
    a2=(1.64915255, 1.92828831, 1.57027864, 1.76810544),
    a3=(-9.86069196, -16.79597770, -11.65588970, -12.07106780),
    a4=(0.19579987, -0.81633314, 0.52942835, 0.77256589),
    r_c=(1.66242117, 1.50195124, 4.86851938, 4.79831327),
    source="Rb (z=37, both isotopes): " + _MSD94_SOURCE)

CS_MODEL_POTENTIAL = ModelPotentialParams(
    z=55, alpha_c=15.6440,
    a1=(3.49546309, 4.69366096, 4.32466196, 3.01048361),
    a2=(1.47533800, 1.71398344, 1.61365288, 1.40000001),
    a3=(-9.72143084, -24.65624280, -6.70128850, -3.20036138),
    a4=(0.02629242, -0.09543125, -0.74095193, 0.00034538),
    r_c=(1.92046930, 2.13383095, 0.93007296, 1.99969677),
    source="Cs (z=55): " + _MSD94_SOURCE + " (all Cs entries agree digit-for-digit)")


def _potential_params_for(sp: Species) -> ModelPotentialParams:
    """MSD94 table for a species; wiring cross-checked against spec 01 alpha_d."""
    if sp.name.startswith("Rb"):
        p = RB_MODEL_POTENTIAL
    elif sp.name.startswith("Cs"):
        p = CS_MODEL_POTENTIAL
    else:
        raise IntegrityError(
            f"no MSD94 model-potential parameters for species {sp.name!r}; "
            "refusing to guess (spec 02 §3)")
    if abs(p.alpha_c - sp.alpha_core_au) > 1e-3:
        raise IntegrityError(
            f"model-potential alpha_c = {p.alpha_c} disagrees with spec 01 "
            f"alpha_core_au = {sp.alpha_core_au} for {sp.name}: table wiring error")
    return p


def effective_charge(p: ModelPotentialParams, l: int,
                     r: np.ndarray) -> np.ndarray:
    """Z_l(r), spec 02 Eq. (2.3) [dimensionless]; r in a0, vectorized.

    Z_l(0) = z, Z_l(inf) = 1. l >= 4 returns ones (pure Coulomb regime,
    spec 02 §2.1).
    """
    r = np.asarray(r, dtype=float)
    if l >= 4:
        return np.ones_like(r)
    i = min(l, 3)
    return (1.0 + (p.z - 1.0) * np.exp(-p.a1[i] * r)
            - r * (p.a3[i] + p.a4[i] * r) * np.exp(-p.a2[i] * r))


def model_potential(p: ModelPotentialParams, l: int, j: float, r: np.ndarray,
                    *, include_so: bool = True) -> np.ndarray:
    """V_lj(r) [Hartree], spec 02 Eqs. (2.2) + (2.4); r in a0, r > 0.

    l <= 3: MSD94 core potential -Z_l(r)/r - (alpha_c/2r^4)(1 - e^-(r/r_c)^6);
    l >= 4: pure Coulomb -1/r (ARC convention). Spin-orbit (l > 0, s = 1/2):
    (alpha_fs^2/2r^3) [j(j+1) - l(l+1) - 3/4]/2 — hydrogenic-limit form; its
    r^-3 divergence sits inside the inner cutoff (spec 02 §2.1). Measured
    effect on Rb 50S->50P3/2 ME: 2.7e-11 relative.
    """
    r = np.asarray(r, dtype=float)
    if l >= 4:
        v = -1.0 / r
    else:
        i = min(l, 3)
        v = (-effective_charge(p, l, r) / r
             - p.alpha_c / (2.0 * r**4) * (1.0 - np.exp(-((r / p.r_c[i]) ** 6))))
    if include_so and l > 0:
        v = v + (ALPHA_FS**2 / (2.0 * r**3)) * (
            j * (j + 1) - l * (l + 1) - 0.75) / 2.0
    return v


# ---------------------------------------------------------------------------
# Numerov solution on the x = sqrt(r) grid (spec 02 §2.2-2.3, §4)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class RadialSolution:
    """Normalized radial state on the uniform x = sqrt(r) grid (spec 02 §5).

    x in sqrt(a0), increasing, uniform step; X(x) = R(r) r^(3/4) with the
    norm 2*trapz(X^2 x^2 dx) = 1 (spec 02 Eq. 2.6); X is zeroed inside the
    divergence-guard truncation radius r_cut [a0]. Arrays are read-only
    (instances are cached). Sign convention: outer tail positive.
    """

    x: np.ndarray
    X: np.ndarray
    n: int
    l: int
    j: float
    nu: float
    r_cut: float
    method: str            # "model_potential" | "coulomb"

    @property
    def r(self) -> np.ndarray:
        """r grid [a0] = x**2."""
        return self.x**2

    @property
    def u(self) -> np.ndarray:
        """u(r) = r R(r) = X sqrt(x) [a0^-1/2]."""
        return self.X * np.sqrt(self.x)


def _divergence_index(u_abs: np.ndarray) -> int | None:
    """ARC-style divergence guard on |u|, sweeping inward (spec 02 §4.2).

    Track the running max of |u| from the outer end inward; after 50
    consecutive steps without a new max, freeze it; the first later (further
    inward) point exceeding the frozen max is the divergence point. Returns
    the index (in increasing-x order) to truncate at, or None.
    """
    a = u_abs[::-1]
    npts = a.size
    prev_max = np.maximum.accumulate(np.concatenate(([0.0], a[:-1])))
    idx = np.arange(npts)
    last_new = np.maximum.accumulate(np.where(a > prev_max, idx, -1))
    frozen = (idx - last_new) > 50
    if not frozen.any():
        return None
    i_freeze = int(np.argmax(frozen))
    m_star = float(np.maximum.accumulate(a)[i_freeze])
    beyond = a[i_freeze + 1:] > m_star
    if not beyond.any():
        return None
    return npts - 1 - (i_freeze + 1 + int(np.argmax(beyond)))


def _solve_scaled(l: int, nu: float, mu_mass: float, potential_of_r,
                  h: float, r_inner: float, r_outer: float,
                  n: int, j: float, method: str) -> RadialSolution:
    """Shared Numerov engine: build the x-grid, integrate (2.5) inward, apply
    the divergence guard, unit-normalize (2.6). All a.u."""
    if nu <= l:
        raise IntegrityError(
            f"nu = {nu:.4f} <= l = {l}: no bound QDT orbital "
            "(spec 02 §5 / audit §3 item 9) — refusing")
    if r_inner <= 0 or r_outer <= r_inner:
        raise ValueError(f"need 0 < r_inner < r_outer, got {r_inner}, {r_outer}")
    x_in = math.sqrt(r_inner)
    x_out = math.sqrt(r_outer)
    x = x_in + h * np.arange(int(math.ceil((x_out - x_in) / h)) + 1)
    r = x * x
    energy = -mu_mass / (2.0 * nu * nu)
    g = (8.0 * mu_mass * x * x * (potential_of_r(r) - energy)
         + (2 * l + 0.5) * (2 * l + 1.5) / (x * x))
    X = numerov_inward(x, g, 1e-10, 1.2e-10)
    idx = _divergence_index(np.abs(X) * np.sqrt(x))
    r_cut = float(r[0])
    if idx is not None:
        X[:idx + 1] = 0.0
        r_cut = float(r[idx])
        e_orbit = math.sqrt(max(0.0, 1.0 - ((l + 0.5) / nu) ** 2))
        r1_classical = nu * nu * (1.0 - e_orbit)
        if r_cut > 1.05 * r1_classical:
            warnings.warn(
                f"divergence guard truncated at r_cut = {r_cut:.3f} a0, above "
                f"the inner classical turning point {r1_classical:.3f} a0 "
                f"(n={n}, l={l}, j={j}, nu={nu:.3f}): wavefunction unreliable "
                "well into the classical region (spec 02 §4.2)", stacklevel=3)
    norm2 = 2.0 * float(np.trapezoid(X * X * x * x, dx=h))
    if not (norm2 > 0.0 and math.isfinite(norm2)):
        raise IntegrityError(
            f"radial norm collapsed (norm^2 = {norm2}) for n={n}, l={l}, "
            f"nu={nu:.4f}: pathological truncation — refusing")
    X = X / math.sqrt(norm2)
    x.setflags(write=False)
    X.setflags(write=False)
    return RadialSolution(x=x, X=X, n=n, l=l, j=j, nu=nu,
                          r_cut=r_cut, method=method)


_SOLUTION_CACHE: dict[tuple, RadialSolution] = {}


def clear_radial_cache() -> None:
    """Drop all cached RadialSolutions (spec 02 §5 implementation note)."""
    _SOLUTION_CACHE.clear()


def _validate_lj(l: int, j: float) -> None:
    if l < 0 or int(l) != l:
        raise ValueError(f"l must be a non-negative integer, got {l!r}")
    if abs(abs(j - l) - 0.5) > 1e-9 or j <= 0:
        raise ValueError(f"j must equal l +- 1/2 (got l={l}, j={j})")


def radial_wavefunction(species: Species, n: int, l: int, j: float, *,
                        nu: float | None = None,
                        h: float = 1e-3,
                        r_inner: float | None = None,
                        r_outer: float | None = None,
                        include_so: bool = True,
                        method: str = "model_potential") -> RadialSolution:
    """Solve one radial state for an alkali species (spec 02 §5). All a.u.

    Method A (``method="model_potential"``): MSD94 potential + spin-orbit,
    E = -mu/(2 nu^2) Hartree with nu = n* from spec 01 (E is an input, never
    an eigenvalue — that is what makes inward integration + truncation
    correct). Method B (``method="coulomb"``): V = -1/r exactly, no model
    parameters, no SO — the QDT/Coulomb approximation evaluated stably (the
    production Method B, spec 02 §2.4).

    Defaults (spec 02 §4.1): h = 0.001 sqrt(a0); r_inner = alpha_c^(1/3)
    (Rb 2.0856 a0, Cs 2.5006 a0); r_outer = 2n(n+15) a0. Reduced mass:
    mu = 1/(1 + m_e/M_atom) (ruling R-10). nu defaults to spec 01's
    n_star(n, l, j), which RAISES below the per-species hard floor — low-n
    states require an explicit measured-energy nu (and their dipoles carry
    the documented +8 % model bias, spec 02 §7.1).

    Results are cached on (species, n, l, j, nu, grid, method); returned
    arrays are read-only. Raises IntegrityError if nu <= l; warns if the
    divergence guard truncated above 1.05x the inner classical turning point.
    """
    _validate_lj(l, j)
    if n < l + 1:
        raise ValueError(f"n = {n} < l + 1 = {l + 1}: no bound state")
    if method not in ("model_potential", "coulomb"):
        raise ValueError(f"unknown method {method!r}")
    if nu is None:
        nu = float(n_star(species, n, l, j))
    p = _potential_params_for(species)
    if r_inner is None:
        r_inner = p.alpha_c ** (1.0 / 3.0)
    if r_outer is None:
        r_outer = 2.0 * n * (n + 15.0)
    key = (species.name, n, l, j, method, h, round(nu, 10),
           round(r_inner, 12), round(r_outer, 6), include_so)
    sol = _SOLUTION_CACHE.get(key)
    if sol is not None:
        return sol
    mu_mass = 1.0 / (1.0 + M_E / (species.mass_u * AMU))
    if method == "model_potential":
        def pot(r: np.ndarray) -> np.ndarray:
            return model_potential(p, l, j, r, include_so=include_so)
    else:
        def pot(r: np.ndarray) -> np.ndarray:
            return -1.0 / r
    sol = _solve_scaled(l, nu, mu_mass, pot, h, r_inner, r_outer, n, j, method)
    _SOLUTION_CACHE[key] = sol
    return sol


def coulomb_wavefunction(nu: float, l: int, *,
                         mu_mass: float = 1.0,
                         h: float = 1e-3,
                         r_inner: float = 1e-2,
                         r_outer: float | None = None) -> RadialSolution:
    """Pure-Coulomb (V = -1/r) radial state for arbitrary real nu > l. All a.u.

    The species-independent engine behind Method B, exposed for the hydrogen
    benchmarks (spec 02 §6 B1-B7, B13-B14: nu = n integer, mu_mass = 1) and
    the Whittaker cross-check (B12, non-integer nu, mu_mass = 1). Defaults:
    r_inner = 1e-2 a0 (adequate for nu >= 5; low-n hydrogen benchmarks pass
    smaller values), r_outer = 2 nc (nc + 15) with nc = ceil(nu). Alkali
    Method B goes through ``radial_wavefunction(..., method="coulomb")``,
    which shares Method A's species grid.
    """
    if l < 0 or int(l) != l:
        raise ValueError(f"l must be a non-negative integer, got {l!r}")
    if r_outer is None:
        nc = max(1, int(math.ceil(nu)))
        r_outer = 2.0 * nc * (nc + 15.0)
    return _solve_scaled(l, float(nu), mu_mass, lambda r: -1.0 / r,
                         h, r_inner, r_outer,
                         n=int(math.ceil(nu)), j=l + 0.5, method="coulomb")


def radial_matrix_element(a: RadialSolution, b: RadialSolution,
                          k: int = 1) -> float:
    """<a| r^k |b> = 2 * trapz(Xa Xb x^(2k+2) dx) [a0^k], spec 02 Eq. (2.7).

    Requires both states on the same uniform x-grid (identical h and x[0]);
    grids may differ in outer extent — the integral runs over the shared
    index range (the excess tail is exponentially negligible by the outer-
    cutoff rule, spec 02 §4.1/B14). Sign is convention-laden (outer-tail-
    positive Numerov solutions); only |ME| is comparable across codes
    (spec 02 §2.6). k < 0 (contact-type, weighted at r <~ r_cut) is refused
    per audit §3 item 11.
    """
    if k < 0:
        raise IntegrityError(
            "k < 0 matrix elements are weighted at r <~ r_cut where the "
            "truncated wavefunction is unreliable — out of scope "
            "(spec 02 §7.2 / audit §3 item 11)")
    ha = a.x[1] - a.x[0]
    hb = b.x[1] - b.x[0]
    if abs(ha - hb) > 1e-12 * ha or abs(a.x[0] - b.x[0]) > 1e-9:
        raise ValueError(
            "states are on different grids (h or x[0] mismatch): generate "
            "both on the same grid, never interpolate (spec 02 §4.3)")
    m = min(a.X.size, b.X.size)
    integrand = a.X[:m] * b.X[:m] * a.x[:m] ** (2 * k + 2)
    return 2.0 * float(np.trapezoid(integrand, dx=float(ha)))


# ---------------------------------------------------------------------------
# Method C — Kaulakys (1995) quasiclassical formula (spec 02 §2.5)
# ---------------------------------------------------------------------------

def _anger_j(s: float, w: float) -> float:
    """Anger function J_{-s}(w) = (1/pi) int_0^pi cos(s xi + w sin xi) dxi
    (Kaulakys eq. 24; quad limit >= 400 per spec 02 §4.4 pitfall 5)."""
    val, _ = quad(lambda xi: math.cos(s * xi + w * math.sin(xi)),
                  0.0, math.pi, limit=400)
    return val / math.pi


def _anger_j_prime(s: float, w: float) -> float:
    """dJ_{-s}/dw = -(1/pi) int_0^pi sin xi sin(s xi + w sin xi) dxi."""
    val, _ = quad(lambda xi: math.sin(xi) * math.sin(s * xi + w * math.sin(xi)),
                  0.0, math.pi, limit=400)
    return -val / math.pi


def radial_me_kaulakys(nu1: float, l1: int, nu2: float, l2: int) -> float:
    """|<nu1 l1| r |nu2 l2>| [a0] by Kaulakys (1995) eqs. 19, 21-24 (k=1 only).

    s = nu2 - nu1; nu_c^3 = 2 (nu1 nu2)^2/(nu1 + nu2) (eq. 19); orbit
    eccentricity e = sqrt(1 - ((l1+l2+1)/(2 nu_c))^2) (eq. 17); D_p (eq. 21)
    with upper sign for l2 = l1+1; D_r = D_p + (1-e) sin(pi s)/(pi s)
    (eq. 23); R = nu_c^5/(nu1 nu2)^(3/2) D_r (eq. 22), magnitude returned
    (signs are convention-laden). |s| < 1e-4 branches to the exact s -> 0
    limit R = (3/2) e nu_c^5/(nu1 nu2)^(3/2) (eqs. 30-31; removes the 0/0)
    which reproduces the hydrogen closed form (3/2) n sqrt(n^2 - l^2)
    exactly at nu1 = nu2 = n. Symmetry D(e,-s) = D(e,s)-with-flipped-sign
    (eq. 23') means either state ordering is valid. Accuracy (spec 02 §2.5,
    re-measured this port): 4.4e-5 at nu=50 dn=1 vs Gordon; ~1e-3 at nu=10;
    degrades to ~2e-3 on cancellation-suppressed alkali MEs (B9).
    """
    if abs(l1 - l2) != 1:
        raise ValueError(f"Kaulakys requires |l1 - l2| = 1, got l1={l1}, l2={l2}")
    if nu1 <= l1 or nu2 <= l2:
        raise IntegrityError(
            f"nu <= l (nu1={nu1:.3f}, l1={l1}; nu2={nu2:.3f}, l2={l2}): "
            "no bound orbital — refusing")
    s = nu2 - nu1
    nu_c = (2.0 * (nu1 * nu2) ** 2 / (nu1 + nu2)) ** (1.0 / 3.0)
    ecc2 = 1.0 - ((l1 + l2 + 1) / (2.0 * nu_c)) ** 2
    if ecc2 <= 0.0:
        raise IntegrityError(
            f"(l1+l2+1)/(2 nu_c) >= 1 (nu_c={nu_c:.3f}): no classical orbit, "
            "quasiclassical formula invalid — refusing")
    ecc = math.sqrt(ecc2)
    pref = nu_c**5 / (nu1 * nu2) ** 1.5
    if abs(s) < 1e-4:
        return 1.5 * ecc * pref
    sinc_s = math.sin(math.pi * s) / (math.pi * s)
    upper = l2 == l1 + 1
    d_p = (_anger_j_prime(s, ecc * s)
           + (1.0 if upper else -1.0) * math.sqrt(1.0 / ecc2 - 1.0)
           * (_anger_j(s, ecc * s) - sinc_s)) / s
    d_r = d_p + (1.0 - ecc) * sinc_s
    return abs(pref * d_r)


# ---------------------------------------------------------------------------
# Exact hydrogenic limit — Gordon's formula (spec 02 §2.6)
# ---------------------------------------------------------------------------

def _hyp2f1_terminating(a: int, b: int, c: int, x: Fraction) -> Fraction:
    """2F1(a, b; c; x) for non-positive-integer a or b, exact rationals."""
    total = Fraction(1)
    term = Fraction(1)
    k = 0
    while (a + k) != 0 and (b + k) != 0:
        term = term * Fraction((a + k) * (b + k), (c + k) * (k + 1)) * x
        total += term
        k += 1
        if k > 100_000:      # unreachable for valid (terminating) inputs
            raise RuntimeError("2F1 did not terminate")
    return total


def _gordon_general(n: int, l: int, n_prime: int) -> float:
    """Signed <n' l-1| r |n l> [a0], spec 02 Eq. (2.10), n != n'.

    Exact-rational bracket (fractions.Fraction), log-gamma prefactor
    (spec 02 §4.4 pitfall 3: the reference path is rational arithmetic; no
    float hyp2f1 fast path is shipped). Raises at n = n' (0^0 degenerate —
    audit §3 item 7); callers use the closed form (2.11) there.
    """
    if n == n_prime:
        raise ValueError("Gordon general formula invalid at n = n' — use the "
                         "closed form (3/2) n sqrt(n^2 - l^2) (spec 02 Eq. 2.11)")
    n_r, np_r = n - l - 1, n_prime - l
    x_arg = Fraction(-4 * n * n_prime, (n - n_prime) ** 2)
    f1 = _hyp2f1_terminating(-n_r, -np_r, 2 * l, x_arg)
    f2 = _hyp2f1_terminating(-n_r - 2, -np_r, 2 * l, x_arg)
    bracket = f1 - Fraction((n - n_prime) ** 2, (n + n_prime) ** 2) * f2
    if bracket == 0:
        return 0.0
    ln_pre = (-math.log(4.0) - math.lgamma(2 * l)
              + 0.5 * (math.lgamma(n + l + 1) + math.lgamma(n_prime + l)
                       - math.lgamma(n - l) - math.lgamma(n_prime - l + 1))
              + (l + 1) * math.log(4.0 * n * n_prime)
              + (n + n_prime - 2 * l - 2) * math.log(abs(n - n_prime))
              - (n + n_prime) * math.log(n + n_prime))
    sign = (-1.0) ** (n_prime - l)
    if n < n_prime and (n + n_prime - 2 * l - 2) % 2 == 1:
        sign = -sign
    if bracket < 0:
        sign = -sign
    ln_bracket = math.log(abs(bracket.numerator)) - math.log(bracket.denominator)
    return sign * math.exp(ln_pre + ln_bracket)


def radial_me_gordon(n1: int, l1: int, n2: int, l2: int) -> float:
    """Exact hydrogen <n1 l1| r |n2 l2> [a0], |l1 - l2| = 1 (spec 02 §2.6).

    n1 != n2: Gordon Eq. (2.10) in exact rational arithmetic (validated vs
    direct quad integration to <= 9e-10 in the spec harness; reproduces
    128 sqrt(6)/243 for 1s->2p). n1 = n2: closed form (3/2) n sqrt(n^2 - l^2)
    with l = max(l1, l2) (Eq. 2.11 — the general formula divides by zero
    there). Reference-grade: used in tests and the delta -> 0 limit only.
    """
    for name, n, l in (("1", n1, l1), ("2", n2, l2)):
        if int(n) != n or int(l) != l or l < 0 or n < l + 1:
            raise ValueError(f"invalid hydrogen state {name}: n={n}, l={l}")
    if abs(l1 - l2) != 1:
        raise ValueError(f"dipole requires |l1 - l2| = 1, got l1={l1}, l2={l2}")
    if n1 == n2:
        l = max(l1, l2)
        return 1.5 * n1 * math.sqrt(n1 * n1 - l * l)
    if l1 > l2:
        return _gordon_general(n1, l1, n2)
    return _gordon_general(n2, l2, n1)


# ---------------------------------------------------------------------------
# Whittaker/hyperu closed form + the audit-R4 error-table machinery
# ---------------------------------------------------------------------------

def _whittaker_u_unguarded(nu: float, l: int, r: np.ndarray) -> np.ndarray:
    """Seaton-normalized QDT orbital u(r) via scipy hyperu, NO nu guard.

    u = N W_{nu,l+1/2}(2r/nu), W = e^(-z/2) z^(l+1) U(l+1-nu, 2l+2, z),
    N = [nu^2 Gamma(nu+l+1) Gamma(nu-l)]^(-1/2) (spec 02 Eqs. 2.8-2.9).
    Private: exists so the audit-R4 test can measure the precision collapse
    ABOVE the public cutoff. Production code must call ``whittaker_u``.
    """
    r = np.asarray(r, dtype=float)
    z = 2.0 * r / nu
    ln_norm = -0.5 * (2.0 * math.log(nu) + math.lgamma(nu + l + 1)
                      + math.lgamma(nu - l))
    return np.exp(ln_norm - z / 2.0 + (l + 1) * np.log(z)) * hyperu(
        l + 1 - nu, 2 * l + 2, z)


def whittaker_u(nu: float, l: int, r: np.ndarray) -> np.ndarray:
    """Seaton-normalized QDT orbital u(r) [a0^-1/2], spec 02 Eqs. (2.8)-(2.9).

    Pointwise cross-check ONLY (benchmark B12, nu <= 20); the production
    Method B is the pure-Coulomb Numerov path. Raises IntegrityError for
    nu > 25: scipy.special.hyperu precision collapses at large nu (measured
    on scipy 1.17.1: rel err 5.5e-6 at nu=25, 0.49 at nu=35, 93 at nu=40 —
    regenerated at test time per audit R4). Additionally, on scipy 1.17.1
    hyperu returns NaN for non-integer nu >~ 15 beyond r ~ 0.2-0.5 nu^2;
    NaNs propagate to the caller (documented, version-dependent). Sign
    convention: tail-positive (equals (-1)^(n-l-1) times the origin-positive
    hydrogen convention at integer nu).
    """
    if nu > 25.0:
        raise IntegrityError(
            f"whittaker_u refused for nu = {nu} > 25: scipy hyperu precision "
            "collapse (spec 02 §2.4 / audit R4 + §3 item 6); use the "
            "pure-Coulomb Numerov Method B instead")
    if nu <= l:
        raise IntegrityError(f"nu = {nu} <= l = {l}: no bound QDT orbital")
    return _whittaker_u_unguarded(nu, l, r)


def _u_exact_hydrogen(n: int, l: int, r: np.ndarray) -> np.ndarray:
    """Analytic hydrogen u_nl(r) = r R_nl(r) [a0^-1/2], mu = 1, Z = 1.

    Bethe & Salpeter §3 normalization (origin-positive convention);
    log-space prefactor + scipy eval_genlaguerre. Test oracle for the
    audit-R4 hyperu error table.
    """
    r = np.asarray(r, dtype=float)
    ln_norm = 0.5 * (3.0 * math.log(2.0 / n) + math.lgamma(n - l)
                     - math.log(2.0 * n) - math.lgamma(n + l + 1))
    rho = 2.0 * r / n
    return (np.exp(ln_norm - rho / 2.0 + (l + 1) * np.log(rho))
            * eval_genlaguerre(n - l - 1, 2 * l + 1, rho) * n / 2.0)


def hyperu_hydrogen_error(nu: int, l: int = 0, n_samples: int = 300) -> float:
    """One row of the audit-R4 hyperu error table, regenerated on the
    INSTALLED scipy: max relative deviation of the hyperu-Whittaker orbital
    from analytic hydrogen u_nl over the classical region [0.2, 1.8] nu^2,
    sign-aligned ((-1)^(n-l-1), tail-positive vs origin-positive
    conventions). Dimensionless. Spec 02 §2.4 check values (scipy 1.17.1):
    6e-13 @ nu=10, 3.5e-8 @ nu=20, 5.5e-6 @ nu=25, 0.49 @ nu=35.
    """
    if int(nu) != nu or nu <= l:
        raise ValueError("hyperu_hydrogen_error requires integer nu > l")
    r = np.linspace(0.2 * nu**2, 1.8 * nu**2, n_samples)
    u_ref = _u_exact_hydrogen(int(nu), l, r) * (-1.0) ** (nu - l - 1)
    u_hyp = _whittaker_u_unguarded(float(nu), l, r)
    return float(np.max(np.abs(u_hyp - u_ref)) / np.max(np.abs(u_ref)))


# ---------------------------------------------------------------------------
# Consensus machinery — THE public matrix-element entry point (spec 02 §1/§5)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RadialMEResult:
    """Consensus radial matrix element [a0^k] (spec 02 §5).

    value: Method A (model-potential Numerov), the quoted number, signed in
    this module's tail-positive convention (only |value| is comparable
    across codes). per_method: signed A and B (same machinery, same
    convention) and Kaulakys (|R| sign-aligned to A; the formula's own sign
    is convention-laden). spread_rel/abs: max pairwise |magnitude
    difference| (relative to |value|) — this IS the numerical uncertainty
    estimate shipped with every ME (audit §4 item 5).
    """

    value: float
    per_method: dict[str, float]
    spread_rel: float
    spread_abs: float


def _ab_ceiling(nu_min: float, me_abs: float) -> float:
    """Per-regime A-vs-B spread ceiling (spec 02 §6 B8/B11 + §4.5).

    nu >= 15: 1e-4 (measured 2.0e-6 at n=50, 4.2e-5 at n=20) for MEs
    >= 50 a0; 1e-3 for smaller (cancellation-suppressed) elements (measured
    4.7e-6 on the 13-a0 50D->51F pair). nu < 15: 3e-2 — the Coulomb
    approximation degrades where core overlap matters and the A-B spread IS
    the dominant honest uncertainty there (spec 02 §7.1).
    """
    if nu_min < 15.0:
        return 3e-2
    return 1e-4 if me_abs >= 50.0 else 1e-3


def _ak_ceiling(me_abs: float) -> float:
    """A-vs-Kaulakys ceiling (spec 02 §6 B9): 5e-3, widened to 1e-2 for
    cancellation-suppressed MEs < 50 a0 (measured worst case 2e-3 on Rb
    50D5/2->51F7/2)."""
    return 5e-3 if me_abs >= 50.0 else 1e-2


def radial_matrix_element_consensus(
        species: Species,
        state1: tuple[int, int, float],
        state2: tuple[int, int, float],
        k: int = 1, *,
        check: bool = True,
        h: float = 1e-3,
        include_so: bool = True) -> RadialMEResult:
    """THE public radial-ME entry point (spec 02 §5; single-method calls are
    private — spec 02 §4.4 pitfall 8 / audit §3 item 8). Returns [a0^k].

    Runs Method A (MSD94 Numerov) and Method B (pure-Coulomb Numerov) on the
    shared grid r in [alpha_c^(1/3), 2 n_max (n_max + 15)] a0, plus Method C
    (Kaulakys) when k = 1, |l1 - l2| = 1 and min(nu) >= 10 (quasiclassical
    validity, spec 02 §7.4). states are (n, l, j); nu comes from spec 01
    (n_star), so the per-species hard floors apply — this entry point is
    Rydberg machinery, low-n dipoles come from experiment (spec 02 §7.1).

    Refusals (IntegrityError): fewer than two methods available (refuse-to-
    guess: single-method MEs never ship); with check=True (default), any
    pairwise spread above the per-regime ceilings of spec 02 §6 — no
    averaging over a failed consensus. The spread is reported as the
    numerical uncertainty either way.
    """
    n1, l1, j1 = state1
    n2, l2, j2 = state2
    _validate_lj(l1, j1)
    _validate_lj(l2, j2)
    if k < 1:
        raise IntegrityError(
            "consensus MEs are defined for k >= 1 (k < 0 is contact-type and "
            "refused, audit §3 item 11; k = 0 overlap is a diagnostic only)")
    nu1 = float(n_star(species, n1, l1, j1))
    nu2 = float(n_star(species, n2, l2, j2))
    n_max = max(n1, n2)
    r_outer = 2.0 * n_max * (n_max + 15.0)
    per_method: dict[str, float] = {}
    a1 = radial_wavefunction(species, n1, l1, j1, h=h, r_outer=r_outer,
                             include_so=include_so, method="model_potential")
    a2 = radial_wavefunction(species, n2, l2, j2, h=h, r_outer=r_outer,
                             include_so=include_so, method="model_potential")
    per_method["model_potential"] = radial_matrix_element(a1, a2, k)
    b1 = radial_wavefunction(species, n1, l1, j1, h=h, r_outer=r_outer,
                             method="coulomb")
    b2 = radial_wavefunction(species, n2, l2, j2, h=h, r_outer=r_outer,
                             method="coulomb")
    per_method["coulomb"] = radial_matrix_element(b1, b2, k)
    me_a = per_method["model_potential"]
    if k == 1 and abs(l1 - l2) == 1 and min(nu1, nu2) >= 10.0:
        per_method["kaulakys"] = math.copysign(
            radial_me_kaulakys(nu1, l1, nu2, l2), me_a)
    if len(per_method) < 2:
        raise IntegrityError(
            f"only {len(per_method)} method(s) available for "
            f"{species.name} {state1}->{state2} (k={k}): single-method "
            "matrix elements never ship (spec 02 §1 / audit §3 item 8)")
    mags = {m: abs(v) for m, v in per_method.items()}
    scale = abs(me_a)
    if scale == 0.0:
        scale = max(mags.values()) or 1.0
    spread_abs = max(mags.values()) - min(mags.values())
    spread_rel = spread_abs / scale
    if check:
        nu_min = min(nu1, nu2)
        ab = abs(mags["model_potential"] - mags["coulomb"]) / scale
        ceiling = _ab_ceiling(nu_min, abs(me_a))
        if ab > ceiling:
            raise IntegrityError(
                f"A-vs-B consensus FAILED for {species.name} {state1}->"
                f"{state2}: spread {ab:.2e} > ceiling {ceiling:.0e} "
                f"(per_method = {per_method}) — refusing to average "
                "(spec 02 §6 B8 / audit §3 item 8)")
        if "kaulakys" in mags:
            ak = abs(mags["model_potential"] - mags["kaulakys"]) / scale
            ceiling = _ak_ceiling(abs(me_a))
            if ak > ceiling:
                raise IntegrityError(
                    f"A-vs-Kaulakys consensus FAILED for {species.name} "
                    f"{state1}->{state2}: spread {ak:.2e} > ceiling "
                    f"{ceiling:.0e} (per_method = {per_method}) — refusing "
                    "(spec 02 §6 B9 / audit §3 item 8)")
    return RadialMEResult(value=me_a, per_method=per_method,
                          spread_rel=spread_rel, spread_abs=spread_abs)
