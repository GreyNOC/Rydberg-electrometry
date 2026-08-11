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
     scipy at test time) and additionally returns non-finite values for
     non-integer nu >~ 11 inside the classical region (measured, scipy
     1.17.1). BOTH failure modes raise; neither returns a degraded number.
  C. ``kaulakys`` — Kaulakys, J. Phys. B 28, 4963 (1995) quasiclassical
     Anger-function formula (eqs. 19, 21-24, 30-31; full text in-repo at
     ``kaulakys_text.txt``, spot-verified by audit item 14). k = 1 only.

House rule (spec 02 §1): no single-method number ships. The public entry
point for alkali matrix elements is ``radial_matrix_element_consensus``,
which runs every available method, reports the spread as the numerical
uncertainty, refuses (IntegrityError) when fewer than two methods are
available, and refuses when any pairwise deviation exceeds its per-regime
allowance (audit §3 item 8: no consensus => raise, don't average). The
allowances are tested against the ORBIT SCALE (nu1 nu2)^k, not against the
relative spread, wherever the element is cancellation-suppressed — see
``check_ab_consensus``, which is exported so that no consumer needs a
private copy of these rules.

What the A-vs-B spread does NOT test: the MSD94 parameter tables. Energy is
an input here (nu from spec 01's measured defects), so the model potential
only acts inside the divergence-guard truncation radius; doubling Rb
a3(l=0) moves the A-B spread from 2.02e-6 to 2.40e-6, 40x inside the §6 B8
ceiling. ``model_potential_defect`` solves the model as a genuine
eigenvalue problem — the quantity MSD94 was fitted to — and is the tripwire
audit R20 assumed B8 was providing.

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
from .numerov import numerov_inward, numerov_outward
from .provenance import IntegrityError

__all__ = [
    "ModelPotentialParams", "RB_MODEL_POTENTIAL", "CS_MODEL_POTENTIAL",
    "A4_L1_NOTE", "effective_charge", "model_potential",
    "model_potential_defect",
    "RadialSolution", "radial_wavefunction", "coulomb_wavefunction",
    "radial_matrix_element", "radial_me_gordon", "radial_me_kaulakys",
    "whittaker_u", "WHITTAKER_NU_MAX", "hyperu_hydrogen_error", "RadialMEResult",
    "radial_matrix_element_consensus", "check_ab_consensus",
    "clear_radial_cache",
]

# ---------------------------------------------------------------------------
# MSD94 model-potential parameter tables (spec 02 §3.1-3.3)
# ---------------------------------------------------------------------------

#: Spec 02 §3.3 disclosure — the ONE transcription discrepancy in the MSD94
#: tables: Rb a4(l=1) is -0.81633314 in pairinteraction/ryd-numerov (8
#: significant digits, consistent formatting — ADOPTED) but -0.8163314 in ARC
#: master (7 digits). Confidence of the last digit: UNVERIFIED (two secondary
#: sources disagree; primary PRA 49, 982 Table I unfetched).
#:
#: Impact bound (RE-DERIVED 2026-08-10; the figures previously carried here
#: and in spec 02 §3.3 / audit R20 — "delta-a4 = 8e-8 ... perturbs Z_1 by
#: < 4e-8" — were arithmetically wrong by 21.8x and 6.3x respectively):
#:   |delta a4| = |0.81633314 - 0.8163314| = 1.740e-06   (2.131e-06 relative)
#:   delta Z_1(r) = |delta a4| r^2 exp(-a2[1] r), maximal at r = 2/a2[1]
#:   = 1.0372 a0 where r^2 exp(-a2 r) = 0.145586
#:   => max |delta Z_1| = 2.533e-07
#: Both figures are regenerated from the two transcriptions at test time
#: (tests/test_radial.py::test_a4_l1_impact_bound_is_reproducible), and the
#: physical conclusion is unchanged. Measured consequences of adopting the
#: ARC reading instead: Rb 50S->50P consensus ME moves 6.5e-14 relative, and
#: the reproduced Rb P-series quantum defect (``model_potential_defect``,
#: the one quantity in this module that IS sensitive to the tables) moves
#: 4.7e-8 — both far below every spec 02 §6 tolerance.
A4_L1_NOTE = (
    "Rb a4(l=1): adopted -0.81633314 (ryd-numerov); ARC reads -0.8163314; "
    "last digit UNVERIFIED, |delta a4| = 1.74e-6 perturbs Z_1(r) by "
    "<= 2.53e-7 (max at r = 1.04 a0) (spec 02 §3.3 / audit R20)"
)

#: Hard validity ceiling on the Whittaker/hyperu cross-check route.
#: Integrator ruling 2026-08-10: moved 25 -> 20 so the fence sits where the
#: method meets benchmark B12's own 1e-6 pointwise contract (20 < nu <= 25
#: measures up to 5.7e-6, 5.7x outside contract). Supersedes spec 02 §7.3
#: and integrity-audit refusal #6,
#: which also mis-state the exception type as ValueError. See whittaker_u.
WHITTAKER_NU_MAX = 20.0

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
    from .atom import element_symbol

    # element_symbol is the single species -> element source (audit R10);
    # it refuses for a species with no declared element rather than
    # defaulting, which the old name-slicing form could not do.
    try:
        el = element_symbol(sp)
    except IntegrityError:
        el = ""
    if el == "Rb":
        p = RB_MODEL_POTENTIAL
    elif el == "Cs":
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
                         r_inner: float = 1e-4,
                         r_outer: float | None = None) -> RadialSolution:
    """Pure-Coulomb (V = -1/r) radial state for arbitrary real nu > l. All a.u.

    The species-independent engine behind Method B, exposed for the hydrogen
    benchmarks (spec 02 §6 B1-B7: nu = n integer, mu_mass = 1) and the
    Whittaker cross-check (B12, non-integer nu, mu_mass = 1).

    NOT B13 or B14, despite what this line said before the 2026-08-10 spec
    reconciliation: B13 (Kaulakys vs Gordon) solves no wavefunction at all,
    and B14 (outer-cutoff adequacy) runs the Rb-87 model-potential path via
    radial_wavefunction. The false claim had propagated from here into spec
    02 §5, where it contradicted that document's own §6 note. Defaults:
    r_outer = 2 nc (nc + 15) with nc = ceil(nu); r_inner = 1e-4 a0. Alkali
    Method B goes through ``radial_wavefunction(..., method="coulomb")``,
    which shares Method A's species grid (r_inner = alpha_c^(1/3), where the
    model potential stops being physical — that cutoff is unaffected).

    ``r_inner`` default CHANGED 2026-08-10 from 1e-2 to 1e-4 a0. It is a
    pure hydrogen-benchmark knob (no production caller), and 1e-2 silently
    truncated the low-n rows: spec 02 §6 B1 (H 1s->2p) is quoted "rel <= 1e-7
    (measured 4e-12)" and B7 (h^4 order on B1) "measured 16.2", neither of
    which is reachable at 1e-2. Measured through this function +
    ``radial_matrix_element`` (h = 1e-3, |rel| vs exact Gordon):

        r_inner |     B1      B2      B3      B4      B5      B6 | B7 ratio
        --------+---------------------------------------------- + --------
          1e-2  | 6.6e-7  8.2e-8  1.0e-11 6.3e-10 5.5e-10 1.9e-9 |   1.02
          1e-3  | 6.6e-10 9.4e-11 1.3e-11 9.2e-11 5.6e-10 1.3e-9 |    —
          1e-4  | 4.7e-12 2.9e-12 2.7e-11 2.7e-11 5.5e-10 1.9e-9 |  16.29
          1e-5  | 6.7e-13 1.6e-11 5.4e-8  5.2e-11 5.5e-10 1.9e-9 |    —

    1e-4 is the optimum and reproduces both of the spec's stated
    measurements; below it the l = 2 row (B3) degrades because the
    centrifugal term (2l+1/2)(2l+3/2)/x^2 drives h^2 g/12 toward 1 near the
    inner edge, where the Numerov auxiliary f = 1 - h^2 g/12 loses meaning.
    The tolerances in §6 were NOT touched.
    """
    if l < 0 or int(l) != l:
        raise ValueError(f"l must be a non-negative integer, got {l!r}")
    if r_outer is None:
        nc = max(1, int(math.ceil(nu)))
        r_outer = 2.0 * nc * (nc + 15.0)
    return _solve_scaled(l, float(nu), mu_mass, lambda r: -1.0 / r,
                         h, r_inner, r_outer,
                         n=int(math.ceil(nu)), j=l + 0.5, method="coulomb")


# ---------------------------------------------------------------------------
# MSD94 eigenvalue problem — the ONLY quantity in this module that is
# genuinely sensitive to the model-potential parameter tables (audit R20)
# ---------------------------------------------------------------------------

#: Largest quantum defect searched by ``model_potential_defect``. The alkali
#: low-l defects covered by spec 01 top out at Cs nS (delta ~ 4.05); 4.6
#: leaves margin without admitting the next lower eigenvalue (spacing 1).
_DEFECT_SEARCH_SPAN = 4.6


def _shoot_model(p: ModelPotentialParams, l: int, j: float, nu: float,
                 mu_mass: float, h: float, r_min: float,
                 include_so: bool) -> tuple[float, int]:
    """One trial energy of the MSD94 bound-state problem (spec 02 Eq. 2.5).

    Integrates the SAME scaled equation the production solver uses, but in
    both directions and with E treated as an EIGENVALUE rather than an
    input: outward from r_min with the regular behaviour X ~ x^(2l+3/2)
    (R ~ r^l, X = R r^(3/4)), inward from r_outer with the decaying seed.
    Returns (Wronskian mismatch at the matching radius r = nu^2, node count
    of the matched solution). The Wronskian W = X_out' X_in - X_in' X_out
    (each branch scaled to unit maximum) vanishes exactly at an eigenvalue
    and — unlike the log-derivative difference — has no poles, so simple
    bisection is safe.
    """
    r_max = 2.0 * nu * (nu + 15.0)
    x_in = math.sqrt(r_min)
    x_out = math.sqrt(r_max)
    npt = int(math.ceil((x_out - x_in) / h)) + 1
    x = x_in + h * np.arange(npt)
    r = x * x
    v = model_potential(p, l, j, r, include_so=include_so)
    energy = -mu_mass / (2.0 * nu * nu)
    g = (8.0 * mu_mass * x * x * (v - energy)
         + (2 * l + 0.5) * (2 * l + 1.5) / (x * x))
    im = int(np.searchsorted(r, nu * nu))
    im = max(3, min(npt - 4, im))
    y_out = numerov_outward(x, g, x[0] ** (2 * l + 1.5), x[1] ** (2 * l + 1.5))
    y_in = numerov_inward(x, g, 1e-10, 1.2e-10)
    s_out = float(np.max(np.abs(y_out[:im + 2])))
    s_in = float(np.max(np.abs(y_in[im - 2:])))
    if not (s_out > 0.0 and s_in > 0.0):
        return float("nan"), -1
    y_out = y_out / s_out
    y_in = y_in / s_in
    d_out = (y_out[im + 1] - y_out[im - 1]) / (2.0 * h)
    d_in = (y_in[im + 1] - y_in[im - 1]) / (2.0 * h)
    wronskian = float(d_out * y_in[im] - d_in * y_out[im])
    if y_in[im] != 0.0:
        matched = np.concatenate([y_out[:im], y_in[im:] * (y_out[im] / y_in[im])])
    else:
        matched = np.concatenate([y_out[:im], y_in[im:]])
    c = matched[5:]                       # skip the seed transient
    c = c[c != 0.0]
    nodes = int(np.count_nonzero(np.diff(np.sign(c)) != 0))
    return wronskian, nodes


def model_potential_defect(species: Species, n: int, l: int, j: float, *,
                           h: float = 4e-3, r_min: float = 1e-6,
                           include_so: bool = True,
                           params: ModelPotentialParams | None = None) -> float:
    """Quantum defect delta = n - nu PREDICTED by the MSD94 potential [—].

    Solves V_lj(r) (spec 02 Eqs. 2.2-2.4) as a genuine bound-state
    EIGENVALUE problem — outward/inward shooting on the x = sqrt(r) grid,
    Wronskian matching at r = nu^2, node count pinned to n - l - 1 — and
    returns n minus the eigenvalue's effective quantum number.

    Why this function exists (audit R20 / spec 02 §6 B8). Everywhere else in
    this module the energy is an INPUT taken from spec 01's measured
    defects, so the model potential only shapes the wavefunction inside the
    divergence-guard truncation radius and its effect on matrix elements is
    ~1e-7. Measured: doubling Rb a3(l=0) moves the 50S->50P consensus ME by
    4e-7 relative and the A-vs-B spread from 2.02e-6 to 2.40e-6 — 40x inside
    the B8 ceiling; flipping the sign of Rb a4(l=1) moves it by 6e-8. The
    declared "B8 catches any material potential error" mitigation is
    therefore inoperative for the a1/a3/a4/r_c rows, and the UNVERIFIED Rb
    a4(l=1) digit was unguarded. The eigenvalue IS what MSD94 fitted, so it
    is the quantity that moves: the same a3(l=0) doubling shifts the
    reproduced Rb S defect by +0.41 (1300x the 3.3e-4 baseline residual) and
    the a4(l=1) sign flip shifts the Rb P centroid by -0.044.
    tests/test_radial.py turns that into the missing tripwire.

    Accuracy of the model itself (measured, this implementation, vs spec 01
    defects at n = 12, l-centroids): Rb S +3.25e-4, P +2.18e-3, D +9.98e-4,
    F +8.47e-4; Cs S +5.64e-4, P +5.38e-3, D +1.32e-3, F -1.27e-4. MSD94's
    spin-orbit term is the hydrogenic form, so the model reproduces the
    l-CENTROID of a fine-structure doublet, not its two components (measured
    model Rb nP splitting 3.9e-4 vs the 1.3e-2 experimental one) — compare
    centroids. Numerical robustness of the returned defect: halving h moves
    it by <= 1.6e-5, dropping r_min to 1e-8 by <= 1.1e-4, dropping the
    spin-orbit term by <= 1.4e-7 (all worst-case over Rb l = 0..3, n = 12).

    Bracketing is by NODE COUNT, which is a monotone step function of nu:
    the plateau where the matched solution carries exactly n - l - 1 nodes
    contains exactly one eigenvalue, and the Wronskian (pole-free) changes
    sign once inside it. Raises IntegrityError when the search window
    nu in [max(l+1, n-4.6), n) contains no such plateau or the Wronskian
    does not change sign across it — a corrupted parameter table that pushes
    the state out of the window refuses rather than returning a wrong root.
    """
    _validate_lj(l, j)
    if n < l + 1:
        raise ValueError(f"n = {n} < l + 1 = {l + 1}: no bound state")
    p = params if params is not None else _potential_params_for(species)
    mu_mass = 1.0 / (1.0 + M_E / (species.mass_u * AMU))
    target = n - l - 1
    lo = max(l + 1.0 + 1e-3, n - _DEFECT_SEARCH_SPAN)
    hi = float(n) - 1e-5
    if hi <= lo:
        raise ValueError(f"empty defect search window for n={n}, l={l}")

    def shoot(nu: float) -> tuple[float, int]:
        return _shoot_model(p, l, j, nu, mu_mass, h, r_min, include_so)

    def refuse(why: str) -> IntegrityError:
        return IntegrityError(
            f"MSD94 eigenvalue search for {species.name} (n={n}, l={l}, "
            f"j={j}) failed over nu in [{lo:.3f}, {hi:.3f}] ({target} nodes "
            f"expected): {why} — the model potential does not bind this "
            "state where the measured defect puts it; refusing to guess "
            "(spec 02 §3 / audit R20)")

    w_lo, nodes_lo = shoot(lo)
    w_hi, nodes_hi = shoot(hi)
    if nodes_lo > target or nodes_hi < target:
        raise refuse(f"node count runs {nodes_lo} -> {nodes_hi} across the "
                     "window, which never sits at the target")

    def edge(want: int) -> tuple[float, float]:
        """Bracket (a, b) of the step where the node count reaches ``want``:
        nodes(a) < want <= nodes(b), by bisection on the monotone count."""
        a, b = lo, hi
        for _ in range(60):
            m = 0.5 * (a + b)
            if shoot(m)[1] >= want:
                b = m
            else:
                a = m
            if b - a < 1e-6:
                break
        return a, b

    nu_a = lo if nodes_lo == target else edge(target)[1]
    nu_b = hi if nodes_hi == target else edge(target + 1)[0]
    if nu_b <= nu_a:
        raise refuse("the target-node plateau is empty")
    w_a, n_a = shoot(nu_a)
    w_b, n_b = shoot(nu_b)
    if n_a != target or n_b != target:
        raise refuse(f"plateau endpoints carry {n_a}/{n_b} nodes")
    if not (math.isfinite(w_a) and math.isfinite(w_b)) or w_a * w_b >= 0.0:
        raise refuse("the Wronskian does not change sign across the plateau")
    a, b = nu_a, nu_b
    for _ in range(80):
        m = 0.5 * (a + b)
        w_m, _ = shoot(m)
        if w_m * w_a > 0.0:
            a, w_a = m, w_m
        else:
            b = m
        if b - a < 1e-10:
            break
    return n - 0.5 * (a + b)


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

    Pointwise cross-check ONLY (benchmark B12); the production Method B is
    the pure-Coulomb Numerov path. Sign convention: tail-positive (equals
    (-1)^(n-l-1) times the origin-positive hydrogen convention at integer
    nu).

    TWO independent scipy.special.hyperu failure modes are refused, both
    re-measured at test time on the installed scipy (audit R4):

    1. Large-nu precision collapse -> IntegrityError for nu > 20. Measured
       on scipy 1.17.1 via ``hyperu_hydrogen_error``: 3.5e-8 @ nu=20,
       5.5e-6 @ 25, 1.9e-4 @ 28, 1.8e-3 @ 30, 0.49 @ 35, 93 @ 40.

       NORMATIVE FENCE MOVED 25 -> 20 (integrator ruling, 2026-08-10,
       superseding spec 02 §7.3 / integrity-audit refusal #6). Rationale:
       the fence must sit where the method meets its OWN stated contract.
       Benchmark B12 demands 1e-6 pointwise; the band 20 < nu <= 25
       measures up to 5.7e-6 (worst over l = 0..2 at nu = 25: 5.48e-6,
       4.71e-6, 5.72e-6), i.e. 5.7x outside contract, against 3.5e-8 at
       nu = 20 — a factor ~160 step across the fence.

       CORRECTION (2026-08-10, same day): this rationale originally read
       "~1e-5, ten times worse", which the spec-reconciliation audit caught
       as ~2x overstated and inconsistent with the error table three lines
       above. The ruling stands on the measurement — 5.7x outside a stated
       contract is sufficient to move a fence — but the multiplier was
       wrong and is corrected here at the source. The same wording had
       propagated verbatim into spec 02 §7.3 and integrity-audit refusal #6.

       This function is a validation instrument, and a validation
       instrument running outside its claimed accuracy is the "plausible
       but wrong" hazard the house rule exists to stop — a warning is not
       sufficient for a number a cross-check will trust.
       Nothing on the production Rydberg path calls this (Method B is the
       pure-Coulomb Numerov ``coulomb_wavefunction``), so the fence costs
       no physics. ``_whittaker_u_unguarded`` remains available in-module
       for deliberate out-of-contract investigation, e.g. the R4 table.
    2. Non-finite returns -> IntegrityError when ANY requested sample comes
       back non-finite. On scipy 1.17.1 hyperu returns NaN for non-integer
       nu >~ 11 beyond r ~ 0.2-0.9 nu^2, i.e. INSIDE the classically allowed
       region where the orbital is largest (measured: nu=18.5, l=1 ->
       4884/16547 samples NaN, first at 0.30 nu^2; nu=19.99, l=0 ->
       9673/17880, first at 0.21 nu^2). Handing those arrays back would let
       a np.allclose-style B12 comparison pass on an all-NaN slice, i.e. a
       scipy bump could silently disable the module's only Numerov-
       independent cross-check. ``_whittaker_u_unguarded`` stays available
       inside the module for the R4 error table itself.
    """
    if nu > WHITTAKER_NU_MAX:
        raise IntegrityError(
            f"whittaker_u refused for nu = {nu} > {WHITTAKER_NU_MAX:g}: "
            "scipy hyperu carries up to 5.7e-6 relative error just above this "
            "(5.7x), above "
            "benchmark B12's own 1e-6 tolerance, rising to 0.49 at nu = 35 "
            "(measured, audit R4). Use the pure-Coulomb Numerov Method B "
            "(coulomb_wavefunction) — it has no such limit.")
    if nu <= l:
        raise IntegrityError(f"nu = {nu} <= l = {l}: no bound QDT orbital")
    u = _whittaker_u_unguarded(nu, l, r)
    bad = ~np.isfinite(u)
    n_bad = int(np.count_nonzero(bad))
    if n_bad:
        r_arr = np.atleast_1d(np.asarray(r, dtype=float))
        r_first = float(r_arr[int(np.argmax(np.atleast_1d(bad)))])
        raise IntegrityError(
            f"scipy.special.hyperu returned {n_bad}/{u.size} non-finite "
            f"values for nu = {nu}, l = {l}, first at r = {r_first:.4g} a0 "
            f"= {r_first / (nu * nu):.3f} nu^2 (known breakdown for "
            "non-integer nu; scipy-version-dependent) — refusing to return a "
            "silently degraded array; use the pure-Coulomb Numerov Method B "
            "(spec 02 §2.4 / audit refuse-to-guess)")
    return u


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

    orbit_scale_a0k: (nu1 nu2)^k [a0^k], the ~<r^k> size of the pair.
    spread_over_orbit: spread_abs / orbit_scale_a0k. For a
    cancellation-suppressed element (|value| << orbit_scale_a0k) the
    RELATIVE spread is not a meaningful uncertainty — the methods' absolute
    error floors are unchanged while |value| -> 0, so spread_rel can exceed
    1 on a numerically perfect result (Rb87 30D5/2->31F7/2: spread_rel 1.65
    from the Kaulakys arm, spread_over_orbit 3.4e-5, |A|-|B| 3.3e-8 of
    scale). Quote spread_over_orbit whenever
    |value| < 1e-3 * orbit_scale_a0k.
    """

    value: float
    per_method: dict[str, float]
    spread_rel: float
    spread_abs: float
    orbit_scale_a0k: float
    spread_over_orbit: float


#: |ME| below this fraction of the orbit scale (nu1 nu2)^k is
#: "cancellation-suppressed": the integrand passes through a node in Delta-n
#: and |ME| -> 0 while every method's ABSOLUTE error floor stays put, so the
#: relative spread diverges on numerically perfect results.
SUPPRESSED_ME_FRACTION = 1e-3
#: A-vs-B ceiling in the suppressed regime, as a fraction of the orbit scale
#: (adopted from the already-shipped rule in rydsim.stark._pair_me_a0).
#: Measured worst |A|-|B| over Rb87/Cs133 S->P, P->D, D->F, F->G suppressed
#: pairs, n = 10..70, Delta-n = 0,1,2: 1.04e-6 of scale (Cs 13D->15F); the
#: Rydberg-n cases sit at ~3e-8 (Rb 30D->31F: 2.95e-5 a0 / 887.8 a0).
SUPPRESSED_AB_ORBIT_CEILING = 1e-4
#: A-vs-B ceiling when BOTH states have l >= 4. There Method A IS Method B
#: (model_potential returns -1/r for l >= 4) apart from the ~1e-13 spin-orbit
#: term, so A-vs-B is not a model cross-check at all — the residual is
#: divergence-guard truncation noise. Same rule and rationale as stark.py.
HIGH_L_AB_ORBIT_CEILING = 1e-3
#: Absolute error floor of the Kaulakys quasiclassical formula [a0^k]. Its
#: deviation from Method A is an ABSOLUTE floor, not a relative one: measured
#: |A| - |K| over the same sweep sits at 2.8e-2 .. 6.8e-2 a0 essentially
#: independent of |ME| (Rb 50S->51P |ME| = 851 a0 -> 3.7e-2; Rb 30D->31F
#: |ME| = 0.018 a0 -> 3.0e-2). Worst case over every pair with |ME| < 15 a0,
#: where this floor rather than the §6 B9 relative ceiling binds: 4.1e-2 a0
#: (Rb 16D->17F). Floor set at 0.15 a0 = 3.7x that worst case.
KAULAKYS_ABS_FLOOR_A0 = 0.15


def _ab_ceiling(nu_min: float, me_abs: float,
                l1: int | None = None, l2: int | None = None) -> float:
    """Per-regime RELATIVE A-vs-B spread ceiling (spec 02 §6 B8/B11 + §4.5).

    nu >= 15: 1e-4 (measured 2.0e-6 at n=50, 4.2e-5 at n=20) for MEs
    >= 50 a0; 1e-3 for smaller (cancellation-suppressed) elements (measured
    4.7e-6 on the 13-a0 50D->51F pair). nu < 15: 3e-2 — the Coulomb
    approximation degrades where core overlap matters and the A-B spread IS
    the dominant honest uncertainty there (spec 02 §7.1).

    PAIR-CLASS SCOPING (integrator reconciliation 2026-08-10). The 1e-4 row
    is SCOPED to S<->P pairs, the only class spec 02 measured it on: B8 is
    "Rb 50S1/2->50P3/2 ... 2.0e-6; also 51P/49P/60S pairs <= 4.2e-6" and B11
    is "Cs 50S1/2->50P3/2 ... 2.8e-6". Applying that number to near-
    degenerate P<->D and D<->F pairs is an UNSOURCED EXTRAPOLATION: those
    pairs measure 1.0e-4..1.2e-4, so a 1e-4 ceiling fails them on nothing
    but the absence of a measurement. They keep the sourced 1e-3 regime
    ceiling, which still leaves ~an order of magnitude of tripwire margin
    over what they actually measure.

    This rule originated in ``lifetimes._ab_ceiling``, which had diverged
    from this gate and declared the divergence. Two gates disagreeing about
    the same physics IS the R10 constant-forking hazard, so the
    better-sourced rule was adopted here and lifetimes now delegates.
    Requested spec amendment: spec 02 §6's B8 row should state that its
    1e-4 figure is S<->P-measured. Passing l1/l2 is optional only for
    backward compatibility — omitting them applies the unscoped (stricter)
    reading.
    """
    if nu_min < 15.0:
        return 3e-2
    if l1 is not None and l2 is not None and {l1, l2} != {0, 1}:
        return 1e-3          # not the S<->P class the 1e-4 row was measured on
    return 1e-4 if me_abs >= 50.0 else 1e-3


def _ak_ceiling(me_abs: float) -> float:
    """A-vs-Kaulakys RELATIVE ceiling (spec 02 §6 B9): 5e-3, widened to 1e-2
    for MEs < 50 a0 (measured worst case 2e-3 on Rb 50D5/2->51F7/2). Used as
    one arm of a max() against ``KAULAKYS_ABS_FLOOR_A0``."""
    return 5e-3 if me_abs >= 50.0 else 1e-2


def check_ab_consensus(me_a: float, me_b: float, *,
                       nu1: float, nu2: float, l1: int, l2: int,
                       k: int = 1, label: str = "",
                       check: bool = True) -> tuple[float, float]:
    """THE A-vs-B consensus gate. Returns (spread_rel, spread_over_orbit).

    ``me_a`` / ``me_b`` are the signed Method-A (MSD94 Numerov) and Method-B
    (pure-Coulomb Numerov) elements [a0^k]; only magnitudes are compared
    (spec 02 §2.6). Raises IntegrityError on breach — never average a failed
    consensus (spec 02 §6 B8 / audit §3 item 8).

    Three regimes, because the relative spread is the right quantity in only
    one of them (this is the single gate the audit asks for — ``stark.py``
    and ``lifetimes.py`` carried private copies of two of these rules):

    1. **Cancellation-suppressed**, |A| < 1e-3 (nu1 nu2)^k. The nD -> (n+1)F
       channel passes through a node around n ~ 30 for Rb, so |ME| -> 0 while
       the solver floor stays at ~3e-8 of the orbit scale. Gate the ABSOLUTE
       deviation at ``SUPPRESSED_AB_ORBIT_CEILING`` x (nu1 nu2)^k. Under the
       old relative rule Rb87 (30,2,2.5)->(31,3,3.5) was refused on a
       |A|-|B| of 2.95e-5 a0 against an 887.8 a0 orbit — a deviation of
       3.3e-8 of scale, i.e. a numerically perfect result.
    2. **High-l**, min(l1, l2) >= 4: A and B are the same ODE (l >= 4 uses
       the pure Coulomb potential by the ARC convention of spec 02 §2.1), so
       the difference measures truncation noise, not the core model. Gate at
       ``HIGH_L_AB_ORBIT_CEILING`` x (nu1 nu2)^k.
    3. **Everything else** — the model-sensitive low-l pairs: the spec 02 §6
       B8 relative ceilings verbatim, unchanged.

    The relative spread ships with the result in every case; for regime 1 it
    is meaningless by construction and ``spread_over_orbit`` is the figure to
    quote.
    """
    mag_a, mag_b = abs(me_a), abs(me_b)
    diff = abs(mag_a - mag_b)
    scale = mag_a or mag_b or 1.0
    orbit = (nu1 * nu2) ** k
    spread_rel = diff / scale
    spread_orbit = diff / orbit if orbit > 0.0 else float("inf")
    if not check:
        return spread_rel, spread_orbit
    tag = f" for {label}" if label else ""
    if mag_a < SUPPRESSED_ME_FRACTION * orbit:
        allowed = SUPPRESSED_AB_ORBIT_CEILING * orbit
        if diff > allowed:
            raise IntegrityError(
                f"A-vs-B consensus FAILED{tag} on a cancellation-suppressed "
                f"element: |A|-|B| = {diff:.3e} > {allowed:.3e} a0^{k} "
                f"= {SUPPRESSED_AB_ORBIT_CEILING:.0e} x (nu1 nu2)^k "
                f"(A={me_a:.6e}, B={me_b:.6e} a0^{k}) — refusing to average "
                "(spec 02 §6 B8 / audit §3 item 8)")
    elif min(l1, l2) >= 4:
        allowed = HIGH_L_AB_ORBIT_CEILING * orbit
        if diff > allowed:
            raise IntegrityError(
                f"A-vs-B consensus FAILED{tag} on a high-l pair: |A|-|B| = "
                f"{diff:.3e} > {allowed:.3e} a0^{k} = "
                f"{HIGH_L_AB_ORBIT_CEILING:.0e} x (nu1 nu2)^k "
                f"(A={me_a:.6e}, B={me_b:.6e} a0^{k}; both methods integrate "
                "the pure Coulomb potential there, so this is solver noise) "
                "— refusing (spec 02 §6 B8 / audit §3 item 8)")
    else:
        ceiling = _ab_ceiling(min(nu1, nu2), mag_a, l1, l2)
        if spread_rel > ceiling:
            raise IntegrityError(
                f"A-vs-B consensus FAILED{tag}: spread {spread_rel:.2e} > "
                f"ceiling {ceiling:.0e} (A={me_a:.6e}, B={me_b:.6e} a0^{k}) "
                "— refusing to average (spec 02 §6 B8 / audit §3 item 8)")
    return spread_rel, spread_orbit


def _check_ak_consensus(me_a: float, me_k: float, *, k: int,
                        label: str) -> None:
    """A-vs-Kaulakys gate (spec 02 §6 B9). Raises IntegrityError on breach.

    The allowance is ``max(relative ceiling x |A|, KAULAKYS_ABS_FLOOR_A0)``.
    The relative arm is spec 02 §6 B9 verbatim and still binds on ordinary
    elements; the absolute arm exists because the quasiclassical formula's
    error is an absolute floor (~0.03 a0) that does not shrink with |ME|, so
    on a cancellation-suppressed element the relative measure reports >100 %
    on a result whose A-vs-B agreement is 3e-8 of the orbit scale. Kaulakys
    stays in ``per_method`` there as the order-of-magnitude cross-check it
    actually is.
    """
    allowed = max(_ak_ceiling(abs(me_a)) * abs(me_a), KAULAKYS_ABS_FLOOR_A0)
    diff = abs(abs(me_a) - abs(me_k))
    if diff > allowed:
        raise IntegrityError(
            f"A-vs-Kaulakys consensus FAILED for {label}: |A|-|K| = "
            f"{diff:.3e} > {allowed:.3e} a0^{k} = max({_ak_ceiling(abs(me_a)):.0e}"
            f" x |A|, {KAULAKYS_ABS_FLOOR_A0:g} a0) (A={me_a:.6e}, "
            f"K={me_k:.6e} a0^{k}) — refusing "
            "(spec 02 §6 B9 / audit §3 item 8)")


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
    pairwise deviation above the per-regime allowances — no averaging over a
    failed consensus. The spread is reported as the numerical uncertainty
    either way. See ``check_ab_consensus`` and ``_check_ak_consensus`` for
    the regimes: the ceilings are tested against the ORBIT SCALE (nu1 nu2)^k
    rather than the relative spread wherever the element is
    cancellation-suppressed, because a relative measure on a near-zero
    integral diverges on numerically perfect results.
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
    orbit = (nu1 * nu2) ** k
    spread_abs = max(mags.values()) - min(mags.values())
    spread_rel = spread_abs / scale
    spread_orbit = spread_abs / orbit if orbit > 0.0 else float("inf")
    label = f"{species.name} {state1}->{state2} (k={k})"
    check_ab_consensus(per_method["model_potential"], per_method["coulomb"],
                       nu1=nu1, nu2=nu2, l1=l1, l2=l2, k=k, label=label,
                       check=check)
    if check and "kaulakys" in per_method:
        _check_ak_consensus(me_a, per_method["kaulakys"], k=k, label=label)
    return RadialMEResult(value=me_a, per_method=per_method,
                          spread_rel=spread_rel, spread_abs=spread_abs,
                          orbit_scale_a0k=orbit,
                          spread_over_orbit=spread_orbit)
