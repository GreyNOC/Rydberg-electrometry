"""Stark effect, polarizabilities and Stark maps (spec 07).

Owns (docs/spec/00-conventions.md §4): perturbative scalar/tensor
polarizabilities alpha_0/alpha_2, m_j-resolved alpha(v, m_j), dynamic
alpha(omega), full Stark maps by diagonalization in |n l j m_j>, the
hydrogen linear/quadratic benchmark generators, Inglis-Teller / classical-
ionization validity fields, DC/LF readout formulas, and the cell-screening
INTERFACE form (parameter values are owned by spec 05).

Conventions (docs/spec/00-conventions.md):
  lock #14: DeltaE = -(1/2) alpha E^2 (alpha > 0 <=> level shifts down);
    alpha(v, m_j) = alpha_0 + alpha_2 [3 m_j^2 - j(j+1)]/[j(2j-1)],
    alpha_2 == 0 identically for j <= 1/2; AC: DeltaE = -(1/4) alpha(omega)
    E_amp^2 (peak amplitude, lock #3). Unit factors are derived at runtime
    from scipy.constants (rydsim.constants); 2.488318e-4 Hz/(V/cm)^2 per
    a.u. is a CHECK value that appears only in tests.
  lock #1: SI internally (J, V/m, C m); a.u. / MHz/(V/cm)^2 representations
    are populated alongside SI on every result object as the spec 07 §3.1
    unit tripwire. Radial integrals cross the rydsim.radial boundary in a0.
  Symbol rule: the electric field is ``field_Vm`` / ``e_bias_Vm`` in code —
    the letter F is reserved for the hyperfine quantum number (§3).
  R-5: alpha_0(Rb 50S) = 50.5 MHz/(V/cm)^2 (O'Sullivan-Stoicheff/Yerokhin),
    NOT spec 04's stale ~6e2 placeholder; benchmark RS-07-05 enforces.
  R-19: sign anchors are tested on alpha_0, never on alpha(m_j); for
    Rb nD5/2, alpha(m_j=1/2) = alpha_0 - 0.8 alpha_2 < 0 while alpha_0 > 0.

Method (spec 07 §2.2, NORMATIVE): the m_j-resolved second-order sum
Eq. (7.1) with matrix elements factorized per Eq. (7.2) into spec 02
consensus-grade radial integrals x spec 03 angular factors. The Yerokhin
6j closed forms Eq. (7.5) are the redundant cross-check (must agree with
the m_j fit to 1e-9, pure algebra). Stark maps: real-symmetric H(E) =
diag(E0) + E_field * D, scipy.linalg.eigh per m_j block (spec 07 §2.6),
adiabatic tracking by eigenvector overlap with adaptive bisection.

Radial-integral policy: every alkali pair element is evaluated by BOTH
Method A (MSD94 model-potential Numerov) and Method B (pure-Coulomb
Numerov) on a shared grid, with the spec 02 §6 per-regime spread ceilings
enforced (IntegrityError on breach — no averaging over a failed consensus).
This reproduces the ``radial_matrix_element_consensus`` contract while
solving each basis state once per method: the consensus entry point fixes
r_outer per PAIR, which would re-solve the target state O(window) times
per polarizability. Equivalence with the public consensus entry point is
pinned by a test (agreement <= 1e-9; the Kaulakys third method is exercised
there). Matrix elements are cached module-wide (floats); wavefunction
caches are cleared after each window/basis build (memory bound).

Integrity-audit compliance (docs/spec/00-integrity-audit.md §3 items 24-30):
  24. perturbative alpha refuses n < 10 (EngineValidityError — continuum
      omission; H 1s misses ~19 %, documented by RS-07-16).
  25. quasi-degenerate partner (E_mix < 50 V/m) or high-l (l >= 4) target
      -> DegeneracyError; high-l states must use the manifold map. The
      E_mix gate is enforced by the SHARED helper _assemble_alpha_checked,
      so alpha_perturbative and alpha_dynamic cannot drift apart, and
      E_mix_Vm ships on both result objects.
  26. dynamic alpha inside the resonance guard band -> ResonanceError.
  27. dynamic results always carry the TRK completeness S and the
      crossover bound |1-S| e^2/(m_e omega^2) — the truncated sum is never
      presented bare (DynamicPolarizability fields).
  28. no Rb nD closed-form fit is provided (MISSING by declaration).
  29. hyperpolarizability gamma is a DIAGNOSTIC output only.
  30. fields above F_ion: the map attaches an ``above_F_ion`` validity flag
      (hand-off with cap); the perturbative shift helper refuses beyond its
      quadratic validity cap.
  Screening (item 20 / R12): uncalibrated ScreeningModel refuses absolute
      external-NEF output unless ``estimate_ok=True``.

Sources for numeric contracts (transcribed in tests with tags):
  - Yerokhin, Buhmann, Fritzsche & Surzhykov, PRA 94, 032503 (2016)
    [arXiv:1608.04515] — alpha_0/alpha_2 anchors, Eqs. (10)-(11) closed
    forms [VERIFIED, spec 07 §2.3/§3.2; incl. the a0^5 label anomaly note].
  - O'Sullivan & Stoicheff, PRA 31, 2718 (1985) — Rb nS fit Eq. (7.6)
    [VERIFIED]. The 1986 nD fit is paywalled: MISSING, not invented.
  - Bethe & Salpeter §51-52 — hydrogen Eqs. (7.7)-(7.8) [VERIFIED].
  - Zimmerman, Littman, Kash & Kleppner, PRA 20, 2251 (1979) — the
    canonical alkali Stark-map method [VERIFIED as method reference].
  - Jau & Carter, PRApplied 13, 054034 (2020) — screening anchor
    [VERIFIED abstract level; parameters owned by spec 05].
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment, minimize_scalar

from .angular import angular_factor, reduced_dipole_j
from .atom import SPECIES, Species, binding_energy_hz
from .constants import (A0, AU_DIPOLE, AU_EFIELD, AU_POLARIZABILITY, E_CHARGE,
                        E_H, H, HBAR, M_E,
                        polarizability_au_to_hz_per_v2_cm2)
from .provenance import IntegrityError
from .radial import (_ab_ceiling, clear_radial_cache, radial_matrix_element,
                     radial_me_gordon, radial_wavefunction)
from .wigner import wigner_6j

__all__ = [
    "EngineValidityError", "DegeneracyError", "ResonanceError",
    "Polarizability", "DynamicPolarizability",
    "alpha_perturbative", "alpha_scalar_tensor", "alpha_dynamic",
    "alpha_ponderomotive", "hydrogen_1s_discrete_alpha_au",
    "StarkBasis", "StarkMatrices", "StarkMapResult", "MapCurvature",
    "ManifoldGap",
    "basis_states", "build_stark_matrices", "stark_map",
    "stark_map_curvature", "alpha_map_curvature", "manifold_gap",
    "hydrogen_manifold_slopes_Cm", "hydrogen_stark_energy_J",
    "inglis_teller_field_Vm", "classical_ionization_field_Vm",
    "stark_shift_J", "ac_stark_shift_J",
    "stark_responsivity_hz_per_Vm", "unbiased_min_field_Vm",
    "optimal_bias_Vm", "nef_dc_Vm_sqrtHz",
    "ScreeningModel", "screening_transfer", "nef_external_Vm_sqrtHz",
    "clear_stark_cache",
]


class EngineValidityError(IntegrityError):
    """Requested quantity is outside the engine's validity domain
    (audit §3 item 24: perturbative alpha for n < 10 — continuum omission)."""


class DegeneracyError(IntegrityError):
    """Perturbation theory through a (quasi-)degeneracy (audit §3 item 25):
    use the manifold diagonalization, never Eq. (7.1)."""


class ResonanceError(IntegrityError):
    """Dynamic alpha requested inside the resonance guard band (audit §3
    item 26): dressed-state physics belongs to spec 06/08."""


# Quasi-degeneracy floor (spec 07 §5 API note: |dE| < 100 |coupling x 1 V/m|
# <=> E_mix < 50 V/m) and the default dynamic-alpha guard band (spec 07 §4.5
# requires 10 x max(Gamma_k, Omega_Rabi); Rydberg Gamma including BBR is
# <~ 2pi x 100 kHz, so 2pi x 1 MHz is a conservative floor — callers supply
# the true 10 x max(Gamma, Omega) when driving fields are involved).
E_MIX_FLOOR_VM = 50.0
RESONANCE_GUARD_DEFAULT_RAD_S = 2.0 * math.pi * 1e6


# ---------------------------------------------------------------------------
# Validity fields (spec 07 §2.5, Eq. 7.9) — derived from scipy.constants
# ---------------------------------------------------------------------------

def inglis_teller_field_Vm(n: int) -> float:
    """Inglis-Teller field F_IT = F0/(3 n^5) [V/m], spec 07 Eq. (7.9).

    F0 = E_h/(e a0) from rydsim.constants (never typed). Check values:
    n = 30 -> 70.54 V/cm (Hogan arXiv:1603.04432: '71 V/cm' [VERIFIED]).
    Beyond F_IT perturbation theory is dead; only the map is valid.
    """
    if n < 1 or int(n) != n:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    return AU_EFIELD / (3.0 * n**5)


def classical_ionization_field_Vm(n: int) -> float:
    """Classical ionization field F_ion = F0/(16 n^4) [V/m] (spec 07 §2.5).

    Saddle point of -e^2/(4 pi eps0 r) - e E z at the zero-field energy
    (analytic, VERIFIED). Validity ceiling of the bound-state model:
    n = 35 -> 214 V/cm.
    """
    if n < 1 or int(n) != n:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    return AU_EFIELD / (16.0 * n**4)


# ---------------------------------------------------------------------------
# Hydrogen closed forms (spec 07 §2.5, Eqs. 7.7-7.8 — exact generators)
# ---------------------------------------------------------------------------

def hydrogen_stark_energy_J(n: int, q: int, m: int, field_Vm: float, *,
                            order: int = 2) -> float:
    """Exact hydrogen Stark energy [J] in parabolic quantum numbers.

    E = -E_h/(2 n^2) + (3/2) n q (e a0) E  -  (1/16) n^4 (17 n^2 - 3 q^2
    - 9 m^2 + 19) F_au^2 E_h  (spec 07 Eqs. 7.7-7.8, Bethe-Salpeter
    [VERIFIED]; infinite-nuclear-mass convention — the benchmark oracle).
    q = n1 - n2; validity requires n1 = (n - 1 - |m| + q)/2 to be a
    non-negative integer <= n - 1 - |m| (raises ValueError otherwise —
    e.g. (n, q, m) = (10, 0, 0) does NOT exist: n - 1 - |m| = 9 is odd, so
    q is odd; see RS-07-04 test note). order = 1 or 2.
    """
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order!r}")
    if n < 1 or int(n) != n:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    n_perp = n - 1 - abs(m)
    n1_twice = n_perp + q
    if n_perp < 0 or n1_twice % 2 or not 0 <= n1_twice // 2 <= n_perp:
        raise ValueError(
            f"(n={n}, q={q}, m={m}) is not a parabolic hydrogen state: "
            f"need n1 = (n-1-|m|+q)/2 integer in [0, {max(n_perp, 0)}]")
    f_au = field_Vm / AU_EFIELD
    e_au = -0.5 / n**2 + 1.5 * n * q * f_au
    if order == 2:
        e_au -= (n**4 / 16.0) * (17.0 * n**2 - 3.0 * q**2 - 9.0 * m**2
                                 + 19.0) * f_au**2
    return e_au * E_H


def _hydrogen_angular(l: int, lp: int, m: int) -> float:
    """<l' m|cos(theta)|l m> for the spinless hydrogen basis, exact closed
    form [dimensionless] (standard; VERIFIED against spec 03 machinery for
    m = 0 via the l-basis reduction)."""
    if lp == l + 1:
        return math.sqrt(((l + 1) ** 2 - m**2)
                         / ((2.0 * l + 1.0) * (2.0 * l + 3.0)))
    if lp == l - 1:
        return math.sqrt((l**2 - m**2) / ((2.0 * l - 1.0) * (2.0 * l + 1.0)))
    return 0.0


def hydrogen_manifold_slopes_Cm(n: int, m: int = 0) -> np.ndarray:
    """Linear Stark slopes of the hydrogen n-manifold [C m], ascending.

    Eigenvalues of the within-manifold e z matrix (Gordon diagonal-n radial
    elements x exact angular factors): equal (3/2) n q e a0 with
    q in {-(n-1-|m|), ..., +(n-1-|m|)} step 2 (spec 07 Eq. 7.7, exact).
    dE/dE_field of the map at small field; benchmark RS-07-03.
    """
    ls = [l for l in range(abs(m), n)]
    nl = len(ls)
    d = np.zeros((nl, nl))
    for i, l in enumerate(ls):
        for k, lp in enumerate(ls):
            if abs(l - lp) == 1:
                d[i, k] = (radial_me_gordon(n, l, n, lp)
                           * _hydrogen_angular(l, lp, m))
    return np.sort(np.linalg.eigvalsh(d)) * AU_DIPOLE


def hydrogen_1s_discrete_alpha_au(n_max: int = 100) -> tuple[float, float]:
    """DIAGNOSTIC: discrete-bound-only alpha(H 1s) [a.u.] and its discrete
    TRK sum, from exact Gordon matrix elements (spec 07 §3.3 / RS-07-16).

    Documents the continuum omission of any bound-state sum: the exact
    alpha(H 1s) = 9/2 a.u. includes the continuum; the discrete series
    saturates at 3.6633 a.u. = 0.8141 x 9/2 (discrete TRK sum 0.5650).
    This function deliberately bypasses the n >= 10 engine floor — it IS
    the documentation of why that floor exists (audit §3 item 24). The
    bound-series tail beyond n_max is ~3.2/n_max^2 a.u. (measured; the
    spec's printed 5.4/n_max^3 underestimates it — see test note).
    """
    alpha = 0.0
    f_sum = 0.0
    for n in range(2, n_max + 1):
        r2 = radial_me_gordon(1, 0, n, 1) ** 2 / 3.0    # |<n p, m=0|z|1s>|^2
        de = 0.5 * (1.0 - 1.0 / n**2)                    # Hartree
        alpha += 2.0 * r2 / de
        f_sum += 2.0 * de * r2
    return alpha, f_sum


# ---------------------------------------------------------------------------
# Shared-grid dual-method radial matrix elements (spec 02 contract; cached)
# ---------------------------------------------------------------------------

_ME_CACHE: dict[tuple, tuple[float, float]] = {}


def clear_stark_cache() -> None:
    """Drop the cached pair matrix elements (and the radial solution cache)."""
    _ME_CACHE.clear()
    clear_radial_cache()


def _pair_me_a0(sp: Species, s1: tuple[int, int, float],
                s2: tuple[int, int, float], n_top: int) -> tuple[float, float]:
    """Dual-method radial dipole element <s1| r |s2> [a0] with spread.

    Methods A (MSD94) + B (pure Coulomb) on the shared grid r_outer =
    2 n_top (n_top + 15) a0, checked against the spec 02 §6 per-regime
    ceilings (same ``_ab_ceiling`` the consensus entry point uses);
    IntegrityError on breach — never average a failed consensus. Returns
    (signed Method-A value, relative |A|-|B| spread). Cached on the state
    pair; the cached value is grid-insensitive to <= 1e-9 (outer tails are
    exponentially negligible per spec 02 §4.1/B14, pinned by the
    consensus-equivalence test).
    """
    key = (sp.name, s1, s2) if s1 <= s2 else (sp.name, s2, s1)
    hit = _ME_CACHE.get(key)
    if hit is not None:
        return hit
    r_outer = 2.0 * n_top * (n_top + 15.0)
    a = radial_wavefunction(sp, *s1, r_outer=r_outer, method="model_potential")
    b = radial_wavefunction(sp, *s2, r_outer=r_outer, method="model_potential")
    me_a = radial_matrix_element(a, b, 1)
    a2 = radial_wavefunction(sp, *s1, r_outer=r_outer, method="coulomb")
    b2 = radial_wavefunction(sp, *s2, r_outer=r_outer, method="coulomb")
    me_b = radial_matrix_element(a2, b2, 1)
    scale = abs(me_a) or max(abs(me_b), 1e-300)
    spread = abs(abs(me_a) - abs(me_b)) / scale
    diff_abs = abs(abs(me_a) - abs(me_b))
    orbit_scale = a.nu * b.nu                    # ~<r> scale of the pair [a0]
    # Three gate regimes (all measured this implementation; the relative
    # spread ships with the result in every case):
    #  1. deeply suppressed elements (|ME| < 1e-3 of the nu1*nu2 orbit
    #     scale, e.g. Rb 30D->31F at 0.01 a0): the relative spread is
    #     meaningless on a near-zero integral — gate the ABSOLUTE
    #     deviation at 1e-4 of the orbit scale (measured floor ~3e-8).
    #  2. l >= 4 pairs: both potentials are pure Coulomb apart from the
    #     ~1e-13 spin-orbit term, so A-vs-B is NOT a model cross-check —
    #     the difference is divergence-guard truncation noise (~1.2e-4 of
    #     scale on near-circular Delta-n=2 pairs) — gate at 1e-3 of the
    #     orbit scale (a real phase/convention bug moves O(scale)).
    #  3. everything else (the model-sensitive low-l pairs every
    #     polarizability sum uses): the spec 02 §6 B8 ceilings verbatim.
    if abs(me_a) < 1e-3 * orbit_scale:
        if diff_abs > 1e-4 * orbit_scale:
            raise IntegrityError(
                f"A-vs-B agreement FAILED on suppressed pair {sp.name} "
                f"{s1}->{s2}: |A|-|B| = {diff_abs:.3e} a0 > 1e-4 x nu1 nu2 "
                f"= {1e-4 * orbit_scale:.3e} a0 — refusing")
    elif min(s1[1], s2[1]) >= 4:
        if diff_abs > 1e-3 * orbit_scale:
            raise IntegrityError(
                f"A-vs-B agreement FAILED on high-l pair {sp.name} "
                f"{s1}->{s2}: |A|-|B| = {diff_abs:.3e} a0 > 1e-3 x nu1 nu2 "
                f"= {1e-3 * orbit_scale:.3e} a0 (solver noise floor is "
                "~1.2e-4 of scale) — refusing")
    else:
        ceiling = _ab_ceiling(min(a.nu, b.nu), abs(me_a))
        if spread > ceiling:
            raise IntegrityError(
                f"A-vs-B radial consensus FAILED for {sp.name} {s1}->{s2}: "
                f"spread {spread:.2e} > ceiling {ceiling:.0e} "
                f"(A={me_a:.6e}, B={me_b:.6e} a0) — refusing to average "
                "(spec 02 §6 B8 / audit §3 item 8)")
    _ME_CACHE[key] = (me_a, spread)
    return _ME_CACHE[key]


def _resolve_species(species: Species | str) -> Species:
    if isinstance(species, Species):
        return species
    sp = SPECIES.get(species)
    if sp is None:
        raise IntegrityError(
            f"unknown species {species!r}: no sourced data (spec 01); "
            f"available: {sorted(SPECIES)} (+ 'H' for map benchmarks only)")
    return sp


def _species_floor(sp: Species) -> int:
    """Per-species hard validity floor of the quantum-defect machinery."""
    return max(row.n_min_hard for row in sp.series.values())


def _partner_series(l: int, j: float) -> list[tuple[int, float]]:
    """E1-allowed (l', j') partner series: l' = l +- 1, |j' - j| <= 1."""
    out = []
    for lp in (l - 1, l + 1):
        if lp < 0:
            continue
        for jp in (lp - 0.5, lp + 0.5):
            if jp <= 0 or abs(jp - j) > 1.0 + 1e-9:
                continue
            out.append((lp, jp))
    return out


def _window_partner_terms(sp: Species, n: int, l: int, j: float,
                          n_window: int) -> list[tuple]:
    """All dipole-coupled partner terms of |n l j> in the n' window.

    Returns [(n', l', j', dE_J, R_a0, spread_rel), ...] with dE = E_k - E_v
    [J] from spec 01 binding energies (E_I cancels exactly) and R from the
    dual-method pair machinery [a0]. m_j-independent — one radial window
    serves every m_j and both static and dynamic sums (spec 07 §4.1).
    """
    eb_v = binding_energy_hz(sp, n, l, j)
    floor = _species_floor(sp)
    n_top = n + n_window
    terms = []
    for lp, jp in _partner_series(l, j):
        for n_p in range(max(floor, lp + 1, n - n_window), n + n_window + 1):
            de_j = (eb_v - binding_energy_hz(sp, n_p, lp, jp)) * H
            r_a0, spread = _pair_me_a0(sp, (n, l, j), (n_p, lp, jp), n_top)
            terms.append((n_p, lp, jp, de_j, r_a0, spread))
    return terms


# ---------------------------------------------------------------------------
# Perturbative polarizability (spec 07 §2.2-2.3, NORMATIVE Eq. 7.1)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Polarizability:
    """m_j-resolved polarizability with the three-representation unit
    tripwire (spec 07 §3.1: all representations ALWAYS populated).

    alpha_SI [C^2 m^2 / J]; alpha_au [a.u.]; alpha_Hz_per_Vcm2 = alpha/h
    [Hz/(V/cm)^2] — all for the REQUESTED m_j at the requested omega.
    alpha0_au / alpha2_au: Eq. (7.4) decomposition [a.u.] (alpha2 = 0.0
    exactly for j <= 1/2, benchmark RS-07-11). sigma_rel: propagated
    radial-spread + truncation-tail uncertainty (spec 07 §4.1, diagonal
    quadrature). E_mix_Vm: perturbative validity field cap min_k |dE| /
    (2 |e z_k|) [V/m] — quadratic results are meaningful only for fields
    << E_mix. tail_rel: relative contribution of the outermost n' ring
    (truncation indicator). trk_completeness: discrete TRK sum S over the
    window (spec 07 §2.7). converged: True only when the doubling check
    ran (spec 07 §4.1). mj_alphas_SI: alpha(m_j) for every m_j in
    (1/2 .. j) at the same omega [C^2 m^2/J].
    """

    species: str
    n: int
    l: int
    j: float
    mj: float
    omega_rad_s: float
    alpha_SI: float
    alpha_au: float
    alpha_Hz_per_Vcm2: float
    alpha0_au: float
    alpha2_au: float
    sigma_rel: float
    E_mix_Vm: float
    n_window: int
    converged: bool
    trk_completeness: float
    tail_rel: float
    mj_alphas_SI: tuple[float, ...]


def _bracket(j: float, mj: float) -> float:
    """Tensor bracket [3 m_j^2 - j(j+1)] / [j(2j-1)] of Eq. (7.4)."""
    return (3.0 * mj * mj - j * (j + 1.0)) / (j * (2.0 * j - 1.0))


def _validate_target(sp: Species, n: int, l: int, j: float,
                     mj: float) -> None:
    if int(l) != l or l < 0 or n < l + 1:
        raise ValueError(f"invalid (n={n}, l={l})")
    if abs(abs(j - l) - 0.5) > 1e-9 or j <= 0:
        raise ValueError(f"j must equal l +- 1/2 (got l={l}, j={j})")
    if abs(2 * mj - round(2 * mj)) > 1e-9 or abs(mj) > j + 1e-9:
        raise ValueError(f"m_j={mj} invalid for j={j}")
    if n < 10:
        raise EngineValidityError(
            f"perturbative alpha refused for n = {n} < 10: continuum and "
            "core contributions are not represented (H 1s misses ~19 %, "
            "spec 07 §4.1/RS-07-16; audit §3 item 24)")
    if l >= 4:
        raise DegeneracyError(
            f"l = {l} >= 4 target is quasi-degenerate within its manifold: "
            "Eq. (7.1) diverges — use the manifold diagonalization "
            "(spec 07 §2.5-2.6; audit §3 item 25)")


def _assemble_alpha(terms: list[tuple], l: int, j: float, mj: float,
                    omega_rad_s: float) -> tuple[float, float, float, float]:
    """(alpha_SI, sigma_abs, E_mix_Vm, tail_ring_abs) for one m_j.

    alpha = 2 e^2 sum |<k|z|v>|^2/(E_k - E_v) (Eq. 7.1); dynamic terms
    carry the omega_k^2/(omega_k^2 - omega^2) factor (Eq. 7.10).
    """
    a_key = {}
    total = 0.0
    var = 0.0
    e_mix = math.inf
    tail = 0.0
    n_edge_lo = min(t[0] for t in terms)
    n_edge_hi = max(t[0] for t in terms)
    for n_p, lp, jp, de_j, r_a0, spread in terms:
        if (lp, jp) not in a_key:
            a_key[(lp, jp)] = (angular_factor(l, j, mj, lp, jp, mj, 0)
                               if jp >= abs(mj) else 0.0)
        ang = a_key[(lp, jp)]
        if ang == 0.0:
            continue
        dz = E_CHARGE * A0 * r_a0 * ang            # C m
        term = 2.0 * dz * dz / de_j
        if omega_rad_s != 0.0:
            wk = de_j / HBAR
            term *= wk * wk / (wk * wk - omega_rad_s**2)
        total += term
        var += (2.0 * term * spread) ** 2
        e_mix = min(e_mix, abs(de_j) / (2.0 * abs(dz)))
        if n_p in (n_edge_lo, n_edge_hi):
            tail += term
    return total, math.sqrt(var), e_mix, abs(tail)


def _check_e_mix(e_mix: float) -> None:
    """Audit §3 item 25 quasi-degeneracy refusal, shared by every consumer
    of the second-order sum (static AND dynamic).

    Second-order PT through a quasi-degeneracy is divergent, not merely
    inaccurate: one denominator becomes comparable to the field-induced
    coupling, so the sum is arbitrary. Raise, never guess (spec 07 §4.1).
    """
    if e_mix < E_MIX_FLOOR_VM:
        raise DegeneracyError(
            f"quasi-degenerate partner: E_mix = {e_mix:.1f} V/m < "
            f"{E_MIX_FLOOR_VM} V/m — perturbative alpha invalid, use the "
            "Stark map (spec 07 §4.1; audit §3 item 25)")


def _assemble_alpha_checked(terms: list[tuple], l: int, j: float, mj: float,
                            omega_rad_s: float) -> tuple[float, float, float,
                                                         float]:
    """``_assemble_alpha`` with the item-25 quasi-degeneracy gate applied.

    The single entry point every public alpha must go through: E_mix is a
    ZERO-FIELD property of the partner set, so the same cap governs the
    static and the dynamic sums (they share the terms and the denominators).
    """
    total, sig, e_mix, tail = _assemble_alpha(terms, l, j, mj, omega_rad_s)
    _check_e_mix(e_mix)
    return total, sig, e_mix, tail


def _decompose(j: float, mj_alphas: dict[float, float]) -> tuple[float, float]:
    """(alpha0_SI, alpha2_SI) from the m_j set via Eq. (7.4) least squares.

    j <= 1/2: alpha2 = 0.0 EXACTLY (angular identity, RS-07-11). j >= 3/2:
    least squares over all m_j; any residual beyond 1e-6 relative is an
    m_j^4 term that cannot exist at second order -> IntegrityError
    (spec 07 §2.3 implementation rule; RS-07-18).
    """
    mjs = sorted(mj_alphas)
    if j <= 0.5 + 1e-9:
        return mj_alphas[mjs[0]], 0.0
    mat = np.array([[1.0, _bracket(j, m)] for m in mjs])
    y = np.array([mj_alphas[m] for m in mjs])
    coef, *_ = np.linalg.lstsq(mat, y, rcond=None)
    scale = max(abs(coef[0]), np.max(np.abs(y)))
    resid = float(np.max(np.abs(mat @ coef - y))) / scale
    if resid > 1e-6:
        raise IntegrityError(
            f"alpha(m_j) violates the alpha0 + alpha2*bracket form at "
            f"{resid:.1e} relative (> 1e-6): an m_j^4 residual at second "
            "order is a bug (spec 07 §2.3 / RS-07-18)")
    return float(coef[0]), float(coef[1])


def _resonance_check(terms: list[tuple], omega: np.ndarray | float,
                     guard_rad_s: float) -> None:
    om = np.atleast_1d(np.asarray(omega, dtype=float))
    if not om.size or np.all(om == 0.0):
        return
    wk = np.array([abs(t[3]) / HBAR for t in terms])
    dist = np.min(np.abs(np.abs(om)[:, None] - wk[None, :]), axis=1)
    bad = np.abs(om) > 0.0
    bad &= dist < guard_rad_s
    if np.any(bad):
        raise ResonanceError(
            f"omega = {om[bad][0]:.4e} rad/s is within the resonance guard "
            f"band ({guard_rad_s:.2e} rad/s) of a coupled transition: the "
            "dressed-state treatment of spec 06/08 is normative there "
            "(spec 07 §4.5; audit §3 item 26)")


def alpha_perturbative(species: Species | str, n: int, l: int, j: float,
                       mj: float, *, n_window: int = 15,
                       omega_rad_s: float = 0.0,
                       check_convergence: bool = False,
                       resonance_guard_rad_s: float =
                       RESONANCE_GUARD_DEFAULT_RAD_S) -> Polarizability:
    """m_j-resolved second-order polarizability, spec 07 Eq. (7.1)/(7.10).

    NORMATIVE path: alpha(v, m_j) = 2 e^2 sum_k |<k|z|v>|^2/(E_k - E_v)
    [C^2 m^2/J] over l' = l +- 1 partners in the n' window (defaults
    n_window = 15, spec 07 §4.1), radial elements from the dual-method
    spec 02 machinery, angular factors from spec 03. omega_rad_s != 0
    gives the dynamic alpha (Eq. 7.10, angular rad/s per lock #2).

    Refuses (audit §3 items 24-26): n < 10 (EngineValidityError), l >= 4
    or a quasi-degenerate partner with E_mix < 50 V/m (DegeneracyError),
    omega inside the guard band of any coupled transition (ResonanceError;
    default guard 2pi x 1 MHz — pass 10 x max(Gamma, Omega_Rabi) when
    driving fields are present, spec 07 §4.5).

    check_convergence=True re-evaluates with the window doubled and raises
    IntegrityError unless |delta alpha/alpha| < 1e-4 (spec 07 §4.1 —
    convergence is demonstrated, never assumed); the result then carries
    converged=True and the doubled-window value.
    """
    sp = _resolve_species(species)
    _validate_target(sp, n, l, j, mj)
    terms = _window_partner_terms(sp, n, l, j, n_window)
    _resonance_check(terms, omega_rad_s, resonance_guard_rad_s)

    mj_all = [0.5 + k for k in range(int(j + 0.5))]
    alphas = {}
    sig = e_mix = tail = None
    for m in mj_all:
        a, s, em, tl = _assemble_alpha(terms, l, j, m, omega_rad_s)
        alphas[m] = a
        if abs(m - abs(mj)) < 1e-9:
            sig, e_mix, tail = s, em, tl
    alpha_si = alphas[abs(mj)]
    _check_e_mix(e_mix)
    a0_si, a2_si = _decompose(j, alphas)

    trk = 0.0
    for n_p, lp, jp, de_j, r_a0, spread in terms:
        ang = angular_factor(l, j, mj, lp, jp, mj, 0) if jp >= abs(mj) else 0.0
        trk += 2.0 * M_E * (de_j / HBAR) * (A0 * r_a0 * ang) ** 2 / HBAR

    converged = False
    if check_convergence:
        wide = _window_partner_terms(sp, n, l, j, 2 * n_window)
        a_wide, sig, e_mix, tail = _assemble_alpha(wide, l, j, abs(mj),
                                                   omega_rad_s)
        if abs(a_wide / alpha_si - 1.0) > 1e-4:
            raise IntegrityError(
                f"n'-window not converged: doubling n_window {n_window} -> "
                f"{2 * n_window} moved alpha by "
                f"{abs(a_wide / alpha_si - 1.0):.2e} (> 1e-4) — fail loudly "
                "(spec 07 §4.1)")
        alphas = {}
        for m in mj_all:
            alphas[m], *_ = _assemble_alpha(wide, l, j, m, omega_rad_s)
        alpha_si = alphas[abs(mj)]
        a0_si, a2_si = _decompose(j, alphas)
        converged = True
    clear_radial_cache()

    scale = abs(alpha_si) or 1.0
    return Polarizability(
        species=sp.name, n=n, l=l, j=j, mj=mj, omega_rad_s=omega_rad_s,
        alpha_SI=alpha_si, alpha_au=alpha_si / AU_POLARIZABILITY,
        alpha_Hz_per_Vcm2=polarizability_au_to_hz_per_v2_cm2(
            alpha_si / AU_POLARIZABILITY),
        alpha0_au=a0_si / AU_POLARIZABILITY,
        alpha2_au=a2_si / AU_POLARIZABILITY,
        sigma_rel=(sig + tail) / scale,
        E_mix_Vm=e_mix, n_window=n_window, converged=converged,
        trk_completeness=trk, tail_rel=tail / scale,
        mj_alphas_SI=tuple(alphas[m] for m in mj_all))


def alpha_scalar_tensor(species: Species | str, n: int, l: int, j: float, *,
                        n_window: int = 15, omega_rad_s: float = 0.0,
                        method: Literal["mj-fit", "6j"] = "mj-fit",
                        ) -> tuple[float, float]:
    """(alpha0_au, alpha2_au) [a.u.], spec 07 §2.3.

    method='mj-fit' (NORMATIVE): direct m_j-resolved sums Eq. (7.1)
    decomposed via Eq. (7.4). method='6j': ALSO evaluates the Yerokhin
    closed forms Eq. (7.5) (transcribed from arXiv:1608.04515 Eqs. 10-11
    [VERIFIED]) and raises IntegrityError unless both agree to 1e-9
    relative (pure algebra); the m_j-fit values are returned either way.
    Sign anchors (tested on alpha_0/alpha_2, never alpha(m_j) — R-19):
    alpha2(Rb nP3/2) < 0, alpha2(Rb nD) > 0, alpha0(Cs nD5/2) < 0.
    """
    pol = alpha_perturbative(species, n, l, j, 0.5, n_window=n_window,
                             omega_rad_s=omega_rad_s)
    if method == "mj-fit":
        return pol.alpha0_au, pol.alpha2_au
    if method != "6j":
        raise ValueError(f"unknown method {method!r}")
    sp = _resolve_species(species)
    terms = _window_partner_terms(sp, n, l, j, n_window)
    s0 = 0.0
    s2 = 0.0
    red_cache: dict[tuple, float] = {}
    for n_p, lp, jp, de_j, r_a0, spread in terms:
        if (lp, jp) not in red_cache:
            red_cache[(lp, jp)] = reduced_dipole_j(l, j, lp, jp)
        red = red_cache[(lp, jp)]
        if red == 0.0:
            continue
        d2 = (E_CHARGE * A0 * r_a0 * red) ** 2
        f_dyn = 1.0
        if omega_rad_s != 0.0:
            wk = de_j / HBAR
            f_dyn = wk * wk / (wk * wk - omega_rad_s**2)
        s0 += d2 / de_j * f_dyn
        s2 += ((-1.0) ** round(j + jp) * wigner_6j(1, 1, 2, j, j, jp)
               * d2 / de_j * f_dyn)
    clear_radial_cache()
    a0_6j = 2.0 / (3.0 * (2.0 * j + 1.0)) * s0 / AU_POLARIZABILITY
    pref = math.sqrt(40.0 * j * (2.0 * j - 1.0)
                     / (3.0 * (j + 1.0) * (2.0 * j + 1.0) * (2.0 * j + 3.0)))
    a2_6j = pref * s2 / AU_POLARIZABILITY
    scale = max(abs(pol.alpha0_au), abs(pol.alpha2_au), 1e-300)
    if (abs(a0_6j - pol.alpha0_au) > 1e-9 * scale
            or abs(a2_6j - pol.alpha2_au) > 1e-9 * scale):
        raise IntegrityError(
            f"6j closed form disagrees with the m_j fit beyond 1e-9: "
            f"alpha0 {a0_6j:.9e} vs {pol.alpha0_au:.9e}, alpha2 {a2_6j:.9e} "
            f"vs {pol.alpha2_au:.9e} a.u. — phase-convention regression "
            "(spec 07 §2.3)")
    return pol.alpha0_au, pol.alpha2_au


@dataclass(frozen=True)
class DynamicPolarizability:
    """Dynamic alpha(omega) sweep (spec 07 §2.7, Eq. 7.10) with the TRK
    completeness bookkeeping of audit §3 item 27: the truncated sum is
    never presented without S and the crossover bound.

    alpha_SI: alpha(v, m_j; omega) [C^2 m^2/J] per omega. trk_completeness:
    discrete TRK sum S over the window (dimensionless; the crossover model
    uncertainty is |1 - S| e^2/(m_e omega^2), spec 07 §2.7 — for
    omega >> all coupled omega_k, alpha -> -S e^2/(m_e omega^2), so the
    TRUNCATED sum can only reproduce the ponderomotive limit to |1 - S|).
    omega_k_max_rad_s: highest coupled transition frequency in the window.
    E_mix_Vm: the perturbative validity field cap min_k |dE_k|/(2|e z_k|)
    [V/m] of the SAME partner set the static sum uses — spec 07 §4.1
    requires the result object to carry it, so a caller can reproduce the
    audit §3 item 25 gate downstream (the gate itself is enforced inside
    ``alpha_dynamic``).
    """

    species: str
    n: int
    l: int
    j: float
    mj: float
    omega_rad_s: np.ndarray
    alpha_SI: np.ndarray
    alpha_static_SI: float
    trk_completeness: float
    omega_k_max_rad_s: float
    n_window: int
    E_mix_Vm: float


def alpha_dynamic(species: Species | str, n: int, l: int, j: float, mj: float,
                  omega_rad_s: np.ndarray | Sequence[float], *,
                  n_window: int = 15,
                  resonance_guard_rad_s: float =
                  RESONANCE_GUARD_DEFAULT_RAD_S) -> DynamicPolarizability:
    """Vectorized dynamic polarizability alpha(v, m_j; omega), Eq. (7.10).

    alpha(omega) = (2/hbar) sum_k omega_k |<k|d_z|v>|^2/(omega_k^2 -
    omega^2) [C^2 m^2/J]; omega in rad/s (lock #2). Same refusals as
    ``alpha_perturbative`` — n < 10 (EngineValidityError), l >= 4 target
    (DegeneracyError) and a quasi-degenerate partner with E_mix < 50 V/m
    (DegeneracyError, audit §3 item 25: the SAME denominators appear in
    Eq. 7.10, so the static cap governs the dynamic sum) — plus the
    per-omega resonance guard. The result carries E_mix_Vm and the
    discrete TRK completeness S; per spec 07 §2.7, in the crossover use
    the bound |1-S| e^2/(m_e omega^2) and for omega >> omega_k switch to
    ``alpha_ponderomotive`` when S < 0.98.
    alpha(omega -> 0) reproduces the static value (RS-07-14).
    """
    sp = _resolve_species(species)
    _validate_target(sp, n, l, j, mj)
    om = np.asarray(omega_rad_s, dtype=float)
    terms = _window_partner_terms(sp, n, l, j, n_window)
    _resonance_check(terms, om, resonance_guard_rad_s)
    # audit §3 item 25 — evaluated on the static (omega = 0) denominators
    # BEFORE any coefficient is built, so no invalid number is ever formed
    _, _, e_mix, _ = _assemble_alpha_checked(terms, l, j, abs(mj), 0.0)
    coeff = []
    wks = []
    trk = 0.0
    for n_p, lp, jp, de_j, r_a0, spread in terms:
        ang = angular_factor(l, j, mj, lp, jp, mj, 0) if jp >= abs(mj) else 0.0
        if ang == 0.0:
            continue
        dz = E_CHARGE * A0 * r_a0 * ang
        coeff.append(2.0 * dz * dz / de_j)
        wks.append(de_j / HBAR)
        trk += 2.0 * M_E * (de_j / HBAR) * (A0 * r_a0 * ang) ** 2 / HBAR
    clear_radial_cache()
    c = np.array(coeff)
    wk = np.array(wks)
    alpha = np.sum(c[None, :] * wk[None, :] ** 2
                   / (wk[None, :] ** 2 - om.reshape(-1, 1) ** 2), axis=1)
    alpha = alpha.reshape(np.shape(om))
    return DynamicPolarizability(
        species=sp.name, n=n, l=l, j=j, mj=mj,
        omega_rad_s=om, alpha_SI=alpha, alpha_static_SI=float(np.sum(c)),
        trk_completeness=trk, omega_k_max_rad_s=float(np.max(np.abs(wk))),
        n_window=n_window, E_mix_Vm=e_mix)


def alpha_ponderomotive(omega_rad_s: np.ndarray | float) -> np.ndarray | float:
    """Free-electron/ponderomotive polarizability -e^2/(m_e omega^2)
    [C^2 m^2/J], spec 07 Eq. (7.11) (TRK-saturated high-frequency limit;
    the AC shift +e^2 E0^2/(4 m_e omega^2) = U_p is UP, alpha < 0).
    omega in rad/s.
    """
    om = np.asarray(omega_rad_s, dtype=float)
    out = -E_CHARGE**2 / (M_E * om**2)
    return out.item() if out.ndim == 0 else out


# ---------------------------------------------------------------------------
# Stark map by diagonalization (spec 07 §2.6, NORMATIVE)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StarkBasis:
    """Fixed-m_j basis window for the Stark map (spec 07 §2.6).

    species: 'Rb85' | 'Rb87' | 'Cs133', or 'H' (spinless hydrogen test
    backend: exact Gordon radial elements, m = mj integer, infinite-mass
    energies — the benchmark oracle). l_max = None -> full manifolds
    (l up to n-1). Alkali m_j > 0 (with no B field the +-m_j blocks are
    degenerate — compute m_j > 0 only).
    """

    species: str
    mj: float
    n_min: int
    n_max: int
    l_max: int | None = None

    def __post_init__(self) -> None:
        if self.n_min < 1 or self.n_max < self.n_min:
            raise ValueError(f"bad n window [{self.n_min}, {self.n_max}]")
        if self.species == "H":
            if abs(self.mj - round(self.mj)) > 1e-9:
                raise ValueError("hydrogen backend is spinless: mj must be "
                                 f"an integer m, got {self.mj}")
        else:
            _resolve_species(self.species)
            if self.mj <= 0 or abs(2 * self.mj - round(2 * self.mj)) > 1e-9:
                raise ValueError(f"alkali m_j must be a positive half-integer "
                                 f"(+-m_j degenerate), got {self.mj}")


def basis_states(basis: StarkBasis) -> list[tuple]:
    """Basis labels: alkali (n, l, j) with j >= m_j; hydrogen (n, l) with
    l >= |m|. Fixed m_j; l <= min(l_max, n-1)."""
    out: list[tuple] = []
    for n in range(basis.n_min, basis.n_max + 1):
        l_top = n - 1 if basis.l_max is None else min(basis.l_max, n - 1)
        if basis.species == "H":
            out.extend((n, l) for l in range(abs(int(basis.mj)), l_top + 1))
            continue
        for l in range(0, l_top + 1):
            for j in (l - 0.5, l + 0.5):
                if j <= 0 or j < basis.mj - 1e-9:
                    continue
                out.append((n, l, j))
    return out


@dataclass(frozen=True)
class StarkMatrices:
    """H(E_field) = diag(e0_J) + E_field * d_Cm (spec 07 §2.6). Built once;
    swept by scaling. labels index both matrices; max_pair_spread_rel is
    the worst dual-method radial spread in d_Cm (0 for hydrogen: exact)."""

    basis: StarkBasis
    labels: tuple[tuple, ...]
    e0_J: np.ndarray
    d_Cm: np.ndarray
    max_pair_spread_rel: float
    n_top: int


def build_stark_matrices(basis: StarkBasis, *,
                         n_top: int | None = None) -> StarkMatrices:
    """Build (E0 diagonal [J], D [C m]) for the basis (spec 07 §2.6).

    D_kv = e <k|z|v> = e a0 R_kv A_kv, real symmetric, block-tridiagonal in
    l (Delta l = +-1, Delta j in {0, +-1}, m_j conserved). Alkali: spec 01
    energies (SAME model as the radial integrals — mixing energy sources
    is forbidden, spec 07 §4.2) + dual-method radial elements on the shared
    n_top grid. Hydrogen: exact Gordon elements, E_n = -E_h/(2 n^2).
    Pass n_top >= n_max to share the radial grid with a widened
    convergence rerun. Clears the wavefunction cache when done (memory).
    """
    labels = tuple(basis_states(basis))
    n_states = len(labels)
    if basis.species == "H":
        e0 = np.array([-0.5 * E_H / s[0] ** 2 for s in labels])
        d = np.zeros((n_states, n_states))
        m = int(basis.mj)
        for i, (n1, l1) in enumerate(labels):
            for k in range(i + 1, n_states):
                n2, l2 = labels[k]
                if abs(l1 - l2) != 1:
                    continue
                d[i, k] = d[k, i] = (AU_DIPOLE
                                     * radial_me_gordon(n1, l1, n2, l2)
                                     * _hydrogen_angular(l1, l2, m))
        return StarkMatrices(basis=basis, labels=labels, e0_J=e0, d_Cm=d,
                             max_pair_spread_rel=0.0, n_top=basis.n_max)
    sp = _resolve_species(basis.species)
    top = basis.n_max if n_top is None else max(n_top, basis.n_max)
    e0 = np.array([-H * binding_energy_hz(sp, *s) for s in labels])
    d = np.zeros((n_states, n_states))
    worst = 0.0
    mj = basis.mj
    for i, s1 in enumerate(labels):
        for k in range(i + 1, n_states):
            s2 = labels[k]
            if abs(s1[1] - s2[1]) != 1 or abs(s1[2] - s2[2]) > 1.0 + 1e-9:
                continue
            ang = angular_factor(s1[1], s1[2], mj, s2[1], s2[2], mj, 0)
            if ang == 0.0:
                continue
            r_a0, spread = _pair_me_a0(sp, s1, s2, top)
            worst = max(worst, spread)
            d[i, k] = d[k, i] = E_CHARGE * A0 * r_a0 * ang
    clear_radial_cache()
    return StarkMatrices(basis=basis, labels=labels, e0_J=e0, d_Cm=d,
                         max_pair_spread_rel=worst, n_top=top)


@dataclass
class StarkMapResult:
    """Adiabatically tracked Stark map (spec 07 §2.6).

    energies_J[f, i]: energy of tracked state i at fields_Vm[f] [J];
    column i connects continuously to the state dominating it at the first
    field. character_idx/character_w: dominant zero-field basis label index
    and weight per (field, state) — states are labelled by character, not
    by eigenvalue order (spec 07 §2.6). tracking_ok: False if any overlap
    assignment stayed below overlap_min after adaptive bisection (ambiguous
    only at near-exact crossings — hydrogen). flags: validity flags
    (audit §4 item 8), e.g. 'above_F_ion' (audit §3 item 30 hand-off).

    Convergence record (audit §4 item 6 — the demonstration ships as DATA,
    never as a bare boolean), populated when conv_check_state is given:
      conv_move_hz  |E_wide - E_base|/h at max(fields) [Hz] — what widening
                    the basis actually moved the tracked eigenvalue;
      conv_shift_hz |E_base(F_max) - E_base(0)|/h [Hz] — the FIELD-INDUCED
                    shift the movement is judged against (the physically
                    relevant quantity; the total eigenvalue is ~1e15 Hz of
                    binding energy that no basis change touches);
      conv_tol_hz   the tolerance actually applied [Hz].
    """

    mats: StarkMatrices
    fields_Vm: np.ndarray
    energies_J: np.ndarray
    character_idx: np.ndarray
    character_w: np.ndarray
    tracking_ok: bool
    converged: bool
    f_ion_Vm: float
    flags: list[str] = field(default_factory=list)
    conv_move_hz: float = math.nan
    conv_shift_hz: float = math.nan
    conv_tol_hz: float = math.nan


def _eig_at(mats: StarkMatrices, field_Vm: float) -> tuple[np.ndarray,
                                                           np.ndarray]:
    return eigh(np.diag(mats.e0_J) + field_Vm * mats.d_Cm)


def _assign(v_prev: np.ndarray, v_new: np.ndarray) -> tuple[np.ndarray,
                                                            float]:
    """Adiabatic column assignment prev->new by |overlap|; returns
    (permutation, worst matched overlap)."""
    ov = np.abs(v_prev.T @ v_new)
    row, col = linear_sum_assignment(-ov)
    perm = np.empty_like(col)
    perm[row] = col
    return perm, float(ov[row, col].min())


def stark_map(basis: StarkBasis | StarkMatrices,
              fields_Vm: np.ndarray | Sequence[float], *,
              overlap_min: float = 0.9, max_bisect: int = 10,
              conv_check_state: tuple | None = None,
              conv_tol_hz: float = 1e5,
              conv_rel_tol: float = 1e-3,
              conv_floor_hz: float = 1.0) -> StarkMapResult:
    """Diagonalize H(E) = diag(E0) + E_field D over a field sweep with
    adiabatic state tracking (spec 07 §2.6). fields_Vm ascending [V/m].

    Tracking: eigenvector-overlap assignment between consecutive fields
    (scipy linear_sum_assignment); when the worst matched overlap drops
    below overlap_min the field step is bisected adaptively (up to
    max_bisect insertions per step); if ambiguity persists (near-exact
    crossings — hydrogen) tracking_ok = False and states must be read by
    character weights, not adiabatic index.

    conv_check_state: zero-field label (n, l, j) — reruns the map at
    max(fields) with the window widened by 2 in n and 2 in l (spec 07
    §2.6) and RAISES IntegrityError if that state's eigenvalue moves too
    far (spec 07 §4.2: FAIL, do not average).

    Convergence tolerance (audit §4 item 6). The movement is compared
    against the FIELD-INDUCED shift |E(F_max) - E(0)|, not against the
    total eigenvalue: the total is dominated by ~1e15 Hz of binding energy
    that basis truncation barely touches, so a fixed absolute tolerance
    certifies a statement whose strength varies by orders of magnitude
    across n (the shift at 0.03 F_IT falls as n^-3, so 100 kHz is 15 % of
    the whole shift for Rb 50S and > 50 % by n = 80). The applied
    tolerance is
        tol = min(conv_tol_hz, max(conv_rel_tol * |shift|, conv_floor_hz))
    i.e. the spec 07 §2.6 absolute ceiling AND a 0.1 % relative-to-shift
    requirement, with conv_floor_hz as the numerical-noise floor so the
    gate can never demand sub-Hz agreement. All three numbers plus the
    measured movement ship on the result (conv_move_hz / conv_shift_hz /
    conv_tol_hz).

    Fields above F_ion(n_max) = F0/(16 n^4) are flagged 'above_F_ion', not
    refused — the Hermitian map tracks resonance positions qualitatively
    there (spec 07 §7.3; audit §3 item 30).
    """
    mats = basis if isinstance(basis, StarkMatrices) else \
        build_stark_matrices(basis)
    bas = mats.basis
    fields = np.asarray(fields_Vm, dtype=float)
    if fields.ndim != 1 or fields.size < 1 or np.any(np.diff(fields) <= 0):
        raise ValueError("fields_Vm must be a strictly increasing 1-D array")
    f_ion = classical_ionization_field_Vm(bas.n_max)
    flags = []
    if fields.max() > f_ion:
        flags.append(f"above_F_ion: max field {fields.max():.4g} V/m exceeds "
                     f"F_ion(n={bas.n_max}) = {f_ion:.4g} V/m — bound-state "
                     "map tracks resonance positions only (spec 07 §7.3)")
    n_states = len(mats.labels)
    energies = np.empty((fields.size, n_states))
    chi = np.empty((fields.size, n_states), dtype=int)
    chw = np.empty((fields.size, n_states))
    tracking_ok = True
    w, v = _eig_at(mats, fields[0])
    energies[0] = w
    chi[0] = np.argmax(np.abs(v), axis=0)
    chw[0] = np.max(np.abs(v), axis=0) ** 2
    f_prev, v_prev = fields[0], v
    for fi in range(1, fields.size):
        f_lo, v_lo = f_prev, v_prev
        stack = [fields[fi]]
        bisections = 0
        while stack:
            f_try = stack[-1]
            w, v = _eig_at(mats, f_try)
            perm, worst = _assign(v_lo, v)
            if worst < overlap_min and bisections < max_bisect:
                stack.append(0.5 * (f_lo + f_try))
                bisections += 1
                continue
            if worst < overlap_min:
                tracking_ok = False
            v_lo = v[:, perm]
            f_lo = f_try
            stack.pop()
            if not stack:
                energies[fi] = w[perm]
                chi[fi] = np.argmax(np.abs(v_lo), axis=0)
                chw[fi] = np.max(np.abs(v_lo), axis=0) ** 2
        f_prev, v_prev = f_lo, v_lo
    converged = False
    move_hz = shift_hz = tol_hz = math.nan
    if conv_check_state is not None:
        target = tuple(conv_check_state)
        if target not in mats.labels:
            raise ValueError(f"conv_check_state {target} not in basis")
        floor = 1 if bas.species == "H" else _species_floor(
            _resolve_species(bas.species))
        l_top = (None if bas.l_max is None
                 else bas.l_max + 2)
        wide = StarkBasis(bas.species, bas.mj,
                          max(floor, bas.n_min - 2), bas.n_max + 2, l_top)
        wmats = build_stark_matrices(wide, n_top=bas.n_max + 2)
        f_max = fields[-1]
        e_base = _tracked_state_energy(mats, target, f_max)
        e_wide = _tracked_state_energy(wmats, target, f_max)
        e_zero = _tracked_state_energy(mats, target, 0.0)
        move_hz = abs(e_wide - e_base) / H
        shift_hz = abs(e_base - e_zero) / H
        tol_hz = min(conv_tol_hz, max(conv_rel_tol * shift_hz, conv_floor_hz))
        if move_hz > tol_hz:
            raise IntegrityError(
                f"basis-window convergence FAILED: widening the window moved "
                f"the {target} eigenvalue at {f_max:.4g} V/m by "
                f"{move_hz:.3e} Hz (> {tol_hz:.3e} Hz = min(abs ceiling "
                f"{conv_tol_hz:.0e} Hz, max({conv_rel_tol:.0e} x field-"
                f"induced shift {shift_hz:.3e} Hz, floor "
                f"{conv_floor_hz:.3g} Hz))) — do not average "
                "(spec 07 §2.6/§4.2; audit §4 item 6)")
        converged = True
    return StarkMapResult(mats=mats, fields_Vm=fields, energies_J=energies,
                          character_idx=chi, character_w=chw,
                          tracking_ok=tracking_ok, converged=converged,
                          f_ion_Vm=f_ion, flags=flags,
                          conv_move_hz=move_hz, conv_shift_hz=shift_hz,
                          conv_tol_hz=tol_hz)


def _tracked_state_energy(mats: StarkMatrices, target: tuple,
                          field_Vm: float) -> float:
    """Eigenvalue at field_Vm of the state with max overlap onto the
    zero-field basis vector `target` (weak-field identification)."""
    idx = mats.labels.index(target)
    w, v = _eig_at(mats, field_Vm)
    return float(w[np.argmax(np.abs(v[idx, :]))])


@dataclass(frozen=True)
class MapCurvature:
    """Quadratic curvature extraction from a Stark map (spec 07 §2.6).

    alpha_SI [C^2 m^2/J] = -2 c2 from the even-power fit E(F) = E0 + c2
    E_field^2 + c4 E_field^4; gamma_SI is the F^4 coefficient recast as a
    hyperpolarizability estimate — DIAGNOSTIC ONLY, never a finding
    (spec 07 §7.7 / audit §3 item 29). quartic_fraction = |c4| F^4 /
    (|c2| F^2) at the fit edge (must be < 1 %, spec 07 §4.2).

    conv_move_hz / conv_shift_hz / conv_tol_hz: the StarkMapResult
    convergence record forwarded verbatim (audit §4 item 6) — the measured
    basis-widening movement, the field-induced shift it was judged
    against, and the tolerance applied. ``converged`` alone is a boolean;
    these three are the evidence behind it.
    """

    alpha_SI: float
    alpha_au: float
    alpha_Hz_per_Vcm2: float
    gamma_SI_diagnostic: float
    fit_max_field_Vm: float
    quartic_fraction: float
    e_mix_Vm: float
    converged: bool
    conv_move_hz: float = math.nan
    conv_shift_hz: float = math.nan
    conv_tol_hz: float = math.nan


def stark_map_curvature(res: StarkMapResult, state: tuple, *,
                        fit_max_field_Vm: float | None = None,
                        ) -> tuple[float, float]:
    """(alpha_SI, gamma_SI_diagnostic) from an even-power fit of a tracked
    state's map energies (spec 07 §2.6). The tracked column is the one
    whose first-field character is `state`. Fits E0 + c2 F^2 + c4 F^4 over
    fields <= fit_max_field_Vm (needs >= 5 points); shrinks the window
    until the quartic term is < 1 % of the quadratic at the edge, raising
    IntegrityError if that leaves fewer than 5 points. gamma is diagnostic
    only (audit §3 item 29).
    """
    try:
        idx = res.mats.labels.index(tuple(state))
    except ValueError:
        raise ValueError(f"state {state} not in the map basis") from None
    cols = np.where(res.character_idx[0] == idx)[0]
    if not cols.size:
        raise IntegrityError(
            f"no tracked column is dominated by {state} at the first field: "
            "start the sweep at a weaker field (spec 07 §2.6 tracking)")
    col = int(cols[0])
    fmax = res.fields_Vm[-1] if fit_max_field_Vm is None else fit_max_field_Vm
    sel = res.fields_Vm <= fmax * (1 + 1e-12)
    fields = res.fields_Vm[sel]
    energy = res.energies_J[sel, col]
    while True:
        if fields.size < 5:
            raise IntegrityError(
                "curvature fit window shrank below 5 points before the "
                "quartic term dropped under 1 % of the quadratic — field "
                "range too large for a quadratic extraction (spec 07 §4.2)")
        # fit on x = E/E_max in [0, 1] — the raw SI design matrix
        # (1, E^2, E^4) is catastrophically ill-conditioned
        f_top = fields[-1]
        x = fields / f_top
        mat = np.column_stack([np.ones_like(x), x**2, x**4])
        b0, b2, b4 = np.linalg.lstsq(mat, energy, rcond=None)[0]
        quart = abs(b4) / max(abs(b2), 1e-300)     # |c4 F^4/c2 F^2| at edge
        if quart < 0.01:
            return float(-2.0 * b2 / f_top**2), float(-24.0 * b4 / f_top**4)
        fields = fields[:-1]
        energy = energy[:-1]


def alpha_map_curvature(species: Species | str, n: int, l: int, j: float,
                        mj: float, *, n_span: int = 4, l_span: int = 4,
                        n_points: int = 8, conv_check: bool = True,
                        conv_tol_hz: float = 1e5,
                        conv_rel_tol: float = 1e-3) -> MapCurvature:
    """Polarizability of |n l j m_j> from Stark-map curvature — the
    perturbative cross-check route (spec 07 §2.6; benchmark RS-07-09).

    Basis: n +- n_span (never < 4, spec 07 §2.6), l <= l + l_span
    (weak-field curvature rule; conv_check widens by 2 in n and l and
    fails loudly if the target moves beyond the ``stark_map`` tolerance —
    min(conv_tol_hz, max(conv_rel_tol x field-induced shift, 1 Hz)); the
    measured movement, the shift and the tolerance are forwarded onto the
    result). Fit fields: n_points points up to E_fit = min(0.03 F_IT(n),
    0.1 E_mix) with E_mix read from the built matrices (spec 07 §2.6
    fit-window rule). NOTE that E_fit is deliberately small, so the
    field-induced shift is small too — which is exactly why the
    convergence gate must be relative to it and not to the eigenvalue.
    """
    sp = _resolve_species(species)
    if n_span < 4:
        raise ValueError("n_span >= 4 (spec 07 §2.6: never less than n0 +- 4)")
    floor = _species_floor(sp)
    basis = StarkBasis(sp.name, abs(mj), max(floor, n - n_span), n + n_span,
                       min(l + l_span, n - 1))
    mats = build_stark_matrices(basis, n_top=n + n_span + 2)
    target = (n, l, j)
    idx = mats.labels.index(target)
    coupled = np.nonzero(mats.d_Cm[idx])[0]
    e_mix = float(np.min(np.abs(mats.e0_J[coupled] - mats.e0_J[idx])
                         / (2.0 * np.abs(mats.d_Cm[coupled, idx]))))
    fit_max = min(0.03 * inglis_teller_field_Vm(n), 0.1 * e_mix)
    fields = np.linspace(fit_max / n_points, fit_max, n_points)
    res = stark_map(mats, fields,
                    conv_check_state=target if conv_check else None,
                    conv_tol_hz=conv_tol_hz, conv_rel_tol=conv_rel_tol)
    alpha_si, gamma_si = stark_map_curvature(res, target)
    # |c4 F^4| / |c2 F^2| at the fit edge, from the fitted coefficients
    quart = (abs(gamma_si) / 24.0) * fit_max**2 / max(abs(alpha_si) / 2.0,
                                                      1e-300)
    alpha_au = alpha_si / AU_POLARIZABILITY
    return MapCurvature(
        alpha_SI=alpha_si, alpha_au=alpha_au,
        alpha_Hz_per_Vcm2=polarizability_au_to_hz_per_v2_cm2(alpha_au),
        gamma_SI_diagnostic=gamma_si, fit_max_field_Vm=fit_max,
        quartic_fraction=quart, e_mix_Vm=e_mix, converged=res.converged,
        conv_move_hz=res.conv_move_hz, conv_shift_hz=res.conv_shift_hz,
        conv_tol_hz=res.conv_tol_hz)


@dataclass(frozen=True)
class ManifoldGap:
    """Fan-edge separation of two adjacent manifolds (spec 07 §2.6 core-
    effect regression, benchmark RS-07-08b) WITH its convergence record.

    Two distinct quantities are reported, because they answer different
    questions and only one of them is step-independent:

    gap_J — the ADIABATIC separation E_{k2}(F) - E_{k1}(F) at the FIXED
      ascending-eigenvalue indices k1 < k2 that the two fan edges occupy at
      fields_Vm[0]. For a real-symmetric one-parameter family the sorted
      eigenvalues are continuous and (generically) never cross, so this IS
      the step -> 0 limit of any correct adiabatic tracking: it needs no
      tracking parameters, is >= 0 by construction, and is the quantity the
      convergence gates are applied to. Its minimum over the sweep is the
      core-induced avoided-crossing gap (finite for an alkali, ~0 for
      hydrogen, where the parabolic separability makes the crossing exact).

    gap_tracked_J — the SIGNED separation from eigenvector-overlap tracking
      of the two edge states (joint 2-state assignment + adaptive
      bisection). It follows character rather than eigenvalue order and so
      changes sign at a genuine crossing; ``crosses`` is read from it.
      Tracking is diabatic through anticrossings narrower than the field
      step, which is exactly why this quantity alone must never carry the
      physics conclusion without the step-halving check.

    min_gap_J / min_field_Vm: refined (bounded parabolic search, not the
      grid argmin) minimum of the ADIABATIC gap and the field where it
      occurs — for ``crosses`` it is refined inside the first sign-change
      bracket, i.e. the crossing field itself.
    levels_between: number of eigenstates lying between the two edges at
      fields_Vm[0]. Alkali low-l intruders (Rb 33P, 32D between the n = 30
      and n = 31 manifolds) make this > 0; if a basis is too narrow to hold
      them the gap is wrong, so this integer is a convergence observable in
      its own right and must not change when the basis is widened.
    worst_overlap / tracking_ok / bisections: tracking quality.
    converged / convergence: the halved-step and widened-basis re-runs and
      the measured movements (audit §4 item 6 — evidence, not a boolean).
    """

    mats: StarkMatrices
    n_lower: int
    n_upper: int
    fields_Vm: np.ndarray
    gap_J: np.ndarray
    gap_tracked_J: np.ndarray
    min_gap_J: float
    min_field_Vm: float
    crosses: bool
    levels_between: int
    character_top: float
    character_bot: float
    worst_overlap: float
    tracking_ok: bool
    bisections: int
    manifold_spacing_J: float
    converged: bool
    convergence: dict
    flags: list[str] = field(default_factory=list)


def _fan_edge_indices(mats: StarkMatrices, n_lower: int, n_upper: int,
                      field_Vm: float, l_min_fan: int,
                      character_min: float) -> tuple[int, int, float, float]:
    """(k_top, k_bot, w_top, w_bot): ascending-eigenvalue indices of the
    top of the n_lower fan and the bottom of the n_upper fan at field_Vm.

    Fan membership = summed eigenvector weight > character_min on basis
    states carrying the manifold's n with l >= l_min_fan (low-l alkali
    states carry the manifold's n label but sit manifolds lower in energy —
    quantum defects; hence the mask). Refuses when the identification is
    ambiguous or when the two fans have already crossed at this field.
    """
    lab = mats.labels
    is_h = len(lab[0]) == 2
    l_cut = 0 if is_h else l_min_fan
    mask_lo = np.array([s[0] == n_lower and s[1] >= l_cut for s in lab])
    mask_hi = np.array([s[0] == n_upper and s[1] >= l_cut for s in lab])
    if not (mask_lo.any() and mask_hi.any()):
        raise IntegrityError(
            f"basis {mats.basis.n_min}..{mats.basis.n_max} does not contain "
            f"both n = {n_lower} and n = {n_upper} fan states (l >= {l_cut})")
    w, v = _eig_at(mats, field_Vm)
    w_lo = (v[mask_lo, :] ** 2).sum(axis=0)
    w_hi = (v[mask_hi, :] ** 2).sum(axis=0)
    i_lo = np.where(w_lo > character_min)[0]
    i_hi = np.where(w_hi > character_min)[0]
    if not (i_lo.size and i_hi.size):
        raise IntegrityError(
            f"fan identification failed at {field_Vm:.4g} V/m: no eigenstate "
            f"carries > {character_min} weight on the n = {n_lower} / "
            f"{n_upper} fans — start the sweep where the two manifold fans "
            "are resolved and not yet overlapping (spec 07 §2.6)")
    k_top = int(i_lo[np.argmax(w[i_lo])])
    k_bot = int(i_hi[np.argmin(w[i_hi])])
    if k_bot <= k_top:
        raise IntegrityError(
            f"the n = {n_lower} and n = {n_upper} fans already overlap at "
            f"the first field {field_Vm:.4g} V/m (edge eigen-indices "
            f"{k_top} >= {k_bot}): the sweep must START below the crossing "
            "or the 'first anticrossing' is not bracketed (spec 07 §2.6)")
    return k_top, k_bot, float(w_lo[k_top]), float(w_hi[k_bot])


def _adiabatic_gap_J(mats: StarkMatrices, k_lo: int, k_hi: int,
                     field_Vm: float) -> float:
    """E_{k_hi}(F) - E_{k_lo}(F) [J] at fixed ascending-eigenvalue index."""
    w = np.linalg.eigvalsh(np.diag(mats.e0_J) + field_Vm * mats.d_Cm)
    return float(w[k_hi] - w[k_lo])


def _track_fan_pair(mats: StarkMatrices, k_top: int, k_bot: int,
                    fields: np.ndarray, overlap_min: float,
                    max_bisect: int) -> tuple[np.ndarray, float, bool, int]:
    """Overlap-track the two fan-edge states; signed gap per field.

    The two states are matched JOINTLY (``linear_sum_assignment`` on the
    2 x N overlap matrix), so they can never collapse onto the same
    eigenvector — the failure mode of two independent argmax lookups. When
    the worse of the two matched overlaps drops below overlap_min the field
    step is bisected (up to max_bisect insertions per step), mirroring
    ``stark_map``; if the ambiguity survives, tracking_ok goes False.
    """
    w, v = _eig_at(mats, fields[0])
    v_top, v_bot = v[:, k_top], v[:, k_bot]
    out = [w[k_bot] - w[k_top]]
    worst_all = 1.0
    tracking_ok = True
    bisections = 0
    f_lo = float(fields[0])
    for fi in range(1, fields.size):
        stack = [float(fields[fi])]
        used = 0
        while stack:
            f_try = stack[-1]
            w, v = _eig_at(mats, f_try)
            ov = np.abs(np.vstack([v_top, v_bot]) @ v)
            _, col = linear_sum_assignment(-ov)
            a, b = int(col[0]), int(col[1])
            worst = min(ov[0, a], ov[1, b])
            if worst < overlap_min and used < max_bisect:
                stack.append(0.5 * (f_lo + f_try))
                used += 1
                bisections += 1
                continue
            if worst < overlap_min:
                tracking_ok = False
            worst_all = min(worst_all, float(worst))
            v_top, v_bot = v[:, a], v[:, b]
            f_lo = f_try
            stack.pop()
            if not stack:
                out.append(w[b] - w[a])
    return np.array(out), worst_all, tracking_ok, bisections


def _locate_gap_min(mats: StarkMatrices, k_lo: int, k_hi: int,
                    fields: np.ndarray, gap_ad: np.ndarray,
                    tracked: np.ndarray) -> tuple[float, float]:
    """Refined (min adiabatic gap [J], field [V/m]).

    When the tracked gap changes sign the branches genuinely cross, and the
    minimum is refined inside that first sign-change bracket (for hydrogen
    the adiabatic gap has many near-degenerate dips, so a global argmin
    would be numerical noise). Otherwise the global grid argmin is used and
    must be INTERIOR — an endpoint minimum means the closest approach is
    not bracketed by the sweep, which is a refusal, not a result.
    """
    neg = np.where(tracked < 0.0)[0]
    if neg.size:
        i = int(neg[0])
        lo, hi = float(fields[i - 1]), float(fields[i])
    else:
        i = int(np.argmin(gap_ad))
        if i == 0 or i == fields.size - 1:
            raise IntegrityError(
                f"the fan-edge separation is minimal at the sweep endpoint "
                f"{fields[i]:.4g} V/m: the closest approach is not bracketed "
                "— widen fields_Vm (spec 07 §2.6)")
        lo, hi = float(fields[i - 1]), float(fields[i + 1])
    res = minimize_scalar(lambda f: _adiabatic_gap_J(mats, k_lo, k_hi, f),
                          bounds=(lo, hi), method="bounded",
                          options={"xatol": (hi - lo) * 1e-6})
    if float(res.fun) < gap_ad[i]:
        return float(res.fun), float(res.x)
    return float(gap_ad[i]), float(fields[i])


def _gap_run(mats: StarkMatrices, n_lower: int, n_upper: int,
             fields: np.ndarray, l_min_fan: int, character_min: float,
             overlap_min: float, max_bisect: int) -> dict:
    """One complete fan-edge evaluation on a given basis and field grid."""
    k_top, k_bot, w_top, w_bot = _fan_edge_indices(
        mats, n_lower, n_upper, float(fields[0]), l_min_fan, character_min)
    gap_ad = np.array([_adiabatic_gap_J(mats, k_top, k_bot, f)
                       for f in fields])
    tracked, worst, ok, nbis = _track_fan_pair(
        mats, k_top, k_bot, fields, overlap_min, max_bisect)
    min_gap, min_field = _locate_gap_min(mats, k_top, k_bot, fields,
                                         gap_ad, tracked)
    return {"k_top": k_top, "k_bot": k_bot, "w_top": w_top, "w_bot": w_bot,
            "gap_J": gap_ad, "tracked_J": tracked, "min_gap_J": min_gap,
            "min_field_Vm": min_field, "crosses": bool(np.any(tracked < 0.0)),
            "levels_between": k_bot - k_top - 1, "worst_overlap": worst,
            "tracking_ok": ok, "bisections": nbis}


def manifold_gap(mats: StarkMatrices, n_lower: int, n_upper: int,
                 fields_Vm: np.ndarray | Sequence[float], *,
                 l_min_fan: int = 4, character_min: float = 0.5,
                 overlap_min: float = 0.9, max_bisect: int = 10,
                 conv_check: bool = True, conv_n_widen: int = 1,
                 conv_rel_gap: float = 0.05, conv_rel_field: float = 0.02,
                 conv_abs_frac: float = 1e-3) -> ManifoldGap:
    """Separation of the top of the n_lower fan and the bottom of the
    n_upper fan over a field sweep (spec 07 §2.6; benchmark RS-07-08b).

    This is the adjudicator of the core-effect regression — "alkali fan
    edges ANTICROSS, hydrogen fan edges genuinely CROSS" — so it carries
    the gates that claim demands. See ``ManifoldGap`` for the two reported
    gap quantities and why only the adiabatic one is step-independent.

    Refuses (IntegrityError) rather than reporting a conclusion it cannot
    support:
      * fan identification ambiguous, or the fans already overlap at
        fields_Vm[0] (the crossing must be bracketed by the sweep);
      * the minimum sits at a sweep endpoint;
      * eigenvector tracking stayed ambiguous after ``max_bisect``
        bisections (``tracking_ok`` False) — the ``crosses`` verdict would
        then be a step-size artifact;
      * conv_check=True and either re-run disagrees (below).

    conv_check defaults to True (convergence is DEMONSTRATED, never
    assumed — spec 07 §4.1/§2.6); pass False only for exploratory calls,
    where ``converged`` then ships False. It runs the two demonstrations
    spec 07 §2.6 requires and the audit §4 item 6 record:
      1. HALVED FIELD STEP on the same basis — catches exactly the failure
         this gate exists for: with a coarse step the tracker hops
         diabatically across the anticrossing and reports the hydrogen-like
         'edges cross' conclusion.
      2. WIDENED BASIS, n_min - conv_n_widen .. n_max + conv_n_widen (same
         l_max) — catches a window too narrow to hold the low-l intruders
         that sit energetically between the two fans.
    Both must reproduce ``crosses`` and ``levels_between`` exactly, and the
    refined minimum within
        |d gap| <= max(conv_rel_gap * gap, conv_abs_frac * manifold spacing)
        |d field| <= conv_rel_field * field
    The absolute branch exists because a genuine crossing has gap ~ 0, on
    which a relative tolerance is meaningless. The manifold spacing is the
    zero-field |E(n_upper) - E(n_lower)| of the two fan centres.
    """
    fields = np.asarray(fields_Vm, dtype=float)
    if fields.ndim != 1 or fields.size < 3 or np.any(np.diff(fields) <= 0):
        raise ValueError("fields_Vm must be a strictly increasing 1-D array "
                         "with at least 3 points")
    bas = mats.basis
    base = _gap_run(mats, n_lower, n_upper, fields, l_min_fan, character_min,
                    overlap_min, max_bisect)
    if not base["tracking_ok"]:
        raise IntegrityError(
            f"fan-edge tracking is ambiguous (worst matched overlap "
            f"{base['worst_overlap']:.3f} < {overlap_min} after "
            f"{max_bisect} bisections): the cross/anticross verdict would be "
            "a field-step artifact — refine fields_Vm or raise max_bisect "
            "(spec 07 §2.6)")

    # Manifold spacing = |E0(n_upper fan) - E0(n_lower fan)| at zero field,
    # taken from the built matrices themselves (the highest-l state of each
    # manifold, which carries a negligible quantum defect) — the natural
    # scale for the absolute branch of the gap tolerance.
    is_h = len(mats.labels[0]) == 2
    l_cut = 0 if is_h else l_min_fan
    e_lo = [e for s, e in zip(mats.labels, mats.e0_J)
            if s[0] == n_lower and s[1] >= l_cut]
    e_hi = [e for s, e in zip(mats.labels, mats.e0_J)
            if s[0] == n_upper and s[1] >= l_cut]
    spacing = abs(max(e_hi) - max(e_lo))

    flags: list[str] = []
    f_ion = classical_ionization_field_Vm(bas.n_max)
    if fields.max() > f_ion:
        flags.append(f"above_F_ion: max field {fields.max():.4g} V/m exceeds "
                     f"F_ion(n={bas.n_max}) = {f_ion:.4g} V/m — bound-state "
                     "map tracks resonance positions only (spec 07 §7.3)")

    converged = False
    record: dict = {"tol_gap_J": max(conv_rel_gap * base["min_gap_J"],
                                     conv_abs_frac * spacing),
                    "tol_field_rel": conv_rel_field,
                    "manifold_spacing_hz": spacing / H}
    if conv_check:
        tol_gap = record["tol_gap_J"]
        half = np.linspace(fields[0], fields[-1], 2 * fields.size - 1)
        runs = {"step_halved": _gap_run(mats, n_lower, n_upper, half,
                                        l_min_fan, character_min,
                                        overlap_min, max_bisect)}
        floor = 1 if bas.species == "H" else _species_floor(
            _resolve_species(bas.species))
        wide_basis = StarkBasis(bas.species, bas.mj,
                                max(floor, bas.n_min - conv_n_widen),
                                bas.n_max + conv_n_widen, bas.l_max)
        wmats = build_stark_matrices(wide_basis,
                                     n_top=bas.n_max + conv_n_widen)
        runs["basis_wide"] = _gap_run(wmats, n_lower, n_upper, fields,
                                      l_min_fan, character_min, overlap_min,
                                      max_bisect)
        for name, run in runs.items():
            d_gap = abs(run["min_gap_J"] - base["min_gap_J"])
            d_field = abs(run["min_field_Vm"] / base["min_field_Vm"] - 1.0)
            record[name] = {
                "min_gap_hz": run["min_gap_J"] / H,
                "min_field_Vm": run["min_field_Vm"],
                "crosses": run["crosses"],
                "levels_between": run["levels_between"],
                "tracking_ok": run["tracking_ok"],
                "d_gap_hz": d_gap / H, "d_field_rel": d_field}
            if not run["tracking_ok"]:
                raise IntegrityError(
                    f"manifold-gap convergence re-run '{name}' lost tracking "
                    f"(worst overlap {run['worst_overlap']:.3f}) — refusing")
            if run["crosses"] != base["crosses"]:
                raise IntegrityError(
                    f"manifold-gap convergence FAILED ({name}): the "
                    f"cross/anticross verdict INVERTED (base crosses="
                    f"{base['crosses']}, {name} crosses={run['crosses']}) — "
                    "the physics conclusion is a discretization artifact, "
                    "refusing (spec 07 §2.6; audit §4 item 6)")
            if run["levels_between"] != base["levels_between"]:
                raise IntegrityError(
                    f"manifold-gap convergence FAILED ({name}): the number "
                    f"of levels between the two fan edges changed "
                    f"{base['levels_between']} -> {run['levels_between']}; "
                    "the basis does not hold every state that lies between "
                    "the manifolds (low-l intruders) — refusing")
            if d_gap > tol_gap:
                raise IntegrityError(
                    f"manifold-gap convergence FAILED ({name}): the minimum "
                    f"gap moved {d_gap / H:.4g} Hz (> {tol_gap / H:.4g} Hz = "
                    f"max({conv_rel_gap:g} x gap, {conv_abs_frac:g} x "
                    f"manifold spacing {spacing / H:.4g} Hz)) — refusing "
                    "(spec 07 §2.6)")
            if d_field > conv_rel_field:
                raise IntegrityError(
                    f"manifold-gap convergence FAILED ({name}): the field of "
                    f"the minimum moved {d_field:.3%} "
                    f"({base['min_field_Vm']:.4g} -> "
                    f"{run['min_field_Vm']:.4g} V/m, allowed "
                    f"{conv_rel_field:.1%}) — refusing (spec 07 §2.6)")
        record["basis_wide"]["n_min"] = wide_basis.n_min
        record["basis_wide"]["n_max"] = wide_basis.n_max
        converged = True

    return ManifoldGap(
        mats=mats, n_lower=n_lower, n_upper=n_upper, fields_Vm=fields,
        gap_J=base["gap_J"], gap_tracked_J=base["tracked_J"],
        min_gap_J=base["min_gap_J"], min_field_Vm=base["min_field_Vm"],
        crosses=base["crosses"], levels_between=base["levels_between"],
        character_top=base["w_top"], character_bot=base["w_bot"],
        worst_overlap=base["worst_overlap"], tracking_ok=base["tracking_ok"],
        bisections=base["bisections"], manifold_spacing_J=spacing,
        converged=converged, convergence=record, flags=flags)


# ---------------------------------------------------------------------------
# Stark shifts and DC/LF sensing (spec 07 §2.1, §2.8)
# ---------------------------------------------------------------------------

def stark_shift_J(pol: Polarizability, field_Vm: float) -> float:
    """Quadratic DC Stark shift DeltaE = -(1/2) alpha E^2 [J] (lock #14).

    Refuses fields beyond the quadratic validity cap min(0.1 E_mix,
    0.03 F_IT(n)) — beyond it use the Stark map (spec 07 §2.6/§2.8 caps;
    audit §3 item 30).
    """
    cap = min(0.1 * pol.E_mix_Vm, 0.03 * inglis_teller_field_Vm(pol.n))
    if abs(field_Vm) > cap:
        raise IntegrityError(
            f"field {field_Vm:.4g} V/m exceeds the quadratic validity cap "
            f"{cap:.4g} V/m = min(0.1 E_mix, 0.03 F_IT) for "
            f"{pol.species} n={pol.n}: use the Stark map "
            "(spec 07 §2.6/§2.8; audit §3 item 30)")
    return -0.5 * pol.alpha_SI * field_Vm**2


def ac_stark_shift_J(alpha_SI: float, e_amp_Vm: float) -> float:
    """Time-averaged AC Stark shift DeltaE = -(1/4) alpha(omega) E_amp^2
    [J] for E(t) = E_amp cos(omega t) — E_amp is the PEAK amplitude, never
    RMS (locks #3/#14; the 1/4 vs the DC 1/2 is the cos^2 average).
    Validity (far-off-resonance) is the caller's: pair with alpha_dynamic.
    """
    return -0.25 * alpha_SI * e_amp_Vm**2


def stark_responsivity_hz_per_Vm(alpha_SI: float,
                                 e_bias_Vm: float) -> float:
    """Biased-quadratic responsivity R = |d nu/d E| = (|alpha|/h) E_b
    [Hz/(V/m)], spec 07 Eq. (7.12). Zero bias -> zero linear responsivity
    (the square-root pathology; a bias is mandatory for linear DC/LF
    sensing, spec 07 §2.8). |alpha| is used: readout linearizes the same
    for downward (alpha > 0) and upward (alpha < 0, Cs nD5/2) shifts.
    """
    return abs(alpha_SI) / H * abs(e_bias_Vm)


def unbiased_min_field_Vm(alpha_SI: float, dnu_min_hz: float) -> float:
    """Unbiased detectable field delta_E = sqrt(2 h delta_nu / |alpha|)
    [V/m] (spec 07 §2.8 zero-bias pathology; Rb 50S with delta_nu = 1 kHz
    -> 6.3 mV/cm)."""
    return math.sqrt(2.0 * H * dnu_min_hz / abs(alpha_SI))


def optimal_bias_Vm(alpha_SI: float, gamma0_hz: float, frac_inhomog: float,
                    *, e_cap_Vm: float) -> tuple[float, float]:
    """Optimal bias field, spec 07 Eq. (7.13) [V/m].

    E_b* = sqrt(h Gamma0 / (|alpha| eta)) with eta the fractional rms bias
    inhomogeneity; clipped at e_cap_Vm (the caller's quadratic-validity
    cap, spec 07 §2.8: E_b <= min(0.1 E_mix, ~0.03 F_IT)). eta -> 0 pushes
    to the cap. Returns (E_b*, NEF scale factor 2 sqrt(eta Gamma0 h /
    |alpha|) [V/m per unit sqrt-resolution] — proportionality of (7.13),
    diagnostic).
    """
    if gamma0_hz <= 0 or e_cap_Vm <= 0:
        raise ValueError("gamma0_hz and e_cap_Vm must be positive")
    if frac_inhomog < 0:
        raise ValueError("frac_inhomog must be >= 0")
    if frac_inhomog == 0.0:
        return e_cap_Vm, 0.0
    e_star = math.sqrt(H * gamma0_hz / (abs(alpha_SI) * frac_inhomog))
    nef_scale = 2.0 * math.sqrt(frac_inhomog * gamma0_hz * H
                                / abs(alpha_SI))
    return min(e_star, e_cap_Vm), nef_scale


def nef_dc_Vm_sqrtHz(alpha_SI: float, e_bias_Vm: float,
                     dnu_hz_sqrthz: float) -> float:
    """Field-referred DC/LF noise floor NEF = h S_nu^(1/2) / (|alpha| E_b)
    [(V/m)/sqrt(Hz)] (spec 07 Eq. 7.12 inverse; amplitude convention,
    lock #3). dnu_hz_sqrthz: spectroscopic frequency-resolution ASD from
    spec 08 [Hz/sqrt(Hz)]."""
    if e_bias_Vm <= 0:
        raise ValueError("e_bias_Vm must be > 0 (zero-bias readout is "
                         "square-root, not linear — spec 07 §2.8)")
    return H * dnu_hz_sqrthz / (abs(alpha_SI) * e_bias_Vm)


# ---------------------------------------------------------------------------
# Low-frequency screening interface (spec 07 §2.9; unified form ruling R-7)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScreeningModel:
    """Cell screening transfer-function parameters (interface form owned
    here; VALUES and calibration owned by spec 05 — ruling R-7).

    Unified normative form (spec 00 R-7):
      T(f) = S_geo [T_dc + (1 - T_dc) (i 2 pi f tau_s)^beta /
                                       (1 + (i 2 pi f tau_s)^beta)]
    Spec 07 Eq. (7.14) is the (S_geo = 1, beta = 1) special case; spec 05's
    high-pass is the (T_dc = 0) special case. calibrated=False marks
    default/estimated parameters: absolute external-field claims then
    refuse (audit §3 item 20 / R12). Anchor: sapphire cells screen with
    tau_s up to O(1 s) (Jau & Carter 2020 [VERIFIED abstract level]);
    borosilicate tau_s is MISSING — never guessed here.
    """

    T_dc: float
    tau_s_s: float
    S_geo: float = 1.0
    beta: float = 1.0
    calibrated: bool = False
    source: str = ""


def screening_transfer(f_hz: np.ndarray | float,
                       m: ScreeningModel) -> np.ndarray | complex:
    """Complex internal/external field transfer T(f) (R-7 unified form).

    f_hz >= 0 [Hz] (one-sided, lock #12). T(0) = S_geo T_dc;
    |T| -> S_geo for f >> 1/(2 pi tau_s).
    """
    f = np.asarray(f_hz, dtype=float)
    if np.any(f < 0):
        raise ValueError("f_hz must be >= 0 (one-sided convention, lock #12)")
    x = (1j * 2.0 * np.pi * f * m.tau_s_s) ** m.beta
    t = m.S_geo * (m.T_dc + (1.0 - m.T_dc) * x / (1.0 + x))
    return t.item() if t.ndim == 0 else t


def nef_external_Vm_sqrtHz(f_hz: np.ndarray, nef_internal_Vm_sqrtHz:
                           np.ndarray | float, m: ScreeningModel, *,
                           estimate_ok: bool = False) -> np.ndarray:
    """External-field NEF through the cell screening, spec 07 Eq. (7.15):
    NEF_ext(f) = NEF_int(f) / |T(f)| [(V/m)/sqrt(Hz)].

    Refuses (audit §3 item 20 / R12) when the screening parameters are not
    calibrated, unless estimate_ok=True — uncalibrated sub-kHz absolute
    field claims are meaningless and every output must carry the parameter
    tuple (the ScreeningModel travels with the result).
    """
    if not m.calibrated and not estimate_ok:
        raise IntegrityError(
            "screening parameters are not calibrated (tau_s default/"
            "estimate): refusing an absolute external-field NEF — pass "
            "estimate_ok=True to get a flagged estimate (spec 07 §2.9 / "
            "audit §3 item 20, R12)")
    t_abs = np.abs(screening_transfer(f_hz, m))
    with np.errstate(divide="ignore"):
        return np.asarray(nef_internal_Vm_sqrtHz, dtype=float) / t_abs
