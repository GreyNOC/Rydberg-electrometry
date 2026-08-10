"""Radiative + BBR lifetimes and the EIT dephasing budget (spec 04).

Owns (docs/spec/00-conventions.md SS4): lifetimes, blackbody depopulation and
photoionization, the term-by-term ladder-EIT dephasing budget, and the
collision / self-broadening coefficients. Species: Rb-85, Rb-87, Cs-133.

Conventions (docs/spec/00-conventions.md, normative):
* SI internally; every rate is angular [rad/s == 1/s for rates] (locks #1-2).
  Hz appears only in ``*_hz`` I/O fields with the single 2*pi conversion at
  this module's boundary.
* Gamma = population decay = 1/tau; gamma_ij = coherence decay (lock #7).
* Laser linewidth: gamma = pi * Dnu (FWHM Hz); the ladder two-photon rate is
  gamma_gr = pi (Dnu_p + Dnu_c + 2 c sqrt(Dnu_p Dnu_c)) (lock #8 / R-16).
* Transit: gamma_t = sqrt(2 ln 2) <v_perp>/w0 from ``rydsim.cell.transit_rate``
  (R-3) — never re-derived here; FWHM equivalent Dnu = gamma_t/pi.
* Densities in m^-3, beta in Hz m^3, C6 stored as C6/h in Hz m^6 (R-21, R-26).
* Stark inhomogeneous term takes alpha as a PARAMETER (R-5): the spec-04
  "~6e2 MHz/(V/cm)^2" figure for Rb 50S is WRONG (12x too large) and BANNED
  from code; the verified value is 50.5 MHz/(V/cm)^2 (spec 07, Yerokhin /
  O'Sullivan & Stoicheff). It must be supplied by the caller (rydsim.stark).
* Cs nD5/2 defects etc. come from spec 01 machinery (R-14) via
  ``rydsim.atom.n_star`` — no defect literals are duplicated here.

Method 1 (A-sum): Einstein-A sums over lower states, spec 04 Eq. 2.1, using
``rydsim.angular`` (reduced_dipole_j, einstein_A) and radial integrals from
``rydsim.radial`` solved with BOTH the MSD94 model potential (Method A) and
the pure-Coulomb QDT potential (Method B) on a shared grid; the A-vs-B spread
is carried as the numerical uncertainty of every rate. States below the
spec-01 Ritz hard floor (Rb n<8, Cs n<12) use measured NIST-ASD term
energies (table below, fetched from physics.nist.gov this session) — never
the Ritz expansion (conventions SS6 gap 3). The D-line radial element is
replaced by the experimentally-derived Steck value (spec 02 SS7.1: D-line
dipoles come from experiment; the model potential carries a documented
+5..10% low-n bias).

Deviation note (documented): for sub-floor partner states the A-vs-B radial
spread may exceed the spec-02 low-nu consensus ceiling (3e-2). The lifetime
engine does NOT refuse there (refusing would make every tau_0 uncomputable);
it records the per-method Gamma_0 spread and ships it as the uncertainty.
Spec 04 SS4.1 item 3 declares exactly this low-n' bottleneck and the sum
benchmarks (B2-B8b, 10%) fence the resulting error. Above-floor pairs still
enforce the spec-02 ceilings (a breach there is a regression -> raise).

Method 2 (Beterov fits): PRA 79, 052504 (2009) Table II lifetime fits and
Table I BBR fits (verbatim coefficients incl. the fitted constants 2.14e10
and 315780 K), Eq. 16 effective lifetime; NJP 11, 013052 (2009) Eq. 27
direct BBR photoionization. Farley-Wing high-T limit as cross-check only.

Integrity (docs/spec/00-integrity-audit.md):
* audit SS3 item 14: Beterov fit paths outside 15 <= n <= 80 raise
  ``rydsim.provenance.IntegrityError``.
* audit SS3 item 15: inhomogeneous budget terms (Doppler, vdW nn-distance,
  DC-Stark P(E), beam profile) are NEVER converted to Lindblad rates — they
  live in ``DephasingBudget.inhomo`` as distribution descriptors; in
  particular Dk*sigma_v is never added to gamma_gr (named forbidden error).
* audit SS3 item 16: buffer-gas cells are out of scope (no buffer-gas
  parameter exists; transit/collision models assume ballistic flight).
* audit R1: ``build_budget`` refuses a Stark term without a caller-supplied
  alpha (R-5); the banned 6e2 literal appears nowhere.
* Missing data rows (e.g. a sub-floor level not in the NIST table, a C6 or
  scattering length for an unsourced series) raise IntegrityError — never a
  guessed number.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .angular import einstein_A, reduced_dipole_j, steck_to_racah
from .atom import Species, binding_energy_hz, n_star, rydberg_constant_hz
from .cell import number_density_m3, transit_rate
from .constants import (A0, ALPHA_FS, AMU, AU_DIPOLE, E_H, H, HBAR, KB, M_E)
from .provenance import Confidence, CrossCheck, IntegrityError, SourcedValue
from .radial import radial_matrix_element, radial_wavefunction

__all__ = [
    "LOW_N_TERM_CM", "NIST_ION_LIMIT_CM", "binding_hz_any",
    "DecayChannel", "DecayTable", "decay_table", "clear_lifetime_cache",
    "einstein_rate", "radiative_lifetime_sum", "bbr_rate_sum",
    "effective_lifetime_sum",
    "BETEROV_TAU_FIT", "BETEROV_BBR_FIT", "beterov_series_key",
    "radiative_lifetime_fit", "bbr_rate_fit", "effective_lifetime_fit",
    "bbr_rate_farley_wing", "NJP_QD_DIFF", "NJP_A_L", "bbr_ionization_rate",
    "lifetime_crosscheck", "bbr_crosscheck",
    "self_broadening_beta_hz_m3", "SELF_BROADENING_MEASURED",
    "self_broadening_fwhm_hz", "transit_fwhm_hz",
    "AU_C6_HZ_M6", "c6_over_h_rb_ns", "vdw_dephasing_hz",
    "A_S_ELECTRON_RB5S_A0", "fermi_shift_hz",
    "laser_dephasing_rates", "stark_inhomogeneous_fwhm_hz",
    "CellParams", "LaserParams", "DephasingBudget", "build_budget",
    "default_cell",
]


# ---------------------------------------------------------------------------
# Low-n intermediate levels — NIST ASD data rows (conventions SS6 gap 3)
# ---------------------------------------------------------------------------

_NIST_SOURCE = ("NIST ASD (physics.nist.gov/cgi-bin/ASD/energy1.pl, Rb I / "
                "Cs I level tables, fetched 2026-08-10) [VERIFIED this "
                "session]")

#: Ionization limits [cm^-1] as printed by NIST ASD for the SAME level table
#: the term energies below come from (internal consistency; the Cs value is
#: digit-identical to Deiglmayr 2016, the Rb value matches spec 01 E_I to
#: < 0.01 cm^-1). Source: NIST ASD [VERIFIED this session].
NIST_ION_LIMIT_CM: dict[str, float] = {"Rb": 33690.81, "Cs": 31406.4677325}

#: Term energies [cm^-1 above the ground level] of the low-lying states that
#: sit BELOW the spec-01 Ritz hard floor (Rb n<8, Cs n<12) and are needed by
#: the Einstein-A sums. Keyed (element, n, l, 2j). Every row fetched from
#: NIST ASD this session [VERIFIED]; the first P rows cross-check against
#: the Steck D-line frequencies in tests (test_lifetimes.py, NIST-vs-Steck
#: self-check). Note the level orderings the data itself settles: Rb 4d is
#: INVERTED (J=5/2 below J=3/2) while Rb 5d-7d are normal; Rb/Cs nF have
#: J=5/2 above J=7/2 (consistent with the spec-01 defect ordering).
LOW_N_TERM_CM: dict[tuple[str, int, int, int], float] = {
    # Rb I (natural-mix NIST values; isotope shifts < 0.01 cm^-1, far below
    # lifetime accuracy)
    ("Rb", 5, 1, 1): 12578.950, ("Rb", 5, 1, 3): 12816.545,
    ("Rb", 6, 1, 1): 23715.081, ("Rb", 6, 1, 3): 23792.591,
    ("Rb", 7, 1, 1): 27835.02, ("Rb", 7, 1, 3): 27870.11,
    ("Rb", 6, 0, 1): 20132.510, ("Rb", 7, 0, 1): 26311.437,
    ("Rb", 4, 2, 3): 19355.649, ("Rb", 4, 2, 5): 19355.203,   # inverted
    ("Rb", 5, 2, 3): 25700.536, ("Rb", 5, 2, 5): 25703.498,
    ("Rb", 6, 2, 3): 28687.127, ("Rb", 6, 2, 5): 28689.390,
    ("Rb", 7, 2, 3): 30280.113, ("Rb", 7, 2, 5): 30281.620,
    ("Rb", 4, 3, 5): 26792.118, ("Rb", 4, 3, 7): 26792.092,
    ("Rb", 5, 3, 5): 29277.787, ("Rb", 5, 3, 7): 29277.768,
    ("Rb", 6, 3, 5): 30627.978, ("Rb", 6, 3, 7): 30627.962,
    ("Rb", 7, 3, 5): 31441.730, ("Rb", 7, 3, 7): 31441.718,
    # Cs I
    ("Cs", 6, 1, 1): 11178.26815870, ("Cs", 6, 1, 3): 11732.3071041,
    ("Cs", 7, 1, 1): 21765.348, ("Cs", 7, 1, 3): 21946.397,
    ("Cs", 8, 1, 1): 25708.83548, ("Cs", 8, 1, 3): 25791.508,
    ("Cs", 9, 1, 1): 27636.9966, ("Cs", 9, 1, 3): 27681.6782,
    ("Cs", 10, 1, 1): 28726.8123, ("Cs", 10, 1, 3): 28753.6769,
    ("Cs", 11, 1, 1): 29403.42310, ("Cs", 11, 1, 3): 29420.824,
    ("Cs", 7, 0, 1): 18535.5286, ("Cs", 8, 0, 1): 24317.149400,
    ("Cs", 9, 0, 1): 26910.6627, ("Cs", 10, 0, 1): 28300.2287,
    ("Cs", 11, 0, 1): 29131.73004,
    ("Cs", 5, 2, 3): 14499.2568, ("Cs", 5, 2, 5): 14596.84232,
    ("Cs", 6, 2, 3): 22588.8210, ("Cs", 6, 2, 5): 22631.6863,
    ("Cs", 7, 2, 3): 26047.8342, ("Cs", 7, 2, 5): 26068.7730,
    ("Cs", 8, 2, 3): 27811.2400, ("Cs", 8, 2, 5): 27822.8802,
    ("Cs", 9, 2, 3): 28828.6820, ("Cs", 9, 2, 5): 28835.79192,
    ("Cs", 10, 2, 3): 29468.2878, ("Cs", 10, 2, 5): 29472.93995,
    ("Cs", 11, 2, 3): 29896.3399, ("Cs", 11, 2, 5): 29899.54646,
    ("Cs", 4, 3, 5): 24472.2269, ("Cs", 4, 3, 7): 24472.0455,
    ("Cs", 5, 3, 5): 26971.3030, ("Cs", 5, 3, 7): 26971.1535,
    ("Cs", 6, 3, 5): 28329.5133, ("Cs", 6, 3, 7): 28329.4075,
    ("Cs", 7, 3, 5): 29147.98188, ("Cs", 7, 3, 7): 29147.90818,
    ("Cs", 8, 3, 5): 29678.74280, ("Cs", 8, 3, 7): 29678.68970,
    ("Cs", 9, 3, 5): 30042.31405, ("Cs", 9, 3, 7): 30042.27515,
    ("Cs", 10, 3, 5): 30302.16537, ("Cs", 10, 3, 7): 30302.13624,
    ("Cs", 11, 3, 5): 30494.28809, ("Cs", 11, 3, 7): 30494.26583,
}

# 1 cm^-1 in Hz (derived, lock #15)
from .constants import C as _C
_INV_CM_HZ = _C * 100.0

#: Lowest existing n' per (element, l') — ground-configuration minima
#: (spec 04 SS4.1 step 1: Rb n'>=5 for S/P, 4 for D/F; Cs 6/6/5/4).
#: Source: NIST ASD level tables (lowest listed level) [VERIFIED].
_N_MIN_SERIES: dict[str, dict[int, int]] = {
    "Rb": {0: 5, 1: 5, 2: 4, 3: 4},
    "Cs": {0: 6, 1: 6, 2: 5, 3: 4},
}


def _element(sp: Species) -> str:
    return "Rb" if sp.name.startswith("Rb") else "Cs"


def _hard_floor(sp: Species) -> int:
    """Per-species Ritz hard floor (Rb 8 / Cs 12), read from the spec-01
    series rows — not duplicated as a literal (audit R10 pattern)."""
    return next(iter(sp.series.values())).n_min_hard


def binding_hz_any(sp: Species, n: int, l: int, j: float) -> float:
    """Binding energy E_b/h > 0 [Hz] for ANY bound (n, l, j), spec 04 SS4.1.

    n >= hard floor: spec 01 quantum-defect machinery (the n < n_min_mhz
    100-MHz-grade warning is suppressed here — 100 MHz on an optical
    interval is < 1e-6 of any lifetime-relevant omega). Below the floor:
    measured NIST-ASD term energies (LOW_N_TERM_CM), binding = (NIST limit
    - term); missing rows raise IntegrityError (refuse-to-guess; the Ritz
    expansion below the floor is banned, conventions SS6 gap 3).
    """
    floor = _hard_floor(sp)
    if n == sp.ground_n and l == 0:
        # Ground state: binding energy IS the ionization energy E_I/h
        # (centroid convention, ruling R-11) [VERIFIED, spec 01].
        return sp.e_ion_hz
    if n >= floor or l >= 5:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(binding_energy_hz(sp, n, l, j))
    el = _element(sp)
    key = (el, n, l, int(round(2 * j)))
    term = LOW_N_TERM_CM.get(key)
    if term is None:
        raise IntegrityError(
            f"{sp.name} (n={n}, l={l}, j={j}) is below the Ritz hard floor "
            f"(n<{floor}) and has no NIST-ASD data row in LOW_N_TERM_CM — "
            "refusing to synthesize (spec 01 SS4.2 / conventions SS6 gap 3); "
            "add the NIST row with a source tag")
    return (NIST_ION_LIMIT_CM[el] - term) * _INV_CM_HZ


# ---------------------------------------------------------------------------
# Einstein rate helper (spec 04 Eq. 2.1 via the spec 03 angular layer)
# ---------------------------------------------------------------------------

def einstein_rate(omega_abs: float, radial_a0: float,
                  l_i: int, j_i: float, l_f: int, j_f: float) -> float:
    """Directional spontaneous-form rate coefficient [1/s], spec 04 Eq. 2.1.

    rate(i->f) = omega^3 e^2 a0^2 R^2 (2 j_f + 1) sixj{l_i j_i s; j_f l_f 1}^2
    L_max / (3 pi eps0 hbar c^3), with omega_abs = |E_i - E_f|/hbar [rad/s]
    and R the radial integral [a0]. For a DOWNWARD transition this is the
    Einstein A coefficient; for an upward partner it is the coefficient that
    multiplies nbar in the BBR-stimulated rate (spec 04 SS2.2 — the (2J'+1)
    of the destination state carries the g'/g asymmetry). Implemented via
    ``rydsim.angular.einstein_A`` with the bra-final reduced element
    (algebraically identical; unit-tested against the closed formula and the
    S1 6j-reduction identity, benchmark 04/B14).
    """
    d_racah = AU_DIPOLE * radial_a0 * reduced_dipole_j(l_f, j_f, l_i, j_i)
    return float(einstein_A(omega_abs, d_racah, j_i))


def _dline_radial_override(sp: Species, n1: int, l1: int, j1: float,
                           n2: int, l2: int, j2: float) -> float | None:
    """Experimentally-derived |R| [a0] for the D-line pair, else None.

    Spec 02 SS7.1 mandate: D-line dipoles come from experiment — the model
    potential carries a documented +5..10% bias exactly there. For the
    (ground nS1/2, ground_n P_j) pair the radial integral is rebuilt from
    the Steck reduced dipole [VERIFIED, rev 2.3.4]:
    |R| = sqrt(2 j_lower + 1) * d_Steck / |<S1/2||er||P_j>_Racah / (e R)|.
    This is what ties the A-sum machinery to Steck (benchmarks B1/B1b).
    """
    ng = sp.ground_n
    states = {(n1, l1, round(2 * j1)), (n2, l2, round(2 * j2))}
    if (ng, 0, 1) not in states:
        return None
    pj = None
    for (n, l, tj) in states:
        if l == 1 and n == ng:
            pj = tj / 2.0
    if pj is None:
        return None
    dline = sp.d2 if pj == 1.5 else sp.d1
    if dline is None:
        raise IntegrityError(f"{sp.name}: no D-line dataset for j'={pj}")
    d_racah_ea0 = steck_to_racah(dline.d_reduced_ea0, 0.5)
    ang = abs(reduced_dipole_j(0, 0.5, 1, pj))
    return d_racah_ea0 / ang


# ---------------------------------------------------------------------------
# Decay-channel tables (Method 1 engine)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DecayChannel:
    """One E1-coupled partner of a Rydberg state (spec 04 SS2.1-2.2).

    omega_hz: SIGNED transition frequency (E_state - E_partner)/h [Hz];
    > 0 means the partner lies BELOW (spontaneous decay allowed).
    rate_A / rate_B [1/s]: spec 04 Eq. 2.1 rate coefficient with the
    Method-A (model-potential) / Method-B (Coulomb) radial integral; for
    downward channels this is the Einstein A, for upward the nbar
    coefficient. radial_a0: Method-A radial integral [a0] (or the Steck
    experimental value on the D-line pair). spread_rel: |A-vs-B| radial
    magnitude spread (the shipped per-channel uncertainty, audit SS4 item 5).
    """

    n: int
    l: int
    j: float
    omega_hz: float
    rate_A: float
    rate_B: float
    radial_a0: float
    spread_rel: float

    @property
    def downward(self) -> bool:
        return self.omega_hz > 0.0


@dataclass(frozen=True)
class DecayTable:
    """All E1 decay/BBR channels of one state on a shared radial grid.

    gamma0_rad_s: 0-K radiative rate = sum of downward A (Method-A radials)
    [1/s]; gamma0_spread_rel: relative |Method A - Method B| spread of that
    sum — the shipped numerical uncertainty of 1/tau_0. bbr_rate(T) and
    effective_lifetime(T) evaluate the Planck-weighted sums (spec 04 SS2.2)
    without re-solving anything.
    """

    species: str
    n: int
    l: int
    j: float
    channels: tuple[DecayChannel, ...]
    n_top: int
    h: float

    @property
    def gamma0_rad_s(self) -> float:
        return sum(c.rate_A for c in self.channels if c.downward)

    @property
    def gamma0_spread_rel(self) -> float:
        a = self.gamma0_rad_s
        b = sum(c.rate_B for c in self.channels if c.downward)
        return abs(a - b) / a if a else 0.0

    def bbr_rate(self, T_K: float, *, check_tail: bool = True,
                 tail_tol: float = 1e-3) -> float:
        """Gamma_BBR(T) [1/s]: sum over ALL channels of rate * nbar(|omega|).

        nbar via expm1 with the x > 700 overflow guard (spec 04 SS4.2).
        check_tail: assert the three highest-n' upward channels contribute
        < tail_tol of the total — else the window n_top = n + dn_up was too
        small: IntegrityError, never a silently truncated rate. Default
        tail_tol = 1e-3 matches Beterov's own convergence statement
        ("n' <= n + 30 captures > 99.9 percent at 300 K", spec 04 SS4.2);
        the discrete-bound tail beyond n_top decays slowly (the free-bound
        continuation is the photoionization channel, modeled separately by
        Eq. 27) and bounds the truncation bias at ~1 percent of Gamma_BBR —
        far inside the 15 percent sum benchmarks (B9).
        """
        if T_K <= 0.0:
            return 0.0
        total = 0.0
        up: list[tuple[int, float]] = []
        for c in self.channels:
            x = H * abs(c.omega_hz) / (KB * T_K)
            nbar = 0.0 if x > 700.0 else 1.0 / math.expm1(x)
            w = c.rate_A * nbar
            total += w
            if not c.downward:
                up.append((c.n, w))
        if check_tail and up:
            top3 = {n for n, _ in sorted(up, key=lambda t: -t[0])[:3]}
            tail = sum(w for n, w in up if n in top3)
            if total > 0 and tail > tail_tol * total:
                raise IntegrityError(
                    f"BBR sum unconverged for {self.species} (n={self.n}, "
                    f"l={self.l}, j={self.j}) at T={T_K} K: top-3 upward "
                    f"channels carry {tail/total:.1e} of the rate "
                    f"(> {tail_tol:.0e}). Increase dn_up (spec 04 SS4.2) — "
                    "refusing a truncated rate")
        return total

    def effective_lifetime(self, T_K: float) -> float:
        """tau_eff(T) = 1/(Gamma_0 + Gamma_BBR) [s] (Beterov Eq. 6)."""
        return 1.0 / (self.gamma0_rad_s + self.bbr_rate(T_K))


_TABLE_CACHE: dict[tuple, DecayTable] = {}


def clear_lifetime_cache() -> None:
    """Drop cached DecayTables (the radial layer keeps its own cache)."""
    _TABLE_CACHE.clear()


def _partner_series(l: int, j: float) -> list[tuple[int, float]]:
    """E1+dipole-allowed (l', j') partner series: l' = l+-1, |j-j'| <= 1."""
    out = []
    for lp in (l - 1, l + 1):
        if lp < 0:
            continue
        for jp in (lp - 0.5, lp + 0.5):
            if jp > 0 and abs(j - jp) <= 1.0 + 1e-9:
                out.append((lp, jp))
    return out


def _nu_for(sp: Species, n: int, l: int, j: float) -> float:
    """Effective quantum number for the radial solver: spec-01 n* above the
    hard floor, sqrt(c R_M / E_b(measured)) below it (NIST rows)."""
    if n >= _hard_floor(sp) or l >= 5:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(n_star(sp, n, l, j))
    return math.sqrt(rydberg_constant_hz(sp) / binding_hz_any(sp, n, l, j))


def _ab_ceiling_low(nu_min: float, me_abs: float) -> float:
    """Spec 02 SS6 A-vs-B spread ceilings (B8/B11): 1e-4 for MEs >= 50 a0 at
    nu >= 15, 1e-3 for smaller elements, 3e-2 below nu = 15 (transcribed
    from spec 02 with the same regime boundaries; enforced only above the
    Ritz floor — see module docstring deviation note)."""
    if nu_min < 15.0:
        return 3e-2
    return 1e-4 if me_abs >= 50.0 else 1e-3


def decay_table(sp: Species, n: int, l: int, j: float, *,
                dn_up: int = 25, h: float = 1e-3) -> DecayTable:
    """Build (and cache) the full E1 channel table of |n l j> (spec 04 SS4).

    Enumerates every partner series (l' = l +- 1, dipole-allowed j') from
    the ground-configuration minimum n' up to n_top = n + dn_up; solves all
    radial states with Method A (MSD94 model potential) AND Method B
    (pure Coulomb) on ONE shared grid r in [alpha_c^(1/3), 2 n_top (n_top +
    15)] a0 with step h [sqrt(a0)] (spec 02 grid rules — sharing the grid is
    what makes the sums affordable; each state is solved exactly once per
    method). Transition frequencies come from actual level energies, never
    n_eff differences (spec 04 SS4.2 pitfall 3). Rates via
    :func:`einstein_rate`; the D-line pair uses the Steck experimental
    radial element (see ``_dline_radial_override``).

    Above the Ritz floor an A-vs-B spread beyond the spec-02 ceiling raises
    IntegrityError (regression); below the floor the spread is recorded and
    shipped as uncertainty (documented deviation, module docstring).
    """
    key = (sp.name, n, l, j, dn_up, h)
    hit = _TABLE_CACHE.get(key)
    if hit is not None:
        return hit
    el = _element(sp)
    floor = _hard_floor(sp)
    n_top = n + dn_up
    e_state = binding_hz_any(sp, n, l, j)

    def solve(nn: int, ll: int, jj: float, method: str):
        # Each state on its NATURAL outer extent 2 n (n + 15) a0 (spec 02
        # SS4.1): a deeply bound state on the n_top grid would traverse an
        # e^~4000 forbidden-region dynamic range and underflow (measured —
        # the divergence guard then annihilates the solution). Grids share
        # h and x[0], so radial_matrix_element integrates the shared index
        # range; the excess tail is exponentially negligible (spec 02 B14).
        return radial_wavefunction(
            sp, nn, ll, jj, nu=_nu_for(sp, nn, ll, jj), h=h,
            r_outer=2.0 * nn * (nn + 15.0), method=method)

    wf_a0 = solve(n, l, j, "model_potential")
    wf_b0 = solve(n, l, j, "coulomb")
    nu0 = wf_a0.nu
    channels: list[DecayChannel] = []
    for lp, jp in _partner_series(l, j):
        n_min = _N_MIN_SERIES[el].get(lp)
        if n_min is None:
            raise IntegrityError(
                f"no sourced ground-configuration minimum for {el} l'={lp}; "
                "l' >= 4 partner series are out of scope for the lifetime "
                "sums (no sub-floor data rows) — refusing")
        for npp in range(max(n_min, lp + 1), n_top + 1):
            e_part = binding_hz_any(sp, npp, lp, jp)
            omega_hz = e_part - e_state       # >0: partner below (downward)
            if omega_hz == 0.0:
                continue
            omega_rad = 2.0 * math.pi * abs(omega_hz)
            override = _dline_radial_override(sp, n, l, j, npp, lp, jp)
            wa = solve(npp, lp, jp, "model_potential")
            wb = solve(npp, lp, jp, "coulomb")
            r_a = radial_matrix_element(wf_a0, wa, 1)
            r_b = radial_matrix_element(wf_b0, wb, 1)
            spread = abs(abs(r_a) - abs(r_b)) / abs(r_a) if r_a else 0.0
            nu_min = min(nu0, wa.nu)
            if (min(n, npp) >= floor
                    and spread > _ab_ceiling_low(nu_min, abs(r_a))):
                raise IntegrityError(
                    f"A-vs-B radial consensus failed above the Ritz floor "
                    f"for {sp.name} ({n},{l},{j})->({npp},{lp},{jp}): "
                    f"spread {spread:.2e} (A={r_a:.4e}, B={r_b:.4e} a0) — "
                    "radial regression (spec 02 SS6 / audit SS3 item 8)")
            if override is not None:
                r_use_a = r_use_b = override
                spread = 0.0
            else:
                r_use_a, r_use_b = r_a, r_b
            channels.append(DecayChannel(
                n=npp, l=lp, j=jp, omega_hz=omega_hz,
                rate_A=einstein_rate(omega_rad, abs(r_use_a), l, j, lp, jp),
                rate_B=einstein_rate(omega_rad, abs(r_use_b), l, j, lp, jp),
                radial_a0=float(r_use_a), spread_rel=spread))
    if not any(c.downward for c in channels):
        raise IntegrityError(
            f"{sp.name} ({n},{l},{j}): no downward E1 channel found — "
            "not a valid excited state for the lifetime engine")
    table = DecayTable(species=sp.name, n=n, l=l, j=j,
                       channels=tuple(channels), n_top=n_top, h=h)
    _TABLE_CACHE[key] = table
    return table


def radiative_lifetime_sum(sp: Species, n: int, l: int, j: float, *,
                           dn_up: int = 25, h: float = 1e-3) -> float:
    """tau_0 [s] from the Einstein-A sum (Method 1; Beterov Eq. 4).

    1/tau_0 = sum of A over all lower states. The 0-K sum needs no upward
    states, but the table is built with the BBR window anyway so the cached
    solves are shared with tau_eff. tau_0/1e-9 gives ns (lock #1: SI out)."""
    return 1.0 / decay_table(sp, n, l, j, dn_up=dn_up, h=h).gamma0_rad_s


def bbr_rate_sum(sp: Species, n: int, l: int, j: float, T_K: float, *,
                 dn_up: int = 25, h: float = 1e-3) -> float:
    """Gamma_BBR(T) [1/s] from the full Planck-weighted sum (Beterov Eq. 5)."""
    return decay_table(sp, n, l, j, dn_up=dn_up, h=h).bbr_rate(T_K)


def effective_lifetime_sum(sp: Species, n: int, l: int, j: float,
                           T_K: float, *, dn_up: int = 25,
                           h: float = 1e-3) -> float:
    """tau_eff(T) = 1/(Gamma_0 + Gamma_BBR) [s] from the sums (Beterov Eq. 6)."""
    return decay_table(sp, n, l, j, dn_up=dn_up, h=h).effective_lifetime(T_K)


# ---------------------------------------------------------------------------
# Method 2 — Beterov fitted scalings (spec 04 SS2.1-2.2, tables 3.1-3.2)
# ---------------------------------------------------------------------------

#: tau_0 = tau_s * n_eff^delta [ns], valid 15 <= n <= 80.
#: Source: Beterov, Ryabtsev, Tretyakov, Entin, PRA 79, 052504 (2009) +
#: Erratum PRA 80, 059902 (2009), Table II via arXiv:0810.0339v4; audit R21
#: re-verified verbatim from ar5iv this corpus [VERIFIED]. Identical fits
#: for Rb-85/Rb-87 (spec 04 SS3.1).
BETEROV_TAU_FIT: dict[tuple[str, str], tuple[float, float]] = {
    ("Rb", "S1/2"): (1.368, 3.0008),
    ("Rb", "P1/2"): (2.4360, 2.9989),
    ("Rb", "P3/2"): (2.2214, 3.0026),
    ("Rb", "D3/2"): (1.0761, 2.9898),
    ("Rb", "D5/2"): (1.0687, 2.9897),
    ("Cs", "S1/2"): (1.2926, 3.0005),
    ("Cs", "P1/2"): (2.9921, 2.9892),
    ("Cs", "P3/2"): (3.2849, 2.9875),
    ("Cs", "D3/2"): (0.6580, 2.9944),
    ("Cs", "D5/2"): (0.6681, 2.9941),
}

#: Gamma_BBR fit coefficients (A, B, C, D), Beterov Eq. 14 / Table I
#: (arXiv:0810.0339v4) [VERIFIED, audit R21].
BETEROV_BBR_FIT: dict[tuple[str, str], tuple[float, float, float, float]] = {
    ("Rb", "S1/2"): (0.134, 0.251, 2.567, 4.426),
    ("Rb", "P1/2"): (0.053, 0.128, 2.183, 3.989),
    ("Rb", "P3/2"): (0.046, 0.109, 2.085, 3.901),
    ("Rb", "D3/2"): (0.033, 0.084, 1.912, 3.716),
    ("Rb", "D5/2"): (0.032, 0.082, 1.898, 3.703),
    ("Cs", "S1/2"): (0.123, 0.231, 2.517, 4.375),
    ("Cs", "P1/2"): (0.041, 0.072, 1.693, 3.607),
    ("Cs", "P3/2"): (0.038, 0.056, 1.552, 3.505),
    ("Cs", "D3/2"): (0.038, 0.076, 1.790, 3.656),
    ("Cs", "D5/2"): (0.036, 0.073, 1.770, 3.636),
}

# Fitted constants of Beterov Eqs. 14/16 — these roundings are PART OF THE
# FIT (spec 04 SS2.2: 315780 K is their rounding of E_h/kB = 315775.02 K;
# 2.14e10 s^-1 their 21.4 ns^-1). Never "corrected" to CODATA.
_BETEROV_RATE_CONST = 2.14e10        # [1/s]
_BETEROV_EH_KB = 315780.0            # [K]

_SERIES_NAME = {(0, 0.5): "S1/2", (1, 0.5): "P1/2", (1, 1.5): "P3/2",
                (2, 1.5): "D3/2", (2, 2.5): "D5/2"}


def beterov_series_key(l: int, j: float) -> str:
    """Series label for the Beterov tables; IntegrityError for series they
    never fitted (F and higher — refuse, don't extrapolate)."""
    name = _SERIES_NAME.get((l, float(j)))
    if name is None:
        raise IntegrityError(
            f"Beterov 2009 fits exist only for nS/nP/nD (got l={l}, j={j}) "
            "— use the sum path (spec 04 SS3.1)")
    return name


def _fit_n_eff(sp: Species, n: np.ndarray | int, l: int, j: float
               ) -> np.ndarray | float:
    """n_eff for the fit paths from spec 01 machinery (R-14) with the
    15 <= n <= 80 validity gate (audit SS3 item 14 -> IntegrityError)."""
    n_arr = np.atleast_1d(np.asarray(n))
    if np.any(n_arr < 15) or np.any(n_arr > 80):
        raise IntegrityError(
            f"Beterov fit valid only for 15 <= n <= 80, got n={n} "
            "(spec 04 SS2.1/SS7.1 / audit SS3 item 14) — use method='sum'")
    return n_star(sp, n, l, j)


def radiative_lifetime_fit(sp: Species, n: np.ndarray | int, l: int,
                           j: float) -> np.ndarray | float:
    """tau_0 [s] from the Beterov fit tau_s * n_eff^delta (Method 2).

    Vectorized over n. Sources: BETEROV_TAU_FIT [VERIFIED]. n_eff from the
    spec-01 machinery incl. Form B for Cs S/D5/2 (ruling R-14; the deviation
    from Beterov's own n_eff is << the fit's 5% accuracy)."""
    tau_s, delta = BETEROV_TAU_FIT[(_element(sp), beterov_series_key(l, j))]
    ne = np.asarray(_fit_n_eff(sp, n, l, j), dtype=float)
    out = tau_s * ne**delta * 1e-9
    return float(out) if np.ndim(n) == 0 else out


def bbr_rate_fit(sp: Species, n: np.ndarray | int, l: int, j: float,
                 T_K: float) -> np.ndarray | float:
    """Gamma_BBR [1/s], Beterov Eq. 14 analytic fit (Method 2).

    Gamma = (A/n_eff^D) * 2.14e10 / (exp(315780 B / (n_eff^C T)) - 1),
    coefficients verbatim (BETEROV_BBR_FIT [VERIFIED]); the fitted constants
    2.14e10 and 315780 are part of the fit, kept as printed. expm1 for the
    denominator. Stated accuracy < 5% for 15 < n < 80 vs their sums."""
    A, B, Cc, D = BETEROV_BBR_FIT[(_element(sp), beterov_series_key(l, j))]
    ne = np.asarray(_fit_n_eff(sp, n, l, j), dtype=float)
    if T_K <= 0.0:
        return 0.0 if np.ndim(n) == 0 else np.zeros_like(ne)
    x = _BETEROV_EH_KB * B / (ne**Cc * T_K)
    out = (A / ne**D) * _BETEROV_RATE_CONST / np.expm1(x)
    return float(out) if np.ndim(n) == 0 else out


def effective_lifetime_fit(sp: Species, n: np.ndarray | int, l: int,
                           j: float, T_K: float) -> np.ndarray | float:
    """tau_eff(T) [s], Beterov Eq. 16 analytic combination of both fits."""
    g0 = 1.0 / np.asarray(radiative_lifetime_fit(sp, n, l, j), dtype=float)
    gb = np.asarray(bbr_rate_fit(sp, n, l, j, T_K), dtype=float)
    out = 1.0 / (g0 + gb)
    return float(out) if np.ndim(n) == 0 else out


def bbr_rate_farley_wing(sp: Species, n: np.ndarray | int, l: int, j: float,
                         T_K: float) -> np.ndarray | float:
    """Farley-Wing high-T/low-omega BBR limit [1/s] — CROSS-CHECK ONLY.

    Gamma_FW = (4/3) alpha_fs^3 (kB T / E_h) / n_eff^2 in atomic units of
    rate, converted with E_h/hbar — every factor computed from
    scipy.constants at runtime (lock #15; the spec's 2.034e7 (T/300)/n^2 is
    a check value reproduced in tests). Expected within ~+30/-20% of the
    full sum for n ~ 40-70 at 300 K (spec 04 SS2.2 / benchmark B9b).
    Source: Farley & Wing, PRA 23, 2397 (1981) [LITERATURE-RECALL]."""
    ne = np.asarray(n_star(sp, n, l, j), dtype=float)
    rate_au = (4.0 / 3.0) * ALPHA_FS**3 * (KB * T_K / E_H) / ne**2
    out = rate_au * (E_H / HBAR)
    return float(out) if np.ndim(n) == 0 else out


# ---------------------------------------------------------------------------
# Direct BBR photoionization — Beterov NJP 11, 013052 (2009) Eq. 27
# ---------------------------------------------------------------------------

#: Quantum-defect differences of the NJP Eq. 27 fit, Table 1
#: (arXiv:0807.2535) [VERIFIED]. Part of the fit — kept verbatim, NOT
#: recomputed from spec 01 defects.
NJP_QD_DIFF: dict[tuple[str, str], float] = {
    ("Rb", "S-P"): 0.490134, ("Rb", "P-D"): 1.29456, ("Rb", "D-F"): 1.34636,
    ("Cs", "S-P"): 0.458701, ("Cs", "P-D"): 1.12661, ("Cs", "D-F"): 2.43295,
}

#: Empirical scaling coefficients A_L, NJP Table 2 [VERIFIED].
NJP_A_L: dict[str, dict[int, float]] = {
    "Rb": {0: 1.0, 1: 1.0, 2: 0.6},
    "Cs": {0: 0.85, 1: 1.1, 2: 0.35},
}

# Fitted constants of NJP Eq. 27 (their roundings; 157890 = 315780/2).
_NJP_CONST = 11500.0
_NJP_RYD_K = 157890.0


def bbr_ionization_rate(sp: Species, n: int, l: int, j: float,
                        T_K: float) -> float:
    """Direct BBR photoionization rate W [1/s], Beterov NJP 11 Eq. (27).

    W = A_L * 11500 T / n_eff^(7/3) * [cos^2(D+ + pi/6) + cos^2(D- - pi/6)]
        * ln(1/(1 - exp(-157890/(T n_eff^2))))
    with D+ = pi (mu_L - mu_{L+1}), D- = pi (mu_{L-1} - mu_L) from the NJP
    Table-1 defect differences (part of the fit); nS keeps only the first
    term. The log is evaluated as -log1p(-exp(-x)) (stable both limits).
    Accuracy ~50% with tabulated A_L, AND systematically LOW by ~1.3-2x vs
    total ionization (direct channel only — spec 04 SS2.2/SS7.4). Validity
    15 <= n <= 80 (same window as the parent fits); l > 2 unfitted -> raise.
    Worked value (benchmark B13): Rb 50S, 300 K -> ~1.5e2 1/s.
    """
    el = _element(sp)
    if l > 2:
        raise IntegrityError(
            f"NJP Eq. 27 coefficients exist only for S/P/D (got l={l}) — "
            "refusing (NJP Table 2)")
    ne = float(_fit_n_eff(sp, n, l, j))
    if T_K <= 0.0:
        return 0.0
    names = ["S-P", "P-D", "D-F"]
    d_plus = math.pi * NJP_QD_DIFF[(el, names[l])]
    bracket = math.cos(d_plus + math.pi / 6.0) ** 2
    if l >= 1:
        d_minus = math.pi * NJP_QD_DIFF[(el, names[l - 1])]
        bracket += math.cos(d_minus - math.pi / 6.0) ** 2
    x = _NJP_RYD_K / (T_K * ne * ne)
    log_term = -math.log1p(-math.exp(-x)) if x < 700.0 else math.exp(-x)
    return (NJP_A_L[el][l] * _NJP_CONST * T_K / ne ** (7.0 / 3.0)
            * bracket * log_term)


# ---------------------------------------------------------------------------
# Cross-check machinery (spec 04 SS4.3 S2/S3 — first-class validation of the
# whole dipole chain)
# ---------------------------------------------------------------------------

def lifetime_crosscheck(sp: Species, n: int, l: int, j: float, *,
                        tolerance_rel: float = 0.10, dn_up: int = 25,
                        h: float = 1e-3) -> CrossCheck:
    """tau_0: A-sum vs Beterov fit (self-check S2, band 10 percent).

    A disagreement beyond the band means the dipole chain (radial x angular
    x energies) or the fit transcription regressed; ``.require()`` raises."""
    return CrossCheck(
        quantity=f"tau_0({sp.name} n={n} l={l} j={j})",
        methods={
            "einstein_sum": radiative_lifetime_sum(sp, n, l, j,
                                                   dn_up=dn_up, h=h),
            "beterov_fit": float(radiative_lifetime_fit(sp, n, l, j)),
        },
        unit="s", tolerance_rel=tolerance_rel)


def bbr_crosscheck(sp: Species, n: int, l: int, j: float, T_K: float, *,
                   tolerance_rel: float = 0.20, dn_up: int = 25,
                   h: float = 1e-3) -> CrossCheck:
    """Gamma_BBR: full sum vs Beterov analytic fit (self-check S3, 20%)."""
    return CrossCheck(
        quantity=f"Gamma_BBR({sp.name} n={n} l={l} j={j}, T={T_K} K)",
        methods={
            "planck_sum": bbr_rate_sum(sp, n, l, j, T_K, dn_up=dn_up, h=h),
            "beterov_fit": float(bbr_rate_fit(sp, n, l, j, T_K)),
        },
        unit="1/s", tolerance_rel=tolerance_rel)


# ---------------------------------------------------------------------------
# Resonant self-broadening (spec 04 SS2.3.5(i), ruling R-6; SI per R-21)
# ---------------------------------------------------------------------------

#: Measured self-broadening beta/2pi [Hz m^3] (R-6; converted from the
#: printed Hz cm^3 at data-load time: 1 Hz cm^3 = 1e-6 Hz m^3, R-21).
SELF_BROADENING_MEASURED: dict[tuple[str, str], SourcedValue] = {
    ("Rb", "D2"): SourcedValue(
        1.10e-13, "Hz m^3",
        "Kondo et al., PRA 73, 062504 (2006) via Weller 2011 Table I "
        "(1.10(17)e-7 Hz cm^3)", Confidence.VERIFIED, uncertainty=0.17e-13),
    ("Rb", "D1"): SourcedValue(
        0.69e-13, "Hz m^3",
        "Weller et al., J. Phys. B 44, 195006 (2011) (0.69(4)e-7 Hz cm^3)",
        Confidence.VERIFIED, uncertainty=0.04e-13),
    ("Cs", "D2"): SourcedValue(
        1.15e-13, "Hz m^3",
        "Akulshin et al., JETP Lett. 36, 303 (1982) via Weller Table I "
        "(1.15(23)e-7 Hz cm^3)", Confidence.VERIFIED, uncertainty=0.23e-13),
}


def self_broadening_beta_hz_m3(sp: Species, line: Literal["D1", "D2"] = "D2",
                               *, which: Literal["theory", "measured"]
                               = "theory") -> SourcedValue:
    """Resonant self-broadening coefficient beta/2pi [Hz m^3] (R-6, R-21).

    theory (closed form, Weller Eqs. 2-3 / Lewis 1980):
        beta_D1/2pi = Gamma_1 (lambda_1/2pi)^3
        beta_D2/2pi = sqrt(2) Gamma_2 (lambda_2/2pi)^3
    computed at runtime from the Steck Gamma and vacuum wavelength (no
    typed coefficient; benchmark B11 checks 1.03e-7 Hz cm^3 Rb / 1.16e-7 Cs
    to 3%). measured: the Kondo/Akulshin values (SELF_BROADENING_MEASURED).
    The probe-line FWHM contribution is (beta/2pi) N [Hz]; the coherence
    rate entry is gamma_ge += pi (beta/2pi) N (lock #7)."""
    dline = sp.d2 if line == "D2" else sp.d1
    if dline is None:
        raise IntegrityError(f"{sp.name} carries no {line} dataset")
    if which == "measured":
        sv = SELF_BROADENING_MEASURED.get((_element(sp), line))
        if sv is None:
            raise IntegrityError(
                f"no measured self-broadening coefficient for "
                f"{_element(sp)} {line} — use which='theory' (spec 04 SS3.5)")
        return sv
    factor = math.sqrt(2.0) if line == "D2" else 1.0
    lam_bar = dline.lambda_vac_m / (2.0 * math.pi)
    val = factor * dline.gamma_rad_s * lam_bar**3
    return SourcedValue(
        val, "Hz m^3",
        f"closed form {'sqrt(2) ' if line == 'D2' else ''}Gamma "
        f"(lambda/2pi)^3, Weller 2011 Eqs. 2-3 / Lewis 1980, from Steck "
        f"rev 2.3.4 {sp.name} {line} data", Confidence.COMPUTED)


def self_broadening_fwhm_hz(sp: Species, line: Literal["D1", "D2"],
                            N_m3: np.ndarray | float, *,
                            which: Literal["theory", "measured"] = "theory",
                            ) -> np.ndarray | float:
    """Self-broadening FWHM contribution (beta/2pi) N [Hz]; N in m^-3 (R-21).

    Valid in the impact regime (detunings below the Weisskopf frequency
    ~2pi*4 GHz, spec 04 SS7.5). gamma_ge (rad/s) adds pi * this value."""
    beta = self_broadening_beta_hz_m3(sp, line, which=which).value
    out = beta * np.asarray(N_m3, dtype=float)
    return float(out) if np.ndim(N_m3) == 0 else out


# ---------------------------------------------------------------------------
# Transit, vdW, Fermi, laser and Stark budget terms
# ---------------------------------------------------------------------------

def transit_fwhm_hz(mass_kg: float, T_K: float, w_m: float) -> float:
    """Transit-time FWHM Delta_nu_tt = gamma_t / pi [Hz] (ruling R-3).

    gamma_t from rydsim.cell.transit_rate = sqrt(2 ln 2) <v_perp>/w0 [rad/s]
    (single implementation, audit R17); equals the spec 04 SS2.3.3 form
    0.3748 sqrt(pi kB T / 2 m)/w. Convention accuracy ~+-20% (benchmark
    B12: 106 kHz for Rb-87, 300 K, w = 0.75 mm). w_m = 1/e^2 INTENSITY
    radius."""
    return transit_rate(mass_kg, T_K, w_m) / math.pi


#: 1 a.u. of C6 expressed as C6/h [Hz m^6] — computed, not typed (lock #15);
#: check value 1.4448e-19 GHz um^6 (spec 04 SS2.3.5(iii)).
AU_C6_HZ_M6 = (E_H / H) * A0**6


def c6_over_h_rb_ns(n: int) -> SourcedValue:
    """|C6|/h [Hz m^6] for a Rb nS1/2 pair (R-26 bookkeeping).

    Singer et al., J. Phys. B 38, S295 (2005) fit
    |C6| = n^11 |11.97 - 0.8486 n + 3.385e-3 n^2| a.u.
    [LITERATURE-RECALL — order-of-magnitude gate only, audit R11; self-check
    |C6|(60S) ~ 140 GHz um^6 in tests]. The pair interaction is repulsive
    for Rb nS. Stored as C6/h so Delta_nu_vdW = (C6/h)/r^6 with NO second
    /h (ruling R-26)."""
    c6_au = abs(n**11 * (11.97 - 0.8486 * n + 3.385e-3 * n * n))
    return SourcedValue(
        c6_au * AU_C6_HZ_M6, "Hz m^6",
        "Singer et al., J. Phys. B 38, S295 (2005) Rb nS1/2 fit",
        Confidence.LITERATURE_RECALL,
        note="order-of-magnitude warning gate (audit R11); repulsive pair")


def vdw_dephasing_hz(c6_over_h_hz_m6: float, N_r_m3: float) -> float:
    """Nearest-neighbour vdW shift scale Delta_nu = (C6/h)/r_nn^6 [Hz].

    r_nn = (3/(4 pi N_r))^(1/3); scales as N_r^2 n^11. INHOMOGENEOUS
    (quasi-static nn-distance distribution) — never a Lindblad rate (audit
    SS3 item 15); the budget carries it in ``inhomo`` and warns above
    0.1 MHz (spec 04 SS2.3.5(iii))."""
    if N_r_m3 <= 0.0:
        return 0.0
    r_nn = (3.0 / (4.0 * math.pi * N_r_m3)) ** (1.0 / 3.0)
    return c6_over_h_hz_m6 / r_nn**6


#: e^- - Rb(5S) triplet scattering length [a0]; Bahrim & Thumm (2000/2001)
#: [LITERATURE-RECALL; self-check: Fermi coefficient -9.9e-8 Hz cm^3].
A_S_ELECTRON_RB5S_A0 = SourcedValue(
    -16.1, "a0", "Bahrim & Thumm, PRA 61, 022722 (2000)",
    Confidence.LITERATURE_RECALL)


def fermi_shift_hz(sp: Species, N_g_m3: float) -> float:
    """Mean-field Fermi line shift of the Rydberg level [Hz], spec 04
    SS2.3.5(ii): Delta_nu = (hbar a_s / m_e) N_g (n-independent, n >~ 30).

    Coefficient computed at runtime from A_S_ELECTRON_RB5S_A0 (check value
    -9.9e-8 Hz cm^3 for Rb). Only Rb has a sourced scattering length —
    Cs raises IntegrityError rather than guessing (refuse-to-guess)."""
    if _element(sp) != "Rb":
        raise IntegrityError(
            f"no sourced e^-/ground scattering length for {sp.name} — "
            "refusing the Fermi shift (spec 04 SS3.5 lists Rb only)")
    return HBAR * (A_S_ELECTRON_RB5S_A0.value * A0) / M_E * N_g_m3


def laser_dephasing_rates(fwhm_probe_hz: float, fwhm_coupling_hz: float,
                          correlation: float = 0.0
                          ) -> tuple[float, float, float]:
    """(gamma_ge, gamma_er, gamma_gr) laser pure-dephasing rates [rad/s].

    Phase-diffusion model (lock #8 / spec 04 SS2.3.4): gamma_ge = pi Dnu_p,
    gamma_er = pi Dnu_c, and for the LADDER two-photon coherence the rates
    ADD with the mutual-coherence cross term (ruling R-16):
        gamma_gr = pi (Dnu_p + Dnu_c + 2 c sqrt(Dnu_p Dnu_c)),  c in [-1,1].
    c = 0 independent lasers (default); c > 0 common-mode (WORSE — common-
    mode noise does not cancel in a ladder); c < 0 engineered
    anti-correlation only."""
    if not -1.0 <= correlation <= 1.0:
        raise ValueError(f"correlation c={correlation} outside [-1, 1]")
    if fwhm_probe_hz < 0.0 or fwhm_coupling_hz < 0.0:
        raise ValueError("laser FWHM linewidths must be >= 0")
    g_ge = math.pi * fwhm_probe_hz
    g_er = math.pi * fwhm_coupling_hz
    g_gr = math.pi * (fwhm_probe_hz + fwhm_coupling_hz
                      + 2.0 * correlation
                      * math.sqrt(fwhm_probe_hz * fwhm_coupling_hz))
    return g_ge, g_er, g_gr


def stark_inhomogeneous_fwhm_hz(alpha0_over_h_hz_v2m2: float,
                                e_rms_v_m: float) -> float:
    """Inhomogeneous Stark width ~ (1/2) (alpha_0/h) <E^2> [Hz] (SS2.3.6).

    alpha0_over_h in Hz/(V/m)^2 — supplied by the CALLER from the Stark
    engine (ruling R-5: the spec-04 '~6e2 MHz/(V/cm)^2' Rb 50S figure is
    banned; the verified value is 50.5 MHz/(V/cm)^2 = 5050 Hz/(V/m)^2,
    spec 07). INHOMOGENEOUS: enters as a P(E) average, never a Lindblad
    rate (audit SS3 item 15)."""
    return 0.5 * abs(alpha0_over_h_hz_v2m2) * e_rms_v_m**2


# ---------------------------------------------------------------------------
# The dephasing budget (spec 04 SS2.3-2.4, SS5)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CellParams:
    """Vapor-cell parameters, SI internally (ruling R-21; the spec 04 SS5
    *_cm3 field names are display conventions, not storage).

    alpha0_over_h_hz_v2m2: scalar polarizability alpha_0/h of the Rydberg
    state in Hz/(V/m)^2, from the Stark engine (ruling R-5) — REQUIRED
    whenever E_stray_v_m > 0; there is deliberately no default.
    T_bbr_K: BBR environment temperature if different from the cell
    (spec 04 SS7.3); None -> T_cell_K.
    """

    T_cell_K: float = 300.0
    T_bbr_K: float | None = None
    N_ground_m3: float | None = None      # None -> Alcock model (rydsim.cell)
    beam_waist_m: float = 0.75e-3         # 1/e^2 intensity radius
    E_stray_v_m: float = 1.0              # 10 mV/cm
    N_ion_m3: float = 0.0
    N_rydberg_m3: float = 0.0
    alpha0_over_h_hz_v2m2: float | None = None


@dataclass(frozen=True)
class LaserParams:
    """Laser linewidths (Lorentzian FWHM, Hz) + mutual coherence (R-16)."""

    fwhm_probe_hz: float = 2.0e5
    fwhm_coupling_hz: float = 2.0e5
    noise_correlation: float = 0.0        # c in [-1, 1]; ladder: c > 0 hurts


@dataclass(frozen=True)
class DephasingBudget:
    """Term-by-term ladder-EIT dephasing budget (spec 04 SS2.3-2.4, SS5).

    All *rates* are angular [rad/s] (lock #2); /(2 pi) for Hz where a field
    name says so. Homogeneous entries feed Lindblad/gamma_ij; INHOMOGENEOUS
    terms live in ``inhomo`` as {name: descriptor} magnitudes/distributions
    and are NEVER converted to rates (audit SS3 item 15 — adding
    Dk sigma_v to gamma_gr is the named forbidden error).
    """

    species: str
    state: tuple[int, int, float]
    Gamma_e: float                  # intermediate-state population decay
    Gamma_r_rad: float              # Rydberg radiative decay
    Gamma_r_bbr: float              # Rydberg BBR depopulation
    W_bbr_ion: float                # direct BBR photoionization
    gamma_transit: float            # reset rate, fast mode (R-3)
    gamma_laser_ge: float
    gamma_laser_er: float
    gamma_laser_gr: float
    gamma_self_ge: float            # pi (beta/2pi) N_g
    gamma_rydgnd_gr: float          # Rydberg-ground broadening [UNVERIFIED x2]
    shift_fermi_hz: float           # deterministic line shift [Hz]
    Omega_p: float
    Omega_c: float
    N_ground_m3: float
    inhomo: dict = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def Gamma_r(self) -> float:
        """Total Rydberg population decay Gamma_0 + Gamma_BBR [rad/s]."""
        return self.Gamma_r_rad + self.Gamma_r_bbr

    def gamma_ij(self) -> dict[tuple[str, str], float]:
        """Coherence decay matrix [rad/s], spec 04 SS2.4:
        gamma_ij = (Gamma_i + Gamma_j)/2 + pure dephasings touching ij +
        gamma_t (the measure-and-replace transit channel decays every
        coherence at gamma_t — rydsim.lindblad collapse-set convention).
        Unit-tested by S7/B15 (the factor-2 conventions are the #1 bug
        source, spec 04 SS2.4 row 4)."""
        ge = (self.Gamma_e / 2.0 + self.gamma_laser_ge + self.gamma_self_ge
              + self.gamma_transit)
        gr = (self.Gamma_r / 2.0 + self.gamma_laser_gr + self.gamma_rydgnd_gr
              + self.gamma_transit)
        er = ((self.Gamma_e + self.Gamma_r) / 2.0 + self.gamma_laser_er
              + self.gamma_self_ge + self.gamma_rydgnd_gr + self.gamma_transit)
        return {("g", "e"): ge, ("g", "r"): gr, ("e", "r"): er}

    def eit_fwhm_sanity_rad_s(self) -> float:
        """2 gamma_gr + Omega_c^2/(2 gamma_ge) [rad/s] — SANITY FORM ONLY
        (ruling R-24, spec 06 variant); hot-cell linewidths come only from
        the velocity integral. Power broadening is emergent, never a rate
        (spec 04 SS2.3.7)."""
        g = self.gamma_ij()
        return 2.0 * g[("g", "r")] + self.Omega_c**2 / (2.0 * g[("g", "e")])

    def warnings(self) -> list[str]:
        """Operational warnings per spec 04 (vdW gate, UNVERIFIED terms)."""
        out = list(self.notes)
        vdw = self.inhomo.get("vdw", {}).get("delta_nu_hz", 0.0)
        if vdw > 1e5:
            out.append(
                f"vdW dephasing {vdw/1e6:.2f} MHz > 0.1 MHz: dominant "
                "systematic at these powers/densities (spec 04 SS2.3.5(iii))")
        if self.gamma_rydgnd_gr > 0.0:
            out.append(
                "Rydberg-ground broadening = 0.5|Fermi shift| is an "
                "order-of-magnitude estimate with x2 uncertainty "
                "[UNVERIFIED, spec 04 SS2.3.5(ii)]")
        return out


def build_budget(sp: Species, state: tuple[int, int, float],
                 cell: CellParams, lasers: LaserParams,
                 Omega_p: float, Omega_c: float, *,
                 line: Literal["D1", "D2"] = "D2",
                 gamma_r_method: Literal["fit", "sum"] = "fit",
                 self_broadening: Literal["theory", "measured"] = "theory",
                 ) -> DephasingBudget:
    """Assemble the term-by-term dephasing budget (spec 04 SS5 API).

    Omega_p / Omega_c: angular Rabi frequencies [rad/s] (lock #4). Rates:
    Gamma_e from the Steck datasheet; Gamma_r from the Beterov fit
    (gamma_r_method='fit', fast default) or the Einstein/Planck sums
    ('sum'); transit from rydsim.cell.transit_rate (R-3, single
    implementation); laser terms per R-16; self-broadening beta N (R-6);
    Fermi shift+broadening (Rb only — Cs refuses); vdW magnitude from the
    Singer C6 fit (Rb nS only; other series with N_rydberg > 0 refuse
    rather than guess a C6); Stark width from the CALLER-supplied
    alpha (R-5 — E_stray > 0 with alpha None raises IntegrityError).
    Inhomogeneous terms land in ``inhomo`` as descriptors, never rates
    (audit SS3 item 15). No buffer-gas parameter exists by design (audit
    SS3 item 16)."""
    n, l, j = state
    dline = sp.d2 if line == "D2" else sp.d1
    if dline is None:
        raise IntegrityError(f"{sp.name} carries no {line} dataset")
    T = cell.T_cell_K
    T_bbr = cell.T_bbr_K if cell.T_bbr_K is not None else T
    mass_kg = sp.mass_u * AMU
    notes: list[str] = []
    if cell.T_bbr_K is not None and cell.T_bbr_K != T:
        notes.append(f"BBR environment at T_bbr = {T_bbr} K != T_cell = "
                     f"{T} K (spec 04 SS7.3)")

    if cell.N_ground_m3 is not None:
        n_g = cell.N_ground_m3
    else:
        n_g = number_density_m3(_element(sp), T,
                                isotope_fraction=sp.abundance)

    if gamma_r_method == "sum":
        table = decay_table(sp, n, l, j)
        g_rad = table.gamma0_rad_s
        g_bbr = table.bbr_rate(T_bbr)
    else:
        g_rad = 1.0 / float(radiative_lifetime_fit(sp, n, l, j))
        g_bbr = float(bbr_rate_fit(sp, n, l, j, T_bbr))
    w_ion = bbr_ionization_rate(sp, n, l, j, T_bbr) if l <= 2 else 0.0

    g_t = transit_rate(mass_kg, T, cell.beam_waist_m)
    g_ge_l, g_er_l, g_gr_l = laser_dephasing_rates(
        lasers.fwhm_probe_hz, lasers.fwhm_coupling_hz,
        lasers.noise_correlation)
    g_self = math.pi * float(self_broadening_fwhm_hz(
        sp, line, n_g, which=self_broadening))

    if _element(sp) == "Rb":
        shift_fermi = fermi_shift_hz(sp, n_g)
        # broadening ~ |shift| * 0.5, x2 uncertainty [UNVERIFIED, SS2.3.5(ii)]
        g_rydgnd = math.pi * 0.5 * abs(shift_fermi)
    else:
        if n_g > 1e19:
            raise IntegrityError(
                f"{sp.name} hot cell (N_g = {n_g:.2e} m^-3): the Fermi "
                "shift/broadening would be significant but no e^-/Cs(6S) "
                "scattering length is sourced — refusing (spec 04 SS3.5)")
        shift_fermi = 0.0
        g_rydgnd = 0.0
        notes.append("Fermi shift/broadening omitted: no sourced e^-/ground "
                     "scattering length for Cs; valid only at N_g <= 1e19 "
                     "m^-3 (spec 04 SS3.5)")

    inhomo: dict = {
        "doppler": {"sigma_v_m_s": math.sqrt(KB * T / mass_kg),
                    "entry": "velocity integral with k_p, k_c signs "
                             "(spec 04 SS2.4 row 12; NEVER a rate)"},
        "beam": {"w_m": cell.beam_waist_m,
                 "entry": "radial quadrature of Omega(r) (row 13)"},
    }
    if cell.N_rydberg_m3 > 0.0:
        if l == 0 and _element(sp) == "Rb":
            c6 = c6_over_h_rb_ns(n)
            dv = vdw_dephasing_hz(c6.value, cell.N_rydberg_m3)
            inhomo["vdw"] = {
                "delta_nu_hz": dv, "c6_over_h_hz_m6": c6.value,
                "source": c6.source,
                "entry": "nn-distance average (row 10; NEVER a rate)"}
        else:
            raise IntegrityError(
                f"N_rydberg > 0 but no sourced C6 for {sp.name} "
                f"(l={l}, j={j}) — the Singer fit covers Rb nS1/2 only; "
                "refusing to guess (spec 04 SS3.5 / audit R11)")
    if cell.E_stray_v_m > 0.0:
        if cell.alpha0_over_h_hz_v2m2 is None:
            raise IntegrityError(
                "E_stray > 0 requires alpha0_over_h_hz_v2m2 from the Stark "
                "engine (ruling R-5: the spec-04 '~6e2 MHz/(V/cm)^2' Rb 50S "
                "placeholder is WRONG by 12x and banned; the verified value "
                "is 50.5 MHz/(V/cm)^2 = 5050 Hz/(V/m)^2, spec 07). Pass the "
                "polarizability explicitly — refusing to guess")
        inhomo["stark"] = {
            "delta_nu_hz": stark_inhomogeneous_fwhm_hz(
                cell.alpha0_over_h_hz_v2m2, cell.E_stray_v_m),
            "e_rms_v_m": cell.E_stray_v_m,
            "alpha0_over_h_hz_v2m2": cell.alpha0_over_h_hz_v2m2,
            "entry": "P(E) average of -(1/2) alpha E^2 (row 11; NEVER a "
                     "rate)"}
    if cell.N_ion_m3 > 0.0:
        inhomo["ions"] = {
            "N_ion_m3": cell.N_ion_m3,
            "entry": "Holtsmark P(E) option (spec 04 SS2.3.6); keep "
                     "N_ion < 1e12 m^-3 (~1e6 cm^-3)"}

    return DephasingBudget(
        species=sp.name, state=state,
        Gamma_e=dline.gamma_rad_s,
        Gamma_r_rad=g_rad, Gamma_r_bbr=g_bbr, W_bbr_ion=w_ion,
        gamma_transit=g_t,
        gamma_laser_ge=g_ge_l, gamma_laser_er=g_er_l, gamma_laser_gr=g_gr_l,
        gamma_self_ge=g_self, gamma_rydgnd_gr=g_rydgnd,
        shift_fermi_hz=shift_fermi,
        Omega_p=Omega_p, Omega_c=Omega_c, N_ground_m3=n_g,
        inhomo=inhomo, notes=tuple(notes))


@dataclass(frozen=True)
class DefaultCell:
    """One spec 04 SS5.1 default parameter set (engineering choices,
    UNVERIFIED as 'typical' in any statistical sense; the resulting
    typical-cell EIT width must land in the 2-10 MHz band, benchmark B10).
    Omega_* are angular [rad/s]."""

    cell: CellParams
    lasers: LaserParams
    Omega_p: float
    Omega_c: float


def default_cell(kind: Literal["typical", "good"] = "typical") -> DefaultCell:
    """Spec 04 SS5.1 defaults. NOTE the SS5.1 'Stark inhom ~30 kHz' row is
    WRONG (inherited the banned 6e2 placeholder); with the verified spec-07
    alpha the typical-cell Stark term is ~2.5 kHz (ruling R-5). alpha is
    left None here — the caller supplies it from the Stark engine."""
    if kind == "typical":
        return DefaultCell(
            cell=CellParams(T_cell_K=300.0, beam_waist_m=0.75e-3,
                            E_stray_v_m=1.0),
            lasers=LaserParams(2.0e5, 2.0e5, 0.0),
            Omega_p=2.0 * math.pi * 1.0e6, Omega_c=2.0 * math.pi * 3.0e6)
    if kind == "good":
        return DefaultCell(
            cell=CellParams(T_cell_K=300.0, beam_waist_m=1.0e-3,
                            E_stray_v_m=0.2),
            lasers=LaserParams(1.0e4, 1.0e4, 0.0),
            Omega_p=2.0 * math.pi * 0.3e6, Omega_c=2.0 * math.pi * 1.0e6)
    raise ValueError(f"unknown default cell kind {kind!r}")
