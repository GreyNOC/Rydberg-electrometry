"""Full dipole matrix elements — the specs 02+03 integration layer.

Owns (docs/spec/00-conventions.md SS4 / SS7 step 5): the combination of
consensus radial integrals (``rydsim.radial``, spread-based uncertainty) with
angular factors (``rydsim.angular``) into physical transition dipole moments
in C*m, the NIST effective RF dipole p of lock #11 consumed by the
superhet/AT chain, and the D-line closure diagnostic of audit R5. The
Racah/Steck conversion is handled ONCE, here (via ``rydsim.angular``
converters) — no other module re-derives it.

Conventions (docs/spec/00-conventions.md):
* SI internally: every dipole this module returns is in C*m (lock #1); the
  ``value_ea0`` property converts to e*a0 for I/O only.
* Symbol collision rule: d = p = mu_RF are one physical quantity, code name
  ``dipole_Cm`` / ``mu_rf`` (symbol table row d/p/mu_RF).
* lock #11: p = e a0 R |A(l, j, m_j=1/2 -> l', j', 1/2; q=0)| (co-linear pi
  geometry). Non-collinear / elliptical RF needs the coherent multi-q sum of
  spec 03 SS2.3 — never a single p.
* Signs of radial integrals and reduced elements are convention-relative
  (spec 03 SS4.3.8): only |value| is comparable across codes.

Integrity (docs/spec/00-integrity-audit.md):
* R5: the Jing fixture mu_RF(Cs 47D5/2->48P3/2) = 1443.450 e a0 is
  LITERATURE-RECALL and is never used as an input — tests recompute it from
  scratch through this module and assert 2 % agreement.
* SS4 item 4: every returned dipole carries the minimum confidence class
  encountered on its computation path (``confidence_floor``) plus the
  per-method radial dict and spread (SS4 items 5, 7).
* SS4 item 7: every dipole is stamped with its convention.
* Refusals: E1-forbidden pairs raise ``rydsim.provenance.IntegrityError``
  from the effective-dipole path (a silent 0 would invert to an infinite
  field); |l1 - l2| != 1 raises everywhere (not a dipole pair).

Sources for numeric contracts (transcribed in tests with tags):
- Simons, Gordon & Holloway, J. Appl. Phys. 120, 123103 (2016) — NIST
  effective-dipole convention [VERIFIED, spec 03 SS2.5].
- Steck alkali datasheets rev 2.3.4 (8 Aug 2025) — D-line dipoles for the
  closure check [VERIFIED, spec 01/03].
- Spec 02 SS7.1 / B15 — the documented +5..10 % model-potential bias of
  low-n D-line dipoles (Rb 5S->5P3/2 computes ~5.57 a0 vs 5.18 a0
  experiment-derived) [computed VERIFIED; bias band LITERATURE-RECALL].
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Mapping

from .angular import (angular_factor, racah_to_steck, reduced_dipole_j)
from .atom import DLine, Species, rydberg_constant_hz
from .constants import AU_DIPOLE
from .provenance import IntegrityError
from .radial import (A4_L1_NOTE, RadialMEResult, radial_matrix_element,
                     radial_matrix_element_consensus, radial_wavefunction)

__all__ = [
    "DipoleResult", "DLineClosureResult",
    "full_dipole_matrix_element", "mu_rf", "dline_closure",
    "DLINE_BIAS_BAND", "MU_RF_CONVENTIONS",
]

# ---------------------------------------------------------------------------
# Confidence-taint machinery (audit SS4 item 4)
# ---------------------------------------------------------------------------

# Ordering per audit SS4 item 4: VERIFIED > VERIFIED-ARC > LITERATURE-RECALL
# > UNVERIFIED > MISSING-blocked. Lower rank index = higher confidence.
_CONF_ORDER = ("VERIFIED", "VERIFIED-ARC", "LITERATURE-RECALL",
               "UNVERIFIED", "MISSING")


def _conf_rank(tag: str) -> int:
    """Rank a free-form confidence tag by its leading class marker.

    'VERIFIED-ARC (digits); unc. LITERATURE-RECALL' classifies as
    VERIFIED-ARC: the digits (which the dipole value consumes) carry that
    class; the uncertainty qualifier does not degrade the value itself.
    Unrecognized tags rank as UNVERIFIED (refuse-to-flatter default).
    """
    t = tag.strip().upper()
    if t.startswith("VERIFIED-ARC"):
        return _CONF_ORDER.index("VERIFIED-ARC")
    for i, marker in enumerate(_CONF_ORDER):
        if t.startswith(marker):
            return i
    return _CONF_ORDER.index("UNVERIFIED")


def _confidence_floor(tags: list[str]) -> str:
    """Minimum confidence class over a computation path (audit SS4 item 4)."""
    if not tags:
        raise IntegrityError("confidence floor of an empty path (no sources "
                             "recorded) — no-fabrication rule")
    return _CONF_ORDER[max(_conf_rank(t) for t in tags)]


def _series_provenance(sp: Species, l: int, j: float) -> tuple[str, str]:
    """(source, confidence) of the quantum-defect series feeding n*(n, l, j).

    Mirrors the spec 01 lookup (l = 4 falls back across j; l >= 5 uses the
    core-polarization formula whose only sourced input is alpha_d
    [VERIFIED-ARC, Marinescu 1994 via ARC]).
    """
    if l >= 5:
        return (f"spec 01 Eq. (1.6) core-polarization defect with alpha_d = "
                f"{sp.alpha_core_au} a.u. (Marinescu 1994 via ARC)",
                "VERIFIED-ARC")
    row = sp.series.get((l, float(j)))
    if row is None and l == 4:
        for (ll, _), r in sp.series.items():
            if ll == 4:
                row = r
                break
    if row is None:
        raise IntegrityError(
            f"no sourced quantum-defect series for {sp.name} (l={l}, j={j}); "
            "refusing to guess (spec 01 SS3.4-3.5 / audit SS3 item 2)")
    return row.source, row.confidence


# MSD94 model-potential confidence for the radial layer (audit R20: LOW —
# both transcriptions diffed; the lone Rb a4(l=1) last-digit discrepancy is
# disclosed and bounded < 4e-8, see rydsim.radial.A4_L1_NOTE).
_MSD94_CONF = "VERIFIED"
_ANGULAR_SOURCE = ("spec 03 Eq. (2.9) Wigner-Eckart chain (exact algebra, "
                   "dual-implementation validated) [VERIFIED]")


# ---------------------------------------------------------------------------
# Result dataclasses (light 12-point-schema carrier, audit SS4)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DipoleResult:
    """A transition dipole moment with provenance (audit SS4 items 4, 5, 7).

    value_Cm [C*m]: e a0 R A for the stated convention; sign is
    convention-relative (spec 03 SS4.3.8) except where the convention takes
    magnitudes (mu_rf), in which case value_Cm >= 0.
    unc_Cm [C*m]: numerical uncertainty = |value| * radial_spread_rel — the
    cross-method radial spread IS the shipped uncertainty (audit SS4 item 5).
    radial_a0 [a0]: signed Method-A consensus radial integral.
    radial_methods: per-method radial dict [a0] (model_potential / coulomb /
    kaulakys), verbatim from ``rydsim.radial``.
    angular_factor: signed dimensionless A (or the rms A for the manifold
    convention).
    convention: dipole convention stamp, one of MU_RF_CONVENTIONS or
    'racah_mj_resolved' (audit SS4 item 7).
    confidence_floor: minimum confidence class on the path (audit SS4 item 4).
    """

    value_Cm: float
    unc_Cm: float
    radial_a0: float
    radial_spread_rel: float
    radial_methods: Mapping[str, float]
    angular_factor: float
    convention: str
    q: int | None
    mj: float | None
    mjp: float | None
    species: str
    state1: tuple[int, int, float]
    state2: tuple[int, int, float]
    confidence_floor: str
    sources: tuple[str, ...]

    @property
    def value_ea0(self) -> float:
        """Dipole in atomic units e*a0 (I/O convenience, lock #1)."""
        return self.value_Cm / AU_DIPOLE

    @property
    def unc_ea0(self) -> float:
        """Uncertainty in e*a0."""
        return self.unc_Cm / AU_DIPOLE


@dataclass(frozen=True)
class DLineClosureResult:
    """D-line closure diagnostic (audit R5 self-check pattern; spec 02 B15).

    Recomputes the D-line reduced dipole from model-potential radial x
    angular (with nu from MEASURED term energies, since Ritz defects are
    invalid at n = 5/6) and exposes BOTH numbers next to the documented
    model-potential bias band — never claiming exact agreement.

    radial_a0 [a0]: Method-A radial integral (measured-energy nu).
    d_racah_computed_ea0 / d_steck_computed_ea0 [e*a0]: computed reduced
    dipole in both conventions (spec 03 Eq. 2.3 conversion applied here).
    d_steck_measured_ea0 [e*a0]: Steck rev 2.3.4 datasheet value [VERIFIED].
    bias_rel: computed/measured - 1; expected inside ``band`` (default
    +5..10 %, spec 02 SS7.1 — the KNOWN low-n model-potential bias).
    """

    species: str
    line: str
    nu_lower: float
    nu_upper: float
    radial_a0: float
    d_racah_computed_ea0: float
    d_steck_computed_ea0: float
    d_steck_measured_ea0: float
    bias_rel: float
    band: tuple[float, float]
    in_band: bool
    sources: tuple[str, ...]


# Documented low-n model-potential bias band for D-line dipoles: computed
# comes out +5..10 % ABOVE experiment (Rb 5S->5P3/2: +7.6 %; spec 02 SS7.1 /
# B15, audit R5 pattern). Band edges LITERATURE-RECALL (spec 02 text); the
# bias itself is re-measured by dline_closure every run.
DLINE_BIAS_BAND: tuple[float, float] = (0.05, 0.10)


# ---------------------------------------------------------------------------
# Core assembly helpers
# ---------------------------------------------------------------------------

def _dipole_pair_radial(sp: Species, state1: tuple[int, int, float],
                        state2: tuple[int, int, float]) -> RadialMEResult:
    """Consensus radial integral for an E1 pair [a0]; |l1-l2| = 1 enforced."""
    _, l1, _ = state1
    _, l2, _ = state2
    if abs(l1 - l2) != 1:
        raise IntegrityError(
            f"(l1={l1}, l2={l2}): |l1 - l2| != 1 is not an E1 dipole pair — "
            "refusing (spec 03 SS2.4)")
    return radial_matrix_element_consensus(sp, state1, state2, k=1)


def _path_provenance(sp: Species, state1: tuple[int, int, float],
                     state2: tuple[int, int, float]) -> tuple[tuple[str, ...], str]:
    """(sources, confidence_floor) for the radial x angular path."""
    tags: list[str] = []
    sources: list[str] = []
    for (_, l, j) in (state1, state2):
        src, conf = _series_provenance(sp, l, j)
        sources.append(f"quantum defects (l={l}, j={j}): {src} [{conf}]")
        tags.append(conf)
    sources.append(
        "MSD94 model potential (PRA 49, 982 Table I, dual-transcription "
        f"diffed) [{_MSD94_CONF}]")
    tags.append(_MSD94_CONF)
    if sp.name.startswith("Rb") and 1 in (state1[1], state2[1]):
        sources.append(A4_L1_NOTE)
    sources.append(_ANGULAR_SOURCE)
    tags.append("VERIFIED")
    return tuple(sources), _confidence_floor(tags)


def full_dipole_matrix_element(sp: Species, state1: tuple[int, int, float],
                               state2: tuple[int, int, float],
                               q: int, mj: float,
                               mjp: float | None = None) -> DipoleResult:
    """Full <state1 mj| e r_q |state2 mjp> [C*m] (ownership map SS7 step 5).

    d = e * a0 * R_consensus * A(l1, j1, mj; l2, j2, mjp; q), with R from
    ``rydsim.radial.radial_matrix_element_consensus`` (spread-based
    uncertainty) and A from ``rydsim.angular.angular_factor`` (spec 03
    Eq. 2.9). states are (n, l, j); mjp defaults to mj - q (the Racah 3j
    m-condition m = m' + q, spec 03 SS2.2). q in {-1, 0, +1} = sigma-/pi/
    sigma+. Signed value; sign convention-relative (spec 03 SS4.3.8).

    Returns exact 0.0 (with full provenance) when only the (m, q) selection
    rules forbid the element — zeros are physical entries of a coupling
    matrix. Raises IntegrityError when |l1 - l2| != 1 (not an E1 pair) or
    when the radial consensus fails (rydsim.radial refusal rules).
    """
    if mjp is None:
        mjp = mj - q
    _, l1, j1 = state1
    _, l2, j2 = state2
    rad = _dipole_pair_radial(sp, state1, state2)
    A = angular_factor(l1, j1, mj, l2, j2, mjp, q)
    value = AU_DIPOLE * rad.value * A
    sources, floor = _path_provenance(sp, state1, state2)
    return DipoleResult(
        value_Cm=value, unc_Cm=abs(value) * rad.spread_rel,
        radial_a0=rad.value, radial_spread_rel=rad.spread_rel,
        radial_methods=dict(rad.per_method), angular_factor=A,
        convention="racah_mj_resolved", q=q, mj=mj, mjp=mjp,
        species=sp.name, state1=state1, state2=state2,
        confidence_floor=floor, sources=sources)


# ---------------------------------------------------------------------------
# Effective RF dipole mu_RF (lock #11) — what the superhet/AT chain consumes
# ---------------------------------------------------------------------------

#: Supported effective-RF-dipole conventions (audit SS4 item 7 stamp values).
#: - 'nist_pi'  (DEFAULT, lock #11): co-linear pi geometry, m_j = +-1/2,
#:   p = e a0 R |A(l,j,1/2 -> l',j',1/2; q=0)| — Simons/Gordon/Holloway 2016.
#: - 'stretched': stretched fine-structure pair m_j = j -> m_j' = j',
#:   q = j - j' (the sigma+-ladder cold-atom convention; Tu 2024 config —
#:   their 5S1/2 F=2 mF=2 ground + sigma+ chain populates m_j = j).
#: - 'pi_manifold_rms': sqrt(mean over populated m_j of A(m_j, q=0)^2) —
#:   the rms pi-manifold dipole over |m_j| <= min(j, j'). For D5/2->P3/2
#:   this is R/sqrt(5); it reproduces Sedlacek 2012's printed effective
#:   4-level mu_RF = 1.37e-26 C*m to 0.3 % (their own "stretched hyperfine
#:   states" description does NOT — documented convention tension,
#:   spec 09 C5e / SS4.9).
MU_RF_CONVENTIONS = ("nist_pi", "stretched", "pi_manifold_rms")


def _rf_angular(l1: int, j1: float, l2: int, j2: float,
                convention: str) -> tuple[float, int | None, float | None,
                                          float | None]:
    """(|A| or rms A, q, mj, mjp) for one mu_RF convention. Dimensionless."""
    if convention == "nist_pi":
        A = angular_factor(l1, j1, 0.5, l2, j2, 0.5, 0)
        return abs(A), 0, 0.5, 0.5
    if convention == "stretched":
        q = round(j1 - j2)
        if q not in (-1, 0, 1):
            raise IntegrityError(
                f"stretched convention needs |j1 - j2| <= 1, got j1={j1}, "
                f"j2={j2} (spec 03 SS2.4)")
        A = angular_factor(l1, j1, j1, l2, j2, j2, q)
        return abs(A), q, j1, j2
    if convention == "pi_manifold_rms":
        j_min = min(j1, j2)
        mjs = [0.5 + k for k in range(int(j_min + 0.5))]
        sq = [angular_factor(l1, j1, m, l2, j2, m, 0) ** 2 for m in mjs]
        return math.sqrt(sum(sq) / len(sq)), 0, None, None
    raise ValueError(
        f"unknown mu_RF convention {convention!r}; supported: "
        f"{MU_RF_CONVENTIONS}")


def mu_rf(sp: Species, state1: tuple[int, int, float],
          state2: tuple[int, int, float], *,
          convention: str = "nist_pi") -> DipoleResult:
    """Effective RF dipole p = mu_RF [C*m] for a Rydberg pair (lock #11).

    The number the superhet/AT chain consumes: hbar Omega_RF = mu_rf * E_RF
    with E_RF the peak amplitude (locks #3/#4). Default convention
    'nist_pi' is normative (spec 00 lock #11 / spec 03 SS2.5): co-linear pi
    geometry, m_j = +-1/2, p = e a0 |R| |A(l,j,1/2 -> l',j',1/2; q=0)|.
    See MU_RF_CONVENTIONS for the two literature-matching alternates; the
    convention is stamped on the result (audit SS4 item 7). value_Cm >= 0
    (magnitude convention). Uncertainty = |value| * radial spread (audit SS4
    item 5).

    Raises IntegrityError for an E1-forbidden pair rather than returning a
    zero dipole that would silently invert to an infinite field (spec 03
    SS2.4-2.5), and propagates every rydsim.radial consensus refusal.
    """
    _, l1, j1 = state1
    _, l2, j2 = state2
    if abs(l1 - l2) != 1:
        raise IntegrityError(
            f"mu_rf: (l1={l1}, l2={l2}) is not an E1 pair — refusing "
            "(spec 03 SS2.4)")
    A_abs, q, mj, mjp = _rf_angular(l1, j1, l2, j2, convention)
    if A_abs == 0.0:
        raise IntegrityError(
            f"mu_rf: transition {sp.name} {state1} -> {state2} is "
            f"E1-forbidden under convention {convention!r}; refusing to "
            "return a zero effective dipole (spec 03 SS2.4-2.5)")
    rad = _dipole_pair_radial(sp, state1, state2)
    value = AU_DIPOLE * abs(rad.value) * A_abs
    sources, floor = _path_provenance(sp, state1, state2)
    return DipoleResult(
        value_Cm=value, unc_Cm=value * rad.spread_rel,
        radial_a0=rad.value, radial_spread_rel=rad.spread_rel,
        radial_methods=dict(rad.per_method), angular_factor=A_abs,
        convention=convention, q=q, mj=mj, mjp=mjp,
        species=sp.name, state1=state1, state2=state2,
        confidence_floor=floor, sources=sources)


# ---------------------------------------------------------------------------
# D-line closure (audit R5 self-check pattern; spec 02 B15 / spec 03 SS2.7)
# ---------------------------------------------------------------------------

def _measured_nu(sp: Species, binding_hz: float, l: int) -> float:
    """Effective quantum number nu = sqrt(c R_M / E_b) from a MEASURED
    binding energy [dimensionless]. The low-n path: Ritz defects are invalid
    below the hard floor, so the D-line closure feeds Numerov with nu from
    Steck term energies (spec 02 SS7.1 / B15)."""
    if binding_hz <= 0:
        raise IntegrityError(
            f"non-positive binding energy {binding_hz} Hz: state above the "
            "ionization limit — no bound orbital")
    nu = math.sqrt(rydberg_constant_hz(sp) / binding_hz)
    if nu <= l:
        raise IntegrityError(f"measured nu = {nu:.4f} <= l = {l}: no bound "
                             "QDT orbital (spec 02 SS5)")
    return nu


def dline_closure(sp: Species, line: Literal["D1", "D2"] = "D2", *,
                  require_band: bool = True,
                  band: tuple[float, float] = DLINE_BIAS_BAND,
                  ) -> DLineClosureResult:
    """Recompute a D-line reduced dipole from radial x angular and compare
    against Steck's measured value (audit R5 self-check pattern).

    Method: nu(ground) and nu(upper) from MEASURED energies (E_I centroid
    and E_I - nu0(line), spec 01 data [VERIFIED]); model-potential Numerov
    radial integral R [a0] on the shared grid; reduced dipole
    |<S1/2||er||Pj'>|_Racah = |R| * |reduced_dipole_j(0, 1/2, 1, j')| e a0
    (spec 03 Eq. 2.5-2.8), converted to the Steck convention ONCE here via
    spec 03 Eq. (2.3). The comparison asserts the DOCUMENTED model-potential
    bias band (default +5..10 % above experiment, spec 02 SS7.1: model
    potentials do not guarantee dipole accuracy where core overlap matters)
    — NOT exact agreement — and exposes both numbers.

    require_band=True (default): bias outside ``band`` raises IntegrityError
    (the radial machinery or the datasheet transcription has regressed —
    spec 02 B15 is a bias-STABILITY regression). This is a diagnostic of the
    radial engine; production D-line dipoles come from experiment
    (spec 02 SS7.1, docs 03/04).
    """
    dline: DLine | None = sp.d2 if line == "D2" else sp.d1
    if dline is None:
        raise IntegrityError(f"{sp.name} carries no {line} dataset")
    j_up = 1.5 if line == "D2" else 0.5
    nu_g = _measured_nu(sp, sp.e_ion_hz, 0)
    nu_e = _measured_nu(sp, sp.e_ion_hz - dline.nu0_hz, 1)
    n0 = sp.ground_n
    r_outer = 2.0 * n0 * (n0 + 15.0)
    a = radial_wavefunction(sp, n0, 0, 0.5, nu=nu_g, r_outer=r_outer,
                            method="model_potential")
    b = radial_wavefunction(sp, n0, 1, j_up, nu=nu_e, r_outer=r_outer,
                            method="model_potential")
    R = radial_matrix_element(a, b, 1)
    d_racah = abs(R) * abs(reduced_dipole_j(0, 0.5, 1, j_up))     # e a0
    d_steck = racah_to_steck(d_racah, 0.5)                        # e a0
    bias = d_steck / dline.d_reduced_ea0 - 1.0
    in_band = band[0] <= bias <= band[1]
    result = DLineClosureResult(
        species=sp.name, line=line, nu_lower=nu_g, nu_upper=nu_e,
        radial_a0=R, d_racah_computed_ea0=d_racah,
        d_steck_computed_ea0=d_steck,
        d_steck_measured_ea0=dline.d_reduced_ea0,
        bias_rel=bias, band=band, in_band=in_band,
        sources=(
            f"measured term energies: {sp.name} E_I + {line} centroid "
            "(Steck rev 2.3.4 / spec 01 tables) [VERIFIED]",
            "MSD94 model potential, single-method Numerov (diagnostic path; "
            "consensus entry refuses n below the Ritz hard floor) "
            f"[{_MSD94_CONF}]",
            "bias band +5..10 %: spec 02 SS7.1/B15 (Rb 5S->5P3/2 computes "
            "+7.6 % above the Steck-derived 5.18 a0) [LITERATURE-RECALL]",
            _ANGULAR_SOURCE,
        ))
    if require_band and not in_band:
        raise IntegrityError(
            f"D-line closure bias out of band for {sp.name} {line}: computed "
            f"{d_steck:.4f} e a0 vs measured {dline.d_reduced_ea0} e a0 "
            f"(bias {bias:+.2%}, expected {band[0]:+.0%}..{band[1]:+.0%}) — "
            "radial engine or datasheet transcription regressed "
            "(spec 02 B15 / audit R5)")
    return result
