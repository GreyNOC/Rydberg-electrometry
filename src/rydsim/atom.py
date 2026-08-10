"""Atomic structure and energy levels for Rb-85, Rb-87, Cs-133 (spec 01).

Owns (docs/spec/00-conventions.md §4): quantum defects, energies, n*, E_I,
D-line + hyperfine data, coupling wavelengths lambda_c. All frequencies are
plain Hz (energy/h) at this module's API — angular quantities do not appear
here; downstream rate modules apply the single 2*pi conversion at their own
boundary (lock #2).

Normative sources (docs/spec/01-atomic-structure.md §3, all transcribed with
per-row source + confidence tags below):
  - Steck alkali D-line datasheets, revision 2.3.4 (2025-08-08) [VERIFIED]
  - Mack et al., PRA 83, 052515 (2011) — Rb-87 E_I + defects [VERIFIED]
  - Sanguinetti et al., J. Phys. B 42, 165004 (2009) — Rb-85 E_I [VERIFIED]
  - Deiglmayr et al., PRA 93, 013424 (2016) — Cs E_I + S/P/D5/2 defects [VERIFIED]
  - Li et al., PRA 67, 052502 (2003); Han et al., PRA 74, 054502 (2006) —
    Rb-85 defects [VERIFIED / VERIFIED-ARC digits]
  - Bai et al., PRA 108, 022804 (2023) — Cs nF defects [VERIFIED]
  - Lorenzen & Niemax 1984, Weber & Sansonetti 1987 — Cs D3/2, G7/2
    [VERIFIED-ARC digits only; uncertainties MISSING]
  - Marinescu et al., PRA 49, 982 (1994) — core polarizabilities [VERIFIED-ARC]

Ruling compliance:
  R-10: single reduced-mass convention R_M = R_inf/(1 + m_e/M_atom).
  R-11: E_I(centroid) = E_I(from F) + dE_hfs(F); the tabulated e_ion_hz values
        below are centroid-referenced already.
  R-12: c*R_inf check value is GHz-scale (3 289 841.960 GHz), not THz.
  R-14: Cs nS1/2 / nD5/2 use Form B (fixed-point) — Form A on the same
        coefficients errs 49.2 MHz at Cs 12S (benchmark AS-11).
  R-15: lambda_c is computed from these energies at runtime, never hard-coded.

Integrity-audit compliance (docs/spec/00-integrity-audit.md §3, items 1-5):
  1. Below the per-species hard floor (Rb n=8, Cs n=12) energies RAISE
     rydsim.provenance.IntegrityError; below the MHz-grade floor (19/25) a
     warning is attached.
  2. Low-lying intermediate levels (Rb 5D/6P, Cs 7P, ...) are MISSING —
     unreachable through Eq. (1.1) because of the hard floor; they must be
     added as NIST-ASD data rows, never synthesized.
  3. Rydberg hyperfine A for unsourced series returns exactly 0 with the
     documented bound; sourced coefficients carry uncertainty tags.
  4. The Cs E_I erratum value (PRA 112, 049902(E) 2025, +0.47 MHz offset) is
     NOT encoded anywhere in this module; only the 2016 Deiglmayr value ships.
     The offset is declared as a systematic via e_ion_systematic_hz.
  5. Rb-85 absolute optical frequencies carry a +-40 MHz E_I systematic
     (Lee-1978 vs Sanguinetti-2009 6-sigma tension) via e_ion_systematic_hz.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Literal, Mapping

import numpy as np

from .constants import AMU, C, M_E, RYD_INF
from .provenance import Confidence, IntegrityError, SourcedValue

__all__ = [
    "RitzForm", "RitzSeries", "HyperfineConstants", "DLine", "Species",
    "RB85", "RB87", "CS133", "SPECIES",
    "rydberg_constant_hz", "quantum_defect", "polarization_defect",
    "ritz_form_a", "ritz_form_b", "n_star", "binding_energy_hz", "energy_hz",
    "transition_hz", "rf_transition_hz", "hyperfine_shift_hz",
    "probe_transition_hz", "coupling_laser_hz", "coupling_wavelength_m",
    "rydberg_hfs_A_hz", "rydberg_hfs_coefficient", "RYDBERG_HFS_COEFF",
    "RB87_E_ION_FROM_5S12_F1", "RB87_E_ION_FROM_5P32_F3",
]

RitzForm = Literal["modified", "iterated"]  # Form A / Form B of spec 01 §2.2

# 1 cm^-1 in Hz, derived (spec 01 §2.1: 1 cm^-1 = 29 979.2458 MHz exactly).
INV_CM_HZ = C * 100.0


@dataclass(frozen=True)
class RitzSeries:
    """Quantum-defect Ritz coefficients for one (l, j) series (spec 01 §2.2).

    form='modified' -> Form A, Eq. (1.3): delta = d0 + d2/(n-d0)^2 + ...
    form='iterated' -> Form B, Eq. (1.4): fixed-point in delta(n).
    n_min_mhz / n_min_hard: validity floors per spec 01 §4.2 (warn / raise).
    source, confidence: mandatory provenance strings (no-fabrication rule).
    """

    l: int
    j: float
    d0: float
    d2: float = 0.0
    d4: float = 0.0
    d6: float = 0.0
    d8: float = 0.0
    form: RitzForm = "modified"
    n_min_mhz: int = 19
    n_min_hard: int = 8
    source: str = ""
    confidence: str = ""

    def __post_init__(self) -> None:
        if not self.source or not self.confidence:
            raise ValueError("RitzSeries requires source and confidence tags")


@dataclass(frozen=True)
class HyperfineConstants:
    """A, B, C in Hz (energy/h) for one fine-structure level; spec 01 Eq. (1.7).

    The magnetic-octupole C is stored for provenance only (Cs 6P3/2:
    0.560(70) kHz) and does not enter hyperfine_shift_hz (negligible).
    """

    A_hz: float
    B_hz: float = 0.0
    C_hz: float = 0.0
    source: str = ""


@dataclass(frozen=True)
class DLine:
    """One D-line dataset (Steck rev 2.3.4, spec 01 §3.6). SI, vacuum, Hz.

    gamma_rad_s is the population decay rate 1/tau = Einstein A [1/s], equal
    to the angular FWHM natural linewidth (lock #7); I_sat rows in W/m^2
    (Steck mW/cm^2 x 10), None where Steck prints no value.
    """

    name: Literal["D1", "D2"]
    nu0_hz: float                  # hyperfine-centroid transition frequency
    nu0_unc_hz: float
    lambda_vac_m: float
    tau_s: float
    gamma_rad_s: float
    f_osc: float
    d_reduced_ea0: float           # <J||er||J'> (Steck convention)
    isat_cycling_w_m2: float | None
    isat_iso_w_m2: float | None
    isat_detuned_pi_w_m2: float
    upper_hfs: HyperfineConstants  # A, B(, C) of the 5P/6P upper level
    source: str = "Steck rev 2.3.4 (2025-08-08) [VERIFIED]"


@dataclass(frozen=True)
class Species:
    """Immutable per-isotope data bundle (spec 01 §5). Instances: RB85, RB87, CS133.

    e_ion_hz is E_I/h from the ground hyperfine CENTROID (ruling R-11);
    e_ion_systematic_hz is the declared systematic on ABSOLUTE optical
    frequencies (Rb-85: 40 MHz Lee/Sanguinetti tension; Cs: 0.47 MHz erratum
    offset, not applied) — it cancels in all Rydberg-Rydberg intervals.
    """

    name: str
    mass_u: float
    nuclear_spin: float
    abundance: float
    e_ion_hz: float
    e_ion_unc_hz: float
    ground_n: int
    alpha_core_au: float           # static dipole polarizability of the ion core
    series: Mapping[tuple[int, float], RitzSeries]
    ground_hfs: HyperfineConstants | None = None
    d1: DLine | None = None
    d2: DLine | None = None
    e_ion_systematic_hz: float = 0.0
    e_ion_note: str = ""
    mass_source: str = ""


# ---------------------------------------------------------------------------
# Quantum-defect tables (spec 01 §3.4-3.5) — every row carries source + tag
# ---------------------------------------------------------------------------

_RB_NMIN = dict(n_min_mhz=19, n_min_hard=8)    # spec 01 §4.2 (Mack/ARC)
_CS_NMIN = dict(n_min_mhz=25, n_min_hard=12)   # spec 01 §4.2 (Deiglmayr/ARC)

_RB85_SERIES: dict[tuple[int, float], RitzSeries] = {
    (0, 0.5): RitzSeries(0, 0.5, 3.1311804, 0.1784, **_RB_NMIN,
                         source="Li et al., PRA 67, 052502 (2003), via Mack Table I",
                         confidence="VERIFIED"),
    (1, 0.5): RitzSeries(1, 0.5, 2.6548849, 0.2900, **_RB_NMIN,
                         source="Li et al. 2003",
                         confidence="VERIFIED-ARC (digits); unc. LITERATURE-RECALL"),
    (1, 1.5): RitzSeries(1, 1.5, 2.6416737, 0.2950, **_RB_NMIN,
                         source="Li et al. 2003",
                         confidence="VERIFIED-ARC (digits); unc. LITERATURE-RECALL"),
    (2, 1.5): RitzSeries(2, 1.5, 1.3480917, -0.6029, **_RB_NMIN,
                         source="Li et al. 2003, via Mack Table I",
                         confidence="VERIFIED"),
    (2, 2.5): RitzSeries(2, 2.5, 1.3464657, -0.5960, **_RB_NMIN,
                         source="Li et al. 2003, via Mack Table I",
                         confidence="VERIFIED"),
    (3, 2.5): RitzSeries(3, 2.5, 0.0165192, -0.085, **_RB_NMIN,
                         source="Han et al., PRA 74, 054502 (2006)",
                         confidence="VERIFIED-ARC (digits); unc. LITERATURE-RECALL"),
    (3, 3.5): RitzSeries(3, 3.5, 0.0165437, -0.086, **_RB_NMIN,
                         source="Han et al. 2006",
                         confidence="VERIFIED-ARC (digits); unc. LITERATURE-RECALL"),
    # G-series j-splitting unresolved in the adopted data (spec 01 §7.6):
    # one coefficient set stored under both j keys.
    (4, 3.5): RitzSeries(4, 3.5, 0.0039990, -0.0202, **_RB_NMIN,
                         source="Raithel-group 2020 as adopted by ARC",
                         confidence="VERIFIED-ARC only"),
    (4, 4.5): RitzSeries(4, 4.5, 0.0039990, -0.0202, **_RB_NMIN,
                         source="Raithel-group 2020 as adopted by ARC",
                         confidence="VERIFIED-ARC only"),
}

_RB87_SERIES: dict[tuple[int, float], RitzSeries] = {
    # Isotope-specific replacements (Mack 2011 Table I); all other series
    # fall back to the Rb-85 rows (residual isotope dependence of delta is
    # < 3e-6, i.e. <= 0.2 MHz at n = 50 — spec 01 §3.4).
    (0, 0.5): RitzSeries(0, 0.5, 3.1311807, 0.1787, **_RB_NMIN,
                         source="Mack et al., PRA 83, 052515 (2011), Table I",
                         confidence="VERIFIED"),
    (2, 1.5): RitzSeries(2, 1.5, 1.3480948, -0.6054, **_RB_NMIN,
                         source="Mack 2011 Table I",
                         confidence="VERIFIED"),
    (2, 2.5): RitzSeries(2, 2.5, 1.3464622, -0.5940, **_RB_NMIN,
                         source="Mack 2011 Table I",
                         confidence="VERIFIED"),
    (4, 3.5): RitzSeries(4, 3.5, 0.00405, 0.0, **_RB_NMIN,
                         source="Afrousheh et al. 2006 as adopted by ARC",
                         confidence="VERIFIED-ARC only"),
    (4, 4.5): RitzSeries(4, 4.5, 0.00405, 0.0, **_RB_NMIN,
                         source="Afrousheh et al. 2006 as adopted by ARC",
                         confidence="VERIFIED-ARC only"),
    (1, 0.5): RitzSeries(1, 0.5, 2.6548849, 0.2900, **_RB_NMIN,
                         source="Li et al. 2003 (Rb-85 value; isotope dep. < 3e-6)",
                         confidence="VERIFIED-ARC (digits)"),
    (1, 1.5): RitzSeries(1, 1.5, 2.6416737, 0.2950, **_RB_NMIN,
                         source="Li et al. 2003 (Rb-85 value; isotope dep. < 3e-6)",
                         confidence="VERIFIED-ARC (digits)"),
    (3, 2.5): RitzSeries(3, 2.5, 0.0165192, -0.085, **_RB_NMIN,
                         source="Han et al. 2006 (Rb-85 value; isotope dep. < 3e-6)",
                         confidence="VERIFIED-ARC (digits)"),
    (3, 3.5): RitzSeries(3, 3.5, 0.0165437, -0.086, **_RB_NMIN,
                         source="Han et al. 2006 (Rb-85 value; isotope dep. < 3e-6)",
                         confidence="VERIFIED-ARC (digits)"),
}

_CS133_SERIES: dict[tuple[int, float], RitzSeries] = {
    # Ruling R-14: nS1/2 and nD5/2 are Form B (Deiglmayr Table IV) — Form A
    # on the same coefficients errs 49.2 MHz at 12S (benchmark AS-11).
    (0, 0.5): RitzSeries(0, 0.5, 4.0493532, 0.2391, 0.06, 11.0, -209.0,
                         form="iterated", **_CS_NMIN,
                         source="Deiglmayr et al., PRA 93, 013424 (2016), Table IV",
                         confidence="VERIFIED"),
    (1, 0.5): RitzSeries(1, 0.5, 3.5915871, 0.36273, form="modified", **_CS_NMIN,
                         source="Deiglmayr 2016 Table II (n = 27-74)",
                         confidence="VERIFIED"),
    (1, 1.5): RitzSeries(1, 1.5, 3.5590676, 0.37469, form="modified", **_CS_NMIN,
                         source="Deiglmayr 2016 Table II",
                         confidence="VERIFIED"),
    (2, 1.5): RitzSeries(2, 1.5, 2.4754562, 0.009320, -0.43498, -0.76358,
                         -18.0061, form="iterated", **_CS_NMIN,
                         source="Lorenzen & Niemax, Z. Phys. A 315, 127 (1984)",
                         confidence="VERIFIED-ARC (digits); uncertainties MISSING"),
    (2, 2.5): RitzSeries(2, 2.5, 2.4663144, 0.01381, -0.392, -1.9,
                         form="iterated", **_CS_NMIN,
                         source="Deiglmayr 2016 Table IV",
                         confidence="VERIFIED"),
    (3, 2.5): RitzSeries(3, 2.5, 0.03341537, -0.2014, form="iterated", **_CS_NMIN,
                         source="Bai et al., PRA 108, 022804 (2023)",
                         confidence="VERIFIED"),
    (3, 3.5): RitzSeries(3, 3.5, 0.0335646, -0.2052, form="iterated", **_CS_NMIN,
                         source="Bai et al. 2023 (NOT ARC's F7/2 = F5/2 shortcut, "
                                "which errs ~9 MHz at n = 47 — spec 01 §3.5)",
                         confidence="VERIFIED"),
    (4, 3.5): RitzSeries(4, 3.5, 0.00703865, -0.049252, 0.01291,
                         form="iterated", **_CS_NMIN,
                         source="Weber & Sansonetti, PRA 35, 4650 (1987)",
                         confidence="VERIFIED-ARC (digits); uncertainties MISSING"),
}


# ---------------------------------------------------------------------------
# Hyperfine constants (Steck rev 2.3.4, spec 01 §3.7 — all VERIFIED)
# ---------------------------------------------------------------------------

_RB85_GROUND_HFS = HyperfineConstants(
    A_hz=1.011_910_813_0e9, source="Steck Rb-85 rev 2.3.4 [VERIFIED]")
_RB87_GROUND_HFS = HyperfineConstants(
    A_hz=3.417_341_305_452_145e9, source="Steck Rb-87 rev 2.3.4 [VERIFIED]")
_CS_GROUND_HFS = HyperfineConstants(
    A_hz=2.298_157_942_5e9,
    source="Steck Cs rev 2.3.4 [VERIFIED; exact — defines the SI second]")

_RB85_5P12_HFS = HyperfineConstants(
    A_hz=120.527e6, source="Steck Rb-85 rev 2.3.4 [VERIFIED]")
_RB85_5P32_HFS = HyperfineConstants(
    A_hz=25.0354e6, B_hz=25.898e6, source="Steck Rb-85 rev 2.3.4 [VERIFIED]")
_RB87_5P12_HFS = HyperfineConstants(
    A_hz=407.25e6, source="Steck Rb-87 rev 2.3.4 [VERIFIED]")
_RB87_5P32_HFS = HyperfineConstants(
    A_hz=84.7185e6, B_hz=12.4965e6, source="Steck Rb-87 rev 2.3.4 [VERIFIED]")
_CS_6P12_HFS = HyperfineConstants(
    A_hz=291.9201e6, source="Steck Cs rev 2.3.4 [VERIFIED]")
_CS_6P32_HFS = HyperfineConstants(
    A_hz=50.28827e6, B_hz=-0.4934e6, C_hz=560.0,
    source="Steck Cs rev 2.3.4 [VERIFIED; C = 0.560(70) kHz, provenance only]")


# ---------------------------------------------------------------------------
# D-line data (Steck rev 2.3.4, spec 01 §3.6 — every row VERIFIED from PDFs)
# I_sat: Steck mW/cm^2 x 10 -> W/m^2 (conversion exact).
# ---------------------------------------------------------------------------

_RB85_D2 = DLine(
    name="D2", nu0_hz=384.230_406_373e12, nu0_unc_hz=14e3,
    lambda_vac_m=780.241_368_271e-9, tau_s=26.2348e-9, gamma_rad_s=38.117e6,
    f_osc=0.69577, d_reduced_ea0=4.22753,
    isat_cycling_w_m2=16.6932, isat_iso_w_m2=38.951,
    isat_detuned_pi_w_m2=25.0399, upper_hfs=_RB85_5P32_HFS)
_RB85_D1 = DLine(
    name="D1", nu0_hz=377.107_385_690e12, nu0_unc_hz=46e3,
    lambda_vac_m=794.979_014_933e-9, tau_s=27.679e-9, gamma_rad_s=36.129e6,
    f_osc=0.34231, d_reduced_ea0=2.9931,
    isat_cycling_w_m2=None, isat_iso_w_m2=None,
    isat_detuned_pi_w_m2=44.876, upper_hfs=_RB85_5P12_HFS)
_RB87_D2 = DLine(
    name="D2", nu0_hz=384.230_484_468_5e12, nu0_unc_hz=6.2e3,
    lambda_vac_m=780.241_209_686e-9, tau_s=26.2348e-9, gamma_rad_s=38.117e6,
    f_osc=0.69577, d_reduced_ea0=4.22752,
    isat_cycling_w_m2=16.6933, isat_iso_w_m2=35.771,
    isat_detuned_pi_w_m2=25.0399, upper_hfs=_RB87_5P32_HFS)
_RB87_D1 = DLine(
    name="D1", nu0_hz=377.107_463_380e12, nu0_unc_hz=11e3,
    lambda_vac_m=794.978_851_156e-9, tau_s=27.679e-9, gamma_rad_s=36.129e6,
    f_osc=0.34231, d_reduced_ea0=2.9931,
    isat_cycling_w_m2=None, isat_iso_w_m2=None,
    isat_detuned_pi_w_m2=44.876, upper_hfs=_RB87_5P12_HFS)
_CS_D2 = DLine(
    name="D2", nu0_hz=351.725_718_50e12, nu0_unc_hz=110e3,
    lambda_vac_m=852.347_275_82e-9, tau_s=30.405e-9, gamma_rad_s=32.889e6,
    f_osc=0.7164, d_reduced_ea0=4.4837,
    isat_cycling_w_m2=11.049, isat_iso_w_m2=27.119,
    isat_detuned_pi_w_m2=16.573, upper_hfs=_CS_6P32_HFS)
_CS_D1 = DLine(
    name="D1", nu0_hz=335.116_048_807e12, nu0_unc_hz=41e3,
    lambda_vac_m=894.592_959_86e-9, tau_s=34.791e-9, gamma_rad_s=28.743e6,
    f_osc=0.34486, d_reduced_ea0=3.1869,
    isat_cycling_w_m2=None, isat_iso_w_m2=None,
    isat_detuned_pi_w_m2=25.055, upper_hfs=_CS_6P12_HFS)


# ---------------------------------------------------------------------------
# Species instances (spec 01 §3.2-3.3)
# Masses: AME/NIST digits as adopted in ARC; cross-check Steck 2.3.4 agrees
# to <= 2e-8 u [VERIFIED both routes]. Core polarizabilities alpha_d:
# Marinescu et al., PRA 49, 982 (1994) via ARC [VERIFIED-ARC].
# E_I: ground-hyperfine-CENTROID convention, ruling R-11.
# ---------------------------------------------------------------------------

RB85 = Species(
    name="Rb85", mass_u=84.911_789_7379, nuclear_spin=2.5, abundance=0.7217,
    e_ion_hz=1.010_024_700e15, e_ion_unc_hz=7e6,
    ground_n=5, alpha_core_au=9.076,
    series=_RB85_SERIES, ground_hfs=_RB85_GROUND_HFS,
    d1=_RB85_D1, d2=_RB85_D2,
    e_ion_systematic_hz=40e6,
    e_ion_note=("Sanguinetti 2009 Method-3 (comb, centroid-referenced) "
                "[VERIFIED]. Steck/Lee-1978 sits 41 MHz (6 sigma) above -> "
                "+-40 MHz systematic on Rb-85 ABSOLUTE optical frequencies; "
                "cancels in Rydberg-Rydberg intervals (spec 01 §3.3/§7.2)."),
    mass_source="AME via ARC; Steck 2.3.4 cross-check [VERIFIED]")

RB87 = Species(
    name="Rb87", mass_u=86.909_180_5310, nuclear_spin=1.5, abundance=0.2783,
    e_ion_hz=1.010_024_892_9e15, e_ion_unc_hz=0.3e6,
    ground_n=5, alpha_core_au=9.076,
    series=_RB87_SERIES, ground_hfs=_RB87_GROUND_HFS,
    d1=_RB87_D1, d2=_RB87_D2,
    e_ion_note=("Mack 2011: E_I(from 5S1/2 F=1)/h = 1 010 029 164.6(3) MHz, "
                "converted to centroid via ruling R-11 (+dE_hfs(F=1) = "
                "-4271.68 MHz) [VERIFIED]."),
    mass_source="AME via ARC; Steck 2.3.4 cross-check [VERIFIED]")

CS133 = Species(
    name="Cs133", mass_u=132.905_451_9610, nuclear_spin=3.5, abundance=1.0,
    e_ion_hz=941.542_215_86e12, e_ion_unc_hz=0.04e6,
    ground_n=6, alpha_core_au=15.644,
    series=_CS133_SERIES, ground_hfs=_CS_GROUND_HFS,
    d1=_CS_D1, d2=_CS_D2,
    e_ion_systematic_hz=0.47e6,
    e_ion_note=("Deiglmayr 2016 [VERIFIED]; self-consistent with the 2016 "
                "quantum-defect fit (UV lines close < 80 kHz, AS-08). A 2025 "
                "erratum (PRA 112, 049902(E)) shifts E_I by +0.47 MHz; per "
                "audit R3 the erratum VALUE is not encoded — the offset is "
                "declared as a systematic on Cs absolute energies only and "
                "cancels in all Rydberg-Rydberg intervals."),
    mass_source="AME via ARC; Steck 2.3.4 cross-check [VERIFIED]")

SPECIES: dict[str, Species] = {"Rb85": RB85, "Rb87": RB87, "Cs133": CS133}


# Auxiliary Mack 2011 anchors for absolute benchmarks (spec 01 §3.3, AS-04..06)
RB87_E_ION_FROM_5S12_F1 = SourcedValue(
    1.010_029_164_6e15, "Hz",
    "Mack et al., PRA 83, 052515 (2011): E_I from 5S1/2(F=1)",
    Confidence.VERIFIED, uncertainty=0.3e6)
RB87_E_ION_FROM_5P32_F3 = SourcedValue(
    625.794_214_8e12, "Hz",
    "Mack 2011 Table II: E_i from 5P3/2(F=3)",
    Confidence.VERIFIED, uncertainty=0.3e6)


# c*R_M cross-check anchors (spec 01 §2.1 / §4.1 step 1; tolerance 0.5 MHz on
# c*R_M — catches mass/CODATA regressions). Check values, never inputs.
_RM_ANCHORS_HZ: dict[str, tuple[float, float]] = {
    # Mack 2011 quotes 3 289 821 194.66(2) MHz [VERIFIED]
    "Rb87": (3.289_821_194_66e15, 0.5e6),
    # spec 01 §2.1 computed cross-check [VERIFIED (calc)]
    "Rb85": (3.289_820_706_08e15, 0.5e6),
    # Deiglmayr 2016: R_M(Cs) = 109 736.862 733 9(6) cm^-1 [VERIFIED]
    "Cs133": (109_736.862_733_9 * INV_CM_HZ, 5e-6 * INV_CM_HZ),
}


def rydberg_constant_hz(sp: Species) -> float:
    """c*R_M [Hz] via Eq. (1.2): R_M = R_inf/(1 + m_e/M_atom) (ruling R-10).

    Spec 01 §2.1/§4.1: asserted against the published anchors to 0.5 MHz
    (Rb) / 5e-6 cm^-1 (Cs); a breach raises IntegrityError (mass or CODATA
    regression — refuse to continue on corrupted constants).
    """
    c_rm = C * RYD_INF / (1.0 + M_E / (sp.mass_u * AMU))
    anchor = _RM_ANCHORS_HZ.get(sp.name)
    if anchor is not None:
        value, tol = anchor
        if abs(c_rm - value) > tol:
            raise IntegrityError(
                f"c*R_M({sp.name}) = {c_rm:.6e} Hz deviates from the spec 01 "
                f"anchor {value:.6e} Hz by {abs(c_rm - value)/1e6:.3f} MHz "
                f"(> {tol/1e6:.3f} MHz): mass/CODATA regression")
    return float(c_rm)


# ---------------------------------------------------------------------------
# Quantum defects
# ---------------------------------------------------------------------------

def _validate_lj(l: int, j: float) -> None:
    if l < 0 or int(l) != l:
        raise ValueError(f"l must be a non-negative integer, got {l!r}")
    if abs(abs(j - l) - 0.5) > 1e-9 or j <= 0:
        raise ValueError(f"j must equal l +- 1/2 (got l={l}, j={j})")


def _validate_n(n_arr: np.ndarray, l: int) -> None:
    if np.any(n_arr != np.floor(n_arr)):
        raise ValueError("n must be integer-valued")
    if np.any(n_arr < l + 1):
        raise ValueError(f"n must satisfy n >= l+1 (l={l}): no bound state")


def _series_for(sp: Species, l: int, j: float) -> RitzSeries:
    """Coefficient row for (l, j); l = 4 falls back across j (G-series
    j-splitting unresolved, spec 01 §7.6). Missing series -> IntegrityError
    (refuse-to-guess; audit §3 item 2)."""
    key = (l, float(j))
    if key in sp.series:
        return sp.series[key]
    if l == 4:
        for (ll, _), row in sp.series.items():
            if ll == 4:
                return row
    raise IntegrityError(
        f"no sourced quantum-defect series for {sp.name} (l={l}, j={j}); "
        "refusing to guess (spec 01 §3.4-3.5 / audit §3)")


def ritz_form_a(series: RitzSeries, n: np.ndarray) -> np.ndarray:
    """Form A (modified Ritz), spec 01 Eq. (1.3): delta0 in the denominators."""
    m2 = (n - series.d0) ** 2
    return (series.d0 + series.d2 / m2 + series.d4 / m2**2
            + series.d6 / m2**3 + series.d8 / m2**4)


def ritz_form_b(series: RitzSeries, n: np.ndarray,
                tol: float = 1e-12, max_iter: int = 30) -> np.ndarray:
    """Form B (self-consistent Ritz), spec 01 Eq. (1.4), fixed-point iteration.

    Contraction ~2*d2/n^3 << 1; spec 01 §4.1: iterate to |d_new - d| < 1e-12,
    max 30 iterations, raise IntegrityError if unconverged.
    """
    d = np.full_like(n, series.d0, dtype=float)
    for _ in range(max_iter):
        m2 = (n - d) ** 2
        d_new = (series.d0 + series.d2 / m2 + series.d4 / m2**2
                 + series.d6 / m2**3 + series.d8 / m2**4)
        if np.max(np.abs(d_new - d)) < tol:
            return d_new
        d = d_new
    raise IntegrityError(
        f"Form B fixed point not converged in {max_iter} iterations "
        f"(l={series.l}, j={series.j})")


def polarization_defect(alpha_d_au: float, n: np.ndarray | float,
                        l: int) -> np.ndarray | float:
    """Core-polarization quantum defect, spec 01 Eq. (1.6) [dimensionless].

    delta_pol(n,l) = (alpha_d/4)*[3n^2 - l(l+1)] / [n^2 (l-1/2) l (l+1/2)
    (l+1) (l+3/2)]; n = np.inf gives the n >> l asymptote. alpha_d in a.u.
    Normative for l >= 5 (l = 4 uses measured coefficients); benchmark AS-09
    checks it reproduces the measured G defects at l = 4 to <= 5 %.
    """
    if l < 1:
        raise ValueError("polarization defect requires l >= 1")
    n_arr = np.asarray(n, dtype=float)
    denom = (l - 0.5) * l * (l + 0.5) * (l + 1) * (l + 1.5)
    with np.errstate(invalid="ignore"):
        radial = np.where(np.isinf(n_arr), 3.0, 3.0 - l * (l + 1) / n_arr**2)
    out = (alpha_d_au / 4.0) * radial / denom
    return float(out) if np.ndim(n) == 0 else out


def quantum_defect(sp: Species, n: np.ndarray | int, l: int,
                   j: float) -> np.ndarray | float:
    """delta_lj(n) [dimensionless], spec 01 §2.2-2.3.

    Dispatches on the series' declared form (Form A closed form, Form B
    fixed point — ruling R-14); l >= 5 uses the core-polarization formula
    Eq. (1.6). Raises IntegrityError below n_min_hard (Rb 8 / Cs 12, audit
    §3 item 1); warns below n_min_mhz (19 / 25).
    """
    _validate_lj(l, j)
    n_arr = np.asarray(n, dtype=float)
    scalar = n_arr.ndim == 0
    n_arr = np.atleast_1d(n_arr).astype(float)
    _validate_n(n_arr, l)
    if l >= 5:
        delta = np.asarray(polarization_defect(sp.alpha_core_au, n_arr, l))
    else:
        series = _series_for(sp, l, j)
        if np.any(n_arr < series.n_min_hard):
            raise IntegrityError(
                f"{sp.name} n={np.min(n_arr):.0f} (l={l}, j={j}) is below the "
                f"hard validity floor n={series.n_min_hard}: single-channel "
                "Ritz breaks (core penetration/perturbers). Low-lying levels "
                "are data-only (spec 01 §4.2/§7.1) — refusing to guess.")
        if np.any(n_arr < series.n_min_mhz):
            warnings.warn(
                f"{sp.name} (l={l}, j={j}): n < {series.n_min_mhz} is below "
                "the MHz-grade validity window (spec 01 §4.2); expect only "
                "100-MHz-grade accuracy", stacklevel=2)
        if series.form == "iterated":
            delta = ritz_form_b(series, n_arr)
        else:
            delta = ritz_form_a(series, n_arr)
    return float(delta[0]) if scalar else delta


def n_star(sp: Species, n: np.ndarray | int, l: int,
           j: float) -> np.ndarray | float:
    """Effective principal quantum number n* = n - delta_lj(n) [dimensionless]."""
    return np.asarray(n, dtype=float) - quantum_defect(sp, n, l, j) \
        if np.ndim(n) else float(n) - quantum_defect(sp, n, l, j)


# ---------------------------------------------------------------------------
# Energies and transitions
# ---------------------------------------------------------------------------

def binding_energy_hz(sp: Species, n: np.ndarray | int, l: int,
                      j: float) -> np.ndarray | float:
    """Binding energy E_b/h = c*R_M/n*^2 > 0 [Hz], spec 01 Eq. (1.1)."""
    return rydberg_constant_hz(sp) / np.asarray(n_star(sp, n, l, j)) ** 2 \
        if np.ndim(n) else rydberg_constant_hz(sp) / n_star(sp, n, l, j) ** 2


def energy_hz(sp: Species, n: np.ndarray | int, l: int, j: float, *,
              ref: Literal["ionization", "ground_centroid"] = "ionization",
              ) -> np.ndarray | float:
    """E/h of |n l j> [Hz], spec 01 Eq. (1.1).

    ref='ionization': negative binding energy -c*R_M/n*^2.
    ref='ground_centroid': e_ion_hz - c*R_M/n*^2 (absolute optical scale,
    from the ground hyperfine centroid — ruling R-11). Rb-85 / Cs absolute
    values carry the declared systematic sp.e_ion_systematic_hz (+-40 MHz /
    +0.47 MHz) which the caller must propagate (audit §3 items 4-5); it
    cancels in intervals, which is why transition_hz never touches E_I.
    """
    e_b = binding_energy_hz(sp, n, l, j)
    if ref == "ionization":
        return -e_b
    if ref == "ground_centroid":
        return sp.e_ion_hz - e_b
    raise ValueError(f"unknown ref {ref!r}")


def transition_hz(sp: Species, n1: np.ndarray | int, l1: int, j1: float,
                  n2: np.ndarray | int, l2: int, j2: float,
                  ) -> np.ndarray | float:
    """Signed transition frequency (E2 - E1)/h [Hz], spec 01 Eq. (1.9).

    Computed directly from binding energies — E_I cancels exactly (immune to
    the Rb-85/Cs absolute-scale systematics; spec 01 §4.1 step 4). Positive
    means state 2 lies above state 1. Note the ordering trap (spec 01 §2.5):
    (n+1)P3/2 lies BELOW nD5/2, so nD5/2 -> (n+1)P3/2 is negative (downward).
    """
    return binding_energy_hz(sp, n1, l1, j1) - binding_energy_hz(sp, n2, l2, j2)


def rf_transition_hz(sp: Species, n1: int, l1: int, j1: float,
                     n2: int, l2: int, j2: float) -> tuple[float, int]:
    """Rydberg-Rydberg RF/mm-wave transition: (|nu| [Hz], sign).

    sign = +1 if state 2 lies above state 1, -1 if below (spec 01 §2.5
    'report nu > 0 with an explicit sign field'). Downstream spec 06 consumes
    the sign for AT sideband asymmetries.
    """
    nu = float(transition_hz(sp, n1, l1, j1, n2, l2, j2))
    return abs(nu), (1 if nu >= 0 else -1)


# ---------------------------------------------------------------------------
# Hyperfine structure
# ---------------------------------------------------------------------------

def _half_int(x: float, name: str) -> Fraction:
    two_x = 2.0 * x
    if abs(two_x - round(two_x)) > 1e-9:
        raise ValueError(f"{name} must be integer or half-integer, got {x}")
    return Fraction(int(round(two_x)), 2)


def hyperfine_shift_hz(hfs: HyperfineConstants, i_nuc: float, j_elec: float,
                       f_tot: float) -> float:
    """F-level shift from the fine-structure centroid [Hz], spec 01 Eq. (1.7).

    dE(F)/h = (A/2) K + B [ (3/2)K(K+1) - 2 I(I+1) J(J+1) ] /
    [4 I(2I-1) J(2J-1)], K = F(F+1) - I(I+1) - J(J+1). K and the B-term
    ratio are evaluated in exact rationals; B term only for I, J > 1/2.
    The octupole C term (Cs 6P3/2, 0.56 kHz) is provenance-only, not applied.
    """
    I = _half_int(i_nuc, "I")
    J = _half_int(j_elec, "J")
    F = _half_int(f_tot, "F")
    if not abs(I - J) <= F <= I + J:
        raise ValueError(f"F={f_tot} outside |I-J|..I+J for I={i_nuc}, J={j_elec}")
    K = F * (F + 1) - I * (I + 1) - J * (J + 1)
    shift = hfs.A_hz * float(K) / 2.0
    if I > Fraction(1, 2) and J > Fraction(1, 2):
        num = Fraction(3, 2) * K * (K + 1) - 2 * I * (I + 1) * J * (J + 1)
        den = 4 * I * (2 * I - 1) * J * (2 * J - 1)
        shift += hfs.B_hz * float(num / den)
    return shift


def probe_transition_hz(sp: Species, line: Literal["D1", "D2"],
                        F: int, Fp: int) -> float:
    """Hyperfine-resolved D-line frequency [Hz], spec 01 Eq. (1.10).

    nu(F -> F') = nu_centroid + dE_hfs(upper, F')/h - dE_hfs(ground, F)/h.
    Validates F/F' against the I (+-) J ranges; hyperfine_shift_hz enforces
    them exactly. Frequencies are vacuum, Steck rev 2.3.4 centroids.
    """
    dline = sp.d2 if line == "D2" else sp.d1
    if dline is None or sp.ground_hfs is None:
        raise IntegrityError(f"{sp.name} carries no {line} dataset")
    j_up = 1.5 if line == "D2" else 0.5
    low = hyperfine_shift_hz(sp.ground_hfs, sp.nuclear_spin, 0.5, F)
    up = hyperfine_shift_hz(dline.upper_hfs, sp.nuclear_spin, j_up, Fp)
    return dline.nu0_hz + up - low


# ---------------------------------------------------------------------------
# Coupling laser (ruling R-15: always computed at runtime, never hard-coded)
# ---------------------------------------------------------------------------

def coupling_laser_hz(sp: Species, line: Literal["D1", "D2"],
                      n: np.ndarray | int, l: int, j: float,
                      ) -> np.ndarray | float:
    """Coupling-laser frequency [Hz]: D-line upper level -> |n l j>.

    nu_c = [E_I - c*R_M/n*^2] - nu0(line), both measured from the ground
    hyperfine centroid; the intermediate level is taken at its hyperfine
    centroid. Ruling R-15: lambda_c is state-dependent and computed here at
    runtime — 480.0 nm (Rb) / 509.4 nm (Cs) are benchmark-fixture values only.
    E_I systematics (Rb-85 +-40 MHz, Cs +0.47 MHz) apply — sub-1e-7 in nu_c.
    """
    dline = sp.d2 if line == "D2" else sp.d1
    if dline is None:
        raise IntegrityError(f"{sp.name} carries no {line} dataset")
    return energy_hz(sp, n, l, j, ref="ground_centroid") - dline.nu0_hz


def coupling_wavelength_m(sp: Species, line: Literal["D1", "D2"],
                          n: np.ndarray | int, l: int, j: float,
                          ) -> np.ndarray | float:
    """Vacuum coupling wavelength lambda_c = c/nu_c [m] (ruling R-15)."""
    return C / coupling_laser_hz(sp, line, n, l, j)


# ---------------------------------------------------------------------------
# Rydberg-state hyperfine scaling (spec 01 §2.4 / §3.8)
# ---------------------------------------------------------------------------

# A(n) = coeff * n*^-3. Per-row provenance mandatory (audit R7): the Rb-85 nS
# and Cs nS coefficients are ESTIMATES (+-30/40 %) pending Sassmannshausen
# 2013; outputs must carry these uncertainties. No sourced coefficients exist
# for nP3/2/nD/nF at any n: rydberg_hfs_A_hz returns exactly 0 there with the
# documented bound (audit §3 item 3), never a scaled guess.
RYDBERG_HFS_COEFF: dict[tuple[str, int, float], SourcedValue] = {
    ("Rb87", 0, 0.5): SourcedValue(
        16.75e9, "Hz*n*^3",
        "Mack 2011 / Li 2003: 2A n*^3 = 33.5(9) GHz",
        Confidence.VERIFIED, uncertainty=0.45e9),
    ("Rb85", 0, 0.5): SourcedValue(
        4.96e9, "Hz*n*^3",
        "spec 01 §2.4 DERIVED-ESTIMATE (A_5S(85)/A_5S(87) x 16.75 GHz), +-30%",
        Confidence.UNVERIFIED, uncertainty=1.5e9),
    ("Cs133", 0, 0.5): SourcedValue(
        17.1e9, "Hz*n*^3",
        "spec 01 §2.4 ground-state-scaling estimate ONLY, +-40%; fetch "
        "Sassmannshausen/Merkt/Deiglmayr PRA 87, 032519 (2013) before relying",
        Confidence.UNVERIFIED, uncertainty=6.8e9),
    ("Cs133", 1, 0.5): SourcedValue(
        3.85e9, "Hz*n*^3",
        "derived from Deiglmayr 2016 measured 27p1/2 F=3-4 interval "
        "1.2(1) MHz = 4A",
        Confidence.VERIFIED, uncertainty=0.32e9),
}


def rydberg_hfs_coefficient(sp: Species, l: int, j: float) -> SourcedValue | None:
    """Sourced A_S coefficient [Hz*n*^3] for (l, j), or None if unsourced.

    Callers needing hyperfine-resolved output MUST check the confidence tag:
    Cs nS is UNVERIFIED (+-40 %) — refuse hyperfine-resolved modes there
    (audit R7 / §3 item 3).
    """
    return RYDBERG_HFS_COEFF.get((sp.name, l, float(j)))


def rydberg_hfs_A_hz(sp: Species, n: np.ndarray | int, l: int,
                     j: float) -> np.ndarray | float:
    """Rydberg hyperfine A(n) = A_S * n*^-3 [Hz], spec 01 §2.4/§3.8.

    Sourced series only: (l, j) = (0, 1/2) all species and Cs (1, 1/2); the
    Rb-85 nS and Cs nS coefficients are estimates (+-30/40 %, see
    rydberg_hfs_coefficient for tags). All other series return exactly 0
    with the documented bound: |A(n)| < A_S(same species, S or P1/2)*n*^-3
    (contact-interaction dominance, spec 01 §3.8) — never a scaled guess.
    """
    coeff = rydberg_hfs_coefficient(sp, l, j)
    ns = n_star(sp, n, l, j)
    if coeff is None:
        return np.zeros_like(np.asarray(ns, dtype=float)) if np.ndim(n) else 0.0
    return coeff.value * np.asarray(ns, dtype=float) ** -3 \
        if np.ndim(n) else coeff.value * float(ns) ** -3
