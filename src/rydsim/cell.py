"""Vapor-cell physics: alkali vapor pressure and number density.

Spec: docs/spec/05-vapor-cell-physics.md §2.a-2.b (Alcock/Steck two-
parameter model, VERIFIED against Steck's tabulated 25 C values).
Beam/Doppler/transit machinery lives in rydsim.doppler and rydsim.eit;
optical propagation in rydsim.eit (Beer-Lambert).

Accuracy: +-5% over 298-550 K (Alcock's stated fit accuracy); outside that
window the model is an extrapolation and vapor_pressure_torr /
number_density_m3 emit a UserWarning naming the band (audit §3 item 17:
"warn + flag extrapolation in the result; never bare"). Audit R22 also
requires the +-5% band and the temperature sensitivity to appear in every
density-dependent output: DENSITY_MODEL_UNC_REL and
density_sensitivity_per_k() supply both, the latter derived from the fit
coefficients rather than transcribed.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .atom import SPECIES
from .constants import KB

TORR_TO_PA = 133.322  # exact conversion within stated digits (spec 05)


@dataclass(frozen=True)
class VaporCoefficients:
    """log10(P_torr) = offset + a - b/T for one phase (Alcock/Steck)."""

    a: float
    b: float
    offset: float = 2.881


# Alcock, Itkin & Horrigan, Can. Metall. Q. 23, 309 (1984) via Steck
# (Rb rev 2.3.4, Cs data sheet). Confidence: VERIFIED (spec 05 §3).
_COEFFS = {
    ("Rb", "solid"): VaporCoefficients(a=4.857, b=4215.0),
    ("Rb", "liquid"): VaporCoefficients(a=4.312, b=4040.0),
    ("Cs", "solid"): VaporCoefficients(a=4.711, b=3999.0),
    ("Cs", "liquid"): VaporCoefficients(a=4.165, b=3830.0),
}

MELTING_POINT_K = {"Rb": 312.45, "Cs": 301.65}

# Natural isotopic abundances. NOT a second transcription: derived from the
# spec-01 Species rows in rydsim.atom so the two tables cannot fork (audit
# R10 — "duplicated constants are how normative values fork"). Steck,
# VERIFIED (spec 05 §2.b); keys are species names, not element symbols.
NATURAL_ABUNDANCE: dict[str, float] = {
    name: sp.abundance for name, sp in SPECIES.items()}

# Alcock's stated fit accuracy over the validity window (spec 05 §2.a,
# audit R22): 1-sigma-equivalent relative band on the density/pressure.
DENSITY_MODEL_UNC_REL = 0.05
VALID_T_RANGE_K = (298.0, 550.0)


def _warn_if_extrapolating(temperature_k: float) -> str:
    """Audit §3 item 17: never return a bare extrapolated density.

    Returns the validity flag string ('' when inside the window) and emits
    the matching UserWarning so a caller that ignores the return value still
    sees it.
    """
    lo, hi = VALID_T_RANGE_K
    if lo <= temperature_k <= hi:
        return ""
    flag = (f"vapor model extrapolated: T = {temperature_k:.2f} K is outside "
            f"the Alcock validity window {lo:g}-{hi:g} K; the +-5% stated "
            "accuracy does not apply (spec 05 §2.a / audit §3 item 17)")
    warnings.warn(flag, stacklevel=3)
    return flag


def vapor_pressure_torr(element: str, temperature_k: float) -> float:
    """Alkali vapor pressure [torr]; solid branch below the melting point.

    Warns when T is outside the 298-550 K Alcock validity window.
    """
    if element not in MELTING_POINT_K:
        raise ValueError(f"unsupported element {element!r} (Rb, Cs)")
    _warn_if_extrapolating(temperature_k)
    phase = "solid" if temperature_k < MELTING_POINT_K[element] else "liquid"
    c = _COEFFS[(element, phase)]
    return float(10.0 ** (c.offset + c.a - c.b / temperature_k))


def number_density_m3(element: str, temperature_k: float,
                      isotope_fraction: float = 1.0) -> float:
    """Vapor number density n = P/(kB T) [m^-3], times the isotope fraction.

    `isotope_fraction` selects the semantics and the caller must choose
    deliberately (the 04<->05 interface): 1.0 gives the TOTAL ELEMENTAL
    ground-state density — the convention the collisional coefficients are
    defined against (Weller self-broadening beta*N, the Fermi pseudopotential
    shift: every ground-state perturber counts, whatever its isotope) —
    while NATURAL_ABUNDANCE[species] gives the ISOTOPE-PARTIAL density, which
    is what the probe susceptibility of one isotope needs. Passing the
    partial density into a collisional coefficient understates it by 1/eta
    (3.59x for Rb-87, 1.39x for Rb-85).

    Warns when T is outside the 298-550 K Alcock window; the +-5% model band
    is DENSITY_MODEL_UNC_REL and the T-sensitivity density_sensitivity_per_k.
    """
    p_pa = vapor_pressure_torr(element, temperature_k) * TORR_TO_PA
    return float(isotope_fraction * p_pa / (KB * temperature_k))


def density_sensitivity_per_k(element: str, temperature_k: float) -> float:
    """d ln(n)/dT [1/K] of the Alcock density — audit R22's "11 %/K" warning.

    Derived, not transcribed: with log10 P = offset + a - b/T and
    n = P/(kB T), d ln n/dT = ln(10)*b/T^2 - 1/T. Gives 0.1058/K for Rb at
    298 K (the audit's ~11 %/K), 0.0641/K at 100 C. Multiply by a cell
    temperature stability to get the induced density uncertainty.
    """
    if element not in MELTING_POINT_K:
        raise ValueError(f"unsupported element {element!r} (Rb, Cs)")
    phase = "solid" if temperature_k < MELTING_POINT_K[element] else "liquid"
    b = _COEFFS[(element, phase)].b
    return float(np.log(10.0) * b / temperature_k**2 - 1.0 / temperature_k)


def mean_speed(mass_kg: float, temperature_k: float) -> float:
    """Mean 3D thermal speed sqrt(8 kB T / (pi m)) [m/s]."""
    return float(np.sqrt(8.0 * KB * temperature_k / (np.pi * mass_kg)))


def mean_transverse_speed(mass_kg: float, temperature_k: float) -> float:
    """Thermal mean transverse speed <v_perp> = sqrt(pi kB T / (2 m)) [m/s]."""
    return float(np.sqrt(np.pi * KB * temperature_k / (2.0 * mass_kg)))


def transit_rate(mass_kg: float, temperature_k: float,
                 beam_waist_m: float) -> float:
    """Transit dephasing rate gamma_t = sqrt(2 ln 2) * <v_perp> / w0 [rad/s].

    NORMATIVE form (docs/spec/00-conventions.md lock #19 / ruling R-3):
    per-transverse-velocity-class rate with the thermal shortcut
    <v_perp> = sqrt(pi kB T / 2 m); w0 is the 1/e^2 intensity radius. FWHM
    equivalent Delta_nu_tt = gamma_t / pi. (Supersedes the demoted spec 06
    estimator ~u_2D/(2 w0), which was a factor sqrt(2 ln 2)*sqrt(2)~2.35 low.)
    Yields gamma_t/2pi = 39.8 kHz for Rb-87, 300 K, w0 = 1 mm.
    """
    v_perp = mean_transverse_speed(mass_kg, temperature_k)
    return float(np.sqrt(2.0 * np.log(2.0)) * v_perp / beam_waist_m)
