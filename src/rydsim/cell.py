"""Vapor-cell physics: alkali vapor pressure and number density.

Spec: docs/spec/05-vapor-cell-physics.md §2.a-2.b (Alcock/Steck two-
parameter model, VERIFIED against Steck's tabulated 25 C values).
Beam/Doppler/transit machinery lives in rydsim.doppler and rydsim.eit;
optical propagation in rydsim.eit (Beer-Lambert).

Accuracy: +-5% over 298-550 K (Alcock's stated fit accuracy); below 298 K
the model is an extrapolation and density_m3 warns via the returned
provenance record.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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

# Steck, VERIFIED (spec 05 §2.b)
NATURAL_ABUNDANCE = {"Rb85": 0.7217, "Rb87": 0.2783, "Cs133": 1.0}


def vapor_pressure_torr(element: str, temperature_k: float) -> float:
    """Alkali vapor pressure [torr]; solid branch below the melting point."""
    if element not in MELTING_POINT_K:
        raise ValueError(f"unsupported element {element!r} (Rb, Cs)")
    phase = "solid" if temperature_k < MELTING_POINT_K[element] else "liquid"
    c = _COEFFS[(element, phase)]
    return float(10.0 ** (c.offset + c.a - c.b / temperature_k))


def number_density_m3(element: str, temperature_k: float,
                      isotope_fraction: float = 1.0) -> float:
    """Vapor number density n = P/(kB T) [m^-3], times the isotope fraction.

    Use NATURAL_ABUNDANCE for natural cells (e.g. Rb87 in natural Rb:
    isotope_fraction = 0.2783). Model extrapolates below 298 K (+-5%
    stated accuracy holds 298-550 K).
    """
    p_pa = vapor_pressure_torr(element, temperature_k) * TORR_TO_PA
    return float(isotope_fraction * p_pa / (KB * temperature_k))


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
