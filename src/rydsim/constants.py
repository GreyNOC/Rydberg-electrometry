"""Physical constants and unit conversions for RydSim.

Provenance policy
-----------------
Every fundamental constant here is taken from ``scipy.constants`` (CODATA as
shipped with the installed scipy) rather than typed from memory, so the values
are externally sourced and update with the library. Derived quantities
(atomic units, conversion factors) are *computed* from those primitives.

Species-specific data (masses, quantum defects, ionization limits, ...) does
NOT live here — it lives in ``rydsim.atom`` with per-value source tags, per
the integrity policy in docs/spec/00-integrity-audit.md.
"""

from __future__ import annotations

import math

import scipy.constants as _sc

# ---------------------------------------------------------------------------
# SI-exact defining constants (exact by the 2019 SI redefinition)
# ---------------------------------------------------------------------------
H = _sc.h                     # Planck constant [J s] (exact)
HBAR = _sc.hbar               # reduced Planck constant [J s] (exact)
C = _sc.c                     # speed of light [m/s] (exact)
E_CHARGE = _sc.e              # elementary charge [C] (exact)
KB = _sc.k                    # Boltzmann constant [J/K] (exact)

# ---------------------------------------------------------------------------
# CODATA measured constants (via scipy.constants)
# ---------------------------------------------------------------------------
EPS0 = _sc.epsilon_0          # vacuum permittivity [F/m]
M_E = _sc.m_e                 # electron mass [kg]
A0 = _sc.physical_constants["Bohr radius"][0]              # [m]
RYD_INF = _sc.physical_constants["Rydberg constant"][0]    # R_infinity [1/m]
E_H = _sc.physical_constants["Hartree energy"][0]          # Hartree [J]
AMU = _sc.physical_constants["atomic mass constant"][0]    # [kg]
ALPHA_FS = _sc.alpha          # fine-structure constant
MU_B = _sc.physical_constants["Bohr magneton"][0]          # [J/T]

# ---------------------------------------------------------------------------
# Derived atomic-unit conversions (computed, not typed)
# ---------------------------------------------------------------------------
# atomic unit of electric dipole moment: e * a0  [C m]
AU_DIPOLE = E_CHARGE * A0
# atomic unit of electric field: E_h / (e a0)  [V/m]
AU_EFIELD = E_H / (E_CHARGE * A0)
# atomic unit of time: hbar / E_h  [s]
AU_TIME = HBAR / E_H
# atomic unit of energy in frequency units: E_h / h  [Hz]
AU_ENERGY_HZ = E_H / H
# polarizability: atomic unit e^2 a0^2 / E_h  [C^2 m^2 / J]
AU_POLARIZABILITY = E_CHARGE**2 * A0**2 / E_H
# Rydberg energy in various units
RYD_J = H * C * RYD_INF       # hc R_inf [J]
RYD_HZ = C * RYD_INF          # c R_inf [Hz]

# Common lab-unit conversions
def polarizability_au_to_hz_per_v2_cm2(alpha_au: float) -> float:
    """Convert polarizability from atomic units to Hz/(V/cm)^2.

    alpha[Hz/(V/cm)^2] = alpha[au] * AU_POLARIZABILITY / h * (100)^2
    since 1 V/cm = 100 V/m and the Stark shift is dE = -1/2 alpha E^2.
    """
    return alpha_au * AU_POLARIZABILITY / H * 100.0**2


def dipole_au_to_si(d_au: float) -> float:
    """Dipole moment: atomic units (e a0) -> SI (C m)."""
    return d_au * AU_DIPOLE


def efield_v_per_cm_to_si(e_v_cm: float) -> float:
    """Electric field: V/cm -> V/m."""
    return e_v_cm * 100.0


def efield_si_to_v_per_cm(e_v_m: float) -> float:
    """Electric field: V/m -> V/cm."""
    return e_v_m / 100.0


def rabi_from_field(d_si: float, e_si: float) -> float:
    """Angular Rabi frequency Omega = d E / hbar [rad/s].

    Convention (docs/spec/00-conventions.md): Omega is the *angular* Rabi
    frequency; spectroscopic splittings in Hz are Omega / (2 pi).
    """
    return d_si * e_si / HBAR


def field_from_rabi(d_si: float, omega: float) -> float:
    """Electric field amplitude E = hbar Omega / d [V/m]."""
    return HBAR * omega / d_si


def wavelength_to_angular_freq(lambda_m: float) -> float:
    """Vacuum wavelength [m] -> angular frequency [rad/s]."""
    return 2.0 * math.pi * C / lambda_m


CODATA_SOURCE = f"scipy.constants (scipy {_sc.__name__} shipped CODATA)"
