"""Superheterodyne transduction and noise chain.

Physics-agnostic receiver mathematics: given the atomic transfer curve
P_probe(E_RF) (transmitted probe power vs RF field amplitude, from the
EIT/Lindblad engine), this module finds the optimal LO operating point,
computes the transduction slope, assembles the technical-noise budget, and
converts everything to noise-equivalent field. The atom-projection /
standard-quantum-limit expressions live here too but are wired to the
formulas fixed in docs/spec/08 (see SQL section).

Superhet principle (Jing et al., Nat. Phys. 16, 911 (2020)): with a strong
resonant LO E_LO and weak signal E_sig at offset delta, the total field
amplitude is E_LO + E_sig cos(delta t + phi) for E_sig << E_LO, so probe
power oscillates at delta with amplitude |dP/dE| * E_sig — linear-in-signal
readout with phase preserved.

Conventions: one-sided power spectral densities [W^2/Hz]; E in V/m;
NEF in (V/m)/sqrt(Hz).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .constants import C, H, HBAR


@dataclass
class NoiseBudget:
    """One-sided PSDs of probe-power fluctuations at the detector [W^2/Hz]."""

    shot: float
    rin: float
    detector: float
    frequency: float = 0.0   # laser frequency noise via the EIT slope (spec 08 §2.4.4)

    @property
    def total(self) -> float:
        return self.shot + self.rin + self.detector + self.frequency

    def as_dict(self) -> dict[str, float]:
        return {"shot": self.shot, "rin": self.rin,
                "detector": self.detector, "frequency": self.frequency,
                "total": self.total}


def frequency_noise_psd(dp_dnu: float, linewidth_hz: float) -> float:
    """Laser-frequency-noise PSD referred to detected power [W^2/Hz].

    S_P = (dP/dnu)^2 * S_nu with the white-frequency-noise equivalence
    S_nu = Delta_nu / pi [Hz^2/Hz] for a Lorentzian FWHM Delta_nu
    (spec 00 lock #8, spec 08 §2.4.4). dp_dnu in W/Hz.
    """
    return dp_dnu**2 * (linewidth_hz / np.pi)


def shot_noise_psd(p_opt: float, wavelength: float, eta: float = 1.0) -> float:
    """One-sided photon shot-noise PSD: S = 2 h nu P / eta [W^2/Hz].

    eta = detector quantum efficiency (spec 08 §2.4.1). Self-check identity
    (spec B4): equals the photocurrent form (2 e i / R^2) with i = R P,
    R = eta e/(h nu) — verified in tests.
    """
    nu = C / wavelength
    return 2.0 * H * nu * p_opt / eta


def shot_noise_psd_photocurrent_form(p_opt: float, wavelength: float,
                                     eta: float = 1.0) -> float:
    """Same quantity via S_i = 2 e i, referred back to optical power.

    Exists only for the B4 unit self-check; must equal shot_noise_psd to
    machine precision.
    """
    from scipy.constants import e as q_e
    nu = C / wavelength
    resp = eta * q_e / (H * nu)          # A/W
    i_mean = resp * p_opt
    s_i = 2.0 * q_e * i_mean             # A^2/Hz
    return s_i / resp**2                 # W^2/Hz


def rin_psd(p_opt: float, rin_db_per_hz: float,
            f_hz: float | None = None, f_corner_hz: float = 0.0) -> float:
    """Laser RIN PSD: S = RIN_w (1 + f_c/f) P^2 [W^2/Hz] (spec 08 §2.4.2).

    rin_db_per_hz is the white floor; with f_hz and f_corner_hz given, the
    1/f knee is included — the reason superhets detect at beat frequencies
    above the corner.
    """
    rin_w = 10.0 ** (rin_db_per_hz / 10.0)
    if f_hz is not None and f_corner_hz > 0 and f_hz > 0:
        rin_w *= (1.0 + f_corner_hz / f_hz)
    return rin_w * p_opt**2


def envelope_fourier(u: float, n_harmonic: int = 1,
                     n_samples: int = 4096) -> float:
    """Exact Fourier coefficient of the beat envelope (spec 08 §2.1).

    E_env(t)/E_LO = sqrt(1 + u^2 + 2u cos(x)); returns the cosine Fourier
    coefficient of harmonic n (a_n, in units of E_LO). Small-u expansion:
    a_1 = u - u^3/8, a_2 = -u^2/4 (verified in tests B1/B2).
    """
    x = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    env = np.sqrt(1.0 + u * u + 2.0 * u * np.cos(x))
    return float(2.0 * np.mean(env * np.cos(n_harmonic * x)))


@dataclass
class OperatingPoint:
    """Superhet operating point on the transfer curve."""

    e_lo: float                 # LO field amplitude [V/m]
    p_probe: float              # transmitted probe power at E_LO [W]
    slope: float                # dP/dE at E_LO [W/(V/m)]
    nef: float                  # noise-equivalent field [(V/m)/sqrt(Hz)]
    budget: NoiseBudget


def transfer_slope(transfer: Callable[[float], float], e_lo: float,
                   rel_step: float = 1e-3) -> float:
    """Central-difference dP/dE with Richardson refinement."""
    h1 = max(e_lo * rel_step, 1e-12)
    d1 = (transfer(e_lo + h1) - transfer(e_lo - h1)) / (2 * h1)
    h2 = h1 / 2
    d2 = (transfer(e_lo + h2) - transfer(e_lo - h2)) / (2 * h2)
    return (4 * d2 - d1) / 3  # O(h^4)


def optimize_lo(transfer: Callable[[float], float],
                e_lo_grid: np.ndarray,
                wavelength: float,
                rin_db_per_hz: float = -150.0,
                detector_nep_w_per_rthz: float = 1e-12,
                extra_psd: Callable[[float], float] | None = None
                ) -> OperatingPoint:
    """Scan the LO grid, return the operating point minimizing TOTAL NEF.

    NEF(E_LO) = sqrt(S_total(P(E_LO))) / |dP/dE|_{E_LO}

    extra_psd(E_LO) -> W^2/Hz adds a caller-computed noise term (e.g. laser
    frequency noise, whose slope dP/dnu the caller evaluates from the OBE).
    It enters the optimization, not just the report: omitting a dominant
    term from the objective moves the optimum, not only the number.
    """
    best: OperatingPoint | None = None
    for e_lo in e_lo_grid:
        p = float(transfer(float(e_lo)))
        k = transfer_slope(transfer, float(e_lo))
        if abs(k) < 1e-30:
            continue
        budget = NoiseBudget(
            shot=shot_noise_psd(p, wavelength),
            rin=rin_psd(p, rin_db_per_hz),
            detector=detector_nep_w_per_rthz**2,
            frequency=float(extra_psd(float(e_lo))) if extra_psd else 0.0,
        )
        nef = float(np.sqrt(budget.total) / abs(k))
        if best is None or nef < best.nef:
            best = OperatingPoint(e_lo=float(e_lo), p_probe=p, slope=k,
                                  nef=nef, budget=budget)
    if best is None:
        raise ValueError("transfer curve has no usable slope on the LO grid")
    return best


def min_detectable_field(nef: float, integration_time: float,
                         snr: float = 1.0) -> float:
    """E_min = SNR * NEF / sqrt(t) [V/m] for integration time t [s]."""
    return snr * nef / np.sqrt(integration_time)


def compression_point(transfer: Callable[[float], float], e_lo: float,
                      e_sig_grid: np.ndarray) -> float | None:
    """1-dB compression: signal amplitude where demodulated response drops
    to 10^(-1/20) of the small-signal linear extrapolation.

    Simulates the amplitude-modulated field E_LO + E_sig cos(delta t) through
    the static transfer curve (valid for delta within the atomic bandwidth)
    and demodulates the fundamental.
    """
    phases = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    slope = transfer_slope(transfer, e_lo)
    if abs(slope) < 1e-30:
        return None
    for e_sig in e_sig_grid:
        # fundamental Fourier amplitude of P(E_LO + E_sig cos(phase))
        p_t = np.array([transfer(e_lo + e_sig * np.cos(ph)) for ph in phases])
        fund = 2.0 * np.abs(np.mean(p_t * np.exp(-1j * phases)))
        linear = abs(slope) * e_sig
        if linear > 0 and fund / linear < 10 ** (-1.0 / 20.0):
            return float(e_sig)
    return None


def sql_nef(dipole_si: float, n_eff: float, tau_coh: float,
            convention: str = "hbar", mode: str = "continuous",
            rep_rate: float | None = None) -> float:
    """Atom-projection-noise (SQL) NEF [(V/m)/sqrt(Hz)] — spec 08 §2.4.5.

    continuous: NEF = (hbar/mu) / sqrt(N_eff * tau)
    pulsed:     NEF = (hbar/mu) / (tau * sqrt(N_eff * R))
    convention='h' multiplies by 2*pi (some papers' bookkeeping; the
    Sci. Adv. 2024 SQL of 3.70 nV/cm/rtHz reproduces under 'h' with
    T2 = 57.8 ns, or under 'hbar'-pulsed with tau = 3.83 us at R = 100 Hz).
    Every SQL claim must be reported WITH (N_eff, tau, convention, mode) —
    never as a bare number.
    """
    if mode == "continuous":
        nef = (HBAR / dipole_si) / np.sqrt(n_eff * tau_coh)
    elif mode == "pulsed":
        if not rep_rate:
            raise ValueError("pulsed SQL needs rep_rate")
        nef = (HBAR / dipole_si) / (tau_coh * np.sqrt(n_eff * rep_rate))
    else:
        raise ValueError(f"unknown mode {mode!r}")
    if convention == "h":
        nef *= 2.0 * np.pi
    elif convention != "hbar":
        raise ValueError(f"unknown convention {convention!r}")
    return float(nef)


def sql_field(dipole_si: float, gamma_eff: float, n_atoms: float,
              integration_time: float) -> float:
    """E_SQL(t) = (hbar/mu) sqrt(gamma_eff/(N t)) [V/m] — the continuous
    hbar-convention SQL with tau = 1/gamma_eff (spec 08: identical form)."""
    return sql_nef(dipole_si, n_atoms, 1.0 / gamma_eff) / np.sqrt(integration_time)


ETA_0 = 376.730313668  # vacuum impedance [ohm], CODATA


def noise_temperature(nef: float, frequency_hz: float,
                      gain_ref: float = 1.64) -> tuple[float, float]:
    """(T_eq [K], NF [dB]) via the standard-antenna comparison (spec 08 §2.5).

    T_eq = G lambda^2 NEF^2 / (8 pi eta0 kB); NF = 10 log10(1 + T_eq/290).
    Convention validated: 10.0 nV/cm/rtHz @ 36.9 GHz, G=1.64 -> 828 K vs
    830 K published (Sci. Adv. 2024).
    """
    from scipy.constants import k as k_b
    lam = C / frequency_hz
    t_eq = gain_ref * lam**2 * nef**2 / (8.0 * np.pi * ETA_0 * k_b)
    nf_db = 10.0 * np.log10(1.0 + t_eq / 290.0)
    return float(t_eq), float(nf_db)
