"""Experiment facade: declarative configs -> simulation runs -> Findings.

This is the layer the CLI and GUI both drive. A LadderConfig fully describes
a simulated EIT/AT experiment; run_* functions execute it and return data
plus provenance-tagged Finding objects (rydsim.provenance).

Species-aware construction (rubidium/cesium presets computing dipole moments
and rates from atomic structure) is layered on in rydsim.schemes once the
atomic-data modules are in place; this facade is deliberately usable with
directly specified rates so that custom/what-if configurations are always
possible.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from . import __version__
from .constants import H, HBAR
from .lindblad import LadderSystem
from .provenance import Finding
from .spectroscopy import ATMeasurement, eit_fwhm, measure_at_splitting
from .superhet import NoiseBudget, OperatingPoint, optimize_lo

STANDING_CAVEATS = [
    "Model potential / quantum-defect accuracy bounds apply to all atomic data.",
    "Neglected: Rydberg-Rydberg interactions at high density, free-ion fields, "
    "cell-wall microphysics beyond the phenomenological screening model.",
    "Laser technical noise modeled as stationary PSDs; real lasers drift.",
    "Steady-state OBE: dynamics faster than the atomic response bandwidth "
    "are outside the model's validity.",
]


@dataclass
class LadderConfig:
    """Complete description of a simulated ladder-EIT experiment.

    All rates are ANGULAR frequencies [rad/s] (convention: spec 00).
    Fields are amplitudes [V/m]. RF link Doppler shifts are neglected
    (k_RF ~ 0 vs optical k).
    """

    name: str = "custom"
    # ladder drive
    omega_probe: float = 2 * np.pi * 100e3
    omega_coupling: float = 2 * np.pi * 5e6
    omega_rf: float = 0.0
    delta_probe: float = 0.0
    delta_coupling: float = 0.0
    delta_rf: float = 0.0
    # decay/dephasing
    gamma_e: float = 2 * np.pi * 6.07e6      # intermediate state decay
    gamma_r: float = 2 * np.pi * 2e3         # Rydberg population decay (incl. BBR)
    gamma_rp: float = 2 * np.pi * 2e3        # second Rydberg state decay
    deph_r: float = 2 * np.pi * 100e3        # extra Rydberg dephasing (laser+transit+collisions)
    transit: float = 0.0
    # vapor / geometry
    doppler: bool = True
    temperature: float = 300.0               # K
    mass: float = 86.909 * 1.66053906892e-27  # kg (Rb-87 default; spec 01 data via schemes)
    lambda_probe: float = 780.241e-9
    lambda_coupling: float = 480.0e-9
    counter_propagating: bool = True
    # RF transition dipole moment for field inversion [C m]; None = unknown
    rf_dipole: float | None = None
    # absolute-transmission chain (spec 06 §2.4): number density of the
    # sensed isotope [m^-3] — None resolves from the Alcock/Steck vapor-
    # pressure model (rydsim.cell) at `temperature` for element/isotope —
    # and optical path length [m].
    number_density: float | None = None
    element: str = "Rb"
    isotope_fraction: float = 0.2783   # Rb-87 in natural Rb (Steck)
    cell_length: float = 0.05

    def resolved_density(self) -> float:
        """Explicit number_density if set, else the vapor-pressure model."""
        if self.number_density is not None:
            return self.number_density
        from .cell import number_density_m3
        return number_density_m3(self.element, self.temperature,
                                 self.isotope_fraction)
    # provenance notes for whatever populated this config
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        return d

    def k_signed(self, n_fields: int) -> np.ndarray:
        kp = 2 * np.pi / self.lambda_probe
        kc = 2 * np.pi / self.lambda_coupling
        ks = [kp, -kc if self.counter_propagating else kc]
        while len(ks) < n_fields:
            ks.append(0.0)
        return np.array(ks)


def _build_arrays(cfg: LadderConfig, with_rf: bool):
    omegas = [cfg.omega_probe, cfg.omega_coupling]
    deltas = [cfg.delta_probe, cfg.delta_coupling]
    decays = [0.0, cfg.gamma_e, cfg.gamma_r]
    dephs = [0.0, 0.0, cfg.deph_r]
    if with_rf:
        omegas.append(cfg.omega_rf)
        deltas.append(cfg.delta_rf)
        decays.append(cfg.gamma_rp)
        dephs.append(cfg.deph_r)
    return (np.array(omegas), np.array(deltas), np.array(decays), np.array(dephs))


def _chi_params(cfg: LadderConfig, with_rf: bool) -> "LadderChiParams":
    """Map a LadderConfig onto weak-probe continued-fraction parameters.

    Coherence decays against ground: G_k = (pop decay of level k)/2 +
    pure dephasing + transit (spec 00 convention, validated against the
    Lindblad engine in tests/test_eit.py).
    """
    from .eit import LadderChiParams

    omegas = [cfg.omega_probe, cfg.omega_coupling]
    deltas = [cfg.delta_probe, cfg.delta_coupling]
    gs = [cfg.gamma_e / 2 + cfg.transit,
          cfg.gamma_r / 2 + cfg.deph_r + cfg.transit]
    if with_rf:
        omegas.append(cfg.omega_rf)
        deltas.append(cfg.delta_rf)
        gs.append(cfg.gamma_rp / 2 + cfg.deph_r + cfg.transit)
    return LadderChiParams(
        omegas=np.array(omegas), deltas=np.array(deltas),
        k_signed=cfg.k_signed(len(omegas)),
        coherence_decays=np.array(gs),
    )


def spectrum(cfg: LadderConfig, probe_detunings: np.ndarray,
             check_convergence: bool = False,
             n_doppler_nodes: int | None = None) -> np.ndarray:
    """Complex normalized WEAK-PROBE response vs probe detuning [rad/s].

    Real part ~ absorption, imaginary ~ dispersion (per rydsim.lindblad).

    Doppler path: analytic continued-fraction susceptibility averaged on a
    resonance-refined velocity grid (rydsim.eit) — converged by
    construction, optionally proven by grid doubling (check_convergence).
    The full Lindblad engine remains available for strong-probe physics and
    is cross-validated against this path in tests/test_eit.py.

    n_doppler_nodes is accepted for CLI/back-compat but ignored: the grid
    is built from the physics (feature widths), not a node count.
    """
    with_rf = cfg.omega_rf > 0
    if cfg.doppler:
        from .eit import doppler_average

        params = _chi_params(cfg, with_rf)
        return doppler_average(params, np.asarray(probe_detunings, float),
                               cfg.mass, cfg.temperature,
                               check_convergence=check_convergence)
    omegas, deltas, decays, dephs = _build_arrays(cfg, with_rf)
    out = np.empty(len(probe_detunings), dtype=complex)
    for i, dp in enumerate(probe_detunings):
        d = deltas.copy()
        d[0] = dp
        s = LadderSystem(omegas=omegas, deltas=d, decays=decays,
                         dephasings=dephs, transit=cfg.transit)
        rho = s.steady_state()
        out[i] = rho[1, 0] / (1j * cfg.omega_probe / 2)
    return out


@dataclass
class ATResult:
    detunings: np.ndarray
    transmission: np.ndarray
    measurement: ATMeasurement | None
    field_v_per_m: float | None       # inverted field, if dipole known & resolved
    mismatch_factor: float            # k_p/k_c correction applied (1.0 if cold/co-prop)


def at_experiment(cfg: LadderConfig, span: float | None = None,
                  n_points: int = 401, n_doppler_nodes: int = 64) -> ATResult:
    """Simulate an AT-splitting field measurement and invert to E.

    Applies the wavelength-mismatch correction k_c/k_p to the probe-scanned
    splitting when Doppler averaging is on and beams are counter-propagating
    (the emergent factor validated in tests/test_doppler.py).
    """
    if span is None:
        span = max(4 * cfg.omega_rf, 2 * np.pi * 20e6)
    dps = np.linspace(-span / 2, span / 2, n_points)
    resp = spectrum(cfg, dps, n_doppler_nodes=n_doppler_nodes)
    trans = -resp.real
    m = measure_at_splitting(dps, trans)

    kp = 2 * np.pi / cfg.lambda_probe
    kc = 2 * np.pi / cfg.lambda_coupling
    mismatch = (kp / kc) if (cfg.doppler and cfg.counter_propagating) else 1.0

    e_field = None
    if m is not None and m.resolved and cfg.rf_dipole:
        omega_rabi = m.splitting / mismatch     # undo probe-axis scaling
        e_field = HBAR * omega_rabi / cfg.rf_dipole
    return ATResult(dps, trans, m, e_field, mismatch)


def at_finding(cfg: LadderConfig, result: ATResult,
               validation_state: str = "see rydsim validate") -> Finding:
    """Package an AT measurement as a provenance-tagged Finding."""
    meas = result.measurement
    res: dict = {"resolved": bool(meas and meas.resolved)}
    if meas:
        res["AT splitting (probe axis)"] = f"{meas.splitting/(2*np.pi)/1e6:.4f} MHz"
        res["mismatch factor applied"] = f"{result.mismatch_factor:.4f}"
        res["inferred Rabi frequency"] = (
            f"{meas.splitting/result.mismatch_factor/(2*np.pi)/1e6:.4f} MHz")
    if result.field_v_per_m is not None:
        res["E-field"] = f"{result.field_v_per_m:.6g} V/m"
    unc = {}
    if meas and meas.uncertainty is not None:
        unc["fit statistical"] = f"{meas.uncertainty/(2*np.pi)/1e6:.4f} MHz (1-sigma, doublet fit)"
    unc["finite-coupling systematic"] = (
        "AT peaks pulled inward by finite Omega_c; metrology-grade inversion "
        "requires the correction of spec 06(e) — reported value is uncorrected.")
    f = Finding(
        title=f"AT-splitting field measurement [{cfg.name}]",
        result=res,
        config=cfg.to_dict(),
        uncertainty_budget=unc,
        constants_used=list(cfg.sources),
        caveats=list(STANDING_CAVEATS),
        validation_state=validation_state,
    )
    return f.finalize(__version__)


def superhet_transfer(cfg: LadderConfig, e_to_omega: float,
                      e_grid: np.ndarray, probe_power_in: float,
                      n_doppler_nodes: int = 32) -> np.ndarray:
    """Transmitted probe power [W] vs RF field amplitude (LO transfer curve).

    Absolute chain (spec 06 §2.4, prefactors locked by the B-2 sum rule):
    normalized response S -> chi = i N p_ge^2/(eps0 hbar) S -> Beer-Lambert
    T = exp(-k_p L Im chi) -> P = P_in * T. The probe dipole p_ge is tied
    to gamma_e via the closed-transition identity (dipole_from_linewidth).
    e_to_omega: conversion E [V/m] -> Omega_RF [rad/s], i.e. d_RF/hbar.
    """
    from .constants import wavelength_to_angular_freq
    from .eit import chi_si, dipole_from_linewidth, transmission

    d_ge = dipole_from_linewidth(
        cfg.gamma_e, wavelength_to_angular_freq(cfg.lambda_probe))
    out = np.empty(len(e_grid))
    for i, e in enumerate(e_grid):
        c = dataclasses.replace(cfg, omega_rf=float(e) * e_to_omega)
        s_norm = spectrum(c, np.array([c.delta_probe]),
                          n_doppler_nodes=n_doppler_nodes)[0]
        chi = chi_si(np.array([s_norm]), cfg.resolved_density(), d_ge)
        t = transmission(chi, cfg.cell_length, cfg.lambda_probe)[0]
        out[i] = probe_power_in * float(t)
    return out
