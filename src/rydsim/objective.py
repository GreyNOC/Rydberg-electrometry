"""DESIGNER objective API (milestone D0) — config -> receiver performance.

The single seam between the validated physics engine and any search layer
(docs/DESIGNER.md). A search proposes a `SensorDesign`; this module returns
an `Evaluation` carrying the objectives, their validity flags, and the
provenance needed to publish the point as a Finding.

FIREWALL (DESIGNER §5): this module only ever reports numbers computed by
the validated engine. Every failure mode is explicit — an evaluation that
cannot be trusted returns `feasible=False` with a reason, and never a
silently coerced number. Search layers must discard infeasible points, not
reinterpret them.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from .constants import AU_DIPOLE, HBAR
from .experiment import LadderConfig, superhet_transfer
from .lindblad import LadderSystem, ibw_3db
from .provenance import IntegrityError
from .superhet import (
    compression_point,
    min_detectable_field,
    noise_temperature,
    optimize_lo,
    sql_nef,
    transfer_slope,
)


def velocity_participation(cfg: LadderConfig, omega_rf: float) -> float:
    """Effective fraction f_vel of the thermal ensemble that carries signal.

    Spec 08 §2.4.5 requires N_eff's velocity factor to be COMPUTED from the
    velocity-resolved response weight, never taken as a folklore fraction.
    We use the participation ratio of the RF-sensitivity weight
    w(v) = |dS/dOmega_RF|(v) over the Maxwell-Boltzmann distribution:

        f_vel = [<|w|>]^2 / <w^2>

    which is 1 when every velocity class contributes equally and small when
    only a narrow slice does (the hot-vapor EIT case). Cauchy-Schwarz
    guarantees f_vel <= 1.
    """
    from .eit import chi_ladder
    from .experiment import _chi_params
    from .constants import KB

    params = _chi_params(cfg, with_rf=True)
    sigma = float(np.sqrt(KB * cfg.temperature / cfg.mass))
    v = np.linspace(-5 * sigma, 5 * sigma, 4001)
    f = np.exp(-v**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

    d_om = max(omega_rf * 1e-3, 1.0)
    p_hi = dataclasses.replace(params, omegas=params.omegas.copy())
    p_hi.omegas[2] = omega_rf + d_om
    p_lo = dataclasses.replace(params, omegas=params.omegas.copy())
    p_lo.omegas[2] = omega_rf - d_om
    dp = np.array([cfg.delta_probe])
    w = np.abs((chi_ladder(p_hi, dp, v)[0] - chi_ladder(p_lo, dp, v)[0])
               / (2 * d_om))

    mean_abs = float(np.trapezoid(f * w, v))
    mean_sq = float(np.trapezoid(f * w**2, v))
    if mean_sq <= 0:
        return 0.0
    return float(min(mean_abs**2 / mean_sq, 1.0))


def parse_state(label: str) -> tuple[int, float]:
    """'D5/2' -> (l=2, j=2.5). Spectroscopic label to (l, j)."""
    letters = {"S": 0, "P": 1, "D": 2, "F": 3, "G": 4, "H": 5}
    label = label.strip()
    if not label or label[0].upper() not in letters:
        raise ValueError(f"unknown state label {label!r}")
    l = letters[label[0].upper()]
    jtxt = label[1:].strip()
    if "/" in jtxt:
        num, den = jtxt.split("/")
        j = float(num) / float(den)
    else:
        j = float(jtxt)
    return l, j


@dataclass(frozen=True)
class SensorDesign:
    """A candidate sensor configuration — the search layer's decision vector.

    Give either the physical atomic choice (`n`, `state`, `rf_partner`,
    optionally `rf_partner_n`) — resolved to a dipole and RF frequency
    through the validated `rydsim.atom`/`rydsim.dipoles` chain — or an
    explicit `rf_dipole_ea0` override, which is reported with a
    LITERATURE-RECALL confidence taint. Never both silently.
    """

    # atomic choice
    species: str = "Rb87"
    n: int | None = None                     # Rydberg principal quantum number
    state: str | None = None                 # e.g. "D5/2"
    rf_partner: str | None = None            # e.g. "P3/2"
    rf_partner_n: int | None = None          # defaults to n + 1 for the partner
    rf_dipole_ea0: float | None = None       # explicit override [e a0]
    # Effective-dipole convention (rydsim.dipoles.MU_RF_CONVENTIONS).
    # 'nist_pi' is normative (spec 00 lock #11). Papers differ: Sci. Adv.
    # 2024's 1218 e*a0 for Rb 39D5/2->40P3/2 is the 'stretched' convention,
    # ~29% above the NIST-pi value for the SAME transition — a difference
    # that propagates directly into field inversion, so it is a design axis,
    # never an implicit choice.
    dipole_convention: str = "nist_pi"
    # drive
    coupling_rabi_hz: float = 5e6
    probe_rabi_hz: float = 50e3
    probe_power_w: float = 100e-6
    # cell / environment
    temperature_k: float = 300.0
    cell_length_m: float = 0.05
    beam_waist_m: float = 1e-3
    # readout hardware
    rin_db_per_hz: float = -150.0
    detector_nep_w_per_rthz: float = 1e-12
    # laser frequency noise (Lorentzian FWHM). Published vapor-cell superhets
    # attribute their floor largely to this term plus transit noise (Jing
    # 2020); spec 08 §3.5 defaults are UNVERIFIED vendor-typical.
    probe_linewidth_hz: float = 100e3
    coupling_linewidth_hz: float = 100e3
    # Probe lock offset from line centre. At EXACTLY 0 the transmission is at
    # a stationary point, so first-order frequency-noise coupling vanishes by
    # symmetry (dP/dnu = 0) — an idealization no real lock achieves. Set a
    # realistic offset to see the frequency-noise term that published work
    # identifies as dominant.
    probe_detuning_hz: float = 0.0
    # Rydberg dephasing: still a lab-technical input (laser linewidths,
    # stray fields, collisions) — NOT derived from first principles.
    rydberg_dephasing_hz: float = 100e3
    # Rydberg population decay: used ONLY when no state is named. With a
    # real state the rate comes from rydsim.lifetimes (Beterov radiative +
    # BBR); see rydberg_decay_hz().
    rydberg_decay_hz: float = 2e3

    def key(self) -> tuple:
        return dataclasses.astuple(self)


@dataclass
class Evaluation:
    """Objectives + validity for one design. All SI unless noted."""

    design: SensorDesign
    feasible: bool
    reason: str = ""
    # objectives (None when infeasible)
    nef: float | None = None                 # (V/m)/sqrt(Hz), TOTAL incl. SQL
    nef_technical: float | None = None       # shot + RIN + detector only
    nef_sql: float | None = None             # atom-projection-noise floor
    n_eff: float | None = None               # atoms participating (computed)
    f_vel: float | None = None               # velocity participation fraction
    tau_coh: float | None = None             # coherence time used for the SQL [s]
    below_sql: bool = False                  # technical noise under the SQL floor
    ibw_hz: float | None = None              # instantaneous bandwidth [Hz]
    e_lo: float | None = None                # optimal LO field [V/m]
    slope: float | None = None               # dP/dE [W/(V/m)]
    e_1db: float | None = None               # 1 dB compression [V/m]
    dynamic_range_db: float | None = None
    noise_temp_k: float | None = None
    noise_figure_db: float | None = None
    rf_frequency_hz: float | None = None
    rf_sign: int = 0                         # +1 partner above, -1 below
    dipole_cm: float | None = None           # RF dipole used [C m]
    confidence_taint: str = "UNKNOWN"
    noise_budget: dict = field(default_factory=dict)

    @property
    def nef_nv_per_cm_rthz(self) -> float | None:
        return None if self.nef is None else self.nef / 100.0 * 1e9

    def objectives(self) -> dict[str, float]:
        """Minimization-form objective vector for a multi-objective search."""
        if not self.feasible:
            raise IntegrityError("infeasible design has no objectives")
        return {
            "nef": float(self.nef),
            "neg_ibw": -float(self.ibw_hz),
            "neg_dynamic_range": -float(self.dynamic_range_db or 0.0),
        }


def _rydberg_states(design: SensorDesign):
    """(Species, state1, state2) tuples for the atomic chain, or None."""
    if design.n is None or not design.state or not design.rf_partner:
        return None
    from .atom import SPECIES

    sp = SPECIES.get(design.species)
    if sp is None:
        raise IntegrityError(
            f"unknown species {design.species!r}; known: {sorted(SPECIES)}")
    l1, j1 = parse_state(design.state)
    l2, j2 = parse_state(design.rf_partner)
    n2 = design.rf_partner_n if design.rf_partner_n is not None else design.n + 1
    return sp, (design.n, l1, j1), (n2, l2, j2)


def _resolve_dipole(design: SensorDesign) -> tuple[float, str]:
    """(dipole [C m], confidence taint) from the validated atomic chain.

    Refuses (IntegrityError) rather than guessing when neither a computed
    dipole nor an explicit override is available — the DESIGNER firewall's
    first line. Every refusal raised by rydsim.dipoles/radial (E1-forbidden
    pairs, low-n floors, failed cross-method consensus) propagates.
    """
    states = _rydberg_states(design)
    if states is not None:
        from .dipoles import mu_rf

        sp, s1, s2 = states
        d = mu_rf(sp, s1, s2, convention=design.dipole_convention)
        return d.value_Cm, f"{d.confidence_floor} [{d.convention}]"
    if design.rf_dipole_ea0 is not None:
        return design.rf_dipole_ea0 * AU_DIPOLE, "LITERATURE-RECALL (explicit override)"
    raise IntegrityError(
        "no RF dipole available: give (n, state, rf_partner) for a computed "
        "dipole, or an explicit rf_dipole_ea0 — this function will not guess")


def _rf_frequency(design: SensorDesign) -> tuple[float, int] | None:
    """(|RF frequency| [Hz], sign) from rydsim.atom.

    sign = +1 if the partner state lies above, -1 below; it drives AT
    sideband asymmetry downstream, so it is carried, not discarded.
    """
    states = _rydberg_states(design)
    if states is None:
        return None
    from .atom import rf_transition_hz

    sp, (n1, l1, j1), (n2, l2, j2) = states
    freq, sign = rf_transition_hz(sp, n1, l1, j1, n2, l2, j2)
    return float(freq), int(sign)


def rydberg_decay_hz(design: SensorDesign) -> tuple[float, str]:
    """(Rydberg population decay rate [Hz], provenance) for this design.

    Computed from `rydsim.lifetimes` (Beterov radiative + BBR fits) whenever
    the design names a real state — the rate scales roughly as n*^-3, so
    holding it fixed across an n-sweep suppresses a term that OPPOSES the
    n^2 dipole gain. Falls back to the design's explicit
    `rydberg_decay_hz` only when no state is named.
    """
    if design.n is None or not design.state:
        return design.rydberg_decay_hz, "fixed default (no state named)"
    from .atom import SPECIES
    from .lifetimes import effective_lifetime_fit

    sp = SPECIES[design.species]
    l, j = parse_state(design.state)
    tau_s = float(effective_lifetime_fit(sp, design.n, l, j,
                                         design.temperature_k))
    return 1.0 / (2 * np.pi * tau_s), "Beterov fit (radiative + BBR)"


def species_cell_parameters(design: SensorDesign) -> dict:
    """Vapor-cell parameters for the design's ACTUAL species (audit CRIT-1).

    Every species-dependent quantity the EIT/vapor engine needs — element,
    isotope fraction, atomic mass, probe and coupling wavelengths, and the
    intermediate-state linewidth — resolved from `rydsim.atom`. Omitting
    these silently simulates a Cs or Rb-85 design inside a Rb-87 cell.

    The coupling wavelength is computed per state via
    `atom.coupling_wavelength_m` (ruling R-15: lambda_c is state-dependent
    and must never be a hard-coded 480 nm).
    """
    from .atom import SPECIES, coupling_wavelength_m, element_symbol
    from .cell import NATURAL_ABUNDANCE
    from .constants import AMU

    sp = SPECIES[design.species]
    # single source of truth for species -> element (audit R10); never slice
    # the isotope name — that hack was duplicated in three modules
    element = element_symbol(sp)
    d2 = sp.d2
    if d2 is None:
        raise IntegrityError(
            f"{sp.name} has no D2 line data; cannot build a vapor cell "
            "configuration (refusing to substitute another species)")
    out = {
        "element": element,
        "isotope_fraction": NATURAL_ABUNDANCE.get(sp.name, sp.abundance),
        "mass": sp.mass_u * AMU,
        "lambda_probe": d2.lambda_vac_m,
        "gamma_e": d2.gamma_rad_s,
    }
    if design.n is not None and design.state:
        l, j = parse_state(design.state)
        out["lambda_coupling"] = float(
            coupling_wavelength_m(sp, "D2", design.n, l, j))
    return out


def _to_ladder_config(design: SensorDesign, dipole: float) -> LadderConfig:
    tau = 2 * np.pi
    gamma_r_hz, _ = rydberg_decay_hz(design)
    return LadderConfig(
        name=f"designer:{design.species}:{design.n}{design.state or ''}",
        delta_probe=tau * design.probe_detuning_hz,
        omega_probe=tau * design.probe_rabi_hz,
        omega_coupling=tau * design.coupling_rabi_hz,
        gamma_r=tau * gamma_r_hz,
        gamma_rp=tau * gamma_r_hz,
        deph_r=tau * design.rydberg_dephasing_hz,
        temperature=design.temperature_k,
        cell_length=design.cell_length_m,
        rf_dipole=dipole,
        **species_cell_parameters(design),
    )


def evaluate(design: SensorDesign, lo_points: int = 48,
             lo_rabi_span: tuple[float, float] = (0.02, 5.0)) -> Evaluation:
    """Evaluate one design through the validated engine.

    The LO search grid is defined in units of the coupling Rabi frequency
    (`lo_rabi_span` x Omega_c) and converted to field via E = hbar*Omega/d,
    so the search is SCALE-INVARIANT in the dipole moment. A grid fixed in
    absolute V/m would place the optimum outside the searchable range as d
    grows (d ~ n^2), making stronger atoms look worse — a gridding artifact
    that would silently corrupt any Pareto frontier across n.

    Returns an Evaluation; on any physics-validity failure (unconverged
    average, no usable transduction slope, missing dipole) returns
    feasible=False with a reason rather than a number. Never raises for
    ordinary infeasibility — search loops depend on that.
    """
    try:
        dipole, taint = _resolve_dipole(design)
    except IntegrityError as exc:
        return Evaluation(design=design, feasible=False, reason=str(exc))

    cfg = _to_ladder_config(design, dipole)
    e_to_omega = dipole / HBAR
    omega_c = 2 * np.pi * design.coupling_rabi_hz

    try:
        e_grid = np.geomspace(lo_rabi_span[0] * omega_c / e_to_omega,
                              lo_rabi_span[1] * omega_c / e_to_omega,
                              lo_points)
        p_curve = superhet_transfer(cfg, e_to_omega, e_grid,
                                    design.probe_power_w)
        from scipy.interpolate import CubicSpline

        transfer = CubicSpline(e_grid, p_curve)

        # laser frequency noise -> intensity noise through the EIT slope
        # (spec 08 §2.4.4). Evaluated across the LO grid so it enters the
        # LO optimization, not just the final report.
        d_nu = 2 * np.pi * 0.2e6                      # probe-detuning step [rad/s]
        cfg_p = dataclasses.replace(cfg, delta_probe=cfg.delta_probe + d_nu)
        cfg_m = dataclasses.replace(cfg, delta_probe=cfg.delta_probe - d_nu)
        p_plus = superhet_transfer(cfg_p, e_to_omega, e_grid, design.probe_power_w)
        p_minus = superhet_transfer(cfg_m, e_to_omega, e_grid, design.probe_power_w)
        dp_dnu = (p_plus - p_minus) / (2 * d_nu / (2 * np.pi))   # W/Hz
        dp_dnu_spline = CubicSpline(e_grid, dp_dnu)
        total_lw = design.probe_linewidth_hz + design.coupling_linewidth_hz

        def _freq_psd(e: float) -> float:
            from .superhet import frequency_noise_psd
            return frequency_noise_psd(float(dp_dnu_spline(e)), total_lw)

        op = optimize_lo(lambda e: float(transfer(e)), e_grid[2:-2],
                         cfg.lambda_probe,
                         rin_db_per_hz=design.rin_db_per_hz,
                         detector_nep_w_per_rthz=design.detector_nep_w_per_rthz,
                         extra_psd=_freq_psd)
    except IntegrityError as exc:
        return Evaluation(design=design, feasible=False,
                          reason=f"physics gate: {exc}")
    except (ValueError, np.linalg.LinAlgError) as exc:
        return Evaluation(design=design, feasible=False,
                          reason=f"evaluation failed: {exc}")

    # instantaneous bandwidth from the OBE resolvent at the LO operating point
    tau = 2 * np.pi
    sys = LadderSystem(
        omegas=[tau * design.probe_rabi_hz, tau * design.coupling_rabi_hz,
                op.e_lo * e_to_omega],
        deltas=[0.0, 0.0, 0.0],
        decays=[0.0, cfg.gamma_e, cfg.gamma_r, cfg.gamma_rp],
        dephasings=[0.0, 0.0, cfg.deph_r, cfg.deph_r],
    )
    try:
        ibw = ibw_3db(sys, 3, delta_max=tau * 100e6) / tau
    except (ValueError, np.linalg.LinAlgError) as exc:
        return Evaluation(design=design, feasible=False,
                          reason=f"IBW extraction failed: {exc}")

    # --- atom projection noise (SQL): the floor no technical improvement
    # can cross. Without it the objective is unphysical and a search will
    # optimize into fiction (this gate caught exactly that during D0).
    tau_coh = 1.0 / (cfg.gamma_r / 2 + cfg.deph_r + cfg.transit)
    f_vel = velocity_participation(cfg, op.e_lo * e_to_omega)
    v_int = np.pi * design.beam_waist_m**2 * design.cell_length_m
    n_eff = cfg.resolved_density() * v_int * f_vel   # f_state = 1 (optimistic)
    nef_sql = (sql_nef(dipole, n_eff, tau_coh) if n_eff > 0 else float("inf"))
    nef_total = float(np.hypot(op.nef, nef_sql))
    below_sql = op.nef < nef_sql

    e_min_1s = min_detectable_field(nef_total, 1.0)
    e_1db = compression_point(
        lambda e: float(transfer(e)), op.e_lo,
        np.linspace(0.02 * op.e_lo,
                    max(float(e_grid[-1]) - op.e_lo, 0.1 * op.e_lo), 100))
    dr_db = (20 * np.log10(e_1db / e_min_1s)) if e_1db else None

    rf_info = _rf_frequency(design)
    f_rf, rf_sign = (rf_info if rf_info else (None, 0))
    t_eq = nf_db = None
    if f_rf:
        t_eq, nf_db = noise_temperature(nef_total, f_rf)

    budget = op.budget.as_dict()
    budget["sql_field_referred_nef"] = nef_sql

    return Evaluation(
        design=design, feasible=True, nef=nef_total, nef_technical=op.nef,
        nef_sql=nef_sql, n_eff=n_eff, f_vel=f_vel, tau_coh=tau_coh,
        below_sql=below_sql, ibw_hz=ibw, e_lo=op.e_lo,
        slope=op.slope, e_1db=e_1db, dynamic_range_db=dr_db,
        noise_temp_k=t_eq, noise_figure_db=nf_db, rf_frequency_hz=f_rf,
        rf_sign=rf_sign, dipole_cm=dipole, confidence_taint=taint,
        noise_budget=budget,
    )


def evaluate_many(designs, **kw) -> list[Evaluation]:
    """Evaluate a batch; infeasible points are returned, not dropped, so the
    search layer can see (and report) what fraction of its proposals were
    physically invalid."""
    return [evaluate(d, **kw) for d in designs]


def pareto_front(evals: list[Evaluation],
                 keys: tuple[str, ...] = ("nef", "neg_ibw")) -> list[Evaluation]:
    """Non-dominated subset under minimization of the named objectives.

    Pure bookkeeping over oracle-computed numbers (no model, no inference) —
    safe to report directly.
    """
    ok = [e for e in evals if e.feasible]
    pts = [tuple(e.objectives()[k] for k in keys) for e in ok]
    front: list[Evaluation] = []
    for i, pi in enumerate(pts):
        dominated = any(
            all(pj[d] <= pi[d] for d in range(len(keys)))
            and any(pj[d] < pi[d] for d in range(len(keys)))
            for j, pj in enumerate(pts) if j != i
        )
        if not dominated:
            front.append(ok[i])
    return front
