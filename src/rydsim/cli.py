"""RydSim command-line interface.

    rydsim spectrum   — simulate an EIT/AT probe spectrum
    rydsim at         — AT-splitting field measurement (+ finding report)
    rydsim superhet   — LO optimization + NEF noise budget (+ finding report)
    rydsim validate   — run the scientific validation suite
    rydsim info       — constants, conventions, provenance policy
    rydsim gui        — launch the GreyNOC GUI

Config: every physics command accepts --config FILE.json (a LadderConfig)
and/or individual overrides; --save-config writes the effective config so
any run is reproducible from its artifact. Findings land in findings/.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import subprocess
import sys

import numpy as np

from . import __version__
from .constants import AU_DIPOLE, HBAR
from .experiment import (
    STANDING_CAVEATS,
    ATResult,
    LadderConfig,
    at_experiment,
    at_finding,
    converged_superhet_operating_point,
    spectrum,
)
from .provenance import Finding, IntegrityError
from .superhet import min_detectable_field

BANNER = f"GreyNOC RydSim v{__version__} — Rydberg electrometry simulator"
# Resolved at call time, not import time: in a frozen build the location
# depends on where the .exe lives and whether that path is writable
# (rydsim.paths). Import-time binding is what put findings in %TEMP%.
from .paths import findings_dir


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_config(args) -> LadderConfig:
    cfg = LadderConfig()
    if getattr(args, "config", None):
        data = json.loads(pathlib.Path(args.config).read_text(encoding="utf-8"))
        known = {f.name for f in dataclasses.fields(LadderConfig)}
        cfg = LadderConfig(**{k: v for k, v in data.items() if k in known})
        # TRUST BOUNDARY. `sources` is rendered into the "Constants on the
        # critical path" block of a GreyNOC-branded finding. A config file
        # received from a third party could therefore inject lines like
        # "NIST SP 1234 primary standard, VERIFIED against traceable
        # source" and have them appear as OUR provenance. Tag every
        # config-supplied entry so a reader cannot mistake it for a claim
        # RydSim stands behind. (Found by the pre-release security audit,
        # which generated exactly that finding.)
        if cfg.sources:
            cfg = dataclasses.replace(cfg, sources=[
                f"[UNVERIFIED — supplied by config file "
                f"{pathlib.Path(args.config).name}, not checked by RydSim] {s}"
                for s in cfg.sources])
    # numeric overrides given in lab units (MHz, K, nm, ea0) for humans
    mhz = 2 * np.pi * 1e6
    over = {
        "omega_probe": ("probe_rabi_mhz", mhz),
        "omega_coupling": ("coupling_rabi_mhz", mhz),
        "omega_rf": ("rf_rabi_mhz", mhz),
        "delta_probe": ("probe_detuning_mhz", mhz),
        "delta_coupling": ("coupling_detuning_mhz", mhz),
        "delta_rf": ("rf_detuning_mhz", mhz),
        "gamma_r": ("rydberg_decay_khz", 2 * np.pi * 1e3),
        "deph_r": ("rydberg_dephasing_khz", 2 * np.pi * 1e3),
        "temperature": ("temperature_k", 1.0),
    }
    for attr, (opt, scale) in over.items():
        v = getattr(args, opt, None)
        if v is not None:
            cfg = dataclasses.replace(cfg, **{attr: v * scale})
    if getattr(args, "rf_dipole_ea0", None) is not None:
        cfg = dataclasses.replace(cfg, rf_dipole=args.rf_dipole_ea0 * AU_DIPOLE)
    if getattr(args, "no_doppler", False):
        cfg = dataclasses.replace(cfg, doppler=False)
    if getattr(args, "name", None):
        cfg = dataclasses.replace(cfg, name=args.name)
    return cfg


def _maybe_save_config(args, cfg: LadderConfig) -> None:
    if getattr(args, "save_config", None):
        p = pathlib.Path(args.save_config)
        p.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
        print(f"[config] saved -> {p}")


def _write_finding(f: Finding, stem: str) -> None:
    out_dir = findings_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"{stem}-{f.config_hash}"
    base.with_suffix(".json").write_text(f.to_json(), encoding="utf-8")
    base.with_suffix(".md").write_text(f.to_markdown(), encoding="utf-8")
    print(f"[finding] {base.with_suffix('.md')}")


def _add_common_physics_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="LadderConfig JSON file")
    p.add_argument("--save-config", help="write effective config JSON here")
    p.add_argument("--name", help="config name label")
    p.add_argument("--probe-rabi-mhz", type=float, default=None)
    p.add_argument("--coupling-rabi-mhz", type=float, default=None)
    p.add_argument("--rf-rabi-mhz", type=float, default=None)
    p.add_argument("--probe-detuning-mhz", type=float, default=None)
    p.add_argument("--coupling-detuning-mhz", type=float, default=None)
    p.add_argument("--rf-detuning-mhz", type=float, default=None)
    p.add_argument("--rydberg-decay-khz", type=float, default=None)
    p.add_argument("--rydberg-dephasing-khz", type=float, default=None)
    p.add_argument("--temperature-k", type=float, default=None)
    p.add_argument("--rf-dipole-ea0", type=float, default=None,
                   help="RF transition dipole moment in units of e*a0")
    p.add_argument("--no-doppler", action="store_true",
                   help="cold-atom mode (no thermal averaging)")


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_spectrum(args) -> int:
    cfg = _load_config(args)
    _maybe_save_config(args, cfg)
    span = 2 * np.pi * args.span_mhz * 1e6
    dps = np.linspace(-span / 2, span / 2, args.points)
    resp = spectrum(cfg, dps, n_doppler_nodes=args.doppler_nodes)
    out = np.column_stack([dps / (2 * np.pi * 1e6), resp.real, resp.imag])
    header = ("probe_detuning_MHz absorption_norm dispersion_norm  "
              f"# rydsim {__version__} config={cfg.name}")
    if args.output:
        np.savetxt(args.output, out, header=header)
        print(f"[spectrum] {args.points} pts -> {args.output}")
    else:
        print(header)
        for row in out[:: max(1, len(out) // 40)]:
            bar = "#" * int(40 * max(row[1], 0) / max(out[:, 1].max(), 1e-30))
            print(f"{row[0]:+9.3f}  {row[1]:+.4e}  {bar}")
    return 0


def cmd_at(args) -> int:
    cfg = _load_config(args)
    _maybe_save_config(args, cfg)
    if cfg.omega_rf <= 0:
        print("error: AT experiment needs --rf-rabi-mhz > 0", file=sys.stderr)
        return 2
    res: ATResult = at_experiment(cfg, n_points=args.points,
                                  n_doppler_nodes=args.doppler_nodes)
    m = res.measurement
    print(BANNER)
    print(f"config: {cfg.name}  (doppler={'on' if cfg.doppler else 'off'}, "
          f"T={cfg.temperature} K)")
    if m is None:
        print("result: NO RESOLVED DOUBLET — field below the AT-resolvable "
              "regime for this configuration. (This is the honest answer; "
              "use superhet mode for weak fields.)")
        return 0
    print(f"AT splitting (probe axis): {m.splitting/(2*np.pi)/1e6:.4f} MHz "
          f"[{m.method}, resolved={m.resolved}]")
    print(f"wavelength-mismatch factor: {res.mismatch_factor:.4f}")
    omega = m.splitting / res.mismatch_factor
    print(f"RF Rabi frequency:         {omega/(2*np.pi)/1e6:.4f} MHz")
    if res.field_v_per_m is not None:
        print(f"E-field (E = hbar*Omega/d): {res.field_v_per_m:.6g} V/m")
    elif cfg.rf_dipole is None:
        print("E-field: (provide --rf-dipole-ea0 to invert to V/m)")
    if args.finding:
        _write_finding(at_finding(cfg, res), "at-measurement")
    return 0


def cmd_superhet(args) -> int:
    cfg = _load_config(args)
    _maybe_save_config(args, cfg)
    if cfg.rf_dipole is None:
        if args.rf_dipole_ea0 is None:
            print("error: superhet needs --rf-dipole-ea0", file=sys.stderr)
            return 2
    d = cfg.rf_dipole
    e_to_omega = d / HBAR
    print(BANNER)
    print(f"[superhet] log-spaced LO scan E in [{args.e_min}, {args.e_max}] "
          f"V/m from {args.lo_points} points, refined to convergence ...")
    try:
        conv = converged_superhet_operating_point(
            cfg, e_to_omega, args.e_max, args.probe_power_w,
            e_min=args.e_min, n_points=args.lo_points,
            rin_db_per_hz=args.rin_db, detector_nep_w_per_rthz=args.nep)
    except IntegrityError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 3
    op = conv.op
    transfer = conv.spline()
    print(f"LO grid converged at {conv.n_points} points "
          f"(NEF change on last doubling {conv.nef_rel_change:.2%})")
    nef_cm = op.nef / 100.0  # V/m -> V/cm
    print(f"optimal LO field:   {op.e_lo:.4g} V/m "
          f"(Omega_LO = {op.e_lo*e_to_omega/(2*np.pi)/1e6:.3f} MHz)")
    print(f"transduction slope: {op.slope:.4g} W/(V/m)")
    print(f"noise budget [W^2/Hz]: shot={op.budget.shot:.3g} "
          f"rin={op.budget.rin:.3g} detector={op.budget.detector:.3g}")
    print(f"NEF: {op.nef:.4g} (V/m)/rtHz  =  {nef_cm*1e9:.4g} nV/cm/rtHz")
    e_min_1s = min_detectable_field(op.nef, 1.0)
    print(f"min detectable field (1 s, SNR=1): {e_min_1s:.4g} V/m")
    from .superhet import compression_point
    c1db = compression_point(lambda e: float(transfer(e)), op.e_lo,
                             np.linspace(0.02 * op.e_lo, args.e_max - op.e_lo,
                                         120))
    if c1db is not None:
        dr_db = 20 * np.log10(c1db / e_min_1s)
        print(f"1 dB compression:   E_sig = {c1db:.4g} V/m")
        print(f"dynamic range (1 s): {dr_db:.1f} dB")
    else:
        c1db = dr_db = None
        print("1 dB compression: not reached within the scanned range")
    if args.finding:
        f = Finding(
            title=f"Superhet LO optimization [{cfg.name}]",
            result={
                "optimal E_LO": f"{op.e_lo:.6g} V/m",
                "NEF": f"{nef_cm*1e9:.4g} nV/cm/rtHz",
                "noise budget (W^2/Hz)": op.budget.as_dict(),
                "slope": f"{op.slope:.4g} W/(V/m)",
                "1dB compression": (f"{c1db:.4g} V/m" if c1db else "not reached"),
                "dynamic range (1 s)": (f"{dr_db:.1f} dB" if c1db else "n/a"),
                "number density": f"{cfg.resolved_density():.4g} m^-3",
            },
            config={**cfg.to_dict(),
                    "probe_power_w": args.probe_power_w,
                    "rin_db": args.rin_db, "nep": args.nep,
                    "lo_grid": [args.e_min, args.e_max, args.lo_points]},
            uncertainty_budget={
                "LO grid convergence": (
                    f"NEF moved {conv.nef_rel_change:.2%}, E_LO moved "
                    f"{conv.e_lo_rel_change:.2%} on the last grid doubling "
                    f"({conv.n_points} log-spaced points)"),
                "transfer-curve model": "optical-depth chain pending spec 05 "
                "integration; slope-normalized NEF insensitive to overall scale",
            },
            constants_used=list(cfg.sources),
            caveats=["Technical-noise realism limited to stationary RIN/NEP model."],
        ).finalize(__version__)
        _write_finding(f, "superhet-nef")
    return 0


def cmd_cosite(args) -> int:
    """T3: hidden weak signal beside a strong co-channel interferer."""
    from . import cosite as cs
    from .provenance import CrossCheck

    cfg = _load_config(args)
    _maybe_save_config(args, cfg)
    if cfg.rf_dipole is None:
        print("error: cosite needs --rf-dipole-ea0", file=sys.stderr)
        return 2
    if not 0 < args.ei_min_frac < args.ei_max_frac < 1:
        print("error: need 0 < --ei-min-frac < --ei-max-frac < 1",
              file=sys.stderr)
        return 2
    if args.ei_points < 2 or args.integration_s <= 0 or args.harmonics < 1:
        print("error: need --ei-points >= 2, --integration-s > 0, "
              "--harmonics >= 1", file=sys.stderr)
        return 2
    e_to_omega = cfg.rf_dipole / HBAR
    f_i = args.interferer_offset_khz * 1e3
    f_s = args.signal_offset_khz * 1e3
    print(BANNER)
    try:
        conv = converged_superhet_operating_point(
            cfg, e_to_omega, args.e_max, args.probe_power_w,
            e_min=args.e_min, n_points=args.lo_points,
            rin_db_per_hz=args.rin_db, detector_nep_w_per_rthz=args.nep)
        e_lo = conv.op.e_lo
        ibw = cs.instantaneous_bandwidth_hz(cfg, e_lo, e_to_omega)
        cs.require_quasi_static([f_i, f_s], ibw)
        cs.require_no_spur_collision(f_s, f_i, args.harmonics)
        spl = conv.spline()
        dspl = spl.derivative()
        domain = (0.0, float(conv.e_grid[-1]))
        e_i_grid = np.geomspace(args.ei_min_frac, args.ei_max_frac,
                                args.ei_points) * e_lo
        pts = cs.cosite_scan(spl, dspl, e_lo, e_i_grid, domain,
                             cfg.lambda_probe, rin_db_per_hz=args.rin_db,
                             detector_nep_w_per_rthz=args.nep,
                             integration_time=args.integration_s)
        mid = pts[len(pts) // 2]
        xc = CrossCheck(
            quantity=f"hidden-signal gain at E_I = {mid.e_i:.4g} V/m",
            methods={
                "phase-average": mid.gain,
                "two-tone DFT": cs.two_tone_fft_gain(
                    spl, e_lo, mid.e_i, 1e-5 * e_lo, domain),
            },
            unit="W/(V/m)", tolerance_rel=1e-4)
        xc.require()
        # Interpolation sensitivity: same physics, half the transfer samples.
        from scipy.interpolate import CubicSpline
        coarse = CubicSpline(conv.e_grid[::2], conv.p_curve[::2])
        g_coarse = cs.small_signal_gain(coarse, coarse.derivative(), e_lo,
                                        mid.e_i, domain).gain
    except IntegrityError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 3
    interp_rel = abs(g_coarse - mid.gain) / mid.gain

    g0 = pts[0].gain / 10 ** (-pts[0].desense_db / 20)
    e_min0 = (np.sqrt(conv.op.budget.total) / abs(conv.op.slope)
              / np.sqrt(args.integration_s))
    b1 = cs.blocking_level(pts, 1.0)
    b3 = cs.blocking_level(pts, 3.0)
    law = cs.small_interferer_law(pts, e_lo)
    law_txt = (f"desense ~= {law[1]:.4g} dB x (E_I/E_LO)^{law[0]:.3f}"
               if law else "not enough weak-interferer points to fit")
    e_spur = b1 if b1 is not None else pts[-1].e_i
    spurs = cs.interferer_spurs(spl, e_lo, e_spur, domain, args.harmonics)
    spur_db = [20 * np.log10(max(s / g0, 1e-300) / e_min0) for s in spurs]

    print(f"LO: {e_lo:.4g} V/m   NEF(no interferer): "
          f"{conv.op.nef / 100 * 1e9:.4g} nV/cm/rtHz   IBW: {ibw / 1e3:.4g} kHz")
    print(f"offsets: interferer {f_i / 1e3:g} kHz, hidden signal "
          f"{f_s / 1e3:g} kHz (quasi-static limit "
          f"{cs.QUASI_STATIC_MAX_FRACTION * ibw / 1e3:.4g} kHz)")
    print(f"\n{'E_I [V/m]':>11} {'desense dB':>11} {'NEF nV/cm/rtHz':>15} "
          f"{'E_min [V/m]':>12} {'hidden below E_I dB':>20}")
    for p in pts:
        print(f"{p.e_i:11.4g} {p.desense_db:11.3f} {p.nef / 100 * 1e9:15.4g} "
              f"{p.e_min:12.4g} {p.hidden_below_interferer_db:20.1f}")
    fmt = (lambda b: f"{b:.4g} V/m ({b / e_lo:.3f} x E_LO)"
           if b is not None else "not reached in scan")
    print(f"\nweak-interferer law: {law_txt}")
    print(f"1 dB blocking: {fmt(b1)}\n3 dB blocking: {fmt(b3)}")
    print(f"interferer spurs at E_I = {e_spur:.4g} V/m "
          "(field-equivalent, dB above the no-interferer floor):")
    for n, s in enumerate(spur_db, 1):
        print(f"  {n} x {f_i / 1e3:g} kHz: {s:+.1f} dB")
    print(f"cross-check: phase-average vs two-tone DFT agree to "
          f"{xc.spread_rel:.1e}; transfer-curve interpolation sensitivity "
          f"{interp_rel:.1e}")

    if args.finding:
        best = max(pts, key=lambda p: p.hidden_below_interferer_db)
        f = Finding(
            title=f"T3 co-site: hidden signal beside a strong interferer [{cfg.name}]",
            result={
                "LO operating point": f"{e_lo:.6g} V/m",
                "NEF, no interferer": f"{conv.op.nef / 100 * 1e9:.4g} nV/cm/rtHz",
                "instantaneous bandwidth (resolvent, Doppler-free)": f"{ibw:.4g} Hz",
                "interferer / signal offsets": f"{f_i:.6g} Hz / {f_s:.6g} Hz",
                "weak-interferer desensitization law": law_txt,
                "1 dB blocking level": fmt(b1),
                "3 dB blocking level": fmt(b3),
                "largest hidden-below-interferer margin in scan": (
                    f"{best.hidden_below_interferer_db:.1f} dB at "
                    f"E_I = {best.e_i:.4g} V/m ({args.integration_s:g} s, SNR 1)"),
                "scan (E_I | desense | NEF | E_min | hidden below E_I)": [
                    f"{p.e_i:.4g} V/m | {p.desense_db:.4f} dB | "
                    f"{p.nef / 100 * 1e9:.4g} nV/cm/rtHz | {p.e_min:.4g} V/m | "
                    f"{p.hidden_below_interferer_db:.1f} dB"
                    for p in pts],
                f"interferer spurs at E_I = {e_spur:.4g} V/m "
                "(dB above no-interferer floor)": {
                    f"{n} x {f_i:.6g} Hz": round(float(s), 2)
                    for n, s in enumerate(spur_db, 1)},
            },
            config={**cfg.to_dict(), "probe_power_w": args.probe_power_w,
                    "rin_db": args.rin_db, "nep": args.nep,
                    "lo_scan": [args.e_min, args.e_max, args.lo_points],
                    "interferer_offset_hz": f_i, "signal_offset_hz": f_s,
                    "ei_frac": [args.ei_min_frac, args.ei_max_frac,
                                args.ei_points],
                    "integration_s": args.integration_s,
                    "harmonics": args.harmonics},
            uncertainty_budget={
                "LO grid convergence": (
                    f"NEF moved {conv.nef_rel_change:.2%} on the last "
                    f"doubling ({conv.n_points} points)"),
                "transfer-curve interpolation": (
                    f"gain changes {interp_rel:.1e} (relative) when the "
                    "transfer curve is resampled at half density"),
                "theta quadrature": "converged to 1e-9 relative by node "
                                    "doubling at every scan point",
                "quasi-static model": (
                    f"offsets held below {cs.QUASI_STATIC_MAX_FRACTION:g} x IBW, "
                    "where the blocking-level error vs the exact Floquet "
                    "solver is <= ~0.3% (scales as 7 (Delta/IBW)^2)"),
            },
            cross_checks=[{"quantity": xc.quantity, "methods": xc.methods,
                           "spread_rel": xc.spread_rel,
                           "tolerance_rel": xc.tolerance_rel}],
            constants_used=list(cfg.sources),
            validation_state="see rydsim validate (tests/test_cosite.py, "
                             "tests/test_lo_convergence.py cover this command)",
            caveats=list(STANDING_CAVEATS) + [
                "Interferer and hidden signal are assumed to drive the same "
                "Rydberg RF transition as the LO (co-channel). Out-of-band "
                "interferers beyond the atomic response bandwidth are refused, "
                "not modeled: they need a time-domain / Floquet OBE solver.",
                "IBW from the single-velocity-class OBE resolvent; Doppler "
                "averaging of the response bandwidth is not applied.",
                "Noise budget is shot + RIN + detector at the interferer-"
                "shifted mean probe power; laser frequency noise is omitted "
                "here (included in objective.evaluate, not in this command).",
                "Signal offsets on an interferer harmonic are refused; the "
                "spur table gives the masking level there instead.",
            ],
        ).finalize(__version__)
        _write_finding(f, "t3-cosite")
    return 0


def cmd_selectivity(args) -> int:
    """T3: blocking level vs interferer offset, in and out of band."""
    from . import floquet as fq
    from .lindblad import LadderSystem
    from .provenance import CrossCheck

    cfg = _load_config(args)
    _maybe_save_config(args, cfg)
    if cfg.doppler:
        print("error: selectivity uses the single-velocity-class Floquet "
              "solver; pass --no-doppler to acknowledge that (Doppler "
              "averaging is not implemented for it)", file=sys.stderr)
        return 2
    if (args.offsets < 2 or args.offset_min_khz <= 0
            or args.offset_max_mhz * 1e3 <= args.offset_min_khz
            or args.signal_offset_khz <= 0 or args.ei_max_ratio <= 0.01
            or args.ei_points < 2):
        print("error: need --offsets >= 2, 0 < offset-min < offset-max, "
              "--signal-offset-khz > 0, --ei-max-ratio > 0.01, "
              "--ei-points >= 2", file=sys.stderr)
        return 2
    tau = 2 * np.pi
    base = LadderSystem(
        omegas=[cfg.omega_probe, cfg.omega_coupling, max(cfg.omega_rf, 1.0)],
        deltas=[cfg.delta_probe, cfg.delta_coupling, cfg.delta_rf],
        decays=[0.0, cfg.gamma_e, cfg.gamma_r, cfg.gamma_rp],
        dephasings=[0.0, 0.0, cfg.deph_r, cfg.deph_r], transit=cfg.transit)
    print(BANNER)
    try:
        if cfg.omega_rf > 0:
            w_l = cfg.omega_rf
            lo_note = "user-set (--rf-rabi-mhz)"
        else:
            w_l = fq.optimal_thin_lo(base, 4 * cfg.omega_coupling)
            lo_note = "steepest-slope point of the thin transfer curve"
        om = base.omegas.copy()
        om[2] = w_l
        base = LadderSystem(omegas=om, deltas=base.deltas, decays=base.decays,
                            dephasings=base.dephasings, transit=base.transit)
        solver = fq.TwoToneFloquet(base)
        offsets = tau * np.geomspace(args.offset_min_khz * 1e3,
                                     args.offset_max_mhz * 1e6, args.offsets)
        grid = np.geomspace(0.01, args.ei_max_ratio, args.ei_points) * w_l
        d_s = tau * args.signal_offset_khz * 1e3
        d_dc = tau * 10.0
        curve = fq.selectivity_curve(solver, offsets, d_s, grid)
        curve_dc = fq.selectivity_curve(solver, offsets, d_dc, grid)
        s_crit = fq.ac_stark_shift_crit(base)
        checks: list[CrossCheck] = []
        far = curve_dc[-1]
        if far.blocking_rabi is not None:
            checks.append(CrossCheck(
                quantity=(f"far-offset blocking at {far.offset / tau / 1e6:.4g} "
                          "MHz (signal -> DC)"),
                methods={"Floquet": far.blocking_rabi / w_l,
                         "AC-Stark law": np.sqrt(4 * far.offset * s_crit) / w_l},
                unit="x Omega_LO", tolerance_rel=0.05))
        # In-band anchor: at 0.005 x IBW the quasi-static error is ~1e-4
        # (it scales as (Delta/IBW)^2), so Floquet must reproduce the
        # independent phase-average blocking level there.
        from .lindblad import ibw_3db
        ibw = ibw_3db(base, 3, delta_max=tau * 100e6) / tau
        near_off = tau * 0.005 * ibw
        near, _ = fq.blocking_rabi(solver, near_off, d_dc, grid)
        qs = fq.quasi_static_thin_blocking(base, grid)
        if near is not None and qs is not None:
            checks.append(CrossCheck(
                quantity=(f"in-band blocking at {near_off / tau / 1e3:.4g} kHz "
                          "= 0.005 x IBW (signal -> DC)"),
                methods={"Floquet": near / w_l,
                         "quasi-static phase average": qs / w_l},
                unit="x Omega_LO", tolerance_rel=2e-3))
        for c in checks:
            c.require()
    except IntegrityError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 3

    d_si = cfg.rf_dipole
    to_field = (lambda w: HBAR * w / d_si) if d_si else None

    def cell(w: float | None) -> str:
        return f"{w / w_l:9.4f}" if w is not None else f"{'>' + format(args.ei_max_ratio, 'g'):>9}"

    print(f"LO: Omega = {w_l / tau / 1e6:.4g} MHz ({lo_note})"
          + (f" = {to_field(w_l):.4g} V/m" if to_field else ""))
    print(f"atomic response bandwidth (resolvent): {ibw / 1e3:.4g} kHz; "
          "offsets below 0.02 x IBW are quasi-static")
    print(f"hidden signal at +{args.signal_offset_khz:g} kHz; 'DC' column "
          "uses +10 Hz (the AC-Stark-law limit)")
    print(f"\n{'offset':>12} {'1dB block':>9} {'1dB block':>9} {'AC-Stark':>9}  "
          f"(x Omega_LO)\n{'':>12} {'@signal':>9} {'@DC':>9} {'law':>9}")
    rows = []
    for p, q in zip(curve, curve_dc):
        law = np.sqrt(4 * p.offset * s_crit)
        f_txt = (f"{p.offset / tau / 1e6:9.4g} MHz")
        print(f"{f_txt:>12} {cell(p.blocking_rabi)} {cell(q.blocking_rabi)} "
              f"{law / w_l:9.4f}")
        rows.append((p, q, law))
    hi = [(p.offset, p.blocking_rabi) for p in curve[len(curve) * 3 // 4:]
          if p.blocking_rabi is not None]
    slope_txt = "not enough far-offset points"
    if len(hi) >= 2:
        k = np.polyfit(np.log10([o for o, _ in hi]),
                       np.log10([b for _, b in hi]), 1)[0]
        slope_txt = (f"{20 * k:.2f} dB/decade in field "
                     f"(exponent {k:.3f}; AC-Stark limit 0.5 = 10 dB/decade)")
    print(f"\nfar-offset rejection: {slope_txt}")
    print(f"AC-Stark critical light shift: s_crit = {s_crit / tau / 1e3:.4g} kHz "
          "-> Omega_block = sqrt(4 Delta s_crit)")
    for c in checks:
        print(f"cross-check: {c.quantity}: methods agree to {c.spread_rel:.2%}")

    if args.finding:
        def fmt(w: float | None) -> str:
            if w is None:
                return f"> {args.ei_max_ratio:g} x Omega_LO"
            out = f"{w / w_l:.4g} x Omega_LO"
            return out + (f" ({to_field(w):.4g} V/m)" if to_field else "")
        f = Finding(
            title=f"T3 selectivity: blocking vs interferer offset [{cfg.name}]",
            result={
                "LO": (f"Omega_LO = {w_l / tau / 1e6:.4g} MHz, {lo_note}"
                       + (f", {to_field(w_l):.4g} V/m" if to_field else "")),
                "hidden-signal offset": f"{args.signal_offset_khz:g} kHz",
                "atomic response bandwidth (resolvent)": f"{ibw:.4g} Hz",
                "far-offset rejection": slope_txt,
                "AC-Stark critical shift": (
                    f"s_crit = {s_crit / tau / 1e3:.4g} kHz; asymptotic "
                    "1 dB blocking Omega_I = sqrt(4 Delta s_crit)"),
                "selectivity (offset | block @signal | block @DC | AC-Stark law)": [
                    f"{p.offset / tau / 1e6:.4g} MHz | {fmt(p.blocking_rabi)} | "
                    f"{fmt(q.blocking_rabi)} | {law / w_l:.4g} x Omega_LO"
                    for p, q, law in rows],
            },
            config={**cfg.to_dict(), "omega_lo": w_l,
                    "offset_min_khz": args.offset_min_khz,
                    "offset_max_mhz": args.offset_max_mhz,
                    "offsets": args.offsets,
                    "signal_offset_khz": args.signal_offset_khz,
                    "ei_max_ratio": args.ei_max_ratio,
                    "ei_points": args.ei_points},
            uncertainty_budget={
                "Floquet truncation": "sideband count doubled until the gain "
                                      "moved < 1e-8 (relative) at every point",
                "blocking interpolation": (
                    f"log-interpolated on a {args.ei_points}-point log grid "
                    f"from 0.01 to {args.ei_max_ratio:g} x Omega_LO"),
                "time-domain agreement": "harmonic balance vs direct OBE "
                                         "integration agrees to ~1e-8 "
                                         "(tests/test_floquet.py)",
            },
            cross_checks=[{"quantity": c.quantity, "methods": c.methods,
                           "spread_rel": c.spread_rel,
                           "tolerance_rel": c.tolerance_rel} for c in checks],
            constants_used=list(cfg.sources),
            validation_state="see rydsim validate (tests/test_floquet.py)",
            caveats=list(STANDING_CAVEATS) + [
                "Single velocity class: no Doppler averaging. Hot-vapor "
                "selectivity will differ.",
                "Optically thin readout (probe absorption ~ Im rho_10): absolute "
                "blocking levels differ from the thick-medium `cosite` ones; "
                "the offset dependence is what this finding claims.",
                "Gain desensitization only: the noise budget is not "
                "re-evaluated (constant to first order in the thin limit).",
                "4-level ladder: interferers far enough off to reach OTHER "
                "Rydberg transitions are outside the model. Check the real "
                "level structure before trusting GHz offsets.",
                "Blocking depends on the hidden signal's own offset: the "
                "interferer's light shift opens a slow (Rydberg-decay-scale) "
                "response mode, so a kHz-offset signal is blocked earlier than "
                "a near-DC one. Both columns are reported.",
            ],
        ).finalize(__version__)
        _write_finding(f, "t3-selectivity")
    return 0


def cmd_validate(args) -> int:
    print(BANNER)
    if getattr(sys, "frozen", False):
        # The portable binary bundles the ENGINE, not the validation suite:
        # tests/ and pytest are deliberately excluded (shipping a test
        # runner inside a distributed artifact widens its attack surface
        # for no user benefit). Refuse clearly rather than emitting a
        # misleading failure — a validation command that cannot validate
        # must say so, not return a red herring.
        print("error: 'validate' is unavailable in the portable build.\n"
              "  The validation suite is not bundled. Run it from a source\n"
              "  checkout instead:\n"
              "      pip install -e .[dev] && python -m pytest tests -q\n"
              f"  This binary reports itself as rydsim {__version__}; the\n"
              "  suite result for that version is recorded in the release\n"
              "  notes and in docs/AUDIT-2026-08-10.md.", file=sys.stderr)
        return 2
    root = pathlib.Path(__file__).resolve().parents[2]
    tests = root / "tests"
    if not tests.is_dir():
        print(f"error: no tests directory at {tests} — run from a source "
              "checkout.", file=sys.stderr)
        return 2
    cmd = [sys.executable, "-m", "pytest", str(tests),
           "-v" if args.verbose else "-q", "--tb=short"]
    r = subprocess.run(cmd, cwd=root)
    return r.returncode


def cmd_info(args) -> int:
    from . import constants as c
    print(BANNER)
    print("\nConventions: docs/spec/00-conventions.md")
    print("Integrity:   docs/spec/00-integrity-audit.md — no fabrication; "
          "every constant sourced; findings carry provenance.")
    print(f"\nCODATA (via scipy.constants):")
    print(f"  a0        = {c.A0:.12e} m")
    print(f"  R_inf     = {c.RYD_INF:.6f} 1/m")
    print(f"  e*a0      = {c.AU_DIPOLE:.12e} C m")
    print(f"  E_h/(e a0)= {c.AU_EFIELD:.6e} V/m")
    from .paths import is_frozen
    print(f"\nFindings dir: {findings_dir()}"
          + ("  (portable build: beside the executable)" if is_frozen() else ""))
    return 0


def cmd_gui(args) -> int:
    from .gui.app import main as gui_main
    return gui_main()


# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rydsim", description=BANNER)
    p.add_argument("--version", action="version", version=f"rydsim {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    ps = sub.add_parser("spectrum", help="simulate EIT/AT probe spectrum")
    _add_common_physics_args(ps)
    ps.add_argument("--span-mhz", type=float, default=40.0)
    ps.add_argument("--points", type=int, default=401)
    ps.add_argument("--doppler-nodes", type=int, default=64)
    ps.add_argument("-o", "--output", help="write spectrum data file")
    ps.set_defaults(func=cmd_spectrum)

    pa = sub.add_parser("at", help="AT-splitting field measurement")
    _add_common_physics_args(pa)
    pa.add_argument("--points", type=int, default=401)
    pa.add_argument("--doppler-nodes", type=int, default=64)
    pa.add_argument("--finding", action="store_true",
                    help="write a provenance-tagged finding report")
    pa.set_defaults(func=cmd_at)

    ph = sub.add_parser("superhet", help="superhet LO optimization + NEF")
    _add_common_physics_args(ph)
    ph.add_argument("--probe-power-w", type=float, default=100e-6)
    ph.add_argument("--rin-db", type=float, default=-150.0,
                    help="laser RIN [dB/Hz]")
    ph.add_argument("--nep", type=float, default=1e-12,
                    help="detector NEP [W/rtHz]")
    ph.add_argument("--e-min", type=float, default=0.002,
                    help="lower end of the log-spaced LO scan [V/m]")
    ph.add_argument("--e-max", type=float, default=20.0,
                    help="upper end of the LO scan [V/m]")
    ph.add_argument("--lo-points", type=int, default=96,
                    help="initial log-spaced LO points; doubled until the "
                         "NEF converges to 1%%")
    ph.add_argument("--doppler-nodes", type=int, default=24)
    ph.add_argument("--finding", action="store_true")
    ph.set_defaults(func=cmd_superhet)

    pc = sub.add_parser(
        "cosite", help="T3: hidden weak signal beside a strong interferer")
    _add_common_physics_args(pc)
    pc.add_argument("--probe-power-w", type=float, default=100e-6)
    pc.add_argument("--rin-db", type=float, default=-150.0)
    pc.add_argument("--nep", type=float, default=1e-12)
    pc.add_argument("--e-min", type=float, default=0.002)
    pc.add_argument("--e-max", type=float, default=20.0)
    pc.add_argument("--lo-points", type=int, default=96)
    pc.add_argument("--interferer-offset-khz", type=float, default=50.0,
                    help="interferer offset from the LO [kHz]")
    pc.add_argument("--signal-offset-khz", type=float, default=13.0,
                    help="hidden-signal offset from the LO [kHz]")
    pc.add_argument("--ei-min-frac", type=float, default=1e-3,
                    help="smallest interferer amplitude, as a fraction of E_LO")
    pc.add_argument("--ei-max-frac", type=float, default=0.9,
                    help="largest interferer amplitude, as a fraction of E_LO")
    pc.add_argument("--ei-points", type=int, default=25)
    pc.add_argument("--integration-s", type=float, default=1.0)
    pc.add_argument("--harmonics", type=int, default=5,
                    help="interferer harmonics to report as spurs")
    pc.add_argument("--finding", action="store_true")
    pc.set_defaults(func=cmd_cosite)

    pl = sub.add_parser(
        "selectivity",
        help="T3: blocking level vs interferer offset (Floquet, cold)")
    _add_common_physics_args(pl)
    pl.add_argument("--offset-min-khz", type=float, default=50.0)
    pl.add_argument("--offset-max-mhz", type=float, default=50.0)
    pl.add_argument("--offsets", type=int, default=16,
                    help="log-spaced interferer offsets")
    pl.add_argument("--signal-offset-khz", type=float, default=13.0)
    pl.add_argument("--ei-max-ratio", type=float, default=100.0,
                    help="largest interferer Rabi frequency, x Omega_LO")
    pl.add_argument("--ei-points", type=int, default=41)
    pl.add_argument("--finding", action="store_true")
    pl.set_defaults(func=cmd_selectivity)

    pv = sub.add_parser("validate", help="run the scientific validation suite")
    pv.add_argument("-v", "--verbose", action="store_true")
    pv.set_defaults(func=cmd_validate)

    pi = sub.add_parser("info", help="constants, conventions, provenance policy")
    pi.set_defaults(func=cmd_info)

    pg = sub.add_parser("gui", help="launch the GreyNOC GUI")
    pg.set_defaults(func=cmd_gui)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    effective = sys.argv[1:] if argv is None else argv

    if not effective:
        # No arguments: show help and exit 0 rather than argparse's
        # "the following arguments are required" + exit 2.
        #
        # This binary deliberately does NOT try to detect a double-click and
        # launch the GUI itself. An earlier version probed console ownership
        # (GetConsoleProcessList) to do exactly that; the probe answers
        # differently depending on how the process was started, so on a real
        # Explorer launch it fell through to this help text inside a console
        # Windows then destroyed instantly — the "terminal flashes and
        # disappears" report. Guessing the launch context is the bug.
        # The portable bundle ships a separate WINDOWED binary (RydSim.exe,
        # rydsim.gui.__main__) as the double-click target, which cannot make
        # that mistake because it has no console to lose.
        print(BANNER)
        parser.print_help()
        if getattr(sys, "frozen", False):
            print("\nTip: double-click RydSim.exe for the graphical "
                  "interface; this binary is the command-line tool.")
        return 0

    args = parser.parse_args(effective)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
