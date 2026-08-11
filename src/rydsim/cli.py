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
    ATResult,
    LadderConfig,
    at_experiment,
    at_finding,
    spectrum,
    superhet_transfer,
)
from .provenance import Finding
from .superhet import min_detectable_field, optimize_lo

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
    e_grid = np.linspace(args.e_min, args.e_max, args.lo_points)
    print(BANNER)
    print(f"[superhet] scanning {args.lo_points} LO points "
          f"E in [{args.e_min}, {args.e_max}] V/m ...")
    p_curve = superhet_transfer(cfg, e_to_omega, e_grid, args.probe_power_w,
                                n_doppler_nodes=args.doppler_nodes)
    from scipy.interpolate import CubicSpline
    transfer = CubicSpline(e_grid, p_curve)
    op = optimize_lo(lambda e: float(transfer(e)),
                     e_grid[2:-2], cfg.lambda_probe,
                     rin_db_per_hz=args.rin_db,
                     detector_nep_w_per_rthz=args.nep)
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
                "LO grid resolution": f"{(args.e_max-args.e_min)/args.lo_points:.3g} V/m",
                "transfer-curve model": "optical-depth chain pending spec 05 "
                "integration; slope-normalized NEF insensitive to overall scale",
            },
            constants_used=list(cfg.sources),
            caveats=["Technical-noise realism limited to stationary RIN/NEP model."],
        ).finalize(__version__)
        _write_finding(f, "superhet-nef")
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
    ph.add_argument("--e-min", type=float, default=0.05)
    ph.add_argument("--e-max", type=float, default=20.0)
    ph.add_argument("--lo-points", type=int, default=60)
    ph.add_argument("--doppler-nodes", type=int, default=24)
    ph.add_argument("--finding", action="store_true")
    ph.set_defaults(func=cmd_superhet)

    pv = sub.add_parser("validate", help="run the scientific validation suite")
    pv.add_argument("-v", "--verbose", action="store_true")
    pv.set_defaults(func=cmd_validate)

    pi = sub.add_parser("info", help="constants, conventions, provenance policy")
    pi.set_defaults(func=cmd_info)

    pg = sub.add_parser("gui", help="launch the GreyNOC GUI")
    pg.set_defaults(func=cmd_gui)
    return p


def _console_was_created_for_us() -> bool:
    """True when this process owns its console, i.e. it was double-clicked.

    Windows gives a GUI-launched console app a fresh console with only that
    process attached; a program started from an existing terminal shares
    that terminal with its parent shell. GetConsoleProcessList returning 1
    is therefore the standard double-click test. Any failure answers False
    (assume a real terminal) — the fallback must never hijack a scripted run.
    """
    if not sys.platform.startswith("win"):
        return False
    try:
        import ctypes

        buf = (ctypes.c_uint * 4)()
        n = ctypes.windll.kernel32.GetConsoleProcessList(buf, 4)
        return n == 1
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    effective = sys.argv[1:] if argv is None else argv

    if not effective:
        # A portable .exe gets double-clicked — that is the whole point of
        # shipping one. Bare argparse answers "error: the following
        # arguments are required: command", exits 2, and Windows tears the
        # console down instantly, so the binary appears to "error out
        # immediately" with nothing readable. Give the double-click the GUI
        # (what a user reaching for a desktop app wants), and give a real
        # terminal the help text rather than an error.
        if _console_was_created_for_us():
            try:
                return cmd_gui(None)
            except Exception as exc:                     # pragma: no cover
                print(f"{BANNER}\n\nCould not start the GUI: {exc}\n",
                      file=sys.stderr)
                parser.print_help()
                input("\nPress Enter to close...")       # keep the window up
                return 1
        print(BANNER)
        parser.print_help()
        return 0

    args = parser.parse_args(effective)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
