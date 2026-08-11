"""GreyNOC RydSim GUI — operator console for the electrometry simulator.

Tabs: Spectrum Lab (EIT/AT), Superhet (LO optimization + NEF budget),
Findings (provenance-tagged reports), Validation (live test-suite gate).
All physics runs on worker threads; the interface never blocks.
"""

from __future__ import annotations

import dataclasses
import pathlib
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk

import matplotlib
import numpy as np

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .. import __version__
from ..constants import AU_DIPOLE, HBAR
from ..experiment import LadderConfig, at_experiment, at_finding, spectrum, superhet_transfer
from ..provenance import IntegrityError
from ..superhet import min_detectable_field, optimize_lo
from . import theme

ROOT_DIR = pathlib.Path(__file__).resolve().parents[3]
FINDINGS_DIR = ROOT_DIR / "findings"


class ParamForm(ttk.Frame):
    """Labeled-entry form with typed getters, lab units."""

    def __init__(self, master, fields: list[tuple[str, str, str]]):
        super().__init__(master, style="Panel.TFrame")
        self.vars: dict[str, tk.StringVar] = {}
        for row, (key, label, default) in enumerate(fields):
            ttk.Label(self, text=label, style="Muted.TLabel").grid(
                row=row, column=0, sticky="w", padx=(8, 6), pady=2)
            var = tk.StringVar(value=default)
            self.vars[key] = var
            ttk.Entry(self, textvariable=var, width=12,
                      font=theme.FONT_MONO).grid(row=row, column=1,
                                                 sticky="ew", padx=(0, 8), pady=2)
        self.columnconfigure(1, weight=1)

    def get(self, key: str) -> float:
        return float(self.vars[key].get())


def form_to_config(form: ParamForm, doppler: bool, rf: bool) -> LadderConfig:
    mhz = 2 * np.pi * 1e6
    khz = 2 * np.pi * 1e3
    return LadderConfig(
        name="gui",
        omega_probe=form.get("probe_rabi") * mhz,
        omega_coupling=form.get("coupling_rabi") * mhz,
        omega_rf=(form.get("rf_rabi") * mhz) if rf else 0.0,
        delta_coupling=form.get("coupling_det") * mhz,
        delta_rf=form.get("rf_det") * mhz,
        gamma_e=form.get("gamma_e") * mhz,
        gamma_r=form.get("gamma_r") * khz,
        gamma_rp=form.get("gamma_r") * khz,
        deph_r=form.get("deph_r") * khz,
        temperature=form.get("temp"),
        doppler=doppler,
        rf_dipole=form.get("dipole") * AU_DIPOLE if form.get("dipole") > 0 else None,
    )


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"GreyNOC RydSim v{__version__}")
        self.geometry("1280x860")
        self.minsize(1100, 720)
        theme.apply_ttk_theme(self)
        matplotlib.rcParams.update(theme.MPL_RC)

        self.jobs: queue.Queue = queue.Queue()
        self._build_header()
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=8, pady=(4, 0))
        self._build_spectrum_tab()
        self._build_superhet_tab()
        self._build_findings_tab()
        self._build_validation_tab()
        self._build_statusbar()
        self.after(100, self._poll_jobs)

    # ---------------- chrome ----------------

    def _build_header(self):
        head = ttk.Frame(self, style="App.TFrame")
        head.pack(fill="x", padx=8, pady=(8, 0))
        ttk.Label(head, text="GreyNOC · RydSim", style="Banner.TLabel",
                  font=("Consolas", 16, "bold")).pack(side="left")
        ttk.Label(head, text=theme.TAGLINE, style="Tagline.TLabel").pack(
            side="left", padx=16, pady=(6, 0))

    def _build_statusbar(self):
        bar = ttk.Frame(self, style="App.TFrame")
        bar.pack(fill="x", side="bottom", padx=8, pady=4)
        self.status = ttk.Label(bar, text="ready", style="Tagline.TLabel")
        self.status.pack(side="left")
        ttk.Label(bar, text=f"v{__version__} · findings -> {FINDINGS_DIR.name}/",
                  style="Tagline.TLabel").pack(side="right")

    def set_status(self, text: str, error: bool = False):
        self.status.configure(text=text,
                              foreground=theme.RED if error else theme.FG2)

    # ---------------- worker plumbing ----------------

    def run_bg(self, fn, on_done):
        def work():
            try:
                result = fn()
                self.jobs.put((on_done, result, None))
            except Exception as exc:  # surfaced to the status bar
                self.jobs.put((on_done, None, exc))
        threading.Thread(target=work, daemon=True).start()

    def _poll_jobs(self):
        try:
            while True:
                on_done, result, exc = self.jobs.get_nowait()
                if exc is not None:
                    if isinstance(exc, IntegrityError):
                        self.set_status(f"INTEGRITY GATE: {exc}", error=True)
                    else:
                        self.set_status(f"error: {exc}", error=True)
                else:
                    on_done(result)
        except queue.Empty:
            pass
        self.after(100, self._poll_jobs)

    # ---------------- Spectrum Lab ----------------

    def _build_spectrum_tab(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="  Spectrum Lab  ")

        left = ttk.Frame(tab, style="Panel.TFrame")
        left.pack(side="left", fill="y", padx=(0, 6), pady=6)

        box = ttk.Labelframe(left, text="LADDER CONFIGURATION")
        box.pack(fill="x", padx=6, pady=6)
        self.spec_form = ParamForm(box, [
            ("probe_rabi", "Probe Rabi [MHz]", "0.05"),
            ("coupling_rabi", "Coupling Rabi [MHz]", "5.0"),
            ("rf_rabi", "RF Rabi [MHz]", "18.0"),
            ("coupling_det", "Coupling det. [MHz]", "0.0"),
            ("rf_det", "RF det. [MHz]", "0.0"),
            ("gamma_e", "Gamma_e [MHz]", "6.07"),
            ("gamma_r", "Rydberg decay [kHz]", "2.0"),
            ("deph_r", "Rydberg deph. [kHz]", "100.0"),
            ("temp", "Temperature [K]", "300.0"),
            ("dipole", "RF dipole [e*a0]", "1000.0"),
            ("span", "Scan span [MHz]", "40.0"),
            ("points", "Scan points", "501"),
        ])
        self.spec_form.pack(fill="x")

        self.var_doppler = tk.BooleanVar(value=True)
        self.var_rf_on = tk.BooleanVar(value=True)
        self.var_converge = tk.BooleanVar(value=True)
        for text, var in [("Doppler averaging (hot vapor)", self.var_doppler),
                          ("RF field on", self.var_rf_on),
                          ("Prove convergence (grid doubling)", self.var_converge)]:
            ttk.Checkbutton(box, text=text, variable=var).pack(
                anchor="w", padx=8, pady=1)

        btns = ttk.Frame(left, style="Panel.TFrame")
        btns.pack(fill="x", padx=6, pady=4)
        ttk.Button(btns, text="Run spectrum", style="Accent.TButton",
                   command=self.on_run_spectrum).pack(fill="x", pady=2)
        ttk.Button(btns, text="AT field measurement",
                   command=self.on_run_at).pack(fill="x", pady=2)
        ttk.Button(btns, text="Write finding report",
                   command=self.on_write_at_finding).pack(fill="x", pady=2)

        out = ttk.Labelframe(left, text="MEASUREMENT")
        out.pack(fill="both", expand=True, padx=6, pady=6)
        self.spec_out = tk.Text(out, width=38, height=12, bg=theme.BG2,
                                fg=theme.FG0, insertbackground=theme.ACCENT,
                                font=theme.FONT_MONO_SM, relief="flat", wrap="word")
        self.spec_out.pack(fill="both", expand=True, padx=4, pady=4)

        fig = Figure(figsize=(7, 5), dpi=100)
        self.spec_ax = fig.add_subplot(111)
        self.spec_canvas = FigureCanvasTkAgg(fig, master=tab)
        self.spec_canvas.get_tk_widget().pack(side="right", fill="both",
                                              expand=True, padx=6, pady=6)
        self._style_axes(self.spec_ax, "probe detuning [MHz]",
                         "probe transmission [norm]")
        self.spec_canvas.draw()
        self._last_at = None

    def _style_axes(self, ax, xlabel, ylabel):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_facecolor(theme.BG0)

    def _spectrum_job(self):
        cfg = form_to_config(self.spec_form, self.var_doppler.get(),
                             self.var_rf_on.get())
        span = self.spec_form.get("span")
        n = int(self.spec_form.get("points"))
        dps = 2 * np.pi * 1e6 * np.linspace(-span / 2, span / 2, n)
        resp = spectrum(cfg, dps, check_convergence=self.var_converge.get())
        return cfg, dps, resp

    def on_run_spectrum(self):
        self.set_status("running spectrum ...")
        self.run_bg(self._spectrum_job, self._show_spectrum)

    def _show_spectrum(self, payload):
        cfg, dps, resp = payload
        mhz = dps / (2 * np.pi * 1e6)
        self.spec_ax.clear()
        self._style_axes(self.spec_ax, "probe detuning [MHz]",
                         "probe transmission [norm]")
        trans = -resp.real
        self.spec_ax.plot(mhz, trans / max(abs(trans.max()), 1e-30),
                          color=theme.ACCENT, lw=1.6)
        mode = "hot vapor (converged avg)" if cfg.doppler else "cold atoms"
        self.spec_ax.set_title(f"EIT/AT spectrum — {mode}")
        self.spec_canvas.draw()
        self.set_status("spectrum done"
                        + (" · convergence proven" if self.var_converge.get()
                           and cfg.doppler else ""))

    def on_run_at(self):
        self.set_status("running AT measurement ...")

        def job():
            cfg = form_to_config(self.spec_form, self.var_doppler.get(), True)
            res = at_experiment(cfg, n_points=int(self.spec_form.get("points")))
            return cfg, res
        self.run_bg(job, self._show_at)

    def _show_at(self, payload):
        cfg, res = payload
        self._last_at = (cfg, res)
        mhz = res.detunings / (2 * np.pi * 1e6)
        self.spec_ax.clear()
        self._style_axes(self.spec_ax, "probe detuning [MHz]",
                         "probe transmission [norm]")
        t = res.transmission / max(abs(res.transmission.max()), 1e-30)
        self.spec_ax.plot(mhz, t, color=theme.ACCENT, lw=1.6)
        lines = []
        m = res.measurement
        if m is None:
            lines.append("NO RESOLVED DOUBLET")
            lines.append("Field below AT-resolvable regime;")
            lines.append("use superhet mode for weak fields.")
        else:
            for x in m.peak_positions:
                self.spec_ax.axvline(x / (2 * np.pi * 1e6), color=theme.AMBER,
                                     lw=0.9, ls="--")
            omega = m.splitting / res.mismatch_factor
            lines.append(f"splitting (probe axis): "
                         f"{m.splitting/(2*np.pi)/1e6:.4f} MHz")
            lines.append(f"resolved: {m.resolved}  [{m.method}]")
            lines.append(f"mismatch factor: {res.mismatch_factor:.4f}")
            lines.append(f"RF Rabi: {omega/(2*np.pi)/1e6:.4f} MHz")
            if res.field_v_per_m is not None:
                lines.append(f"E-field: {res.field_v_per_m:.6g} V/m")
                lines.append(f"        ({res.field_v_per_m/100:.6g} V/cm)")
            if m.uncertainty:
                lines.append(f"fit 1-sigma: "
                             f"{m.uncertainty/(2*np.pi)/1e6:.4f} MHz")
            lines.append("")
            lines.append("systematics: finite-coupling inward")
            lines.append("pull uncorrected (see spec 06e)")
        self.spec_out.delete("1.0", "end")
        self.spec_out.insert("1.0", "\n".join(lines))
        self.spec_ax.set_title("AT field measurement")
        self.spec_canvas.draw()
        self.set_status("AT measurement done")

    def on_write_at_finding(self):
        if not self._last_at:
            self.set_status("run an AT measurement first", error=True)
            return
        cfg, res = self._last_at
        f = at_finding(cfg, res)
        FINDINGS_DIR.mkdir(exist_ok=True)
        base = FINDINGS_DIR / f"at-measurement-{f.config_hash}"
        base.with_suffix(".json").write_text(f.to_json(), encoding="utf-8")
        base.with_suffix(".md").write_text(f.to_markdown(), encoding="utf-8")
        self.set_status(f"finding written: {base.name}.md")
        self._refresh_findings()

    # ---------------- Superhet ----------------

    def _build_superhet_tab(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="  Superhet  ")

        left = ttk.Frame(tab, style="Panel.TFrame")
        left.pack(side="left", fill="y", padx=(0, 6), pady=6)
        box = ttk.Labelframe(left, text="RECEIVER CONFIGURATION")
        box.pack(fill="x", padx=6, pady=6)
        self.sh_form = ParamForm(box, [
            ("coupling_rabi", "Coupling Rabi [MHz]", "5.0"),
            ("gamma_e", "Gamma_e [MHz]", "6.07"),
            ("gamma_r", "Rydberg decay [kHz]", "2.0"),
            ("deph_r", "Rydberg deph. [kHz]", "100.0"),
            ("temp", "Temperature [K]", "300.0"),
            ("dipole", "RF dipole [e*a0]", "1000.0"),
            ("power", "Probe power [uW]", "100.0"),
            ("rin", "Laser RIN [dB/Hz]", "-150.0"),
            ("nep", "Detector NEP [pW/rtHz]", "1.0"),
            ("emax", "LO scan max [V/m]", "20.0"),
            ("lopts", "LO scan points", "50"),
        ])
        self.sh_form.pack(fill="x")
        self.sh_doppler = tk.BooleanVar(value=True)
        ttk.Checkbutton(box, text="Doppler averaging",
                        variable=self.sh_doppler).pack(anchor="w", padx=8)
        ttk.Button(left, text="Optimize LO / compute NEF",
                   style="Accent.TButton",
                   command=self.on_run_superhet).pack(fill="x", padx=6, pady=6)
        out = ttk.Labelframe(left, text="RECEIVER REPORT")
        out.pack(fill="both", expand=True, padx=6, pady=6)
        self.sh_out = tk.Text(out, width=38, height=16, bg=theme.BG2,
                              fg=theme.FG0, insertbackground=theme.ACCENT,
                              font=theme.FONT_MONO_SM, relief="flat", wrap="word")
        self.sh_out.pack(fill="both", expand=True, padx=4, pady=4)

        fig = Figure(figsize=(7, 5), dpi=100)
        self.sh_ax1 = fig.add_subplot(211)
        self.sh_ax2 = fig.add_subplot(212)
        fig.subplots_adjust(hspace=0.45)
        self.sh_canvas = FigureCanvasTkAgg(fig, master=tab)
        self.sh_canvas.get_tk_widget().pack(side="right", fill="both",
                                            expand=True, padx=6, pady=6)
        self._style_axes(self.sh_ax1, "LO field [V/m]", "probe power [uW]")
        self._style_axes(self.sh_ax2, "LO field [V/m]", "NEF [nV/cm/rtHz]")
        self.sh_canvas.draw()

    def on_run_superhet(self):
        self.set_status("optimizing LO (scanning transfer curve) ...")

        def job():
            mhz = 2 * np.pi * 1e6
            khz = 2 * np.pi * 1e3
            f = self.sh_form
            cfg = LadderConfig(
                name="gui-superhet",
                omega_probe=2 * np.pi * 50e3,
                omega_coupling=f.get("coupling_rabi") * mhz,
                gamma_e=f.get("gamma_e") * mhz,
                gamma_r=f.get("gamma_r") * khz,
                gamma_rp=f.get("gamma_r") * khz,
                deph_r=f.get("deph_r") * khz,
                temperature=f.get("temp"),
                doppler=self.sh_doppler.get(),
                rf_dipole=f.get("dipole") * AU_DIPOLE,
            )
            d = cfg.rf_dipole
            p_in = f.get("power") * 1e-6
            e_grid = np.linspace(0.05, f.get("emax"), int(f.get("lopts")))
            p_curve = superhet_transfer(cfg, d / HBAR, e_grid, p_in)
            from scipy.interpolate import CubicSpline
            transfer = CubicSpline(e_grid, p_curve)
            op = optimize_lo(lambda e: float(transfer(e)), e_grid[2:-2],
                             cfg.lambda_probe,
                             rin_db_per_hz=f.get("rin"),
                             detector_nep_w_per_rthz=f.get("nep") * 1e-12)
            # NEF vs LO curve for the plot
            nefs = []
            for e in e_grid[2:-2]:
                from rydsim.superhet import NoiseBudget, shot_noise_psd, rin_psd, transfer_slope
                k = transfer_slope(lambda x: float(transfer(x)), float(e))
                if abs(k) < 1e-30:
                    nefs.append(np.nan)
                    continue
                b = NoiseBudget(shot=shot_noise_psd(float(transfer(e)), cfg.lambda_probe),
                                rin=rin_psd(float(transfer(e)), f.get("rin")),
                                detector=(f.get("nep") * 1e-12) ** 2)
                nefs.append(np.sqrt(b.total) / abs(k))
            return cfg, e_grid, p_curve, np.array(nefs), op
        self.run_bg(job, self._show_superhet)

    def _show_superhet(self, payload):
        cfg, e_grid, p_curve, nefs, op = payload
        self.sh_ax1.clear()
        self.sh_ax2.clear()
        self._style_axes(self.sh_ax1, "LO field [V/m]", "probe power [uW]")
        self._style_axes(self.sh_ax2, "LO field [V/m]", "NEF [nV/cm/rtHz]")
        self.sh_ax1.plot(e_grid, p_curve * 1e6, color=theme.BLUE, lw=1.5)
        self.sh_ax1.axvline(op.e_lo, color=theme.ACCENT, lw=1.0, ls="--")
        self.sh_ax1.set_title("atomic transfer curve P(E_RF)")
        self.sh_ax2.semilogy(e_grid[2:-2], nefs / 100 * 1e9,
                             color=theme.VIOLET, lw=1.5)
        self.sh_ax2.axvline(op.e_lo, color=theme.ACCENT, lw=1.0, ls="--")
        self.sh_ax2.set_title("noise-equivalent field vs LO")
        self.sh_canvas.draw()

        nef_cm = op.nef / 100
        b = op.budget
        txt = [
            f"optimal LO:  {op.e_lo:.4g} V/m",
            f"slope:       {op.slope:.4g} W/(V/m)",
            "",
            "noise budget [W^2/Hz]:",
            f"  shot:     {b.shot:.3g}",
            f"  RIN:      {b.rin:.3g}",
            f"  detector: {b.detector:.3g}",
            "",
            f"NEF: {nef_cm*1e9:.4g} nV/cm/rtHz",
            f"E_min (1 s):  {min_detectable_field(op.nef, 1.0):.4g} V/m",
            f"E_min (60 s): {min_detectable_field(op.nef, 60.0):.4g} V/m",
            "",
            "caveat: optical-depth chain pending",
            "cell module; slope-normalized NEF.",
        ]
        self.sh_out.delete("1.0", "end")
        self.sh_out.insert("1.0", "\n".join(txt))
        self.set_status("superhet optimization done")

    # ---------------- Findings ----------------

    def _build_findings_tab(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="  Findings  ")
        left = ttk.Frame(tab, style="Panel.TFrame")
        left.pack(side="left", fill="y", padx=(0, 6), pady=6)
        ttk.Button(left, text="Refresh", command=self._refresh_findings).pack(
            fill="x", padx=6, pady=6)
        self.find_list = tk.Listbox(left, width=44, bg=theme.BG2, fg=theme.FG0,
                                    selectbackground=theme.BG3,
                                    selectforeground=theme.ACCENT,
                                    font=theme.FONT_MONO_SM, relief="flat")
        self.find_list.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self.find_list.bind("<<ListboxSelect>>", self._show_finding)
        self.find_view = tk.Text(tab, bg=theme.BG1, fg=theme.FG0,
                                 font=theme.FONT_MONO_SM, relief="flat",
                                 wrap="word", padx=12, pady=10)
        self.find_view.pack(side="right", fill="both", expand=True,
                            padx=6, pady=6)
        self._refresh_findings()

    def _refresh_findings(self):
        self.find_list.delete(0, "end")
        FINDINGS_DIR.mkdir(exist_ok=True)
        self._finding_paths = sorted(FINDINGS_DIR.glob("*.md"),
                                     key=lambda p: p.stat().st_mtime,
                                     reverse=True)
        self._finding_paths = [p for p in self._finding_paths
                               if p.name != "README.md"]
        for p in self._finding_paths:
            self.find_list.insert("end", p.stem)

    def _show_finding(self, _event):
        sel = self.find_list.curselection()
        if not sel:
            return
        text = self._finding_paths[sel[0]].read_text(encoding="utf-8")
        self.find_view.delete("1.0", "end")
        self.find_view.insert("1.0", text)

    # ---------------- Validation ----------------

    def _build_validation_tab(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="  Validation  ")
        top = ttk.Frame(tab, style="Panel.TFrame")
        top.pack(fill="x", padx=6, pady=6)
        ttk.Button(top, text="Run scientific validation suite",
                   style="Accent.TButton",
                   command=self.on_validate).pack(side="left", padx=6, pady=6)
        self.val_state = ttk.Label(top, text="not run", style="Muted.TLabel")
        self.val_state.pack(side="left", padx=12)
        self.val_out = tk.Text(tab, bg=theme.BG2, fg=theme.FG1,
                               font=theme.FONT_MONO_SM, relief="flat")
        self.val_out.pack(fill="both", expand=True, padx=6, pady=(0, 6))

    def on_validate(self):
        self.set_status("running validation suite ...")
        self.val_state.configure(text="running ...", foreground=theme.AMBER)

        def job():
            r = subprocess.run(
                [sys.executable, "-m", "pytest", str(ROOT_DIR / "tests"),
                 "-v", "--tb=short"],
                capture_output=True, text=True, cwd=ROOT_DIR,
                env={**__import__("os").environ,
                     "PYTHONPATH": str(ROOT_DIR / "src")},
            )
            return r
        self.run_bg(job, self._show_validation)

    def _show_validation(self, r):
        self.val_out.delete("1.0", "end")
        self.val_out.insert("1.0", r.stdout[-20000:] + "\n" + r.stderr[-4000:])
        self.val_out.see("end")
        ok = r.returncode == 0
        self.val_state.configure(
            text="ALL BENCHMARKS PASSING" if ok else "FAILURES — see log",
            foreground=theme.ACCENT if ok else theme.RED)
        self.set_status("validation " + ("passed" if ok else "FAILED"),
                        error=not ok)


def main() -> int:
    app = App()
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
