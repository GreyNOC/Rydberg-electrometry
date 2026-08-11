# RydSim — GreyNOC Rydberg Electrometry Simulator

Scientifically rigorous, calibratable simulation of Rydberg-atom RF
electrometry: ladder EIT / Autler–Townes spectroscopy in hot vapor and cold
atoms, superheterodyne receiver transduction, and full noise/sensitivity
budgets — built to produce **reproducible findings**, not plots.

> **House rule:** *reproducible or it didn't happen.* Every number this
> simulator emits is convention-stamped, provenance-tagged, and backed by a
> validation suite anchored to published, verified benchmarks. When a
> result is not numerically converged, the engine raises `IntegrityError`
> instead of returning it.

## What it does

- **EIT/AT spectroscopy** — 3/4-level ladder optical Bloch equations
  (exact steady-state Lindblad solver) and an analytic weak-probe
  continued-fraction susceptibility, cross-validated against each other.
- **Hot-vapor physics** — Maxwell-Boltzmann Doppler averaging on
  resonance-refined velocity grids with mandatory convergence proof
  (grid doubling + domain widening, ≤1e-4). The NIST wavelength-mismatch
  AT factor (λc/λp) *emerges* from the average and is verified to ~1%.
- **Field measurement** — AT-splitting extraction (doublet fit with honest
  unresolved-regime refusal), mismatch correction, inversion
  E = ħΩ/d with uncertainty budget.
- **Superheterodyne receiver** — LO-point optimization on the atomic
  transfer curve, shot/RIN/detector noise budgets, NEF, first-principles
  instantaneous bandwidth from the OBE resolvent transfer function H(δ),
  SQL (both published conventions), noise temperature / noise figure.
- **Findings pipeline** — every measurement can be written as a
  config-hashed JSON+Markdown report carrying its full uncertainty budget,
  constants provenance, caveats, and validation state.

## Quick start

```bash
# run the scientific validation suite (389 checks + 1 documented xfail)
python -m rydsim.cli validate

# hot-vapor AT field measurement with a finding report
python -m rydsim.cli at --rf-rabi-mhz 18 --rf-dipole-ea0 1000 --finding

# superhet receiver: LO optimization + noise budget
python -m rydsim.cli superhet --rf-dipole-ea0 1000 --probe-power-w 100e-6 --finding

# EIT spectrum to a data file
python -m rydsim.cli spectrum --rf-rabi-mhz 25 --span-mhz 40 -o spectrum.dat

# the GreyNOC GUI
python -m rydsim.cli gui
```

(From the repo root with `src` on `PYTHONPATH`, or `pip install -e .`.)

## Architecture

```
docs/spec/        normative physics specs (web-verified, confidence-tagged)
src/rydsim/
  constants.py    CODATA via scipy; atomic-unit conversions
  wigner.py       3j/6j/CG (log-factorial; validated vs sympy exact)
  numerov.py      radial integrator core (validated vs analytic H)
  lindblad.py     N-level ladder Lindblad: steady state, linear response H(δ), IBW
  eit.py          analytic weak-probe chi; converged Doppler averaging
  doppler.py      thermal-vapor helpers + Lindblad cross-validation path
  spectroscopy.py AT-splitting extraction, EIT linewidth measurement
  superhet.py     transduction, noise budgets, NEF, SQL, T_eq/NF
  experiment.py   config -> run -> Finding facade (CLI/GUI backend)
  provenance.py   Finding reports, config hashing, IntegrityError
  cli.py          full CLI
  gui/            GreyNOC-branded Tk GUI (dark theme, live validation tab)
tests/            the validation suite — spec-anchored benchmark tests
findings/         provenance-tagged measurement reports (generated)
docs/MISSION.md   the next-era thesis and test campaigns T1-T6
```

## Scientific integrity

- Spec-first: physics fixed in `docs/spec/` by a multi-specialist,
  web-verified research pass; every constant carries VERIFIED /
  LITERATURE-RECALL / UNVERIFIED confidence tags.
- Cross-method validation: analytic vs Lindblad per velocity class and
  after averaging; resolvent vs time-domain OBE; quadrature convergence
  gates that *raise* rather than return unconverged numbers.
- Published-value anchors: Steck line data, the Sci. Adv. 2024 SQL chain
  (3.70 nV/cm/√Hz → ×2.70 → 10.0 nV/cm/√Hz → T_eq 828 K vs 830 K
  published), Jing 2020 superhet parameters, NIST mismatch-factor papers.
- Development is adversarial by doctrine: the test suite exists to refute
  the engine. Two real bugs (Doppler quadrature aliasing; transit-channel
  factor 2) were caught this way and are now locked by regression tests.

## Status

Core physics engine, CLI, GUI, findings pipeline: **operational, 389 passing /
1 documented xfail** (RS-07-15 strict form — see `tests/test_stark.py`). Atomic-structure data modules (quantum
defects, model-potential wavefunctions, dipole moments, lifetimes, vapor
pressure, Stark maps) land next from specs 01–05/07/09 — after which the
species presets (`--species rb87 --n 53 ...`) replace manually supplied
dipole moments, and calibration campaigns T1–T6 (docs/MISSION.md) run.

---
*GreyNOC security research & engineering · not a certified metrology
instrument · findings carry their own caveats*
