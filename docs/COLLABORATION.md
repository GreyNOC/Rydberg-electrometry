# The RydSim Collaboration

**GreyNOC Rydberg Electrometry Simulation Program — Operating Model**

> This document describes how RydSim is engineered and how its findings are
> produced: as a **virtual scientific collaboration** executed on a
> multi-agent compute fabric. One orchestrator (the "machine") coordinates a
> panel of specialist scientist-agents (the "collaboration"), each owning a
> domain, each cross-examined by the others before anything ships. The model
> is deliberately patterned on how a national-lab metrology group runs: no
> single author, no unreviewed number, no finding without provenance.

---

## 1. The machine

RydSim work is executed as orchestrated multi-agent workflows:

- **Fan-out**: independent specialists work their domains in parallel —
  spec-writing, implementation, validation, calibration studies.
- **Barrier + synthesis**: results merge only through explicit cross-check
  stages; nothing flows from one specialist's output to the release without
  passing through an adversarial reviewer whose job is to *refute* it.
- **Deterministic control flow**: the orchestration script — not any
  individual agent — decides what runs, what blocks, and what constitutes
  done. Agents provide judgment; the machine provides discipline.
- **Provenance by construction**: every constant, equation, and benchmark in
  the codebase traces to a spec document written in the fan-out phase, and
  every spec value carries a confidence tag
  (`VERIFIED` / `LITERATURE-RECALL` / `UNVERIFIED`).

## 2. The collaboration — the virtual panel

Each seat is an agent charter, scoped like a PI's responsibility in a real
collaboration. Seats, not personalities:

| Seat | Charter | Owns |
|---|---|---|
| **Atomic-structure theorist** | Energy levels, quantum defects, Rydberg–Ritz machinery | `docs/spec/01`, `rydsim.atom` |
| **Numerical-methods lead** | Numerov integration, model potentials, cross-validated matrix elements | `docs/spec/02`, `rydsim.radial` |
| **Angular-momentum algebraist** | Wigner algebra, dipole moments, selection rules, sum rules | `docs/spec/03`, `rydsim.angular` |
| **Decoherence & linewidth specialist** | Lifetimes, BBR, transit, collisions — the dephasing budget | `docs/spec/04`, `rydsim.lifetimes` |
| **Vapor-cell experimentalist** | Density, Doppler, propagation, cell-wall screening | `docs/spec/05`, `rydsim.cell` |
| **Quantum-optics theorist** | Lindblad/OBE, EIT, Autler–Townes, lineshapes | `docs/spec/06`, `rydsim.eit` |
| **Stark-physics specialist** | Polarizabilities, Stark maps, DC/low-frequency sensing | `docs/spec/07`, `rydsim.stark` |
| **RF/metrology engineer** | Superheterodyne, noise model, NEF/SQL, receiver figures of merit | `docs/spec/08`, `rydsim.superhet` |
| **Validation librarian** | The corpus of published results the simulator must reproduce | `docs/spec/09`, `tests/` |
| **Chief theorist (consistency)** | One set of conventions across every module; dimensional analysis of every load-bearing equation | `docs/spec/00-conventions.md` |
| **Integrity auditor (adversarial)** | Hunts fabricated-looking constants; defines the self-checks that catch them | `docs/spec/00-integrity-audit.md` |

The two `00-` seats are structural: they run *after* the domain seats and
have authority to overrule them. A convention conflict or an unsourced
constant is a blocking defect, not a footnote.

## 3. How a finding is produced

A **finding** is a quantitative claim produced by calibrating the simulation —
e.g. "shot-noise-limited NEF of configuration X is Y nV/cm/√Hz ± Z". Findings
are held to the GreyNOC house rule: **reproducible or it didn't happen.**

The pipeline every finding passes through:

1. **Specification** — the governing equations and constants are written down
   with sources *before* implementation (`docs/spec/`).
2. **Implementation with redundancy** — load-bearing quantities are computed
   by ≥2 independent methods where feasible (e.g. radial matrix elements via
   Numerov *and* Coulomb approximation *and* semiclassical formula); their
   disagreement is reported as numerical uncertainty, and excessive
   disagreement fails loudly rather than averaging quietly.
3. **Validation gate** — the full benchmark suite (analytic exact values,
   published experimental results, cross-module consistency) must pass at its
   declared tolerances. Benchmarks are graded honestly
   (exact / tight / order-of-magnitude / qualitative), never flattened into a
   wall of green ticks.
4. **Calibration study** — the actual experiment-in-silico: parameter sweeps,
   optimizations, sensitivity analyses, run from a declarative config so the
   run is reproducible from the artifact alone.
5. **Finding report** — emitted with full provenance: config hash, code
   version, every input parameter, the uncertainty budget, the confidence
   tags of every constant on the critical path, and the model-limitation
   caveats from the validation corpus. A finding that rests on any
   `UNVERIFIED` constant says so on its face.

## 4. Rules of the collaboration

- **No fabrication.** A missing number tagged `MISSING` outranks a plausible
  invented one. Constants carry sources; outputs carry provenance.
- **Adversarial before released.** Every artifact gets a reviewer whose
  explicit brief is refutation, not approval.
- **Disagreement is data.** When two independent methods disagree, the
  disagreement is published as the uncertainty — not silenced by picking the
  prettier number.
- **The model states its own limits.** Every report carries the standing
  caveats: model-potential accuracy bounds, neglected physics (ion effects,
  high-density interactions, cell-wall microphysics), and the difference
  between a simulated sensitivity and a measured one.
- **Traceability mirrors the physics.** Rydberg electrometry's claim to fame
  is SI-traceability — field values traceable to Planck's constant and atomic
  structure. The simulator holds itself to the software analogue: every
  output traceable to sourced constants and versioned code.

## 5. Artifacts map

```
docs/spec/00-conventions.md      ← normative symbols/units/conventions (chief theorist)
docs/spec/00-integrity-audit.md  ← risk register + mandatory self-checks (auditor)
docs/spec/01..09-*.md            ← domain specifications (the panel)
src/rydsim/                      ← implementation, one module per seat
tests/                           ← the validation gate (pytest)
configs/                         ← declarative experiment configs (the lab notebook)
findings/                        ← calibration-study reports with provenance
```

---

*GreyNOC · RydSim collaboration charter · 2026-08-10*
