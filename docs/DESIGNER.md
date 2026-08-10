# RydSim DESIGNER — AI-Driven Inverse Design of Rydberg Sensors

**Fork B of the RydSim program.** Companion to [MISSION.md](MISSION.md).
Status: specified, pending the atomic-data core (Fork A).

---

## 1. The problem this solves

Today — in the literature and in RydSim's own T2 campaign — Rydberg sensors
are designed **forward**: a physicist picks a configuration (species, n,
state ladder, RF transition, Rabi frequencies, beam geometry, cell
conditions), simulates it, and reports what that one choice does. Mapping a
trade-off means a grid sweep, and the grid is only ever swept over the two
or three axes someone thought to vary.

The design space is far larger than that: species × n × angular-momentum
state × RF transition partner × Ω_c × Ω_LO × beam waists × probe power ×
density/temperature × cell material × bias/dressing scheme. It is
high-dimensional, expensive to evaluate, multi-objective, and riddled with
sharp resonant structure. **No human sweep covers it, and nobody has
published its Pareto frontier.**

DESIGNER inverts the direction of the question:

> Give the machine a mission spec — *"2–6 GHz, NEF < 50 nV/cm/√Hz,
> IBW > 20 MHz, two lasers, room-temperature cell"* — and get back the
> atomic recipe and operating point that meets it, Pareto-ranked, with an
> uncertainty budget, reproducible from a config hash.

## 2. Why now, and why this program

Three preconditions are met here and essentially nowhere else:

1. **A validated physics oracle.** RydSim maps config → (NEF, IBW, dynamic
   range, traceability) from first principles, anchored to the published
   record, with convergence gates that refuse to emit unconverged numbers.
   An optimizer is only as good as its objective function; ours is
   defensible.
2. **A hard integrity doctrine.** The failure mode of "AI designs a sensor"
   is confident nonsense. Our house rule structurally prevents it (§5).
3. **The field's central open problem is a trade-off, not a number.**
   Sensitivity, bandwidth, and coverage fight each other (MISSION §2.1);
   the literature reports operating points, not frontiers.

## 3. Architecture

```
mission spec ──► SAMPLER ──► SURROGATE ──► OPTIMIZER ──► candidate configs
                    ▲            │              │
                    │            ▼              ▼
                    └──── active learning ── RydSim ORACLE (validated)
                                                 │
                                            FINDINGS (hashed, caveated)
```

**Layer 1 — Sampler / feasibility.** Enumerates *physically legal*
configurations: real states from `rydsim.atom`, real RF transition partners
and dipoles from `rydsim.dipoles`, validity fences (Inglis–Teller, RWA,
weak-probe, n floors). Illegal regions are rejected by the physics modules,
not learned around.

**Layer 2 — Surrogate + active learning.** Each oracle evaluation costs
seconds; a real search wants 10⁴–10⁶ points. A Gaussian-process (or small
neural) surrogate learns config → objectives, and an acquisition function
spends oracle calls only where the frontier is uncertain. Every surrogate
prediction is provisional until the oracle confirms it.

**Layer 3 — Multi-objective optimization.** qNEHVI / NSGA-II over
(NEF, IBW, coverage, SWaP proxy, traceability) to trace the **whole Pareto
surface**, not one optimum. Published operating points are located on that
surface as anchors.

**Layer 4 — Amortized inverse model.** A conditional generative model
(normalizing flow / cVAE) trained on oracle+surrogate data turns a target
spec into candidate recipes in milliseconds; each candidate is then
confirmed by the exact engine before it is reported.

**Layer 5 — Learned field decoder** *(the receiver's brain).* A small
temporal network trained on simulated (field → probe-transmission) pairs
inverts the beat note **in the nonlinear/saturating regime** where the
closed-form inversion breaks. This is what opens the strong-signal and
co-site-interference regime (MISSION T3) that analytic methods cannot
reach.

**Layer 6 — LLM scientist-in-the-loop.** The multi-agent pattern already
used for the spec corpus: agents propose novel ladders, dressing and
Zeeman-tuning schemes from the literature, and critique the frontier for
what the search has *not* explored. Proposals enter as configurations, never
as claims.

## 4. The north star

One falsifiable target:

> **Find a configuration whose NEF·IBW product beats the best published
> operating point — or demonstrate the bound that says none exists.**

Either outcome is a genuine finding no human sweep would have produced, and
both serve the EW/spectrum-awareness mission: design a better receiver, and
understand the limits of an adversary's.

Secondary targets: the first machine-mapped NEF-vs-IBW Pareto frontier with
uncertainty bands; a continuous-coverage recipe for a stated band; a
quantified price of SI-traceability (MISSION T4) read off the frontier.

## 5. The credibility firewall (non-negotiable)

**The AI proposes; the validated physics disposes.**

- No AI component ever emits a performance number. Models nominate
  *configurations*; all reported physics comes from the validated engine.
- Surrogate-only results are never reported as findings — they are search
  state. Anything published is oracle-confirmed.
- Every design point is a config-hashed `Finding` with its uncertainty
  budget, the confidence taint of its inputs, and the standing caveats.
- The optimizer inherits the refusal machinery: a candidate whose evaluation
  raises `IntegrityError` (unconverged, out-of-validity, single-method
  dipole) is discarded, never silently coerced to a number.
- Surrogate quality is itself reported (held-out error on oracle points), so
  readers know how much of the frontier is measured vs interpolated.

## 6. Milestones

| # | Deliverable | Gate |
|---|---|---|
| D0 | Objective API: config → (NEF, IBW, DR, traceability) + validity flags | Fork A complete; reproduces T1 anchors |
| D1 | Feasible-config sampler over real states/transitions | legality checks pass; no fabricated states |
| D2 | Oracle campaign + surrogate with held-out error reported | surrogate error < oracle uncertainty on holdout |
| D3 | **First machine-mapped NEF-vs-IBW Pareto frontier** (Rb subspace) | published points land on/inside the frontier |
| D4 | Multi-objective search with coverage + SWaP; inverse model | round-trip: spec → recipe → oracle confirms spec met |
| D5 | Learned decoder for the nonlinear regime | beats closed-form inversion beyond 1 dB compression |
| D6 | North-star attempt + honest bound statement | oracle-confirmed, adversarially reviewed |

D3 is the first genuinely novel scientific output and the point at which
this stops being infrastructure.

## 7. What this is not

Not a replacement for experiment: it designs candidates and bounds, which
hardware must confirm. Not a certified metrology tool. Not a black box —
every recommendation traces to physics a reviewer can re-run. And the model
uncertainty of the simulator (model potentials, neglected interactions,
cell-wall physics, technical-noise realism) bounds every claim, exactly as
in MISSION T6.

---

*GreyNOC · RydSim DESIGNER · reproducible or it didn't happen*
