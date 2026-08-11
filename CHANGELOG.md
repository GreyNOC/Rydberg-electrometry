# Changelog

All notable changes to RydSim. Format follows [Keep a Changelog](https://keepachangelog.com/1.1.0/);
this project uses [Semantic Versioning](https://semver.org/) (pre-1.0: minor
versions may change interfaces).

## [0.2.0] — 2026-08-11

First release with a portable Windows binary. The engine is unchanged in
intent from 0.1.0 but substantially corrected: this release is the product
of an adversarial audit, a remediation pass, a spec reconciliation, and a
pre-release security review — each of which found real defects.

**Validation suite: 391 → 605 tests** (1 documented xfail).

### Added
- `rydsim.designer` — surrogate-accelerated design-space mapping (DESIGNER
  D2/D3): GP surrogate with held-out error reported against the oracle's own
  uncertainty, feasibility-weighted active learning, and `confirm_frontier()`,
  which re-evaluates every Pareto point through the exact oracle before it can
  be claimed. Optional extra: `pip install 'rydsim[designer]'`.
- `rydsim.paths` — frozen-aware filesystem resolution, honouring
  `RYDSIM_FINDINGS_DIR`.
- Portable single-file Windows binary (`packaging/rydsim.spec`), built from a
  locked environment (`packaging/requirements-build.lock`) with a CycloneDX
  SBOM and third-party notices.
- `docs/AUDIT-2026-08-10.md`, `docs/SPEC-RECONCILIATION-2026-08-10.md` — the
  full findings registers, including the findings that were *refuted*.

### Fixed — physics and correctness
- **Species parameters were discarded** when building the ladder config, so Cs
  and Rb-85 designs were simulated inside a Rb-87 vapour cell. Now resolved per
  species through `rydsim.atom`, including the state-dependent coupling
  wavelength (ruling R-15, previously dead code behind a hard-coded 480 nm).
- **No optical-thickness refusal**: a Cs 313 K / 5 cm design returned
  `NEF = 5.4e9 nV/cm/√Hz` instead of refusing. Added an optical-depth gate.
- **Wigner 3j/6j returned mathematically impossible values** for generic large
  j — `3j(100,90,80,0,0,0)` gave `+2.52` against an elementary bound of
  `0.0705` (exact `-6.78e-3`), inside the stated contract. Exact-rational
  fallback plus a bound gate; the contract now holds unconditionally in j and
  rank.
- **Removed `pi_manifold_rms`**, an invented, unsourced dipole convention that
  had been used to force the Sedlacek benchmark to pass. That fixture is now a
  documented literature tension (residual exactly √2) per audit R5.
- **MSD94 model-potential tables were unguarded** — a corrupted parameter left
  the entire suite green. Real tripwire added via `model_potential_defect`.
- Whittaker/hyperu fence moved 25 → **20**, where the method meets benchmark
  B12's own 1e-6 contract (3.5e-8 at ν=20 vs up to 5.7e-6 above it).
- A-vs-B consensus gates in `radial` and `lifetimes` had **forked and
  disagreed** about the same physics; unified on the better-sourced rule
  (spec 02's 1e-4 row is S↔P-measured), with `lifetimes` delegating.
- Rydberg decay now comes from `rydsim.lifetimes` (Beterov fits, validated to
  0.7% against Table VII) instead of a fixed default that made IBW
  n-independent.
- Over-strict gates that made routine physics uncomputable were corrected to
  test the right quantity (orbit scale rather than relative spread for
  cancellation-suppressed elements).
- Species→element mapping unified on `atom.element_symbol`; the name-slicing
  form was duplicated across four modules and silently misclassified unknown
  species.

### Fixed — packaging and release integrity
- **The portable binary could not start**: the PyInstaller entry script used a
  relative import, which raises `ImportError` under a frozen build with no
  package context. The build emitted no warning, so a build-only gate would
  have shipped a dead binary.
- **Findings were written to `%TEMP%`** in the frozen build (path derived from
  `__file__` inside the extracted bundle) — i.e. into a directory Windows
  purges. Now resolved beside the executable, with a documents-area fallback
  when that location is not writable.
- **The GUI's validation tab reported the shipped binary's own physics as
  FAILING** — it invoked pytest against a path that does not exist in a frozen
  build. It now explains that the suite is source-checkout only, matching the
  guard the CLI already had.
- **A config file could inject fabricated provenance** into the "Constants on
  the critical path" block of a GreyNOC-branded finding, complete with forged
  `VERIFIED` tags. Config-supplied sources are now tagged at the trust
  boundary as unverified and not checked by RydSim.
- **`numpy>=1.26` was a factually wrong floor**: six core physics call sites
  use `np.trapezoid`, which is NumPy 2.0+. Corrected to `numpy>=2.0`.
- `scikit-learn` and `sympy` were used but undeclared; both are now declared
  (as the `designer` extra and a dev dependency respectively).
- The build is now performed in an isolated, locked virtualenv. Building from
  the developer's global interpreter pulled PyInstaller's optional-backend
  hooks toward torch/CUDA and would have produced an untruthful SBOM.

### Documentation
- 42 edits reconciled the normative specs with the code, **0 benchmarks
  weakened**. This included the spec 03 sentence that *caused* a confirmed 2×
  oscillator-strength error.

### Known limitations
- The Windows binary is **unsigned** — see the release notes.
- `rydsim validate` is unavailable in the portable build by design; the
  validation suite and its symbolic oracle are deliberately not bundled.
- High optical-depth operation (how several published experiments actually
  run) requires the z-propagation solver of spec 06 §7.2, which does not exist
  yet; such configurations are refused rather than approximated.

## [0.1.0] — 2026-08-10

Initial internal version: physics engine (EIT/Autler-Townes, Stark,
superheterodyne), CLI, GreyNOC GUI, findings pipeline, spec corpus.
