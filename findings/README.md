# findings/

Calibration-study outputs. Each finding is emitted as a JSON + Markdown pair
by `rydsim run` / the GUI, carrying:

- the headline result with units and uncertainty budget,
- the full declarative config and its hash (reproduce with
  `rydsim run --config ...`),
- RydSim version and environment,
- provenance strings for every constant on the critical path
  (`VERIFIED` / `LITERATURE-RECALL` / `UNVERIFIED` / `COMPUTED`),
- cross-method check results,
- standing model-limitation caveats,
- the validation-suite state at the time of the run.

A finding resting on any `UNVERIFIED` constant says so on its face.
Generated files are gitignored; commit a finding deliberately when it is
reviewed and worth keeping.
