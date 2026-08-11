# Finding: D3 v2: sensitivity-bandwidth trade on Rb-87 nD5/2, measured INSIDE the transfer-chain validity regime — the trade is far weaker than the retracted v1 claimed

*GreyNOC RydSim 0.1.0 · 2026-08-11T00:17:36+00:00 · config `6f7f20848dca40a1` · python 3.11.9 / Windows 10*

## Result

- **controlled sweep (Rb-87 60D5/2->61P3/2, 300 K, L = 2 mm, OD << 5)**: NEF is U-shaped in Omega_c with the optimum at Omega_c/2pi = 30.3 MHz (NEF 2.90 nV/cm/rtHz, IBW 55.9 MHz)
- **trade branch above the optimum (4 pts, IBW 55.9 -> 91.4 MHz)**: NEF ~ IBW^0.115 (R^2 = 0.920); mechanism IBW ~ Omega_c^1.00, |kappa| ~ Omega_c^-0.14
- **replication at n=50 and at 310 K / 1 mm**: exponents 0.054-0.066 with the same U-shape — consistently WEAK, an order of magnitude below the retracted alpha ~ 0.5
- **headline**: in the valid thin-medium weak-probe regime, bandwidth above the NEF optimum is nearly free: a 1.6x IBW increase costs only ~6% in NEF until the feasibility edge (~90 MHz at 60D5/2). The strong alpha ~ 0.5 trade of the retracted v1 was an optical-collapse artifact, not atomic physics.
- **spec 08 SS2.6 hypothesis (NEF x IBW invariant to ~3x)**: still REJECTED, but for the opposite reason v1 claimed: the product grows almost linearly with IBW because NEF is nearly flat above the optimum (1.7x drift over the measured branch from IBW alone)
- **what invalidated v1**: the audit's CRIT-2 (no optical-depth gate): every v1 configuration ran at OD 5-100 where transmitted probe power collapses; the apparent kappa ~ Omega_c^-0.52 there was transmission collapse, not transduction physics

## Uncertainty budget

- **trade exponent**: 0.05-0.14 across replications; quote alpha < 0.15 with R^2 0.83-0.92 on short (4-pt) branches — the branch is short because feasibility ends, an honest limitation
- **oracle NEF common-mode**: ~0.5% (radial dipole spread), irrelevant to the exponent
- **IBW numerical**: ~1%

## Validation state

- engine under post-audit remediation (34 confirmed findings, 2 critical fixed by integrator and regression-locked); this measurement uses only the fixed paths

## Constants on the critical path

- rydsim.atom (Steck rev 2.3.4 / Mack / Deiglmayr, audited)
- rydsim.radial three-method consensus
- rydsim.lifetimes Beterov fits (audited)
- CODATA 2022 via scipy.constants

## Caveats (standing model limitations)

- ABSOLUTE NEF here (~3 nV/cm/rtHz at 2 mm) is worse than v1's retracted numbers because a thin cell holds fewer atoms — and it remains an idealized bound (no laser frequency noise, perfect lock, stationary RIN/NEP).
- The trade branch spans only 1.6x in IBW before the feasibility edge; the exponent should not be extrapolated beyond ~90 MHz.
- A valid-regime Pareto FRONTIER (v1's other deliverable) is NOT reissued here: with the OD gate binding, the sampled frontier rides the feasibility boundary (temperature marches down along it), confounding the trade — mapping it honestly needs density and cell length as free axes. Deferred, stated openly.
- High-OD operation (how Jing 2020 actually runs a 5 cm Cs cell) requires the z-propagation solver of spec 06 §7.2 — until it exists, RydSim cannot model those operating points at all.
- Single ladder (nD5/2 -> (n+1)P3/2), pi polarization, no Zeeman tuning, no multi-tone dressing.

## Reproduce

```bash
rydsim run --config <saved config for 6f7f20848dca40a1>
```

*Reproducible or it didn't happen.*
