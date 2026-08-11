> **RETRACTED 2026-08-10 (same day).** Every configuration in this
> campaign ran at optical depth 5-100, far beyond the transfer-chain
> validity ceiling installed after audit finding CRIT-2 (the engine now
> REFUSES these configurations). The reported trade exponent alpha ~ 0.5
> is an artifact of operating in the optically-collapsed regime. The
> adversarial audit that exposed the missing gate, and the superseding
> measurement in the valid regime, are recorded in
> `d3-trade-law-v2` (see findings directory). Kept for the record per
> the house rule: a retraction is itself a finding.

# Finding: D3: sensitivity-bandwidth trade law on the Rb-87 nD5/2 frontier — NEF ~ sqrt(IBW), not 1/IBW

*GreyNOC RydSim 0.1.0 · 2026-08-10T23:25:47+00:00 · config `e812470c37ac98f1` · python 3.11.9 / Windows 10*

## Result

- **spec 08 SS2.6 hypothesis (NEF x IBW invariant to ~3x)**: REJECTED — the product drifts 14.8x across the frontier (2.22 -> 32.94 in nV/cm/rtHz x MHz)
- **measured trade law (frontier, n>=60 branch, 8 pts)**: NEF ~ IBW^0.537 (R^2 = 0.869)
- **measured trade law (controlled Omega_c sweep at fixed n=60, trade branch)**: NEF ~ IBW^0.482 (R^2 = 0.957)
- **conclusion**: alpha = 0.51 +- 0.03 => the invariant is NEF/sqrt(IBW), i.e. NEF^2/IBW is conserved, NOT the folklore product NEF x IBW
- **mechanism (from the controlled sweep)**: above the NEF optimum, IBW ~ Omega_c^1.00 while the transduction slope kappa ~ Omega_c^-0.52; since NEF = sqrt(S_P)/kappa, NEF ~ Omega_c^0.52 ~ sqrt(IBW)
- **NEF is U-shaped in Omega_c**: optimum at Omega_c/2pi = 10.25 MHz for Rb-87 60D5/2 at 313 K; a single power law across the whole sweep is INVALID (R^2 ~ 0.16) — the trade exists only ABOVE the optimum
- **frontier**: 9 oracle-confirmed Pareto points, NEF 0.121-0.523 nV/cm/rtHz over IBW 18.5-63.0 MHz, from 800 oracle evaluations (753 feasible)
- **published operating points**: Sedlacek 2012 (53D5/2), Sci. Adv. 2024 (39D5/2) and a typical 50D5/2 lab config all land strictly INSIDE the frontier (dominated by 2.1x, 4.0x, 2.4x in NEF at equal-or-greater IBW)

## Uncertainty budget

- **trade exponent alpha**: 0.482 (controlled, R^2 0.957) to 0.537 (frontier, R^2 0.869); quote 0.51 +- 0.03
- **oracle model uncertainty (NEF)**: ~0.5% common-mode, from the cross-method radial dipole spread (NEF ~ 1/d). This is a SYSTEMATIC that shifts all points together — it does not affect the trade exponent or the ranking of designs.
- **IBW numerical**: ~1% (resolvent bisection + scan grid)
- **frontier reproducibility**: every frontier point re-evaluated through the oracle and required to reproduce to 1e-6 (confirm_frontier)

## Validation state

- engine at 391 passing validation benchmarks; frontier points oracle-reconfirmed; two independent routes to the exponent agree

## Constants on the critical path

- rydsim.atom quantum defects (VERIFIED / VERIFIED-ARC per row)
- rydsim.radial three-method consensus radial integrals (model potential / Coulomb-QDT / Kaulakys, spread <= 1e-4)
- rydsim.lifetimes Beterov PRA 79, 052504 (2009) fits
- CODATA 2022 via scipy.constants

## Caveats (standing model limitations)

- ABSOLUTE NEF VALUES ARE AN IDEALIZED BOUND, NOT A HARDWARE PREDICTION. They sit ~20-80x below the published state of the art (10 nV/cm/rtHz) because the noise model here carries only photon shot noise, RIN, detector NEP and the atom-projection floor, with a perfectly locked on-resonance probe. Laser frequency noise, servo residuals, cell etalon/RF-inhomogeneity and LO amplitude noise are NOT included. The TRADE EXPONENT is the transferable result; the absolute scale is not.
- Rydberg dephasing is a fixed 100 kHz lab-technical input, not derived from first principles. The POSITION of the Omega_c optimum depends on it directly; the exponent above the optimum is far less sensitive.
- The Rydberg population decay is now computed from rydsim.lifetimes (Beterov radiative + BBR, validated to 0.7% against Table VII), but at 100 kHz dephasing it is subdominant (1-4 kHz): the coherence is dephasing-limited, so the n^3 lifetime gain does NOT buy bandwidth here. In a low-dephasing or cold-atom regime this term would matter and the frontier could change shape.
- Frontier points cluster at the top of the n range (62-70), i.e. against the sampled boundary. The true optimum may lie at higher n, where the model's neglected physics (Rydberg-Rydberg interactions, ionization, n-dependent stray-field sensitivity) becomes important. The frontier should not be extrapolated past n = 70.
- Single species/ladder (Rb-87 nD5/2 -> (n+1)P3/2). No claim is made about other ladders, Zeeman-tuned schemes, or multi-tone dressing.

## Reproduce

```bash
rydsim run --config <saved config for e812470c37ac98f1>
```

*Reproducible or it didn't happen.*
