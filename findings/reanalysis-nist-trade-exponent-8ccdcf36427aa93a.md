# Finding: First trade exponent for a Rydberg sensor measured within one apparatus: alpha = 0.15, reanalysed from Manchaiah et al. (NIST) — the folklore constant-product law overpredicts the bandwidth penalty by >= 42x

*GreyNOC RydSim 0.2.0 · 2026-08-11T21:21:56+00:00 · config `8ccdcf36427aa93a` · python 3.11.9 / Windows 10*

## Result

- **method**: Reanalysis of published STATED values (not figure digitisation) from Manchaiah, Oliver, Berweger, Holloway & Prajapati, arXiv:2509.20632, 'Probing Bandwidth and Sensitivity in Rydberg Atom Sensing via Optical Homodyne and RF Heterodyne Detection'. Their Fig. 4 reports sensitivity vs beatnote frequency at several coupling powers on ONE apparatus — the configuration the literature audit identified as the field's only within-apparatus trade data. They extract no exponent; this does.
- **trade exponent alpha (NEF ~ IBW^alpha)**: alpha = 0.145 to 0.158. Point estimate 0.158 from their own bandwidth DEFINITION (NEF at the bandwidth edge = 2x the low-frequency value) over the 100 kHz -> 8 MHz span (80x); upper bound 0.145 from the separate statement that NEF 'remains below 20 uV/m/rtHz up to 8 MHz'. The two readings are mildly inconsistent because 2 x 10.6 = 21.2 > 20.
- **the folklore is refuted by a factor of at least 42**: A constant NEF x IBW product (alpha = 1) predicts NEF(8 MHz) = 848 uV/m/rtHz from the stated NEF(100 kHz) = 10.6. Observed: <= 20. The folklore overpredicts the bandwidth penalty by >= 42x.
- **stated values used (160 mW coupling)**: NEF(100 kHz) = 10.6 uV/m/rtHz = 106 nV/cm/rtHz; NEF remains <= 20 uV/m/rtHz = 200 nV/cm/rtHz out to 8 MHz. Best overall: 9.9(4) uV/m/rtHz at 100 kHz with 140 mW.
- **interior optimum in coupling power — CONFIRMED as structure**: 20 mW -> ~20 uV/m/rtHz; 140 mW -> 9.9(4); 160 mW -> 10.6. The minimum is INTERIOR: 160 mW is worse than 140 mW, so NEF is not monotone in coupling power. This is prediction #1 of docs/EXPERIMENTS.md.
- **authors' own conclusion, verbatim**: 'Producing a higher coupling Rabi frequency through the use of smaller beam sizes and also using the optical homodyne technique enables the simultaneous improvement of both the sensitivity and bandwidth of a Rydberg sensor.' They also name the opposing belief as folklore: higher bandwidth via smaller beams 'is thought to compromise sensitivity'.
- **comparison with RydSim's blind prediction**: RydSim (corrected engine, Rb-87 60D5/2, 300 K, 5 cm) predicted alpha = 0.179 and an interior optimum. Measured: alpha = 0.145-0.158. The prediction is ~13-23% STEEPER than measured — a mild tension, stated as such, not a claimed confirmation. Both are an order of magnitude below the folklore's alpha = 1.

## Uncertainty budget

- **alpha range**: 0.145-0.158, the spread being the inconsistency between their two stated characterisations of the same curve.
- **reference frequency**: The 'low frequency' reference is taken as the stated 100 kHz point. If their plateau extends below 100 kHz the span is larger and alpha SMALLER, so 0.158 is conservative.
- **their 3 dB / 6 dB inconsistency**: They equate 'twice the field' with a '3 dB drop'; a factor 2 in field is 6 dB. If a genuine 3 dB (factor 1.414) was meant, alpha = 0.079 — shallower still. Every reading stays far below alpha = 1.
- **no error bars on the curve**: Only the 9.9(4) point carries a stated uncertainty. The 140-vs-160 mW difference is 1.7 sigma on that value.

## Validation state

- reanalysis of published stated values; arithmetic reproduced in-session; no figure digitisation; conclusions bounded by the authors' own definitional inconsistency

## Constants on the critical path

- arXiv:2509.20632 — all numerical inputs, VERIFIED (fetched and quoted this session; abstract and HTML full text)

## Caveats (standing model limitations)

- The interior optimum is CONFIRMED AS STRUCTURE but its statistical strength is weak: 9.9(4) vs 10.6 is only ~1.7 sigma, and the intermediate coupling powers were not recoverable from the text. Treat 'an optimum exists' as supported and 'it lies at 140 mW' as suggestive.
- THE NUMERICAL AGREEMENT ON THE OPTIMUM'S LOCATION IS PROBABLY COINCIDENTAL. Their optimum sits near Omega_c/2pi = 12.2 MHz and RydSim predicted 12.85 MHz, but their probe waist is 83 um against our 1 mm, giving a transit rate of 1.69 MHz versus our ~40 kHz — a ~40x difference in a dephasing term that helps set the optimum. Their Rydberg state was not recovered either. This is NOT a validated absolute prediction.
- alpha here is measured along the BEATNOTE-FREQUENCY axis at fixed configuration, i.e. the NEF(delta) roll-off. That is the receiver-relevant trade for a fixed sensor, but it is not identical to a trade traced by re-optimising the sensor at each bandwidth.
- One apparatus, one group, one geometry. This is the field's first within-apparatus exponent, not a universal constant.

## Reproduce

```bash
rydsim run --config <saved config for 8ccdcf36427aa93a>
```

*Reproducible or it didn't happen.*
