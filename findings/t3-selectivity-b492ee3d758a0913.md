# Finding: T3 selectivity: blocking vs interferer offset [rb87-cold-thin-1000ea0]

*GreyNOC RydSim 0.2.0 · 2026-11-25T02:34:06+00:00 · config `b492ee3d758a0913` · python 3.13.18 / Linux 6.18.44-fc-v880*

## Result

- **LO**: Omega_LO = 0.5433 MHz, steepest-slope point of the thin transfer curve, 0.04246 V/m
- **hidden-signal offset**: 13 kHz
- **atomic response bandwidth (resolvent)**: 4.637e+05 Hz
- **far-offset rejection**: 10.33 dB/decade in field (exponent 0.516; AC-Stark limit 0.5 = 10 dB/decade)
- **AC-Stark critical shift**: s_crit = 77.16 kHz; asymptotic 1 dB blocking Omega_I = sqrt(4 Delta s_crit)
- **selectivity (offset | block @signal | block @DC | AC-Stark law)**: ['0.05 MHz | 0.4607 x Omega_LO (0.01956 V/m) | 0.4549 x Omega_LO (0.01931 V/m) | 0.2286 x Omega_LO', '0.07924 MHz | 0.5118 x Omega_LO (0.02173 V/m) | 0.5047 x Omega_LO (0.02143 V/m) | 0.2878 x Omega_LO', '0.1256 MHz | 0.5928 x Omega_LO (0.02517 V/m) | 0.5775 x Omega_LO (0.02452 V/m) | 0.3624 x Omega_LO', '0.1991 MHz | 0.694 x Omega_LO (0.02947 V/m) | 0.6763 x Omega_LO (0.02872 V/m) | 0.4562 x Omega_LO', '0.3155 MHz | 0.782 x Omega_LO (0.03321 V/m) | 0.7586 x Omega_LO (0.03221 V/m) | 0.5743 x Omega_LO', '0.5 MHz | 0.8123 x Omega_LO (0.03449 V/m) | 0.8005 x Omega_LO (0.03399 V/m) | 0.723 x Omega_LO', '0.7924 MHz | 0.8097 x Omega_LO (0.03438 V/m) | 0.8048 x Omega_LO (0.03417 V/m) | 0.9103 x Omega_LO', '1.256 MHz | 0.7876 x Omega_LO (0.03344 V/m) | 0.7948 x Omega_LO (0.03375 V/m) | 1.146 x Omega_LO', '1.991 MHz | 0.7938 x Omega_LO (0.03371 V/m) | 0.8184 x Omega_LO (0.03475 V/m) | 1.443 x Omega_LO', '3.155 MHz | 1.04 x Omega_LO (0.04415 V/m) | 1.124 x Omega_LO (0.04771 V/m) | 1.816 x Omega_LO', '5 MHz | 1.64 x Omega_LO (0.06964 V/m) | 1.807 x Omega_LO (0.07671 V/m) | 2.286 x Omega_LO', '7.924 MHz | 2.341 x Omega_LO (0.09942 V/m) | 2.604 x Omega_LO (0.1106 V/m) | 2.878 x Omega_LO', '12.56 MHz | 3.156 x Omega_LO (0.134 V/m) | 3.419 x Omega_LO (0.1452 V/m) | 3.624 x Omega_LO', '19.91 MHz | 4.033 x Omega_LO (0.1712 V/m) | 4.383 x Omega_LO (0.1861 V/m) | 4.562 x Omega_LO', '31.55 MHz | 5.107 x Omega_LO (0.2168 V/m) | 5.561 x Omega_LO (0.2361 V/m) | 5.743 x Omega_LO', '50 MHz | 6.445 x Omega_LO (0.2737 V/m) | 7.024 x Omega_LO (0.2983 V/m) | 7.23 x Omega_LO']

## Uncertainty budget

- **Floquet truncation**: sideband count doubled until the gain moved < 1e-8 (relative) at every point
- **blocking interpolation**: log-interpolated on a 41-point log grid from 0.01 to 100 x Omega_LO
- **time-domain agreement**: harmonic balance vs direct OBE integration agrees to ~1e-8 (tests/test_floquet.py)

## Cross-method checks

- {'quantity': 'far-offset blocking at 50 MHz (signal -> DC)', 'methods': {'Floquet': 7.024316980817571, 'AC-Stark law': np.float64(7.230452601934347)}, 'spread_rel': np.float64(0.02850936621334457), 'tolerance_rel': 0.05}
- {'quantity': 'in-band blocking at 2.319 kHz = 0.005 x IBW (signal -> DC)', 'methods': {'Floquet': 0.42014390705602067, 'quasi-static phase average': 0.4200673342497199}, 'spread_rel': 0.00018225375880682565, 'tolerance_rel': 0.002}

## Validation state

- see rydsim validate (tests/test_floquet.py)

## Constants on the critical path

- (none recorded)

## Caveats (standing model limitations)

- Model potential / quantum-defect accuracy bounds apply to all atomic data.
- Neglected: Rydberg-Rydberg interactions at high density, free-ion fields, cell-wall microphysics beyond the phenomenological screening model.
- Laser technical noise modeled as stationary PSDs; real lasers drift.
- Steady-state OBE: dynamics faster than the atomic response bandwidth are outside the model's validity.
- Single velocity class: no Doppler averaging. Hot-vapor selectivity will differ.
- Optically thin readout (probe absorption ~ Im rho_10): absolute blocking levels differ from the thick-medium `cosite` ones; the offset dependence is what this finding claims.
- Gain desensitization only: the noise budget is not re-evaluated (constant to first order in the thin limit).
- 4-level ladder: interferers far enough off to reach OTHER Rydberg transitions are outside the model. Check the real level structure before trusting GHz offsets.
- Blocking depends on the hidden signal's own offset: the interferer's light shift opens a slow (Rydberg-decay-scale) response mode, so a kHz-offset signal is blocked earlier than a near-DC one. Both columns are reported.

## Reproduce

```bash
rydsim run --config <saved config for b492ee3d758a0913>
```

*Reproducible or it didn't happen.*
