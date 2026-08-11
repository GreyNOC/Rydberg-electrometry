# Confirming RydSim's findings — an experimental protocol

**Purpose.** Everything RydSim has produced is simulation. This document lists
the experiments that would confirm or refute its findings, from desk checks
costing nothing to the one benchtop measurement that would be a genuine
contribution to the field.

**Read this first.** The predictions below are stated *in advance* and are
falsifiable. That is the point. If an experiment returns something outside the
stated band, the simulator is wrong and the finding must be retracted — this
program has already retracted one finding (`findings/d3-trade-law-*`) for
exactly that reason. Do not adjust the prediction after seeing data.

**Convention warning.** Rydberg electrometry is riddled with ±3 dB traps:
amplitude vs RMS field, one-sided vs two-sided PSD, RBW vs ENBW, and
log-averaging under-read (spec 08 §2.7). Every number here is an **amplitude**
spectral density, one-sided, unless stated otherwise. Record which convention
your instrument uses before comparing anything.

---

## Tier 0 — Desk checks

**Cost: nothing. Time: an hour. Equipment: a calculator.**

These confirm arithmetic and analytic identities, not physics. They are worth
doing first because they are the cheapest way to catch a whole class of error,
and two real defects in this project were found exactly this way.

### T0.1 — The absorption-chain sum rule

**Tests:** the absolute susceptibility chain (density → dipole → χ → optical
depth), which underpins every transmission number RydSim produces.

**Procedure**

1. For a closed two-level transition the resonant absorption cross-section is
   exactly σ₀ = 3λ²/2π, independent of every atomic detail.
2. Evaluate at the Rb D2 wavelength λ = 780.241 nm.
3. Compare to RydSim: `tests/test_spec06_absolute.py::test_b2_resonant_cross_section_sum_rule`.

**Prediction:** σ₀ = 2.9070 × 10⁻¹³ m² = 2.9070 × 10⁻⁹ cm².

**Falsifies if:** the values differ by more than 1 part in 10⁸. Any mismatch
means a prefactor error in the χ chain, and every optical-depth and NEF number
downstream is wrong by that factor.

### T0.2 — The probe dipole (the defect this program actually shipped)

**Tests:** the correction described in `CHANGELOG` under spec 10 R10-10 — the
one that made the medium 2.40× too absorbing.

**Procedure**

1. Take the Steck reduced dipole for Rb-87 D2: ⟨J‖er‖J'⟩ = 4.22752 e·a₀
   (Steck, *Rubidium 87 D Line Data*, rev 2.3.4 — a free PDF).
2. The far-detuned π element for an unresolved excited hyperfine structure is
   d_eff = ⟨J‖er‖J'⟩/√3.
3. Compare with the closed-cycling dipole derived from Γ = ω³d²/(3πε₀ħc³) with
   Γ/2π = 6.0666 MHz.

**Prediction:** d_eff = 2.44076 e·a₀; cycling = 2.98930 e·a₀; ratio² = 1.500.
With the ground hyperfine fraction p_F = 5/8 for Rb-87 F=2, the total
optical-depth error from using the wrong element is **2.40×**.

**Falsifies if:** you get a materially different ratio. This is pure arithmetic
from published constants — if it disagrees, the correction is wrong.

### T0.3 — Beer–Lambert exactness

**Tests:** spec 10 R10-1/R10-3, which overturned this program's stated plan.

**Procedure**

1. Note that in strict linear response χ does not depend on the probe field.
2. Then dE/dz = i(k/2)χE is linear with constant coefficient.
3. Integrate it numerically (any ODE solver, any step size small enough) and
   compare to exp(−k·Im χ·L) over optical depths from 10⁻⁶ to 500.

**Prediction:** they agree to solver precision at **all** optical depths.
Beer–Lambert is the exact solution, not a thin-medium approximation.

**Consequence if confirmed:** "thick cell" is not itself a modelling problem.
What fails at depth is the *weak-probe* and *undepleted-coupling* assumptions,
which optical depth does not measure. Any gate placed on OD alone — including
the one this program shipped — is gating the wrong quantity.

---

## Tier 1 — Independent computational replication

**Cost: nothing. Time: a day or two. Equipment: a computer.**

This is the highest value-per-dollar tier and it genuinely confirms findings:
it re-derives RydSim's numbers with independently written, widely used code.
If ARC and QuTiP agree with RydSim, the atomic data and the master-equation
solution are not the error source.

### T1.1 — Atomic data cross-check against ARC

**Tests:** quantum defects, radial matrix elements, dipole moments, lifetimes —
the whole spec 01–04 chain.

**Materials**

- Python 3.11+
- `pip install ARC-Alkali-Rydberg-Calculator`

**Procedure**

1. For Rb-87, compute with ARC the radial matrix element ⟨50S₁/₂|r|50P₃/₂⟩.
2. Compute the RF transition frequency 53D₅/₂ → 54P₃/₂ for Rb-85.
3. Compute the effective lifetime of 50D₅/₂ at 300 K.
4. Compare each against RydSim (`rydsim.radial.radial_matrix_element_consensus`,
   `rydsim.atom.rf_transition_hz`, `rydsim.lifetimes.effective_lifetime_fit`).

**Predictions**

| Quantity | RydSim | Tolerance |
|---|---|---|
| Rb-87 ⟨50S₁/₂‖r‖50P₃/₂⟩ | 2510.91 a₀ | 1 % |
| Rb-85 ν(53D₅/₂→54P₃/₂) | 14.2317 GHz | 5 MHz |
| Rb-87 τ_eff(50D₅/₂, 300 K) | ≈ 65 µs | 10 % |

**Falsifies if:** any disagreement beyond tolerance. Note ARC is *not* fully
independent — RydSim's quantum defects are partly ARC-sourced (tagged
VERIFIED-ARC), so agreement on defects is weaker evidence than agreement on
matrix elements, which the two codes compute by different routes.

### T1.2 — Master equation cross-check against QuTiP

**Tests:** the Lindblad solver and EIT lineshape — spec 06.

**Materials**

- `pip install qutip`

**Procedure**

1. Build the 3-level ladder Hamiltonian in QuTiP with the spec 00 convention:
   H/ħ = −Δ_p|e⟩⟨e| − (Δ_p+Δ_c)|r⟩⟨r| − ½(Ω_p|g⟩⟨e| + Ω_c|e⟩⟨r| + h.c.).
2. Collapse operators: √Γ_e|g⟩⟨e|, plus Rydberg dephasing.
3. Solve for the steady state with `qutip.steadystate`, extract ρ_eg.
4. Sweep the probe detuning and compare the resulting EIT lineshape to
   RydSim's `rydsim.lindblad.LadderSystem.steady_state`.

**Prediction:** agreement to better than 10⁻⁶ relative at every detuning.
Both solve the same linear system; disagreement means a convention error in
one of them — most likely a factor-2 in a Rabi frequency or a sign in a
detuning.

**Note:** this checks the *homogeneous* solution. The Doppler average is where
RydSim previously had a real bug (quadrature aliasing), so also compare a
velocity-averaged spectrum if you can afford the QuTiP runtime.

### T1.3 — Reproduce the retraction

**Tests:** the honesty of this program's own record.

**Procedure**

1. Read `findings/d3-trade-law-*` — the original finding and its successor.
2. Re-run the original campaign configuration.
3. Confirm the optical-depth gate now refuses it, or that the corrected
   absorption chain changes the answer.

**Why do this:** if you cannot reproduce a program's retraction, you should not
trust its findings either.

---

## Tier 2 — Benchtop absorption spectroscopy

**Cost: roughly $3 000 – $8 000, or free if you have access to an
undergraduate atomic-physics lab. Time: a week.**

This is the cheapest *physical* experiment that tests something RydSim actually
got wrong, and it needs no Rydberg excitation at all — only the 780 nm probe.
It directly confirms or refutes the 2.40× absorption correction (T0.2).

### Materials

| Item | Spec | Approx. cost |
|---|---|---|
| ECDL or DFB laser, 780 nm | ≥ 5 mW, mode-hop-free scan ≥ 6 GHz | $2 000 – $6 000 |
| Rb vapor cell, natural abundance | 75 mm × 25 mm, no buffer gas | $300 – $1 500 |
| Photodiode + transimpedance amp | DC – 1 MHz, low noise | $150 – $600 |
| Oscilloscope | ≥ 2 channels, ≥ 10 kSa/s | often already owned |
| Neutral-density filters | OD 1–3, to keep the probe weak | $100 |
| Cell heater + thermocouple | ambient to 60 °C, ±0.5 K | $200 – $800 |
| Optics, mounts, breadboard | mirrors, λ/2, PBS, irises | $500 – $2 000 |
| Fabry–Pérot etalon *(optional)* | for a linear frequency axis | $1 000 – $3 000 |

**Safety.** 780 nm is invisible and this is a Class 3R/3B beam. Laser safety
eyewear rated for 780 nm, beam at bench height, no reflective jewellery, and
formal laser-safety training before switching on. Rubidium cells contain an
alkali metal — do not break one; handle per the supplier's MSDS.

### Procedure

1. **Set up.** Collimate the probe to a 1–2 mm waist through the cell onto the
   photodiode. Keep the beam single-pass and normal to the cell windows to
   avoid etalon fringes.
2. **Make the probe genuinely weak.** Attenuate until the measured absorption
   depth stops changing with power. Record the power at which it saturates,
   then work at least 10× below it. *This step is not optional:* the entire
   prediction assumes the weak-probe limit, and a strong probe bleaches the
   line and flatters the model.
3. **Calibrate the frequency axis.** Either use the known Rb-85/Rb-87 hyperfine
   splittings as fiducials, or an etalon of known FSR. Do not trust the laser's
   piezo voltage as linear in frequency.
4. **Take a reference.** Scan with the cell removed (or well below room
   temperature) to record the laser power envelope P₀(ν).
5. **Measure.** Stabilise the cell at 20 °C. Scan across the full Doppler-
   broadened D2 manifold and record P(ν). Transmission is P(ν)/P₀(ν).
6. **Repeat** at 25, 30, 40 and 50 °C, allowing ≥ 20 min to reach thermal
   equilibrium at each point and recording the cell temperature, not the
   heater setpoint.
7. **Extract** the peak optical depth OD = −ln(T_min) at each temperature.

### Predictions (75 mm natural-Rb cell, weak probe, Rb-87 F=2 line)

| Cell temperature | Predicted peak OD | Predicted T_min |
|---|---|---|
| 20 °C | 0.317 | 0.728 |
| 25 °C | 0.539 | 0.583 |
| 30 °C | 0.900 | 0.407 |
| 40 °C | 2.462 | 0.085 |
| 50 °C | 5.892 | 0.0028 |

**Tolerance.** The vapour-pressure model itself carries ±5 % (Alcock), and
d ln n/dT ≈ 0.106 /K near room temperature — so **a 1 K temperature error moves
the density by 11 %** and dominates everything else. Agreement within ±20 % on
OD, with the correct *trend*, confirms the chain. Measure your cell temperature
well or this experiment cannot discriminate.

**What this confirms.** If measured OD matches the table, the corrected
d_eff/p_F chain is right. **If measured OD comes in ≈ 2.4× higher than
predicted, the correction was wrong and the original cycling-dipole chain was
right** — that is the specific falsification this experiment exists to deliver.

**Caution at 50 °C:** T_min = 0.003 means you are measuring 0.3 % transmission;
detector dark current and stray light will dominate. Treat the 40 and 50 °C
rows as order-of-magnitude only unless your detector floor is characterised.

---

## Tier 3 — The decisive experiment

**Cost: roughly $120 000 – $250 000 for a full build, or a collaboration with
an existing Rydberg group. Time: months.**

This is the one that matters. Our central finding is that **the
sensitivity–bandwidth trade is weak** — α ≈ 0.18, where the folklore
constant-product law needs α ≈ 1. As far as this program's literature audit can
establish, *nobody has published a sensitivity–bandwidth curve measured within
a single apparatus by varying one parameter.* The published "frontier" is
assembled from incomparable points across different labs, species, transitions
and bandwidth definitions.

Measuring one curve, in one apparatus, would settle it.

### Materials

| Item | Spec | Approx. cost |
|---|---|---|
| Probe laser, 780 nm ECDL | frequency-stabilised, < 500 kHz linewidth | $8 000 – $20 000 |
| Coupling laser, 480 nm | frequency-doubled 960 nm, **≥ 500 mW**, < 1 MHz linewidth | $60 000 – $120 000 |
| Wavemeter | ≥ 100 MHz absolute accuracy | $8 000 – $25 000 |
| Rb vapor cell | 50–75 mm, natural or enriched | $300 – $1 500 |
| Reference cell + sat-abs lock | for probe stabilisation | $2 000 – $5 000 |
| RF synthesiser ×2 | to ≥ 20 GHz, low phase noise (LO + signal) | $20 000 – $60 000 |
| Horn antenna, calibrated | matched to the chosen Rydberg transition | $1 500 – $6 000 |
| Photodetector, low-noise | DC – 10 MHz, NEP ≤ 5 pW/√Hz | $1 000 – $3 000 |
| Lock-in amplifier / spectrum analyser | ENBW-configurable | $5 000 – $25 000 |
| Anechoic enclosure or absorber | to control RF reflections | $2 000 – $15 000 |
| Optics, AOMs, mounts, table | vibration-isolated | $15 000 – $40 000 |

The 480 nm coupling laser is the budget. **The coupling power requirement is
what makes this expensive, and it is also the independent variable** — see
below.

**Safety.** Class 3B/4. 480 nm at 500 mW is a serious hazard and, unlike the
780 nm, it is visible and will trigger a blink reflex that does *not* protect
you. Interlocked enclosure, wavelength-specific eyewear for both lines,
institutional laser-safety sign-off. Do not attempt this without a trained
laser-safety officer.

### Procedure

1. **Establish Rydberg EIT.** Counter-propagate probe and coupling through the
   cell. Lock the probe to the D2 line. Scan the coupling and find the EIT
   transparency peak for your chosen Rydberg state (e.g. Rb-87 60D₅/₂).
2. **Confirm the ladder.** Apply a known resonant RF field and observe
   Autler–Townes splitting. Verify the splitting is linear in field amplitude
   — this is the self-calibration that makes the sensor SI-traceable.
3. **Set up superheterodyne readout.** Apply a strong resonant LO field and a
   weak signal field offset by δ (150 kHz is a common choice). Demodulate the
   probe transmission at δ.
4. **Measure NEF.** With the signal off, record the noise spectral density at
   δ. With a *known* signal field on, record the response. NEF = (known field)
   × 10^(−SNR_dB/20) / √ENBW. **Record your ENBW explicitly** — the 1 Hz RBW vs
   1 s integration confusion is a factor √2 and is the most common error in
   this measurement.
5. **Measure IBW.** Sweep δ and find where the demodulated response falls to
   1/√2 of its low-δ value. **Report this as the atomic response bandwidth, and
   verify your detection chain is faster than the atoms** — otherwise you are
   measuring your electronics, which is the defect our literature audit found
   in much of the published record.
6. **NOW VARY ONE PARAMETER.** Change the coupling Rabi frequency Ω_c by
   varying coupling power, and repeat steps 4–5 at each setting. Ω_c ∝ √P, so
   calibrate it from the observed AT splitting rather than assuming.
7. **Plot** NEF against IBW on log–log axes and fit the slope above the NEF
   minimum.

### Predictions (Rb-87 60D₅/₂ → 61P₃/₂, 300 K, 5 cm cell)

| Ω_c/2π | Predicted NEF | Predicted IBW |
|---|---|---|
| 3 MHz | 0.68 nV/cm/√Hz | 6.2 MHz |
| 6 MHz | 0.47 | 12.8 |
| 13 MHz | **0.42 (minimum)** | 21.7 |
| 25 MHz | 0.44 | 41.8 |
| 46 MHz | 0.53 | 76.9 |

**The three falsifiable claims, in order of importance:**

1. **NEF is U-shaped in Ω_c, with an interior minimum near 13 MHz.**
   Falsified if NEF decreases monotonically across the whole accessible range.
2. **The trade branch is weak: α ≈ 0.18.** A 3.55× bandwidth increase should
   cost only 1.26× in sensitivity. **Falsified if α ≳ 0.5**, and decisively so
   if α ≈ 1 (the folklore constant-product law).
3. **The minimum's location scales predictably** with Γ_e and Rydberg
   dephasing, not with atom number.

### Read this before comparing absolute numbers

**Do not expect the absolute NEF values to match.** RydSim predicts ≈ 0.4
nV/cm/√Hz; the published state of the art is ≈ 10 nV/cm/√Hz, and real hardware
will land near the published figure or worse. The model carries only photon
shot noise, RIN, detector NEP and the projection floor, with a perfectly locked
on-resonance probe. It omits laser frequency noise, servo residuals, cell
etalon effects, and RF field inhomogeneity — and two systematics remain
*unbounded* in the model (RF internal field, radiation trapping; see spec 10).

**The transferable claim is the exponent and the optimum's existence, not the
scale.** That is the whole thesis of a laboratory-less program: absolute
sensitivities are hostage to uncalibrated parameters, while relative structure
survives. An experiment that finds α ≈ 0.18 while measuring 10 nV/cm/√Hz
absolute *confirms* our finding. One that finds α ≈ 1 refutes it, regardless of
the absolute level.

---

## What each tier actually buys you

| Tier | Cost | Confirms |
|---|---|---|
| 0 | nothing | arithmetic, prefactors, analytic identities |
| 1 | nothing | atomic data and the master-equation solution, against independent codes |
| 2 | $3–8 k | the absorption chain and the 2.40× dipole correction — a real defect this program shipped |
| 3 | $120–250 k | the headline finding, and a measurement the field appears never to have published |

If you can only do one: **Tier 1**, today, for free. If you can fund one
physical experiment: **Tier 2**, because it tests a specific correction with a
sharp two-way falsification criterion. Tier 3 is a research programme, best
done as a collaboration with a group that already owns the 480 nm laser.

---

*GreyNOC · RydSim · predictions generated from the validated engine at the
commit recorded in each finding's `config_hash`. Not a certified metrology
instrument. If an experiment contradicts these numbers, the simulator is wrong
and this document is the receipt.*
