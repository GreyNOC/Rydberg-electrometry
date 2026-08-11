# 10 — Optically Thick Propagation · §6 Validation Corpus

**RydSim physics specification, module 10, validation section. Status: subordinate to
`00-conventions.md` (20 locks, R-1…R-28) and `00-integrity-audit.md` (§3 refusal list).**
Network was AVAILABLE during authoring; every row below carries a source and one of
VERIFIED (primary fetched this session and quoted), VERIFIED-COMPUTED (reproduced numerically
this session from VERIFIED inputs, script named), VERIFIED-ARC, LITERATURE-RECALL, UNVERIFIED,
MISSING.

**One-line statement of the problem this corpus exists to solve.** `rydsim.experiment`
currently raises `IntegrityError` above `max_optical_depth = 5.0`
(`src/rydsim/experiment.py:102, 317`). Every flagship vapour-cell electrometry experiment in
the published record operates *at or above* that ceiling and simultaneously *outside* the
weak-probe gate. This corpus is the set of numbers that decides whether the replacement
propagation solver is right.

---

## 6.1 Design principle: a benchmark that a thin-cell solver can pass is not a benchmark

The single most important structural fact about this module — and the reason a naive corpus
would be worthless — is:

> **For a linear medium (χ independent of the probe field, coupling undepleted),
> `T = exp(−k_p Im χ L)` is EXACT at every optical depth, not just small OD.**

Beer–Lambert is not a thin-cell *approximation*; it is the closed-form solution of the
propagation equation when χ is z-independent. So a benchmark that only varies OD at weak probe
**cannot** distinguish the new solver from `rydsim.eit.transmission()`. It only tests the
z-integrator. Both classes of row are needed, and they must be labelled differently:

| Class | What it tests | Can the existing thin-cell code pass it? |
|---|---|---|
| **Structural (S)** | z-integrator correctness, step control, reduction limits | Yes, by construction — these are *consistency* gates |
| **Discriminating (D)** | physics the thin path cannot produce | **No** — a solver that silently reduces to old behaviour fails |

The three physical channels that make the medium genuinely nonlinear along z, in the order
their magnitude was established below:

1. **Probe saturation with de-saturation along z** (dominant; §6.5 shows all three flagship
   experiments run at s₀ = 2Ω_p²/Γ_e² ≈ 2–49).
2. **Coupling depletion** (sub-dominant in Rydberg ladders — the intermediate state is nearly
   empty — but must be *measured*, not assumed: benchmark P-17).
3. **Transverse × longitudinal coupling** (radial averaging and z-propagation do not commute
   once the medium saturates: benchmark P-18).

---

## 6.2 Symbols and units used by this corpus

| Symbol | Meaning | SI unit | Notes / normative source |
|---|---|---|---|
| `z` | propagation coordinate along the probe | m | probe along +z, coupling counter-propagating (lock #10, spec 06 §2.6) |
| `L` | cell (or interaction) length | m | *interaction* length where it differs from the glass length (Sedlacek) |
| `α(z,ν)` | local power absorption coefficient | m⁻¹ | `α = k_p Im χ` (lock, spec 00 §2 row "α (absorption)") |
| `OD` | optical depth | — | `OD = ∫₀^L α dz`; **linear/weak-probe OD** written `OD_lin = α₀L` |
| `α₀` | unsaturated (weak-probe) absorption coefficient | m⁻¹ | evaluated at Ω_p → 0 |
| `T` | **intensity** transmission | — | `T = P_out/P_in`; never amplitude (lock #3) |
| `s₀` | on-resonance saturation parameter | — | `s₀ ≡ I/I_sat = 2\|Ω_p\|²/Γ_e²` under lock #4 (Ω = dℰ/ħ, ℰ peak) |
| `I_sat` | saturation intensity | W/m² | must be paired with the matching dipole (spec 00 §2 row `I_sat`) |
| `Ω_p(z)`, `Ω_c(z)` | z-resolved Rabi frequencies | rad/s | recomputed from (d, ℰ(z)) — never imported (R-22) |
| `Δφ` | accumulated dispersive phase | rad | `Δφ = (k_p/2)∫Re χ dz` (spec 05 §2.f) |
| `C_EIT` | EIT peak height / contrast | — | `T_peak − T_baseline`, the definition used by Su 2022 |
| `r` | EIT contrast ratio | — | `r ≡ α_max/α_min` (off-resonance vs on-EIT absorption) |
| `κ_E` | transduction slope | W/(V/m) | spec 08; `NEF = √S_P/\|κ_E\|` |
| `w₀p, w₀c` | probe / coupling 1/e² intensity radii | m | lock, spec 00 §2 row `w₀` |

Propagation equation (spec 05 §2.f, normative):

```
dΩ_p/dz = i (k_p/2) · χ(Ω_p(z), Ω_c(z); z) · Ω_p(z)
```

with `χ` re-evaluated — including the full velocity integral (R-2 grid) — at every z level.

---

## 6.3 Tier A — analytic limits (GATING; these cannot be fudged)

All five were **derived and numerically verified in this session**; the scripts are
`scratchpad/verify_thick.py` and `scratchpad/verify_b9.py`. Each row states its falsifier.

### A1. Linear-medium exactness at arbitrary OD (P-1)

With χ frozen (weak probe, undepleted coupling), the solver must return `exp(−OD_lin)` to
integrator tolerance **at OD = 0.01, 1, 5, 14.216, 20**, not merely at small OD.

*Why this is not trivial:* measured this session, an RK4 z-integrator using a fixed 16 steps
returns

| OD_lin | 16 steps, rel. error | 64 steps | spec-05 step rule (`ΔOD ≤ 0.05` ⇒ n ≥ 20·OD) |
|---|---|---|---|
| 1.000 | 1.34e−7 | 5.03e−10 | — |
| 5.000 | 5.16e−4 | 1.66e−6 | n = 100 → **2.72e−7** |
| 14.216 | **1.68e−1** | 3.47e−4 | n = 285 → **7.65e−7** |
| 20.000 | **2.09e+0 (209 %)** | 2.07e−3 | n = 400 → **1.09e−6** |

**Falsifier:** any solver whose high-OD error exceeds 1e−4 relative, or whose step controller
does not enforce `|ΔOD| ≤ 0.05` per step. The 209 % figure at OD = 20 is why spec 05's step
rule is normative rather than advisory.

### A2. The OD → 0 reduction, with its exact second-order coefficient (P-2)

This is the row the brief calls the single most important check, and "it converges" is too
weak a statement — the *coefficient* is known in closed form. For a homogeneously broadened
saturable medium of unsaturated depth `x ≡ α₀L` and input saturation `s ≡ I_in/I_sat`:

```
T_prop − T_thin  =  −(1/2)·s/(1+s)³ · x²  +  O(x³)
      where T_thin = exp(−x/(1+s))   (frozen-χ Beer–Lambert at the INPUT intensity)
```

Derived here by series inversion of the implicit law of A3; measured this session at
x = 1e−4:

| s | measured `(T_prop−T_thin)/x²` | analytic `−s/(2(1+s)³)` |
|---|---|---|
| 0.25 | −0.06398733 | −0.06400000 |
| 1.00 | −0.06249686 | −0.06250000 |
| 3.00 | −0.02343715 | −0.02343750 |
| 10.0 | −0.00375656 | −0.00375657 |

**Falsifier:** a solver that reduces to the thin answer with the *wrong* order (O(x) or O(x³))
or the wrong coefficient. A solver that is "correct by accident" as OD → 0 — e.g. one that
simply calls the thin path below a threshold — fails, because the ratio at finite x will not
track −s/(2(1+s)³). **This row is the one that catches silent reduction to the old behaviour.**
Note the limit is *doubly* correct: at s → 0 the coefficient vanishes and A1 (exactness at all
OD) takes over.

### A3. Saturable-absorber implicit transmission law (P-3, the strongest discriminator)

For `dI/dz = −α₀ I/(1 + I/I_sat)` (homogeneously broadened, single velocity class, Ω_c = 0):

```
ln T + (I_in/I_sat)·(T − 1) = −α₀ L
      equivalently   T = W( s·T₀·e^s ) / s ,   s = I_in/I_sat,  T₀ = e^{−α₀L}
```

(W = Lambert W; this is the "LambertW model" of fast/steady-state saturable absorbers in the
laser literature.) Verified this session at α₀L = 3.0:

| I_in/I_sat | T (implicit law) | T (RK4, 4000 steps) | rel. diff | Beer–Lambert `e^{−α₀L}` | BL error |
|---|---|---|---|---|---|
| 1e−6 | 0.0497871157 | 0.0497871157 | 7.4e−13 | 0.049787 | 0.00 % |
| 0.1 | 0.0547229392 | 0.0547229392 | 1.5e−12 | 0.049787 | 9.02 % |
| 1.0 | 0.1200282390 | 0.1200282390 | 1.5e−13 | 0.049787 | **58.5 %** |
| 10 | 0.7312939746 | 0.7312939746 | 2.3e−15 | 0.049787 | **93.2 %** |

**Falsifier:** any solver reproducing `e^{−α₀L}` at s ≳ 0.1 is running the thin path. The
discriminating power is enormous (a factor 15 in T at s = 10) and requires no literature at all.
The RydSim-side identity that makes this executable is `s₀ = 2|Ω_p|²/Γ_e²` under lock #4, which
must itself be asserted from the ladder solver's own two-level steady state (Ω_c = 0), not
typed in.

### A4. Shot-noise-limited NEF is optimal at OD = 2 (P-7)

For a linear medium of fixed density, `κ_E = P_in e^{−OD}·(∂OD/∂E)` with `∂OD/∂E ∝ L ∝ OD`, and
shot noise `S_P ∝ P_out = P_in e^{−OD}`. Hence

```
NEF(OD) ∝ e^{OD/2} / OD     ⇒     OD_opt = 2 exactly
```

Verified this session: numerical argmin over OD ∈ [0.05, 8] with 8×10⁵ samples gives
**OD_opt = 1.999996**. Penalties: `NEF(2)/NEF(0.1) = 0.1293`, `NEF(2)/NEF(1) = 0.8244`,
`NEF(2)/NEF(5) = 0.5578`.

**Why this row matters more than it looks:** it is the quantitative statement that the
thin-cell regime (spec 05's OD ≤ 0.1 gate) is **7.7× worse in sensitivity than the optimum**.
The regime RydSim currently refuses is not a corner case; it is where the instrument works.

### A5. EIT contrast is maximised at a specific OD (P-8), and that OD is above the thin gate

With `C(L) = e^{−α_min L} − e^{−α_max L}` and `r = α_max/α_min`:

```
OD_bg*  ≡  α_max L*  =  ln r / (1 − 1/r)
```

Verified this session (2×10⁶-point scan):

| r | OD_bg* numeric | analytic | C_max |
|---|---|---|---|
| 2 | 1.38631 | 1.38629 | 0.25000 |
| 5 | 2.01181 | 2.01180 | 0.53499 |
| 10 | 2.55844 | 2.55843 | 0.69684 |
| 100 | 4.65168 | 4.65169 | 0.94500 |

`OD_bg* → 1` as r → 1⁺ and crosses **5.0 at r = 143.3** — i.e. the optimum is *always* above
spec 05's OD ≤ 0.1 thin gate, and for high-contrast media above RydSim's present refusal
ceiling.

### A6. Transparency-window narrowing as 1/√OD (P-9)

Expanding `α(δ) = α_min + a₂δ²` about two-photon resonance, the transmitted peak is Gaussian
in δ with

```
FWHM(window) = 2 √( ln 2 / (a₂ L) )     ⇒     FWHM ∝ OD^(−1/2)
```

Verified this session over six octaves in L: the width ratio between successive doublings is
**1.41421, 1.41421, 1.41422, 1.41422, 1.41422** against √2 = 1.414214.

**Named trap (measured here):** the common form quoted with a `2 ln 2` inside the root is the
half-*depth* convention; the half-*peak-transmission* convention gives `ln 2`, a factor √2
apart. Both appear in the literature. RydSim must define the width on the
baseline-subtracted signal and stamp the convention (a √2 error here is exactly the class of
bug lock #12's ±3 dB traps exist to prevent). *Confidence:* the scaling and the coefficient are
**derived + verified in-session**; the frequently cited RMP 77, 633 (2005) statement was
**NOT fetched** (paywalled) and is not relied upon.

---

## 6.4 Tier B — the two-level Doppler reference cell (GATING)

The Doppler-broadened, coupling-off Rb D2 profile is the one place where the corpus has an
experimentally validated absolute model. Spec 05 §2.f cites **Siddons, Adams, Ge & Hughes,
J. Phys. B 41, 155004 (2008)**.

**What Siddons actually publishes (VERIFIED, arXiv:0805.1139 abstract + ar5iv full text fetched
this session):** a 7.5 cm natural-Rb cell; temperatures 16.5 / 25.0 / 25.4 / 36.6 °C; weak
probe "32 nW/mm²" (I/I_sat = 0.002) and a hyperfine-pumping study at "1.6 μW/mm²"
(I/I_sat = 0.1); transmissions "ranging from 5 to 95 %"; and the headline
**"an rms error better than 0.2 % for the D₂ line at 16.5 degrees C"**.

**What Siddons does NOT publish:** tabulated transmission minima or peak optical depths. Spec
05's B9 table is therefore **the project's own model output**, correctly tagged "V-computed" —
it is *not* a transcription of Siddons numbers, and any release note claiming otherwise would
be a fabrication. The Siddons contribution is the **method's 0.2 % rms experimental validation**,
which is what licenses using these rows as absolute anchors.

**Independent verification performed this session.** I re-implemented spec 05 Eq. 2.f from
scratch (`scratchpad/verify_b9.py`: Voigt sum over both isotopes, `S_FF'` from
`rydsim.angular`, Steck rev 2.3.4 hyperfine constants from `rydsim.atom`, densities from
`rydsim.cell`) without calling any RydSim propagation routine. Result — spec 05's B9 rows
**reproduce**:

| Spec-05 row | Printed | Recomputed this session | Δ |
|---|---|---|---|
| 25.0 °C, ⁸⁷Rb F=2: OD / T_min | 0.481 / 0.619 | **0.4805 / 0.618489** | <0.1 % |
| 25.0 °C, ⁸⁵Rb F=3: OD / T_min | 1.298 / 0.273 | **1.2982 / 0.273031** | <0.02 % |
| 25.0 °C, ⁸⁵Rb F=2: OD / T_min | 0.956 / 0.385 | **0.9558 / 0.384523** | <0.15 % |
| 25.0 °C, ⁸⁷Rb F=1: OD / T_min | 0.312 / 0.732 | **0.3124 / 0.731713** | <0.15 % |
| 16.5 °C, ⁸⁵Rb F=3: T_min (OD) | 0.594 (0.521) | **0.594036 (0.5208)** | <0.1 % |
| 50.0 °C, ⁸⁵Rb F=3: OD / T_min | 14.2 / 6.7e−7 | **14.2160 / 6.69992e−7** | <0.15 % |
| dip positions vs ⁸⁷Rb centroid | −2.424/−1.288/+1.619/+4.094 GHz | **−2.4241/−1.2883/+1.6195/+4.0942** | ≤0.5 MHz |

Supporting closures also reproduced: `N(Rb, 298.15 K) = 1.2918e16 m⁻³` (B3a: 1.292e16),
`N(Rb, 323.15 K) = 1.4672e17 m⁻³` (B3c: 1.467e17), `N(Cs, 298.15 K) = 4.8941e16 m⁻³`
(B3e: 4.894e16).

**These are the exact rows a propagation solver must reproduce (P-5, P-6).** The 50 °C row
carries **OD = 14.216**, i.e. **2.8× above the engine's present refusal ceiling** — it is the
gating high-OD stability row, and by A1 its correct answer is exactly `exp(−14.216)`.

**Amendment recommended to spec 05 (B9c).** The apparent ⁸⁷Rb F=2 dip position drifts with
temperature as the Doppler width reweights the hyperfine blend: measured **−2.4231 GHz at
16.5 °C, −2.4241 at 25.0 °C, −2.4267 at 51 °C, −2.4280 at 65 °C** — a 4.9 MHz span against
B9c's ±3 MHz tolerance. B9c must be restated as *per-temperature*, or its tolerance widened to
±6 MHz with the drift documented. Grading a high-OD spectrum against the 25 °C positions at a
±3 MHz window is a false failure waiting to happen.

**Second named trap at high OD:** at 50 °C the transmission floor is 6.7e−7 — a numerically
black region several hundred MHz wide. Dip *positions* must be recovered by fitting the wings
of `OD(ν) = −ln T`, never by `argmin T` on the transmission trace, and `exp(−OD)` needs the
spec 05 §5 underflow guard (OD > 700).

**Independent cross-validation (non-pytest, recommended):** ElecSus — Zentile, Keaveney,
Weller, Whiting, Adams & Hughes, *Comput. Phys. Commun.* **189**, 162 (2015),
arXiv:1409.1873 — is the community reference implementation of exactly this weak-probe
susceptibility. Compare, do **not** vendor (spec 05 §6 already says this).

---

## 6.5 Tier C — published Rydberg-EIT rows in thick cells, and what is actually checkable

Everything in this subsection was fetched this session. The decisive, program-level finding:

> **Every flagship vapour-cell electrometry experiment runs at OD of order 1–17 *and* with a
> saturating probe (s₀ ≈ 2–49) simultaneously.** Neither the thin-medium assumption nor the
> weak-probe assumption survives; the two failures are not independent and cannot be patched
> one at a time.

| Experiment | Cell | Ω_p / Γ_e | s₀ = 2Ω_p²/Γ_e² | weak-probe OD (computed here) |
|---|---|---|---|---|
| Mohapatra 2007 (Rb) | 75 mm, room T | — (P = 1 μW, w₀ = 0.4 mm ⇒ I/I_sat ≈ 0.16) | ≈0.16 | 1.298 (⁸⁵Rb F=3, 25 °C) |
| Sedlacek 2012 (Rb) | 10 cm glass, **7.5 cm effective**, room T | 6.0/6.0666 = 0.989 | **1.956** | **0.4805** (⁸⁷Rb F=2, 25 °C, 7.5 cm) |
| Jing 2020 (Cs) | 5 cm, room T, N₀ = 4.89e10 cm⁻³ | 5.7/5.234 = 1.089 | **2.372** | **4.639** (F=4 dip, 5 cm, 25 °C) |
| Su 2022 (Rb) | Thorlabs GC25075-RB, 27–65 °C | 30/6.0666 = 4.945 | **48.91** | 0.566 (27 °C) → 17.10 (65 °C) |

### C1. Jing et al., *Nat. Phys.* **16**, 911 (2020) — the best published thick-cell number

**Fetched verbatim from ar5iv:1902.11063 this session:**
- "The cell is [5]-cm-long and contains ground-state atoms with a total density N₀=4.89×10¹⁰ cm⁻³."
- "the 1/e² beam diameter is 1.70±0.04 mm, and the optical power incident to the vapor cell is 120±4 μW, yielding effectively Ωₚ=5.7±0.6 MHz."
- **"After absorption by Cs atoms, the power of the probe light incident on the detector is about 10 μW."**
- coupling: "1/e² beam diameter is 2.00±0.05 mm … incident optical power is 34±1 mW, yielding Ωc=0.97±0.12 MHz"; states 47D₅/₂ → 48P₃/₂ at 6.94 GHz; E_L = 3.0 mV/cm; sensitivity 55 nV cm⁻¹ Hz⁻¹ᐟ².

The 120 μW → ~10 μW pair is a **directly published end-to-end transmission through an
optically thick cell**: `T_meas = 0.0833`, `OD_eff = 2.485`. It is, as far as this session's
search found, the only such pair printed in the flagship electrometry literature.

**Cell-length adjudication (numerical, from published numbers alone).** The ar5iv rendering
reads "55-cm-long"; spec 08 §3.2 records 5 cm as VERIFIED from arXiv v1. **5 cm is the only
self-consistent reading:** at 55 cm the weak-probe OD would be 51.0, and even fully
Doppler-saturated at their own s₀ it is 27.8, giving `T = 8.5e−13` ⇒ 1e−10 μW at the detector
against a published ~10 μW. The 55 cm reading is excluded by ~11 orders of magnitude. Spec 08's
5 cm stands.

**The three-way bracket (this is the corpus's centrepiece).** Using Jing's own stated density
and the spec 05 §2.f Cs D2 model (computed this session, `verify_thick.py`):

| Model | Predicted `OD` | Predicted `T` | Predicted detector power | vs published ~10 μW |
|---|---|---|---|---|
| Weak-probe Beer–Lambert (present thin path) | 4.639 | 0.00966 | **1.16 μW** | **8.6× TOO LOW** |
| Frozen homogeneous saturation `α₀/(1+s₀)` | 1.376 | 0.2526 | **30.3 μW** | **3.0× TOO HIGH** |
| Frozen Doppler saturation `α₀/√(1+s₀)` | 2.527 | 0.0799 | 9.59 μW | ~ published |
| **Published (Jing v1)** | 2.485 | 0.0833 | **~10 μW** | — |

**Honest reading of that table.** The near-coincidence of the third row with the measurement
(1.7 % in OD) is **not** a validation — it is the net of several corrections that partly cancel
and that a real solver must model individually: (i) `α₀/√(1+s₀)` is itself a *frozen* law and a
z-resolved solver will predict *more* absorption as the probe de-saturates along z; (ii) Jing
operates on the EIT/AT resonance with the coupling and LO on, which raises T back; (iii)
uncoated borosilicate windows cost ~8–15 % (with 0.92/0.87 window factors the measured atomic
OD becomes 2.40/2.35, ratios to weak-probe 0.518/0.506 rather than 0.536). The inhomogeneous
saturation law `α = α₀/√(1+s₀)` is **LITERATURE-RECALL** (standard for inhomogeneously
broadened lines; no primary fetched this session) and is used here only to *bracket*, never to
grade.

**Therefore the row is split:** the **bracket direction is GATING** (P-10) and the absolute
power is **DIAGNOSTIC** at a factor-2 window (P-11).

### C2. Sedlacek et al., *Nat. Phys.* **8**, 819 (2012)

**Fetched verbatim from ar5iv:1205.4461 this session:** "room temperature 10 cm Rb vapor cell";
"the effective interaction length is 7.5 cm"; probe locked to ⁸⁷Rb 5S₁/₂(F=2)→5P₃/₂(F=3);
"the probe beam size is 750 μm"; "Ωₚ=2π×6 MHz"; "The coupling laser beam size is 100 μm";
"Ωc=2π×2 MHz"; "53D₅/₂→54P₃/₂ … ∼14.233 GHz"; "μᴿᶠ=1.37×10⁻²⁶ C m"; EIT window
"∼4−5 MHz"; effective laser linewidth "∼700 kHz"; **"A maximum increase of ∼4.5%"**;
sensitivity ∼30 μV cm⁻¹ Hz⁻¹ᐟ²; minimum field ∼8 μV cm⁻¹.

Quantitatively checkable: OD (0.4805 over 7.5 cm at 25 °C, computed here) and s₀ = 1.956.
Only *semi*-checkable: the ~4.5 % transmission increase, because
**w₀c/w₀p = 100/750 ⇒ only (100/750)² = 1.78 % of the probe area sees the coupling beam**. Any
prediction of Sedlacek's contrast therefore requires the spec 05 §2.g unequal-waist radial
average (exponent ratio `w₀p²/w₀c² = 56.25`) *composed with* z-propagation — which is exactly
benchmark P-18. Amplitude at the tens-of-percent level only (spec 06 §7.1). **DIAGNOSTIC.**

Note the standing tension already on the books: R-22 records that Sedlacek's printed
`℘ = 1.37e−26 C·m` is reproduced by neither member of the closed convention set. Nothing in
this corpus may be tuned to it.

### C3. Mohapatra, Jackson & Adams, *PRL* **98**, 113003 (2007)

**Fetched verbatim from ar5iv:quant-ph/0612200 this session:** "room temperature rubidium
vapor cell of length 75 mm"; probe "λp=780.24 nm", "power 1 μW", "beam size 0.4 mm (1/e²
radius)"; coupling "λc=479.2−483.9 nm", "power up to 200 mW", "spot size of 0.8 mm (1/e²
radius)"; "The line–width of the EIT resonance is between 22 and 44 MHz depending on the laser
power and the transition"; **"produces a change in the probe transmission of 5%"** (n = 45,
Ωc = 2π×3.5 MHz) and "the change in the probe transmission is reduced to about 1%" (n = 80,
Ωc = 2π×1.5 MHz).

This is the same 75 mm natural-Rb geometry as the Tier-B rows, so its OD is *already known
absolutely* from P-5 (1.298 at the ⁸⁵Rb F=3 dip at 25 °C). The probe: 1 μW at w₀ = 0.4 mm gives
I₀ = 3.98 W/m² = 0.16 I_sat(det-π) — **16× the spec 05 weak-probe gate**. The 5 % / 1 % pair is
a *ratio* test that is largely free of amplitude-model error (same cell, same probe, differing
only in n and Ω_c). **DIAGNOSTIC**, ratio graded tighter than absolutes.

*(Note: spec 06 B-12 currently grades this qualitatively — 3 booleans. This corpus upgrades it
to the quantitative 5:1 ratio, at ORDER tolerance.)*

### C4. Su, Liou, Lin & Chen, *Opt. Express* **30**, 1499 (2022) — the OD-scan paper

This is the paper the brief asks for: **EIT peak height vs optical depth, published
quantitatively.** Fetched verbatim from ar5iv:2111.13408 this session:

- "We performed the Rydberg EIT spectral measurements in an Rb vapor cell at the temperature ranging from 27∘C to 65∘C."
- **"The determined α varied from 0.42 (27∘C) to 5.0 (65∘C)."** (α is their symbol for OD)
- their baseline model: **`T_B = exp(−α Γe²/(Γe² + 2Ωp²))`** — a *frozen*, saturation-corrected Beer–Lambert
- probe 780 nm, "1.5 μW to 17 mW", optimum "0.044 W/cm²", waist "0.81 mm (full width at e⁻² maximum)" ⇒ w₀ = 0.405 mm, Ωp/2π = 30 MHz
- coupling 480 nm, "27 mW", "5.3 W/cm²", Ωc/2π = 0.38 ± 0.03 MHz
- states |33D₃/₂⟩, |33D₅/₂⟩, splitting "336.4 MHz"; B ≈ 0 optimal; "The EIT linewidth was around 10 MHz"
- **maximum EIT peak height "13%" at T = 51 °C**, Ip = 0.044 W/cm² — an *interior* optimum in temperature
- their fit uses "Γe/2π = 60 MHz (includes velocity group selection)" — an effective, not first-principles, width

**Cell length is absent from the paper text and figures.** The part number GC25075-RB resolves
(vendor listing, search-snippet level — a direct Thorlabs fetch failed this session) to
"Rubidium Borosilicate Reference Cell, Ø25.4 mm × **71.8 mm**", natural abundance, "transmission
through the cell exceeds 84 %". **Tag: LITERATURE-RECALL/vendor-snippet, not VERIFIED.** Any
absolute-OD row built on it must carry that tag.

**The quantitative discriminator this paper hands us (P-15).** Computed this session at
L = 71.8 mm, natural Rb, ⁸⁷Rb F=2 dip:

| T_cell | spec-05 model OD (this session) | Su's fitted α | ratio |
|---|---|---|---|
| 27 °C | 0.566 | 0.42 | 1.35 |
| 51 °C (their optimum) | **5.521** | — | — |
| 65 °C | 17.096 | 5.0 | **3.42** |
| **ratio 65/27** | **30.19** | **11.90** | **2.54** |

The **65/27 ratio is independent of cell length and of isotopic enrichment** (both cancel), so
the 30.2 vs 11.9 discrepancy is a length-free, enrichment-free statement. It is exactly the
signature of a *frozen*-saturation fit applied to a cell in which the probe de-saturates as it
is attenuated: the fitted α under-reports the true optical depth, increasingly so with density.
The competing explanation is cold-spot calibration (the spec 05 density model runs at 11 %/K —
a ~10 K cold-spot offset at 65 °C with none at 27 °C would also do it).

**RydSim's job is to discriminate those two hypotheses**, not to assume one: the solver must
reproduce Su's measured `T_B(T_cell, I_p)` surface with the RydSim density model and report the
implied α. **DIAGNOSTIC + documented literature tension.**

**And the sentence that justifies this whole module:** Su's *published optimum operating point*
— 51 °C, where the Rydberg-EIT peak height is maximal at 13 % — sits at a computed weak-probe
optical depth of **5.52**, i.e. **just above the depth at which `rydsim.experiment` presently
refuses to answer.**

---

## 6.6 Tier D — refusal fences and self-consistency (GATING)

### D1. The fences must still fire (P-16)

Replacing the refusal with a solver must not delete the refusal. Required, per
`00-integrity-audit.md` §3 items 17–19 and 21:

- OD → the *propagation* path (never the analytic thin path) above OD = 0.1;
- strong probe → propagation path above I_p = 0.01 I_sat;
- vapour density outside 298–550 K → warn + flag;
- Gauss–Hermite as sole velocity quadrature for any EIT/AT spectrum → raise (R-2);
- non-convergent z-step or velocity grid → `converged: False` **as data**, not a docstring.

**Measured gap in the shipped tree (this session).** `ThickCellError` named in audit §3 item 18
**does not exist** in `src/rydsim`; there is **no `I > 0.01 I_sat` gate anywhere** on the
physics path (`grep` over `src/rydsim/*.py` for `i_sat|I_sat|saturation` returns only
`lindblad.py:252 weak_probe_chi3`); and the only fence is
`experiment.py:102 max_optical_depth = 5.0` raising a generic `IntegrityError`. Spec 10 must
either implement item 18 as written or amend the audit. **This is a spec-vs-code divergence,
recorded rather than papered over.**

### D2. Coupling depletion must be measured, not assumed (P-17)

Spec 05 §2.f already requires "Check coupling-beam absorption once per run … assert < 1 % and
warn otherwise". Promote to a corpus row: report `1 − P_c(L)/P_c(0)` for every Tier-C fixture
and fail the *run* (not the benchmark) if it exceeds 1 % while the solver was configured with
an undepleted coupling. Rationale: in a Rydberg ladder the intermediate state is nearly empty
under weak probe, so the assumption is usually good — but "usually" is not a number, and at
s₀ ≈ 49 (Su) the intermediate population is *not* small.

### D3. Radial × axial ordering (P-18)

Spec 05 §2.g's Gauss–Laguerre radial average and the z-integration **do not commute** once χ
depends on intensity. Required assertion: `⟨propagate⟩_r ≠ propagate(⟨·⟩_r)` by more than the
solver tolerance whenever s₀ > 0.1, and the implemented order is radial-average-of-propagated-
shells (each shell propagated independently at its own Ω_p(r), Ω_c(r); shells do not exchange
energy — spec 05 §7.6). Sedlacek's 56.25 waist-ratio fixture is the stress case.
**Named pitfall (spec 05 §2.g, verbatim): "radial averaging washes out Autler–Townes contrast —
never simulate at peak intensity only."**

### D4. Kramers–Kronig / dispersive-phase consistency (P-19)

The solver accumulates `Δφ = (k_p/2)∫Re χ dz` alongside the amplitude. Re χ and Im χ from the
same run must satisfy a discrete Hilbert-transform closure over the scanned band to ≤ 1 %
(band-truncation-limited). Catches a solver that propagates |Ω_p| but drops the phase — which
would silently kill self-focusing/lensing diagnostics and any future four-wave-mixing work.

---

## 6.7 Master benchmark table

**Gate column:** **G** = gating (must pass for release); **D** = diagnostic (failure triggers
review, never a tolerance edit — spec 09 §7 rule 5 / audit §3 item 38).

| ID | Gate | Quantity / setup | Expected | Tolerance | Source | Confidence |
|---|---|---|---|---|---|---|
| **P-1** | G | Linear medium (weak probe, undepleted coupling): `T_solver` vs `exp(−OD_lin)` at OD ∈ {0.01, 0.1, 1, 5, 14.216, 20} | equal | rel ≤ 1e−6 with spec-05 step rule (`ΔOD ≤ 0.05`/step) | in-session RK4 study | VERIFIED-COMPUTED |
| **P-1b** | G | Step-control necessity canary: same at fixed 16 steps, OD = 20 | rel error ≈ 2.09 (209 %) | must be **detected and refused**, not returned | in-session | VERIFIED-COMPUTED |
| **P-2** | G | OD → 0 reduction coefficient `(T_prop − T_thin)/OD_lin²` at s = 0.25/1/3/10, OD_lin = 1e−4 | −0.06400 / −0.06250 / −0.023438 / −0.0037566 | rel ≤ 1e−3 | derived + verified in-session (`−s/(2(1+s)³)`) | VERIFIED-COMPUTED |
| **P-3** | G | Saturable absorber, α₀L = 3, s ∈ {1e−6, 0.1, 1, 10}: T vs `ln T + s(T−1) = −α₀L` | 0.0497871157 / 0.0547229392 / 0.1200282390 / 0.7312939746 | rel ≤ 1e−9 | LambertW model; derived + verified in-session | VERIFIED-COMPUTED (law), LITERATURE-RECALL (name) |
| **P-3b** | G | Same setup, discrimination margin vs Beer–Lambert at s = 1 and s = 10 | 58.5 % and 93.2 % | must exceed 10 % (else solver is thin) | in-session | VERIFIED-COMPUTED |
| **P-4** | G | z-integrator order: error vs step halving, OD = 5, 14.216, 20 | order 4.03 measured | order ∈ [3.8, 4.2] | in-session | VERIFIED-COMPUTED |
| **P-5a** | G | Nat-Rb, 75.0 mm, 25.0 °C, weak probe: OD / T_min at ⁸⁵Rb F=3 | 1.2982 / 0.273031 | ±1 % (phys ±6 %) | spec 05 B9a; **independently recomputed this session** | VERIFIED-COMPUTED |
| **P-5b** | G | same cell: ⁸⁷Rb F=2 | 0.4805 / 0.618489 | ±1 % | spec 05 B9b; recomputed | VERIFIED-COMPUTED |
| **P-5c** | G | same cell: ⁸⁵Rb F=2 / ⁸⁷Rb F=1 OD | 0.9558 / 0.3124 | ±1 % | spec 05 §2.f table; recomputed | VERIFIED-COMPUTED |
| **P-5d** | G | Nat-Rb 75.0 mm, 16.5 °C: T_min(⁸⁵Rb F=3) | 0.594036 (OD 0.5208) | ±1 % (phys: Siddons 0.2 % rms) | spec 05 B9d + Siddons J. Phys. B 41, 155004 | VERIFIED-COMPUTED (model), VERIFIED (0.2 % rms claim) |
| **P-5e** | G | dip positions vs ⁸⁷Rb centroid at 25.0 °C | −2.4241 / −1.2883 / +1.6195 / +4.0942 GHz | ±3 MHz **at 25 °C only** | spec 05 B9c; recomputed | VERIFIED-COMPUTED |
| **P-5f** | D | dip-position temperature drift (⁸⁷Rb F=2), 16.5 → 65 °C | −2.4231 → −2.4280 GHz (4.9 MHz span) | ±1 MHz | recomputed this session | VERIFIED-COMPUTED |
| **P-6** | G | Nat-Rb 75.0 mm, **50.0 °C**: OD / T_min at ⁸⁵Rb F=3 (**2.8× above present ceiling**) | 14.2160 / 6.69992e−7 | OD ±1 %; T rel ≤ 1e−3 | spec 05 §2.f; recomputed | VERIFIED-COMPUTED |
| **P-7** | G | Shot-noise-limited NEF optimum, linear medium | `OD_opt = 2` (numeric 1.999996); NEF(2)/NEF(0.1) = 0.1293 | OD_opt ±1 %; ratios ±1 % | derived + verified in-session | VERIFIED-COMPUTED |
| **P-8** | G | EIT-contrast-optimal depth `OD_bg* = ln r/(1−1/r)`, r ∈ {2,5,10,100} | 1.38629 / 2.01180 / 2.55843 / 4.65169 | rel ≤ 1e−3 | derived + verified in-session | VERIFIED-COMPUTED |
| **P-9** | G | Transparency-window narrowing: FWHM ratio per OD doubling | √2 = 1.414214 (measured 1.41421–1.41422) | ±1 % | derived + verified in-session | VERIFIED-COMPUTED |
| **P-10** | **G** | **Jing discriminator (bracket direction).** Cs, 5 cm, N₀ = 4.89e10 cm⁻³, Ω_p = 2π·5.7 MHz, Ω_c = 2π·0.97 MHz | `T_solver ≥ 3 × T_weak-probe` where `T_weak = 0.00966` | strict inequality | Jing v1 (arXiv:1902.11063, fetched) + in-session OD | VERIFIED (inputs), VERIFIED-COMPUTED (bound) |
| **P-11** | D | Jing absolute: transmitted probe power from 120 μW input | ~10 μW (T = 0.0833, OD_eff = 2.485) | factor 2 window [5, 20] μW | Jing v1, fetched verbatim | VERIFIED (published pair); model comparison bundles window loss + EIT |
| **P-11b** | D | Jing cell-length adjudication: 55 cm reading | excluded (predicts ≤1e−10 μW) | — | in-session arithmetic on published numbers | VERIFIED-COMPUTED |
| **P-12** | D | Sedlacek 2012: max probe-transmission increase, 7.5 cm effective, s₀ = 1.956, w₀p/w₀c = 7.5 | ~4.5 % | ORDER (factor 3) | ar5iv:1205.4461, fetched | VERIFIED (published), amplitude model tens-of-% (spec 06 §7.1) |
| **P-13** | D | Mohapatra 2007: probe-transmission change ratio, n = 45 (Ωc = 2π·3.5 MHz) vs n = 80 (Ωc = 2π·1.5 MHz), same 75 mm cell | 5 % and 1 % → **ratio 5.0** | absolutes ORDER; **ratio factor 2** | ar5iv:quant-ph/0612200, fetched | VERIFIED (published) |
| **P-13b** | D | Mohapatra EIT linewidth band | 22–44 MHz | must fall inside | ibid. | VERIFIED |
| **P-14** | D | **Su 2022: EIT peak height vs OD.** Rb, 71.8 mm(?), Ω_p/2π = 30 MHz, Ω_c/2π = 0.38 MHz, B ≈ 0, 27→65 °C | max peak height **13 %** at an **interior** optimum near 51 °C; EIT linewidth ≈ 10 MHz | height ORDER (factor 2); **optimum must be interior, not at the hottest point** | Opt. Express 30, 1499; ar5iv:2111.13408, fetched | VERIFIED (published); cell length vendor-snippet only |
| **P-15** | D | **Su fitted-α tension (length- and enrichment-independent).** α(65 °C)/α(27 °C) | published **11.90**; spec-05 model **30.19** | discrepancy must be **explained** (saturation-along-z vs cold-spot), not fitted away | ibid. + in-session model | VERIFIED (published pair), VERIFIED-COMPUTED (model) |
| **P-16** | G | Refusal fences: OD > 0.1 and I_p > 0.01 I_sat both route to the propagation path; GH-only quadrature raises; non-converged grid returns `converged: False` | all fire | boolean | audit §3 items 17–19, 21; R-2 | VERIFIED (policy) |
| **P-16b** | G | Fence-coverage regression: `I_sat` gate exists at all | currently **ABSENT** in `src/rydsim` | must exist before release | measured this session (`grep`) | VERIFIED-COMPUTED (code state) |
| **P-17** | G | Coupling depletion `1 − P_c(L)/P_c(0)` reported for every Tier-C fixture | < 1 % (else warn + require depleted-coupling mode) | as stated | spec 05 §2.f verbatim | VERIFIED (policy) |
| **P-18** | G | Radial × axial non-commutation at Sedlacek's waist ratio (w₀p²/w₀c² = 56.25), s₀ = 1.956 | `⟨propagate⟩_r ≠ propagate(⟨·⟩_r)` beyond solver tolerance | must differ by > 1 % | spec 05 §2.g | VERIFIED (structure), magnitude to be measured |
| **P-19** | D | Kramers–Kronig closure of (Re χ, Im χ) accumulated by the solver | ≤ 1 % over the scanned band | 1 % | standard KK | LITERATURE-RECALL |
| **P-20** | G | Convergence records shipped as data: z-halving `|ΔT| < 1e−4`, velocity-grid halving < 1e−4, radial nodes 12 vs 32 < 1e−5 | all `True` | as stated | spec 05 §2.f/§2.d/§2.g | VERIFIED (policy) |

**Tolerance-change rule (binding):** no number in this table may be edited to make a run pass.
Expected values and tolerances change only by editing this spec with a rationale
(audit §3 item 38). UNVERIFIED-confidence rows never gate a release (audit §3 item 36).

---

## 6.8 Fixture constants used by this corpus

| Quantity | Value | Unit | Source | Confidence |
|---|---|---|---|---|
| Rb-87 D2 centroid ν₀ | 384 230 484 468 500 | Hz | Steck rev 2.3.4 via `rydsim.atom.RB87.d2` | VERIFIED |
| Rb-85 D2 centroid ν₀ | 384 230 406 373 000 | Hz | ibid. | VERIFIED |
| Rb-85 − Rb-87 D2 isotope shift | −78.0955 | MHz | derived from the two rows above | VERIFIED-COMPUTED |
| Rb D2 Γ_e | 3.8117e7 | rad/s | Steck rev 2.3.4 | VERIFIED |
| Cs-133 D2 Γ_e/2π | 5.234(13) | MHz | Steck rev 2.3.4 (audit §1 check 2) | VERIFIED |
| N(Rb, 298.15 K) | 1.2918e16 | m⁻³ | `rydsim.cell`; spec 05 B3a 1.292e16 | VERIFIED-COMPUTED |
| N(Rb, 323.15 K) | 1.4672e17 | m⁻³ | spec 05 B3c 1.467e17 | VERIFIED-COMPUTED |
| N(Cs, 298.15 K) | 4.8941e16 | m⁻³ | spec 05 B3e 4.894e16; **matches Jing's stated 4.89e10 cm⁻³** | VERIFIED-COMPUTED |
| Rb natural abundances | 0.7217 / 0.2783 (⁸⁵/⁸⁷) | — | Steck | VERIFIED |
| Siddons reference cell | 7.5 cm, natural Rb, 16.5/25.0/25.4/36.6 °C, I/I_sat = 0.002 | — | arXiv:0805.1139 (fetched) | VERIFIED |
| Siddons model accuracy | rms error better than 0.2 % (D2, 16.5 °C) | — | ibid., verbatim | VERIFIED |
| Sedlacek cell / beams | 10 cm glass, 7.5 cm effective; w_p = 750 μm, w_c = 100 μm; Ω_p = 2π·6 MHz, Ω_c = 2π·2 MHz | — | ar5iv:1205.4461 (fetched) | VERIFIED |
| Mohapatra cell / beams | 75 mm; probe 1 μW, w₀ = 0.4 mm; coupling ≤200 mW, w₀ = 0.8 mm | — | ar5iv:quant-ph/0612200 (fetched) | VERIFIED |
| Jing cell / beams | 5 cm Cs, N₀ = 4.89e10 cm⁻³; probe 120±4 μW, 1/e² dia 1.70 mm, Ω_p = 2π·5.7 MHz; coupling 34±1 mW, 2.00 mm, Ω_c = 2π·0.97 MHz | — | ar5iv:1902.11063 (fetched); spec 08 §3.2 | VERIFIED |
| Jing transmitted power | ~10 μW from 120 μW | — | ibid., verbatim | VERIFIED |
| Su OD range | α = 0.42 (27 °C) → 5.0 (65 °C) | — | ar5iv:2111.13408 (fetched) | VERIFIED |
| Su peak height / optimum | 13 % at 51 °C, I_p = 0.044 W/cm², Ω_p/2π = 30 MHz, Ω_c/2π = 0.38 MHz | — | ibid. | VERIFIED |
| Su cell (GC25075-RB) | Ø25.4 mm × 71.8 mm borosilicate, natural Rb, window T > 84 % | — | Thorlabs vendor listing via search snippet; **direct fetch failed** | LITERATURE-RECALL |
| Saturation parameter identity | s₀ = 2\|Ω_p\|²/Γ_e² under lock #4 | — | two-level steady state; must be asserted from the solver, never typed | VERIFIED-COMPUTED |
| Doppler-saturation law | α = α₀/√(1+s₀) at line centre (inhomogeneous) | — | standard inhomogeneous-broadening result; **no primary fetched** | LITERATURE-RECALL — bracketing only, never grading |

---

## 6.9 Corpus execution rules and named pitfalls

1. **Velocity quadrature (R-2, non-negotiable).** Every z level re-solves the velocity integral
   on the spec 05 §2.d / 06 §4.4 uniform-or-composite grid with halving convergence.
   Gauss–Hermite alone is forbidden for any EIT/AT spectrum, and *at every z*, not only at
   z = 0 — a solver that converges the grid once at the entrance and reuses the node set is
   under-resolved at the exit, where the EIT feature has narrowed by √OD (P-9).
2. **Cost discipline.** Naive nesting is (z levels) × (velocity nodes ~1e5) × (Δ_p grid) ×
   (radial nodes 12) full Liouvillian solves. Use the affine decomposition (spec 06 §2.3,
   `L = L₀ + Δ_p L_p + Δ_c L_c + v L_v`) and batch; the *only* z-dependent quantity is Ω_p(z)
   (and Ω_c(z) when depletion is on), which enters L₀ alone.
3. **Underflow.** `exp(−OD)` for OD > 700 underflows; the 50 °C row already reaches T = 6.7e−7.
   Work in OD (log) space for the spectrum and exponentiate once, guarded (spec 05 §5 item 7).
4. **Black-region peak extraction.** At OD ≳ 10 the transmission trace has a flat zero floor
   hundreds of MHz wide; fit `−ln T` wings, never `argmin T`. (Measured: 50 °C, 75 mm →
   T_min = 6.7e−7.)
5. **Step control is physics, not taste.** `|ΔOD| ≤ 0.05` **and** `|ΔΩ_p|/|Ω_p| ≤ 2 %` per step
   (spec 05 §2.f). Fixed-step RK4 at 16 steps is wrong by 209 % at OD = 20 (P-1b).
6. **The width-convention √2 trap** (§6.3 A6): `2√(ln2/(a₂L))` vs `2√(2ln2/(a₂L))`. Stamp the
   convention on every reported EIT width.
7. **Ω is never imported** (R-22): Sedlacek's 2π·6 MHz, Jing's 2π·5.7 MHz and Su's 2π·30 MHz are
   used to *characterise their regime*, never as solver inputs. Ω is recomputed from (d, ℰ)
   with ℰ from the stated power and waist under lock #3.
8. **Species defaults (R-28).** Every Tier-C fixture is species-aware: the Cs (Jing) row must
   build all six `LadderConfig` species fields from `rydsim.objective.species_cell_parameters()`
   and assert `species_defaults_in_use() == ()`. A Cs experiment silently simulated in a Rb-87
   cell is the exact audit CRITICAL-1 failure.
9. **Provenance stamp** (audit §4): every corpus row emits the z-scheme + step-halving result,
   velocity-grid scheme + halving result, radial-node convergence, coupling-depletion fraction,
   the saturation parameter s₀ actually used, and the minimum confidence class on the path.
10. **No RNG** anywhere in corpus paths (spec 09 §4.1).

---

## 6.10 Known limitations, MISSING items, and what would close them

1. **MISSING — Rb D2 self-broadening coefficient at the densities this module unlocks.**
   R-6 supplies the *values* (Weller/Kondo, β/2π = 1.03e−7 Hz·cm³ theory, 1.10(17)e−7 measured
   for D2), but spec 05 §7.2's regime warning stands: above N ≈ 1e12 cm⁻³ (Rb ≳ 90–100 °C)
   resonant dipole–dipole broadening is comparable to γ_EIT. The 50 °C P-6 row is at
   1.6e11 cm⁻³ — safe. Su's 65 °C row is at 5.0e11 cm⁻³ — marginal. **Any corpus row above
   1e12 cm⁻³ must be refused until the D2 coefficient is anchored to a primary.**
2. **MISSING — a published Rydberg-EIT spectrum with fully stated absolute axes.** None of the
   four fetched papers publishes a transmission spectrum with an absolute vertical axis *and* a
   complete parameter set. Jing's 120 μW → 10 μW pair is the closest thing in the record and it
   is a single point. Everything else in Tier C is either a contrast percentage (amplitude,
   tens-of-% model error by spec 06 §7.1) or a figure-digitisation claim, and **no row in this
   corpus is a figure digitisation** — that was a deliberate choice: a short table of stated
   numbers beats a long table of traced pixels.
3. **MISSING — Holloway et al., J. Appl. Phys. 121, 233106 (2017)** numeric systematics budget
   (audit §6 gap 6, refusal #35). Still paywalled; not fetched this session. The
   internal-field/etalon systematic it quantifies is precisely a thick-cell effect and it
   remains an empty placeholder. Must be fetched before release.
4. **MISSING — Su et al.'s cell length.** Absent from the paper; only the vendor part number
   resolves it. Until a primary confirms 71.8 mm, P-14/P-15's *absolute* OD numbers carry that
   tag and only the length-independent 65/27 ratio may be graded tightly.
5. **UNVERIFIED — the inhomogeneous saturation law `α₀/√(1+s₀)`.** Used only to bracket
   (§6.5 C1). A primary (Demtröder or equivalent) would upgrade it; until then it never grades.
6. **NOT FETCHED — RMP 77, 633 (2005)** for the √OD narrowing. Replaced by an in-session
   derivation + numerical verification, which is stronger; the citation is deliberately not
   claimed as VERIFIED.
7. **Out of scope for this corpus (declared, not hidden):** optical bistability and
   Rydberg-interaction-driven nonlinearity in dense thermal vapour (the Gärttner & Evers
   density-dependent dephasing `Γ_R,mot/2π = α n₀`, `α = 1.2e−11 MHz·cm³`, fetched this session
   from ar5iv:1305.1458, is a *cold*-atom result and must not be imported into a thermal-cell
   row); four-wave mixing and generated fields; self-focusing/lensing from transverse Re χ
   gradients (spec 05 §7.6 already excludes it); velocity-changing collisions; buffer gases
   (audit §3 item 16 refuses these outright).
8. **Structural risk this corpus deliberately carries:** Tier A is entirely self-validating and
   Tier B is model-internal (Siddons validates the *method* to 0.2 %, not these specific
   numbers). Only Tier C touches measured Rydberg-EIT reality, and it is thin and amplitude-
   dominated. **RydSim must not claim thick-cell Rydberg-EIT amplitude accuracy better than a
   factor of 2 on the strength of this corpus.** Frequency observables (splittings, positions,
   widths) inherit spec 06's much stronger footing; amplitudes do not.
9. **Spec-vs-code divergence recorded (P-16b):** `ThickCellError` (audit §3 item 18) does not
   exist; the `I > 0.01 I_sat` gate does not exist; the only fence is
   `experiment.py: max_optical_depth = 5.0`. Resolve by implementing, or by amending the audit —
   not by leaving both texts standing.

---

*GreyNOC · RydSim spec 10 §6 · authored 2026-08-11, network available · every Tier-A and
Tier-B number recomputed in-session (`scratchpad/verify_b9.py`, `scratchpad/verify_thick.py`)
· house rule: reproducible or it didn't happen.*

---

## Provenance of this draft section
### Sources FETCHED this session
- arXiv:0805.1139 abstract page (fetched) — Siddons, Adams, Ge & Hughes, J. Phys. B 41, 155004 (2008). Took: 'an rms error better than 0.2% for the D2 line at 16.5 degrees C'; 'intensity under one thousandth of the saturation intensity'. Establishes that the 0.2% figure is the METHOD validation, and that no transmission minima are tabulated.
- ar5iv full text of 0805.1139 (fetched) — took: '7.5 cm cell'; temperatures 16.5 / 25.0 / 25.4 / 36.6 °C; weak-probe '32 nW/mm²' (I/I_sat = 0.002) and '1.6 μW/mm²' (I/I_sat = 0.1); transmissions 'ranging from 5 to 95%'; and the explicit confirmation that NO specific peak/minimum transmission values are stated numerically in the text. This is what forced spec 05's B9 rows to be labelled model-internal rather than Siddons transcriptions.
- ar5iv full text of quant-ph/0612200 (fetched) — Mohapatra, Jackson & Adams, PRL 98, 113003 (2007). Took: 'room temperature rubidium vapor cell of length 75 mm'; probe 'power 1 μW', 'beam size 0.4 mm (1/e² radius)'; coupling 'λc=479.2−483.9 nm', 'power up to 200 mW', 'spot size of 0.8 mm'; 'The line-width of the EIT resonance is between 22 and 44 MHz'; 'produces a change in the probe transmission of 5%' (n=45, Ωc=2π×3.5 MHz) and 'about 1%' (n=80, Ωc=2π×1.5 MHz). Became benchmarks P-13/P-13b.
- ar5iv full text of 1205.4461 (fetched) — Sedlacek et al., Nat. Phys. 8, 819 (2012). Took: 'room temperature 10 cm Rb vapor cell'; 'the effective interaction length is 7.5 cm'; 'the probe beam size is 750 μm'; 'Ωp=2π×6 MHz'; 'The coupling laser beam size is 100 μm'; 'Ωc=2π×2 MHz'; '53D5/2→54P3/2 … ∼14.233 GHz'; 'μRF=1.37×10⁻²⁶ C m'; EIT window '∼4−5 MHz'; 'A maximum increase of ∼4.5%'; sensitivity ∼30 μV/cm/√Hz. Became P-12; also gave the w_p/w_c = 7.5 geometric-dilution fact behind P-18.
- ar5iv full text of 1902.11063 (fetched) — Jing et al., Nat. Phys. 16, 911 (2020). Took: total density N₀=4.89×10¹⁰ cm⁻³; 'the optical power incident to the vapor cell is 120±4 μW, yielding effectively Ωp=5.7±0.6 MHz'; 'After absorption by Cs atoms, the power of the probe light incident on the detector is about 10 μW'; coupling '34±1 mW … Ωc=0.97±0.12 MHz'; 47D5/2→48P3/2 at 6.94 GHz; E_L=3.0 mV/cm; 55 nV/cm/√Hz. The 120→10 μW pair is the single best published thick-cell transmission datum in the corpus (P-10/P-11). NOTE: the ar5iv rendering reads '55-cm-long'; adjudicated numerically to 5 cm (consistent with spec 08 §3.2).
- arXiv:2111.13408 abstract page (fetched) — Su, Liou, Lin & Chen, 'Optimizing the Rydberg EIT spectrum in a thermal vapor', Opt. Express 30(2), 1499-1510 (2022). Took title/authors/journal ref and the headline 13% peak height, 'more than twice' the prior room-temperature maximum.
- ar5iv full text of 2111.13408 (fetched) — took the quantitative OD scan: 'We performed the Rydberg EIT spectral measurements in an Rb vapor cell at the temperature ranging from 27∘C to 65∘C'; 'The determined α varied from 0.42 (27∘C) to 5.0 (65∘C)'; their baseline model 'TB = exp(−α Γe²/(Γe² + 2Ωp²))'; probe '1.5 μW to 17 mW', optimum '0.044 W/cm²', waist '0.81 mm (full width at e⁻² maximum)', Ωp/2π = 30 MHz; coupling '27 mW', '5.3 W/cm²', Ωc/2π = 0.38±0.03 MHz; states |33D3/2>,|33D5/2>, splitting '336.4 MHz'; 'Γe/2π = 60 MHz (includes velocity group selection)'; 'The EIT linewidth was around 10 MHz'; max peak height '13%' at 51 °C; and the explicit fact that CELL LENGTH IS ABSENT from text and figures. Became P-14/P-15 — the only quantitative published EIT-peak-height-vs-optical-depth data found.
- ar5iv full text of 1305.1458 (fetched) — Gärttner & Evers, 'Non-linear absorption and density dependent dephasing in Rydberg EIT-media'. Took: the self-consistent probe-attenuation propagation model ('Ωp(L) = Ωp(0)e^(−χkL/2)'), 'Including the probe beam attenuation self-consistently in the RE model is indispensable for the simulation of Rydberg EIT in a dense gas', and the density-dependent dephasing ΓR,mot/2π = α n₀ with α = 1.2×10⁻¹¹ MHz cm³. Used ONLY as an out-of-scope marker (it is a T = 5 μK cold-atom result, explicitly excluded from thermal-cell rows).
- In-repo primary reading (this session): docs/spec/00-conventions.md (locks 1-20, R-1..R-28), docs/spec/00-integrity-audit.md (§3 refusal list, items 17-19/21/36/38), docs/spec/05-vapor-cell-physics.md §2.f/§2.g/§6/§7 (propagation scheme, B9 table, step controls, limitations), docs/spec/06-optical-bloch-eit.md §2.4/§2.6/§7.2 (chi chain, thin-medium limitation), docs/spec/08 §3.2 (Jing fixture, cell length 5 cm VERIFIED from v1).
- In-repo code measurement (this session): src/rydsim/experiment.py:102,313-323 (max_optical_depth = 5.0, generic IntegrityError); src/rydsim/eit.py:200-215 (chi_si, transmission); grep over src/rydsim/*.py for 'ThickCellError' and for 'i_sat|I_sat|saturation' — BOTH ABSENT except lindblad.py:252 weak_probe_chi3. This is the P-16b spec-vs-code divergence.
- In-session numerical verification, scratchpad/verify_b9.py — independent re-implementation of spec 05 Eq. 2.f (Voigt sum over both isotopes, S_FF' from rydsim.angular, Steck 2.3.4 hyperfine from rydsim.atom, densities from rydsim.cell) with NO call to any RydSim propagation routine. Reproduced every spec-05 B9 row to <0.15%: 25 °C OD/T_min = 0.4805/0.618489 (87F2), 1.2982/0.273031 (85F3), 0.9558/0.384523 (85F2), 0.3124/0.731713 (87F1); 16.5 °C 0.5208/0.594036; 50 °C 14.2160/6.69992e-7; dip positions -2.4241/-1.2883/+1.6195/+4.0942 GHz. Also measured the previously undocumented dip-position temperature drift (-2.4231 GHz at 16.5 °C to -2.4280 GHz at 65 °C, 4.9 MHz span vs B9c's ±3 MHz tolerance).
- In-session numerical verification, scratchpad/verify_thick.py — (a) Cs D2 Doppler OD at 25 °C: alpha = 92.787/m at the F=4 dip -> OD(5 cm) = 4.639, OD(55 cm) = 51.03, which adjudicates Jing's cell length; (b) saturable-absorber implicit law verified against RK4 to 1.5e-12 at four saturation levels, with Beer-Lambert errors 0.00/9.02/58.5/93.2%; (c) linear-medium RK4-vs-Beer-Lambert exactness, including the 209% error of a fixed-16-step integrator at OD = 20 and the 1.09e-6 accuracy of the spec-05 step rule (n >= 20*OD); order 4.03 measured; (d) shot-noise NEF optimum argmin OD = 1.999996; (e) EIT-contrast-optimal depth ln r/(1-1/r) verified to 5 digits at r = 2/5/10/100. Plus: the OD->0 reduction coefficient -s/(2(1+s)^3) derived by series inversion and confirmed at s = 0.25/1/3/10 to 4 significant figures; the sqrt(2) window-narrowing ratio confirmed over six octaves (1.41421-1.41422); and the Jing three-way bracket (weak-probe 1.16 μW, homogeneous-saturation 30.3 μW, Doppler-saturation 9.59 μW vs published ~10 μW).
- WebSearch snippet (NOT a direct fetch — thorlabs.com fetch returned no content): Thorlabs GC25075-RB is a 'Rubidium Borosilicate Reference Cell, Ø25.4 mm x 71.8 mm', natural abundance, 'transmission through the cell exceeds 84% for light in the 350 nm to 2.2 µm range'. Supplies the cell length missing from Su et al.; tagged LITERATURE-RECALL, not VERIFIED.
- WebSearch (citation confirmation only): Zentile, Keaveney, Weller, Whiting, Adams & Hughes, 'ElecSus: A program to calculate the electric susceptibility of an atomic ensemble', Comput. Phys. Commun. 189, 162-174 (2015), arXiv:1409.1873 — recommended independent cross-validation for the Tier-B weak-probe rows. Full text not fetched.

### UNVERIFIED / recall-only
- The inhomogeneous (Doppler) saturated-absorption law alpha = alpha0/sqrt(1+s0) at line centre — LITERATURE-RECALL. It is standard for inhomogeneously broadened lines but no primary (Demtroder/Siegman or equivalent) was fetched this session. It is used ONLY to bracket the Jing row and never to grade a benchmark. It also has a known defect for this purpose: it is a *frozen* law (uses the input intensity), so a correct z-resolved solver should predict MORE absorption than it does.
- The Fleischhauer/Imamoglu/Marangos RMP 77, 633 (2005) statement of the 1/sqrt(OD) transparency-window narrowing — NOT FETCHED (paywalled; search returned no usable text). The scaling law and its exact coefficient are instead derived and numerically verified in-session, which is why P-9 is tagged VERIFIED-COMPUTED and the RMP is deliberately not cited as VERIFIED.
- The name 'LambertW model' for the steady-state saturable-absorber transmission law — search-snippet level only (an Optics Communications paper 'Modeling absorption in saturable absorbers' was surfaced but not fetched). The LAW itself is derived and verified in-session to 1.5e-12; only the literature attribution is unverified.
- Su et al.'s cell length. ABSENT from the paper text and figures (confirmed by the ar5iv extraction). The 71.8 mm value comes from resolving the Thorlabs part number GC25075-RB through a vendor search snippet; the direct Thorlabs fetch failed. All ABSOLUTE OD numbers in P-14/P-15 inherit this tag; only the length-independent 65/27 alpha ratio is graded tightly.
- Su et al.'s cell temperature calibration (quoted cell temperature vs cold-spot temperature). This is the leading competing explanation for the P-15 alpha-ratio discrepancy (spec 05's density model runs at 11 %/K, so a ~10 K cold-spot offset at 65 °C with none at 27 °C would reproduce the gap). RydSim must discriminate between saturation-along-z and cold-spot calibration, not assume the former.
- Jing et al.'s window/optical losses. The published 120 uW -> ~10 uW pair is end-to-end and bundles uncoated-borosilicate window reflections (~8-15%), the EIT/AT transparency at the operating point, and hyperfine optical pumping. The 1.7% agreement between OD_eff/OD_weak = 0.536 and 1/sqrt(1+s0) = 0.545 is therefore a coincidence of partially cancelling corrections, NOT a validated model agreement. Stated as such in the spec; the row is diagnostic at a factor-2 window.
- Jing's cell length as rendered by ar5iv ('55-cm-long'). Adjudicated to 5 cm by numerical impossibility of the 55 cm reading (predicts <=1e-10 uW against ~10 uW measured) plus spec 08 §3.2's VERIFIED-from-v1 record — but the arXiv v1 PDF itself was not re-parsed this session, so the adjudication rests on arithmetic, not on re-reading the source.
- Sedlacek's ~4.5% maximum transmission increase — the ar5iv extraction attributes it to a figure caption, and it is ambiguous whether it refers to the EIT peak on the Doppler background or to the bright-resonance feature that is the paper's subject. P-12 is diagnostic at ORDER tolerance for this reason.
- Holloway et al., J. Appl. Phys. 121, 233106 (2017) numeric systematics budget — still paywalled and NOT fetched (audit §6 gap 6, refusal #35 remain open). The internal-field/etalon systematic it quantifies is a thick-cell effect and its placeholder in spec 09 (E9.2+) must stay empty.
- P-18's magnitude (radial x axial non-commutation at Sedlacek's 56.25 waist ratio) is specified structurally but its numerical value was NOT measured this session — the benchmark states the required inequality, not an expected number. It needs a measurement pass once the solver exists.
- The Rb D2 self-broadening impact at the densities this module unlocks. R-6 supplies the coefficient values, but the regime boundary (N ~ 1e12 cm^-3) versus Su's 65 °C row (5.0e11 cm^-3) is marginal and was not evaluated numerically this session.
- Whether Su et al.'s cell is natural-abundance or enriched Rb was not confirmed from the paper; the vendor listing implies natural. The 65/27 ratio benchmark was chosen precisely because it is insensitive to this.

### Open questions
- Should the OD > 0.1 / I_p > 0.01*I_sat refusal (audit 3 item 18, 'ThickCellError') be implemented as written, or amended? Measured this session: ThickCellError does not exist in src/rydsim, there is NO saturation gate anywhere on the physics path, and the only fence is experiment.py:102 max_optical_depth = 5.0 raising a generic IntegrityError. Spec and code currently disagree and one of them must move.
- What replaces max_optical_depth = 5.0 once the solver exists? Su et al.'s published OPTIMUM operating point (51 C, 13 % peak height) sits at a computed weak-probe OD of 5.52 - just above the present ceiling. A ceiling that excludes the literature optimum is not a safety fence, it is a capability gap. Proposal: replace the OD ceiling with (a) a density ceiling tied to the self-broadening regime (N < 1e12 cm^-3 until the Rb D2 coefficient is primary-sourced) and (b) a convergence-based refusal (z-halving, velocity-halving, radial-node), i.e. refuse when the ANSWER is not converged rather than when the OD is large.
- Spec 05 B9c's +-3 MHz dip-position tolerance is measured here to be temperature-specific: the 87Rb F=2 apparent dip drifts 4.9 MHz between 16.5 C and 65 C. Amend B9c to per-temperature expected values, or widen to +-6 MHz with the drift documented? Grading a high-OD spectrum against the 25 C positions at +-3 MHz will produce false failures.
- Is Su et al.'s fitted-alpha discrepancy (11.90 published vs 30.19 model over 27->65 C, length- and enrichment-independent) explained by saturation-along-z, or by cold-spot temperature calibration? The spec-05 density model runs at 11 %/K, so a ~10 K cold-spot offset at 65 C with none at 27 C would also reproduce it. RydSim must discriminate; this is a concrete post-solver experiment, not a matter of opinion.
- Should P-11's tolerance be tightened once the window-loss and EIT contributions are modelled separately? The published 120 uW -> ~10 uW pair is end-to-end; the ~1.7 % agreement with a frozen Doppler-saturation estimate is a coincidence of competing corrections (frozen saturation over-predicts T because it ignores de-saturation along z; EIT transparency pushes T back up). A solver that models all three should be able to close the row much tighter than the factor-2 window given here.
- Does anyone publish EIT contrast vs optical depth beyond Su 2022? Su is the only quantitative source this session's search found (alpha = 0.42 -> 5.0, peak height 13 % at an interior optimum). One paper is a thin evidential base for a headline capability claim. A second independent OD-scan dataset would materially strengthen the corpus.
- Do we need a benchmark for the transverse-shell approximation itself? Spec 05 7.6 assumes radial shells do not diffract into each other, valid for L << 2 z_R and modest transverse OD gradients. At Sedlacek's 56.25 waist ratio the transverse OD gradient is extreme. A gating row would need either an experimental anchor or a beam-propagation reference calculation - neither exists in this corpus.
- Holloway et al., J. Appl. Phys. 121, 233106 (2017) is still paywalled and unfetched (audit 6 gap 6). Its internal-field/etalon systematics budget is a THICK-CELL effect and is now on the critical path for this module, not just for spec 09's E9.2 placeholder. Who fetches it, and does release block on it?
- Should P-18's expected magnitude be measured now with a stand-in solver, or left as a structural inequality until the real solver lands? Leaving it unquantified is honest but weakens the gate.

### Proposed benchmarks

| id | quantity | expected | tol | source | conf |
|---|---|---|---|---|---|
| P-1 | Linear-medium exactness: T_solver vs exp(-OD_lin) at OD in {0.01, 0.1, 1, 5, 14.216, 20}, weak probe, undepleted coupling | equal (Beer-Lambert is the EXACT solution when chi is z-independent, at any OD) | rel <= 1e-6 with the spec-05 step rule \|dOD\| <= 0.05 per step | derived + verified in-session (scratchpad/verify_thick.py); spec 05 2.f | VERIFIED-COMPUTED |
| P-1b | Step-control necessity canary: fixed 16-step RK4 at OD = 20 | rel error 2.09 (209 %) - must be DETECTED and refused, not returned | boolean (refusal must fire) | in-session measurement | VERIFIED-COMPUTED |
| P-2 | OD->0 reduction coefficient (T_prop - T_thin)/OD_lin^2 at s = 0.25 / 1 / 3 / 10, OD_lin = 1e-4 | -0.06400 / -0.06250 / -0.023438 / -0.0037566, i.e. exactly -s/(2(1+s)^3) | rel <= 1e-3 | derived by series inversion of the saturable-absorber implicit law; verified in-session at 4 values of s | VERIFIED-COMPUTED |
| P-3 | Saturable absorber, alpha0*L = 3: T vs ln T + s(T-1) = -alpha0 L at s = 1e-6 / 0.1 / 1 / 10 | 0.0497871157 / 0.0547229392 / 0.1200282390 / 0.7312939746 | rel <= 1e-9 | LambertW model; derived + RK4-verified in-session to 1.5e-12 | VERIFIED-COMPUTED (law); LITERATURE-RECALL (attribution) |
| P-3b | Discrimination margin of P-3 vs Beer-Lambert at s = 1 and s = 10 | 58.5 % and 93.2 % error if the thin path is used | measured margin must exceed 10 % (else the solver is thin) | in-session | VERIFIED-COMPUTED |
| P-4 | z-integrator convergence order from step halving at OD = 5, 14.216, 20 | 4.03 (RK4) | order in [3.8, 4.2] | in-session | VERIFIED-COMPUTED |
| P-5a | Natural Rb, 75.0 mm, 25.0 C, weak probe: OD / T_min at the 85Rb F=3 dip | 1.2982 / 0.273031 | +-1 % (phys +-6 %: vapour model + T) | spec 05 B9a; INDEPENDENTLY RECOMPUTED this session (scratchpad/verify_b9.py); method validated to 0.2 % rms by Siddons J. Phys. B 41, 155004 (2008) | VERIFIED-COMPUTED |
| P-5b | Same cell: OD / T_min at the 87Rb F=2 dip | 0.4805 / 0.618489 | +-1 % | spec 05 B9b; recomputed this session | VERIFIED-COMPUTED |
| P-5c | Same cell: OD at 85Rb F=2 and 87Rb F=1 dips | 0.9558 and 0.3124 (T_min 0.384523 and 0.731713) | +-1 % | spec 05 2.f table; recomputed this session | VERIFIED-COMPUTED |
| P-5d | Natural Rb, 75.0 mm, 16.5 C: T_min (OD) at 85Rb F=3 | 0.594036 (OD 0.5208) | +-1 % (phys: Siddons 2008 agree 0.2 % rms) | spec 05 B9d; recomputed; Siddons rms figure VERIFIED from arXiv:0805.1139 | VERIFIED-COMPUTED (model) / VERIFIED (0.2 % claim) |
| P-5e | Dip positions vs the 87Rb D2 centroid at 25.0 C | -2.4241 / -1.2883 / +1.6195 / +4.0942 GHz | +-3 MHz AT 25 C ONLY | spec 05 B9c; recomputed this session | VERIFIED-COMPUTED |
| P-5f | Dip-position temperature drift of the 87Rb F=2 blend, 16.5 -> 65 C | -2.4231 -> -2.4280 GHz (4.9 MHz span) | +-1 MHz | measured this session; motivates widening/per-temperature restatement of spec 05 B9c | VERIFIED-COMPUTED |
| P-6 | Natural Rb, 75.0 mm, 50.0 C: OD / T_min at 85Rb F=3 - 2.8x above the engine's present refusal ceiling | 14.2160 / 6.69992e-7 | OD +-1 %; T rel <= 1e-3 | spec 05 2.f table; recomputed this session | VERIFIED-COMPUTED |
| P-7 | Shot-noise-limited NEF optimum in a linear medium: argmin of exp(OD/2)/OD | OD_opt = 2 (numeric 1.999996); NEF(2)/NEF(0.1) = 0.1293, NEF(2)/NEF(1) = 0.8244, NEF(2)/NEF(5) = 0.5578 | OD_opt +-1 %; ratios +-1 % | derived + verified in-session | VERIFIED-COMPUTED |
| P-8 | EIT-contrast-optimal background depth OD_bg* = ln r/(1 - 1/r) for contrast ratio r = 2 / 5 / 10 / 100 | 1.38629 / 2.01180 / 2.55843 / 4.65169 (crosses 5.0 at r = 143.3) | rel <= 1e-3 | derived + verified in-session (2e6-point scan) | VERIFIED-COMPUTED |
| P-9 | Transparency-window FWHM ratio per doubling of OD | sqrt(2) = 1.414214 (measured 1.41421-1.41422 over six octaves) | +-1 % | derived + verified in-session; RMP 77,633 (2005) deliberately NOT relied upon (not fetched) | VERIFIED-COMPUTED |
| P-10 | JING DISCRIMINATOR (bracket direction). Cs, 5 cm, N0 = 4.89e10 cm^-3, Om_p = 2pi*5.7 MHz, Om_c = 2pi*0.97 MHz: solver transmission vs the weak-probe value | T_solver >= 3 x T_weak-probe, where T_weak = 0.00966 (OD 4.639) - the thin path predicts 1.16 uW against a published ~10 uW, 8.6x too low | strict inequality (gating) | Jing et al., Nat. Phys. 16, 911 (2020) / ar5iv:1902.11063 fetched; OD computed in-session from their stated density | VERIFIED (inputs) / VERIFIED-COMPUTED (bound) |
| P-11 | Jing absolute: transmitted probe power from 120 uW incident | ~10 uW (T = 0.0833, OD_eff = 2.485) | factor 2 window, [5, 20] uW | ar5iv:1902.11063, verbatim published pair | VERIFIED (published); model comparison bundles window loss + EIT, so diagnostic only |
| P-11b | Jing cell-length adjudication: is the ar5iv '55-cm' reading admissible? | EXCLUDED - 55 cm predicts <=1e-10 uW at the detector against a published ~10 uW; 5 cm stands (spec 08 3.2) | n/a (11 orders of magnitude) | in-session arithmetic on published numbers | VERIFIED-COMPUTED |
| P-12 | Sedlacek 2012: maximum probe-transmission increase; 7.5 cm effective length, OD = 0.4805, s0 = 1.956, w_p/w_c = 7.5 (area dilution 1.78 %) | ~4.5 % | ORDER (factor 3) | ar5iv:1205.4461 fetched; OD computed in-session | VERIFIED (published number, figure-caption level); amplitude model tens-of-% per spec 06 7.1 |
| P-13 | Mohapatra 2007: probe-transmission change, n = 45 (Om_c = 2pi*3.5 MHz) vs n = 80 (Om_c = 2pi*1.5 MHz), same 75 mm cell | 5 % and 1 % -> ratio 5.0 | absolutes ORDER (factor 3); RATIO factor 2 | ar5iv:quant-ph/0612200 fetched | VERIFIED (published) |
| P-13b | Mohapatra 2007: EIT linewidth band across their power/transition range | 22-44 MHz | simulated width must fall inside the band | ibid., verbatim | VERIFIED |
| P-14 | Su 2022 EIT PEAK HEIGHT vs OPTICAL DEPTH. Rb, L = 71.8 mm (vendor), Om_p/2pi = 30 MHz, Om_c/2pi = 0.38 MHz, B ~ 0, 27-65 C | maximum peak height 13 % at an INTERIOR optimum near 51 C; EIT linewidth ~10 MHz | height ORDER (factor 2); the optimum must be interior, NOT at the hottest point | Su, Liou, Lin & Chen, Opt. Express 30, 1499 (2022) / ar5iv:2111.13408 fetched | VERIFIED (published); cell length LITERATURE-RECALL (vendor snippet) |
| P-15 | Su fitted-alpha tension, LENGTH- and ENRICHMENT-INDEPENDENT: alpha(65 C)/alpha(27 C) | published 11.90; spec-05 weak-probe model 30.19 (absolutes 0.42 vs 0.566 at 27 C; 5.0 vs 17.10 at 65 C) | the 2.54x discrepancy must be EXPLAINED (saturation-along-z vs cold-spot calibration), never fitted away | ar5iv:2111.13408 + in-session spec-05 model computation | VERIFIED (published pair) / VERIFIED-COMPUTED (model) |
| P-16 | Refusal fences: OD > 0.1 and I_p > 0.01*I_sat both route to the propagation path; Gauss-Hermite-only quadrature raises; non-converged grid returns converged=False as data | all fire | boolean | 00-integrity-audit.md 3 items 17-19, 21; ruling R-2 | VERIFIED (policy) |
| P-16b | Fence-coverage regression: does an I_sat / weak-probe gate exist in src/rydsim at all? | currently ABSENT (measured); ThickCellError also absent; only experiment.py:102 max_optical_depth = 5.0 exists | must exist before release, or the audit must be amended | measured this session by grep over src/rydsim/*.py | VERIFIED-COMPUTED (code state) |
| P-17 | Coupling depletion 1 - P_c(L)/P_c(0), reported for every Tier-C fixture | < 1 % (else warn and require the depleted-coupling mode) | 1 % absolute | spec 05 2.f verbatim ('assert < 1% and warn otherwise') | VERIFIED (policy) |
| P-18 | Radial x axial non-commutation at Sedlacek's waist ratio (w0p^2/w0c^2 = 56.25), s0 = 1.956 | <propagate>_r differs from propagate(<.>_r) beyond solver tolerance; implemented order must be radial-average-of-propagated-shells | must differ by > 1 % (magnitude NOT yet measured - needs a pass once the solver exists) | spec 05 2.g (unequal-waist radial average; 'radial averaging washes out Autler-Townes contrast') | VERIFIED (structure) / MISSING (magnitude) |
| P-19 | Kramers-Kronig closure between the Re chi and Im chi accumulated by the solver over the scanned band | consistent | <= 1 % (band-truncation-limited) | standard KK relation | LITERATURE-RECALL |
| P-20 | Convergence records shipped as data: z-step halving, velocity-grid halving, radial-node 12-vs-32 | \|dT\| < 1e-4, < 1e-4, < 1e-5 respectively; all flags True | as stated | spec 05 2.f / 2.d / 2.g; audit 4 item 6 | VERIFIED (policy) |
