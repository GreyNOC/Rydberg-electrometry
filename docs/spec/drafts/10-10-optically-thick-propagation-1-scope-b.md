# 10 — Optically Thick Propagation · §1 Scope Boundary and Non-Claims

**RydSim physics specification, document 10, section 1. Status: NORMATIVE for the scope, refusal
and claim boundaries of `rydsim.propagate`.** Subordinate to `docs/spec/00-conventions.md` (20
locks, rulings R-1…R-28) in every particular; where this section proposes an amendment to 00 it
says so explicitly (§7.4). Network **was** available during authoring; every fetch is dated
2026-08-11 and listed in §8.

House rule in force: *reproducible or it didn't happen.* Every claim below carries a source and a
confidence tag, and every criterion is stated as a number that a test can falsify.

---

## 1. What this section is for

RydSim has no laboratory. Everything it produces is simulation, so the boundary of the model is
not a footnote — it is the deliverable that makes the numbers inside the boundary worth reading.
This section fixes four things before any propagation code is written:

1. exactly which present limitations thick-cell propagation removes, and which currently-refused
   published configurations become computable (§2);
2. exactly which physics remains absent, with a *criterion* for when each matters and a *size* for
   the error it induces (§3);
3. what a no-lab program may legitimately claim from the resulting solver — assessed critically,
   not asserted (§5);
4. the conditions under which the solver must raise `IntegrityError` rather than return a number
   (§7).

**Symbols used in this section** (SI internal, per lock #1; angular rates per lock #2):

| Symbol | Meaning | SI unit |
|---|---|---|
| `N` | ground-state number density of the sensed species | m⁻³ (cm⁻³ at I/O) |
| `N_F` | density in the addressed ground hyperfine level, `N_F = p_F·N` | m⁻³ |
| `N_r` | Rydberg-state number density | m⁻³ |
| `L` | optical path length through the vapor | m |
| `k_p`, `k_c` | probe/coupling wavenumbers, `2π/λ` | rad/m |
| `χ` | probe electric susceptibility (dimensionless) | — |
| `OD` | optical depth `= k_p·Im χ·L` (lock, spec 00 §2 row "α (absorption)") | — |
| `Ω_p`, `Ω_c`, `Ω_RF` | Rabi frequencies, `Ω = d·ℰ/ħ`, peak amplitude (lock #4) | rad/s |
| `Γ_e` | intermediate-state population decay (FWHM in angular units, lock #7) | rad/s |
| `s_p` | probe saturation parameter `2Ω_p²/Γ_e²` | — |
| `β` | resonant self-broadening coefficient (`Γ_self = β·N`, FWHM) | Hz·m³ (Hz·cm³ I/O) |
| `w₀` | beam radius, 1/e² **intensity** radius (lock, spec 00 §2) | m |
| `z_R` | Rayleigh range `π w₀²/λ` | m |
| `D` | largest cell dimension seen by the RF field | m |
| `λ_rf` | free-space RF wavelength | m |
| `E_int`, `E_inc` | RF field inside the cell / incident on it | V/m |
| `p_F` | ground hyperfine degeneracy fraction `(2F+1)/Σ(2F+1)` (spec 05 §2.b) | — |
| `κ_E` | transduction slope `dP/dE` at the operating point (spec 08) | W/(V/m) |
| `NEF` | noise-equivalent field, amplitude convention (lock #12) | (V/m)/√Hz |

---

## 2. What thick-cell propagation fixes — and the one thing it does not

### 2.1 The four limitations it removes

| # | Present limitation | Where it lives now | Removed by the z-solver because |
|---|---|---|---|
| F-1 | Probe saturation is *not* z-resolved: `χ` is evaluated once at the input `Ω_p` | `eit.chi_ladder` + `eit.transmission` | `dΩ_p/dz = i(k_p/2)χ(Ω_p(z))Ω_p` re-solves `χ` at the local field (spec 05 §2.f) |
| F-2 | Coupling depletion is assumed zero and never checked | `experiment.superhet_transfer` (no check) | the solver carries `Ω_c(z)` and asserts `<1 %` depletion per run (spec 05 §2.f) |
| F-3 | The transmitted-power collapse at high OD is handled by refusing, not by computing | `LadderConfig.max_optical_depth = 5.0` | saturation raises the transmitted power at fixed `(N, L)`, so the refusal ceiling can be replaced by a *physics* ceiling |
| F-4 | Radial averaging (spec 05 §2.g) and longitudinal propagation are not composed | nothing composes them | shell-resolved `T(s, L)` with `s = 2r²/w₀²`, then Gauss–Laguerre in `s` |

### 2.2 The load-bearing negative result: propagation is EXACTLY a no-op in the weak-probe limit

**Claim (proved, then measured).** In the strict weak-probe limit `χ` is independent of `Ω_p`
(linear response), and `Ω_c`, `N`, `f(v)` are z-independent. The propagation ODE
`dΩ_p/dz = i(k_p/2)χ Ω_p` therefore has **constant coefficients** and integrates in closed form to

```
|Ω_p(L)/Ω_p(0)|² = exp(−k_p·Im χ·L) = exp(−OD)
```

— which is bit-for-bit the existing `eit.transmission`. **Measured this session** (RK4, 20 001
steps, Rb-87, 60 °C, L = 5 cm, OD = 10.0558): RK4 gives `4.2937676186e-05`, closed form gives
`4.2937676186e-05`, relative difference **2.8×10⁻¹⁴**.

Two consequences, both binding:

* **The `OD → 0` reduction test is degenerate.** It is satisfied by construction at *every* OD, not
  only small OD, so it cannot validate the new module. It must still be run (S-1), but the
  discriminating test is a different limit — see §6, benchmarks S-1/S-2.
* **Everything the module can change comes from relaxing weak probe, or from transverse coupling.**
  A thick-cell solver that keeps the weak-probe `χ` is a re-derivation of `exp(−OD)`. This is the
  single most important scope statement in this document, and it reorders the work: the *strong-probe*
  z-coupled solve is the deliverable; "thick cell" alone is not.

Confidence: **VERIFIED (analytic + numerical, this session).** Falsifier: any parameter set in the
declared weak-probe domain where the z-integrated result differs from `exp(−OD)` by more than
1×10⁻¹⁰ relative indicates a solver bug, not new physics.

### 2.3 Which currently-refused published configurations become computable

Optical depths computed this session with the shipped chain
(`experiment.spectrum` → `eit.chi_si` → `k_p·L·Im χ`, weak probe, line centre, EIT on), densities
from `rydsim.cell` (Alcock/Steck):

| Corpus entry | Species / cell | `N` (cm⁻³) | shipped OD | vs gate 5.0 | corrected OD† |
|---|---|---|---|---|---|
| E1 Sedlacek 2012 | Rb-87, 294 K, L = 7.5 cm eff. | 2.30×10⁹ | 0.750 | passes | 0.31 |
| E2 Holloway 2014 | Rb-85, 296 K, L = 7.5 cm | 7.41×10⁹ | 1.260 | passes | 0.49 |
| **E3 Jing 2020** | **Cs-133, 298.15 K, L = 5 cm** | **4.894×10¹⁰** | **15.909** | **REFUSED** | **5.97 — still refused** |

† corrected for the two normative factors the shipped estimator omits (§7.4 D-1): the effective
hot-vapor probe dipole `d_eff,far = ⟨J‖er‖J'⟩/√3` rather than the cycling `⟨J‖er‖J'⟩/√2`
(factor 2/3 in `d²`, spec 00 §6 gap 7), and the ground hyperfine fraction `p_F` (5/8 Rb-87 F=2,
7/12 Rb-85 F=3, 9/16 Cs F=4, spec 05 §2.b).

Reproduction of the refusal, verbatim from this session:

```
Jing-like superhet_transfer: REFUSED -> optical depth 15.91 exceeds the operating ceiling
5.00 for Cs at 298.1 K over 5.0 cm (N = 4.89e+16 m^-3): transmitted probe power is
numerically dead across the ...
```

Independent corroboration that the density is right, not the model: the Alcock/Steck model gives
`N_Cs(298.15 K) = 4.8941×10¹⁰ cm⁻³` against Jing's **printed** `N₀ = 4.89×10¹⁰ cm⁻³` — agreement to
three significant figures, i.e. the OD is a property of the published cell, not of RydSim's vapor
model. Confidence: **VERIFIED** (printed value from spec 09 §3.5, itself fetched from
arXiv:1902.11063 on 2026-08-10; density recomputed this session).

**So: exactly one corpus entry — E3, the headline superheterodyne benchmark — is currently
refused, and it is refused for the wrong reason.** At the corrected OD ≈ 6 the transmitted power
is 297 nW out of 120 µW in, three orders of magnitude above the spec-08 default detector NEP of
5 pW/√Hz. The claim in the `max_optical_depth` docstring that "transmitted probe power is
numerically dead" is true at OD 15.9 (14.9 pW out) and false at the corrected OD ≈ 6. The gate is
mis-calibrated by the same 2.67× the estimator is (§7.4 D-1).

### 2.4 The regime the corpus actually occupies is strong-probe, not merely thick

Computed this session from the papers' own printed Rabi frequencies (spec 09 §3.5):

| Config | `Ω_p/2π` | `s_p = 2Ω_p²/Γ_e²` | audit refusal #21 ceiling `0.01·min(Γ_e, Ω_c)/2π` | exceeded by |
|---|---|---|---|---|
| E1 Sedlacek 2012 (Rb) | 6.00 MHz | 1.96 | 20.0 kHz | **300×** |
| E3 Jing 2020 (Cs) | 5.70 MHz | 2.37 | 9.7 kHz | **588×** |
| E6 multi-dress 2026 (Rb) | 17.16 MHz | 16.0 | 60.7 kHz | **283×** |

Jing's printed beam parameters (120 µW, 1/e² diameter 1.70 mm) give `I₀ = 10.57 mW/cm²`, i.e.
`I₀/I_sat = 9.6` against the Cs cycling `I_sat = 1.1049 mW/cm²` or `3.9` against the π-effective
`2.7119 mW/cm²`. The three saturation figures (2.37 from `Ω_p`, 3.9 and 9.6 from the beam) do not
agree — a printed-parameter inconsistency of exactly the class ruling **R-22** governs (spec 08
already flags Jing's `Ω_L = 7.9 MHz` vs `E_LO = 3.0 mV/cm`). Per R-22 the resolution is never to
ingest the printed number: `Ω_p` is recomputed from `(d_eff,far, ℰ)` and the residual is recorded
as a fixture tension. All three figures agree on the only thing that matters here: **the published
corpus runs 200–600× outside RydSim's own weak-probe validity gate.**

**Therefore the scope statement is:** thick-cell propagation makes E3 *computable in principle*;
it makes E1/E2/E3/E6 *physically defensible* only when it is the strong-probe, z-coupled solve.
The weak-probe z-solver closes none of the gap (§2.2).

---

## 3. What it does NOT fix — the residual-physics register

Each entry: the physics, the criterion for when it matters (a number a test can evaluate), and the
size of the error induced. Ordered by how badly the corpus violates the criterion.

### 3.1 Vapor-cell etalon and the internal-vs-incident RF field — **the dominant unmodelled systematic**

Spec 06 §2.8 item 6 flags this as "major in practice" and does not model it. It remains unmodelled
after this document; the thick-cell solver propagates the *optical* fields and says nothing about
the RF field's amplitude at the atoms.

**Mechanism (VERIFIED, fetched 2026-08-11):** Fan, Kumar, Sheng, Shaffer, Holloway & Gordon,
*Phys. Rev. Applied* **4**, 044015 (2015):

> "The FP effect occurs because when a rf wave is incident onto a hollow glass vapor cell, standing
> waves can develop inside the vapor cell due to reflections from the glass walls, forming a rf FP
> cavity."

**Criterion (VERIFIED, quoted from the same paper):**

> "we show experimentally that the accuracy is greater than current methods in the frequency range
> 10–30 GHz and is not limited by the vapor-cell geometry provided `D/λ_rf < 0.1`."

and, on the wall-loss channel specifically:

> "For a rf E-field at 12.6 GHz, the absorption by 1 mm of Pyrex is 0.066%."

so absorption is negligible and **interference is the whole effect**. Their cells (8 mm and 9 mm
cubic Pyrex) span `D/λ_rf = 0.05–0.72`, i.e. the criterion was established by sweeping across it.

**How badly the corpus violates it** (computed this session from spec 09 §3.5 cell dimensions and RF
frequencies; `D_long` = cell length, `D_tran` = diameter/width):

| Corpus entry | `f_RF` | `λ_rf` | `D_long/λ_rf` | `D_tran/λ_rf` | violation of `<0.1` |
|---|---|---|---|---|---|
| E1 Sedlacek 2012 | 14.233 GHz | 2.11 cm | 4.75 | 1.19 | **47×** |
| E2 Holloway 2014 | 17.04 GHz | 1.76 cm | 4.26 | 1.42 | **43×** |
| E2 Holloway 2014 | 104.77 GHz | 0.29 cm | 26.2 | 8.74 | **262×** |
| E3 Jing 2020 | 6.94 GHz | 4.32 cm | 1.16 | 0.46 | **12×** |
| E6 multi-dress 2026 | 16.03 GHz | 1.87 cm | 2.67 | 1.07 | **27×** |
| E7 Zeeman 2026 | 1 GHz | 30.0 cm | 0.17 | 0.03 | **2×** |
| E7 Zeeman 2026 | 40 GHz | 0.75 cm | 6.67 | 1.33 | **67×** |

**Every corpus configuration violates the published criterion, by 2× to 262×.** This is not a
corner case; it is the norm for the experiments RydSim exists to reproduce.

**Size of the induced error: MISSING.** Fan et al. establish the *criterion* and show the variation
falls as `D/λ_rf` falls, but the per-configuration `|E_int/E_inc|` for RydSim's cell geometries is
not derivable from what was fetched. What would resolve it: a full-wave EM solve (FDTD/FEM) of each
declared cell geometry with the measured permittivity, or the per-cell measurement of Richardson
et al., arXiv:2604.11785 (2026, VERIFIED fetched 2026-08-11), which reports effective permittivities
`3.8+0j` (unfilled quartz), `4.5+0j` (unfilled borosilicate), `9+0j` (sapphire portion) over
10–300 MHz and observes "field reduction and spatial degradation" inside filled cells — a
measurement campaign, not a model. **Until then, every absolute-field claim from RydSim carries an
un-bounded multiplicative systematic on `E_RF`, and RydSim must say so in words, not swallow it.**
This is the largest known gap in the program.

Ruling implication: the AT-splitting inversion `E = ħΩ_RF/℘` returns the field **at the atoms**,
never the incident field. The two are the same quantity only when `D/λ_rf < 0.1`. `EFieldResult`
must carry a `field_reference: Literal["at_atoms","incident"]` stamp, always `"at_atoms"`, plus the
computed `D/λ_rf`.

### 3.2 RF field inhomogeneity across the cell

Distinct from §3.1: even with a perfectly matched cell, `E_RF(r, z)` varies over the interaction
volume, so the AT splitting is a *distribution*, not a line.

**Criterion.** The doublet is broadened rather than shifted when the fractional field spread across
the interrogated volume exceeds the fractional splitting resolution:

```
σ(E_RF)/⟨E_RF⟩  >  Γ_EIT^obs / Ω_RF          (broadening dominates)
```

Below that, the effect is a second-order shift of the fitted splitting. **Size:** first order in the
spread; a 10 % field spread over the beam produces a ~10 % broadening of each AT peak and a
sub-percent shift of the fitted separation for a symmetric distribution. Spec 06 §2.8 item 7 already
provides the machinery (`average E over profile if map provided`) — the gap is that **no map exists**
for any corpus entry, because no corpus entry publishes one. Confidence: **LITERATURE-RECALL**
(standard inhomogeneous-broadening argument); the criterion is derived here and is self-checking
(S-8).

The thick-cell solver makes this *worse* in one specific way that must be stated: once the probe is
z-resolved, the weighting of each z-slice in the observed signal is `∝ exp(−OD(z))`, so the front of
the cell dominates. Any `E_RF(z)` gradient is therefore sampled with a **non-uniform, OD-dependent
weight** — the effective `⟨E_RF⟩` moves as the cell is heated. Falsifier: with a linear `E_RF(z)`
ramp of ±10 %, the inverted field must move between the OD → 0 and OD = 5 solutions by the
predicted weight shift; if it does not, the z-weighting is wrong.

### 3.3 Transverse beam structure — radial averaging and diffraction

Two separate failures, with separate criteria.

**(a) Radial intensity profile (plane-wave/on-axis inadequacy).** A TEM₀₀ beam has
`Ω(r) = Ω₀ exp(−r²/w₀²)` (field ∝ √intensity, spec 05 §2.g). For whole-beam detection with equal
probe and coupling waists, the probe-power-weighted mean of `Ω_c²` is, **exactly**,

```
⟨Ω_c²⟩ / Ω_c0² = ∫₀^∞ e^{−s}·e^{−s} ds / ∫₀^∞ e^{−s} ds = 1/2
⇒ Ω_c,eff = Ω_c0/√2 = 0.7071·Ω_c0
```

**On-axis evaluation therefore overstates the effective coupling Rabi frequency by 29.3 %**, and the
EIT width (`∝ Ω_c²/2γ_ge`, spec 06 §2.5) by a factor of **2**. Verified numerically this session
(24-node Gauss–Laguerre reproduces 0.500000). For unequal waists `w₀p/w₀c`:
`⟨Ω_c²⟩/Ω_c0² = 1/(1 + (w₀p/w₀c)²)` — measured 0.800 at ratio 0.5 and 0.200 at ratio 2.0.
Confidence: **VERIFIED (analytic identity, numerically confirmed).**

*Criterion:* the plane-wave/on-axis treatment is inadequate whenever the observable depends on
`Ω_c` or `Ω_p` at second order or higher — i.e. **always for EIT widths, contrasts, transduction
slopes and NEF**, and *never* for AT peak positions, which depend on `Ω_RF` alone and `Ω_RF` is
radially uniform on the scale of an optical beam (`w₀ ≪ λ_rf`, by 10²–10³). This asymmetry is the
structural reason frequency observables survive and amplitude observables do not.

*Size:* 29.3 % in `Ω_c,eff`, 2× in EIT width, and — since the AT resolvability threshold is
`Δf_AT ≳ Γ_EIT^obs/2π` (Holloway 2017, VERIFIED-abstract-level in spec 06 §2.7) — a **√2 error in
the minimum resolvable field** if peak-intensity-only simulation is used. Spec 05 §4.8 already
forbids peak-intensity-only comparison with experiment; this section elevates it to a refusal
(§7.1 R-P4) because the thick-cell solver multiplies the error: each radial shell has a *different*
OD, so radial and longitudinal averaging do not commute.

**(b) Diffraction — the collimated-beam assumption fails for half the corpus.** Spec 05 §2.g requires
`L ≪ 2z_R`. Computed this session:

| Corpus beam | `w₀` | `z_R` | `2z_R/L` | verdict |
|---|---|---|---|---|
| E1 Sedlacek probe | 375 µm | 566 mm | 15.1 | collimated OK |
| **E1 Sedlacek coupling** | **50 µm** | **16.4 mm** | **0.44** | **FAILS** |
| **E2 Holloway probe** | **40 µm** | **6.4 mm** | **0.17** | **FAILS** |
| E3 Jing probe | 850 µm | 2663 mm | 107 | collimated OK |
| E3 Jing coupling | 1000 µm | 6167 mm | 247 | collimated OK |
| **E6 probe** | **52.5 µm** | **11.1 mm** | **0.44** | **FAILS** |
| **E6 coupling** | **29 µm** | **5.5 mm** | **0.22** | **FAILS** |

*Criterion:* `2z_R/L ≥ 10` → collimated treatment valid; `1 ≤ 2z_R/L < 10` → marginal, the waist
varies by up to ~40 % over the cell and the radial quadrature must be z-dependent;
`2z_R/L < 1` → **the 1-D z-propagation model is not applicable**; the beam expands by more than √2
in radius within the cell and `Ω(r, z)` cannot be factorized.

*Size:* at `2z_R/L = 0.44` (E1 coupling, E6 probe) the on-axis intensity at the cell exit is
`1/(1+(L/2z_R)²) ≈ 1/(1+5.2) = 0.16` of the waist value — a **6× drop in `I_c`, i.e. 2.5× in `Ω_c`**
across the cell. That is not a correction; it is a different experiment at each z. Confidence:
**VERIFIED (Gaussian-optics identity; corpus beam parameters from spec 09 §3.5, fetched 2026-08-10).**

Spec 09 §8 item 8 already flags E6's 29 µm waist as stressing the transit model. This section adds
the harder statement: **for E1's coupling beam, E2's probe and both of E6's beams, the geometry
violates the propagation model's own precondition, and thick-cell propagation as a 1-D z-integration
inherits that violation rather than fixing it.** A paraxial 2-D `(r, z)` split-step solver is the
fix; it is out of scope here and is named as future work.

### 3.4 Alkali adsorbate screening at cell walls

Alkali adsorbed on the inner wall makes glass conductive; free/surface charge redistributes and
cancels quasi-static interior fields, and the effect is photo-activated. Spec 05 §2.h owns the
phenomenological model `T(f)` (ruling **R-7**) with per-cell parameters.

*Criterion for when it matters:* it applies to the **applied low-frequency signal field only**, never
to the optical fields and never to a resonant GHz field (spec 05 §4.10). Numerically, the interior
field is suppressed below the corner `f_c = 1/(2π τ_s,eff)`; a borosilicate cell with
`τ_s ≈ 1×10⁻⁴ s` gives `f_c = 1.59 kHz` and `|S(60 Hz)|/S_geo = 0.038`, i.e. **26× suppression at
mains frequency**. Under illumination the corner moves: Jau & Carter measured 3-dB low-cutoffs of
≈64 Hz at `P_480 = 10 mW` and ≈770 Hz at 120 mW on their best sapphire cell (spec 05 §2.h,
VERIFIED-in-repo, **not re-fetched this session**).

*What thick-cell propagation changes:* **nothing.** Screening enters before the atoms; the
propagation solver consumes `E_int` and never sees `E_ext`. The one interaction is indirect and
must be stated: the photo-activation rate scales with the coupling power **at the cell**, and a
thick cell attenuates the *probe*, not the coupling — so `κ_ph·P_c` is z-independent to the same
accuracy as F-2 (coupling depletion < 1 %). If the F-2 assertion fails, `τ_s,eff` becomes
z-dependent and the screening model must be applied per slice. Falsifier: S-9.

*Second, harder channel — DC adsorbate patch fields.* Sedlacek, Kim, Rittenhouse, Weck, Sadeghpour
& Shaffer, *Phys. Rev. Lett.* **116**, 133201 (2016), "Electric Field Cancellation on Quartz by Rb
Adsorbate-Induced Negative Electron Affinity" — **citation VERIFIED (title/journal/authors confirmed
from three independent indexes this session: UNLV institutional record, OSTI 1244776, PubMed
27081976); numeric field magnitudes MISSING** (the CfA-hosted PDF would not extract). These patch
fields Stark-shift the Rydberg level directly and are not a transfer function. What would resolve
it: the field magnitudes and atom–surface distances from PRL 116, 133201, plus a
beam-centre-to-wall distance for each declared cell. Until then, RydSim must not claim a DC or
sub-kHz absolute field for a glass cell — which is already the standing position (spec 05 §7.5,
audit refusal #20).

### 3.5 Rydberg–Rydberg interactions and ionization at high density

*What is absent:* dipole–dipole/vdW shifts and broadening between Rydberg atoms, Penning
ionization, free-charge (ion/electron) fields, and the resulting optical bistability.

**Primary anchor (VERIFIED, fetched 2026-08-11).** Weller, Urvoy, Rico, Löw & Kübler,
"Charge-induced optical bistability in thermal Rydberg vapor," *Phys. Rev. A* **94**, 063820 (2016)
[arXiv:1609.02330]. Abstract, verbatim: *"we conclude that the large polarizability of Rydberg
states in combination with electric fields of spontaneously ionized Rydberg atoms is the relevant
interaction mechanism … Both these experiments allow us to rule out dipole-dipole interactions, and
support our hypothesis of a charge-induced bistability."* Extracted quantities: Rb densities
`N₈₅ = 1.8×10¹² cm⁻³`, `N₈₇ = 0.7×10¹² cm⁻³` (total ≈ 2.5×10¹² cm⁻³) at `T_res = 80–120 °C`,
`T_cell = 135 °C`; Cs `N = 1.2×10¹³ cm⁻³`; Rydberg fraction ≈ 2 %; ion density
`N_ion ≤ 1×10¹⁰ cm⁻³`, *"this ion density matches 27 % of the Rydberg density"*; ionization
cross-section *"up to σ = 1×10⁻³ µm² = 0.03·σ_geo"*; states Rb 32S/41S, Cs 23D₃/₂/28S₁/₂.

**This is the load-bearing number for §4:** the ground density at which charge-induced bistability
has been *observed* in a thermal Rydberg vapor is ≈2.5×10¹² cm⁻³ (Rb). RydSim's own OD table (§4)
puts `OD(5 cm, Rb) = 100` at `N = 3.51×10¹² cm⁻³`. **The density that buys OD ≈ 100 is the density
at which the medium is known to go bistable.**

*Criterion (weak-probe, computed this session).* Rydberg density
`N_r ≈ N·f_vel·ρ_rr` with `f_vel ≈ Γ_e/Δν_D ≈ 6.07/511.3 = 0.0119` (Rb 300 K) and `ρ_rr` from the
steady state; mean nearest-neighbour spacing `r_nn = (3/4πN_r)^{1/3}`; vdW dephasing
`Δν_vdW = (C₆/h)/r_nn⁶` (ruling **R-26**, no second /h). At `ρ_rr = 10⁻³`:

| Operating point | `N` (cm⁻³) | `N_r` (cm⁻³) | `r_nn` | `Δν_vdW` (C₆/h = 19 GHz·µm⁶) | (140 GHz·µm⁶) |
|---|---|---|---|---|---|
| OD ≈ 10 (5 cm, 60 °C) | 3.36×10¹¹ | 4.0×10⁶ | 39.1 µm | 0.01 kHz | 0.04 kHz |
| OD ≈ 100 (5 cm, 92 °C) | 3.51×10¹² | 4.2×10⁷ | 17.9 µm | 0.58 kHz | 4.3 kHz |
| OD ≈ 1000 (5 cm, 131 °C) | 3.69×10¹³ | 4.4×10⁸ | 8.2 µm | 63.9 kHz | 471 kHz |

**Non-obvious result worth stating loudly: at weak probe, Rydberg–Rydberg vdW is *not* the binding
constraint** — it is sub-kHz up to OD ≈ 100, two to three orders below the ground-perturber terms of
§4. The binding constraint is ground-state collisional physics. But at the corpus's *actual*
Rydberg fraction (Weller's 2 %, i.e. 20× larger than assumed here), `N_r` rises 20×, `r_nn` falls
2.7×, and `Δν_vdW` rises by 2.7⁶ ≈ **390×** — into the MHz. The trap is that `ρ_rr` is exactly what
the strong-probe solver will start producing, and RydSim has **no ionization model at all**: no
Penning rate, no free-charge field, no bistability branch. Confidence: **derived this session from
spec-04 coefficients (C₆ fit LITERATURE-RECALL, R11) — order-of-magnitude only.**

*What would resolve it:* a Penning/ionization rate model with a primary source, plus the
ion-field-induced Stark broadening. Weller's `σ ≤ 1×10⁻³ µm²` is a fitted bound, not a rate law.
**MISSING.**

### 3.6 Velocity-changing collisions

*What is absent:* RydSim's Doppler average (spec 05 §2.d) assumes each atom keeps its velocity
class for the whole interrogation. Velocity-changing collisions (VCC) redistribute atoms between
classes, partially collapsing the velocity-selective EIT structure.

**Anchor (VERIFIED, fetched 2026-08-11).** Lei, Eckel, Norrgard, Prajapati, Artusio-Glimpse, Simons
& Holloway, "Revisiting collisional broadening of ⁸⁵Rb Rydberg levels: conclusions for vapor cell
manufacture," arXiv:2408.16669 → *Phys. Rev. Applied* **23**, 034028 (2025). States 25D, 27S, 30D,
32S, 35D, 37S. Broadening coefficients e.g. 25D + He `8.2(6)×10⁻¹⁰ Hz·cm³`; 25D + Ar broadening
`21.6(1.2)×10⁻¹⁰ Hz·cm³`, shift `−103.6(1.1)×10⁻¹⁰ Hz·cm³`. VCC rate coefficients `⟨σv⟩` are
tabulated in units of `10⁻⁹ cm³/s`; on the ratio of EIT broadening to VCC rate: *"Our observed
ratios … do not vary dramatically over the four gases measured, the total variation is only about a
factor of 5."*

**Criterion (VERIFIED, quoted):** *"roughly 0.02 mbar of contaminant gas would be required to add
roughly 1 MHz of additional broadening"*, and *"to get an additional broadening of 1 MHz would
require p ∼ 0.01 mbar at 30 °C."* Their conclusion: *"It is unlikely that contaminant gases are the
sole cause of vapor cells with odd EIT lineshapes because the required pressures for measurable
effects are large, on the order of 0.01 mbar, and vapor cells are generally evacuated to well below
10⁻⁵ mbar prior to sealing."*

**Size, and why this one is *good* news:** at the ≤10⁻⁵ mbar residual pressure of a well-made
evacuated cell, foreign-gas VCC contributes ≲1 kHz — negligible against every other term in §4.
**RydSim may legitimately neglect foreign-gas VCC for evacuated alkali-only cells, and this is now
a sourced neglect rather than an assumption.** It may **not** neglect it for buffer-gas cells, which
remain refused outright (audit refusal #16, unchanged).

*The residual gap is Rb–Rb VCC*, which this paper does not address ("The study focuses exclusively
on collisional effects from external perturber gases"). Alkali–alkali VCC scales with `N` and
therefore rises exactly where the density trap bites. **MISSING**; what would resolve it is an
alkali–alkali VCC cross-section with a primary source. Until then, the density ceiling of §7.1
(R-P6) is the operative fence and must be justified on the self-broadening/Fermi terms alone, which
it is.

---

## 4. THE DENSITY TRAP — quantified

The thick-cell module exists so RydSim can raise OD. OD is raised by raising `N` (or `L`). But every
collisional term in the model scales with `N`, so "optically thick" and "dilute, non-interacting
atoms" (spec 06 §7.4) are the same knob pulled in opposite directions. This section fixes where the
boundary sits.

**Inputs.** Self-broadening `Γ_self = β·N` (FWHM; `γ_ge += β·N/2`), `β/2π = 1.03×10⁻⁷ Hz·cm³` for
Rb D2 (theory, Lewis 1980 via Weller et al. 2011 Table I) and `1.16×10⁻⁷` for Cs D2 — **VERIFIED in
spec 04 §3.5 per ruling R-6; not re-fetched this session.** Rydberg–ground Fermi shift
`Δν_Fermi = (ħa_s/m_e)·N = −9.9×10⁻⁸ Hz·cm³ · N[cm⁻³]` with broadening `≈ 0.5·|shift|`
(**UNVERIFIED in spec 04, 2× uncertainty flag** — audit R11). Densities from `rydsim.cell`
(Alcock/Steck, ±5 %, 298–550 K). ODs computed this session with the shipped chain, natural Rb,
Rb-87 sensed, `Ω_c/2π = 5 MHz`, weak probe, line centre.

| Target OD | `T` (5 cm) | `T` (7.5 cm) | `N_total` (cm⁻³) | `Γ_self` FWHM | as % of `Γ_e` | Fermi shift | Fermi broad. as % of default `deph_r` (2π·100 kHz) |
|---|---|---|---|---|---|---|---|
| 0.1 | 12.1 °C | 8.6 °C | 3.12×10⁹ | 0.3 kHz | 0.005 % | −0.3 kHz | 0.2 % |
| 1 | 33.9 °C | 29.8 °C | 3.23×10¹⁰ | 3.3 kHz | 0.05 % | −3.2 kHz | 1.6 % |
| **10** | **59.9 °C** | **54.9 °C** | **3.36×10¹¹** | **34.6 kHz** | **0.57 %** | **−33.2 kHz** | **16.6 %** |
| **100** | **91.8 °C** | **85.7 °C** | **3.51×10¹²** | **361 kHz** | **5.96 %** | **−347 kHz** | **174 %** |
| 1000 | 130.6 °C | 123.2 °C | 3.69×10¹³ | 3.80 MHz | 62.6 % | −3.65 MHz | 1824 % |

Cs, 5 cm (Jing geometry, `Ω_c/2π = 0.97 MHz`): OD 1 at 2.0 °C (`N = 4.02×10⁹`), OD 10 at 23.4 °C
(`4.17×10¹⁰`), OD 100 at 49.1 °C (`4.33×10¹¹`, `Γ_self = 50 kHz`).

Reference crossings for Rb (computed this session):

* `Γ_self` reaches the transit rate `γ_t/2π = 39.8 kHz` (`w₀ = 1 mm`, 300 K) at `N = 3.86×10¹¹ cm⁻³`;
* `Γ_self = 0.1·Γ_e` at `N = 5.89×10¹² cm⁻³`;
* `Γ_self = Γ_e` at `N = 5.89×10¹³ cm⁻³`;
* Fermi shift reaches 100 kHz at `N = 1.01×10¹² cm⁻³`.

### 4.1 Where the boundary sits — the normative reading

1. **`OD ≤ 10` (Rb `N ≤ 3.4×10¹¹ cm⁻³`, ≤60 °C at 5 cm): the model is sound.** Every collisional
   term is ≤17 % of an already-modelled rate, and all of them *are* modelled (self-broadening into
   `γ_ge`, Fermi shift into the line centre, Fermi broadening into `γ_gr`). They must be switched
   on — they are not, today, in `LadderConfig` — but the physics is present.
2. **`10 < OD ≤ 100` (`N ≤ 3.5×10¹² cm⁻³`, ≤92 °C at 5 cm): computable with mandatory flags.**
   Self-broadening reaches 6 % of `Γ_e`; the Rydberg–ground broadening reaches 174 % of the default
   Rydberg dephasing, i.e. **the collisional term becomes the dominant Rydberg linewidth and it is
   the term carrying a 2× UNVERIFIED uncertainty.** Any output here must inherit that 2× on the
   EIT/AT linewidth, hence on the resolvability threshold, hence on `E_min`. Spec 05 §7.2's own
   warning threshold (`n ≈ 10¹² cm⁻³`) sits inside this band and stands.
3. **`OD > 100` (`N > 3.5×10¹² cm⁻³`): REFUSE.** Three independent reasons, any one sufficient:
   * it is the density at which charge-induced optical bistability is *observed* (Weller 2016,
     Rb total ≈2.5×10¹² cm⁻³) and RydSim has no ionization, free-charge or bistability physics;
   * the dominant Rydberg linewidth is the UNVERIFIED Fermi-broadening surrogate, so the answer is
     an extrapolation of a guess;
   * spec 05 §7.2 declares the Durham-grade **D2** self-broadening coefficient MISSING above
     10¹² cm⁻³ (Weller's 0.1 % validation to 3×10¹⁴ cm⁻³ is **D1**), so `β` itself is out of its
     validated band.

   **Normative ceiling: `N_max = 3.5×10¹² cm⁻³` (total elemental ground density), or `OD = 100`,
   whichever binds first.** Confidence: the *value* is derived from VERIFIED coefficients; the
   *choice* of 100 rather than 30 or 300 is a judgement, declared as such, and is the number a
   future spec edit would change with rationale (lock #20).

4. **The trap in one line:** for a 5 cm Rb cell the entire usable window is
   **34 °C ≤ T ≤ 92 °C**, spanning `OD` 1 → 100 and `N` 3.2×10¹⁰ → 3.5×10¹² cm⁻³. Below it the cell
   is thin and the thick-cell solver is a no-op (§2.2); above it the atoms are not the atoms the
   model describes. Every design campaign must report where in that window it sat.

---

## 5. What a no-lab program may legitimately claim from this solver

### 5.1 The thesis under test

*Absolute sensitivities are hostage to uncalibrated parameters (`τ_s`, RIN, NEP, laser lineshape,
`E_int/E_inc`, cell geometry); relative structure — trade exponents, optima locations, orderings,
bounds — survives.* This section assesses that thesis specifically for thick-cell propagation.

### 5.2 The answer, in four grades

**Grade A — invariant under propagation, *exactly*, and provably.**
The transmission is `T(Δ_p) = exp(−k_p L·Im χ(Δ_p))`. Since `exp` is strictly monotone,
`dT/dΔ_p = 0 ⟺ d(Im χ)/dΔ_p = 0`: **the stationary points of the transmitted spectrum coincide with
those of `Im χ` at every optical depth.** Therefore, in the weak-probe limit, AT peak *positions*,
the peak *separation*, the `λ_c/λ_p` compression factor, and hence the inverted field `E = ħΩ_RF/℘`
are propagation-invariant to machine precision. Frequency observables are safe.
Caveat that must ship with the claim: the invariance holds for the *true* stationary points. The
spec-06 §2.7 extraction algorithm fits Lorentzians to a lineshape that OD distorts (`exp(−OD)`
flattens the peaks and steepens the flanks), so **the fitted splitting is not invariant even though
the true one is**. That is an algorithm systematic, and it is the one place OD leaks into a
frequency observable. Benchmark S-3 measures it.

**Grade B — scale changes, structure does not.**
Transmitted power, EIT contrast, transduction slope `κ_E` and `NEF` all pick up the factor
`exp(−OD)` — but the thin-cell engine **already computes that factor**. Consequently, in the
weak-probe limit, propagation changes *neither* scale *nor* structure (§2.2, measured to 2.8×10⁻¹⁴).
Everything below is about what happens once `Ω_p` is finite.

**Grade C — structure that is already present and that propagation does not create.**
The existence and location of an OD optimum are properties of Beer–Lambert plus the noise model,
not of the solver. With `κ_E ∝ P_in·e^{−OD}·OD·(relative slope)`:

| Dominant noise term | `S_P` scaling | `NEF ∝` | optimum |
|---|---|---|---|
| shot noise | `∝ P` | `e^{OD/2}/OD` | **`OD* = 2` exactly** |
| detector NEP | constant | `e^{OD}/OD` | **`OD* = 1` exactly** |
| RIN | `∝ P²` | `1/OD` | monotone — no interior optimum |

(Computed this session by bounded minimisation; the values are exact stationary points of the
closed forms.) **This is a strong, useful, no-lab-legitimate result:** the *location* of the
optical-depth optimum is set by which noise term dominates, is an integer 1 or 2 in the two
detector-limited cases, and is completely independent of every uncalibrated absolute parameter.
RydSim can assert it. It also means the shipped ceiling of 5.0 sits above both optima — the current
engine was never blind to the interesting region, only to the region beyond it.

**Grade D — structure that propagation genuinely changes, and that therefore invalidates thin-cell
trade findings taken at finite `Ω_p`.**
Once `Ω_p` saturates the medium, three things break:
* `OD` stops being `∝ N·L`. The front of the cell is bleached and the back is not, so
  `OD_eff(N, L, Ω_p)` is sublinear in `L` and in `N`. **Any trade exponent measured against `L` or
  `N` at finite `Ω_p` in the thin-cell model is wrong by a z-dependent factor.**
* The probe-power optimum couples to density. **Measured, not asserted** — Su, Liou, Lin & Chen,
  *Opt. Express* **30**, 1499 (2022) [arXiv:2111.13408], VERIFIED fetched 2026-08-11:
  *"For any given temperature, the peak height has a universal behavior that it reaches the maximum
  value with the optimum `I_p`. The optimum `I_p` becomes stronger with a higher vapor temperature"*
  and *"We achieved the maximum EIT peak height of 13 % with the vapor temperature of 51 °C and
  `I_p` of 0.044 W/cm²."* Their fitted optical-density parameter is `α = 165`, corroborating that
  real Rydberg-EIT cells operate at OD ≫ 5. **A two-dimensional optimum whose location moves along
  one axis as the other is varied is a structural feature, and RydSim's weak-probe path cannot
  reproduce it at all** — at weak probe `χ` has no `I_p` dependence, so the peak-height-vs-`I_p`
  maximum does not exist in the model.
* The radial and longitudinal averages stop commuting (§3.3a): each shell has its own OD and its own
  saturation, so `⟨T(s, L)⟩_s ≠ T(⟨s⟩, L)`.

### 5.3 Verdict, stated plainly

**Propagation changes the SCALE, not the STRUCTURE — but only inside the weak-probe limit, where it
changes nothing at all. Outside it, propagation changes the structure of the `(N, L, Ω_p)` trade
and leaves the `Ω_c` trade and every frequency observable alone.**

Consequences the program must accept:

* The retracted D3 trade-law finding (α ≈ 0.5, retracted 2026-08-10 because every configuration ran
  at OD 5–100) is **not** rehabilitated by a thick-cell solver. Those runs were invalid for two
  independent reasons — regime and, per §7.4 D-1, an OD estimate 2.4× too large — and the
  re-measured valid-regime result (`NEF` U-shaped in `Ω_c`; above the optimum `α < 0.15`) is the one
  that stands. What the solver *can* do is extend that measurement's domain upward in OD.
* Because the `Ω_c` trade is not touched by propagation (the coupling is undepleted, F-2, and
  `Ω_c` does not enter the propagation ODE), **the `α < 0.15` weak-trade finding is expected to
  survive.** "Expected" is not "shown": S-6 is the benchmark that must show it, by re-running the
  `Ω_c` sweep through the propagation solver at `Ω_p → 0` (must be bit-identical) and at
  `Ω_p = Γ_e/10, Γ_e/2, Γ_e` (drift reported). Decision rule: `|Δα| < 0.02` ⇒ scale only;
  `|Δα| > 0.05` ⇒ structural, and the thin-cell trade findings are retracted a second time and
  re-derived.
* **RydSim may claim:** OD optima locations; orderings of designs at equal OD; frequency
  observables and their systematics; trade exponents in `Ω_c`; bounds (SQL, shot-noise floors);
  the *shape* of `NEF(N)`, `NEF(L)`, `NEF(Ω_c)` curves inside the §4 window.
* **RydSim may not claim:** any absolute `NEF` in nV/cm/√Hz without the uncalibrated-parameter list
  attached; any absolute *field* without the `field_reference: "at_atoms"` stamp and the `D/λ_rf`
  figure (§3.1); any comparison with a published sensitivity closer than the one-sided ORDER grading
  spec 09 §7 already imposes; any result at `2z_R/L < 1` (§3.3b); anything above the §4 ceiling.

---

## 6. Benchmarks for this section (→ `tests/test_spec10_scope.py`)

All are scope/boundary tests: they check that the module knows where it is, not that the physics is
right (that is the propagation-core section's job).

| ID | Quantity | Setup | Expected | Tolerance | Source / type | Confidence |
|---|---|---|---|---|---|---|
| S-1 | weak-probe degeneracy | Rb-87, 60 °C, L = 5 cm, `Ω_p/2π = 1 Hz`, RK4 20 001 steps | z-solver `= exp(−OD)` at OD = 10.0558 | rel ≤ 1e-10 (**measured 2.8e-14**) | §2.2 analytic identity | VERIFIED (this session) |
| S-2 | the *discriminating* reduction: `OD → 0` at **finite** `Ω_p` | `Ω_p = Γ_e/2`, `N` scaled so OD ∈ {1e-3, 1e-2, 1e-1} | z-solver → thin-cell strong-probe `χ` | rel ≤ 1e-6, and error `∝ OD` (slope 1.00 ± 0.05 on log–log) | §2.2; this is the test S-1 cannot be | VERIFIED (by construction) |
| S-3 | AT position invariance vs fitted-splitting drift | Doppler Rb, `Ω_RF/2π = 20 MHz`, OD ∈ {0.1, 1, 5, 20} | true stationary points invariant to grid precision; **fitted** splitting drift reported, not asserted | positions ≤ 1e-9 rel; drift = measured output | §5.2 Grade A theorem | VERIFIED (theorem), drift measured |
| S-4 | OD-optimum locations | analytic `NEF(OD)` with the three noise scalings | `OD* = 2` (shot), `1` (NEP), none (RIN) | rel ≤ 1e-6 | §5.2 Grade C closed forms | VERIFIED (this session) |
| S-5 | radial `Ω_c` identity | equal waists, 24-node Gauss–Laguerre | `⟨Ω_c²⟩/Ω_c0² = 1/2` exactly; general `1/(1+(w₀p/w₀c)²)` | ≤ 1e-9 | §3.3a analytic identity | VERIFIED (this session) |
| S-6 | trade-structure invariance (the strategic test) | D3 `Ω_c` sweep through the propagation solver, `Ω_p/Γ_e ∈ {→0, 0.1, 0.5, 1.0}` | `α(Ω_p→0)` bit-identical to thin cell; `|Δα|` at finite `Ω_p` **reported** | `Δα < 0.02` ⇒ "scale only"; `> 0.05` ⇒ FAIL-open, structural | §5.3 | self-check; **decision rule normative** |
| S-7 | corpus geometry gate | every spec-09 corpus entry | `D/λ_rf` and `2z_R/L` computed and stamped | must match §3.1/§3.3b tables to 1 % | §3.1, §3.3b | VERIFIED (this session) |
| S-8 | RF-inhomogeneity z-weighting | linear `E_RF(z)` ramp ±10 %, OD ∈ {0, 5} | inverted field shifts by the `exp(−OD(z))`-weighted mean | rel ≤ 2 % vs the analytic weight | §3.2 | derived; self-checking |
| S-9 | screening decoupling | `T(f)` with and without z-resolved `P_c` | identical while coupling depletion < 1 % | rel ≤ 1e-6 | §3.4 | VERIFIED (by construction) |
| S-10 | density-trap ceiling fires | Rb, L = 5 cm, T sweep 20 → 140 °C | `IntegrityError` for `N > 3.5e12 cm⁻³` **or** OD > 100; flag (not raise) for 10 < OD ≤ 100 | exact threshold | §4.1, §7.1 R-P6 | normative choice, declared |
| S-11 | corrected-OD estimator | nat-Rb, 75 mm, 25 °C, `Ω_c = 0`, Rb-87 F=2 | shipped 1.3874 → corrected 0.578 vs spec-05 B9b Voigt model 0.481 | corrected/B9b within 25 % (single-line vs 3-line Voigt) | §7.4 D-1 | VERIFIED (this session) |
| S-12 | vapor model vs Jing's printed density | Cs, 298.15 K | `4.894e10 cm⁻³` vs printed `4.89e10` | ≤ 0.5 % | §2.3 | VERIFIED (this session) |

---

## 7. Refuse-to-guess boundaries for `rydsim.propagate`

### 7.1 The refusal list (audit style: each is a `raise`, not a warning, unless stated)

All raise `rydsim.provenance.IntegrityError` carrying the physics reason and the offending number.

| ID | Condition | Why refusing beats answering |
|---|---|---|
| **R-P1** | `2z_R/L < 1` for **either** beam | the 1-D z-model's own precondition is violated; the beam changes radius by >√2 inside the cell. Warn (not raise) for `1 ≤ 2z_R/L < 10`, with the exit/waist intensity ratio attached. **Fires today for E1-coupling, E2-probe, E6-probe, E6-coupling.** |
| **R-P2** | coupling depletion over the cell `> 1 %` | F-2's premise fails; `Ω_c(z)` and `τ_s,eff(z)` both become z-dependent and the ladder is no longer the modelled one (spec 05 §2.f) |
| **R-P3** | `Ω_p ≥ 0.01·min(Γ_e, |Ω_c|)` on the **analytic weak-probe** path | audit refusal #21, currently unimplemented (§7.4 D-3). The full z-coupled Lindblad path is the only route above it — and it, not the weak-probe path, is what the corpus needs |
| **R-P4** | on-axis-only evaluation requested for any amplitude observable (contrast, `κ_E`, `NEF`, EIT width) at OD > 0.1 | radial and longitudinal averaging do not commute (§3.3a); the error is 29.3 % in `Ω_c,eff` and 2× in EIT width. Frequency observables are exempt (§5.2 Grade A) |
| **R-P5** | radial-quadrature convergence not demonstrated (12 vs 32 Gauss–Laguerre nodes differ by > 1e-5) **or** z-step halving changes `T` by > 1e-4 | spec 05 §2.f/§4.8; convergence ships as data (`converged: bool` + magnitude), never as a docstring (audit §4 item 6) |
| **R-P6** | `N_total > 3.5×10¹² cm⁻³` **or** `OD > 100` | §4.1: the bistability density (Weller 2016), the MISSING D2 coefficient above 10¹² cm⁻³, and an UNVERIFIED dominant linewidth term all coincide there |
| **R-P7** | `10 < OD ≤ 100` **without** the collisional terms enabled and their 2× uncertainty propagated | in this band `Γ_self` is 0.6–6 % of `Γ_e` and Fermi broadening is 17–174 % of the default `deph_r`; running with them off is a silently wrong linewidth |
| **R-P8** | Rydberg density `N_r` (computed, not assumed) implies `Δν_vdW > 0.1 MHz` **or** `N_r > 1×10⁹ cm⁻³` | spec 04's own vdW warning threshold; above it the absent Rydberg–Rydberg/ionization physics (§3.5) is no longer a rounding error. The solver must **compute** `N_r = N·f_vel·ρ_rr`, never assume it |
| **R-P9** | any **absolute field** output without `field_reference="at_atoms"` and the computed `D/λ_rf` in the result object | §3.1: every corpus entry violates `D/λ_rf < 0.1` by 2–262×; a bare "E = … V/m" is a claim the model cannot support |
| **R-P10** | buffer-gas cells, wall-coated cells | audit refusal #16, unchanged and inherited: transit and collision models are invalid and must not be extrapolated |
| **R-P11** | vapor temperature outside 298–550 K **and** an absolute (not relative) OD claim | audit refusal #17 escalated: a warn-only extrapolation is acceptable for a relative sweep, not for an absolute optical depth that gates a refusal |
| **R-P12** | the z-solver returning a result while the propagation path was never actually exercised (weak probe, `Ω_p` below R-P3) — must return the closed form and **stamp** `path="beer_lambert_exact"` | §2.2: a no-op dressed as a computation is the "plausible but wrong" hazard in its purest form |

### 7.2 Warn-only (flagged in `validity_flags`, never silent)

`1 ≤ 2z_R/L < 10`; `0.1 < OD ≤ 10`; `D/λ_rf ≥ 0.1` (always true today — the flag is the honest
default, not an exception); screening-uncalibrated; Rb-85 ±40 MHz systematic; `N > 1×10¹² cm⁻³`
(spec 05 §7.2 hard warning, retained).

### 7.3 Provenance additions required of every propagation output

Beyond audit §4's twelve items: `OD_weak` and `OD_effective`; `path` ∈
{`beer_lambert_exact`, `z_coupled_weak`, `z_coupled_lindblad`}; `n_z`, z-halving magnitude;
radial node count and 12-vs-32 spread; coupling-depletion fraction; `2z_R/L` per beam; `D/λ_rf`;
`N_total`, `N_r`, `Δν_vdW`; `Γ_self`, `Δν_Fermi` and whether they were enabled; `s_p` at entrance and
exit; and the `d_eff,far`/`p_F` factors actually applied to the OD (§7.4 D-1).

### 7.4 Defects found in the shipped tree while writing this section

These are reported, not fixed here. All four are reproducible from the commands in §8.

* **D-1 (HIGH). The shipped OD estimator is not normative and the gate is mis-calibrated.**
  `experiment.superhet_transfer` forms `OD = k_p·L·Im χ` with `χ` built from
  `eit.dipole_from_linewidth`, which returns the **cycling** dipole
  `⟨J‖er‖J'⟩/√2 = 2.534451×10⁻²⁹ C·m` (Rb D2) — verified against Steck this session — where spec 00
  §6 gap 7 mandates `d_eff,far = ⟨J‖er‖J'⟩/√3 = 2.069362×10⁻²⁹ C·m` for **both** `Ω_p` and the `χ`
  prefactor. That is `d²` too large by exactly **1.5×**. It further applies no ground-hyperfine
  fraction `p_F`, a second **1/p_F** (1.60× Rb-87 F=2, 1.71× Rb-85 F=3, 1.78× Cs F=4). Net
  overstatement **2.40×** (Rb-87) to **2.67×** (Cs). Measured: nat-Rb 75 mm 25 °C, `Ω_c = 0` gives
  shipped 1.3874 → corrected 0.578, against spec 05 B9b's normative Voigt model 0.481. Consequence:
  `max_optical_depth = 5.0` actually binds at a true OD of ≈2.1, i.e. **just above the shot-noise
  optimum `OD* = 2`** — the engine has been refusing at almost exactly the operating point it should
  be recommending. Fix: route the OD through spec 05 §2.f's `rb_d2_weak_probe_od`-class model, or at
  minimum through `d_eff,far` and `p_F`, and re-derive the ceiling afterwards.
* **D-2 (MED). Audit refusal #18 is unimplemented and contradicts the shipped gate by 50×.**
  The audit mandates `ThickCellError` for OD > 0.1 through the analytic thin-cell path;
  `ThickCellError` does not exist anywhere in `src/rydsim`, and the only gate is
  `max_optical_depth = 5.0`, applied in `superhet_transfer` alone (not in `spectrum`, not in
  `at_experiment`). **Proposed ruling for spec 00** (this section does not enact it): the 0.1 figure
  was written for a strong-probe-capable path and is wrong for the weak-probe path, where
  Beer–Lambert is exact at any OD (§2.2, 2.8×10⁻¹⁴). Replace it with the pair
  `OD ≤ 100` (validity, §4.1 / R-P6) **and** `Ω_p < 0.01·min(Γ_e, Ω_c)` (R-P3) — a validity fence on
  the medium plus a validity fence on the method, rather than one number standing in for both.
* **D-3 (MED). Audit refusal #21 (weak-probe gate) is unimplemented.** No check on `Ω_p` exists in
  `eit.py`, `experiment.py` or `objective.py`. Since `LadderConfig.omega_probe` defaults to
  `2π·100 kHz` against `omega_coupling = 2π·5 MHz`, the **default configuration is already at
  `Ω_p/min(Γ_e,Ω_c) = 0.02`, twice the gate.** Every corpus config exceeds it by 283–588× (§2.4).
* **D-4 (LOW, doctrinal). The `OD → 0` reduction test cannot validate the new module.** Stated here
  so that no one ships a green S-1 as evidence the solver works. S-2 is the test that discriminates.

---

## 8. Sources, confidence register, and what remains MISSING

### 8.1 Fetched this session (2026-08-11) — VERIFIED

1. **Fan, Kumar, Sheng, Shaffer, Holloway & Gordon**, "Effect of Vapor-Cell Geometry on
   Rydberg-Atom-Based Measurements of Radio-Frequency Electric Fields," *Phys. Rev. Applied* **4**,
   044015 (2015). Full text extracted from the NIST-hosted PDF (`tsapps.nist.gov`, pub_id 918728);
   abstract independently confirmed from the NIST publication record. Used for: the FP/standing-wave
   mechanism; the `D/λ_rf < 0.1` criterion; Pyrex absorption 0.066 % at 12.6 GHz; cell set
   `D/λ_rf = 0.05–0.72`; the "current rf E-field standards ≈1 mV cm⁻¹ Hz⁻¹ᐟ² with an accuracy of
   5 %–20 %" baseline.
2. **Weller, Urvoy, Rico, Löw & Kübler**, "Charge-induced optical bistability in thermal Rydberg
   vapor," *Phys. Rev. A* **94**, 063820 (2016) [arXiv:1609.02330]. Abstract + body via ar5iv and
   arXiv abstract page. Used for: bistability densities (Rb 2.5×10¹² cm⁻³ total, Cs 1.2×10¹³ cm⁻³);
   2 % Rydberg fraction; `N_ion ≤ 1×10¹⁰ cm⁻³` = 27 % of `N_r`; `σ ≤ 1×10⁻³ µm²`; the ruling-out of
   dipole–dipole in favour of charge-induced.
3. **Lei, Eckel, Norrgard, Prajapati, Artusio-Glimpse, Simons & Holloway**, "Revisiting collisional
   broadening of ⁸⁵Rb Rydberg levels: conclusions for vapor cell manufacture," arXiv:2408.16669 →
   *Phys. Rev. Applied* **23**, 034028 (2025). Used for: VCC rate coefficients and the factor-5
   broadening/VCC ratio spread; the 0.01–0.02 mbar-for-1-MHz criterion; the ≤10⁻⁵ mbar seal
   statement that licenses neglecting foreign-gas VCC in evacuated cells.
4. **Su, Liou, Lin & Chen**, "Optimizing the Rydberg EIT spectrum in a thermal vapor," *Opt. Express*
   **30**(2), 1499–1510 (2022) [arXiv:2111.13408]. Full text extracted. Used for: the measured
   probe-intensity optimum and its temperature dependence; 13 % peak height at 51 °C,
   `I_p = 0.044 W/cm²`; fitted optical-density parameter `α = 165`.
5. **Richardson, Dee, Yaeger, Viray, Marsh, Kayim, Sawyer, La Mantia, Wyllie & Westafer**,
   "Extraction of Effective Electromagnetic Material Properties for Rydberg Electrometer Vapor Cells
   from 10–300 MHz," arXiv:2604.11785 (13 April 2026). Used for: effective permittivities
   (quartz 3.8, borosilicate 4.5, sapphire 9) and the observed in-cell field reduction and spatial
   degradation.
6. **Sedlacek, Kim, Rittenhouse, Weck, Sadeghpour & Shaffer**, "Electric Field Cancellation on
   Quartz by Rb Adsorbate-Induced Negative Electron Affinity," *Phys. Rev. Lett.* **116**, 133201
   (2016) — **citation VERIFIED** (three independent indexes); **numeric content MISSING** (PDF text
   would not extract).

### 8.2 Computed this session — VERIFIED (reproducible)

The OD table (§2.3, §4), the density-trap crossings (§4), the Beer–Lambert exactness result
(§2.2, 2.8×10⁻¹⁴), the Rayleigh-range table (§3.3b), the `⟨Ω_c²⟩ = Ω_c0²/2` identity (§3.3a), the
`D/λ_rf` table (§3.1), the `OD* = 2 / 1 / none` optima (§5.2), the `d_eff,far` vs cycling factor and
`p_F` correction (§7.4 D-1), the Cs-density-vs-Jing agreement (§2.3), and the confirmed firing of the
shipped OD gate on the E3 configuration. All from `rydsim` HEAD on branch
`feat/audited-physics-core-and-designer` plus scipy; scripts in the session scratchpad
(`od_scan.py`, `trap.py`, `geom.py`, `odcheck.py`) and to be ported into
`tests/test_spec10_scope.py` before release (audit R4's rule: session-measured numerics without an
in-repo harness are unreproducible assertions).

### 8.3 Carried from the repo, **not** re-fetched this session

Weller et al. 2011 self-broadening `β` values (spec 04 §3.5, ruling R-6) — **VERIFIED-in-repo**;
Jau & Carter 2020 screening parameters (spec 05 §2.h) — **VERIFIED-in-repo**; the Singer 2005 `C₆`
fit (spec 04 §3.5) — **LITERATURE-RECALL**; the Fermi-shift coefficient `−9.9×10⁻⁸ Hz·cm³` and the
`0.5·|shift|` broadening surrogate (spec 04 §2.3.5) — **UNVERIFIED, 2× uncertainty**; corpus
parameters (spec 09 §3.5) — **VERIFIED 2026-08-10**.

### 8.4 MISSING — named, with the resolution path

| # | Missing | Resolves the gap in | What would close it |
|---|---|---|---|
| M-1 | quantitative `E_int/E_inc` for RydSim's declared cell geometries | §3.1 — the largest gap in the program | full-wave FDTD/FEM of each cell with measured permittivity, or a per-cell measurement in the Richardson et al. style |
| M-2 | Holloway et al., *J. Appl. Phys.* **121**, 233106 (2017) numeric systematics budget | §3.1, §3.2, §5.2 Grade A caveat | fetch the paper (paywalled; audit §3 refusal #35 forbids filling from memory) — unchanged and still outstanding |
| M-3 | Sedlacek 2016 adsorbate field magnitudes and distances | §3.4 DC channel | readable copy of PRL 116, 133201 |
| M-4 | Durham-grade Rb **D2** self-broadening above 10¹² cm⁻³ | §4.1 item 3 — the ceiling's own coefficient | the Durham D2 follow-up to Weller 2011 (spec 05 §7.2 declares it MISSING; R-6 adopts the Table-I value, which is not the high-density validation) |
| M-5 | Penning/ionization rate law and free-charge field model for thermal Rydberg vapor | §3.5 | a primary source with rates, not the fitted cross-section bound of Weller 2016 |
| M-6 | alkali–alkali velocity-changing-collision cross-section | §3.6 | primary source; Lei 2024 covers foreign gases only |
| M-7 | a validated Rydberg–ground collisional **broadening** coefficient (today `0.5·|shift|`, UNVERIFIED, 2×) | §4.1 item 2 — it is the *dominant* Rydberg linewidth for `10 < OD ≤ 100` | measured density-dependent hot-cell EIT widths (spec 04's own stated self-check) |

---

## 9. Known limitations of this section itself

1. The `OD = 100` / `N = 3.5×10¹² cm⁻³` ceiling is a **declared judgement**, not a derived
   threshold. It is anchored on three independent facts (§4.1 item 3) but the specific number is a
   choice; changing it requires a spec edit with rationale (lock #20), never a code-side constant.
2. The Rydberg-density estimates in §3.5 assume `ρ_rr = 10⁻³` and `f_vel = Γ_e/Δν_D`. Both are
   order-of-magnitude placeholders; the refusal R-P8 is written to **compute** them, so the table is
   illustrative and the fence is not.
3. §5.2's `OD*` values assume an OD-independent relative transduction slope. That is exact in the
   weak-probe limit and approximate once the medium saturates; S-4 tests the closed forms, not the
   saturated case, and the saturated optimum must be *measured* by the solver.
4. The `D/λ_rf` table uses the cell dimensions as printed in spec 09 §3.5; where a paper printed only
   one dimension, the other was taken from the stated cell type. Two entries (E6, E7 transverse) are
   therefore estimates; they violate the criterion by so large a margin that the estimate cannot
   change the conclusion, but the numbers should not be quoted to two figures without the papers.
5. Nothing in this section validates the propagation *solver*. It fixes where the solver is allowed
   to speak. The core equations, their discretisation and their benchmarks belong to the other
   sections of spec 10.

---

*GreyNOC · RydSim spec 10 §1 · sources fetched 2026-08-11, network available · subordinate to spec
00 (locks 1–20, rulings R-1…R-28) · house rule: reproducible or it didn't happen.*

---

## Provenance of this draft section
### Sources FETCHED this session
- Fan, Kumar, Sheng, Shaffer, Holloway & Gordon, Phys. Rev. Applied 4, 044015 (2015) — FULL TEXT extracted this session from the NIST-hosted PDF (tsapps.nist.gov pub_id=918728), abstract independently confirmed from the NIST publication record page. Took: the Fabry-Perot standing-wave mechanism inside hollow glass cells (verbatim); the criterion 'not limited by the vapor-cell geometry provided D/lambda_rf < 0.1' (verbatim, 10-30 GHz); 'For a rf E-field at 12.6 GHz, the absorption by 1 mm of Pyrex is 0.066%' (verbatim); the studied cell set (8 mm and 9 mm cubic Pyrex, D/lambda_rf = 0.05-0.72); the baseline 'current rf E-field standards is approximately 1 mV cm-1 Hz-1/2 with an accuracy of 5%-20%'.
- Weller, Urvoy, Rico, Loew & Kuebler, 'Charge-induced optical bistability in thermal Rydberg vapor', Phys. Rev. A 94, 063820 (2016) [arXiv:1609.02330] — fetched this session via ar5iv full text and the arXiv abstract page (for the author list and verbatim abstract). Took: ground densities N85 = 1.8e12 cm^-3, N87 = 0.7e12 cm^-3 (T_res 80-120 C, T_cell 135 C), Cs N = 1.2e13 cm^-3; Rydberg fraction ~2%; ion density N_ion <= 1e10 cm^-3 and 'this ion density matches 27% of the Rydberg density'; ionization cross-section 'up to sigma = 1e-3 um^2 = 0.03 sigma_geo'; states Rb 32S/41S, Cs 23D3/2/28S1/2; the conclusion that dipole-dipole is ruled out and charge-induced bistability is the mechanism.
- Lei, Eckel, Norrgard, Prajapati, Artusio-Glimpse, Simons & Holloway, 'Revisiting collisional broadening of 85Rb Rydberg levels: conclusions for vapor cell manufacture', arXiv:2408.16669 -> Phys. Rev. Applied 23, 034028 (2025) — fetched this session (arXiv HTML). Took: broadening coefficients (25D+He 8.2(6)e-10 Hz cm^3; 25D+Ar broadening 21.6(1.2)e-10, shift -103.6(1.1)e-10 Hz cm^3); VCC rate coefficients <sigma v> tabulated in 1e-9 cm^3/s and the statement that the broadening/VCC ratio varies by only ~a factor of 5 across four gases; the tolerance criterion 'roughly 0.02 mbar of contaminant gas would be required to add roughly 1 MHz of additional broadening' and 'p ~ 0.01 mbar at 30 C'; the statement that cells are 'generally evacuated to well below 10^-5 mbar prior to sealing'; states studied 25D, 27S, 30D, 32S, 35D, 37S; explicit absence of Rb-Rb self-collisional broadening from that study.
- Su, Liou, Lin & Chen, 'Optimizing the Rydberg EIT spectrum in a thermal vapor', Opt. Express 30(2), 1499-1510 (2022) [arXiv:2111.13408] — fetched this session (arXiv PDF, text extracted locally with pypdf). Took: 'For any given temperature, the peak height has a universal behavior that it reaches the maximum value with the optimum I_p. The optimum I_p becomes stronger with a higher vapor temperature' (verbatim); 'We achieved the maximum EIT peak height of 13% with the vapor temperature of 51 C and I_p of 0.044 W/cm2' (verbatim); the fitted optical-density parameter alpha = 165 for a typical spectrum; probe/coupling 1/e^2 waists ~0.81 mm; Rb-87 33D3/2 and 33D5/2.
- Richardson, Dee, Yaeger, Viray, Marsh, Kayim, Sawyer, La Mantia, Wyllie & Westafer, 'Extraction of Effective Electromagnetic Material Properties for Rydberg Electrometer Vapor Cells from 10-300 MHz', arXiv:2604.11785 (13 April 2026) — fetched this session (arXiv HTML). Took: effective complex permittivities (unfilled quartz 3.8+0j, unfilled borosilicate 4.5+0j, sapphire portion 9+0j, borosilicate portion 4+0j); the observation of in-cell 'field reduction and spatial degradation' for filled cells; the 10-300 MHz band. Noted explicitly that the paper contains no etalon/standing-wave/resonance discussion.
- NIST publication record for 'Effect of Vapor Cell Geometry on Rydberg Atom-based Radio-frequency Electric Field Measurements' (nist.gov/publications/...) — fetched this session; verbatim abstract used to corroborate the criterion statement independently of the PDF extraction.
- Sedlacek, Kim, Rittenhouse, Weck, Sadeghpour & Shaffer, 'Electric Field Cancellation on Quartz by Rb Adsorbate-Induced Negative Electron Affinity', Phys. Rev. Lett. 116, 133201 (2016) — CITATION ONLY verified this session, from three independent indexes (UNLV institutional repository record, OSTI 1244776, PubMed 27081976) plus arXiv:1511.03754. The PDF (CfA-hosted) would not yield extractable text, so NO numeric content is taken from it; the adsorbate field magnitudes are recorded as MISSING (M-3).
- RydSim repository itself, read this session as primary source for the 'what exists' claims: docs/spec/00-conventions.md, 00-integrity-audit.md, 04-lifetimes-decay-dephasing.md (self-broadening/C6/Fermi sections), 05-vapor-cell-physics.md, 06-optical-bloch-eit.md, 09-validation-corpus.md, and src/rydsim/{eit.py, experiment.py, cell.py}. Took: the shipped max_optical_depth = 5.0 gate and its docstring rationale; the absence of ThickCellError and of any weak-probe gate (grep/AST-level confirmation); dipole_from_linewidth's cycling-dipole convention; corpus parameters (spec 09 section 3.5).

### UNVERIFIED / recall-only
- Rb/Cs D2 resonant self-broadening coefficients (beta/2pi = 1.03e-7 and 1.16e-7 Hz cm^3) — carried from spec 04 section 3.5 (ruling R-6, tagged VERIFIED there against Weller et al. 2011 Table I / Kondo 2006). NOT re-fetched this session. The whole density-trap table depends on them linearly.
- Rydberg-ground (Fermi pseudopotential) shift coefficient -9.9e-8 Hz cm^3 and, worse, the broadening surrogate 'broadening = 0.5 x |shift|' — spec 04 tags the latter UNVERIFIED with a 2x uncertainty flag. This is the DOMINANT Rydberg linewidth term in the 10 < OD <= 100 band, so the band's error bars rest on a guess. Named as MISSING item M-7.
- Singer et al. 2005 C6 fit coefficients used for the van der Waals estimates in section 3.5 — LITERATURE-RECALL in spec 04 (audit R11), not re-fetched. The vdW numbers are order-of-magnitude only and are explicitly labelled so.
- Jau & Carter 2020 screening numbers (tau_s,dark ~ 1 s sapphire, kappa_ph = 1.7 s^-1/mW, 64 Hz / 770 Hz measured cutoffs) — carried from spec 05 section 2.h (VERIFIED there), NOT re-fetched this session.
- Holloway et al., J. Appl. Phys. 121, 233106 (2017) numeric systematics budget — still paywalled and still unfetched (audit refusal #35 forbids filling it from memory). Only the qualitative linearity-regime claim is available, and it is carried from the repo's 2026-08-10 abstract-level verification, not from a fetch this session.
- Sedlacek 2016 PRL 116, 133201 adsorbate field MAGNITUDES, atom-surface distances and time constants — citation verified, numbers not extractable. Nothing numeric from that paper is used anywhere in the section.
- Cell dimensions for two corpus entries used in the D/lambda_rf table (E6 transverse dimension, E7 transverse dimension) are inferred from the stated cell type rather than printed in spec 09 section 3.5. Flagged in section 9 item 4; the violation margins (27x, 67x) are far too large for the inference to change the conclusion, but the individual figures should not be quoted to two significant figures.
- The 'Rydberg fraction rho_rr = 1e-3' and 'f_vel = Gamma_e/Delta_nu_D' used to populate the Rydberg-density table are order-of-magnitude placeholders, not computed steady-state values. The refusal R-P8 is written to compute them at runtime precisely so the fence does not inherit the placeholder.
- The normative ceiling itself (OD = 100 / N = 3.5e12 cm^-3) is a declared judgement, not a derived threshold. Its three anchors are sourced; the specific number is a choice and is labelled as such in section 9 item 1.
- Su et al.'s fitted 'alpha = 165' is quoted as their optical-density fit parameter. Their normalisation of alpha inside their transmission expression was not independently reconstructed, so it must NOT be read as a peak optical depth of 165 without checking their Eq. (5). Used here only as qualitative corroboration that real cells run well above OD 5.

### Open questions
- THE ONE THAT REORDERS THE WORK: in the strict weak-probe limit the z-propagation solver is provably a no-op (Beer-Lambert is exact at any OD; measured 2.8e-14 relative at OD 10). So is the deliverable actually 'thick-cell propagation', or is it 'strong-probe propagation'? The published corpus runs 283-588x outside RydSim's own weak-probe gate, which says the latter. Recommend the spec-10 title and work plan be re-scoped to the z-coupled STRONG-PROBE solve before any code is written.
- Should spec 00 be amended to replace audit refusal #18 ('OD > 0.1 -> ThickCellError') with the pair (OD <= 100 validity fence on the medium) AND (Omega_p < 0.01 min(Gamma_e, Omega_c) validity fence on the method)? The 0.1 figure is wrong for the weak-probe path, where Beer-Lambert is exact at any OD, and the shipped code contradicts it by 50x anyway. This section proposes the amendment but does not enact it — it needs a spec-00 ruling.
- The shipped OD estimator overstates OD by 2.40x (Rb-87) to 2.67x (Cs) — cycling dipole instead of d_eff,far (1.5x in d^2) and no ground-hyperfine fraction p_F. That means max_optical_depth = 5.0 actually binds at a TRUE OD of ~2.1, i.e. just above the shot-noise-limited optimum OD* = 2. Has the engine been refusing at almost exactly the operating point it should be recommending? What does the retracted-and-rerun D3 campaign look like once the estimator is fixed?
- What is |E_int/E_inc| for RydSim's declared cell geometries? Every corpus entry violates Fan et al.'s D/lambda_rf < 0.1 criterion by 2x to 262x, and the size of the induced error is MISSING. This is the largest unbounded systematic in the program. Does it warrant its own EM/cell module (full-wave FDTD/FEM), or does the program accept a permanent 'at_atoms only, never incident' claim boundary?
- Four of seven corpus beams fail the collimated criterion 2 z_R/L >= 1 (E1 coupling 0.44, E2 probe 0.17, E6 probe 0.44, E6 coupling 0.22). A 1-D z-propagation model cannot represent them. Is a paraxial 2-D (r, z) split-step solver in scope for this program, or do E1/E2/E6 get permanently downgraded to QUALITATIVE grading on any amplitude observable?
- The Rydberg-ground (Fermi) collisional broadening surrogate 'broadening = 0.5 x |shift|' is UNVERIFIED with a declared 2x uncertainty, and it becomes the DOMINANT Rydberg linewidth term exactly in the 10 < OD <= 100 band that thick-cell propagation is being built to reach. Is that band publishable at all before M-7 is closed, or should the ceiling drop to OD = 10 until it is?
- RydSim has no ionization, free-charge, or bistability physics at all, and the density that buys OD ~ 100 (Rb 3.5e12 cm^-3) is essentially the density at which Weller et al. observed charge-induced optical bistability (Rb ~2.5e12 cm^-3, 2% Rydberg fraction). The 20x higher Rydberg fraction of a real strong-probe experiment raises the vdW dephasing estimate by ~390x. Does the strong-probe solver need an ionization fence BEFORE it ships, rather than after?
- Audit refusal #21 (the weak-probe gate) is unimplemented, and LadderConfig's own DEFAULTS sit at twice the gate (Omega_p/min = 0.02 vs 0.01). Should the gate be implemented as written — which would refuse the shipped default config — or should the defaults move first?
- S-6 is the strategic benchmark and its outcome is genuinely unknown. If |Delta alpha| > 0.05 at finite Omega_p, the thin-cell trade findings are retracted for a second time. Does the program want that test run before or after the solver is considered done? Recommendation: before, on a stub, because a FAIL changes what the solver is for.
- The session's measured numerics live in scratchpad scripts (od_scan.py, trap.py, geom.py, odcheck.py), not in the repo. Audit R4's rule says that makes them unreproducible assertions. Who ports them into tests/test_spec10_scope.py, and does that block the section from being marked normative?

### Proposed benchmarks

| id | quantity | expected | tol | source | conf |
|---|---|---|---|---|---|
| S-1 | Weak-probe degeneracy: z-propagation solver vs closed-form Beer-Lambert at high OD | \|Omega_p(L)/Omega_p(0)\|^2 = exp(-OD) identically; measured 4.2937676186e-05 both ways at OD = 10.0558 (Rb-87, 60 C, L = 5 cm, Omega_p/2pi = 1 Hz, RK4 20001 steps) | relative <= 1e-10 (measured this session: 2.8e-14) | Analytic: in the weak-probe limit chi is Omega_p-independent so dOmega_p/dz = i(k_p/2)chi Omega_p has constant coefficients. Computed and verified this session with the shipped rydsim chain. | VERIFIED (analytic identity + numerical measurement this session) |
| S-2 | The DISCRIMINATING reduction test: OD -> 0 at FINITE probe Rabi frequency (Omega_p = Gamma_e/2), z-solver vs thin-cell strong-probe chi | z-solver converges to the thin-cell strong-probe answer as OD -> 0, with residual error scaling linearly in OD | relative <= 1e-6 at OD = 1e-3; log-log slope of error vs OD = 1.00 +/- 0.05 over OD in {1e-3, 1e-2, 1e-1} | Derived in this section (2.2): the plain OD -> 0 test is degenerate because it is satisfied at every OD, so it cannot validate the module; the finite-Omega_p limit is the one that discriminates. | VERIFIED (by construction) |
| S-3 | AT peak-position invariance under propagation vs drift of the FITTED splitting | True stationary points of T(Delta_p) coincide exactly with those of Im chi at every OD (since T = exp(-k_p L Im chi) and exp is strictly monotone), so the true AT splitting is propagation-invariant; the Lorentzian-fitted splitting is NOT, and its drift is a measured output | true positions: relative <= 1e-9 across OD in {0.1, 1, 5, 20}; fitted-splitting drift reported as data, not asserted | Theorem derived in this section (5.2 Grade A); extraction algorithm from spec 06 section 2.7 | VERIFIED (theorem); drift is a measurement |
| S-4 | Location of the optical-depth optimum in NEF, by dominant noise term | OD* = 2 exactly (shot-noise-limited, S_P ~ P); OD* = 1 exactly (detector-NEP-limited, S_P constant); no interior optimum (RIN-limited, S_P ~ P^2, NEF ~ 1/OD monotone) | relative <= 1e-6 on OD* | Closed forms derived here from kappa_E ~ P_in exp(-OD) OD; stationary points confirmed numerically this session by bounded minimisation | VERIFIED (analytic + numerical, this session) |
| S-5 | Probe-power-weighted radial average of Omega_c^2 over a Gaussian beam (equal waists) | <Omega_c^2>/Omega_c0^2 = 1/2 exactly, i.e. Omega_c,eff = 0.70711 x on-axis (on-axis overstates by 29.3%); general case 1/(1 + (w0p/w0c)^2), measured 0.800 at ratio 0.5 and 0.200 at ratio 2.0 | <= 1e-9 (24-node Gauss-Laguerre reproduced 0.500000 this session) | Analytic identity from spec 05 section 2.g beam profile; computed this session | VERIFIED (analytic identity, numerically confirmed) |
| S-6 | Trade-structure invariance under propagation: NEF-vs-Omega_c exponent alpha from the D3 sweep, run through the propagation solver at Omega_p/Gamma_e in {->0, 0.1, 0.5, 1.0} | alpha(Omega_p -> 0) bit-identical to the thin-cell value (consequence of S-1); \|Delta alpha\| at finite Omega_p is the measured strategic result | \|Delta alpha\| < 0.02 => 'scale only, thin-cell trade findings stand'; \|Delta alpha\| > 0.05 => FAIL-open, structural change, thin-cell trade findings retracted and re-derived | Decision rule defined in this section (5.3); baseline from the repo's valid-regime d3-trade-law-v2 measurement (NEF U-shaped in Omega_c, alpha < 0.15 above the optimum) | self-check; the decision rule is NORMATIVE, the outcome is unknown until run |
| S-7 | Corpus geometry gate: D/lambda_rf and 2 z_R/L stamped for every spec-09 corpus entry | D_long/lambda_rf = 4.75 (E1, 14.233 GHz), 4.26 (E2, 17.04 GHz), 26.2 (E2, 104.77 GHz), 1.16 (E3, 6.94 GHz), 2.67 (E6, 16.03 GHz), 0.17/6.67 (E7 at 1/40 GHz) — all violate Fan's D/lambda_rf < 0.1 by 2x to 262x. 2 z_R/L = 15.1 (E1 probe), 0.44 (E1 coupling), 0.17 (E2 probe), 107 (E3 probe), 247 (E3 coupling), 0.44 (E6 probe), 0.22 (E6 coupling) | computed values must match these tables to 1% | Fan et al. PRApplied 4, 044015 (2015) criterion (fetched this session); Gaussian-optics z_R = pi w0^2 / lambda; corpus parameters from spec 09 section 3.5; all figures computed this session | VERIFIED (this session) |
| S-8 | RF-inhomogeneity z-weighting: inverted field under a linear E_RF(z) ramp of +/-10% | the inverted field shifts between OD -> 0 and OD = 5 by the exp(-OD(z))-weighted mean of the ramp (front-of-cell weighting), not by the unweighted mean | relative <= 2% against the analytic weight | Derived in this section (3.2); the machinery exists as spec 06 section 2.8 item 7 but no corpus entry publishes an E_RF map | derived; self-checking |
| S-9 | Screening decoupling: T(f) computed with and without a z-resolved coupling power P_c(z) | identical results while the coupling depletion over the cell is below 1% | relative <= 1e-6 | Section 3.4 of this document; screening model is spec 05 section 2.h under ruling R-7 | VERIFIED (by construction, conditional on the R-P2 depletion assertion holding) |
| S-10 | Density-trap ceiling fires: Rb, L = 5 cm, temperature swept 20 -> 140 C | IntegrityError raised for N_total > 3.5e12 cm^-3 (Rb, ~92 C at 5 cm) or OD > 100, whichever binds first; a mandatory validity flag (not a raise) for 10 < OD <= 100; clean pass below OD 10 (<= ~60 C) | exact threshold (no tolerance); the anchor table is T = 33.9 / 59.9 / 91.8 / 130.6 C for OD = 1 / 10 / 100 / 1000 at L = 5 cm | Section 4.1 of this document. Anchors: Weller et al. PRA 94, 063820 (2016) bistability at Rb ~2.5e12 cm^-3 (fetched this session); spec 05 section 7.2 MISSING D2 self-broadening above 1e12 cm^-3; spec 04 UNVERIFIED Fermi broadening surrogate. OD/temperature mapping computed this session. | coefficients VERIFIED-in-repo; the ceiling VALUE is a declared normative judgement |
| S-11 | Corrected optical-depth estimator vs the spec-05 normative Voigt model (nat-Rb, 75 mm, 25 C, Omega_c = 0, Rb-87 F=2) | shipped estimator 1.3874; after the two normative corrections (d_eff,far/d_cycling squared = 2/3, and p_F = 5/8) 0.578; spec 05 B9b Voigt multi-line model 0.481 | corrected value within 25% of B9b (the residual is the single-effective-line vs three-line-Voigt treatment); the 1.5x dipole factor itself must reproduce to <= 1e-4 | spec 00 section 6 gap 7 (d_eff,far = <J\|\|er\|\|J'>/sqrt(3)); spec 05 sections 2.b and 2.f; Steck reduced element 3.58424e-29 C m. All computed this session. | VERIFIED (this session) |
| S-12 | Alcock/Steck vapor model vs Jing 2020's printed ground-state density | N_Cs(298.15 K) = 4.8941e10 cm^-3 vs Jing's printed N0 = 4.89e10 cm^-3 | <= 0.5% | rydsim.cell (Alcock/Steck, spec 05 section 2.a) computed this session; printed value from spec 09 section 3.5 (arXiv:1902.11063, fetched 2026-08-10) | VERIFIED (this session) |
