# 05 — Vapor Cell Physics and Optical Propagation

RydSim physics specification. Species: Rb-85, Rb-87, Cs-133. Python 3.11 + numpy/scipy, first-principles.

**Verification status:** Network WAS available during authoring. All Steck-datasheet numbers below were
extracted directly from the current PDFs (`steck.us/alkalidata`, revision 2.3.4, 8 August 2025, all three
datasheets) and are tagged VERIFIED. Every computed number in this document was produced by executing the
formulas in this spec with CODATA-2018 constants; the scripts are reproducible from the equations as written.
Items tagged LITERATURE-RECALL or UNVERIFIED are explicitly marked and carry a self-check recipe.

**Independent verification pass (2026-08-10, network available):** every Steck-tagged constant in §3 was
re-read from the rev 2.3.4 PDFs (Rb85/Rb87/Cs Tables 2, 3, 5, 7, 8 and the Eq. (1) vapor-pressure text);
every computed table (§2.a densities, §2.c speeds/Doppler widths, §2.d factors, §2.e transit, §2.f B9
spectrum, §2.g beam numbers, §2.h screening examples) was independently recomputed from the equations as
written, including a from-scratch Racah-6j recomputation of §3.4 — all reproduced to displayed digits.
Citations Mohapatra 2007, Siddons 2008, Weller 2011, Bouchiat 1999, Sedlacek 2016, Jau & Carter 2020,
Ma 2022, Zhou 2022, Alcock 1984 were re-verified online; former LITERATURE-RECALL tags upgraded where noted.

---

## 1. Scope

- (a) Alkali vapor pressure P_v(T) and number density n(T) for Rb and Cs, solid and liquid phases.
- (b) Isotope fractions → partial density of the sensed isotope.
- (c) Maxwell–Boltzmann velocity distribution; Doppler widths at probe/coupling wavelengths.
- (d) Doppler averaging for ladder (Rydberg) EIT: co-/counter-propagation, wavelength-mismatch factor,
  velocity quadrature scheme and convergence.
- (e) Transit-time broadening through a Gaussian beam and its correct folding into the velocity average.
- (f) Optical propagation: χ ↔ absorption/dispersion, transmission, optical depth, thick-cell z-integration.
- (g) Beam geometry: Rabi frequency from power/waist, radial averaging.
- (h) Cell-wall screening of low-frequency applied fields: phenomenological S(f) model.

Out of scope here (owned by other spec docs): Rydberg state energies/dipoles (λ_c values, d_c), the N-level
density-matrix steady state ρ(Δ_p, Δ_c, Ω_p, Ω_c, v) (this doc consumes it), Stark physics of the sensed field.

Conventions used throughout: SI units; T in kelvin (T[K] = T[°C] + 273.15); angular frequencies ω, Ω, Γ, γ in
rad/s; ν, Δν, f in Hz; probe propagates along +ẑ; v ≡ v_z is the atom velocity component along +ẑ.

---

## 2. Equations

### 2.a Vapor pressure and number density

Steck/Alcock form (P_v in **torr**, T in K; omit the 2.881 term for atmospheres):

```
Rb (solid,  T < T_melt = 312.45 K):  log10(P_v) = 2.881 + 4.857 − 4215/T
Rb (liquid, T ≥ 312.45 K):           log10(P_v) = 2.881 + 4.312 − 4040/T
Cs (solid,  T < T_melt = 301.65 K):  log10(P_v) = 2.881 + 4.711 − 3999/T
Cs (liquid, T ≥ 301.65 K):           log10(P_v) = 2.881 + 4.165 − 3830/T
```

No `C·log10(T)` term appears in this model (the Alcock et al. fit for Rb/Cs in this range is two-parameter;
some Nesmeyanov-style fits carry a third term — do not mix coefficient sets across models).
Stated accuracy ±5% over 298–550 K; below 298 K it is an extrapolation (flag in code).

Ideal-gas number density (alkali vapor is ideal to ≪1% at these pressures):

```
n(T) = P_v(T) / (kB · T)        [m^-3],  P_v in Pa;  1 torr = 133.322 Pa
```

**Worked example — natural-abundance Rb cell** (all digits reproducible from the formulas above;
kB = 1.380649e-23 J/K exact):

| T | Phase | P_v [torr] | P_v [Pa] | n_total [cm⁻³] | n(⁸⁷Rb) = 0.2783·n | n(⁸⁵Rb) = 0.7217·n |
|---|---|---|---|---|---|---|
| 25.00 °C = 298.15 K | solid | 3.989e-7 | 5.318e-5 | 1.292e10 | 3.595e9 | 9.323e9 |
| 50.00 °C = 323.15 K | liquid | 4.910e-6 | 6.546e-4 | 1.467e11 | 4.083e10 | 1.059e11 |
| 100.00 °C = 373.15 K | liquid | 2.324e-4 | 3.099e-2 | 6.014e12 | 1.674e12 | 4.341e12 |

Cs at 298.15 K (solid): P_v = 1.511e-6 torr → n = 4.894e10 cm⁻³; at 323.15 K: n = 4.670e11 cm⁻³.

Cross-check against Steck's quoted "vapor pressure at 25 °C": Steck evaluates at T = 298 K (not 298.15 K):
Rb 10^(7.738 − 4215/298) = 3.92e-7 torr = Steck's 3.92(20)e-7; Cs 10^(7.592 − 3999/298) = 1.488e-6 =
Steck's 1.488(74)e-6. The implementation must reproduce both (benchmarks B1–B2). Note the sensitivity:
d(ln P_v)/dT = ln(10)·4215/T² ≈ 11%/K for Rb at room temperature — see Pitfalls.

### 2.b Isotope fractions and partial density

```
n_iso = η_iso · n_total(T)
```

Natural abundance: η(⁸⁵Rb) = 0.7217(2), η(⁸⁷Rb) = 0.2783(2) [Steck, VERIFIED]; η(¹³³Cs) = 1 exactly
(only stable Cs isotope). For isotopically enriched cells η is a cell parameter. Isotopes form a nearly ideal
solution in the condensed reservoir, so each isotope's partial pressure is its condensed-phase mole fraction
times the elemental P_v (Raoult's law; isotope vapor-pressure ratio deviates from 1 by ≪1%, negligible here).
The hyperfine ground levels are further populated thermally — at these temperatures the splitting
(3–9 GHz ≪ kB·T/h) gives degeneracy weighting only:

```
p_F = (2F+1) / Σ_F (2F+1)
⁸⁷Rb: p(F=1)=3/8,  p(F=2)=5/8      ⁸⁵Rb: p(F=2)=5/12, p(F=3)=7/12      ¹³³Cs: p(F=3)=7/16, p(F=4)=9/16
```

### 2.c Maxwell–Boltzmann distribution and Doppler widths

3D speed distribution and characteristic speeds (m = atomic mass in kg):

```
f3D(u) du = 4π u² (m/2πkB T)^(3/2) exp(−m u²/2kB T) du
v_p   = sqrt(2 kB T / m)            (most probable)
v_bar = sqrt(8 kB T / (π m))        (mean)
v_rms = sqrt(3 kB T / m)
```

1D distribution along the beam (the one used for Doppler averaging), σ_v = sqrt(kB T/m) = v_p/√2:

```
f1D(v) dv = (1/(v_p √π)) exp(−v²/v_p²) dv          ⟨v⟩=0, ⟨v²⟩ = σ_v²
```

Doppler-broadened (Gaussian) FWHM of an optical line at rest frequency ν₀ = c/λ:

```
Δν_D = ν₀ · sqrt(8 ln2 · kB T / (m c²)) = (2/λ) · sqrt(2 ln2 · kB T / m)   [Hz FWHM]
Gaussian standard deviation: σ_ν = Δν_D / (2 sqrt(2 ln2)) = ν₀ σ_v / c
```

Computed values (VERIFIED constants, arithmetic reproducible):

| Quantity | 300 K | 350 K | 400 K |
|---|---|---|---|
| ⁸⁷Rb v_p / v_bar / v_rms [m/s] | 239.6 / 270.3 / 293.4 | 258.8 / 292.0 / 316.9 | 276.6 / 312.2 / 338.8 |
| Cs v_p / v_bar / v_rms [m/s] | 193.7 / 218.6 / 237.3 | 209.3 / 236.1 / 256.3 | 223.7 / 252.4 / 274.0 |
| Δν_D ⁸⁷Rb @ 780.241 nm [MHz] | 511.3 | 552.3 | 590.4 |
| Δν_D ⁸⁷Rb @ 480.0 nm [MHz] | 831.1 | 897.7 | 959.7 |
| Δν_D Cs @ 852.347 nm [MHz] | 378.5 | 408.8 | 437.0 |
| Δν_D Cs @ 509.4 nm [MHz] | 633.3 | 684.0 | 731.3 |

(480.0 nm and 509.4 nm are nominal coupling wavelengths; the implementation must use the actual λ_c of the
chosen Rydberg state from the energy-level spec doc.)

### 2.d Doppler averaging in the ladder EIT scheme

Frame transformation (probe along +ẑ with wavevector k_p = 2π/λ_p; v > 0 = atom co-moving with probe):

```
Δp(v) = Δp − k_p v
Counter-propagating coupling (k_c along −ẑ):  Δc(v) = Δc + k_c v
Co-propagating coupling      (k_c along +ẑ):  Δc(v) = Δc − k_c v
Two-photon detuning:  δ2(v) = Δp + Δc + (k_c − k_p) v     (counter-prop)
                      δ2(v) = Δp + Δc − (k_c + k_p) v     (co-prop)
```

Since λ_c < λ_p for Rydberg ladders (k_c > k_p), **counter-propagation minimizes but does not cancel** the
two-photon Doppler shift. Raw two-photon inhomogeneous widths at 300 K (⁸⁷Rb, 780.241/480.0 nm):

```
counter: |1 − λp/λc| · Δν_D(probe) = 0.6255 × 511.3 MHz = 319.8 MHz
co:      (1 + λp/λc) · Δν_D(probe) = 2.6255 × 511.3 MHz = 1342 MHz
```

**The observed EIT linewidth is far narrower than 319.8 MHz.** Mechanism: at fixed probe detuning Δp only the
velocity class within ~Γ_e/k_p of one-photon resonance (v* = Δp/k_p) contributes absorption; across that class
the two-photon detuning varies by only (k_c − k_p)·(Γ_e/k_p) = (λp/λc − 1)·Γ_e ≈ 0.63 Γ_e ≈ 2π·3.8 MHz for Rb.
The Doppler-averaged EIT feature width is therefore of order γ_EIT + (λp/λc − 1)Γ_e plus power broadening
(few MHz in practice, consistent with first Rydberg-EIT cell observations: Mohapatra, Jackson, Adams,
PRL 98, 113003 (2007) [VERIFIED: reports EIT linewidth 2 MHz in a room-temperature Rb cell]). **Do not use
any closed-form linewidth in the simulator — it emerges from the velocity integral.** Closed forms are
sanity checks only.

**Wavelength-mismatch (probe-scan compression) factor.** The dominant contribution at probe detuning Δp comes
from v* = Δp/k_p; imposing δ2(v*) = 0 gives the mapping between coupling/Rydberg detuning space and probe-scan
space:

```
Δp_feature = −(k_p/k_c) Δc = −(λc/λp) Δc
```

A splitting Ω (rad/s) of the Rydberg level (e.g. Autler–Townes from the sensed RF field) therefore appears in a
**probe-frequency scan** as Δν_meas = (λc/λp)·Ω/2π, and the field is recovered with the standard scale factor:

```
Ω/2π = (λp/λc) · Δν_meas       λp/λc = 1.6255 (Rb 780.241/480.0),  1.6732 (Cs 852.347/509.4)
```

When the **coupling** laser is scanned with probe locked on resonance, the splitting is read off unscaled.
[Factor direction verified against Rydberg electrometry literature, e.g. Zhou et al., J. Phys. B 55, 075501
(2022) on Doppler mismatch in Rydberg EIT/AT electrometry; the derivation above is self-contained.]

**Doppler-averaged susceptibility.** With ρ_eg(Δp, Δc; v) the steady-state optical coherence from the N-level
solver (doc 04) evaluated at the shifted detunings:

```
χ(Δp) = (2 n d_ge² / (ε0 ħ Ω_p)) ∫ dv f1D(v) ρ_eg(Δp − k_p v, Δc ± k_c v)
```

Velocity-to-frequency conversion constants (Rb 780.241/480.0 nm): k_p/2π = 1.2817 MHz/(m/s),
k_c/2π = 2.0833 MHz/(m/s), (k_c−k_p)/2π = 0.8017 MHz/(m/s). Cs (852.347/509.4 nm): 1.1732 / 1.9631 /
0.7899 MHz/(m/s).

**Quadrature scheme (normative).**

- Narrowest velocity-space features: one-photon Lorentzian width Γ_e/k_p (⁸⁷Rb D2: 6.07 MHz / 1.2817 MHz·s/m
  = 4.73 m/s) and the EIT dip width γ_EIT/(k_c−k_p) (γ_EIT/2π = 1 MHz → 1.25 m/s).
- **Primary scheme: uniform trapezoid grid** over v ∈ [−4 v_p, +4 v_p] with step
  Δv ≤ min(Γ_e/k_p, γ_EIT_est/(k_c−k_p)) / 4. For Rb at 300 K with γ_EIT/2π ≈ 1 MHz: Δv ≈ 0.3 m/s →
  N ≈ 6400 points (Δv = 0.5 m/s → N = 3834; Δv = 1.0 m/s → N = 1917). Vectorize over v with numpy;
  the N-level solve per velocity class is the cost driver, not the grid.
- Optional two-tier composite grid for speed: coarse Δv = 2 m/s over ±4 v_p (captures the Gaussian absorption
  background) plus a fine window Δv = 0.2 m/s over v* ± 30 m/s around each probe detuning's resonant class.
- **Weights:** w_i = f1D(v_i)·Δv_i, then renormalize Σw_i = 1 (kills ±4σ truncation error, ~6e-5).
- **Convergence criterion:** halve Δv (and widen span to 4.5 v_p); accept when max|ΔT| < 1e-4 in transmission
  and the fitted EIT linewidth changes < 0.5%.
- **Gauss–Hermite is NOT acceptable as the primary scheme.** GH nodes near center are spaced
  ≈ π σ_v √2/√(2N); resolving a 1 m/s feature at σ_v = 169 m/s requires N ~ 3e5 nodes. GH (N = 20–40) is fine
  only for smooth Doppler-broadened absorption with no EIT (e.g. §2.f reference spectra) and may be used there.

### 2.e Transit-time broadening (Gaussian beam)

Beam intensity 1/e² radius w0; field envelope seen by an atom crossing the axis with transverse speed v⊥:
E(t) ∝ exp(−(v⊥ t)²/w0²) (field 1/e radius = w0 because field ∝ sqrt(intensity)). Its spectral power is
Gaussian; the exact transit-limited FWHM is:

```
Δν_tt(v⊥) = sqrt(2 ln2) · v⊥ / (π w0)          [Hz FWHM, derived, exact for this geometry]
```

Transverse speed distribution (2D Maxwell, independent of v_z):

```
f2D(v⊥) dv⊥ = (m v⊥ / kB T) exp(−m v⊥²/2kB T) dv⊥,   ⟨v⊥⟩ = sqrt(π kB T / 2m)
```

⁸⁷Rb 300 K: ⟨v⊥⟩ = 212.3 m/s → Δν_tt(⟨v⊥⟩) = 79.6 kHz at w0 = 1 mm (scales as 1/w0).

**Liouvillian model (convention, normative for RydSim):** model transit as relaxation of the full density
matrix toward the thermal ground state (fresh atoms enter, interacted atoms leave) at rate

```
γ_t(v⊥) = sqrt(2 ln2) · v⊥ / w0        [rad/s]   (⁸⁷Rb, 300 K, w0=1 mm: γ_t/2π = 39.8 kHz)
```

chosen so that a γ_t-dominated two-level line reproduces Δν_tt (Lorentzian-for-Gaussian substitution; correct
to ~±30% in shape, adequate because γ_t ≪ Γ_e always holds here). Apply γ_t to **all** populations and
coherences: L_transit[ρ] = γ_t (ρ_thermal − ρ).

**Correct folding into the Doppler average.** v_z (Doppler) and v⊥ (transit) are statistically independent —
the 3D Maxwell distribution factorizes. The "fast atoms Doppler-shift more AND dwell less" correlation exists
only through |v|; once resolved into components there is no correlation between the axial shift and the
transverse dwell. Normative scheme:

```
χ(Δp) = ∫ dv_z f1D(v_z) ∫ dv⊥ f2D(v⊥) · χ_v(Δp − k_p v_z, Δc ± k_c v_z; γ_t(v⊥))
```

Quadrature for v⊥: 4–8 Gauss–Laguerre nodes in x = m v⊥²/2kB T (γ_t enters smoothly). Shortcut
γ_t(⟨v⊥⟩) for all classes is permitted when γ_t < 0.1·γ_EIT; it errs on the EIT linewidth at the few-% level —
gate it behind a config flag and test against the full 2D integral.

### 2.f Optical propagation

Field convention: E(z,t) = ½ Ê(z) e^{i(k_p z − ω t)} + c.c.; Rabi frequency Ω_p = d_ge Ê/ħ;
polarization P = ε0 χ Ê with χ from §2.d. Then n_refr = sqrt(1+χ) ≈ 1 + χ/2 (|χ| ≪ 1 always holds here;
χ ~ 1e-6 at 25 °C) and:

```
Ê(L) = Ê(0) · exp( i k_p L/2 · χ )                      (uniform medium)
Intensity transmission:  T = exp( − k_p · Im(χ) · L )    ← exact factor convention: k_p = 2π/λ_p, no extra 2 or 4π
Optical depth:           OD(ν) ≡ k_p Im(χ(ν)) L = α(ν) L,   α = k_p Im χ = n σ(ν)
Dispersive phase:        Δφ = (k_p L / 2) · Re(χ)
```

Sanity identity (two-level, weak probe, v = 0): α(0) = n σ0 with σ0 = 2 ω d²/(c ε0 ħ Γ) = ħωΓ/(2 I_sat) =
3λ²/2π (closed transition). Numeric self-check: ⁸⁷Rb D2 cycling d = 2.53444e-29 C·m, Γ = 3.8117e7 s⁻¹ →
σ0 = 2.9067e-9 cm², equal to Steck's tabulated 2.906693e-9 cm² (VERIFIED, benchmark B8).

**Weak-probe reference spectrum (Doppler-broadened, no EIT)** — used for OD benchmarks and cell thermometry.
Sum over isotopes i, ground levels F, excited levels F′:

```
α(ν) = Σ_i Σ_F Σ_F′  n_i · p_F · σ0(F,F′) · (π Γν/2) · V(ν − ν_{i,F,F′}; Γν, σ_ν,i)
σ0(F,F′) = 2 ω d_eff²/(c ε0 ħ Γ),   d_eff² = (1/3) S_FF′ |⟨J||er||J′⟩|²
V = unit-area Voigt = Re[wofz( ((ν−ν_t) + i Γν/2) / (σ_ν √2) )] / (σ_ν √(2π)),   Γν = Γ/2π  [Hz FWHM]
ν_{i,F,F′} = ν_centroid,i + E_hf(F′)/h − E_hf(F)/h
E_hf = (A/2)K + B·[ (3/2)K(K+1) − 2I(I+1)J(J+1) ] / [ 2I(2I−1)·2J(2J−1) ],   K = F(F+1) − I(I+1) − J(J+1)
```

This is the Siddons et al. model (J. Phys. B 41, 155004 (2008)), experimentally validated to 0.2% rms on a
75 mm natural-Rb cell (D2, 16.5 °C, probe intensity < I_sat/1000) — the definitive literature anchor for this
subsection. Computed dips for natural Rb, L = 75.0 mm (this spec's own numbers; benchmark B9):

| T_cell | Dip (ground level) | Detuning from ⁸⁷Rb D2 centroid | T_min | OD_peak |
|---|---|---|---|---|
| 25.0 °C | ⁸⁷Rb F=2 | −2.424 GHz | 0.619 | 0.481 |
| 25.0 °C | ⁸⁵Rb F=3 | −1.288 GHz | 0.273 | 1.298 |
| 25.0 °C | ⁸⁵Rb F=2 | +1.619 GHz | 0.385 | 0.956 |
| 25.0 °C | ⁸⁷Rb F=1 | +4.094 GHz | 0.732 | 0.312 |
| 16.5 °C | ⁸⁵Rb F=3 | −1.288 GHz | 0.594 | 0.521 |
| 50.0 °C | ⁸⁵Rb F=3 | −1.288 GHz | 6.7e-7 | 14.2 |

**Validity criteria and thick-cell scheme (normative):**

- Weak-probe: I_p < 0.01·I_sat for ≲1% accuracy (Siddons used < 0.001·I_sat for 0.2%). In Rabi terms
  Ω_p ≲ Γ_e/7.
- Optically thin (use single-exponential with χ evaluated once): OD_peak ≤ 0.1. Above that the probe Rabi
  frequency decays along z, and where the medium is nonlinear (EIT, saturation) χ depends on Ω_p(z):
  integrate

```
dΩ_p/dz = i (k_p/2) χ(Ω_p(z), Ω_c(z); z) Ω_p(z)
```

  with RK4; step control Δz such that per-step |ΔOD| = k_p Im χ Δz ≤ 0.05 and per-step |ΔΩ_p|/|Ω_p| ≤ 2%.
  Check coupling-beam absorption once per run (upper transition from a nearly-empty intermediate state —
  normally negligible; assert < 1% and warn otherwise). Re-solve the velocity integral at each z level.
  Convergence: halve Δz, require |ΔT| < 1e-4.

### 2.g Beam geometry and Rabi frequencies

TEM00 Gaussian beam, total power P, waist w0 (intensity 1/e² radius), Rayleigh range z_R = π w0²/λ (assert
L ≪ 2 z_R or model w(z); for w0 = 1 mm at 780 nm, z_R = 4.0 m ≫ 75 mm — collimated assumption fine):

```
I(r) = I0 exp(−2r²/w0²),   I0 = 2P/(π w0²)
Ê0 = sqrt(2 I0 / (ε0 c))            [V/m, peak field amplitude]
Ω = d · Ê0 / ħ                      [rad/s]   — RydSim convention: Ω = dÊ0/ħ with E = ½Ê0 e^{−iωt} + c.c. ≡ Ê0 cos ωt
```

(Steck defines Ω = −d·Ê0/ħ with E = Ê0 cos ωt — same magnitude. Watch factor-2 traps: some papers define Ω
with the half-amplitude Ê0/2. The Hamiltonian in doc 04 uses ħΩ/2 off-diagonal couplings; keep consistent.)

Worked examples (VERIFIED arithmetic): P = 1 µW, w0 = 1.0 mm → I0 = 0.0637 mW/cm², Ê0 = 21.9 V/m;
⁸⁷Rb D2 far-detuned π dipole (2.06936e-29 C·m): Ω_p/2π = 0.684 MHz ≈ Γ/8.9 — weak-probe OK,
I0/I_sat = 0.025 against the matching far-detuned-π I_sat(det,eff,D2) = 2.50399(73) mW/cm² [Steck, V]
(NOT the cycling I_sat = 1.66933 mW/cm², which pairs with d = 2.53444e-29 C·m; mixing the pairs is a bug).
Coupling: P = 30 mW, w0 = 1.0 mm, d(5P3/2→50D, order 0.012 e·a0, actual value from doc 03) →
Ω_c/2π ≈ 0.58 MHz.

**Radial averaging (measured transmission over the detected beam).** With s ≡ 2r²/w0², the power-weighted
transmission of the probe (whole-beam detection) is exactly:

```
T_meas = ∫0^∞ e^(−s) · T(Ω_p(s), Ω_c(s)) ds,   Ω_p²(s) = Ω_p0² e^(−s),   Ω_c²(s) = Ω_c0² e^(−s·w0p²/w0c²)
```

Evaluate with 8–16 node Gauss–Laguerre in s (integrand smooth; 12 nodes reproduce 32-node result to <1e-5).
Unequal waists enter only through the exponent ratio shown. For an apertured detector, integrate s over
[0, 2r_ap²/w0²] with Gauss–Legendre. Pitfall: radial averaging washes out Autler–Townes contrast — never
simulate at peak intensity only.

### 2.h Cell-wall screening of low-frequency applied fields — phenomenological model

Physical story (all statements sourced): alkali adsorbed on the inner cell wall makes glass surfaces
electrically conductive; free/surface charge redistributes to cancel applied quasi-static fields inside the
cell. Measured: glass cells exposed to Cs vapor become conductive (wall resistance ~5 orders of magnitude
below sapphire; monocrystalline sapphire stays at GΩ level) — Bouchiat, Guéna, Jacquier, Lintz, Papoyan,
"Electrical conductivity of glass and sapphire cells exposed to dry cesium vapor," Appl. Phys. B 68,
1109–1116 (1999) [VERIFIED]. In a monocrystalline-sapphire Rb cell the *intrinsic* (dark) screening
timescale reaches "up to ~ second" vs order 1e-3 s for fused silica; best sapphire cell surface resistivity
R_□,0 = (4.7 ± 1)e12 Ω/sq; screening time scales as ~ ε R_□ V^(1/3) (first-order estimate, V = cell
volume) — Jau & Carter, "Vapor-Cell-Based Atomic Electrometry for Detection Frequencies below 1 kHz,"
Phys. Rev. Applied 13, 054034 (2020) / arXiv:2002.04145 [VERIFIED, values read from paper]. **Critical
caveat from the same paper:** screening is photo-activated — their measured 3-dB low-cutoff was ≈ 64 Hz at
P_480 = 10 mW and ≈ 770 Hz at P_480 = 120 mW (with P_780 = 200 µW), far above the dark-screening corner,
with a fitted photo-activation coefficient γ/P_480 = (1.7 ± 0.06) s⁻¹/mW for their best cell — and the
measured cutoffs still exceeded the value deduced from that coefficient (vapor-density and 780-nm-intensity
dependent). So the operational τ_s is set by illumination, not by wall resistivity alone. DC fields can be
delivered with wall-integrated electrodes: Ma, Viray, Anderson, Raithel, "Measurement of dc and ac Electric
Fields inside an Atomic Vapor Cell with Wall-Integrated Electrodes," Phys. Rev. Applied 18, 024001 (2022)
[VERIFIED; they measure ~5 V/cm dc inside a cell at 10% relative uncertainty and quantify attenuation by
free surface charges]. Related surface physics: adsorbate-induced field cancellation on quartz — Sedlacek,
Kim, Rittenhouse, Weck, Sadeghpour, Shaffer, PRL 116, 133201 (2016) [VERIFIED].

**THIS IS A PHENOMENOLOGICAL MODEL** — it parameterizes, it does not derive. The interior field responds as a
first-order (optionally stretched) high-pass filter to the externally applied field:

```
E_int(f) = S(f) · E_ext(f)
S(f) = S_geo · (i 2πf τ_s)^β / (1 + (i 2πf τ_s)^β)          (complex; use |S| for amplitude, arg S for phase)
|S(f)| = S_geo · x/sqrt(1+x²)  for β=1,  x = f/f_c,  f_c = 1/(2π τ_s)
```

Time domain (β = 1): τ_s dE_int/dt + E_int = S_geo τ_s dE_ext/dt.

**Photo-activated screening (normative extension).** The effective screening time under illumination is

```
1/τ_s,eff = 1/τ_s,dark + κ_ph · P_c        [s⁻¹; P_c = coupling power in mW at the cell]
```

with κ_ph a per-cell calibration constant (Jau & Carter measured γ/P_480 = (1.7 ± 0.06) s⁻¹/mW on their best
sapphire cell, and observed cutoffs *above* this deduction at high density/probe intensity — treat κ_ph as a
lower bound calibrated at operating conditions). Use τ_s,eff in S(f). Consequence: quoting S(f) without the
optical powers it was calibrated at is meaningless; RydSim reports must log (τ_s,dark, κ_ph, P_c, τ_s,eff).

Parameters, defaults, provenance:

| Parameter | Meaning | Default / range | Provenance & confidence |
|---|---|---|---|
| τ_s,dark | intrinsic (dark) screening time | sapphire: 1 s (range 0.1–2 s); fused silica: 1e-3 s (order); borosilicate/pyrex: 1e-4 s (range 1e-5–1e-2 s) | Sapphire "up to ~ second" and fused-silica "order 1e-3 s": Jau & Carter 2020, VERIFIED (read from paper). Borosilicate range: LITERATURE-RECALL — **must be calibrated per cell**; treat as fit parameter. |
| κ_ph | photo-activation coefficient of screening rate | 1.7 s⁻¹/mW (coupling power at cell); range 0–10; treat as lower bound | Jau & Carter 2020, γ/P_480 = (1.7±0.06) s⁻¹/mW, VERIFIED — but condition-dependent (density, probe intensity); calibrate per cell at operating point. |
| S_geo | high-frequency (unscreened) geometric/dielectric factor | slab model below; ≈1.29 (borosilicate), ≈1.34–1.35 (sapphire) for 2 mm walls / 10 mm gap | Derived (exact for parallel-slab geometry); ε_r values LITERATURE-RECALL: borosilicate ≈ 4.6, sapphire ⊥c ≈ 9.4, ∥c ≈ 11.6 |
| β | stretch exponent (non-Debye surface relaxation) | 1.0; fit range 0.5–1 | Modeling freedom; Bouchiat 1999 impedance spectra motivate a distributed-RC (β<1) option. UNVERIFIED as a universal law. |

Slab (capacitive-divider) factor for external plates across a cell with interior gap g and two walls of
thickness t_w, permittivity ε_r (field normal to walls; exact electrostatics for infinite slabs):

```
S_geo = (g + 2 t_w) / (g + 2 t_w/ε_r)
```

Behavior encoded: |S| → S_geo for f ≫ f_c (kHz–MHz fields enter the cell, up to cavity/standing-wave effects
at GHz which are out of scope); |S| → 0 as f → 0 (DC is screened in equilibrium). Example: τ_s = 1e-4 s
(glass) → f_c = 1.59 kHz, |S(60 Hz)|/S_geo = 0.038, |S(1 kHz)|/S_geo = 0.53 — i.e. a glass cell kills 60 Hz
sensing; τ_s,dark = 1 s (sapphire) → f_c = 0.16 Hz, |S(60 Hz)|/S_geo = 1.000 — this is what makes the
kHz-band findings meaningful. But with κ_ph = 1.7 s⁻¹/mW and P_c = 120 mW, τ_s,eff = 4.88 ms →
f_c = 32.6 Hz (and Jau & Carter *measured* 770 Hz under those powers) — a dark-parameter simulation of a
brightly-driven sapphire cell overstates low-frequency response by orders of magnitude. **Every RydSim
output that used S(f) must report (τ_s,dark, κ_ph, P_c, τ_s,eff, S_geo, β) and calibration status.** Calibration recipe: apply known E_ext at several f spanning 0.01–100·f_c_expected,
measure the Stark/AT response, fit (τ_s,eff, S_geo, β) at the operating optical powers; repeat at a second
P_c to separate τ_s,dark from κ_ph. Photo-induced charge drift (Rydberg ionization, photoemission) appears
as a slow additive bias field E_bias(t) — a separate calibration channel; Jau & Carter exploited an
optically-induced internal bias field deliberately.

---

## 3. Constants / parameter tables

Confidence: **V** = VERIFIED (read from source during authoring), **LR** = LITERATURE-RECALL, **U** = UNVERIFIED.

Fundamental (CODATA 2018, as adopted by Steck rev 2.3.4): kB = 1.380649e-23 J/K (exact),
u = 1.66053906660e-27 kg, c = 299792458 m/s (exact), ε0 = 8.8541878128e-12 F/m, ħ = 1.054571817e-34 J·s,
h = 6.62607015e-34 J·s (exact), e = 1.602176634e-19 C (exact), a0 = 5.29177210903e-11 m. All **V**.
1 torr = 133.322 Pa (definition, 101325/760 exact).

### 3.1 Vapor pressure / thermophysical

| Quantity | Value | Source | Conf. |
|---|---|---|---|
| Rb solid coeffs (torr) | 2.881 + 4.857, −4215 | Alcock, Itkin, Horrigan, Can. Metall. Q. 23, 309 (1984) via Steck Rb85/Rb87 rev 2.3.4 | V |
| Rb liquid coeffs (torr) | 2.881 + 4.312, −4040 | ibid. | V |
| Cs solid coeffs (torr) | 2.881 + 4.711, −3999 | ibid. via Steck Cs rev 2.3.4 | V |
| Cs liquid coeffs (torr) | 2.881 + 4.165, −3830 | ibid. | V |
| Model accuracy | ±5%, 298–550 K | Steck (as stated by Alcock) | V |
| Rb melting point | 39.30 °C = 312.45 K | Steck Rb87 Table 2 (CRC) | V |
| Cs melting point | 28.5 °C = 301.65 K | Steck Cs Table 2 (CRC) | V |
| Rb boiling point | 688 °C | Steck Rb87 (CRC) | V |
| P_v(Rb, 298 K) | 3.92(20)e-7 torr | Steck Rb87/Rb85 Table 2 | V |
| P_v(Cs, 298 K) | 1.488(74)e-6 torr | Steck Cs Table 2 | V |

### 3.2 Masses, abundances, D2 line

| Quantity | ⁸⁵Rb | ⁸⁷Rb | ¹³³Cs | Source / Conf. |
|---|---|---|---|---|
| Mass [u] | 84.911789732(14) | 86.909180520(15) | 132.905451931(27) | Steck rev 2.3.4 / V |
| Natural abundance | 0.7217(2) | 0.2783(2) | 1 (only stable) | Steck / V |
| Nuclear spin I | 5/2 | 3/2 | 7/2 | Steck / V |
| D2 centroid ν₀ [THz] | 384.230406373(14) | 384.2304844685(62) | 351.72571850(11) | Steck / V |
| D2 λ_vac [nm] | 780.241368271(27) | 780.241209686(13) | 852.34727582(27) | Steck / V |
| D2 Γ [1e6 s⁻¹] (Γ/2π MHz) | 38.117(11) (6.0666(18)) | 38.117(11) (6.0666(18)) | 32.889(84) (5.234(13)) | Steck / V |
| ⟨J=1/2‖er‖J′=3/2⟩ [e·a0] | 4.22753(62) | 4.22752(62) | 4.4837(57) | Steck / V |
| same [1e-29 C·m] | 3.58425 | 3.58424(52) | 3.8014 | Steck / V |
| I_sat cycling [mW/cm²] | — | 1.66933(49) | — | Steck Rb87 / V (others in resp. datasheets) |
| σ0 cycling [1e-9 cm²] | — | 2.906692937721(93) | — | Steck Rb87 / V |
| ⁸⁷Rb−⁸⁵Rb D2 centroid shift | +78.095 MHz | | | derived from the two V frequencies / V |

### 3.3 Hyperfine constants (D2 manifold, for line positions)

| Constant | ⁸⁵Rb | ⁸⁷Rb | ¹³³Cs | Source / Conf. |
|---|---|---|---|---|
| A(ground S1/2) | h·1.0119108130(20) GHz | h·3.417341305452145(45) GHz | h·2.2981579425 GHz (exact) | Steck rev 2.3.4 / V |
| A(P3/2) | h·25.0354(69) MHz | h·84.7185(20) MHz | h·50.28827(23) MHz | Steck / V |
| B(P3/2) | h·25.898(91) MHz | h·12.4965(37) MHz | h·−0.4934(17) MHz | Steck / V |

### 3.4 Line-strength factors S_FF′ (D2), computed from the 6-j formula, cross-checked against Steck Rb87 Table 8 (exact match) and the sum rule Σ_F′ S_FF′ = 1 (all rows sum to 1 to 1e-9)

| | values | Conf. |
|---|---|---|
| ⁸⁷Rb F=1 | S10=1/6, S11=5/12, S12=5/12 | V (matches Steck Table 8) |
| ⁸⁷Rb F=2 | S21=1/20, S22=1/4, S23=7/10 | V (matches Steck Table 8) |
| ⁸⁵Rb F=2 | S21=3/10, S22=7/18, S23=14/45 | V (computed + sum rule) |
| ⁸⁵Rb F=3 | S32=5/63, S33=5/18, S34=9/14 | V (computed + sum rule) |
| ¹³³Cs F=3 | S32=5/14, S33=3/8, S34=15/56 | V (computed + sum rule) |
| ¹³³Cs F=4 | S43=7/72, S44=7/24, S45=11/18 | V (computed + sum rule) |

### 3.5 Nominal sensing-scheme wavelengths (actual values from the Rydberg-structure doc)

| Quantity | Value | Conf. |
|---|---|---|
| Rb coupling λ_c (5P3/2→nS/nD) | ≈ 479–484 nm, nominal 480.0 nm | U (state-dependent; compute from doc 02 energies) |
| Cs coupling λ_c (6P3/2→nS/nD) | ≈ 508–512 nm, nominal 509.4 nm | U (state-dependent) |

### 3.6 Screening-model parameters — see table in §2.h (phenomenological; τ_s glass is LR and per-cell calibrated)

---

## 4. Numerical method + pitfalls

1. **Temperature sensitivity is the #1 systematic.** d(ln n)/dT ≈ 11–12 %/K (Rb, 300 K). A ±0.5 K cell
   temperature error is a ±6% density error — larger than the ±5% model accuracy. Steck's "25 °C" table
   values are evaluated at 298 K, not 298.15 K (1.7% difference); benchmarks below pin both. Always propagate
   T in kelvin; never round to 3 digits before exponentiating.
2. **Phase selection:** use the solid branch strictly below T_melt, liquid at/above. The branches disagree by
   ~2% at T_melt (fit seam) — do not average or interpolate across it; document which branch was used near Tm.
3. **Units:** the vapor formulas return torr. Convert P → Pa (×133.322) before n = P/kB T. Keep Γ, Ω, γ in
   rad/s and ν, Δν in Hz; the factor 2π lost between them is the classic bug (benchmarks B4/B8 catch it).
4. **Voigt evaluation:** use `scipy.special.wofz` exactly as in §2.f. Never sample a bare Lorentzian on a grid
   coarser than Γν/4, and never convolve numerically with FFT without zero-padding ≥ 8 Δν_D (wrap-around
   contaminates the wings that set the off-resonance dispersion).
5. **Velocity quadrature:** uniform grid per §2.d; renormalize weights; convergence by halving. GH forbidden
   for EIT (quantified in §2.d). When scanning Δp, the fine window tracks v* = Δp/k_p — recompute the
   composite grid per detuning point or use the fixed fine grid covering the scanned v* range.
6. **Doppler sign conventions:** one consistent frame (§2.d). Self-check: with the coupling far detuned, the
   Doppler-averaged absorption dip must sit at Δp = 0 and have FWHM ≈ sqrt(Δν_D² + (Γ/2π)²)-ish (Voigt);
   with counter-propagating beams the EIT feature moves as −(λc/λp)Δc — verify both signs in a unit test.
7. **Weak-probe / thin-cell gates:** enforce I_p < 0.01 I_sat and OD ≤ 0.1 for the analytic path; otherwise
   run the z-propagation RK4 of §2.f with the stated step controls. Guard `exp(−OD)` for OD > 700 (underflow
   → return 0.0 explicitly).
8. **Radial averaging:** Gauss–Laguerre 12 nodes; verify vs 32 nodes once per run (< 1e-5). Peak-intensity-only
   simulation overestimates AT contrast — forbidden for comparisons with experiment.
9. **Transit:** γ_t must scale ∝ 1/w0 and ∝ sqrt(T); assert both in tests. Use the smaller of probe/coupling
   waists for the interaction-zone w0 (conservative; exact treatment would need the overlap profile).
10. **Screening:** S(f) multiplies the *applied low-frequency signal field* only — never the optical fields,
    never a resonant microwave field in the GHz band (different physics: cavity/dielectric effects, out of
    scope). Log calibration status of (τ_s, S_geo, β) with every result.

---

## 5. Recommended Python API

```python
# rydsim/vapor.py — numpy-vectorized; all inputs SI unless suffixed otherwise.
from dataclasses import dataclass
from typing import Callable
import numpy as np
from numpy.typing import ArrayLike

@dataclass(frozen=True)
class VaporPressureModel:
    """log10(P_torr) = offset + A - B/T per phase. Source: Steck rev 2.3.4 / Alcock 1984."""
    A_solid: float; B_solid: float
    A_liquid: float; B_liquid: float
    T_melt_K: float
    T_valid_K: tuple[float, float] = (298.0, 550.0)   # warn outside

@dataclass(frozen=True)
class AlkaliSpecies:
    name: str                  # "Rb85" | "Rb87" | "Cs133"
    mass_kg: float
    natural_fraction: float    # eta_iso; overridable per-cell
    vp_model: VaporPressureModel   # element-level (shared by Rb85/Rb87)

def vapor_pressure_Pa(T_K: ArrayLike, model: VaporPressureModel) -> np.ndarray:
    """Alcock/Steck model; solid branch for T < T_melt_K, liquid otherwise.
    Emits warnings.warn outside T_valid_K. Vectorized over T_K."""

def number_density_m3(T_K: ArrayLike, species: AlkaliSpecies,
                      isotope_fraction: float | None = None) -> np.ndarray:
    """n = eta * P/(kB*T). isotope_fraction=None -> species.natural_fraction."""

@dataclass(frozen=True)
class SpeedScales:
    v_p: float; v_mean: float; v_rms: float; sigma_1d: float; v_perp_mean: float

def maxwell_speeds(T_K: float, mass_kg: float) -> SpeedScales: ...
def doppler_fwhm_Hz(wavelength_m: float, T_K: float, mass_kg: float) -> float: ...
def f1d_velocity_pdf(v: ArrayLike, T_K: float, mass_kg: float) -> np.ndarray: ...

def velocity_grid(T_K: float, mass_kg: float, *, k_p: float, k_c: float,
                  gamma_e: float, gamma_eit_est: float = 2*np.pi*1e6,
                  n_sigma: float = 4.0, mode: str = "uniform",
                  fine_center_mps: float | None = None,
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Return (v, w): nodes [m/s] and renormalized weights (sum w == 1).
    mode='uniform': step = min(gamma_e/k_p, gamma_eit_est/(k_c-k_p))/4 over +-n_sigma*v_p.
    mode='composite': 2 m/s coarse + 0.2 m/s fine window around fine_center_mps (+-30 m/s).
    Raises ValueError for mode='gauss-hermite' if gamma_eit_est is finite (see spec 2.d)."""

def doppler_average(chi_of_v: Callable[[np.ndarray], np.ndarray],
                    v: np.ndarray, w: np.ndarray) -> complex:
    """sum(w * chi_of_v(v)); chi_of_v must be vectorized over v."""

def transit_rate_rad_s(v_perp: ArrayLike, waist_m: float) -> np.ndarray:
    """gamma_t = sqrt(2 ln 2) * v_perp / w0   (Liouvillian relaxation toward thermal ground state)."""

def transit_fwhm_Hz(T_K: float, mass_kg: float, waist_m: float) -> float:
    """sqrt(2 ln2) * <v_perp> / (pi * w0), <v_perp> = sqrt(pi kB T / 2 m)."""

@dataclass(frozen=True)
class GaussianBeam:
    power_W: float; waist_m: float; wavelength_m: float
    @property
    def peak_intensity_W_m2(self) -> float: ...    # 2P/(pi w0^2)
    @property
    def peak_field_V_m(self) -> float: ...         # sqrt(2 I0 / (eps0 c))
    def rabi_rad_s(self, dipole_Cm: float) -> float: ...   # d*E0/hbar
    @property
    def rayleigh_range_m(self) -> float: ...

def susceptibility(rho_eg: complex | np.ndarray, n_m3: float,
                   dipole_Cm: float, omega_rabi: float) -> complex | np.ndarray:
    """chi = 2 n d^2 rho_eg / (eps0 hbar Omega_p); rho_eg in the rotating frame of doc 04."""

def transmission_thin(chi: complex, k_p: float, L_m: float) -> float:
    """exp(-k_p Im(chi) L). Assert k_p*Im(chi)*L <= 0.1 else raise ThickCellError."""

def propagate_probe(chi_fn: Callable[[float, float], complex],  # (omega_p, z) -> chi
                    omega_p0: float, k_p: float, L_m: float, *,
                    dz_max_od: float = 0.05) -> "PropagationResult":
    """RK4 of dOmega_p/dz = i k_p/2 chi Omega_p. Returns T, phase, Omega_p(z) samples,
    and a convergence record (halved-step comparison)."""

def radial_average(T_of_s: Callable[[np.ndarray], np.ndarray], n_nodes: int = 12) -> float:
    """integral_0^inf e^-s T(s) ds by Gauss-Laguerre; s = 2r^2/w0^2."""

def screening_tau_eff(tau_dark_s: float, kappa_ph_per_s_mW: float,
                      P_coupling_mW: float) -> float:
    """1/tau_eff = 1/tau_dark + kappa_ph * P_c (spec 2.h, Jau & Carter 2020 photo-activation).
    kappa_ph is a per-cell lower-bound calibration constant."""

def screening_factor(f_Hz: ArrayLike, tau_s: float, S_geo: float = 1.0,
                     beta: float = 1.0) -> np.ndarray:
    """Complex S(f) = S_geo (i 2 pi f tau)^beta / (1 + (i 2 pi f tau)^beta). PHENOMENOLOGICAL —
    pass tau_s = screening_tau_eff(...) for illuminated cells; callers must surface
    (tau_dark, kappa_ph, P_c, tau_eff, S_geo, beta) and calibration status in reports."""

def slab_geometry_factor(gap_m: float, wall_m: float, eps_r: float) -> float:
    """(g + 2 t)/(g + 2 t/eps_r)."""

def rb_d2_weak_probe_od(detuning_Hz: ArrayLike, T_C: float, L_m: float,
                        eta85: float = 0.7217, eta87: float = 0.2783) -> np.ndarray:
    """Reference Doppler-broadened OD spectrum (spec 2.f: Voigt sum over both isotopes,
    S_FF' line strengths, Steck hyperfine constants). Detuning relative to the Rb-87 D2
    centroid. This is the validation workhorse (benchmarks B9) and the cell thermometer."""
```

Contracts: every function accepts scalars or arrays and broadcasts; no function silently clips — out-of-domain
inputs warn or raise; all defaults cited to this spec section numbers in docstrings.

---

## 6. Validation benchmarks (→ pytest)

Tolerances: "num" = pure-arithmetic reproduction of this spec (tight); "phys" = physical/model accuracy for
comparison against measurement. Implementations must pass "num"; report "phys" in docs.

| ID | Quantity | Expected | Tolerance (num) | Source | Conf. |
|---|---|---|---|---|---|
| B1 | P_v(Rb, 298 K), solid branch | 3.92e-7 torr | ±1% (phys ±5%) | Steck Rb87 Table 2 / Alcock | V |
| B2 | P_v(Cs, 298 K), solid branch | 1.488e-6 torr | ±1% (phys ±5%) | Steck Cs Table 2 / Alcock | V |
| B3a | n_total(Rb, 298.15 K) | 1.292e16 m⁻³ | ±0.5% | computed from V inputs (§2.a) | V |
| B3b | n(⁸⁷Rb) partial, 298.15 K, natural | 3.595e15 m⁻³ | ±0.5% | ibid. ×0.2783 | V |
| B3c | n_total(Rb, 323.15 K) [liquid branch] | 1.467e17 m⁻³ | ±0.5% | ibid. | V |
| B3d | n_total(Rb, 373.15 K) | 6.014e18 m⁻³ | ±0.5% | ibid. | V |
| B3e | n_total(Cs, 298.15 K) | 4.894e16 m⁻³ | ±0.5% | ibid. | V |
| B4a | Δν_D(⁸⁷Rb, 780.241 nm, 300 K) | 511.3 MHz | ±0.1% | §2.c formula, V constants | V |
| B4b | Δν_D(⁸⁷Rb, 780.241 nm, 400 K) | 590.4 MHz | ±0.1% | ibid. | V |
| B4c | Δν_D(⁸⁷Rb, 480.0 nm, 300 K) | 831.1 MHz | ±0.1% | ibid. | V |
| B4d | Δν_D(Cs, 852.347 nm, 300 K) | 378.5 MHz | ±0.1% | ibid. | V |
| B5 | v_p / v_mean / v_rms (⁸⁷Rb, 300 K) | 239.6 / 270.3 / 293.4 m/s | ±0.1% | §2.c | V |
| B6 | λp/λc scale factor (780.241/480.0) | 1.6255 (probe-scan splitting × this = Ω/2π) | exact ratio | §2.d derivation; Zhou 2022 | V |
| B7 | Transit FWHM (⁸⁷Rb, 300 K, w0 = 1 mm) | 79.6 kHz | ±1% | §2.e formula | V (formula-derived) |
| B8 | σ0 cycling ⁸⁷Rb D2 from 2ωd²/(cε0ħΓ) | 2.906693e-9 cm² | ±0.05% | Steck Rb87 Table 7 | V |
| B9a | Nat-Rb 75.0 mm, 25.0 °C: OD at ⁸⁵Rb F=3 dip | 1.298 (T_min = 0.273) | ±1% (phys ±6%: vapor model + T) | §2.f model (Siddons-validated method) | V-computed |
| B9b | same cell: OD at ⁸⁷Rb F=2 dip | 0.481 (T_min = 0.619) | ±1% | ibid. | V-computed |
| B9c | dip positions vs ⁸⁷Rb centroid | −2.424 / −1.288 / +1.619 / +4.094 GHz | ±3 MHz | Steck hyperfine constants | V |
| B9d | Nat-Rb 75.0 mm, 16.5 °C: T_min(⁸⁵Rb F=3) | 0.594 | ±1% (phys: Siddons 2008 agree 0.2% rms) | §2.f + Siddons J.Phys.B 41,155004 | V-computed |
| B10 | S_FF′ sum rule, every ground F, all species | Σ_F′ S_FF′ = 1 | 1e-9 | §3.4 / Steck Eq. 42 | V |
| B11 | ⁸⁷Rb S_2F′ = {1/20, 1/4, 7/10} | exact fractions | 1e-12 | Steck Rb87 Table 8 | V |
| B12 | \|S(f_c)\|/S_geo (screening, β=1) | 1/√2 | 1e-12 | §2.h model self-consistency | model |
| B13 | GaussianBeam: P=1 µW, w0=1 mm → I0, Ê0 | 0.6366 W/m², 21.9 V/m | ±0.1% | §2.g | V |
| B14 | Doppler-average convergence: T(Δp) grid halving | max\|ΔT\| < 1e-4 | — | §2.d scheme | scheme |
| B15 | EIT feature drag: dip position vs Δc slope = −λc/λp | −0.6152 (Rb, 480.0 nm) | ±0.5% | §2.d; catches sign/convention bugs | V-derived |
| B16 | τ_s,eff(τ_dark=1 s, κ_ph=1.7 s⁻¹/mW, P_c=120 mW) → f_c | 4.878 ms → 32.63 Hz | ±0.1% | §2.h model arithmetic (params: Jau & Carter 2020) | model |

Independent cross-validation (non-pytest, manual): compare `rb_d2_weak_probe_od` output against the published
Siddons 2008 (arXiv:0805.1139) figures at 16.5/25.4/36.6 °C, and against the open ElecSus package output if
available in the validation environment (do NOT vendor it — comparison only).

---

## 7. Known limitations / model breakdown

1. **Vapor model:** ±5% (298–550 K); extrapolation below 25 °C; getter/stem effects and cold-spot dynamics
   in real cells mean the *cell* density can deviate from the reservoir model — treat T as the cold-spot
   temperature and calibrate via B9-style absorption thermometry.
2. **Density regime:** above n ≈ 1e12 cm⁻³ (Rb ≳ 90–100 °C) resonant dipole–dipole self-broadening becomes
   non-negligible on the D lines. Anchor: Weller, Bettles, Siddons, Adams, Hughes, "Absolute absorption on
   the rubidium D1 line including resonant dipole–dipole interactions," J. Phys. B 44, 195006 (2011)
   [VERIFIED]: **D1** self-broadening β/2π = (0.69 ± 0.04)e-7 Hz·cm³, linear in n, validated to 0.1% up to
   3e14 cm⁻³ (i.e. 69 kHz at 1e12 cm⁻³ — comparable to γ_EIT). The **D2** coefficient is MISSING here
   (not verified this session — same order of magnitude expected; add from the Durham group's D2 follow-up
   before supporting hot cells, and self-check via the n-linearity of fitted Voigt Lorentzian widths).
   Rydberg-Rydberg interactions and ionization also activate at high density/Ω_c. This doc's linear-in-n
   optics fails there.
3. **No buffer gas, no wall coatings** in the collision model: this spec assumes evacuated
   alkali-only cells. Buffer-gas pressure broadening/shift and Rydberg quenching are absent (out of scope;
   would enter χ via extra γ terms).
4. **Transit model** is a Lorentzian-rate surrogate for a Gaussian process (±30% shape error where γ_t
   dominates; it never dominates for w0 ≥ 0.5 mm at our temperatures). Beam-profile distortion by the cell
   windows is ignored.
5. **Screening model S(f)** is phenomenological (first-order high-pass plus a linear photo-activation
   term). It cannot predict τ_s ab initio; τ_s,eff depends on optical power, vapor density, and cell
   history (alkali dosing, light-induced desorption, photoelectric charging), and relaxation may be
   non-exponential (β < 1). Jau & Carter's measured cutoffs exceeded even their own fitted photo-activation
   deduction — the κ_ph model is a lower bound, not a prediction. Any absolute low-frequency field claim
   requires the in-situ calibration of §2.h. Above ~100 MHz, cell-dielectric/cavity resonances (not modeled) modify the internal
   field instead.
6. **Propagation** assumes a scalar, paraxial, transversely-uniform-per-shell model (radial shells don't
   diffract into each other); valid for L ≪ 2 z_R and OD gradients across the beam ≲ a few. Lensing by the
   dispersive medium at very high OD is not modeled.
7. **Magnetic fields:** Zeeman structure is entirely absent from this doc's line-strength treatment
   (unpolarized-atom, isotropic S_FF′/3 factors). Earth-field-level splittings (~0.5 MHz) matter for
   narrow-EIT metrology; that belongs to doc 04's Hamiltonian.
```
