# 09 — Validation Corpus

**RydSim spec · GreyNOC Rydberg electrometry program · 2026-08-10**
**Status of sources: network was AVAILABLE during authoring.** Every entry marked VERIFIED was
checked today against the primary source (arXiv full text, Nature/Science abstract, Steck datasheet
PDF, or direct numerical computation). Entries marked LITERATURE-RECALL are recalled from memory
with the stated self-check; entries marked UNVERIFIED could not be confirmed and must not gate a release.

---

## 1. Scope

This document is the **complete target list** the simulator must reproduce, the analytic checks that
are independent of any experiment, the cross-module consistency requirements, the honest statement
of what this class of simulation cannot claim, and the grading scheme that turns the suite into a
credible validation report instead of a wall of green ticks.

It covers three benchmark families:

- **A-family** — analytic / exact checks (no experimental input; machine-precision or near).
- **C-family** — internal cross-module consistency (two independent methods inside RydSim must agree).
- **E-family** — published experimental results with the parameters as stated in the papers.

Species in scope: Rb-85, Rb-87, Cs-133. Everything here becomes a pytest case via the API in §5.
The benchmark IDs below (`A1`, `C4b`, `E1.3`, …) are the canonical test names.

---

## 2. Equations

All symbols SI unless noted. `h` Planck constant [J·s], `ħ = h/2π`, `ε0` vacuum permittivity,
`e` elementary charge, `a0` Bohr radius, `d` electric transition dipole moment [C·m],
`E` electric field amplitude [V/m], `Ω = dE/ħ` **angular** Rabi frequency [rad/s]
(spectroscopic splittings in Hz are `Ω/2π`; see docs/spec/00 conventions), `Γ` decay rate [rad/s],
`N` number density [m⁻³], `L` cell length [m], `λp, λc` probe/coupling vacuum wavelengths [m],
`k = 2π/λ` [m⁻¹], `Δp, Δc, ΔRF` detunings (angular) of probe, coupling, RF.

**(2.1) Autler–Townes field inversion** (resonant RF, no Doppler):

    Δf_AT = Ω_RF / 2π = d·E / h        ⇒        E = (h/d)·Δf_AT

**(2.2) Doppler-mismatch scaling of the observed AT splitting** (counter-propagating ladder EIT,
probe scanned, coupling fixed; λc < λp for Rb and Cs two-photon schemes):

    Δf_observed(probe axis) = (λc/λp) · Ω_RF/2π
    ⇒  E = (2πħ/d) · (λp/λc) · Δf_observed        [Holloway 2014, IEEE TAP 62, 6169, Eq. (12) — VERIFIED from arXiv:1405.7066 full text]

Derivation: velocity class v dominates probe absorption at Δp = k_p·v; two-photon (AT) condition
Δp + Δc − (k_p − k_c)v = ±Ω_RF/2 ⇒ Δp·(k_c/k_p) = ±Ω_RF/2 ⇒ probe-axis splitting = (k_p/k_c)·Ω_RF
= (λc/λp)·Ω_RF. For Rb (λp≈780 nm, λc≈480 nm): factor ≈ 0.6154. For Cs (852/510): ≈ 0.5986.
When the **coupling** laser is scanned with the probe locked, the splitting is unscaled to first order.

> **Documented tension (benchmark A11):** Sedlacek arXiv:1205.4461 v1 states the Doppler-averaged
> peaks are "separated by 1.625 × Ω_RF/2π" (λp/λc = 1.625) — the *inverse* of Holloway 2014 Eq. (12)
> and of the derivation above, for the same probe-scan geometry. RydSim must adjudicate this
> **numerically** with the full velocity-averaged OBE (benchmark A11). The corpus records both claims;
> the implementation must not hard-code either factor — it must emerge from the velocity average.

**(2.3) Weak-probe ladder susceptibility** (3-level; add the RF term for 4-level), per velocity class v:

    χ(Δp, v) = i·N·|d_ge|²/(ε0·ħ) · [ γ_ge − i(Δp − k_p v) + (Ωc²/4)/( γ_gr − i(Δp+Δc − (k_p−k_c)v) + (Ω_RF²/4)/( γ_gr' − i(Δp+Δc+ΔRF − (k_p−k_c±k_RF)v) ) ) ]⁻¹

with γ_ge = Γ_e/2 + γ_dephasing etc. Doppler average with Maxwell–Boltzmann weight
W(v) = (1/(√π u))·exp(−v²/u²), u = √(2 k_B T/m). RF Doppler term negligible for microwave k_RF
(Sedlacek 2012 methods — VERIFIED).

**(2.4) Beer–Lambert probe transmission** (intensity):

    T_I = exp( − k_p · L · Im[χ̄] ),   χ̄ = ∫ dv W(v) χ(Δp, v)

(Holloway 2014 writes amplitude |T| = exp(−k_p L Im χ/2); intensity is |T|². Jing 2020 writes
P(t) = P_i·exp(−k L Im[χ(t)]) — same convention as T_I. VERIFIED both.)

**(2.5) Superheterodyne transduction** (Jing 2020, arXiv:1902.11063 — VERIFIED):

    P_out(t) = P_s·cos(δ_s t + φ_s),   P_s = (α·P̄/Γ)·Ω_s = (√2·d_RF·α·P̄/(ħ·Γ))·E_s

α ≤ 1 = fraction of photons participating in EIT; Γ = 1/τ_c coherence rate ≈ Γ_EIT.
Optimal LO operating point: **Ω_L = Γ/√3**, where the AT spectrum is linear near Δ = 0 with maximum
slope **S_max = 3√3·χ0/(8Γ)** (Jing 2020 supplement — VERIFIED). These two relations are benchmarks
(C6), not free parameters.

**(2.6) Quadratic Stark / polarizability**:

    ΔE = −(1/2)·α·E²,    α = 2·Σ_k |⟨0|d̂_z|k⟩|²/(E_k − E_0)   (perturbation form)
    α = −∂²E(E_F)/∂E_F²  evaluated at E_F → 0                   (Stark-map curvature form)

Both forms implemented independently; agreement is benchmark C2. Sign/units convention:
α in Hz/(V/cm)² via `constants.polarizability_au_to_hz_per_v2_cm2`.

**(2.7) Lifetimes and BBR**:

    Γ_rad(i) = Σ_{k<i} A_ik ,    A_ik = (ω_ik³·|⟨i||d||k⟩|²)/(3π·ε0·ħ·c³·g_i)
    Γ_BBR(i) = Σ_k A_ik·n̄(ω_ik,T) + (photoionization term, may be neglected with stated error)
    n̄(ω,T) = 1/(exp(ħω/k_B T) − 1)
    1/τ_eff = Γ_rad + Γ_BBR

Beterov et al. (PRA 79, 052504, 2009; arXiv:0810.0339 — citation VERIFIED) provide fitted
`τ(0K) = τ_s·(n_eff)^δ` and analytic BBR formulas; agreement with our summed-A computation is
benchmark C1. **Fit coefficients: VERIFIED 2026-08-10 from arXiv:0810.0339 full text, Tables I & II**
(see §3.4). BBR fit form (their Eq. 14):

    Γ_BBR = (A/n_eff^D) · 2.14×10¹⁰ / ( exp[315780·B/(n_eff^C·T)] − 1 )   [s⁻¹, T in K]

**(2.8) Kramers–Kronig** (causality of the computed susceptibility):

    Re χ(ω) = (2/π)·P∫₀^∞ ω'·Im χ(ω') / (ω'² − ω²) dω'

**(2.9) Lindblad structural invariants**:

    dρ/dt = −(i/ħ)[H, ρ] + Σ_j ( L_j ρ L_j† − ½{L_j†L_j, ρ} )
    Tr ρ = 1 (all t);  ρ = ρ†;  eig(ρ) ≥ 0;  steady state: L[ρ_ss] = 0

**(2.10) Detailed balance under BBR-only coupling** (two levels i, k, ω_ik):

    ρ_kk/ρ_ii → (g_k/g_i)·exp(−ħω_ik/k_B T)   as t → ∞

**(2.11) Grading metric** for a benchmark with expected value x̂, tolerance τ (relative unless noted):

    r = |x_sim − x̂| / |x̂|;  PASS iff r ≤ τ.  Grade classes in §7 define τ per benchmark.

---

## 3. Constants / parameter tables

### 3.1 D-line anchors (Steck alkali datasheets, fetched & extracted 2026-08-10)

| Quantity | Value | Source | Confidence |
|---|---|---|---|
| Rb-87 D2 λ_vac | 780.241 209 686(13) nm | Steck, "Rubidium 87 D Line Data" (steck.us/alkalidata) | VERIFIED |
| Rb-87 D2 frequency | 384.230 484 468 5(62) THz | Steck Rb-87 | VERIFIED |
| Rb-87 D2 lifetime τ(5P3/2) | 26.2348(77) ns | Steck Rb-87 | VERIFIED |
| Rb-87 D2 Γ | 38.117(11)×10⁶ s⁻¹ = 2π·6.0666(18) MHz | Steck Rb-87 | VERIFIED |
| Rb-87 ⟨J=1/2‖er‖J′=3/2⟩ | 4.227 52(62) e·a0 = 3.584 24(52)×10⁻²⁹ C·m | Steck Rb-87 | VERIFIED |
| Rb-85 D2 λ_vac | 780.241 368 271(27) nm | Steck, "Rubidium 85 D Line Data" | VERIFIED |
| Rb-85 D2 frequency | 384.230 406 373(14) THz | Steck Rb-85 | VERIFIED |
| Rb-85 D2 lifetime / Γ | 26.2348(77) ns / 2π·6.0666(18) MHz (isotope-indistinguishable) | Steck Rb-85 | VERIFIED |
| Cs-133 D2 λ_vac | 852.347 275 82(27) nm | Steck, "Cesium D Line Data" | VERIFIED |
| Cs-133 D2 frequency | 351.725 718 50(11) THz | Steck Cs | VERIFIED |
| Cs-133 D2 lifetime τ(6P3/2) | 30.405(77) ns | Steck Cs | VERIFIED |
| Cs-133 D2 Γ | 32.889(84)×10⁶ s⁻¹ = 2π·5.234(13) MHz | Steck Cs | VERIFIED |
| Cs-133 ⟨J=1/2‖er‖J′=3/2⟩ | 4.4837(57) e·a0 = 3.8014(48)×10⁻²⁹ C·m | Steck Cs | VERIFIED |

Fundamental constants: from `scipy.constants` CODATA only (see `rydsim.constants`); never typed by hand.

### 3.2 Hydrogen exact values (computed independently today with scipy; §2 formulas)

| Quantity | Value | Source | Confidence |
|---|---|---|---|
| ⟨1s|r|2p⟩ radial matrix element | 128√6/243 a0 = 1.290 266 202 0 a0 | exact analytic; reproduced numerically to 4×10⁻¹⁶ rel. | VERIFIED |
| Oscillator strength f(1s→2p) | 0.416 197 | computed from exact matrix element | VERIFIED |
| A(2p→1s) | 6.268 3×10⁸ s⁻¹ ⇒ τ(2p) = 1.5953 ns | computed (CODATA constants) | VERIFIED |
| ⟨r⟩ₙℓ | (3n² − ℓ(ℓ+1))/2 a0; e.g. n=50, ℓ=2 → 3747 a0 exactly | exact analytic | VERIFIED |
| TRK sum rule | Σ_k f(1s→k) = 1 exactly (discrete + continuum) | exact | VERIFIED |
| Discrete part of TRK sum (n ≤ ∞ bound only) | ≈ 0.5650 (continuum carries ≈ 0.4350) | Bethe & Salpeter, standard | LITERATURE-RECALL (self-check: 1 − computed continuum integral) |

### 3.3 Quantum-defect sources (values live in docs/spec/01; corpus needs only the citations + spot checks)

| Item | Source | Confidence |
|---|---|---|
| Rb nS/nP/nD quantum defects | Li, Mourachko, Noel, Gallagher, PRA 67, 052502 (2003); Mack et al. PRA 83, 052515 (2011) | citation VERIFIED; digit values LITERATURE-RECALL — self-check = benchmarks C4a–C4e below |
| Cs quantum defects | Weber & Sansonetti, PRA 35, 4650 (1987); Goy et al. PRA 26, 2733 (1982) | citation LITERATURE-RECALL; self-check = C4b |
| δ→0 hydrogenic limit | exact | VERIFIED (analytic) |

The five RF transition frequencies in §3.5 are the corpus's *external* test of whichever defect
values spec 01 adopts: if the defects are wrong at the 10⁻⁴ level, the predicted mm-wave/microwave
intervals miss by many MHz and C4 fails. This is deliberate: the corpus does not need to restate the
defect tables to test them.

### 3.4 Lifetime-fit reference (Beterov 2009) — **extracted from arXiv:0810.0339 full text 2026-08-10**

Radiative-lifetime fit `τ0 = τ_s·n_eff^δ` [ns] (their Table II, Rb and Cs rows, verbatim):

| Series | Rb τ_s [ns] | Rb δ | Cs τ_s [ns] | Cs δ | Confidence |
|---|---|---|---|---|---|
| nS1/2 | 1.368 | 3.0008 | 1.2926 | 3.0005 | VERIFIED |
| nP1/2 | 2.4360 | 2.9989 | 2.9921 | 2.9892 | VERIFIED |
| nP3/2 | 2.2214 | 3.0026 | 3.2849 | 2.9875 | VERIFIED |
| nD3/2 | 1.0761 | 2.9898 | 0.6580 | 2.9944 | VERIFIED |
| nD5/2 | 1.0687 | 2.9897 | 0.6681 | 2.9941 | VERIFIED |

BBR-fit coefficients for Eq. (14) (their Table I, verbatim; rows are S1/2; P1/2 / P3/2; D3/2 / D5/2):

| Series | Rb A | Rb B | Rb C | Rb D | Cs A | Cs B | Cs C | Cs D | Confidence |
|---|---|---|---|---|---|---|---|---|---|
| nS1/2 | 0.134 | 0.251 | 2.567 | 4.426 | 0.123 | 0.231 | 2.517 | 4.375 | VERIFIED |
| nP1/2 | 0.053 | 0.128 | 2.183 | 3.989 | 0.041 | 0.072 | 1.693 | 3.607 | VERIFIED |
| nP3/2 | 0.046 | 0.109 | 2.085 | 3.901 | 0.038 | 0.056 | 1.552 | 3.505 | VERIFIED |
| nD3/2 | 0.033 | 0.084 | 1.912 | 3.716 | 0.038 | 0.076 | 1.790 | 3.656 | VERIFIED |
| nD5/2 | 0.032 | 0.082 | 1.898 | 3.703 | 0.036 | 0.073 | 1.770 | 3.636 | VERIFIED |

Paper: PRA 79, 052504 (2009) + erratum PRA 80, 059902 (arXiv:0810.0339 — citation and tables VERIFIED).
Beterov's own accuracy statement (verified from full text): the combined τ_eff formula (their Eq. 16)
reproduces their numerical calculations to **better than 5% for 15 < n < 80** — this is the basis of
the C1b tolerance. Note both fits use `n_eff = n − δ_ℓj`, output in ns.

### 3.5 Experimental corpus — parameters as stated in the papers (all fetched 2026-08-10)

**E1 — Sedlacek et al. 2012** (Nature Physics 8, 819; params from arXiv:1205.4461 v1 full text — VERIFIED):

| Parameter | Value |
|---|---|
| Species / states | Rb-87 (probe locked 5S1/2 F=2 → 5P3/2 F=3); EIT to 53D5/2; RF couples 53D5/2 → 54P3/2 |
| RF frequency | measured 14.233 GHz; calculated from quantum defects 14.232 GHz |
| RF dipole (their 4-level model, stretched hyperfine states) | μ_RF = 1.37×10⁻²⁶ C·m (≈1616 e·a0) |
| Lasers | probe ≈780 nm, Ω_p = 2π·6 MHz, beam 750 μm; coupling ≈480 nm, Ω_c = 2π·2 MHz, beam 100 μm |
| Cell | 10 cm Rb vapor cell, room temperature; effective interaction length 7.5 cm |
| Model decays | Γ_s = 2π·6.1 MHz (5P3/2); transit dephasing Γ_tt = 2π·270 kHz on the RF transition; Rydberg decay/BBR/collisions ~2π·1 kHz each (neglected) |
| EIT linewidth | 4–5 MHz (laser-limited; effective laser linewidth ≈700 kHz) |
| Reported results | sensitivity ≈30 μV·cm⁻¹·Hz⁻¹ᐟ²; smallest detected field 8.33 μV/cm (SNR≈1, 1 s integration); field via AT known to 0.5%; agreement with horn-antenna calculation 10%; probe transmission increase up to ≈4.5% peaking at 715 μV/cm; claimed Doppler-averaged peak separation factor 1.625·Ω_RF/2π (see §2.2 tension) |

**E2 — Holloway et al. 2014** (IEEE TAP 62, 6169; params from arXiv:1405.7066 full text — VERIFIED):

| Parameter | Value |
|---|---|
| Species / cell | Rb-85; cylindrical glass cell 75 mm long × 25 mm diameter, room temperature |
| Probe | 780 nm, ~100–175 nW, FWHM 80 μm |
| Measured RF points | 15.59, 17.04, 18.65, 68.64, 104.77 GHz |
| 17.04 GHz config | coupling ≈480.13 nm to 50D5/2; RF couples 50D5/2 → 51P3/2 |
| 104.77 GHz config | coupling ≈482.63 nm to 28D5/2; RF couples 28D5/2 → 29P3/2 |
| Broadband concept | blue 479.32 nm → 2.03 GHz; blue 483.60 nm → 150.40 GHz; span 1–500 GHz |
| Field inversion | E = (2πħ/℘_RF)·(λp/λc)·Δf (probe-scan Doppler scaling λc/λp; 5P3/2 hyperfine splittings scale as 1 − λc/λp) |

**E3 — Jing et al. 2020 superheterodyne** (Nature Physics 16, 911; params from arXiv:1902.11063 full text — VERIFIED):

| Parameter | Value |
|---|---|
| Species / states | Cs-133: 6S1/2 F=4 → 6P3/2 F=5 (probe 852 nm) → 47D5/2 (coupling 510 nm); LO on 47D5/2 → 48P3/2 |
| LO frequency | 6.94 GHz (as printed; more digits UNVERIFIED) |
| Cell | 5 cm, room temperature, ground-state density N0 = 4.89×10¹⁰ cm⁻³ |
| Decays used | γ2 = 5.2 MHz (6P3/2, FWHM units); γ3 = 3.9 kHz (47D5/2), γ4 = 1.7 kHz (48P3/2), incl. BBR to n = 70 |
| Probe | 120±4 μW, 1/e² diameter 1.70±0.04 mm ⇒ Ω_p = 5.7±0.6 MHz |
| Coupling | 34±1 mW, 1/e² diameter 2.00±0.05 mm ⇒ Ω_c = 0.97±0.12 MHz |
| Polarization | both linear ∥ MW field ⇒ |mJ| = 1/2 excitation |
| IF | δ_s = 150.000 kHz |
| Reported results | sensitivity 55 nV·cm⁻¹·Hz⁻¹ᐟ²; minimum detectable field 780 pV/cm (Nature abstract — VERIFIED); QPNL projection ≈700 pV·cm⁻¹·Hz⁻¹ᐟ² (arXiv v1); SNR 44 dB at E_s = 7.8 μV/cm with 1.04 Hz ENBW; optimal Ω_L = Γ/√3, S_max = 3√3χ0/(8Γ) |

**E4 — Tu et al. 2024 SQL** (Sci. Adv. 10, eads0683; params from PMC11661427 — VERIFIED).
**Cold-atom experiment — NOT a vapor cell.** RydSim's thermal-vapor OBE does not simulate it;
only the SQL/noise-budget formulas are benchmarked (see §7 grading and §8 limitations).

| Parameter | Value |
|---|---|
| Species | Rb-87, laser-cooled (2D-MOT), N = 5.2×10⁵ atoms, 200 μK |
| States | ground 5S1/2 F=2 mF=2; RF couples 39D5/2 ↔ 40P3/2 at 36.9 GHz; dipole 1218 e·a0 |
| Lasers | probe 780 nm Ω_p ≈ 2π·4.6–4.7 MHz (5–7.6 μW); coupling 481 nm Ω_c ≈ 2π·6.1 MHz |
| LO | Ω_L = 2π·2.0 MHz |
| Results | sensitivity 10.0 nV·cm⁻¹·Hz⁻¹ᐟ²; SQL 3.7 nV·cm⁻¹·Hz⁻¹ᐟ² (factor 2.6 above); T_eq ≈ 830 K; 3-dB bandwidth 2.3 MHz; rep rate 100 Hz, detection window 2.7 ms; minimum-field run T′ = 420 s |

**E5 — Meyer et al. 2020 wideband assessment** (J. Phys. B 53, 034001; from arXiv:1910.00646 full text — VERIFIED):

| Parameter | Value |
|---|---|
| Scope | sensitivity model 1 kHz–1 THz, analytic + semiclassical Floquet; experimentally validated 1–20 GHz |
| Key numeric anchor | scalar polarizability of Rb |100D5/2, mJ=1/2⟩, low-frequency limit: **−8.6 GHz/(V/cm)²** full numeric sum vs **−45.4 GHz/(V/cm)²** nearest-state-only two-level estimate (documented 5.3× pitfall) |
| Structural claims | SNR ∝ E^β, β ∈ [1,2] by regime; minimum field |100P3/2⟩→|101S1/2⟩ is 1.3× larger than |100D5/2⟩→|101P3/2⟩; P/S series beat D at far-detuned low frequency by ~3× and ~1.5×; passive-dipole diode readout floor ~1 (V/m)/√Hz |

**E6 — Multi-dress-state IBW** (arXiv:2506.10541 v6, June 2026 — VERIFIED full text):

| Parameter | Value |
|---|---|
| Species / cell | thermal Rb-87 vapor, 21 °C |
| States | |g⟩=5S1/2 F=2; |e⟩=5P3/2 F′=3; |r1⟩=51D5/2 mJ=1/2; |r2⟩=52P3/2 mJ=1/2 |
| Lasers | probe 780 nm waist 52.5 μm, Ω_p = 2π·17.16 MHz; coupling 480 nm waist 29 μm, Ω_c = 2π·83.32 MHz; Δp = ΔL = 0 |
| LO | 16.03 GHz |
| Operating point "peak A" | Ω_L = 2π·16.66 MHz, Δc = −2π·16 MHz ⇒ **IBW 54.6 MHz at 140.4 nV·cm⁻¹·Hz⁻¹ᐟ²** |
| Record point | IBW 76.8 MHz at 222.6 nV·cm⁻¹·Hz⁻¹ᐟ² |
| Other points | max IBW 82 MHz (Δc=+2π·6 MHz, Ω_L=2π·8.36 MHz) at degraded 1.25 μV·cm⁻¹·Hz⁻¹ᐟ²; best sensitivity 81.1 nV·cm⁻¹·Hz⁻¹ᐟ² |

Note: the task brief and Study Report quote "54.6 MHz / 140.4 nV" — that is the *peak-A* operating
point, not the paper's headline record. Both are in the corpus.

**E7 — Zeeman continuous coverage** (Communications Physics 2026, s42005-026-02529-3 — VERIFIED abstract+methods):

| Parameter | Value |
|---|---|
| Species / transitions | Cs; 45D5/2→46P3/2, 46D5/2→44F7/2, 49S1/2→49P3/2 |
| Tuning law | Δf = (μ_B·B/h)·(g_J4·m_J4 − g_J3·m_J3) |
| Coverage | 1–40 GHz continuous (1 GHz steps demonstrated); B = 0–412 G; 1.17 GHz tuning around one resonance at 60 G |
| Sensitivity | ≤65 nV·cm⁻¹·Hz⁻¹ᐟ² across band; best <20 nV·cm⁻¹·Hz⁻¹ᐟ² near 8 and 34 GHz |
| Cell | φ10 mm × 50 mm; B-field uniformity <3% over cell |

**E8 — Sapphire-cell kHz sensing** (npj Quantum Materials 2026, s41535-026-00862-y — VERIFIED abstract+methods):

| Parameter | Value |
|---|---|
| Species / states | Cs; 6S1/2 F=4 → 6P3/2 F=5 → 52D5/2; coupling 509 nm |
| Mechanism | coupling-laser photoelectric charging of sapphire wall → laser-induced DC bias field → converts quadratic Stark to linear (self-dressing) |
| Results | 13.5 nV·cm⁻¹·Hz⁻¹ᐟ² at 100 kHz; operation down to 10 kHz; sapphire's higher resistivity suppresses alkali-adsorption screening vs borosilicate |

**E9 — NIST AT-splitting traceability** (Holloway, Simons, Gordon, Dienstfrey, Anderson, Raithel,
J. Appl. Phys. 121, 233106 (2017), "Electric field metrology for SI traceability: Systematic
measurement uncertainties in electromagnetically induced transparency in atomic vapor" — citation
and central claim VERIFIED 2026-08-10 from the AIP abstract: *the linear Ω_RF ↔ AT-splitting
relation holds with minimal error as long as the EIT linewidth is small compared to the AT
splitting*, and the paper quantifies the systematic deviations outside that regime. That qualitative
regime boundary is benchmark E9.1. The paper's specific percentage-deviation tables are paywalled
and were NOT extracted: **numeric budget UNVERIFIED** — implementation task: obtain and add as
E9.2+ before release; the placeholder must not be filled from memory.)

### 3.6 Derived Doppler factors (computed from §3.1 wavelengths; nominal coupling λ)

| System | λc/λp (probe-scan AT compression) | Source | Confidence |
|---|---|---|---|
| Rb 780/480 | 0.6152 (use exact k-ratio from RydSim's own level energies at runtime) | computed | VERIFIED (formula per Holloway 2014) |
| Cs 852/510 | 0.5984 | computed | VERIFIED (same) |

---

## 4. Numerical method + pitfalls (benchmark execution rules)

1. **Determinism.** Every benchmark runs from a frozen config (dataclass, hashed); the report embeds
   the config hash, RydSim version, and scipy CODATA version. No random seeds anywhere in the corpus.
2. **Velocity averaging.** Gauss–Hermite quadrature in v, ≥ 80 nodes (convergence criterion: doubling
   nodes changes T(Δp) by < 10⁻⁶ absolute). Uniform grids need ±4σ_v span and ≥ 401 points — cheaper
   to use Gauss–Hermite. Pitfall: under-resolved velocity grids fake extra EIT linewidth and shift the
   apparent AT scaling factor — exactly the quantity A11 adjudicates.
3. **Probe-scan resolution.** Δp grid step ≤ Γ_EIT/20 across ±(3·Ω_RF + 5Γ). AT peak positions by
   local parabolic fit, never argmax on the raw grid.
4. **Steady state.** Solve L[ρ_ss]=0 by null-space (LU on the vectorized Liouvillian with the trace
   row replaced); cross-check by RK45 integration to t = 50/Γ_slowest. Both must agree to 10⁻⁸ in
   every ρ element (this *is* benchmark A8c). Pitfall: with kHz-scale Rydberg decays the Liouvillian
   is stiff — direct integration alone silently under-converges; the null-space solve is authoritative.
5. **Positivity/trace tolerance.** |Tr ρ − 1| ≤ 10⁻¹⁰; min eigenvalue ≥ −10⁻¹⁰ (allow −10⁻¹⁰ for
   floating-point; anything worse fails A8).
6. **Kramers–Kronig check.** Compute Im χ on a grid spanning ≥ ±100 Γ_D (Doppler width), apply the
   subtractive KK transform (subtract the asymptote before the principal-value integral). Pitfall:
   truncation of the wings dominates the error; grade on the central ±5Γ region with L2 relative
   error ≤ 1%.
7. **BBR sums.** Sum A·n̄ over Δn until the partial sum changes < 0.1%; include both up and down
   transitions; T = 300 K unless the paper states otherwise. Pitfall: truncating at Δn = 5 biases
   nD lifetimes by percent-level; Jing's γ3, γ4 assumed BBR to n = 70.
8. **Stark maps.** Basis n ± 5, all ℓ (or ℓ ≤ ℓ_max with stated convergence), |mJ| fixed. Curvature
   fit for α at fields ≤ 10% of the first avoided crossing. Pitfall (Meyer 2020, verified): the
   nearest-state two-level estimate for 100D5/2 is wrong by 5.3× — never validate α against a
   two-level shortcut.
9. **Dipole-moment conventions.** All corpus dipoles state their convention explicitly: reduced
   ⟨J‖er‖J′⟩ (Steck), vs spherical component ⟨n l j mj|d_q|…⟩, vs the papers' effective 2-level d.
   Benchmarks C5/E-family compare like with like; a silent √(2J′+1) or angular-factor slip is a
   factor ~2 error that AT round-trips (C3) will catch.
10. **Sensitivity benchmarks.** RydSim computes the *photon-shot-noise-limited* NEF for the stated
    optical powers plus the atom-projection floor. Technical noise (laser linewidth beyond stated,
    electronics) is NOT modeled unless the paper quantifies it: therefore simulated NEF must satisfy
    NEF_sim ≤ NEF_published, and match only within the grades of §7. A simulated sensitivity *better*
    than SQL for the stated atom number is an automatic FAIL (unphysical).

---

## 5. Recommended Python API

```python
# rydsim/validation/corpus.py   (numpy-vectorized; no ARC, no qutip)
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Sequence

class Confidence(Enum):
    VERIFIED = "VERIFIED"                  # checked against primary source during spec authoring
    LITERATURE_RECALL = "LITERATURE-RECALL"
    UNVERIFIED = "UNVERIFIED"              # never gates a release

class GradeClass(Enum):
    EXACT = "exact"          # tol_rel <= 1e-8 (analytic identities)
    TIGHT = "tight"          # tol_rel <= 0.05
    MODERATE = "moderate"    # tol_rel <= 0.25
    ORDER = "order"          # within factor 3 (log-space check)
    QUALITATIVE = "qual"     # structural predicate (monotonicity, bound, shape)

@dataclass(frozen=True)
class Benchmark:
    id: str                      # e.g. "C4a"; becomes the pytest id
    family: str                  # "analytic" | "consistency" | "experimental"
    quantity: str                # human-readable, with units
    expected: float | None       # None for QUALITATIVE predicates
    unit: str
    grade_class: GradeClass
    tol: float                   # relative unless grade_class == ORDER (then log10 factor)
    source: str                  # citation string, exactly as in this spec
    confidence: Confidence
    compute: Callable[[], float | bool]   # pure function; pulls everything from frozen configs
    notes: str = ""

@dataclass(frozen=True)
class BenchmarkResult:
    benchmark: Benchmark
    value: float | bool
    rel_error: float | None
    passed: bool
    grade_awarded: str           # "EXACT PASS", "TIGHT PASS", "MODERATE PASS", "FAIL", ...

@dataclass(frozen=True)
class CorpusReport:
    results: tuple[BenchmarkResult, ...]
    config_hash: str
    rydsim_version: str
    codata_source: str
    def summary(self) -> dict[str, int]: ...          # counts per grade, per family
    def worst(self, k: int = 10) -> list[BenchmarkResult]: ...

def build_registry() -> tuple[Benchmark, ...]:
    """Assemble A-, C-, E-family benchmarks from this spec. IDs are stable API."""

def grade(value: float, bench: Benchmark) -> BenchmarkResult:
    """Apply Sec.2.11 metric; ORDER graded as |log10(value/expected)| <= log10(3)."""

def run_corpus(registry: Sequence[Benchmark] | None = None,
               families: Sequence[str] | None = None) -> CorpusReport:
    """Run all (or filtered) benchmarks; never raises on FAIL; raises on compute() exception."""

def report_markdown(report: CorpusReport) -> str:
    """Emit the honest validation report: table sorted worst-first, confidence tags inline,
    UNVERIFIED/LITERATURE-RECALL sources surfaced in a dedicated caveats section, and the
    cannot-claim statement of Sec.8 reproduced verbatim at the bottom. No green-ticks-only view."""
```

pytest binding: `tests/test_corpus.py` parametrizes over `build_registry()`;
`@pytest.mark.xfail(strict=False)` is allowed **only** for benchmarks whose confidence is
UNVERIFIED; VERIFIED benchmarks failing = red build.

---

## 6. Validation benchmarks

Tolerance semantics per §2.11/§5. "ORDER" = within factor 3 unless stated.

### A-family — analytic / exact

| ID | Quantity | Expected | Tolerance | Source | Confidence |
|---|---|---|---|---|---|
| A1 | Hydrogen radial ⟨1s|r|2p⟩ via RydSim Numerov (Z=1, δ=0) | 1.290 266 202 a0 (=128√6/243) | 1e-6 rel | exact analytic (recomputed 2026-08-10) | VERIFIED |
| A2 | Hydrogen f(1s→2p) | 0.416 20 | 1e-4 rel | exact analytic (recomputed) | VERIFIED |
| A3 | Hydrogen τ(2p) from RydSim A-coefficient pipeline | 1.5953 ns | 0.2% rel | computed from exact f + CODATA | VERIFIED |
| A4a | TRK sum rule, discrete Σf(1s→np), n≤30 | ≈0.5646 (approaches 0.5650 discrete total) | 0.5% rel | Bethe & Salpeter | LITERATURE-RECALL (self-check: continuum integral must supply 1−Σ) |
| A4b | TRK total incl. continuum | 1.000 | 1e-3 rel | exact | VERIFIED |
| A5 | ⟨r⟩(n=50, ℓ=2, hydrogen) | 3747.0 a0 | 1e-6 rel | exact analytic | VERIFIED |
| A6 | Alkali energy module with all δℓ forced to 0 vs −R_M/n² | 0 difference | 1e-10 rel | exact | VERIFIED |
| A7 | Weak-probe analytic χ (2.3) vs Lindblad steady-state ρ_ge, Ωp/Γ = 10⁻³, 3-level, v=0 | identical | 1e-6 rel | internal (both from first principles) | VERIFIED |
| A8a | Tr ρ conservation over 50/Γ evolution | 1 | 1e-10 abs | Lindblad structure | VERIFIED |
| A8b | Min eigenvalue of ρ(t), all t | ≥ 0 | −1e-10 abs floor | Lindblad structure | VERIFIED |
| A8c | Null-space steady state vs long-time RK45 | identical | 1e-8 abs per element | internal | VERIFIED |
| A9 | Kramers–Kronig: KK[Im χ] vs computed Re χ, central ±5Γ | identical | 1% L2 rel | causality, §2.8 | VERIFIED |
| A10 | BBR-only two-level detailed balance ρ_kk/ρ_ii | (g_k/g_i)·exp(−ħω/k_BT) | 0.1% rel | §2.10 | VERIFIED |
| A11 | Doppler probe-scan AT splitting ratio (full velocity-averaged OBE, Rb 780/480, Ω_RF = 2π·10 MHz) | (λc/λp) = 0.6152 | 5% rel | Holloway 2014 Eq.(12) derivation §2.2; **adjudicates Sedlacek v1 prose (1.625)** | VERIFIED (formula); the tension itself is documented |
| A12 | Same, coupling-scan configuration | 1.000 (unscaled) | 5% rel | §2.2 derivation | VERIFIED (derivation; no primary-source printout — treat failure as physics finding, not code bug, and investigate) |

### C-family — internal cross-module consistency

| ID | Quantity | Expected | Tolerance | Source | Confidence |
|---|---|---|---|---|---|
| C1a | τ(Rb 30S1/2, 0 K): summed-A vs Beterov fit (τ_s=1.368 ns, δ=3.0008, §3.4) | agreement | 5% rel | Beterov PRA 79, 052504, Table II | VERIFIED (coeffs extracted from full text) |
| C1b | τ_eff(Rb 30S–70S, 300 K) incl. BBR vs Beterov Eq.(16) (§3.4 coeffs) | agreement | 10% rel (their Eq.16 is itself ≤5% vs their numerics, 15<n<80) | Beterov 2009 Tables I+II | VERIFIED (coeffs) |
| C1c | τ_eff(Cs 47D5/2, 300 K) vs Beterov Eq.(16) — anchors Jing's γ3 scale | agreement | 10% rel | Beterov 2009 | VERIFIED (coeffs) |
| C2 | α(Rb nD5/2, n=30..70): perturbation sum vs Stark-map curvature | agreement | 1% rel | internal, §2.6 | VERIFIED (method) |
| C3 | AT round-trip: E_in → Doppler OBE → Δf_obs → E_out via §2.2 | E_out = E_in | 2% rel for Ω_RF ∈ [3Γ_EIT, 0.1·level spacing] | internal | VERIFIED (method) |
| C4a | ν(Rb 53D5/2→54P3/2) from quantum-defect module | 14.233 GHz | ±5 MHz | Sedlacek 2012 (meas.; their calc 14.232) | VERIFIED |
| C4b | ν(Cs 47D5/2→48P3/2) | 6.94 GHz | ±10 MHz | Jing 2020 | VERIFIED (3 s.f.) |
| C4c | ν(Rb-85 50D5/2→51P3/2) | 17.04 GHz | ±10 MHz | Holloway 2014 | VERIFIED |
| C4d | ν(Rb-85 28D5/2→29P3/2) | 104.77 GHz | ±50 MHz | Holloway 2014 | VERIFIED |
| C4e | ν(Rb-87 39D5/2→40P3/2) | 36.9 GHz | ±100 MHz | Tu 2024 (Sci. Adv. eads0683) | VERIFIED (3 s.f.) |
| C4f | ν(Rb-87 51D5/2→52P3/2) | 16.03 GHz | ±20 MHz | arXiv:2506.10541 | VERIFIED |
| C5a | ⟨5S1/2‖er‖5P3/2⟩ Rb from Numerov+model potential | 4.2275 e·a0 | 2% rel | Steck Rb-87 | VERIFIED |
| C5b | ⟨6S1/2‖er‖6P3/2⟩ Cs | 4.4837 e·a0 | 2% rel | Steck Cs | VERIFIED |
| C5c | Γ(Rb D2) from C5a pipeline | 2π·6.0666 MHz | 4% rel | Steck Rb-87 | VERIFIED |
| C5d | Γ(Cs D2) from C5b pipeline | 2π·5.234 MHz | 4% rel | Steck Cs | VERIFIED |
| C5e | d(Rb 53D5/2→54P3/2), Sedlacek convention (stretched-state 4-level) | 1.37×10⁻²⁶ C·m | 5% rel | Sedlacek 2012 | **DOCUMENTED TENSION — not a passing benchmark.** Printed *number* VERIFIED (arXiv:1205.4461 v1); its *convention* **UNVERIFIED** (§9 register). Reproduced by **no** published convention; see the blockquote below. Expected value and tolerance unchanged (§7 rule 5). Never gates a release (§8.10). |
| C5f | d(Rb 39D5/2→40P3/2) | 1218 e·a0 | 5% rel | Tu 2024 | VERIFIED (number; convention as in paper) |
| C6a | Superhet optimal LO: argmax slope at Ω_L | Γ_EIT/√3 | 5% rel | Jing 2020 suppl. | VERIFIED |
| C6b | Superhet max slope | 3√3·χ0/(8Γ) | 5% rel | Jing 2020 suppl. | VERIFIED |
| C7 | Doppler FWHM of simulated Rb D2 absorption at 300 K vs analytic Gaussian √(8ln2·k_BT/m)·ν0/c | ≈0.51 GHz | 1% rel | kinetic theory | VERIFIED (analytic) |
| C8 | Cs vapor density at 25 °C from RydSim vapor-pressure model | 4.89×10¹⁰ cm⁻³ (Jing's stated room-T value) | factor 1.5 (T uncertainty ±3 °C) | Jing 2020 + vapor-pressure fit (spec 05) | VERIFIED (target number) |
| C9 | Zeeman tuning rate (Cs 45D5/2→46P3/2, stretched mJ) | (μ_B/h)·(g_J4 m_J4 − g_J3 m_J3), μ_B/h = 1.39962 MHz/G | 1% rel | Comms. Phys. 2026 formula + CODATA | VERIFIED |

> **Documented tension (benchmark C5e):** Sedlacek's printed effective RF dipole for
> Rb 53D₅/₂→54P₃/₂, **1.37×10⁻²⁶ C·m**, is reproduced by **neither** published convention in
> `rydsim.dipoles.MU_RF_CONVENTIONS` (a closed set: `stretched`, `nist_pi`), evaluated on the
> spec-02 consensus radial **R = 3622.78 a₀** (three methods agreeing to 6×10⁻⁶):
> `stretched` — the paper's *own stated reading* ("4-level model, stretched hyperfine states") —
> gives **1.9426×10⁻²⁶ C·m (+41.8 %)**, and `nist_pi` (spec 00 lock #11, the normative default)
> gives **1.5047×10⁻²⁶ C·m (+9.8 %)**. Both miss C5e's 5 % tolerance *and* audit R5's 2 %.
> Per **audit R5 that flags the FIXTURE, not the code**: the invented `pi_manifold_rms` averaging
> rule that had previously been used to force agreement is **removed from the code**, and no
> convention is ever added to make a benchmark agree. (§4.9's "compare like with like" rule is what
> this benchmark exercises; here the like-for-like comparison is the thing that fails.)
>
> The **printed number stays VERIFIED** (arXiv:1205.4461 v1 full text); **its convention becomes
> UNVERIFIED** (§9 register — same amplitude-vs-RMS artifact class as ruling R-22's Jing √2).
> The residual under the paper's own convention is a clean **√2**: computed/printed = 1.41796,
> i.e. √2 to **0.26 %** — and the radial layer is not the suspect, since three methods agree to
> better than 10⁻⁴ and no radial error of size √2 is available. Independent, **code-free**
> corroboration: Tu 2024 print 1218 e·a₀ for the *same* D₅/₂→P₃/₂ angular channel
> (39D₅/₂→40P₃/₂) under an explicitly stretched σ⁺ ladder, so the two printed dipoles must scale
> as the radial ME alone; published Li-2003 / Mack-2011 defects give
> ν(53D)ν(54P)/ν(39D)ν(40P) = **1.8859** while 1615.88/1218 = **1.3267** — short by **1.4215**,
> i.e. √2 to **0.5 %**.
>
> **The expected value (1.37×10⁻²⁶ C·m) and the 5 % tolerance are NOT changed** — C5e is recorded
> as a tension, not re-toleranced (§7 rule 5; audit §3 item 38). C5e must not be reported as a
> passing TIGHT benchmark, and being UNVERIFIED-convention it can never gate a release (§8.10).

The normative statement is **shipped and test-bound** as `rydsim.dipoles.C5E_CONVENTION_TENSION`
(every digit regenerated from a live run by
`tests/test_dipoles.py::test_c5e_tension_note_digits_track_live_computation`, so this provenance
string cannot go stale), verbatim:

```text
spec 09 C5e (Sedlacek 2012, Rb 53D5/2->54P3/2, printed mu_RF = 1.37e-26 C*m = 1615.88 e*a0) is NOT reproduced by either published convention on the spec-02 consensus radial R = 3622.78 a0 (three methods agreeing to 6e-6): stretched (the paper's own stated reading) = 2291.2 e*a0 (+41.8 %), nist_pi (lock #11) = 1774.8 e*a0 (+9.8 %). The stretched residual is a factor sqrt(2): computed/printed = 1.41796, i.e. sqrt(2) to 0.26 % (ruling R-22 amplitude/RMS artifact). Code-independent corroboration (no RydSim needed): Tu 2024 print 1218 e*a0 for the SAME D5/2->P3/2 angular channel at 39D5/2->40P3/2 under an explicitly stretched sigma+ ladder, so the two printed dipoles must scale as the radial ME alone; published Li-2003 / Mack-2011 quantum defects give nu(53D)nu(54P)/nu(39D)nu(40P) = 1.8859, but 1615.88/1218 = 1.3267 — short by 1.4215, i.e. sqrt(2) to 0.5 %. FIXTURE FLAGGED per audit R5; number VERIFIED (v1 full text), its convention UNVERIFIED (spec 09 SS9 register / SS7 rule 5)
```

### E-family — experimental reproduction

| ID | Quantity | Expected | Tolerance | Source | Confidence |
|---|---|---|---|---|---|
| E1.1 | Sedlacek EIT FWHM at stated Ω_p, Ω_c, +700 kHz laser width | 4–5 MHz | within ×1.5 of 4.5 MHz | Sedlacek 2012 | VERIFIED |
| E1.2 | Probe-transmission enhancement curve: maximum at E_RF | 715 μV/cm | ±20% position; amplitude (≈+4.5%) within ×2 | Sedlacek 2012 Fig.3a | VERIFIED |
| E1.3 | Shot-noise NEF at Sedlacek params | ≤ 30 μV·cm⁻¹·Hz⁻¹ᐟ² (they were laser-noise-limited) | one-sided bound + ORDER | Sedlacek 2012 | VERIFIED |
| E2.1 | AT-derived field vs horn-antenna calculation discrepancy envelope | ≤ 10% | qualitative (sim systematic budget must fit inside 10%) | Sedlacek 2012 / Holloway 2014 | VERIFIED |
| E3.1 | Jing superhet NEF at stated params (photon-shot-noise model) | 55 nV·cm⁻¹·Hz⁻¹ᐟ² | MODERATE ×3 (one-sided: sim ≤ measured requires FAIL review) | Jing 2020 | VERIFIED |
| E3.2 | Jing linear dynamic range: P_s ∝ E_s over stated span; slope max at Ω_L = Γ/√3 | linear, Eq. (3) | QUALITATIVE (R² ≥ 0.99 over their linear span) | Jing 2020 Fig.3 | VERIFIED |
| E3.3 | Jing QPNL projection with N from their cell/beam geometry | ≈700 pV·cm⁻¹·Hz⁻¹ᐟ² | ORDER | Jing 2020 (arXiv v1) | VERIFIED |
| E4.1 | SQL formula with N = 5.2×10⁵, their T2/window | 3.7 nV·cm⁻¹·Hz⁻¹ᐟ² | 30% rel | Tu 2024 | VERIFIED — formula-level only; cold-atom experiment out of thermal-solver scope |
| E5.1 | α(Rb 100D5/2, mJ=1/2) low-freq, full sum | −8.6 GHz/(V/cm)² | 15% rel | Meyer 2020 | VERIFIED |
| E5.2 | Two-level nearest-state α estimate (documented pitfall ratio) | −45.4 GHz/(V/cm)², ratio 5.3× | 20% on ratio | Meyer 2020 | VERIFIED |
| E5.3 | Min-field ratio |100P3/2→101S1/2⟩ / |100D5/2→101P3/2⟩ | 1.3 | 15% rel | Meyer 2020 | VERIFIED |
| E6.1 | Multi-dress peak-A IBW at Ω_L=2π·16.66 MHz, Δc=−2π·16 MHz | 54.6 MHz | within ×1.5 | arXiv:2506.10541 | VERIFIED |
| E6.2 | Multi-dress peak-A sensitivity | 140.4 nV·cm⁻¹·Hz⁻¹ᐟ² | ORDER (×3) | arXiv:2506.10541 | VERIFIED |
| E6.3 | Record point IBW/sens | 76.8 MHz / 222.6 nV·cm⁻¹·Hz⁻¹ᐟ² | ×1.5 / ORDER | arXiv:2506.10541 | VERIFIED |
| E6.4 | Conventional superhet IBW collapses as sensitivity optimized (trade-law shape) | monotone trade | QUALITATIVE | arXiv:2506.10541 + Jing 2020 | VERIFIED |
| E7.1 | Zeeman-tuned coverage: resonance shift 1.17 GHz at 60 G (stated transition) | 1.17 GHz | 10% rel | Comms. Phys. 2026 | VERIFIED |
| E7.2 | Band-wide NEF ≤65 nV·cm⁻¹·Hz⁻¹ᐟ² (1–40 GHz), best <20 near 8 & 34 GHz | envelope | QUALITATIVE (sim envelope shape + one-sided) | Comms. Phys. 2026 | VERIFIED |
| E8.1 | Self-dressed linear Stark readout NEF at 100 kHz | 13.5 nV·cm⁻¹·Hz⁻¹ᐟ² | ORDER; screening model is phenomenological | npj QM 2026 | VERIFIED (number); mechanism model UNVERIFIED at first-principles level |
| E9.1 | AT↔E linearity regime: simulated E_out/E_in → 1 for Δf_AT ≫ Γ_EIT and departs measurably as Δf_AT → Γ_EIT | linear regime + breakdown onset | QUALITATIVE (predicate: relative inversion error <2% for Δf_AT > 5·Γ_EIT, >5% for Δf_AT < Γ_EIT) | Holloway JAP 121, 233106 (2017), central claim per AIP abstract | VERIFIED (claim) |
| E9.2+ | NIST JAP 2017 numeric systematic-uncertainty budget | — | — | Holloway JAP 121, 233106 | UNVERIFIED numerics — fetch before release; placeholder MUST NOT be filled from memory |

---

## 7. Grading scheme (how the report stays honest)

Each benchmark carries exactly one grade class (§5 `GradeClass`), fixed in this spec — the
implementation may not relax a class to make a test pass:

| Grade | Criterion | Applies to |
|---|---|---|
| EXACT | rel. error ≤ 10⁻⁸ (or stated abs) | A6, A8 |
| TIGHT | ≤ 5% (typically ≤ 1–2%) | A1–A5, A7, A9–A12, C-family, E5.1, E7.1 |
| MODERATE | ≤ 25–30% | E4.1 |
| ORDER | within factor 3 (log-space) | sensitivity/NEF numbers: E1.3, E3.1, E3.3, E6.2, E6.3, E8.1 |
| QUALITATIVE | structural predicate; must state the predicate in the test name | E2.1, E3.2, E6.4, E7.2, E9.1 |

Reporting rules (enforced by `report_markdown`):
1. The report lists results **worst-first**, never pass-count-first.
2. Every LITERATURE-RECALL or UNVERIFIED source is surfaced in a caveats block in the same report.
3. Sensitivity benchmarks are **one-sided**: simulated shot-noise NEF must be ≤ the published value
   (papers include technical noise we don't model). A simulated NEF *below the SQL for the stated
   atom number* is an automatic FAIL (unphysical), and a simulated NEF better than published by >10×
   triggers a mandatory noise-budget review — both are report-level rules, not per-test options.
4. Cross-method disagreement inside a passing tolerance is still reported as the model uncertainty
   for that quantity (MISSION.md T6); the corpus is where those numbers come from.
5. A benchmark removed or re-toleranced requires a spec edit here with rationale — never a code-side
   constant change.

---

## 8. Known limitations — what this simulator CANNOT legitimately claim

Any RydSim output presented as a *finding* must carry this caveat block (report_markdown appends it):

1. **Model-potential absolute accuracy.** One-electron model potentials + quantum defects give
   Rydberg–Rydberg dipoles to ~0.1–1% and low-lying dipoles to ~1–2% at best. Nothing downstream
   (fields from AT, NEF, α) can be more accurate than that floor. Fine-structure-resolved defects
   for high-ℓ (>F) states are extrapolations.
2. **No atom–atom interactions.** Dipole–dipole/vdW shifts, blockade, and collision-induced
   ionization are absent. Above ~10¹⁰–10¹¹ cm⁻³ ground density with significant Rydberg fraction,
   simulated lineshapes and NEF are optimistic. (Sedlacek/Jing deliberately operated below this;
   the corpus stays valid, but density sweeps outside it are extrapolation.)
3. **No ion/free-charge physics.** Plasma formation, ion Stark broadening, and space-charge screening
   inside the cell are not modeled — a known real effect in Cs ensembles at high excitation.
4. **Cell-wall physics is phenomenological only.** Alkali adsorption screening, sapphire vs pyrex
   resistivity, and the E8 photoelectric self-dressing mechanism enter as fitted parameters, not
   first principles. kHz-and-below predictions are ORDER-grade at best (MISSION T5 scope).
5. **Multi-level leakage approximations.** Hyperfine manifolds are truncated (e.g. Sedlacek's own
   4-level reduction of a 52-level system); optical-pumping dynamics into stretched states is assumed,
   not always simulated. mJ-resolved RF coupling beyond the stated polarization geometry is truncated.
6. **Laser technical noise is not realistically modeled.** Only shot noise + stated linewidths.
   Published sensitivities include 1/f, intensity, and lock noise we cannot reconstruct — hence the
   one-sided ORDER grading of every NEF benchmark.
7. **Cold-atom results (E4) are out of scope for the thermal solver.** Only the SQL formula is
   benchmarked; RydSim must never present a "vapor-cell SQL reproduction" of Tu 2024.
8. **Transit-time & beam geometry.** Modeled as a single dephasing rate (e.g. Sedlacek's 2π·270 kHz),
   not ray-resolved; strongly focused geometries (E6's 29 μm waist) stress this approximation.
9. **The Sedlacek-vs-Holloway Doppler-factor tension (§2.2/A11)** is resolved here by derivation +
   Holloway's published formula, adjudicated numerically; until A11 passes, field inversions carry a
   possible λp/λc ↔ λc/λp systematic of (λp/λc)² ≈ 2.6×.
10. **UNVERIFIED entries** (§9 list) must never gate a release nor appear in findings without their tag.

---

## 9. Unverified-item register (complete)

| Item | Status | Self-check path |
|---|---|---|
| ~~Beterov 2009 fit coefficients~~ | **RESOLVED 2026-08-10**: Tables I & II extracted verbatim from arXiv:0810.0339 → VERIFIED (§3.4) | C1a–C1c remain as implementation checks |
| Quantum-defect digit values (Li 2003, Mack 2011, Weber–Sansonetti, Goy) | LITERATURE-RECALL (citations verified) | C4a–C4f frequency benchmarks |
| Discrete TRK fraction 0.5650 | LITERATURE-RECALL (Bethe–Salpeter) | A4b continuum computation |
| Jing LO frequency beyond "6.94 GHz" | UNVERIFIED | C4b at 3 s.f. only |
| Jing integration time behind the 780 pV/cm minimum field | UNVERIFIED | not benchmarked |
| Sedlacek 2012 effective-RF-dipole convention behind the printed 1.37e-26 C·m | UNVERIFIED (convention; number VERIFIED from arXiv:1205.4461 v1) | C5e |
| Sedlacek Nature-published version deltas vs arXiv v1 (incl. the 1.625 prose) | UNVERIFIED (paywalled) | A11 numerical adjudication |
| Holloway JAP 2017 numeric uncertainty budget | UNVERIFIED (citation + central linearity claim verified from AIP abstract) | fetch before release (E9.2+); E9.1 qualitative regime check stands now |
| npj QM 2026 cell temperature; Comms. Phys. 2026 exact per-point NEF list | UNVERIFIED | graded QUALITATIVE/ORDER only |
| A12 coupling-scan factor (derived, not found printed in a primary source today) | derivation VERIFIED-internal | treat A12 failure as physics investigation |

---

*GreyNOC · RydSim spec 09 · sources fetched 2026-08-10 · companion to specs 00–08 · house rule: reproducible or it didn't happen.*
