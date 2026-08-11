# Spec 07 — Stark Effect, Polarizability, and DC/Low-Frequency Sensing

**Seat:** Stark-physics specialist. **Module:** `rydsim.stark`.
**Depends on:** spec 01 (energies/quantum defects), spec 02 (radial matrix elements), spec 03 (angular algebra), spec 05 (cell screening model), spec 06 (EIT readout mapping), spec 08 (noise/NEF).
**Network status:** WebSearch/WebFetch were AVAILABLE during authoring (2026-08-10). Key constants and formulas below were verified online; each table row carries a Confidence tag. The full text of arXiv:1608.04515 (Yerokhin et al., published as PRA 94, 032503 (2016)) was retrieved and its equations and tables transcribed directly. **Independent re-verification pass (same date, second session):** Yerokhin Tables IV–VII values and Eqs. (10)–(14) re-extracted digit-for-digit from a freshly downloaded PDF; O'Sullivan & Stoicheff 1985 fit coefficients confirmed against the PRA 31, 2718 record; Jau & Carter 2020 title and performance figures confirmed (OSTI/arXiv:2002.04145); Grimmel NJP 17, 053005 confirmed (exact title: "…Rubidium Rydberg Stark *spectra*"); all unit-chain check values recomputed from `scipy.constants` CODATA; H-1s discrete-bound-only α computed exactly (upgraded to VERIFIED, §3.3).

---

## 1. Scope

Normative specification for:

- (a) Second-order perturbative scalar (α₀) and tensor (α₂) static polarizabilities of alkali Rydberg states |n l j m_j⟩ (Rb-85, Rb-87, Cs-133), sum-over-states with correct angular factors, truncation and convergence rules.
- (b) n*⁷ scaling, sourced fitted values for Rb nS/nD, and the atomic-unit ↔ SI ↔ spectroscopic unit conversions.
- (c) Full Stark maps by matrix diagonalization in the |n l j m_j⟩ basis: Hamiltonian construction, basis truncation, m_j conservation, avoided crossings, Inglis–Teller limit, algorithm and cost.
- (d) Linear Stark effect in hydrogenic (high-l) manifolds and DC sensitivity.
- (e) Dynamic polarizability α(ω), far-off-resonant AC Stark shifts, ponderomotive limit and the crossover.
- (f) Spectroscopic readout of Stark shifts, bias-field strategy, dE/dν responsivity.
- (g) Low-frequency screening: physics, phenomenological transfer function (interface to spec 05), kHz-band NEF prediction pipeline.

Out of scope: resonant AC dressing / Autler–Townes (spec 06/08), Zeeman + combined E×B (future spec), fields with components transverse to the quantization axis (limitation §7), field ionization rates (only the classical threshold is quoted as a validity boundary).

---

## 2. Equations

### 2.1 Conventions and Stark Hamiltonian

Static, uniform field **E** = F ẑ (F ≥ 0, SI unit V/m; ẑ = quantization axis). Electron charge −e, e > 0. Electric dipole operator **d** = −e **r**. Interaction:

```
H_S = − d · E = + e F z                                        [J]
```

Total Hamiltonian in the field: `H = H0 + H_S`, with H0 diagonal in |n l j m_j⟩ with eigenvalues E⁰(n,l,j) from spec 01 (quantum-defect energies, isotope-correct reduced-mass Rydberg constant). Fine structure is included via j; hyperfine structure is neglected for Rydberg states (see §7).

Energy-shift sign convention: for a state shifted by ΔE(F),

```
ΔE = − (1/2) α F²  −  (1/24) γ F⁴ + …                          [J]
```

α > 0 ⇔ level shifts DOWN. α (SI) has units C²·m²·J⁻¹ (equivalently J/(V/m)²). γ is the hyperpolarizability (diagnostic only in this spec; no sourced values).

Symbols: n principal quantum number; l orbital; s = 1/2; j total electronic angular momentum; m_j its z-projection (conserved by H_S, §2.6); n* = n − δ_{lj}(n) effective quantum number (δ from spec 01); ħ = h/2π; a₀ Bohr radius; E_h Hartree energy; R_M reduced-mass Rydberg constant of the species.

### 2.2 Second-order perturbation theory, m_j-resolved (NORMATIVE)

For a nondegenerate target state v = |n l j m_j⟩ (all low-l alkali Rydberg states away from accidental degeneracies):

```
ΔE_v = Σ_{k≠v} |⟨k| e F z |v⟩|² / (E⁰_v − E⁰_k)
     = − (1/2) α(v, m_j) F²
```

so, exactly:

```
α(v, m_j) = 2 e² Σ_{k≠v} |⟨k| z |v⟩|² / (E⁰_k − E⁰_v)          [C² m² J⁻¹]      (7.1)
```

- k runs over |n' l' j' m_j⟩ with the same m_j (H_S conserves m_j), l' = l ± 1 (electric-dipole selection), j' ∈ {j−1, j, j+1} ∩ {|l'−1/2|, l'+1/2}, all n' in the truncation window (§4.1).
- States above the target (E⁰_k > E⁰_v) contribute positively to α (they push v down); states below contribute negatively. The sign of α is decided by the energy ordering of the strongly coupled neighbors — e.g. Cs nD states have α₀ < 0 because (n+1)P₃/₂ lies just below (δ_P ≈ 3.56–3.59, δ_D ≈ 2.47), while Rb nS/nD have α₀ > 0. The code must reproduce these signs (benchmark RS-07-10).

Matrix element factorization (radial from spec 02, angular from spec 03):

```
⟨n' l' j' m_j | z | n l j m_j⟩ = ⟨n' l' | r | n l⟩ · A(l j; l' j'; m_j)          (7.2)
```

with the Wigner–Eckart chain (Edmonds conventions; spec 03 is NORMATIVE for phases):

```
A = (−1)^{j'−m_j} ( j'  1  j ; −m_j 0 m_j ) · ⟨l' ½ j' ‖ C(1) ‖ l ½ j⟩
⟨l' ½ j' ‖ C(1) ‖ l ½ j⟩ = (−1)^{l'+½+j+1} √[(2j+1)(2j'+1)] { l' j' ½ ; j l 1 } ⟨l' ‖ C(1) ‖ l⟩
⟨l' ‖ C(1) ‖ l⟩ = (−1)^{l'} √[(2l'+1)(2l+1)] ( l' 1 l ; 0 0 0 )
```

(  ) = Wigner 3-j, { } = 6-j. Only |A|² enters (7.1); *relative* signs matter in the diagonalization Hamiltonian (§2.6). All matrix elements MUST come from the single spec-03 implementation — never mix phase conventions. The closure self-check (benchmark RS-07-12) guards this:

```
Σ_k |⟨k| z |v⟩|²  =  ⟨v| z² |v⟩        (exact over a complete basis)             (7.3)
```

where ⟨v|z²|v⟩ is computed independently from ⟨r²⟩ (spec 02) times the analytic angular factor for |l j m_j⟩.

### 2.3 Scalar / tensor decomposition

The m_j dependence of the quadratic shift (field ∥ ẑ) is exactly

```
α(v, m_j) = α₀(n l j) + α₂(n l j) · [3 m_j² − j(j+1)] / [j(2j−1)]                (7.4)
```

valid for j ≥ 1; α₂ ≡ 0 for j = 0 and j = 1/2 (so all nS₁/₂ and nP₁/₂ states are purely scalar — enforce identically zero, benchmark RS-07-11). Closed forms (transcribed verbatim from Yerokhin et al., arXiv:1608.04515 Eqs. (10)–(11); one-electron reduction, VERIFIED):

```
α₀(v) = [2 / (3(2j_v+1))] Σ_n [C₁(κ_v,κ_n) R⁽¹⁾_vn]² / (ε_n − ε_v)

α₂(v) = √[ 40 j_v (2j_v − 1) / (3 (j_v+1)(2j_v+1)(2j_v+3)) ]
        × Σ_n (−1)^{j_v + j_n} { 1 1 2 ; j_v j_v j_n } [C₁(κ_v,κ_n) R⁽¹⁾_vn]² / (ε_n − ε_v)   (7.5)
```

where [C₁ R⁽¹⁾] is the reduced dipole matrix element (their Eqs. 12–14; equals ⟨v‖ e r C(1) ‖n⟩ in our nonrelativistic reduction). Their full quadratic shift (their Eq. 7) for arbitrary field orientation reduces, for E ∥ ẑ, to −½[α₀ + α₂(3M²−J(J+1))/(J(2J−1))]E², identical to (7.4).

**Implementation rule (self-validating):** the NORMATIVE computation is the direct m_j-resolved sum (7.1); α₀, α₂ are extracted from (7.4):
- j = 3/2: α₀ = [α(½) + α(3/2)]/2, α₂ = [α(3/2) − α(½)]/2 (the bracket in (7.4) is −1 and +1).
- j = 5/2: brackets are (−0.8, −0.2, +1.0) for m_j = (½, 3/2, 5/2); solve by least squares; the residual MUST vanish to < 10⁻⁶ relative (any m_j⁴ residual at second order is a bug).
The 6-j closed form (7.5) is the redundant second method; it must agree with the m_j-fit to < 10⁻⁹ relative (pure algebra). Published sign anchors that pin the phase convention: α₂(Rb nP₃/₂) < 0, α₂(Rb nD₃/₂), α₂(Rb nD₅/₂) > 0, α₀(Cs nD₅/₂) < 0 (Yerokhin Tables IV–VII; VERIFIED).

### 2.4 Scaling and fitted values

Scaling: radial matrix elements between neighboring Rydberg states scale as n*², energy denominators as n*⁻³ ⇒ **α ∝ n*⁷**. Because the leading fits carry both n*⁶ and n*⁷ terms, the local logarithmic slope d ln α₀/d ln n* for Rb nS over n = 40–70 evaluates to ≈ 6.5–6.7 (derived from the verified fit below; benchmark RS-07-13).

**Rb nS₁/₂ (measured, n = 15–80), O'Sullivan & Stoicheff, PRA 31, 2718 (1985) — VERIFIED:**

```
α₀(nS) [MHz/(V/cm)²] = 2.202(28)×10⁻⁹ · n*⁶ + 5.53(13)×10⁻¹¹ · n*⁷              (7.6)
```

with n* = n − δ(nS). (Spot values from this fit: 30S → 1.39, 35S → 4.15, 50S → 50.8 MHz/(V/cm)².)

**Rb nD:** no verified closed-form fit was retrieved (the O'Sullivan & Stoicheff 1986 nD paper, PRA 33, 1640, is paywalled — MISSING; do not invent one). Use the tabulated values in §3.2 as anchors and the sum-over-states for everything else.

### 2.5 Linear Stark effect, hydrogenic manifolds, Inglis–Teller (NORMATIVE for (d))

High-l states (l ≳ 4 in Rb/Cs; quantum defects < 10⁻²) are quasi-degenerate within a manifold n. A field mixes them at first order; diagonalizing e F z within the degenerate manifold (spinless treatment) gives the exact hydrogen result in parabolic quantum numbers (n₁, n₂, m), q ≡ n₁ − n₂:

```
ΔE⁽¹⁾ = (3/2) n q e a₀ F_au        (atomic units: (3/2) n q F)                    (7.7)
q ∈ { −(n−1−|m|), −(n−3−|m|), …, +(n−1−|m|) }
```

Slopes up to (3/2)n(n−1) e a₀ ≈ n² e a₀ — for n = 50 the extreme state slope is 3675 e a₀ = 4.70 GHz/(V/cm), roughly 10³× the biased-quadratic responsivity of 50S at 0.1 V/cm bias (§2.8). This is why manifold (linear) states give the best DC sensitivity per volt; the cost is optical access (they are reached via low-l admixture at finite field, or engineered dressing — see PRResearch 6, 023138 (2024) for high-l DC sensing in a room-temperature cell).

Second order (hydrogen, exact — Bethe & Salpeter; verified against multiple independent sources):

```
ΔE⁽²⁾ = − (1/16) n⁴ (17 n² − 3 q² − 9 m² + 19) F²      [atomic units]            (7.8)
```

For n=1 (q=m=0): ΔE⁽²⁾ = −(9/4)F² ⇒ α(H, 1s) = 9/2 a.u. exactly (the classic result; includes continuum — see pitfall §4.6).

**Inglis–Teller field** (adjacent manifolds n, n+1 first meet; extreme linear fans cross):

```
F_IT = F₀ / (3 n⁵),   F₀ = E_h/(e a₀) = 5.14220675112(80)×10¹¹ V/m               (7.9)
```

n = 30 ⇒ F_IT = 70.5 V/cm (literature check: "71 V/cm between n = 30 and 31", Hogan review arXiv:1603.04432 — VERIFIED). Beyond F_IT, low-l states are inside the manifold fan: perturbation theory is dead, only diagonalization (§2.6) is valid, and n is no longer a good quantum number. **Classical ionization threshold** (validity ceiling of the bound-state model): F_ion = F₀/(16 n⁴) (saddle point of −e²/4πε₀r − eFz at the zero-field energy; analytic). n = 35 ⇒ 214 V/cm.

### 2.6 Full Stark map by diagonalization (NORMATIVE for (c))

Basis: all |n' l' j' m_j⟩ with fixed m_j, n' ∈ [n_min, n_max], l' ≤ l_max, j' = l' ± ½, j' ≥ |m_j|.

- **m_j conservation:** H_S ∝ z couples only Δm_j = 0 (field ∥ ẑ). Diagonalize each m_j block separately. With no B-field, ±m_j blocks are degenerate (time reversal) — compute m_j > 0 only.
- **Matrix:** H(F) = diag(E⁰(n,l,j)) + F·D, where D_{kv} = e⟨k|z|v⟩ from (7.2). With spec-03 real phase conventions D is real symmetric ⇒ `scipy.linalg.eigh`. Build D **once**; sweep F by scaling.
- **Selection structure:** D couples l' = l ± 1, any Δn, Δj ∈ {0, ±1}: block-tridiagonal in l. Radial integrals cached per (n l, n' l') pair — O(N_states²) elements but only O(unique radial pairs) integrals.
- **Truncation:** for target n₀ and maximum field F_max, include every n' whose zero-field manifold fan can reach the target energy window: |E⁰(n') − E⁰(n₀)| ≤ 2 · (3/2) n'² e a₀ F_max, and never less than n₀ ± 4. l_max = n − 1 (full manifolds) whenever F_max ≳ 0.3 F_IT; for weak-field curvature runs (F ≤ 0.05 F_IT) l_max = l_target + 4 is acceptable **only if** the l_max-convergence check passes (§4.2). Convergence: re-run with the n-window widened by 2 (and l_max+2 where truncated); target eigenvalue at F_max must move < tol (default 100 kHz·h).
- **Cost:** dim N ≈ Σ_{n'} (2n' − 1) per m_j block ≈ 1.1×10³ for n₀ = 50, window ±5. eigh is O(N³) ≈ 30 ms–1 s per field point; a 500-point map runs in minutes single-threaded. Memory O(N²) (~10 MB float64). Vectorize over F by preallocating; do not rebuild D.
- **State tracking / avoided crossings:** eigenvalue order swaps at every avoided crossing, so label states by adiabatic continuation: at each field step match eigenvectors by maximum |⟨ψ_i(F_k)|ψ_j(F_{k+1})⟩|; if the best overlap < 0.9, bisect the field step (adaptive). Alkali low-l states **anticross** with manifold states (finite core-induced gaps); hydrogen states genuinely cross (gap → 0) — a qualitative regression test of core effects. Report per-state character (dominant |n l j⟩ weight) alongside energies.
- **Curvature extraction (perturbative cross-check):** fit E(F) over F ∈ [0, F_fit], F_fit = min(0.03 F_IT, 0.1 E_mix) with E_mix = min_k |E⁰_k − E⁰_v| / (2|e⟨k|z|v⟩|), to E₀ − ½αF² − (1/24)γF⁴ (even powers only). The fitted α must match (7.1) to < 0.5% (benchmark RS-07-09); the F⁴ term is reported as a hyperpolarizability estimate (diagnostic, unsourced).

Method reference: Zimmerman, Littman, Kash & Kleppner, PRA 20, 2251 (1979) (canonical alkali Stark-map method; VERIFIED as the method used and extended to n = 35, 70 and 500 V/cm with ~2 MHz accuracy by Grimmel et al., NJP 17, 053005 (2015)).

### 2.7 Dynamic polarizability, AC Stark shift, ponderomotive limit (NORMATIVE for (e))

General dynamic (real, off-resonant) polarizability — exactly as requested, m_j-resolved:

```
α(v, m_j; ω) = (2/ħ) Σ_{k≠v} ω_k |⟨k| d_z |v⟩|² / (ω_k² − ω²),   ω_k ≡ (E⁰_k − E⁰_v)/ħ    (7.10)
```

Scalar/tensor versions: replace the frequency factor 1/(E_k−E_v) in (7.5) by (2ω_k/ħ)/(ω_k²−ω²)·(1/2)… equivalently multiply each term of the static sums by ω_k²/(ω_k²−ω²). α(v; 0) ≡ static α of (7.1) (continuity benchmark RS-07-14).

- **AC shift:** for E(t) = E₀ cos ωt ẑ, far from every resonance (|ω − |ω_k|| ≫ Ω_Rabi,k, Γ_k):
  `ΔE = −(1/4) α(ω) E₀²` (time-averaged; the ¼ vs the DC ½ is the cos² average).
- **Quasi-static regime (ω ≪ min|ω_k|):** α(ω) ≈ α(0)·[1 + O(ω²/ω_dom²)], ω_dom = dominant-transition frequency (nS ↔ nP, ≈ 2π·31 GHz for Rb n≈50). RF below 1 GHz is quasi-static to < 10⁻³ at n ≈ 50. This is the operating point for "far-off-resonant RF sensing with the DC polarizability".
- **Ponderomotive regime (ω ≫ all strongly coupled ω_k):** using the Thomas–Reiche–Kuhn sum rule (Σ_k f_vk = 1 for a one-electron local potential),
  ```
  α(ω) → − e² / (m_e ω²)     ⇒     ΔE → + e² E₀² / (4 m_e ω²) = U_p              (7.11)
  ```
  the free-electron/ponderomotive energy (shift is UP; α < 0). For Rydberg n ≈ 50 the oscillator strength is concentrated in transitions at ~2R_M·c/n³ (tens of GHz), so (7.11) holds to ~1% already for ω/2π ≳ 1 THz — and in particular for any optical field (this is the physics of ponderomotive Rydberg traps/lattices).
- **Crossover:** between the highest strongly coupled ω_k and the TRK-saturated regime, evaluate (7.10) with the truncated sum AND report the truncation completeness S = Σ_window f_vk; if S < 0.98, widen the window or (for ω ≫ spacings) switch to (7.11); quote |1 − S|·e²/(m_e ω²) as the model uncertainty. Never present the truncated sum alone in the crossover without this bound.
- **Resonances:** (7.10) diverges at ω = ω_k; within Γ or Ω_Rabi of a resonance the dressed-state treatment of spec 06/08 is normative, not this spec.

### 2.8 Spectroscopic readout and DC/low-frequency sensitivity (NORMATIVE for (f))

The Rydberg-state shift is read as a shift of the EIT resonance. This spec quotes shifts in Rydberg-energy frequency units, δν ≡ ΔE/h; the mapping to probe-laser detuning units (Doppler wavelength-mismatch factor) is owned by spec 06 and must be applied exactly once.

Quadratic response with bias field E_b (total field E_b + δE, signal |δE| ≪ E_b):

```
ν(E) = ν₀ − (α/2h) E²
Responsivity   R ≡ |dν/dE| = (α/h) E_b                       [Hz/(V/m)]           (7.12)
Inverse        dE/dν = h/(α E_b)                              [(V/m)/Hz]
Min. field     δE_min = h·δν_min / (α E_b)
```

- **Zero-bias pathology:** R → 0 as E_b → 0; unbiased, a resolution δν_min detects only δE = √(2h δν_min/α) (square-root, not linear). Example, Rb 50S₁/₂ (α/h = 50.5 MHz/(V/cm)²): δν_min = 1 kHz gives 6.3 mV/cm unbiased, but 40 µV/cm at E_b = 0.5 V/cm. **A bias field is mandatory for linear small-signal DC/LF sensing.**
- **Optimal bias:** the bias's own fractional rms inhomogeneity η ≡ δE_b,rms/E_b over the probe volume inhomogeneously broadens the line by δν_inh ≈ (α/h) η E_b². With intrinsic linewidth Γ₀ (spec 06) and δν_min ∝ Γ_eff = Γ₀ + (α/h) η E_b²:
  ```
  minimize [Γ₀ + (α/h)ηE_b²] / [(α/h)E_b]  ⇒  E_b* = √( h Γ₀ / (α η) )
  NEF_min ∝ 2 √( η Γ₀ h / α )                                                   (7.13)
  ```
  subject to hard caps E_b ≤ min(0.1·E_mix, ~0.03 F_IT) (quadratic validity, §2.6). If η → 0 (very homogeneous bias), push E_b to the cap. Practical bias routes: electrodes (in-vacuum cells), or optically induced internal fields for screened vapor cells (Jau & Carter, PRApplied 13, 054034 (2020); photoillumination control, arXiv:1909.05793) — parameters owned by spec 05.
- **Manifold (linear) operation:** replace (α/h)E_b in (7.12) by the state's linear slope k_s = (3/2) n q e a₀/h (up to 4.7 GHz/(V/cm) at n = 50); no bias needed; best raw dE/dν. Trade-offs: optical access, state lifetime in the fan, stray-field-induced state scrambling near crossings — flag, don't model here.

### 2.9 Low-frequency screening and kHz-band NEF (NORMATIVE interface for (g))

Physics (mechanisms; the *parameters* are owned and calibrated by spec 05):

1. **Adsorbed-alkali wall film:** Rb/Cs adsorbed on the inner wall forms a (weakly) conductive layer; the cell interior approaches a lossy Faraday cage. External quasi-static fields terminate on redistributed wall charge with a relaxation time τ_s (RC time of the film–dielectric system). Slow field components are attenuated; fast ones pass.
2. **Free-charge (space-charge) screening:** photoelectrons from windows/walls and ions/electrons from Rydberg ionization (collisions, BBR photoionization) drift to null the internal field, relaxation time ~ ε₀/σ_vapor.
3. **Patch potentials / adsorbate dipoles:** quasi-static offset fields and slow drift; they set an unknown internal E-offset (motivates Zeeman-referenced locking and internal-bias schemes).

Canonical phenomenological transfer function (form defined here; **spec 05 owns the parameter values and may generalize to multi-pole**):

```
E_int(f) = T(f) · E_ext(f)
T(f) = T_DC + (1 − T_DC) · (i 2π f τ_s) / (1 + i 2π f τ_s)                       (7.14)
```

|T| → T_DC as f → 0, |T| → 1 for f ≫ f_c = 1/(2π τ_s). Anchor (VERIFIED): monocrystalline-sapphire Rb cells show screening timescales up to order **seconds** (Jau & Carter, "Vapor-Cell-Based Atomic Electrometry for Detection Frequencies below 1 kHz", PRApplied 13, 054034 (2020); arXiv:2002.04145) ⇒ f_c well below 1 Hz for the wall-film pole; their demonstrated system (11 mm³ active volume, optically induced internal bias field): spectral noise floor ≈ **0.34 (mV/m)/√Hz = 3.4 (µV/cm)/√Hz** with a **3-dB low-frequency cutoff ≈ 770 Hz** — i.e. even the best published sub-kHz system retains a sub-kHz roll-off from residual screening/readout dynamics. Standard borosilicate screens far faster (numeric τ_s: MISSING — spec 05 calibration; treat kHz-band glass-cell predictions as parameter-dependent).

**kHz-band NEF pipeline (the deliverable of this spec + 05 + 08):**

```
NEF_ext(f) = h · δν_min(f) / ( α E_b |T(f)| )      [(V/m)/√Hz]                   (7.15)
```

δν_min(f): spectroscopic frequency-resolution PSD from spec 08 (includes 1/f laser noise at low f); α, E_b from this spec; T(f) from spec 05. Sanity anchor for the assembled pipeline (not a unit test): published 13.5 nV/cm/√Hz at 100 kHz in engineered sapphire cells (npj Quantum Materials 2026, per Study Report §4.5 — LITERATURE-RECALL, cited from the program study report).

---

## 3. Constants and parameter tables

### 3.1 Units and conversions — THE MINEFIELD (all conversion factors MUST be derived at runtime from `scipy.constants` CODATA values; the numbers below are check values, not hard-code values)

| Quantity | Value | Source | Confidence |
|---|---|---|---|
| Atomic unit of polarizability e²a₀²/E_h | 1.64877727212(51)×10⁻⁴¹ C²·m²·J⁻¹ | CODATA 2022 (NIST CUU) | VERIFIED |
| **1 a.u. → α/h in Hz/(V/m)²** | **2.48832×10⁻⁸** | derived: (e²a₀²/E_h)/h | VERIFIED (derived; matches standard factor in Mitroy et al., J. Phys. B 43, 202001 (2010)) |
| **1 a.u. → α/h in Hz/(V/cm)²** | **2.48832×10⁻⁴** (= ×10⁴ above) | derived | VERIFIED |
| **1 a.u. → α/h in MHz/(V/cm)²** | **2.48832×10⁻¹⁰** | derived | VERIFIED |
| 1 MHz/(V/cm)² → a.u. | 4.01878×10⁹ | derived (reciprocal) | VERIFIED |
| Gaussian "polarizability volume" | α[cm³] = α[a.u.] × a₀³; a₀³ = 1.481847×10⁻²⁵ cm³ | definition + CODATA | VERIFIED |
| Steck cross-check | α[cm³] = 5.95531×10⁻²² × α[Hz/(V/cm)²] | Steck Rb-87 datasheet | VERIFIED (consistency: 5.95531e-22 × 2.48832e-4 = 1.48185e-25 = a₀³ ✓) |
| Atomic unit of electric field F₀ = E_h/(e a₀) | 5.14220675112(80)×10¹¹ V/m = 5.14220675×10⁹ V/cm | CODATA 2022 | VERIFIED |
| Atomic unit of dipole, e·a₀ | 8.478354×10⁻³⁰ C·m (use scipy.constants product) | CODATA (derived e × a₀) | VERIFIED |
| **e·a₀/h** | **1.2795448 MHz/(V/cm)** = 1.2795448×10⁴ Hz/(V/m) | derived (CODATA 2022 via scipy.constants) | VERIFIED |

Internal rule: `rydsim.stark` computes in SI (J, V/m, C·m; α in C²m²J⁻¹). Conversions happen only in I/O helpers. Every public result object carries the value in a.u., SI, and Hz/(V/cm)² simultaneously so unit bugs surface immediately.

### 3.2 Sourced polarizability values (static)

All a.u. values below are atomic units of polarizability (multiply by 2.48832×10⁻¹⁰ for MHz/(V/cm)²). Primary source: Yerokhin, Buhmann, Fritzsche & Surzhykov, PRA 94, 032503 (2016) [arXiv:1608.04515, Tables IV–VII, transcribed from the retrieved PDF]. "exp" rows therein: refs [32,33] = O'Sullivan & Stoicheff PRA 31, 2718 (1985) / PRA 33, 1640 (1986); Cs exp refs [35–38] therein. NOTE: the preprint labels the tensor tables "a₀⁵" — dimensional analysis, the direct comparison with experimental α₂ in the same rows, and α₂/α₀ ~ O(1) show this is a typographical unit label; both α₀ and α₂ are in a.u. of polarizability (a₀³-class).

| State | α₀ [a.u.] | α₂ [a.u.] | α₀/h [MHz/(V/cm)²] | Source | Confidence |
|---|---|---|---|---|---|
| Rb 30S₁/₂ | 5.59(6)×10⁹ (exp); 5.55×10⁹ (th) | 0 | 1.391 | Yerokhin T.IV / O'S&S | VERIFIED |
| Rb 35S₁/₂ | 1.69(1)×10¹⁰ (exp); 1.66×10¹⁰ (th) | 0 | 4.21 | Yerokhin T.IV / O'S&S | VERIFIED |
| Rb 45S₁/₂ | 1.00(4)×10¹¹ (exp) | 0 | 24.9 | Yerokhin T.IV / O'S&S | VERIFIED |
| Rb 50S₁/₂ | 2.03(1)×10¹¹ (exp & th) | 0 | 50.5 | Yerokhin T.IV / O'S&S | VERIFIED |
| Rb 35P₃/₂ | 9.48×10¹⁰ (th only) | −8.65×10⁹ (th only) | 23.6 | Yerokhin T.IV–V | VERIFIED (theory value) |
| Rb 35D₅/₂ | 2.53(8)×10¹⁰ (exp); 2.58×10¹⁰ (th) | 4.18(8)×10¹⁰ (exp); 4.15×10¹⁰ (th) | 6.30 | Yerokhin T.IV–V / O'S&S | VERIFIED |
| Rb 50D₅/₂ | 2.89(16)×10¹¹ (exp) | 5.39(20)×10¹¹ (exp) | 71.9 | Yerokhin T.IV–V / O'S&S | VERIFIED |
| Cs 50S₁/₂ | 2.15×10¹¹ (th only) | 0 | 53.5 | Yerokhin T.VI | VERIFIED (theory value) |
| Cs 39D₅/₂ | −4.9(2)×10¹¹ (exp); −4.69×10¹¹ (th) | 5.6(1)×10¹¹ (exp) | −121.9 | Yerokhin T.VI–VII | VERIFIED (note the NEGATIVE α₀) |
| Cs 11D₅/₂ | −1.358(2)×10⁷ (exp) | 1.705(5)×10⁷ (exp) | −3.38×10⁻³ | Yerokhin T.VI–VII ref [36] | VERIFIED (low-n; outside engine validity, sign/order check only) |
| Rb-87 5S₁/₂ (ground) | 319 a.u. ≈ h·0.0794(16) Hz/(V/cm)² | 0 | 7.94×10⁻⁸ | Steck Rb-87 datasheet | VERIFIED (NOT reproducible by this engine — continuum + core; documentation row only) |

Rb-85 vs Rb-87: identical to the precision here (quantum defects are isotope-independent at current accuracy; reduced-mass R_M differences enter at ~10⁻⁶ relative — spec 01).

### 3.3 Other constants

| Quantity | Value | Source | Confidence |
|---|---|---|---|
| O'S&S nS fit coefficients | Eq. (7.6): 2.202(28)×10⁻⁹, 5.53(13)×10⁻¹¹ MHz/(V/cm)² | PRA 31, 2718 (1985) | VERIFIED |
| Hydrogen ΔE⁽¹⁾, ΔE⁽²⁾ | Eqs. (7.7)–(7.8) | Bethe–Salpeter §51–52; cross-checked vs 3 independent sources incl. arXiv:physics/0010038 | VERIFIED |
| α(H 1s) exact | 9/2 a.u. | analytic (follows from (7.8), n=1) | VERIFIED |
| α(H 1s) discrete-bound-only | 3.6633 a.u. = 0.8141 × 9/2 (2p alone: f=0.41620, ΔE=0.375 ⇒ 2.9596 a.u.; discrete TRK sum = 0.5650) | computed exactly from analytic f(1s→np) = 256 n⁵(n−1)^{2n−4}/[3(n+1)^{2n+4}], summed n=2…∞ (mpmath, 30 digits, this session) | VERIFIED (analytic-numeric) |
| F_IT = F₀/(3n⁵) | n=30 ⇒ 70.5 V/cm | Gallagher, *Rydberg Atoms*; Hogan arXiv:1603.04432 ("71 V/cm between n=30 and 31") | VERIFIED |
| F_ion = F₀/(16n⁴) | n=35 ⇒ 214 V/cm | analytic saddle point; standard | VERIFIED (analytic) |
| U_p = e²E₀²/(4 m_e ω²) | (7.11) | standard strong-field result; TRK-derivable | VERIFIED (analytic) |
| Rb quantum defects (illustrative only) | δ(nS)=3.1311804, δ(nP₁/₂)=2.6548849, δ(nP₃/₂)=2.6416737, δ(nD₅/₂)=1.3464657 | Li, Mourachko, Noel, Gallagher PRA 67, 052502 (2003) | LITERATURE-RECALL — spec 01 is normative; do not duplicate values in code |
| Sapphire-cell screening time | up to O(1 s) ⇒ f_c ≪ 1 Hz (wall-film pole) | Jau & Carter, PRApplied 13, 054034 (2020), arXiv:2002.04145 | VERIFIED (abstract level; exact τ_s: spec 05) |
| Sapphire-cell demonstrated NEF | 0.34 (mV/m)/√Hz noise floor; 3-dB low cutoff ≈ 770 Hz; 11 mm³ volume | Jau & Carter 2020 (same) | VERIFIED (pipeline sanity anchor, not a unit test) |
| Borosilicate screening τ_s | — | — | MISSING (spec 05 must calibrate; do not guess) |
| Stark-map validation dataset | Rb n=35, 70 maps to 500 V/cm, ≈2 MHz accuracy | Grimmel et al., NJP 17, 053005 (2015) | VERIFIED (existence/scope; digitization needed for integration test) |

---

## 4. Numerical method and pitfalls

### 4.1 Perturbative sum (7.1): truncation and convergence

- l' = l ± 1 exactly (no other l' contribute — this is exact, not a truncation).
- n' window: n − Δn … n + Δn with **Δn = 15 default** for 10⁻⁴ relative convergence of low-l Rydberg α (dominant terms are Δn = 0, ±1; contributions fall by >3 orders of magnitude by |Δn| ≈ 10 — verify empirically, do not assume an exponent). Convergence criterion: double Δn; require |Δα/α| < 10⁻⁴, else fail loudly.
- Continuum: negligible for Rydberg states (n ≥ 10, < 10⁻⁴ relative — the engine's validity floor); **catastrophic for ground/low states** (H 1s: ~19% missing → §3.3). Enforce `n >= 10` at API level; document, don't silently return bad ground-state numbers.
- Near-degeneracy guard: report E_mix = min_k |E⁰_k − E⁰_v|/(2|e⟨k|z|v⟩|) with every α; perturbative α is only meaningful for applied fields F ≪ E_mix. For Rb nD (close (n+2)P/(n−1)F partners) this cap is much lower than for nS — the result object must carry it.
- Radial elements (spec 02) come with a two-method disagreement estimate; propagate it: σ_α² = Σ (∂α/∂R_k)² σ_{R_k}² (diagonal approximation is fine — terms are additive).

### 4.2 Diagonalization

- Real-symmetric matrix, `scipy.linalg.eigh`, per-m_j block (§2.6). Never `eig` (complex noise breaks tracking).
- Basis rule + convergence re-run as in §2.6. FAIL if the target eigenvalue moves more than tol when the window widens — do not average.
- Adaptive field stepping for tracking (overlap ≥ 0.9 else bisect). Near-exact crossings (hydrogen test case) tracking is ambiguous by nature — label by character weights, not adiabatic index.
- Energy consistency: H₀ diagonal MUST use spec-01 energies from the same model as the wavefunctions that produced the radial integrals (same n*, same defects). Mixing measured term energies for some states with QDT for others produces spurious curvatures at the 1–10 MHz level — forbidden.
- Curvature-fit window rule of §2.6 (F_fit = min(0.03 F_IT, 0.1 E_mix)); check the quartic coefficient is < 1% of the quadratic term at F_fit, else shrink.

### 4.3 Unit discipline

All internal SI. The three-representation result object (§3.1) plus the runtime-derived conversion constants (benchmark RS-07-01) make unit regressions test-visible. Never hard-code 2.48832e-4 etc. in library code — only in tests, as independent check values.

### 4.4 Phase/convention discipline

All angular factors from spec 03's single implementation. The closure test (7.3) and the α₂ sign anchors (§2.3) are the tripwires. |·|² protects the perturbative sum from global phase errors, but the Stark matrix D is phase-sensitive through interference at avoided crossings: a wrong relative sign shows up as wrong anticrossing gaps (caught by RS-07-09 only weakly; the closure + hydrogen-crossing tests catch it directly).

### 4.5 Dynamic α

Same machinery as static with the ω_k²/(ω_k²−ω²) factor; refuse (raise) if any |ω − |ω_k|| < 10 × max(Γ_k, Ω_Rabi) — resonant regime belongs to spec 06/08. In the crossover, apply the TRK completeness bound of §2.7.

### 4.6 Known numeric traps

- Cancellation in radial integrals at large Δn (oscillatory integrands): rely on spec-02's dual-method error, don't chase digits.
- For j = 5/2 the least-squares α₀/α₂ extraction must use all three m_j values and check the residual (§2.3).
- The Yerokhin "a₀⁵" table-label typo (§3.2): anyone re-deriving benchmark values from the paper must read the note or they will "correct" the benchmarks by ×a₀².
- Degenerate PT: (7.1) diverges if a true degeneracy sits in the sum (hydrogenic high-l target states). For l ≥ 4 targets use the manifold diagonalization (§2.5–2.6), never (7.1).

---

## 5. Recommended Python API (`rydsim/stark.py`; numpy-vectorized; Python 3.11)

```python
from dataclasses import dataclass
import numpy as np

# ---- engine protocol: what this module consumes from specs 01/02/03 ----
class AtomEngine(Protocol):
    def energy_J(self, n: int, l: int, j: float) -> float: ...            # spec 01
    def radial_dipole_m(self, n1,l1, n2,l2) -> tuple[float, float]: ...   # spec 02: (value [m], sigma [m])
    def angular_factor(self, l,j,mj, l2,j2) -> float: ...                 # spec 03: A in Eq. (7.2), fixed phases

@dataclass(frozen=True)
class RydState:
    species: str   # "Rb85" | "Rb87" | "Cs133" | "H" (test species)
    n: int; l: int; j: float; mj: float

@dataclass(frozen=True)
class Polarizability:
    """All three representations ALWAYS populated (unit tripwire, spec 07 §3.1)."""
    alpha_SI: float          # C^2 m^2 / J   (per-m_j value, Eq. 7.1)
    alpha_au: float
    alpha_Hz_per_Vcm2: float # alpha/h
    alpha0_au: float; alpha2_au: float   # Eq. 7.4 decomposition (alpha2=0.0 for j<=1/2)
    sigma_rel: float         # propagated radial-element + truncation uncertainty
    E_mix_Vm: float          # perturbative validity field cap (spec 07 §4.1)
    n_window: int; converged: bool

def alpha_perturbative(state: RydState, engine: AtomEngine, *,
                       n_window: int = 15, omega_rad_s: float = 0.0) -> Polarizability:
    """Second-order m_j-resolved sum (7.1)/(7.10). Raises EngineValidityError for n<10,
    DegeneracyError if a quasi-degenerate partner (|dE| < 100*|coupling*1 V/m|) is found,
    ResonanceError if omega is within the guard band of any omega_k (spec 07 §4.5)."""

def alpha_scalar_tensor(species: str, n: int, l: int, j: float, engine: AtomEngine, *,
                        n_window: int = 15, omega_rad_s: float = 0.0,
                        method: str = "mj-fit") -> tuple[float, float]:
    """(alpha0_au, alpha2_au). method='mj-fit' (normative) or '6j' (redundant cross-check,
    Eq. 7.5); the two must agree to 1e-9 rel — library asserts this when method='6j'."""

def alpha_dynamic(state: RydState, engine: AtomEngine,
                  omega_rad_s: np.ndarray, *, n_window: int = 15) -> np.ndarray:
    """Vectorized (7.10) over omega. Also returns TRK completeness S via .attrs or tuple;
    caller must apply spec 07 §2.7 crossover rule."""

def alpha_ponderomotive(omega_rad_s: np.ndarray | float) -> np.ndarray | float:
    """-e^2/(m_e omega^2), SI. Eq. (7.11)."""

# ---- Stark map ----
@dataclass(frozen=True)
class StarkBasis:
    species: str; mj: float; n_min: int; n_max: int; l_max: int | None  # None => l up to n-1

@dataclass
class StarkMapResult:
    fields_Vm: np.ndarray            # (NF,)
    energies_J: np.ndarray           # (NF, N) adiabatically tracked (tracking flag) 
    character: np.ndarray            # (NF, N, k) dominant zero-field weights or sparse repr
    basis: StarkBasis
    zero_field_labels: list[tuple]   # (n,l,j) per column at F=0
    converged: bool

def build_stark_matrices(basis: StarkBasis, engine: AtomEngine) -> tuple[np.ndarray, np.ndarray]:
    """(E0_diag [J], D [C*m]); D real symmetric; H(F) = diag(E0) + F*D. Cached radial pairs."""

def stark_map(basis: StarkBasis, fields_Vm: np.ndarray, engine: AtomEngine, *,
              track: bool = True, overlap_min: float = 0.9,
              conv_tol_Hz: float = 1e5) -> StarkMapResult:
    """eigh per field point; adaptive substepping when overlap < overlap_min;
    auto re-run with widened window until target-window eigenvalues move < conv_tol_Hz*h."""

def stark_map_curvature(res: StarkMapResult, state: tuple, *,
                        fit_max_field_Vm: float | None = None) -> tuple[float, float]:
    """(alpha_SI, gamma_SI_diagnostic) from even-power fit per spec 07 §2.6."""

def inglis_teller_field_Vm(n: int) -> float          # Eq. (7.9)
def classical_ionization_field_Vm(n: int) -> float   # F0/(16 n^4)
def hydrogen_stark_energy_J(n: int, q: int, m: int, F_Vm: float, *, order: int = 2) -> float
    # Eqs. (7.7)-(7.8); exact benchmark generator

# ---- sensing / readout ----
def stark_responsivity_Hz_per_Vm(alpha_SI: float, E_bias_Vm: float) -> float   # (7.12)
def optimal_bias_Vm(alpha_SI: float, gamma0_Hz: float, frac_inhomog: float, *,
                    E_cap_Vm: float) -> tuple[float, float]:
    """(E_b*, NEF scale factor 2*sqrt(eta*Gamma0*h/alpha)); clipped at E_cap_Vm. (7.13)"""
def nef_dc_Vm_sqrtHz(alpha_SI: float, E_bias_Vm: float, dnu_Hz_sqrtHz: float) -> float

@dataclass(frozen=True)
class ScreeningModel:      # canonical form (7.14); spec 05 owns values / generalization
    T_dc: float; tau_s_s: float
def screening_transfer(f_Hz: np.ndarray, m: ScreeningModel) -> np.ndarray   # complex T(f)
def nef_external_Vm_sqrtHz(f_Hz: np.ndarray, nef_internal_Vm_sqrtHz: np.ndarray,
                           m: ScreeningModel) -> np.ndarray                 # (7.15)
```

Contracts: every function raises rather than extrapolating outside validity (n < 10, F > F_ion, resonant ω). All array functions vectorized over their leading argument. No module-level hard-coded unit factors (§4.3).

---

## 6. Validation benchmarks (→ `tests/test_stark.py`)

Tolerances are on relative error unless stated. "map" = diagonalization route; "pert" = Eq. (7.1).

| ID | Quantity | Expected | Tolerance | Source | Confidence |
|---|---|---|---|---|---|
| RS-07-01 | Unit chain: 1 a.u. α → Hz/(V/cm)², derived from scipy.constants | 2.48832×10⁻⁴ | 1×10⁻⁵ | CODATA 2022 (check value) | VERIFIED |
| RS-07-02 | e·a₀/h | 1.2795448 MHz/(V/cm) | 1×10⁻⁶ | CODATA 2022 (derived) | VERIFIED |
| RS-07-03 | H, n=10, q=9, m=0 linear slope (map, within-manifold) | 135 e a₀ = 172.74 MHz/(V/cm) | 1×10⁻³ | Eq. (7.7), exact | VERIFIED |
| RS-07-04 | H, n=10, q=0, m=0 quadratic α (map curvature, Δn ≥ 6) | 2.1488×10⁶ a.u. (= (1/8)n⁴(17n²+19)) | 1% | Eq. (7.8), exact | VERIFIED |
| RS-07-05 | Rb 50S₁/₂ α₀ (pert) | 2.03×10¹¹ a.u. = h·50.5 MHz/(V/cm)² | 3% | O'S&S 1985 exp via Yerokhin T.IV | VERIFIED |
| RS-07-06 | Rb 35S₁/₂ α₀ (pert) | 1.69×10¹⁰ a.u. = h·4.21 MHz/(V/cm)² | 3% | same | VERIFIED |
| RS-07-07 | Rb 35D₅/₂ α₀; α₂ (pert) | 2.53×10¹⁰ a.u.; 4.18×10¹⁰ a.u. | 6%; 6% | O'S&S 1986 exp via Yerokhin T.IV–V | VERIFIED |
| RS-07-08 | Inglis–Teller field, n=30 (formula) | 70.54 V/cm | 0.5% | Eq. (7.9) | VERIFIED |
| RS-07-08b | First n=30/31 anticrossing field in Rb map | within 20% of 70.5 V/cm | qualitative | Eq. (7.9) + defect shifts | VERIFIED (formula) |
| RS-07-09 | pert α vs map curvature, Rb 50S₁/₂ m_j=½, F ≤ 0.03 F_IT | equal | 0.5% | self-consistency | self-check |
| RS-07-10 | Sign tests: α₂(Rb 35P₃/₂) < 0; α₀(Cs 39D₅/₂) = −4.9×10¹¹ a.u. | signs + value | sign exact; 8% | Yerokhin T.V–VII | VERIFIED |
| RS-07-11 | α₂(nS₁/₂) and α₂(nP₁/₂) | exactly 0.0 | exact | angular identity (j=½) | VERIFIED |
| RS-07-12 | Closure: Σ|⟨k|z|v⟩|² / ⟨v|z²|v⟩, Rb 50S, Δn=25 | 1 | 1% | Eq. (7.3) | self-check |
| RS-07-13 | d ln α₀ / d ln n*, Rb nS, n = 40–70 | 6.4 – 6.9 | band | derived from Eq. (7.6) | VERIFIED |
| RS-07-14 | α(ω=2π·1 MHz)/α(0) − 1, Rb 50S | 0 | 1×10⁻⁴ | quasi-static limit of (7.10) | self-check |
| RS-07-15 | α(ω=2π·10 THz), Rb 50S vs −e²/(m_e ω²) | equal | 2% | TRK / Eq. (7.11) | VERIFIED (analytic limit) |
| RS-07-16 | H 1s discrete-only α / exact 4.5 a.u. (documents continuum omission) | 0.8141 (α_disc = 3.6633 a.u.) | ±0.005 abs (finite n'-window tail ≈ 5.4/n_max³ a.u.) | §3.3, exact f(1s→np) sum | VERIFIED (analytic-numeric, this session) |
| RS-07-17 | O'S&S fit consistency: pert α₀(nS) vs Eq. (7.6), n ∈ {30,40,50,60,70} | fit value | 5% | PRA 31, 2718 (1985) | VERIFIED |
| RS-07-18 | j=5/2 m_j-fit residual (α beyond α₀+α₂ form) | 0 | 1×10⁻⁶ | Eq. (7.4) exactness at 2nd order | self-check |

Integration-level (not pytest-gating): reproduce Grimmel et al. NJP 17, 053005 (2015) Rb n=35 Stark map to ≈ 2–5 MHz against digitized figure data (dataset acquisition task for the validation librarian, spec 09); assemble (7.15) with spec 05/08 outputs and compare order-of-magnitude against the 100 kHz sapphire-cell anchor (§2.9).

---

## 7. Known limitations / where the model breaks down

1. **Field orientation:** everything here assumes E ∥ ẑ (m_j conserved). Transverse components require the full m_j-coupled matrix (3-j with q = ±1); not specified here.
2. **Hyperfine structure ignored:** invalid where the Stark shift is comparable to Rydberg hyperfine splittings (relevant only at very small shifts for low n, and for ground/intermediate EIT levels — spec 06 owns those).
3. **Above F_ion** (= F₀/16n⁴): states are ionizing resonances; the Hermitian bound-state map produces energies without widths. Grimmel's data beyond 214 V/cm (n=35) shows this regime — our map may track the resonance positions qualitatively but widths/lifetimes are out of scope.
4. **n < 10:** continuum + core-penetration contributions to α are not represented (H 1s misses ~19%); the engine must refuse. Ground-state row in §3.2 is documentation, not a target.
5. **Perturbative α near quasi-degeneracies** (alkali nD with close P/F partners; any high-l target): use the map; (7.1) carries an explicit validity cap E_mix and errors out on true degeneracy.
6. **Resonant AC fields:** (7.10) excludes |ω − ω_k| ≲ linewidth/Rabi guard band — dressed-state physics is spec 06/08.
7. **Hyperpolarizability γ:** diagnostic output only; no sourced benchmark values — never report γ as a finding without a dedicated validation.
8. **Screening model (7.14):** phenomenological, single-pole + DC leak; real cells show multi-timescale, illumination- and history-dependent screening (Jau & Carter observed seconds-scale dynamics). Spec 05 owns calibration; kHz-band NEF predictions inherit its parameter uncertainty and MUST quote it.
9. **DC "absolute" sensing:** patch potentials and screening make the *internal* DC field differ from the applied one by unknown offsets; this spec predicts responsivity to internal fields. Traceable absolute DC metrology requires the bias/locking strategies referenced in §2.8–2.9.
