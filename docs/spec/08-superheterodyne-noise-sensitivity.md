# 08 — Superheterodyne Architecture, Noise Model and Sensitivity

**RydSim physics specification — module 08 (the sensitivity/"money" module).**
Status of source verification: **network WAS available** during authoring. Primary sources fetched
and quoted this session: Jing et al., *Nat. Phys.* **16**, 911 (2020) [abstract + arXiv:1902.11063 v1
full text via ar5iv]; Wang(?) et al., "Approaching the standard quantum limit of a Rydberg-atom
microwave electrometer", *Sci. Adv.* **10**, eads0683 (2024) [full text via PMC11661427;
arXiv:2307.15617]; Meyer, Castillo, Cox, Kunz, *J. Phys. B* **53**, 034001 (2020)
[arXiv:1910.00646 via ar5iv]. Every derived number in §6 was re-computed numerically during
authoring (script `verify_bench.py`, session scratchpad). **Second-pass verification
2026-08-10 (network available):** headline sensitivities, Sci. Adv. SQL formula
(h-convention as printed), Jing v1 parameters (Ω_p, Ω_c, E_LO*, QPNL prefactor, 90 dB DR),
and all §6 derived rows re-fetched/recomputed independently; corrections folded in
(Ω_p(0)/2π = 4.6 MHz, T′ = 420 s, v1 vs published sensitivity values). Confidence tags:
`VERIFIED` (fetched/recomputed this session), `LITERATURE-RECALL` (recalled, consistent with
fetched material but not directly quoted), `UNVERIFIED` (memory only — treat as default input,
not truth).

Dependencies: this module consumes the steady-state and time-dependent OBE solver, the EIT
susceptibility χ convention, Doppler averaging, and transition dipole moments from the ladder/EIT
modules of this spec series. All symbols SI unless noted. **All angular frequencies internally in
rad/s** (`omega_*`, `Omega_*`, `delta_*`, `Gamma_*`); Hz only at I/O boundaries (`freq_*`, `f_*`).
Unit note used throughout: 1 nV/cm = 10⁻⁷ V/m.

---

## 1. Scope

Models the atomic superheterodyne (superhet) receiver of Jing et al. (2020): a strong resonant
local-oscillator (LO) RF field E_LO on a Rydberg–Rydberg transition plus a weak co-polarized
signal field E_sig detuned by δ, read out via Rydberg-EIT probe transmission. Deliverables:

- (a) beat-note transduction: linear-in-E_sig response, phase recovery, validity conditions,
  optimal LO working point;
- (b) transduction transfer function H(δ) and instantaneous bandwidth (IBW) from OBE linear
  response — computed, not asserted;
- (c) full noise budget (photon shot, RIN, detector/Johnson, laser-frequency conversion, atom
  projection noise / SQL), each mapped to equivalent field noise in (V/m)/√Hz;
- (d) receiver metrics: NEF, minimum detectable field vs integration time, SNR, dynamic range
  (1-dB compression), SFDR, equivalent noise temperature and noise figure via the
  standard-antenna-comparison formalism;
- (e) the sensitivity–bandwidth tradeoff as a numerical experiment;
- (f) digital demodulation (I/Q, lock-in) and the *exact* bookkeeping connecting measurement
  bandwidth, averaging time, and the reported (V/cm)/√Hz figure.

Out of scope here: level structure, dipole matrix elements, EIT lineshapes (earlier modules);
antenna/waveguide field delivery; polarization mismatch (assume co-polarized; see §7).

---

## 2. Equations

### 2.1 Field composition and the beat envelope

Two co-polarized RF fields on the same Rydberg transition r→r′, carrier ω_RF (resonant LO),
signal at ω_RF + δ with relative phase φ:

```
E_tot(t) = E_LO cos(ω_RF t) + E_sig cos((ω_RF + δ) t + φ)
         = Re{ [E_LO + E_sig e^{-i(δt+φ)}] e^{iω_RF t} } ,   (phasor form)
```

Slowly varying envelope (exact):

```
E_env(t) = sqrt( E_LO² + E_sig² + 2 E_LO E_sig cos(δt + φ) )              [V/m]
```

Expansion in u ≡ E_sig/E_LO ≪ 1 (all four coefficients verified numerically to stated order):

```
E_env(t) = E_LO [ 1 + u²/4                          (DC offset, +u²/4)
                + (u − u³/8) cos(δt+φ)              (fundamental)
                − (u²/4) cos(2δt+2φ)                (2nd harmonic)
                + O(u³) harmonics ]
```

So to first order **E_env ≈ E_LO + E_sig cos(δt + φ)**: the LO performs a homodyne-style
projection of the signal field, amplitude *linear* in E_sig and carrying the signal phase φ.
The atoms are driven at instantaneous Rabi frequency

```
Ω(t) = μ_RF E_env(t) / ħ        [rad/s],   μ_RF = ⟨r‖d‖r′⟩ matrix element [C·m]
```

**Validity conditions (all enforced by the API):**

1. `E_sig ≪ E_LO` (linearity; quantified in §2.5 via compression);
2. `δ ≪ atomic response bandwidth` (§2.3), and `δ, Ω ≪ ω_RF` (RWA / envelope adiabaticity);
3. LO resonant with r→r′ (LO detuning shifts the working point — model via OBE, not here);
4. `E_LO = E_LO*`, the optimum maximizing |∂P/∂E| (§2.2). Off-optimum operation is allowed but
   the slope must be evaluated at the actual E_LO.
5. Image ambiguity: signals at ω_RF − δ produce the *same* beat frequency δ with conjugate
   phase. A single-envelope superhet cannot separate the two sidebands; document as receiver
   property (image rejection requires LO stepping or dual-phase acquisition).

### 2.2 Transduction slope

Probe transmission through the cell, P(E) [W], is a static nonlinear function of the RF field
amplitude E (computed by the EIT/OBE module: steady-state ρ_ge(Ω), susceptibility
χ = (2 N₀ d_ge² / (ε₀ ħ Ω_p)) ρ_ge with the sign/normalization convention of the EIT module,
Beer–Lambert `P_out = P_in exp(−k_p L Im χ)`, Doppler-averaged for vapor). For slow envelope
modulation the output power is

```
P(t) = P(E_LO) + κ · E_sig cos(δt + φ) + O(E_sig²) ,
κ ≡ dP/dE |_{E = E_LO}                [W/(V/m)]
```

Chain rule to the OBE-native variable: `κ = (dP/dΩ)·(μ_RF/ħ)` with dP/dΩ in W·s/rad.
The beat amplitude is `P_beat = |κ| E_sig`; the recovered demod phase equals φ (mod the image
ambiguity, §2.1.5). **Optimal LO**: `E_LO* = argmax_E |dP/dE|`. On the microwave-dressed
(Autler–Townes) EIT spectrum this sits on the maximum-slope flank of the dressed line; Jing 2020
found E_LO* = 3.0 mV/cm for their Cs 47D₅/₂→48P₃/₂ configuration (Table 3.2). RydSim must
*compute* E_LO* by 1-D maximization of the OBE-derived |κ(E)| — never hard-code it.

Jing et al.'s own parametrization (arXiv:1902.11063 v1): `P_out(t) = P_s cos(δ_s t + φ_s)` with
`P_s = (ᾱ P̄ / Γ) Ω_s` — i.e. slope ∝ mean transmitted power × dimensionless factor / relaxation
rate. Our κ generalizes this; do not adopt their ᾱ,Γ as separate fit constants.

### 2.3 Transfer function and instantaneous bandwidth — OBE linear response

The envelope approximation of §2.2 holds only for δ slower than the atomic dynamics. Compute the
full transfer function from the linearized master equation. Let ρ evolve under the vectorized
Lindbladian (4-level ladder g–e–r–r′, dimension 16):

```
dρ/dt = L(Ω) ρ ,      L(Ω) = L0 + (Ω − Ω_L) L1     (exactly linear in Ω: H is linear in Ω)
```

where Ω_L = μ_RF E_LO/ħ, L0 = L(Ω_L), and L1 = −(i/ħ)[V_RF, ·] with
V_RF = −(ħ/2)(|r⟩⟨r′| + |r′⟩⟨r|) per unit Rabi frequency (RWA). Steady state: `L0 ρ0 = 0`,
tr ρ0 = 1. Drive `Ω(t) = Ω_L + δΩ cos(δt)` and expand `ρ(t) = ρ0 + Re[ρ1 e^{−iδt}]`:

```
ρ1 = −(L0 + iδ·1)^{−1} L1 ρ0 · δΩ                       (first order in δΩ)
```

Readout functional: `δP(t) = Re[ w† ρ1 e^{−iδt} ]` where w is the linearization of the
Beer–Lambert output about ρ0 (for optically thin: w ∝ −P̄·k_p L·∂Imχ/∂ρ_ge; for thick media
linearize the exponential: same form, weight −P̄·k_p L). Define the **complex transfer function**

```
H(δ) ≡ w† [ −(L0 + iδ·1)^{−1} L1 ρ0 ]        [W·s/rad, response per unit δΩ]
κ_E(δ) = H(δ) · μ_RF/ħ                        [W/(V/m)]  ;  κ ≡ |κ_E(0)|
IBW: f_3dB = δ_3dB/2π  with  |H(δ_3dB)| = |H(0)|/√2
```

Doppler vapor: average the *complex* H(δ; v) over the velocity distribution with
detunings Δ_p → Δ_p − k_p v, Δ_c → Δ_c + k_c v (counter-propagating geometry), then take
moduli. Never average |H|.

**Consistency identity (mandatory self-check):** `|H(0)| = dP/dΩ` from steady-state finite
difference, to <0.1%.

**Fallback single-pole model** (for fast link-budget estimates only, tagged as approximation):

```
H_sp(δ) = κ_Ω / (1 − i δ/Γ_bw)
```

with Γ_bw *fitted* from the full |H(δ)|, not asserted. Physical expectation (Meyer 2020,
VERIFIED quote: "the probe photon scattering rate of the intermediate atomic resonance (of order
10 MHz) is the limiting bandwidth"): Γ_bw is set by the slowest recovery rate of the EIT dark
state — a combination of Ω_c²/Γ_e (dark-state pumping), Γ_e, Rydberg decay/dephasing, and
transit rate. RydSim derives it; papers quoting "IBW" without this analysis are quoting their
electronics.

### 2.4 Noise model — term by term

Every term is expressed as a one-sided PSD of detected optical power S_P [W²/Hz] at Fourier
frequency δ/2π (the beat frequency), then referred to the RF field through the transduction:

```
NEF_x(δ) = sqrt(S_P,x(δ)) / |κ_E(δ)|          [(V/m)/√Hz]     (field-referred noise)
```

Note the denominator carries the roll-off |κ_E(δ)|: detector-side white noise costs more
field-equivalent noise at large δ because the signal transduction falls while the noise floor
does not. Projection noise (item 5) is generated *inside* the transduction and is already
field-referred — do NOT divide it by κ.

**1. Photon shot noise on the probe.** Detected mean optical power P̄ [W], detector quantum
efficiency η, probe frequency ν = c/λ_p:

```
S_P,shot = 2 h ν P̄ / η                       [W²/Hz, one-sided]
NEF_shot = sqrt(2 h ν P̄ / η) / |κ_E(δ)|
```

(Equivalently photocurrent shot noise S_i = 2 e ī with ī = ℜ P̄, ℜ = ηe/hν [A/W]; both forms
must agree — unit self-check.) Worked number: λ_p = 852.347 nm, P̄ = 120 μW, η = 1 →
√S_P = 7.479 pW/√Hz (VERIFIED arithmetic, §6 row B3).

**2. Laser relative intensity noise (RIN).**

```
S_P,RIN(f) = RIN(f) · P̄²   ,   RIN(f) = RIN_w · (1 + f_c/f)      [1/Hz, one-sided]
NEF_RIN(δ) = P̄ · sqrt(RIN(δ/2π)) / |κ_E(δ)|
```

RIN_w = white floor, f_c = 1/f corner. Typical external-cavity diode laser values:
RIN_w ~ 10⁻¹⁴–10⁻¹³ /Hz (−140 to −130 dBc/Hz), f_c ~ 10 kHz–1 MHz
(UNVERIFIED-FROM-MEMORY, vendor-datasheet order of magnitude; **config inputs, never
constants**; self-check: measured RIN input file overrides the model). The superhet's core
advantage is moving detection to δ/2π ≫ f_c, above the 1/f knee — the simulator must reproduce
the NEF(δ) minimum at intermediate δ (cf. Jing's choice δ_s/2π = 150 kHz).

**3. Detector NEP and electronics.**

```
NEF_det = NEP_det(δ) / |κ_E(δ)|
Johnson floor of transimpedance R_f at temperature T_A:
NEP_J = sqrt(4 k_B T_A / R_f) / ℜ            [W/√Hz]
```

Worked number: √(4k_B·300K·50Ω) = 0.910 nV/√Hz (VERIFIED arithmetic). NEP is a config input
(typical Si PD + TIA: 1–20 pW/√Hz, UNVERIFIED-FROM-MEMORY).

**4. Laser frequency noise → intensity noise via the EIT slope.** Probe (and coupling)
frequency noise with one-sided PSD S_ν [Hz²/Hz] converts through the transmission-vs-detuning
slope at the operating point:

```
S_P,fn(f) = (∂P/∂ν_p)² S_ν,p(f) + (∂P/∂ν_c)² S_ν,c(f)        [uncorrelated lasers]
∂P/∂ν = 2π · ∂P/∂Δ   (Δ in rad/s, computed from the steady-state OBE)
NEF_fn(δ) = sqrt(S_P,fn(δ/2π)) / |κ_E(δ)|
```

Modeling assumption (declared): the static slope × the same normalized roll-off is used;
strictly, FM noise enters through a different response port with its own transfer function.
Self-check (§4): time-domain OBE with an FM-modulated probe must reproduce S_P,fn within a
factor 2 at δ ≤ Γ_bw/3, else the module must compute the FM port response the same resolvent way
(replace L1 by ∂L/∂Δ_p). White-frequency-noise ↔ Lorentzian linewidth: S_ν = Δν_L/π
(convention stated so linewidth inputs are unambiguous).

**5. Atom projection noise and the standard quantum limit.** Derivation (Meyer 2020 convention,
VERIFIED quote: SNR = φ/Δφ with φ = Ωτ and Δφ_SQL = 1/√N):

- One shot: N_eff uncorrelated atoms accumulate signal phase φ = Ω_s τ during coherence-limited
  interrogation τ; projection readout gives phase uncertainty Δφ = 1/√N_eff.
- Minimum per-shot Rabi: Ω_min = 1/(τ √N_eff).
- Continuous operation (shots back-to-back, duty 1): n = t/τ independent shots in total time t:

```
Ω_min(t) = 1 / sqrt(N_eff · τ · t)
E_SQL(t) = (ħ/μ_RF) · 1/sqrt(N_eff · τ · t)                    [V/m]  (ħ-convention)
NEF_SQL  = (ħ/μ_RF) · 1/sqrt(N_eff · τ)                        [(V/m)/√Hz]
```

  Equivalently `E_SQL = (ħ/μ_RF)·sqrt(Γ_eff/(N_eff t))` with Γ_eff ≡ 1/τ — the form in the
  assignment; identical.
- Pulsed operation (cold atoms, repetition rate R, interrogation τ ≤ T₂ per shot, duty Rτ ≤ 1):

```
NEF_SQL,pulsed = (ħ/μ_RF) · 1/(τ · sqrt(N_eff · R))            [(V/m)/√Hz]
```

  Continuous and pulsed forms agree when Rτ = 1 (duty penalty factor sqrt(1/(Rτ)) otherwise —
  verified numerically: 51.1× for the Sci. Adv. parameters below).

**Which N.** `N_eff = n₀ · V_int · f_vel · f_state`: n₀ vapor density; V_int the probe–coupling
overlap volume (∫ I_p I_c dV normalization, not the cell volume); f_vel the fraction of the
velocity distribution actually contributing to the Doppler-averaged EIT signal (compute from the
velocity-resolved weight |∂H/∂v|-integrand of §2.3, NOT a folklore "1%"); f_state the fraction
coherently participating (from ρ0: weight of the field-sensitive coherence). Cold atoms:
f_vel = 1 and N_eff = trapped-atom number in the beams (Sci. Adv. 2024 uses N = 5.2×10⁵ this
way, VERIFIED). Every reported SQL must print the N_eff, τ, and convention used.

**Which coherence time.** τ = 1/Γ₂ where Γ₂ is the decay rate of the coherence that accumulates
the signal phase — for the superhet, the LO-dressed Rydberg pair:
`Γ₂ = ½(Γ_r + Γ_r′) + γ_transit + γ_collision + γ_laser(coupling)`. In a driven-EIT continuous
readout the mapping "measurement ↔ Ramsey" carries an O(1) prefactor ambiguity that published
work does not agree on — see convention warning below.

**Convention warning (factor 2π, explicit).** The Sci. Adv. 2024 paper's SQL formula is, as
printed (PMC full text, re-extracted cleanly this session): **`E_SQL·√Hz = h/(μ_MW √(N T₂))`
with Planck's h (not ħ) and T₂ the EIT coherence time** — i.e. the h-convention is definitively
theirs. It reproduces their stated `NEF_at = 3.7 nV/cm/√Hz` for μ = 1218 ea₀, N = 5.2×10⁵ with
T₂ = 57.8 ns (back-inferred; the paper states T₂ ≈ 100 ns, which would give 2.81 nV/cm/√Hz —
consistent at the "approximately" level). The identical 3.7 also follows from our **ħ-pulsed**
form with τ = 3.83 μs at R = 100 Hz (back-inferred; plausible cold-atom Rydberg coherence).
Jing arXiv v1 prints yet another prefactor: `E_QPNL = (√2 ħ/2μ_r)·1/√(N_a τ_c)` (ħ-convention
with a 1/√2; scope of the radical ambiguous in extraction). Published conventions genuinely
differ at the O(2π) level.
**RydSim default: ħ-convention (Meyer-consistent).** The API exposes `convention={"hbar","h"}`
and the benchmark pins numbers per convention. If your implementation disagrees with a published
SQL by ≈ 2π, check this first; do not fudge τ to compensate.

**6. Total.**

```
NEF_tot(δ) = sqrt( Σ_x NEF_x(δ)² )        (uncorrelated-sum assumption, declared)
```

Known limitation: Sci. Adv. 2024 *measured* a correlation coefficient r = −0.78 between atomic
and photon shot noise (VERIFIED quote:
`NEF_in = √(NEF_at² + 2r·NEF_at·NEF_ph + NEF_ph² + NEF_pd²)`). RydSim v1 does not model this
correlation (requires Heisenberg–Langevin treatment); flag in output when NEF_at and NEF_shot
are within 3× of each other. Discrepancy flag (unresolved, stated honestly): recombining their
extracted components {at 3.7, ph 9.1, pd 3.0, ext 3.2} nV/cm/√Hz with r = −0.78 gives
7.95 nV/cm/√Hz, not their headline 10.0 — our reading of which number is which is likely
imperfect; re-verify against the published PDF before using the component values as truth.

### 2.5 Receiver metrics

**NEF and minimum detectable field.** RydSim reports `NEF ≡ sqrt(S_E(δ))`, the one-sided
amplitude spectral density of field-equivalent noise, units (V/m)/√Hz (displayed nV/cm/√Hz).
With phase-locked single-quadrature detection integrated for time t (ENBW = 1/(2t), §2.7):

```
E_min(t) = NEF / sqrt(t)          (SNR = 1 definition)
```

**SNR** of a coherent signal E_sig at beat δ in detection bandwidth B (one-sided ENBW):

```
SNR = (|κ_E(δ)| E_sig)² / (2 · S_P,tot(δ) · B)         [power ratio]
    = E_sig² / (2 · NEF_tot(δ)² · B)
```

(Amplitude convention: E_sig is a field *amplitude*; the ½ is the sinusoid's mean-square factor.
Single-quadrature, phase-known: E_min(B) = NEF·√(2B); the two conventions coincide at B = 1/(2t).
Dual-quadrature |I+iQ| magnitude detection with unknown phase costs a further √2.)

**1-dB compression (upper end of dynamic range).** Two stacked mechanisms:

1. *Envelope arithmetic* (species-independent): exact fundamental Fourier coefficient of
   E_env(u) compresses by 1 dB at **u = E_sig/E_LO = 0.8763** (computed exactly this session;
   the small-u series coefficient is u − u³/8).
2. *Atomic saturation* (usually dominant): curvature of P(E). Cubic fit
   `P(E_LO+e) ≈ P̄ + c₁e + c₂e² + c₃e³` (c₁ = κ, c₂ = ½P″, c₃ = ⅙P‴ by finite differences of
   the OBE steady state) gives the memoryless-nonlinearity result
   `E_1dB = sqrt(0.145·|c₁/c₃|)`. RydSim must ALSO find E_1dB by brute force: sweep E_sig,
   demodulate the fundamental, find the −1 dB point (this captures both mechanisms and any
   dynamic effects at once). Report the sweep result as authoritative.

```
Dynamic range (linear):  DR(B) = 20 log10( E_1dB / E_min(B) )      [dB]
```

Reference point: Jing arXiv v1 states 90 dB linear dynamic range (extraction from ar5iv;
LITERATURE-RECALL — the published version and the dedicated follow-up
"Linear dynamic range of a Rydberg-atom microwave superheterodyne receiver" (2023) should be
checked before this number is used as a benchmark).

**Two-tone SFDR.** Two equal signals A at δ₁, δ₂ produce IM3 at 2δ₁−δ₂ with amplitude
(3/4)|c₃|A³ (products fall inside the band — the atomic nonlinearity is at the transduction,
before any filtering). Input intercept: `A_IIP3 = sqrt(4|c₁|/(3|c₃|))` (field units). With field
levels in dB re 1 V/m, `L(E) = 20 log10(E/1 V/m)`:

```
SFDR(B) = (2/3) · [ L(A_IIP3) − L(E_min(B)) ]        [dB]
```

**Equivalent noise temperature / noise figure (standard-antenna comparison).** A plane wave of
field amplitude E carries time-averaged power flux `S = E²/(2 η₀)`, η₀ = 376.7303 Ω. A matched
reference antenna of gain G captures `P = A_e S` with `A_e = G λ²/(4π)`. The field-noise floor
NEF maps to an input-referred noise power density and thence a noise temperature:

```
T_eq = A_e · NEF² / (2 η₀ k_B) = G λ² NEF² / (8π η₀ k_B)     [K]
NF   = 10 log10(1 + T_eq/290 K)                              [dB]
```

Convention pinned by cross-check: with G = 1.64 (half-wave dipole) and NEF as *amplitude* per
√Hz, this formula reproduces the published Sci. Adv. value: 10.0 nV/cm/√Hz at 36.9 GHz →
T_eq = 828 K computed vs 830 K published (0.2%; VERIFIED — this simultaneously validates the
G and the E²/2η₀ amplitude conventions). Same formula on Jing 2020: 55 nV/cm/√Hz at 6.94 GHz →
T_eq = 7.08×10⁵ K, NF = 33.9 dB — i.e. vapor-cell superhets are far above thermal-antenna noise
floors, the central sober conclusion of Meyer 2020 (their passive-dipole thermal field,
VERIFIED quote: `ℰ_Dipole = sqrt(8 k_B T |Z_a| / ℓ²)`).

**Link budget.** Given transmit EIRP and range r: incident flux S = EIRP/(4πr²), field amplitude
E_sig = sqrt(2η₀S); then SNR from above. Received-power bookkeeping in dBm for RF engineers:
`P_dBm(E) = 10 log10( E² A_e / (2η₀ · 1 mW) )` with the SAME (G, amplitude) convention — never
mix conventions inside one budget.

### 2.6 Sensitivity–bandwidth tradeoff

With the computed transfer function: `NEF_tot(δ) = sqrt(S_P,tot(δ))/|κ_E(δ)|`. For a single-pole
roll-off and white detector-side noise this is exactly

```
NEF(δ) = NEF(0) · sqrt(1 + (δ/Γ_bw)²)
```

so field sensitivity degrades ∝ δ beyond the IBW. The physically meaningful tradeoff is against
control parameters: increasing Ω_c speeds dark-state dynamics (Γ_bw ↑) but dilutes the slope
(κ ↓). **Simulator experiment (deliverable, not asserted law):** sweep Ω_c over ×10 at fixed
optimized E_LO; record (NEF(0), f_3dB); test the hypothesis that the product NEF(0)·f_3dB is
invariant to within a factor ≈3. Publish the curve; if the product drifts more, that IS the
result — report it.

### 2.7 Digital demodulation and reporting conventions (precision bookkeeping)

Detected signal `v(t) = a cos(δt + φ) + n(t)`, a = |κ_E(δ)|E_sig, one-sided noise PSD S_v
flat near δ.

**I/Q demodulation:** `I = LPF[2 v(t) cos δt]`, `Q = −LPF[2 v(t) sin δt]` → I + iQ = a e^{iφ} + ñ.
Multiplication folds S_v(δ−f) + S_v(δ+f) = 2S_v to baseband; after a low-pass of one-sided ENBW
B_n each quadrature has noise variance **σ² = 2 S_v B_n**.

**ENBW table (one-sided, verified conventions):**

| Filter | ENBW |
|---|---|
| Boxcar average, duration t | 1/(2t) |
| 1-pole RC lock-in, time constant τ_LI (6 dB/oct) | 1/(4 τ_LI) |
| 2-pole (12 dB/oct) | 1/(8 τ_LI) |
| FFT bin, rectangular window, record length T | 1/T |
| FFT bin, Hann | 1.5/T |

(Lock-in rows: SRS SR830 manual convention, LITERATURE-RECALL; Hann: Harris 1978,
LITERATURE-RECALL.)

**The chain that produces a published "nV/cm/√Hz":**

```
NEF = sqrt(S_v(δ)) / |κ_E(δ)|              (definition; what RydSim reports)
E_min(t) = NEF/√t                          (boxcar, phase-known, SNR=1; B_n = 1/(2t))
sigma_amplitude per shot: σ_a = NEF·κ_E·sqrt(2 B_n)
```

**Where papers are sloppy — the four ±3 dB traps (each must be an explicit code parameter):**

1. one-sided vs two-sided PSD (factor 2 in S);
2. amplitude vs RMS field in the sensitivity figure (factor √2; the antenna cross-check in §2.5
   pins the *amplitude* convention for both cited papers);
3. B = 1 Hz "RBW" vs t = 1 s integration (ENBW of 1 s boxcar is 0.5 Hz, not 1 Hz: factor √2);
4. spectrum-analyzer noise-marker practice: log-power averaging of Gaussian noise under-reads
   by 2.51 dB (add it back), and RBW ≠ ENBW (Gaussian RBW filters: ENBW ≈ 1.065×RBW)
   (Keysight AN-150 conventions; LITERATURE-RECALL).

Spectrum-analyzer route to sensitivity (Jing 2020 used RBW = 1 Hz at δ_s/2π = 150 kHz):
`NEF = E_sig · 10^{−SNR_dB/20} / sqrt(ENBW)` with SNR_dB read peak-to-noise-floor and corrected
per items 3–4. RydSim's synthetic-measurement mode must implement BOTH the I/Q route and the
SA route and demonstrate they agree to <0.5 dB on the same time series (self-validation).

---

## 3. Constants and parameters

### 3.1 Fundamental constants (as used in all worked numbers)

| Constant | Value | Units | Source | Confidence |
|---|---|---|---|---|
| h | 6.62607015e-34 | J·s | SI exact (CODATA 2018) | VERIFIED |
| ħ | 1.054571817e-34 | J·s | CODATA 2018 (derived) | VERIFIED |
| e | 1.602176634e-19 | C | SI exact | VERIFIED |
| k_B | 1.380649e-23 | J/K | SI exact | VERIFIED |
| c | 299792458 | m/s | SI exact | VERIFIED |
| a₀ | 5.29177210903e-11 | m | CODATA 2018 | LITERATURE-RECALL (CODATA 2022 differs at 1e-10 rel.; immaterial) |
| e·a₀ | 8.478353626e-30 | C·m | derived | VERIFIED (arithmetic) |
| η₀ (Z₀) | 376.730313668 | Ω | CODATA 2018 | LITERATURE-RECALL (same caveat) |

### 3.2 Jing et al., Nat. Phys. 16, 911 (2020) — Cs vapor-cell superhet

| Parameter | Value | Source | Confidence |
|---|---|---|---|
| Sensitivity | 55 nV cm⁻¹ Hz⁻¹ᐟ² (published); arXiv v1: 55.5 = −145 dBV cm⁻¹ Hz⁻¹ᐟ² (dB↔linear verified) | Nat. Phys. abstract; arXiv v1 | VERIFIED (both) |
| Minimum detectable field | 780 pV/cm (published); arXiv v1: 2.4 nV/cm | Nat. Phys. abstract; arXiv v1 | VERIFIED (both; the improvement is a v1→published change, not an error) |
| Implied integration time (55/0.78)² | ≈ 4.97×10³ s | derived this session | VERIFIED (arithmetic) |
| Ladder | 6S₁/₂ F=4 → 6P₃/₂ F=5 → 47D₅/₂; RF 47D₅/₂→48P₃/₂ | arXiv:1902.11063 v1 | VERIFIED (v1 text) |
| Probe | 852 nm, 120±4 μW, waist 1.70±0.04 mm; Ω_p = 5.7±0.6 MHz (as quoted) | arXiv v1 | VERIFIED (v1; published version may differ) |
| Coupling | 510 nm, 34±1 mW, waist 2.00±0.05 mm; Ω_c = 0.97±0.12 MHz (as quoted) | arXiv v1 | VERIFIED (v1) |
| Cell length | 5 cm; density N₀ = 4.89×10¹⁰ cm⁻³ | arXiv v1 | VERIFIED (v1) |
| RF frequency | 6.94 GHz | arXiv v1 | VERIFIED (v1) |
| μ_RF(47D₅/₂→48P₃/₂) | 1443.450 e·a₀ = 1.2238e-26 C·m (radial ME 2946.512 e·a₀) | secondary (survey arXiv:2412.05554-family citing Jing) | LITERATURE-RECALL — recompute from module 02 matrix elements as self-check |
| Optimal LO | E_LO* = 3.0 mV/cm | arXiv v1 | VERIFIED (v1) |
| Ω_L consistency | μ·(3.0 mV/cm)/ħ = 2π×5.54 MHz, but v1 text extraction says "Ω_L = 7.9 MHz" (would need 4.28 mV/cm) | derived | **INCONSISTENT — flagged**, resolve against published PDF; RydSim recomputes Ω from (μ, E) always |
| Beat frequency δ_s/2π | 150.000 kHz; SA RBW 1 Hz | arXiv v1 | VERIFIED (v1) |
| Quantum-projection-noise limit (their estimate) | ~700 pV cm⁻¹ Hz⁻¹ᐟ², formula as printed E_QPNL = (√2 ħ/2μ_r)·1/√(N_a τ_c) | arXiv v1 (re-extracted 2026-08-10) | VERIFIED (v1); radical scope ambiguous in extraction — implied N_a·τ_c ≈ 7.6×10⁻³ s under the per-√Hz reading |
| Dominant noise (their attribution) | transit noise + laser frequency noise above 100 kHz | arXiv v1 | VERIFIED (v1) |
| Linear dynamic range | 90 dB | arXiv v1 (re-extracted 2026-08-10) | VERIFIED (v1); published + 2023 follow-up still unchecked |

### 3.3 "Approaching the SQL…", Sci. Adv. 10, eads0683 (2024); arXiv:2307.15617 — cold Rb-87

| Parameter | Value | Source | Confidence |
|---|---|---|---|
| Sensitivity | 10.0 nV cm⁻¹ Hz⁻¹ᐟ² at 100 Hz repetition | PMC full text | VERIFIED |
| Factor above SQL | 2.6 (their SQL "NEF_at" = 3.7 nV cm⁻¹ Hz⁻¹ᐟ²; note 10.0/3.7 = 2.70) | PMC full text | VERIFIED |
| SQL formula as printed | E_SQL·√Hz = h/(μ_MW √(N T₂)) — **h-convention**, T₂ = EIT coherence time | PMC full text (re-extracted 2026-08-10) | VERIFIED |
| Minimum detectable field | 540 pV/cm at stated T′ = 420 s (NEF/√420 = 488 pV/cm; ratio 1.107 → estimator/ENBW convention, flagged) | PMC + derived | VERIFIED (both numbers as printed; convention gap noted) |
| Atom number N | 5.2×10⁵ laser-cooled ⁸⁷Rb | PMC | VERIFIED |
| States / RF | 5S₁/₂ F=2 ground; RF 39D₅/₂↔40P₃/₂ at 36.9 GHz | PMC | VERIFIED |
| μ_RF | 1218 e·a₀ = 1.0327e-26 C·m | PMC | VERIFIED |
| LO Rabi | Ω_L/2π = 2.0 MHz; probe ≈5 μW (optimized 7.6 μW), Ω_p(0)/2π = 4.6 MHz | PMC (re-extracted 2026-08-10) | VERIFIED |
| Timing | rep 100 Hz, detection window 2.7 ms, 1-ms control pulses | PMC | VERIFIED |
| Implied T₂ (their h-convention SQL formula) | 57.8 ns back-inferred; paper states T₂ ≈ 100 ns (would give 2.81 nV/cm/√Hz) | back-inference this session | UNVERIFIED (inference; consistent at the "approximately" level) |
| Implied τ (ħ-pulsed SQL) | 3.83 μs | back-inference | UNVERIFIED (inference) |
| Noise components (at, ph, pd, ext) | {3.7, 9.1, 3.0, 3.2} nV cm⁻¹ Hz⁻¹ᐟ², correlation r = −0.78 | PMC | VERIFIED as extracted; **quadrature recombination gives 7.95 ≠ 10.0 — mapping suspect, see §2.4.6** |
| Equivalent noise temperature | 830 K (lossless-antenna comparison) | PMC | VERIFIED; independently reproduced: 828 K with G = 1.64 |

### 3.4 Meyer, Castillo, Cox, Kunz, J. Phys. B 53, 034001 (2020)

| Item | Statement | Confidence |
|---|---|---|
| SQL primitive | SNR = φ/Δφ, φ = Ωτ, Δφ_SQL = 1/√N | VERIFIED (ar5iv quote) |
| DC-regime limit | ℰ_Rydberg = N^{−1/4}·sqrt(2/(ατ)) (quadratic Stark → N^{−1/4}); bias-field linearized: ℰ ≈ 1/(α τ √N E_bias) | VERIFIED (ar5iv quote) |
| Antenna thermal field | ℰ_Dipole = sqrt(8 k_B T |Z_a|/ℓ²) | VERIFIED (ar5iv quote) |
| Bandwidth ceiling | probe-photon scattering rate of intermediate state, order 10 MHz | VERIFIED (ar5iv quote) |

### 3.5 Default technical-noise inputs (ALL are config inputs, never constants)

| Input | Default | Confidence |
|---|---|---|
| RIN white floor | 1e-14 /Hz (−140 dBc/Hz) | UNVERIFIED-FROM-MEMORY (vendor-typical); expect ±10 dB spread |
| RIN 1/f corner | 100 kHz | UNVERIFIED-FROM-MEMORY |
| Detector NEP | 5 pW/√Hz | UNVERIFIED-FROM-MEMORY |
| η (quantum efficiency, Si @852 nm) | 0.85 | UNVERIFIED-FROM-MEMORY |
| Probe/coupling linewidths | 100 kHz / 100 kHz (white-noise equivalent S_ν = Δν/π) | UNVERIFIED-FROM-MEMORY |

---

## 4. Numerical method and pitfalls

1. **Slope κ:** central difference of the OBE steady state, step ΔE = 10⁻³·E_LO, one Richardson
   extrapolation; verify linearity by halving ΔE (change < 10⁻⁴ relative). Same machinery with
   steps {±ΔE, ±2ΔE} yields c₂, c₃ for compression/SFDR (use ΔE = 10⁻²·E_LO for the third
   derivative; check plateau in ΔE).
2. **Resolvent (L0 + iδ)⁻¹:** L0 (16×16 complex for the 4-level ladder) is singular at δ = 0
   (zero mode = trace conservation; left null vector is the trace functional). The RHS L1ρ0 is
   traceless (L1 is a commutator), so the system is consistent: solve with `lstsq` /
   pseudo-inverse at |δ| < 10⁻³·Γ_e, direct `solve` elsewhere. Never eigendecompose L0 for the
   steady state — use SVD null space with the trace-normalization row appended.
3. **Doppler averaging:** Gauss–Hermite or trapezoid over v ∈ ±4 v_p (v_p = √(2k_BT/m)),
   ≥ 201 points, convergence: doubling the grid changes |H| by < 0.1%. Average complex H.
   Residual two-photon Doppler uses k_p − k_c (counter-propagating) — sign per geometry module.
4. **Self-validation A vs B (mandatory test):** time-domain integration of the full OBE with
   Ω(t) = Ω_L + δΩ cos δt (δΩ/Ω_L = 10⁻³), ≥ 20 envelope periods after discarding a transient of
   10/Γ_min; FFT the transmission; compare fundamental amplitude & phase against the resolvent
   H(δ). Agreement < 1% amplitude, < 1° phase over δ ∈ [10⁻², 5]×Γ_bw. Stiff integrator:
   `scipy.integrate.solve_ivp(method="BDF", rtol=1e-9, atol=1e-12)` on the vectorized ρ.
5. **Envelope vs two-tone check:** at δ beyond IBW the envelope picture fails; the module must
   optionally drive the OBE with the true two-tone Hamiltonian (bichromatic RWA at ω_RF:
   Ω(t) complex = Ω_L + Ω_s e^{−i(δt+φ)}) and compare. Divergence between envelope-Ω(t)=|·| and
   complex two-tone drive quantifies the RWA-envelope error; report it above δ > Γ_bw.
6. **Synthetic time series:** sample rate f_s ≥ 32·δ/2π; duration ≥ 200 beat periods for < 1%
   amplitude bias. White noise generation: per-sample variance σ² = S_v·f_s/2 (one-sided PSD
   convention — document in code; `scipy.signal.welch` default is one-sided for real input,
   consistent). Fixed RNG seed in tests.
7. **Units discipline:** rad/s internally; the ×2π bug is the most common failure of this
   module class. Enforce via naming convention and a unit self-test (photocurrent-form vs
   optical-power-form shot noise must agree to machine precision, §2.4.1).
8. **Optimization of E_LO:** bounded scalar maximization of |κ(E)| (`scipy.optimize.
   minimize_scalar`, bracket from a coarse 30-point log sweep); beware double-peaked |κ(E)| on
   AT-split spectra — always coarse-sweep first.
9. **Numerical floor:** κ can be ~10⁻⁴ W/(V/m) (Jing-scale) down to ~10⁻⁹ for weak probes —
   normalize P by P_in inside the optimizer to keep conditioning sane.

---

## 5. Recommended Python API (numpy-vectorized, Python 3.11, scipy only)

```python
# rydsim/superhet.py
from dataclasses import dataclass, field
from typing import Callable, Literal
import numpy as np

@dataclass(frozen=True)
class SuperhetOperatingPoint:
    """Working point of the atomic superheterodyne."""
    E_LO: float                 # LO field amplitude [V/m], resonant with r->r'
    mu_rf: float                # RF transition dipole matrix element [C*m]
    omega_rf: float             # RF carrier [rad/s] (validity checks only)
    P_probe_in: float           # probe power at cell entrance [W]
    P_probe_det: float          # mean probe power at detector [W] (after cell/optics)
    # ladder/EIT configuration object from the EIT module:
    eit: "EITConfig"            # states, Omegas, Gammas, detunings, N0, L, T, geometry

@dataclass(frozen=True)
class NoiseInputs:
    """Technical-noise configuration. ALL defaults are UNVERIFIED typicals (spec 3.5)."""
    eta_det: float = 0.85               # detector quantum efficiency
    rin_white: float = 1e-14            # RIN floor [1/Hz]
    rin_corner_hz: float = 1e5          # 1/f corner [Hz]
    nep_det: float = 5e-12              # detector NEP [W/rtHz]
    S_nu_probe: Callable[[np.ndarray], np.ndarray] | None = None   # [Hz^2/Hz] one-sided
    S_nu_coupling: Callable[[np.ndarray], np.ndarray] | None = None
    # projection-noise bookkeeping (every field REQUIRED to be explicit — no silent defaults):
    N_eff: float | None = None          # effective atom number (spec 2.4.5 definition)
    tau_coh: float | None = None        # coherence/interrogation time [s]
    sql_convention: Literal["hbar", "h"] = "hbar"
    sql_mode: Literal["continuous", "pulsed"] = "continuous"
    rep_rate: float | None = None       # [Hz], pulsed mode only

@dataclass(frozen=True)
class NoiseBudget:
    delta: np.ndarray           # beat angular frequencies [rad/s]
    nef_shot: np.ndarray        # each [(V/m)/rtHz], same shape as delta
    nef_rin: np.ndarray
    nef_det: np.ndarray
    nef_freq: np.ndarray
    nef_sql: float              # field-referred, delta-independent in this model
    nef_total: np.ndarray
    kappa_E: np.ndarray         # complex transduction |slope| per delta [W/(V/m)]

def transduction_slope(op: SuperhetOperatingPoint, *, dE_rel: float = 1e-3,
                       order: int = 3) -> tuple[float, float, float]:
    """(c1, c2, c3) = (dP/dE, d2P/dE2 / 2, d3P/dE3 / 6) at E_LO by Richardson central
    differences of the Doppler-averaged OBE steady state. c1 in W/(V/m).
    Contract: halving dE_rel changes c1 by < 1e-4 relative, else raises ConvergenceError."""

def optimize_lo(op_template: SuperhetOperatingPoint, E_bounds: tuple[float, float],
                n_coarse: int = 30) -> tuple[float, float]:
    """(E_LO_star, kappa_max). Coarse log-sweep then bounded refine. Never hard-code E_LO*."""

def linear_response(op: SuperhetOperatingPoint, delta: np.ndarray,
                    doppler: bool = True) -> np.ndarray:
    """Complex H(delta) [W*s/rad] via rho1 = -(L0 + i*delta)^-1 L1 rho0, Doppler-averaged
    (complex average). Contract: H(0) matches transduction_slope c1*hbar/mu_rf to <0.1%.
    Vectorized over delta (batched LU factorizations)."""

def ibw_3db(op: SuperhetOperatingPoint) -> float:
    """f_3dB [Hz] from |H|: bisection on |H(delta)| = |H(0)|/sqrt(2)."""

def sql_nef(mu_rf: float, N_eff: float, tau_coh: float, *,
            convention: Literal["hbar", "h"] = "hbar",
            mode: Literal["continuous", "pulsed"] = "continuous",
            rep_rate: float | None = None) -> float:
    """Projection-noise NEF [(V/m)/rtHz]. continuous: (hbar/mu)/sqrt(N*tau);
    pulsed: (hbar/mu)/(tau*sqrt(N*R)). 'h' convention multiplies by 2*pi.
    Prints/returns provenance record: (N_eff, tau, convention, mode) — spec 2.4.5."""

def noise_budget(op: SuperhetOperatingPoint, noise: NoiseInputs,
                 delta: np.ndarray) -> NoiseBudget:
    """Assemble spec 2.4 term-by-term. Warns if nef_sql and nef_shot within 3x
    (uncorrelated-sum assumption strained; Sci.Adv. 2024 r=-0.78)."""

def e_min(nb: NoiseBudget, t_int: float | np.ndarray, delta_idx: int = 0) -> np.ndarray:
    """E_min(t) = NEF_total/sqrt(t) (phase-known single-quadrature, ENBW=1/(2t))."""

def compression_sweep(op: SuperhetOperatingPoint, E_sig: np.ndarray, delta: float,
                      *, engine: Literal["envelope", "two_tone"] = "two_tone"
                      ) -> tuple[np.ndarray, float]:
    """Fundamental beat amplitude vs E_sig via time-domain OBE + demod; returns
    (amplitudes, E_1dB). Authoritative over the cubic-fit estimate."""

def receiver_metrics(op: SuperhetOperatingPoint, noise: NoiseInputs, *,
                     bandwidth_hz: float = 1.0, gain_ref: float = 1.64) -> "ReceiverReport":
    """Dataclass: nef0, f_3db, E_1dB, DR_dB, A_iip3, SFDR_dB, T_eq_K, NF_dB.
    T_eq = gain_ref * lambda^2 * NEF^2 / (8*pi*eta0*kB); NF = 10log10(1+T_eq/290)."""

def demodulate_iq(t: np.ndarray, v: np.ndarray, delta: float, *,
                  filt: Literal["boxcar", "rc1", "rc2"] = "boxcar",
                  t_avg: float | None = None) -> tuple[float, float, float]:
    """(I, Q, enbw_hz) per spec 2.7 ENBW table. I+iQ = a*exp(i*phi)+noise."""

def nef_from_timeseries(t: np.ndarray, v: np.ndarray, delta: float, kappa_E: float,
                        E_sig_cal: float, *, method: Literal["iq", "sa"] = "iq") -> float:
    """Measured NEF from a synthetic record, by BOTH the I/Q route and the
    spectrum-analyzer route (Welch PSD + ENBW + 2.51 dB log-average correction when
    emulating a log detector). Contract: the two methods agree < 0.5 dB (spec 2.7)."""

def sensitivity_bandwidth_scan(op_template: SuperhetOperatingPoint, noise: NoiseInputs,
                               Omega_c_grid: np.ndarray) -> "TradeoffCurve":
    """Spec 2.6 experiment: per Omega_c re-optimize E_LO, return (nef0, f_3db, product)."""
```

---

## 6. Validation benchmarks (→ pytest)

All "expected" values recomputed this session unless noted. Tolerances are on the simulator
output; "arithmetic fixture" rows validate code plumbing, "physics" rows validate the model.

| # | Quantity | Expected | Tolerance | Source | Confidence |
|---|---|---|---|---|---|
| B1 | Envelope fundamental coefficient, u = 0.1: a₁/E_sig − 1 | −1.25×10⁻³ (= −u²/8) | ±1×10⁻⁴ | analytic identity, verified numerically | VERIFIED |
| B2 | Envelope 2δ harmonic, u = 0.1 | 2.50×10⁻³·E_LO (= u²/4) | ±5% | same | VERIFIED |
| B3 | √(2hνP), 852.347 nm, 120 μW, η=1 | 7.479×10⁻¹² W/√Hz | ±0.5% | CODATA arithmetic | VERIFIED |
| B4 | Shot-noise identity: optical-power form vs photocurrent form | equal | 1×10⁻¹² rel | internal | VERIFIED (definition) |
| B5 | Envelope-only 1-dB compression point | E_sig/E_LO = 0.876 | ±0.01 | exact Fourier coefficient, computed | VERIFIED |
| B6 | SQL, h-convention (their printed formula h/(μ√(N T₂))): μ=1218 ea₀, N=5.2×10⁵, T₂=57.8 ns | 3.70 nV/cm/√Hz | ±3% | Sci. Adv. 2024 formula + stated SQL; T₂ back-inferred | VERIFIED formula and number / UNVERIFIED T₂ provenance |
| B7 | SQL, ħ-pulsed: same μ,N; τ=3.83 μs, R=100 Hz | 3.70 nV/cm/√Hz | ±3% | equivalent reading | VERIFIED (arithmetic) |
| B8 | Duty-cycle relation: continuous ħ SQL with τ=3.83 μs = pulsed/51.1 | 0.0724 nV/cm/√Hz | ±3% | derived | VERIFIED (arithmetic) |
| B9 | Ratio achieved/SQL, Sci. Adv. | 10.0/3.7 = 2.70 (published "2.6") | ±10% | PMC full text | VERIFIED |
| B10 | T_eq(10.0 nV/cm/√Hz, 36.9 GHz, G=1.64) | 828 K (published 830 K) | ±2% | §2.5 formula vs PMC | VERIFIED |
| B11 | T_eq(55 nV/cm/√Hz, 6.94 GHz, G=1.64); NF | 7.08×10⁵ K; 33.9 dB | ±5% | derived (no published cross-value) | VERIFIED (arithmetic) |
| B12 | Jing consistency: t implied by (55 nV/cm/√Hz, 780 pV/cm) | 4.97×10³ s ∈ [4×10³, 6×10³] | — | Nat. Phys. pair | VERIFIED |
| B13 | Sci. Adv. consistency: t implied by (10.0, 0.54 nV/cm) vs their stated T′ | 343 s implied; published T′ = 420 s (ratio 1.107, ENBW convention) | window [250, 450] | PMC pair + stated T′ | VERIFIED |
| B14 | SQL scaling: d ln NEF_SQL/d ln N and /d ln t | −0.500 each | ±1×10⁻⁶ | contract | VERIFIED (definition) |
| B15 | Resolvent vs time-domain OBE: |H(δ)| and arg H, 4-level Cs fixture, δ ≤ 2π×10 MHz | agree | <1% ampl., <1° phase | self-validation §4.4 | VERIFIED (method) |
| B16 | H(0) vs finite-difference slope | equal | <0.1% | self-validation §2.3 | VERIFIED (method) |
| B17 | Optimal LO, Jing fixture (their cell/beam params, computed μ) | E_LO* ∈ [1.5, 6.0] mV/cm (published 3.0) | factor 2 window | Nat. Phys./arXiv v1 | LITERATURE-RECALL (model-dependent) |
| B18 | End-to-end demod: synthetic record with known NEF, IQ vs SA routes | recover NEF; routes agree | ±5% of truth; <0.5 dB between routes | §2.7 contract | VERIFIED (method) |
| B19 | Jing total sensitivity reproduction: full noise budget with §3.5 defaults + their κ | NEF_tot ∈ [18, 165] nV/cm/√Hz (×3 window on 55) | factor 3 | Nat. Phys. | LITERATURE-RECALL (technical-noise inputs uncertain) |
| B20 | Jing QPNL cross-check: N_eff from §2.4.5 recipe with their cell params, τ = 1/Γ₂ | within ×3 of their 700 pV/cm/√Hz | factor 3 | arXiv v1 | LITERATURE-RECALL |

pytest note: B1–B14 are deterministic (no OBE) and must pass exactly; B15–B18 exercise the OBE
engine on a fixed 4-level Cs fixture (params from Table 3.2, μ from module 02); B19–B20 are
order-of-magnitude physics gates — a failure means a real modeling problem, not a tolerance to
widen. If B19/B20 fail by >3×, the first suspects are (in order): N_eff bookkeeping, ×2π unit
slips, Doppler weighting of κ, and the SQL convention flag.

---

## 7. Known limitations / where the model breaks down

1. **Uncorrelated noise sum.** Measured atomic–photon noise correlation (r = −0.78, Sci. Adv.
   2024) is not modeled; near the SQL our NEF_tot can err by ~√2 either way. Requires
   Heisenberg–Langevin input–output treatment (future module).
2. **SQL prefactor convention.** Published conventions genuinely differ at O(2π): Sci. Adv.
   2024 prints the h-convention h/(μ√(N T₂)) (verified as printed); Jing v1 prints an
   ħ-convention with a 1/√2 prefactor; Meyer 2020 works in the ħ/Ramsey picture. RydSim
   defaults to ħ (Meyer-consistent) and stamps every SQL output with (N_eff, τ, convention,
   mode) — never bare numbers. Continuous driven-EIT readout carries a further O(1) mapping
   ambiguity onto the Ramsey picture.
3. **Envelope model fails for δ ≳ IBW** and for LO-signal frequency offsets approaching Ω_L or
   the AT splitting; the two-tone OBE engine (§4.5) is the fallback, at ~100× cost.
4. **Image response** (ω_RF ± δ degenerate) is inherent to single-envelope detection; the
   simulator reproduces it, and any "SSB sensitivity" claims must state the ±3 dB bookkeeping.
5. **Co-polarized plane-wave fields assumed.** Polarization mismatch, standing waves, cell-wall
   RF scattering/Fabry–Pérot effects (real vapor cells: ±3 dB field inhomogeneity) are not in
   this module — field-delivery module's job; here they enter only as an E_LO/E_sig calibration
   uncertainty.
6. **Transit and collision dephasing** enter as scalar rates in Γ₂; velocity-class-resolved
   transit dynamics (re-entering ground-state atoms) are approximated. Jing's dominant "transit
   noise" (their attribution) is therefore modeled only at the PSD-input level (S_ν-like input),
   not ab initio.
7. **No saturation of the photodetector / ADC chain**; electronic 1-dB compression must be
   checked separately in hardware-emulation configs.
8. **LO amplitude/phase noise not modeled** (assumed ideal synthesizer): real LO AM noise maps
   directly through κ like signal — add S_AM,LO·E_LO² as an extra PSD input if the synthesizer
   spec matters; LO PM noise cancels to first order at δ≪IBW for amplitude readout but limits
   phase recovery.
9. **Parameter provenance gaps** flagged in §3: the Jing headline numbers are reconciled
   (v1 55.5 nV/cm/√Hz and 2.4 nV/cm improved to published 55 and 780 pV/cm — a real v1→
   published change, so benchmark against the published pair), but the beam/cell parameters
   remain v1-only; the Ω_L = 7.9 MHz vs E_LO = 3.0 mV/cm inconsistency stands (μ·E/ħ gives
   2π×5.54 MHz; 7.9 MHz needs 4.28 mV/cm — RydSim always recomputes Ω from (μ, E)); Sci. Adv.
   noise-component recombination gives 7.95 ≠ 10.0 nV/cm/√Hz. Each is carried as an open
   item, not silently patched.
