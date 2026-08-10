# 06 — Optical Bloch Equations, EIT and Autler–Townes

RydSim physics specification, module 06. Status: network verification WAS available for this
document; every constant/citation is tagged VERIFIED (checked online during authoring),
LITERATURE-RECALL (standard result recalled with high confidence, not re-checked this session),
or UNVERIFIED (memory only — must be self-checked as described).

---

## 1. Scope

Steady-state and time-dependent solutions of the Lindblad master equation for the 3-level
(probe + coupling) and 4-level (probe + coupling + RF) ladder systems used in Rydberg
electrometry of Rb-85, Rb-87, Cs-133 vapor. Covers: RWA Hamiltonians, collapse operators
(spontaneous emission, pure dephasing/laser linewidth, transit-time refill), Liouvillian
vectorization and linear-system steady state, weak-probe analytic susceptibility
(continued-fraction form), Doppler averaging, EIT/AT lineshape properties, the wavelength-ratio
Doppler-mismatch scaling of AT splitting, splitting-extraction algorithms, and field inversion
E = ħΩ_RF/℘ with uncertainty budget.

Out of scope (other modules): Rydberg energy levels and quantum defects, radial matrix elements
and dipole moments ℘ (consumed here as inputs), vapor-pressure/number-density model n(T),
cell/RF-field inhomogeneity, hyperfine/Zeeman multilevel structure and optical pumping,
beam propagation beyond thin-medium Beer–Lambert.

---

## 2. Equations

### 2.0 Conventions (used everywhere in RydSim; deviations are bugs)

* Level labels, ladder order: `0 = g` (ground, e.g. 5S1/2), `1 = e` (intermediate, e.g. 5P3/2),
  `2 = r` (Rydberg), `3 = r'` (RF-coupled Rydberg). N = 3 or 4.
* Field of step k (k = p, c, RF): real field E_k(t) = ℰ_k cos(ω_k t); positive-frequency
  convention e^{−iω t}.
* Rabi frequency (angular, rad/s, real ≥ 0 WLOG; complex allowed in API):
  **Ω_k = ℘_k ℰ_k / ħ**, with ℘_k the transition dipole matrix element [C·m] of step k and
  ℰ_k the field **amplitude** [V/m] (not RMS).
* Detunings (angular, rad/s): **Δ_k = ω_k − ω_k^atom** (laser/RF frequency minus transition
  frequency; blue detuning positive). Cumulative (multiphoton) detunings:
  δ_1 = Δ_p, δ_2 = Δ_p + Δ_c, δ_3 = Δ_p + Δ_c + Δ_RF.
* All internal rates/frequencies are **angular** (rad/s). API boundary values in Hz carry the
  suffix `_hz`; conversion is exactly 2π. (The classic silent-2π bug is the single most common
  error in this class of code; the naming rule is mandatory.)
* Density matrix in the rotating frame: σ, with σ_ij = ⟨i|σ|j⟩. The lab-frame optical coherence
  is ρ_eg^lab = σ_eg e^{−iω_p t} (this fixes all signs below).

### 2.1 Rotating-frame RWA Hamiltonian

Single stated convention, used consistently (factor of 2 lives on the diagonal so every
off-diagonal element is −ħΩ/2):

3-level (basis g, e, r):

```
H = -(ħ/2) * [ 0        Ω_p       0
               Ω_p      2δ_1      Ω_c
               0        Ω_c       2δ_2 ]
```

4-level (basis g, e, r, r'):

```
H = -(ħ/2) * [ 0        Ω_p       0         0
               Ω_p      2δ_1      Ω_c       0
               0        Ω_c       2δ_2      Ω_RF
               0        0         Ω_RF      2δ_3  ]
```

Symbols: δ_k as in §2.0 [rad/s]; Ω_p, Ω_c, Ω_RF [rad/s]; H [J].
For complex Ω, element H_{k,k+1} = −(ħ/2)Ω_k and H_{k+1,k} = −(ħ/2)Ω_k*.

RF transition direction: r → r' may be upward or downward in energy (e.g. nD5/2 → (n+1)P3/2 is
downward). Define ω_RF^atom = |E_{r'} − E_r|/ħ and Δ_RF = ω_RF − ω_RF^atom; the matrix above is
unchanged. (The rotating frame is built from the applied field frequencies, not the level
ordering; getting this sign wrong flips the AT asymmetry under RF detuning.) Convention
verification: this H reproduces the textbook 2-level result σ_eg = (iΩ_p/2)/(γ_ge − iΔ_p)
(derived in §2.4; that derivation is the internal consistency check).

### 2.2 Lindblad master equation and collapse operators

```
dσ/dt = -(i/ħ)[H, σ] + Σ_k D[C_k]σ,     D[C]σ = C σ C† − (1/2){C†C, σ}
```

Collapse operators (all rates angular, rad/s):

1. **Spontaneous emission** (population decay):
   * `C_e = sqrt(Γ_e) |g⟩⟨e|` — intermediate state decay. Γ_e = 2π × 6.0666 MHz (Rb D2),
     2π × 5.234 MHz (Cs D2); see §3.
   * `C_r = sqrt(Γ_r) |g⟩⟨r|` and (4-level) `C_r' = sqrt(Γ_r') |g⟩⟨r'|` — **modeling choice**:
     real Rydberg states decay through many cascade channels and blackbody transfer, not to g
     directly. Routing all of Γ_r to g is the recommended default because (i) it keeps the
     system closed (trace preserved without fictitious loss), (ii) in the weak-probe regime the
     steady-state Rydberg population is O((Ω_p/Ω_c)²) ≪ 1, so the routing choice perturbs the
     probe coherence at the same negligible order, and (iii) transit refill (below) dominates
     ground repopulation in a cell anyway. The alternative `sqrt(Γ_r)|e⟩⟨r|` (cascade via e) is
     supported in the API; the two options must agree on χ_probe to better than 10⁻³ relative
     in the weak-probe benchmark regime — this is a self-check, not an assumption.
   * Γ_r, Γ_r' are inputs from the lifetime module (typically 2π × (0.1–10 kHz) for n ≈ 30–100
     including blackbody; order-of-magnitude statement, LITERATURE-RECALL).

2. **Pure dephasing / laser linewidth.** For white-frequency-noise (Lorentzian) laser lines the
   exact reduction is: probe linewidth γ_p dephases every level above the probe transition,
   coupling linewidth γ_c every level above the coupling transition, etc. Implement as one
   projector-sum operator per field:
   * `C_p = sqrt(2 γ_p) (P_e + P_r [+ P_r'])`
   * `C_c = sqrt(2 γ_c) (P_r [+ P_r'])`
   * `C_RF = sqrt(2 γ_RF) P_r'`
   with P_k = |k⟩⟨k|. Factor check (mandatory unit test): C = sqrt(2γ)P_e alone gives
   dσ_ge/dt = −γ σ_ge and no population change. Net coherence decay rates that result:
   ```
   γ_ge = Γ_e/2 + γ_p + γ_col
   γ_gr = (Γ_r + γ_t·…)/2 + γ_p + γ_c + γ_col,rr   (see below for exact composition)
   γ_er = (Γ_e + Γ_r)/2 + γ_c + …
   ```
   The implementation must NOT hand-assemble these γ_ij: they emerge from the collapse set.
   The analytic formula (§2.4) uses γ_ij extracted programmatically as
   γ_ij = −Re L[(ij),(ij)] of the Liouvillian with all Ω = 0 — this guarantees the analytic
   and numeric paths share identical decoherence bookkeeping.
   White-noise caveat: real diode lasers have non-Lorentzian noise; treating linewidth as pure
   dephasing is exact only for white frequency noise (LITERATURE-RECALL; standard caveat).
   `γ_col`: optional collisional dephasing input (pressure/density dependent), default 0.

3. **Transit-time refill (hot cell).** Atoms cross the beam in finite time and are replaced by
   fresh thermal ground-state atoms. Exact Lindblad form — the "measure-and-replace" channel:
   ```
   C_i^t = sqrt(γ_t) |g⟩⟨i|   for ALL i ∈ {g, e, r, r'}     (note: i = g included)
   ```
   Summing D[C_i^t] gives exactly dσ/dt|_transit = γ_t (|g⟩⟨g| Tr σ − σ), i.e. uniform decay of
   every population and coherence at γ_t with repopulation of g. Justification: the Kraus set
   {|g⟩⟨i|} is a valid CPTP channel (Σ K†K = 1), so this *is* a Lindblad dissipator, not an
   ad-hoc source term; physically it is the exponential-dwell-time approximation to beam
   transit. Equivalent non-Lindblad writing dσ/dt += γ_t(σ_0 − σ) with σ_0 = |g⟩⟨g| is
   identical for Tr σ = 1 and may be used interchangeably. The i = g operator contributes pure
   dephasing of g-coherences (correct: a replaced atom carries no coherence).
   * Magnitude model: γ_t ≈ ū_2D / (2 w_0) with ū_2D the mean 2-D transverse thermal speed and
     w_0 the 1/e² beam radius — an order-of-magnitude model only (UNVERIFIED as a precise
     formula; different papers use prefactors between ~0.5 and ~2.4 depending on beam-profile
     averaging). For Rb at 300 K and w_0 = 0.5 mm this gives γ_t/2π ~ 40–100 kHz. The API takes
     γ_t as an explicit input; the helper estimator must carry this caveat in its docstring.
     Self-check: EIT linewidth floor in simulation must respond linearly to γ_t when
     γ_t dominates γ_gr.

Real-atom caveat: true alkali ladders have hyperfine/Zeeman substructure and optical pumping;
this module's 3/4-level model absorbs those effects into effective ℘, γ values (see §7).

### 2.3 Steady state as a linear system (exact recipe)

Work with the scaled Hamiltonian H̃ = H/ħ [rad/s]. Vectorize **column-major**
(`vec = sigma.flatten(order='F')`; index map: σ_ij ↔ vec[i + N*j]). Identities (column-stacking):
vec(AσB) = (Bᵀ ⊗ A) vec(σ). Then the Liouvillian L [rad/s], shape (N², N²):

```
L = -i ( I ⊗ H̃  −  H̃ᵀ ⊗ I )
    + Σ_k [ conj(C_k) ⊗ C_k  −  (1/2) I ⊗ (C_k†C_k)  −  (1/2) (C_k†C_k)ᵀ ⊗ I ]
```

(np.kron; note conj not conj-transpose in the first dissipator factor.) Steady state solves
L·vec(σ) = 0 with Tr σ = 1. L is singular by construction (trace conservation ⇒ the row space
is deficient by exactly one for a system with a unique steady state), so:

1. Build b = zeros(N²).
2. Pick row r0 = 0 (the dσ_00/dt row). Overwrite: `L[r0, :] = 0`, then `L[r0, i + N*i] = s`
   for i = 0..N−1, and `b[r0] = s`, with row scale **s = max(|L|)** (or Γ_e) — scaling the
   trace row to the magnitude of the other rows keeps the condition number near its intrinsic
   value; an unscaled 1.0 row next to 10⁸-magnitude rows costs ~8 digits of conditioning.
3. `sigma = np.linalg.solve(L, b).reshape(N, N, order='F')`.
4. Mandatory post-checks (cheap, always on): |Tr σ − 1| < 10⁻¹², ‖σ − σ†‖_max < 10⁻¹⁰,
   min eig(σ_herm) > −10⁻¹⁰. A silent 'C'/'F' ordering bug produces σᵀ and is caught by the
   Hermiticity check combined with a deliberately asymmetric test case (benchmark B-7).

Uniqueness guard: if any level is dynamically disconnected (its Ω = 0 and it has no decay path;
e.g. 4-level with Ω_RF = 0, Γ_r' = 0, γ_t = 0), L has extra null vectors and `solve` returns
garbage without necessarily erroring. Detect via `np.linalg.cond(L) > 1/eps` (or catch
LinAlgError) and raise with a physics-level message.

Batched/vectorized solving: L is **affine in each detuning and in v** (detunings enter only on
the diagonal of H̃). Decompose L(Δ_p, Δ_c, v) = L_0 + Δ_p L_p + Δ_c L_c + v L_v with constant
matrices; L_v = −k_p L_p + k_c L_c for counter-propagating beams (§2.6). Broadcast to shape
(n_Δ, n_v, N², N²) and use the batched `np.linalg.solve` — this is the intended production path
(≈10⁶ 16×16 solves ≈ seconds).

Time-dependent cross-check (method B for self-validation): σ(t) = expm(L t) vec(σ(0)) via
`scipy.linalg.expm` on L·Δt, propagated to t_end = 20/min(nonzero decay rate); must match the
linear-solve steady state elementwise to < 10⁻⁸ (benchmark B-6).

### 2.4 Weak-probe analytic susceptibility (3-level) and 4-level extension

Perturbative solution to O(Ω_p): set σ_gg = 1, keep only σ_eg, σ_rg (and σ_r'g). With the §2.0
conventions the steady-state chain is (derivation reproduced in appendix comment of the
implementation; the two-line version: the coupled linear equations for σ_eg, σ_rg under
H of §2.1 close exactly):

```
σ_eg = (i Ω_p / 2) / D(Δ_p)

3-level:  D = γ_ge − i δ_1 + (Ω_c²/4) / (γ_gr − i δ_2)
4-level:  D = γ_ge − i δ_1 + (Ω_c²/4) / (γ_gr − i δ_2 + (Ω_RF²/4)/(γ_gr' − i δ_3))
```

— a continued fraction; each additional ladder rung nests one more level. γ_ij are the total
coherence decay rates assembled from the collapse set (§2.2, item 2 — extract from L, don't
hand-write). The susceptibility (χ dimensionless, SI):

```
χ(Δ_p) = (2 N ℘_ge² / (ε₀ ħ Ω_p)) · σ_eg
       = i (N ℘_ge² / (ε₀ ħ)) / D(Δ_p)
```

with N = number density [m⁻³], ℘_ge [C·m], ε₀ [F/m], ħ [J·s]; probe power absorption
coefficient α = k_p · Im χ [1/m] (thin medium; transmission T = exp(−α ℓ)), refractive index
n ≈ 1 + Re χ / 2.

**Form/factor verification (VERIFIED this session):** structure χ = i(Nμ²/ħε₀)·(σ_13/Ω_p) with
nested (|Ω_c|²/4)/(complex two-photon denominator), detuning convention Δ = ω_laser − ω_atom,
time convention e^{−iνt}, and α_0 = Nμ²k_p/(2ε₀ħγ_13) — matches Finkelstein, Bali, Firstenberg
& Novikova, "A practical guide to electromagnetically induced transparency in atomic vapor",
New J. Phys. 25, 035001 (2023), Eqs. (9)–(11) [fetched 2026-08-10]. Ladder-specific two-photon
sign (δ_2 = Δ_p + Δ_c rather than Λ-system Δ_1 − Δ_2) per Gea-Banacloche, Li, Jin & Xiao,
Phys. Rev. A 51, 576 (1995) (citation VERIFIED; equation-level detail LITERATURE-RECALL).

**Convention equivalence note:** the frequently seen form with denominators (γ + iΔ) — as in
the task brief — is the same physics under Δ ≡ ω_atom − ω_laser (or e^{+iωt}); Im χ (absorption
spectrum) is identical point-by-point under Δ → −Δ, Re χ flips sign. RydSim pins (γ − iΔ) with
Δ = ω_laser − ω_atom. Do not mix.

Two-level prefactor sum-rule check (mandatory benchmark B-2): with Ω_c = 0, no dephasing and ℘
related to Γ_e by Γ = ω³℘²/(3π ε₀ ħ c³), the resonant absorption cross-section per atom is
exactly σ_0 = α/N = 3λ²/(2π). This ties N, ℘, ε₀, ħ, k_p factors together analytically
(closed two-level transition, aligned dipole; real degenerate atoms deviate — see §7).

AT structure from the continued fraction: with Δ_c = 0 and small γ's, the nested denominator
resonates where γ-terms → 0, i.e. at two-photon detunings

```
δ_2^± = ( −Δ_RF ± sqrt(Δ_RF² + Ω_RF²) ) / 2
```

⇒ peak separation in δ_2: **sqrt(Δ_RF² + Ω_RF²)**, reducing to Ω_RF on RF resonance
(dressed-state/AT result; standard, LITERATURE-RECALL, and enforced numerically by B-4/B-5).
Under RF detuning the doublet becomes asymmetric in height and its centroid shifts by −Δ_RF/2.

### 2.5 EIT lineshape properties (homogeneous)

From §2.4 with Δ_c = 0, no Doppler:

* Transparency dip FWHM (weak-coupling regime Ω_c ≪ Γ_e, and Γ_EIT ≪ Γ_e):
  ```
  Γ_EIT ≈ 2 γ_gr + Ω_c² / (2 γ_ge)      [rad/s, FWHM]
       (= 2 γ_gr + Ω_c²/Γ_e when γ_ge = Γ_e/2)
  ```
  Derived directly from the §2.4 χ by expanding around δ_2 = 0 (self-validating: benchmark B-8
  extracts the FWHM numerically and compares). Residual on-resonance absorption fraction
  relative to the EIT-free peak: 1/(1 + Ω_c²/(4 γ_ge γ_gr)) — perfect transparency requires
  Ω_c² ≫ 4γ_ge γ_gr (matches the NJP 2023 guide's γ_12γ_13/|Ω_c|² suppression factor, VERIFIED).
* Strong-coupling / Autler–Townes regime Ω_c ≳ Γ_e: the single dip splits into two absorption
  peaks at δ_1 ≈ ±Ω_c/2 (probe scan, Δ_c = 0); separation Ω_c; each peak has width
  ≈ (γ_ge + γ_gr)/… ~ Γ_e/2-scale. The EIT↔ATS crossover is at Ω_c ~ Γ_e; there is a published
  taxonomy of the crossover (e.g. Anisimov/Dowling/Sanders PRL 107, 163604 (2011),
  LITERATURE-RECALL) but RydSim needs no analytic crossover formula — the full solve covers
  both regimes continuously.

### 2.6 Doppler averaging and the wavelength-ratio AT scaling

Geometry: probe along +z, coupling counter-propagating (−z) — the standard Rydberg-EIT cell
geometry. An atom with velocity component v (along +z) sees

```
Δ_p → Δ_p − k_p v ,      Δ_c → Δ_c + k_c v ,      Δ_RF → Δ_RF (RF Doppler negligible*)
```

k = 2π/λ. (*RF Doppler shift v/λ_RF ~ 10 kHz at 10 GHz, 300 K — include optionally, off by
default.) Observable:

```
χ_D(Δ_p) = ∫ dv f(v) χ(Δ_p − k_p v, Δ_c + k_c v, Δ_RF),   f(v) = exp(−v²/u²)/(u√π),
u = sqrt(2 k_B T / m)
```

(u ≈ 240 m/s for Rb-87 at 300 K; probe Doppler FWHM = 2 sqrt(ln 2) k_p u / 2π ≈ 0.51 GHz.)
Two-photon residual Doppler slope: dδ_2/dv = −(k_p − k_c) = +(k_c − k_p) > 0 for Rb/Cs ladders
(λ_c < λ_p) — the mismatch (k_c − k_p) both narrows nothing away and rescales AT splittings:

**Wavelength-ratio scaling (exact statement).** Scanning the **probe** with the coupling locked
on resonance, in a Doppler-broadened counter-propagating ladder, the observed AT splitting in
probe frequency is

```
Δf_meas(probe scan) = (λ_c / λ_p) · Ω_RF / 2π          [Hz]
⇔  Ω_RF = 2π (λ_p / λ_c) Δf_meas                        (recovery direction)
```

For Rb: λ_c/λ_p = 480.0/780.24 ≈ 0.6152 (measured splitting is SMALLER than Ω_RF/2π);
recovery factor λ_p/λ_c = 780.24/480.0 = 1.6255. Scanning the **coupling** with the probe
locked on resonance, the factor is 1 (no rescaling). VERIFIED: equation
Δf_p = (λ_c/λ_p)(Ω_RF/2π) = (λ_c/λ_p)(μ/h)E_RF quoted verbatim from arXiv:2306.13256
(⁸⁷Rb Rydberg EIT-AT vapor-cell work, Eq. (1)) [fetched 2026-08-10]; the effect and its use in
metrology are treated at length in Holloway et al., IEEE Trans. Antennas Propag. 62, 6169
(2014) and Holloway et al., J. Appl. Phys. 121, 233106 (2017) (citations VERIFIED; page-level
equation numbers LITERATURE-RECALL).

Mechanism (2-line derivation, also the regime of validity): the AT transparency contribution of
velocity class v sits at δ_2(v) = ±Ω_RF/2, weighted by that class's one-photon probe resonance
|Δ_p − k_p v| ≲ Γ_e. Maximizing the weight along the constraint gives resonant class
v_± = ±Ω_RF/(2 k_c) and peak positions Δ_p = k_p v_± = ±(k_p/k_c)(Ω_RF/2). Validity requires:
(i) counter-propagation, (ii) Doppler width ≫ Γ_e, Ω_c, Ω_RF (so the resonant-class argument
holds and the Gaussian weight is locally flat), (iii) RF on resonance, (iv) splitting resolved
(Δf_meas ≳ Γ_EIT^obs). Outside these conditions the factor drifts — RydSim must *measure* the
effective factor from the simulated spectrum, which is benchmark B-9.

Doppler-averaged lineshape facts the simulation must reproduce (qualitative, all standard):
the EIT feature survives Doppler averaging in the counter-propagating ladder and its observed
width can be **sub-natural** (< Γ_e) — vapor-cell Rydberg EIT with narrow features was
demonstrated by Mohapatra, Jackson & Adams, PRL 98, 113003 (2007) (citation VERIFIED); theory
of ladder EIT in Doppler media: Gea-Banacloche et al., PRA 51, 576 (1995) (VERIFIED). No
closed-form Doppler EIT width is specified here (published forms are regime-dependent); the
width is a numerical output.

### 2.7 AT splitting vs Ω_RF, threshold behavior, extraction algorithm

* Exact relations (resolved regime): homogeneous probe scan → splitting = Ω_RF/2π on resonance,
  sqrt(Ω_RF² + Δ_RF²)/2π detuned (§2.4). Doppler probe scan → multiply by λ_c/λ_p (§2.6).
* **Resolvability threshold**: the doublet is resolved and the linear relation is metrologically
  valid when the AT splitting exceeds the observed EIT linewidth,
  Δf_AT ≳ Γ_EIT^obs/2π; VERIFIED statement (paper abstract): "the linear relationship is valid
  … as long as the EIT linewidth is small compared to the AT splitting" — Holloway, Simons,
  Gordon, Dienstfrey, Anderson & Raithel, J. Appl. Phys. 121, 233106 (2017). Near/below
  threshold the apparent peak separation of two overlapping lines is pulled BELOW the true
  splitting, then the peaks merge (for two equal Lorentzians of FWHM w the double-max exists
  only for separation > w/√3 — analytic property of the sum of two Lorentzians,
  LITERATURE-RECALL; the pulled-peak sign for the full Doppler lineshape must be established
  numerically, not assumed). This defines the minimum measurable field of the AT method,
  E_min ~ ħ Γ_EIT^obs / ℘ (order-of-magnitude).
* Published mitigation approaches (for the low-field regime):
  1. **Full-model least-squares fit**: fit the complete Doppler-averaged Lindblad spectrum with
     E (i.e. Ω_RF) free — valid through and below threshold; this is RydSim's native mode and
     the recommended approach (it is exactly what the simulator exists to do).
  2. **RF detuning**: detune the RF so the splitting sqrt(Ω_RF² + Δ_RF²) is pushed above
     threshold, then invert; Simons, Gordon, Holloway, Anderson, Miller & Raithel, "Using
     frequency detuning to improve the sensitivity of electric field measurements via
     electromagnetically induced transparency and Autler-Townes splitting in Rydberg atoms",
     Appl. Phys. Lett. 108, 174101 (2016) (citation VERIFIED — vol/issue/page confirmed at
     AIP 2026-08-10).
  3. Quadrature-style corrections of the form Δf_true² ≈ Δf_meas² + (c·Γ_EIT)² appear in the
     literature; no specific published coefficient is asserted here (UNVERIFIED) — if used,
     the coefficient must be calibrated against RydSim's own full-model fit, never hard-coded.
* **Recommended extraction algorithm** (deterministic, testable):
  1. Simulate transmission T(Δ_p) (or Im χ_D) on a grid fine enough that the narrowest feature
     has ≥ 15 samples per FWHM.
  2. Baseline-remove: subtract the Ω_c = 0 (EIT-free) Doppler background computed with
     identical parameters → EIT signal S(Δ_p) ≥ 0.
  3. `scipy.signal.find_peaks(S, prominence=0.05*S.max(), distance=⌈w_est/dΔ⌉)`.
  4. Refine each candidate: nonlinear least-squares Lorentzian fit within ±1 FWHM window
     (parabolic vertex interpolation as fallback); peak position uncertainty from the fit
     covariance.
  5. If exactly 2 peaks: splitting = f_+ − f_−; `resolved = (splitting > mean fitted FWHM)`.
     If 1 peak: report unresolved, return NaN splitting + flag. If > 2: raise (indicates
     multilevel/parameter pathology).
  6. Metrology mode (small splittings): bypass 3–5 and do the full-model fit (approach 1).

### 2.8 Field inversion and uncertainty propagation

```
E = ħ Ω_RF / ℘_RF = h Δf_AT / ℘_RF        [V/m]
Δf_AT = Δf_meas                (coupling scan, or homogeneous/cold)
Δf_AT = (λ_p/λ_c) Δf_meas      (Doppler probe scan, §2.6 conditions)
```

℘_RF: dipole matrix element of the r → r' Rydberg transition [C·m] (input from the matrix-
element module; for the reduced-vs-mJ-resolved element and polarization factors see that
module — using the wrong angular factor is a leading systematic).

Uncertainty (uncorrelated inputs):

```
(u(E)/E)² = (u(Δf_meas)/Δf_meas)² + (u(℘)/℘)² + (u_ratio/ratio)² + Σ (systematic terms)
```

u(Δf_meas): from peak-fit covariance + grid/fit-model residual. u(℘): from the matrix-element
module (radial-integral methods for Rydberg states are typically quoted at the 0.1–1 % level —
LITERATURE-RECALL; RydSim must propagate whatever that module reports, not assume). u_ratio:
laser wavelengths are known to ≪10⁻⁶ relative; negligible — the ratio *model validity* error
(§2.6 conditions) dominates and is estimated by re-extracting the factor from simulation at the
user's parameters (B-9 machinery).

Systematics checklist a metrology-grade run must evaluate (each togglable in RydSim; magnitudes
are configuration-dependent and must be computed, not quoted):

| # | Systematic | Handling in RydSim |
|---|---|---|
| 1 | Doppler wavelength-ratio validity (finite Ω/Doppler ratio) | re-derive factor numerically at user parameters |
| 2 | RF detuning: splitting → sqrt(Ω²+Δ_RF²) | fit Δ_RF or bound it |
| 3 | Peak pulling below/near threshold | full-model fit (§2.7) |
| 4 | Probe/coupling power broadening & AC-Stark asymmetry | vary Ω_p, Ω_c in fit; keep Ω_p ≪ Γ_e |
| 5 | Off-resonant RF coupling to other Rydberg levels (AC Stark) | beyond 4-level; bound via level-structure module |
| 6 | Vapor-cell etalon / internal-field ≠ incident-field | NOT modeled here — cell module; flag in report (major in practice: Holloway 2017, VERIFIED-abstract-level) |
| 7 | RF field inhomogeneity across beam / cell | average E over profile if map provided |
| 8 | Gaussian beam averaging of Ω_p, Ω_c | intensity-weighted average of χ over radius |
| 9 | Transit time & collisional broadening | γ_t, γ_col inputs; sensitivity scan |
| 10 | Multiple |m_J| components → multiple ℘ → composite splitting | matrix-element module supplies set; simulate superposition |
| 11 | Optical pumping / hyperfine structure | outside 4-level model; see §7 |
| 12 | Rydberg density effects (interactions, ionization) | keep n, Ω_p low; see §7 |

---

## 3. Constants / parameters

| Quantity | Symbol | Value | Units | Source | Confidence |
|---|---|---|---|---|---|
| Planck constant | h | 6.62607015e-34 (exact) | J s | SI 2019 / CODATA | VERIFIED (defined) |
| Reduced Planck | ħ | h/2π | J s | — | — |
| Boltzmann constant | k_B | 1.380649e-23 (exact) | J/K | SI 2019 | VERIFIED (defined) |
| Speed of light | c | 299792458 (exact) | m/s | SI | VERIFIED (defined) |
| Vacuum permittivity | ε₀ | 8.8541878188(14)e-12 | F/m | CODATA 2022 / NIST | VERIFIED |
| Rb-87 D2 vacuum wavelength (probe) | λ_p | 780.241209686(13) | nm | Steck, Rubidium 87 D Line Data | VERIFIED |
| Rb-87/85 D2 natural linewidth | Γ_e/2π | 6.0666(18) | MHz | Steck Rb87/Rb85 D Line Data | VERIFIED |
| Rb-85 D2 vacuum wavelength | λ_p | 780.241368271(27) | nm | Steck, Rubidium 85 D Line Data | VERIFIED |
| Rb D2 lifetime | τ_e | 26.2348(77) | ns | Steck (Rb85 sheet; consistent with Γ) | VERIFIED |
| Cs-133 D2 vacuum wavelength (probe) | λ_p | 852.34727582(27) | nm | Steck, Cesium D Line Data | VERIFIED |
| Cs-133 D2 natural linewidth | Γ_e/2π | 5.234(13) | MHz | Steck, Cesium D Line Data | VERIFIED |
| Cs-133 D2 lifetime | τ_e | 30.405(77) | ns | Steck, Cesium D Line Data | VERIFIED |
| Rb coupling wavelength (5P3/2→nD/nS) | λ_c | ≈ 479–484 (state-dependent); 480.0 used in benchmarks | nm | level-structure module | LITERATURE-RECALL (range) |
| Cs coupling wavelength (6P3/2→nD/nS) | λ_c | ≈ 508–512 (state-dependent) | nm | level-structure module | LITERATURE-RECALL (range) |
| Atomic masses (Rb-85/87, Cs-133) | m | from constants module (AME-based) | kg | module 01 / AME2020 | see module 01 — do not re-type here |
| Rydberg-state decay Γ_r, Γ_r' | Γ_r | input; typ. 2π×(0.1–10) kHz incl. blackbody, n≈30–100 | rad/s | lifetime module | input (recall for typical scale only) |
| Number density n(T) | N | input | m⁻³ | vapor-pressure module (Steck model) | input |
| Transit rate | γ_t | input; typ. 2π×(10–100) kHz @ 300 K, w₀~0.5 mm | rad/s | §2.2 estimator | UNVERIFIED (order of magnitude) |
| Wavelength-ratio benchmark factor | λ_p/λ_c | 780.24/480.0 = 1.62550 | — | §2.6 (VERIFIED relation) | VERIFIED (relation), fixed numbers by convention |

MISSING (deliberately not specified here — owned by other modules): ℘_ge and ℘_RF values,
per-state Γ_r(n, L, J, T), n(T) coefficients, exact coupling wavelengths per Rydberg state.

---

## 4. Numerical method and pitfalls

1. **Units discipline**: rad/s internally; `_hz` suffix at boundaries; one conversion site.
2. **Liouvillian assembly**: §2.3 recipe; build once as L_0 + Δ_p L_p + Δ_c L_c (+ Δ_RF L_r) +
   v L_v (affine decomposition), then broadcast. Assemble with complex128 throughout.
3. **Conditioning**: scale trace row (§2.3); with rates spanning 2π×1 Hz…2π×10 GHz the raw
   dynamic range is ~10¹⁰ — comfortably within double precision if the trace row is scaled;
   optionally nondimensionalize by Γ_e (divide L by Γ_e) so entries are O(10⁻⁴…10³).
4. **Velocity integration**: the EIT/AT structure occupies a velocity window of width
   δv ≈ Γ_EIT/(k_c − k_p) (≈ 1 m/s for Γ_EIT = 2π×1 MHz, Rb 780/480: (k_c−k_p)/2π ≈
   0.80 MHz per m/s) inside a ±4u ≈ ±1000 m/s domain. Naive Gauss–Hermite with < 100 nodes
   UNDERSAMPLES the narrow structure and silently biases the AT peaks — forbidden as sole
   method. Required scheme: composite grid = (a) coarse trapezoid/GH over ±4.5u for the Doppler
   background + (b) dense uniform windows (step ≤ δv/10) centered on each resonant class
   v = Δ_p/k_p and v_± = ±Ω_RF/(2k_c) (probe scan). Simpler compliant alternative: uniform
   trapezoid over ±4.5u with step ≤ δv/10 (≈ 10⁵ nodes) using the analytic χ (cheap), and the
   composite grid for full-Lindblad runs. Convergence criterion (mandatory): halving the step
   AND widening to ±5.5u changes the spectrum by < 10⁻⁴ relative (max-norm); expose as
   `converged` flag, don't just document it.
5. **Detuning grid**: ≥ 15 points per narrowest FWHM (peak-fit accuracy degrades as (Δgrid)²;
   with Lorentzian refinement 15/FWHM gives ≪1 % splitting error). Auto-estimate the narrowest
   width from §2.5 formula before gridding.
6. **Weak-probe consistency**: production spectra may use the analytic χ (fast path) ONLY when
   Ω_p < 0.01·min(Γ_e, Ω_c); otherwise full Lindblad. The two paths must agree in the overlap
   (benchmark B-1) — run this agreement check in CI, not once.
7. **Steady-state validity (time scales)**: the linear solve gives t → ∞; real scans are
   quasi-static if scan rate ≪ Γ_EIT²/(2π) (adiabatic criterion, LITERATURE-RECALL as standard
   practice); RydSim models CW only (see §7).
8. **Known traps**: (i) 'C' vs 'F' vec ordering (caught by B-7); (ii) Ω vs Ω/2 convention drift
   between H and analytic formulas (caught by B-1); (iii) sign of k_c v for co- vs counter-
   propagation — expose `geometry: Literal["counter", "co"]`, never a bare sign; (iv) using
   FWHM linewidths where HWHM decay rates are meant: Γ_e (population, full rate) vs
   γ_ge = Γ_e/2 + …; (v) forgetting that find_peaks on transmission (peaks) vs absorption
   (dips) inverts the signal; (vi) treating λ-ratio factor as exact — it is an approximation
   (§2.6 validity list).

---

## 5. Recommended Python API (numpy-only, Python 3.11)

```python
# rydsim/obe.py
from dataclasses import dataclass, field
from typing import Literal
import numpy as np
from numpy.typing import NDArray

Complex = NDArray[np.complex128]
Float = NDArray[np.float64]

@dataclass(frozen=True, slots=True)
class LadderParams:
    """All rates ANGULAR (rad/s). n_levels in (3, 4). Index order g,e,r[,r'].

    omega: (n_levels-1,) Rabi frequencies (Ω_p, Ω_c[, Ω_RF]); complex allowed.
    delta: (n_levels-1,) per-step detunings (Δ_p, Δ_c[, Δ_RF]), ω_field − ω_atom.
    gamma_pop: (n_levels,) population decay Γ of each level (Γ_g=0, Γ_e, Γ_r[, Γ_r']).
    decay_route: 'to_ground' (default, §2.2) or 'cascade' (r→e→g).
    gamma_laser: (n_levels-1,) white-noise linewidths (γ_p, γ_c[, γ_RF]).
    gamma_transit: scalar γ_t ≥ 0 (measure-and-replace refill, §2.2).
    gamma_collisional: scalar, extra uniform dephasing of optical coherences.
    """
    n_levels: int
    omega: tuple[complex, ...]
    delta: tuple[float, ...]
    gamma_pop: tuple[float, ...]
    gamma_laser: tuple[float, ...]
    gamma_transit: float = 0.0
    gamma_collisional: float = 0.0
    decay_route: Literal["to_ground", "cascade"] = "to_ground"

def hamiltonian(p: LadderParams, delta_override: Float | None = None) -> Complex:
    """H/ħ (rad/s), shape (..., N, N); §2.1 convention (−Ω/2 off-diag, 2·cumulative Δ diag).
    delta_override: (..., N-1) broadcastable per-step detunings for batched scans."""

def collapse_ops(p: LadderParams) -> list[Complex]:
    """Collapse operators per §2.2 (spont. emission w/ routing, projector dephasing, transit)."""

def liouvillian(h_over_hbar: Complex, c_ops: list[Complex]) -> Complex:
    """Column-stacking (order='F') Liouvillian, shape (..., N², N²); §2.3 formula."""

def liouvillian_affine(p: LadderParams) -> tuple[Complex, tuple[Complex, ...]]:
    """(L0, (L_dp, L_dc[, L_drf])) with L(Δ) = L0 + Σ Δ_k L_k; enables batched scans and
    Doppler shifts via L_v = −k_p·L_dp + s_geom·k_c·L_dc."""

def steady_state(liou: Complex, *, trace_row: int = 0) -> Complex:
    """Batched solve of L·vec(σ)=0, Tr σ=1 (§2.3: scaled trace-row replacement).
    Returns σ (..., N, N). Raises SteadyStateError on cond(L) failure / disconnected level.
    Always verifies trace/Hermiticity/positivity (§2.3 tolerances)."""

def coherence_ge(sigma: Complex) -> Complex:
    """σ_eg = sigma[..., 1, 0] — the probe coherence entering χ (sign per §2.0)."""

def susceptibility_weak_probe(
    delta_p: Float, p: LadderParams, number_density: float, dipole_ge: float
) -> Complex:
    """Analytic continued-fraction χ(Δ_p) (§2.4); γ_ij extracted from the collapse set so the
    analytic and Lindblad paths share decoherence bookkeeping. Valid Ω_p→0; caller enforces
    Ω_p < 0.01·min(Γ_e, |Ω_c|)."""

def susceptibility_lindblad(
    delta_p: Float, p: LadderParams, number_density: float, dipole_ge: float
) -> Complex:
    """χ from full steady state: χ = 2 N ℘²/(ε₀ ħ Ω_p) · σ_eg, vectorized over delta_p."""

@dataclass(frozen=True, slots=True)
class DopplerSpec:
    temperature_K: float
    mass_kg: float
    lambda_probe_m: float
    lambda_coupling_m: float
    geometry: Literal["counter", "co"] = "counter"
    v_halfwidth_sigmas: float = 4.5
    v_step_fraction: float = 0.1     # step = fraction × Γ_EIT/(|k_c − k_p|)
    include_rf_doppler: bool = False

def doppler_average(
    chi_fn, delta_p: Float, dop: DopplerSpec, *, refine_windows: bool = True
) -> tuple[Complex, dict]:
    """∫dv f(v) χ(Δ_p − k_p v, Δ_c ± k_c v) with composite grid (§4.4).
    Returns (χ_D, info) where info = {'converged': bool, 'n_nodes': int, 'max_rel_change': float}
    from the built-in half-step/wider-domain re-check."""

def transmission(chi: Complex, length_m: float, lambda_probe_m: float) -> Float:
    """T = exp(−k_p · Im χ · L). Thin-medium Beer–Lambert; no coupling depletion (§7)."""

@dataclass(frozen=True, slots=True)
class ATSplitResult:
    f_minus_hz: float; f_plus_hz: float
    splitting_hz: float                # NaN if unresolved
    fwhm_hz: tuple[float, float]
    resolved: bool
    u_splitting_hz: float              # from fit covariance
    method: Literal["two_peak_fit", "full_model_fit"]

def extract_at_splitting(
    freq_hz: Float, signal: Float, *, prominence_frac: float = 0.05,
    force_full_model: bool = False, model_fit_fn=None
) -> ATSplitResult:
    """§2.7 algorithm: background-subtracted peak find → Lorentzian refinement → resolve check;
    falls back to full-model fit when unresolved and model_fit_fn is provided."""

@dataclass(frozen=True, slots=True)
class EFieldResult:
    e_field_v_per_m: float
    u_e_field_v_per_m: float
    rabi_rf_rad_s: float
    doppler_ratio_applied: float       # 1.0 or λ_p/λ_c
    systematics_report: dict[str, float | str]

def invert_field(
    at: ATSplitResult, dipole_rf_cm: float, u_dipole_rel: float, *,
    scan: Literal["probe", "coupling"], lambda_probe_m: float, lambda_coupling_m: float,
    doppler_broadened: bool = True,
) -> EFieldResult:
    """E = h·Δf_AT/℘ with Δf_AT = (λ_p/λ_c)·Δf_meas for Doppler probe scans (§2.8).
    Raises if at.resolved is False. Propagates u(Δf), u(℘) in quadrature; systematics_report
    lists §2.8 table items with 'not_evaluated'/value entries — never silently omitted."""
```

Contracts: every function broadcasts over leading axes; no Python loops over detuning/velocity
grids; all docstrings state units; `steady_state` never returns an unchecked σ.

---

## 6. Validation benchmarks (→ `tests/test_obe_eit.py`)

Common parameters unless stated: Rb-87 numbers (Γ_e/2π = 6.0666 MHz, λ_p = 780.241 nm,
λ_c = 480.0 nm), Γ_r/2π = 1 kHz, Γ_r'/2π = 1 kHz, γ_laser = 0, γ_t = 0, Δ_c = Δ_RF = 0,
N = 1 (χ scale-invariant checks) — all self-validating unless a source is cited.

| ID | Quantity | Setup | Expected | Tolerance | Source / type | Confidence |
|---|---|---|---|---|---|---|
| B-1 | max rel. |χ_lindblad − χ_analytic|/|χ_analytic| over Δ_p ∈ ±2π·30 MHz (601 pts) | 3-level; Ω_p/2π = 1 Hz; Ω_c/2π = 5 MHz | 0 | ≤ 1e-10 | method-A-vs-B self-check (weak-probe correction O((Ω_p/Ω_c)²) ≈ 4e-14 here) | VERIFIED (by construction) |
| B-2 | resonant cross-section α/N, two-level (Ω_c = 0), ℘ from Γ = ω³℘²/(3πε₀ħc³) | Ω_p/2π = 1 Hz, Δ_p = 0 | 3λ_p²/2π = 2.9070e-13 m² (λ = 780.241 nm) | rel ≤ 1e-8 | analytic sum rule (§2.4) | VERIFIED (identity) |
| B-3 | fitted HWHM of two-level Im χ | as B-2, scan ±2π·60 MHz | γ_ge = Γ_e/2 → HWHM/2π = 3.0333 MHz | rel ≤ 1e-4 | Lorentzian limit of §2.4; Γ_e Steck | VERIFIED |
| B-4 | AT splitting linearity, homogeneous 4-level | Ω_c/2π = 3 MHz; Ω_RF/2π ∈ {10, 20, 40, 80} MHz; probe scan | splitting = Ω_RF/2π; fit slope 1.000 through origin | each ≤ 1 %; slope ≤ 0.5 % | dressed-state result §2.4 | VERIFIED (self-check vs analytic peak positions) |
| B-5 | detuned-RF splitting | as B-4, Ω_RF/2π = 20 MHz, Δ_RF/2π = 15 MHz | sqrt(20²+15²) = 25.0 MHz | ≤ 1 % | §2.4 dressed states | LITERATURE-RECALL (formula), self-checked numerically |
| B-6 | steady state: solve vs expm time propagation | 4-level, Ω_p/2π = 0.2 MHz (NOT weak), Ω_c/2π = 5 MHz, Ω_RF/2π = 10 MHz, γ_t/2π = 50 kHz; t_end = 20/min-rate | identical σ | max elem ≤ 1e-8 | method-A-vs-B self-check | VERIFIED (by construction) |
| B-7 | trace, Hermiticity, positivity + anti-transpose canary | asymmetric params (Δ_p/2π = 3.7 MHz, complex Ω_c phase 0.3 rad) | Tr = 1; σ = σ†; eig ≥ 0; Im σ_eg matches analytic sign (absorptive > 0) | 1e-12 / 1e-10 / −1e-10 / sign | invariants | VERIFIED (by construction) |
| B-8 | homogeneous EIT FWHM | 3-level, Ω_c/2π = 1.0 MHz, γ_gr/2π = 10 kHz (via Γ_r, γ_c) | 2γ_gr + Ω_c²/(2γ_ge) → /2π: 0.020 + 1.0²/6.0666 = 0.1848 MHz | ≤ 10 % | §2.5 derived formula | VERIFIED (derivation), tolerance covers expansion error |
| B-9 | Doppler wavelength-ratio factor | T = 300 K Rb-87, counter-prop 780.241/480.0 nm, Ω_c/2π = 5 MHz, Ω_RF/2π = 20 MHz, probe scan | splitting/(Ω_RF/2π) = λ_c/λ_p = 0.61519; inferred Ω via ×1.62550 recovers 20 MHz | ≤ 5 % | arXiv:2306.13256 Eq.(1); Holloway JAP 121, 233106 (2017) | VERIFIED (relation) |
| B-10 | coupling-scan factor = 1 | as B-9 but scan Δ_c, Δ_p = 0 | splitting = Ω_RF/2π = 20 MHz | ≤ 5 % | §2.6 | VERIFIED (relation) |
| B-11 | sub-threshold behavior | as B-9, Ω_RF/2π ∈ {0.2, 0.5} × Γ_EIT^obs | `resolved == False`; full-model fit recovers Ω_RF | fit ≤ 5 % | Holloway JAP 121, 233106 (2017): linearity fails when splitting ≲ EIT width | VERIFIED (abstract-level statement) |
| B-12 | qualitative published-spectrum reproduction | Rb room-T cell, counter-prop 780/480, Ω_c/2π ≈ 2–6 MHz, weak probe, no RF | single EIT peak on Doppler-broadened absorption; observed FWHM < Γ_e (sub-natural) and ≫ γ_gr; peak height few % of Doppler absorption depth | qualitative flags (3 booleans) | Mohapatra, Jackson & Adams, PRL 98, 113003 (2007) | VERIFIED (citation + qualitative facts); exact figure values NOT asserted |
| B-13 | transit refill sanity | all Ω = 0, γ_t/2π = 50 kHz, start σ = |r⟩⟨r| (time path) | σ(∞) = |g⟩⟨g| | 1e-10 | §2.2 channel definition | VERIFIED (by construction) |
| B-14 | decay-routing insensitivity | B-1 params, 'to_ground' vs 'cascade' | same χ | rel ≤ 1e-3 | §2.2 argument | VERIFIED (self-check) |

pytest notes: B-1/B-6 are the core convention locks — run first, fail loudly. B-9/B-10/B-11 are
slow (Doppler); mark `@pytest.mark.slow` but gate releases on them. Grids per §4.4–4.5.

---

## 7. Known limitations / model breakdown

1. **3/4-level reduction**: real atoms have hyperfine + Zeeman structure; optical pumping,
   dark hyperfine states, and multiple ℘(m_J) values are absorbed into effective parameters.
   Quantitative amplitude predictions (EIT contrast) are ~tens-of-percent level at best;
   *frequency* observables (splittings, positions) are robust. Multilevel extension is a
   separate module if metrology-grade amplitudes are ever needed.
2. **Thin-medium, no propagation**: Beer–Lambert with constant Ω_c, Ω_RF; no coupling
   absorption/depletion, no probe back-action, no cell etalon or RF standing waves (§2.8 items
   6–7 must come from a cell/EM module). Breaks down for optically thick cells (α ℓ ≳ 1 with
   the coupling also attenuated).
3. **CW steady state only**: no pulsed dynamics, no scan-rate transients (validity §4.7);
   superheterodyne/modulated-RF operation needs the time-dependent path (expm machinery exists
   but no modulation framework here).
4. **Dilute, non-interacting atoms**: no Rydberg–Rydberg interactions, ionization, plasma
   fields, or radiation trapping; keep Rydberg fraction low (weak probe). At high n and high
   density, interaction shifts/broadening invalidate the model (documented experimentally in
   the cold-atom AT literature; LITERATURE-RECALL).
5. **1-D Doppler, no transverse effects**: velocity-changing collisions and beam-profile
   inhomogeneity beyond the γ_t and item-8 averaging are not modeled.
6. **White-noise laser lineshapes** only (§2.2); correlated or 1/f laser noise changes EIT
   linewidth floors in ways pure dephasing cannot capture.
7. **λ-ratio scaling is approximate** (§2.6 validity list); near the resolvability threshold or
   when Ω_RF approaches the Doppler width, use the numerically extracted factor, not 1.6255.
8. **RF strong-field regime**: for Ω_RF comparable to the r–r' transition frequency or when the
   RF couples multiple Rydberg pairs, the 4-level RWA fails (Floquet/multilevel treatment
   required — out of scope).

---

### Source list (verification performed 2026-08-10; network available)

* Steck, D. A., "Rubidium 87 D Line Data", "Rubidium 85 D Line Data", "Cesium D Line Data",
  https://steck.us/alkalidata — λ, Γ_e, τ values (VERIFIED via search snippets this session).
* NIST CODATA 2022, https://physics.nist.gov/cuu — ε₀ (VERIFIED); h, k_B, c exact by SI definition.
* arXiv:2306.13256 (⁸⁷Rb Rydberg EIT/AT vapor cell) — Eq. (1): Δf_p = (λ_c/λ_p)(Ω_RF/2π) =
  (λ_c/λ_p)(μ/h)E_RF (VERIFIED, fetched full text).
* Finkelstein, Bali, Firstenberg, Novikova, New J. Phys. 25, 035001 (2023) — χ conventions,
  EIT suppression factor (VERIFIED, fetched).
* Holloway et al., J. Appl. Phys. 121, 233106 (2017) — systematics; linearity valid for
  EIT width ≪ AT splitting (VERIFIED, abstract).
* Holloway et al., IEEE Trans. Antennas Propag. 62, 6169 (2014); Gordon et al.
  (arXiv:1406.2936, APL 2014); Sedlacek et al., Nat. Phys. 8, 819 (2012) — electrometry
  method + λ-ratio usage (citations VERIFIED; equation numbers LITERATURE-RECALL).
* Gea-Banacloche, Li, Jin, Xiao, Phys. Rev. A 51, 576 (1995) — Doppler ladder EIT theory
  (citation VERIFIED).
* Mohapatra, Jackson, Adams, Phys. Rev. Lett. 98, 113003 (2007) — vapor-cell Rydberg EIT
  (citation VERIFIED).
* Simons et al., Appl. Phys. Lett. 108(17), 174101 (2016) — detuned-RF technique
  (VERIFIED: full citation confirmed via AIP pubs this session).
* Anisimov, Dowling, Sanders, PRL 107, 163604 (2011) — EIT/ATS crossover taxonomy
  (LITERATURE-RECALL).
