# 10 — Optically Thick Propagation of the Probe (and Coupling) Through a Vapor Cell

**RydSim physics specification, module 10, part 1 of N: THEORY.** Status: network verification WAS
available for this document. Every equation, constant and numerical claim carries a source and one
of the tags **VERIFIED** (primary source fetched this session and quoted), **VERIFIED-ARC**
(matches a reputable secondary implementation, primary unfetched), **LITERATURE-RECALL**,
**UNVERIFIED**, **DERIVED-IN-SPEC** (algebra done here; the falsifying numerical check is stated),
or **MISSING**.

**Precedence.** Spec 00 overrides this document. §7 lists the three places where this spec believes
00/05/06 should be **amended**, stated openly rather than diverged from silently, per the brief.

**Ownership.** Propagation is assigned to **05 / `rydsim.vapor`** by the spec-00 §4 ownership map.
This document is the theory that 05 §2.f sketches in six lines; it does not move ownership.

---

## 1. Scope

### 1.1 What this document owns

The **coupled steady-state propagation problem** for a CW Rydberg-EIT vapor cell:

- (a) the Maxwell → paraxial → slowly-varying-envelope (SVEA) reduction, and the exact form and
  sign of `dÊ_p/dz` under the spec-00 field and detuning conventions;
- (b) the exact conditions under which the existing thin-medium answer
  `T = exp(−k_p Im χ L)` is **not an approximation at all** — and what actually breaks when they
  fail (§2.3, the *reduction theorem*, and the single most important falsification test in this
  spec);
- (c) **coupling depletion**: the propagation equation for the coupling field, why it is
  `O(Ω_p²)` and therefore inseparable from probe saturation, and the exact photon-number bound
  that quantifies it (§2.4);
- (d) the **counter-propagating two-point boundary-value problem** (probe known at `z = 0`,
  coupling known at `z = L`), and the coupling strength at which the boundary genuinely matters
  (§2.5);
- (e) how the **velocity average and the spatial integration compose**, why they do not commute
  once the fields vary with `z`, the exact first-order transport correction, and its cost (§2.6);
- (f) **probe saturation with depth** — why the weak-probe approximation *self-heals* going into
  the cell and where the entrance condition binds (§2.7);
- (g) **radiation trapping / reabsorption** — a model-free upper bound that needs no escape-factor
  theory, plus the honest declaration of what is MISSING (§2.8);
- (h) transverse structure: per-shell propagation, and the dispersive self-lensing criterion that
  bounds the paraxial-shell model (§2.9).

### 1.2 What this document does not own

Susceptibility itself (**06**), velocity grids and quadrature convergence (**05 §2.d / 06 §4.4**),
transit rates (**05 §2.e**, ruling R-3), density and vapor pressure (**05 §2.a**), beam geometry and
radial averaging (**05 §2.g**), cell-wall screening (**05 §2.h**), and the superheterodyne readout
chain (**08**). Pulsed/transient propagation, dark-state polaritons, slow-light storage, four-wave
mixing and Rydberg–Rydberg interactions are **out of scope** and appear only in §6.

### 1.3 New empirical content

**None.** This document introduces **no new measured constant.** Every number below is either a
fundamental constant from `scipy.constants` (spec 00 lock #15), a quantity already owned by specs
01–06, or algebra performed here with its falsification test stated. That is deliberate: a
propagation spec that needs new fitted parameters is a propagation spec that is hiding a model.

---

## 2. Equations

### 2.0 Conventions inherited (deviations are bugs)

From spec 00 locks #1–#5 and spec 05 §2.f, used without restatement:

- Real field `E_p(z,t) = ½ Ê_p(z) e^{i(k_p z − ω_p t)} + c.c.`; **`Ê` is the peak amplitude, never
  RMS** (lock #3). Intensity `I = ε₀c|Ê|²/2`.
- `Ω_k = ℘_k Ê_k/ħ` [rad/s], full Rabi frequency (lock #4).
- `Δ = ω_field − ω_atom` (lock #5); weak-probe denominators `(γ − iΔ)`.
- `k_p = 2π/λ_p` is the **vacuum** wavenumber — "no extra 2 or 4π" (spec 05 §2.f, verbatim).
- `χ` dimensionless; `Im χ > 0` ⇔ absorption; `α_p ≡ k_p Im χ` is the **intensity** absorption
  coefficient [m⁻¹]; `OD ≡ α_p L` (spec 00 symbol table, row "α (absorption)").

New symbols introduced by this document (all SI):

| Symbol | Meaning | Unit | Code name |
|---|---|---|---|
| `z` | axial coordinate, probe propagates toward `+z` | m | `z_m` |
| `s_c ≡ L − z` | distance travelled by the **counter-propagating coupling** | m | `s_coupling_m` |
| `χ_p, χ_c` | susceptibility at the probe / coupling frequency | — | `chi_probe`, `chi_coupling` |
| `α_p, α_c` | intensity absorption coefficients, `α_k = k_k Im χ_k` | m⁻¹ | `absorption_coeff_probe/_coupling` |
| `Φ_k` | photon flux `I_k/(ħω_k)` | m⁻² s⁻¹ | `photon_flux_probe/_coupling` |
| `s_sat` | probe saturation parameter `I_p/I_sat` (**collides** with spec 05 §2.g's radial `s = 2r²/w₀²`) | — | `s_sat` vs `s_radial` — never bare `s` |
| `L_grad` | field-gradient scale `\|Ω_p/(dΩ_p/dz)\| = 2/α_p` | m | `grad_length_m` |
| `ε_transp` | non-locality parameter of the local-response approximation, Eq. (10.14) | — | `transport_epsilon` |
| `γ_trap` | extra `g–r` dephasing from reabsorbed resonance photons | rad/s | `gamma_trap` |
| `g_esc` | Holstein escape factor (**MISSING closed form**, §2.8) | — | `holstein_escape_factor` |
| `η_Ωc` | tolerated fractional error in `Ω_c` (accuracy target) | — | `omega_c_tolerance` |

**Symbol-collision additions for spec 00 §3:** `s` (saturation parameter vs radial variable vs line
strength vs PSD); `α` gains `absorption_coeff_probe`/`_coupling` as distinct names; `L` (cell length
vs Liouvillian) — `cell_length_m` vs `liouvillian`.

---

### 2.1 Maxwell → paraxial → SVEA: the probe propagation equation

Start from the scalar wave equation for one transverse polarization component in a medium of
polarization `P` (Gaussian-beam shells treated independently — see §2.9 for the validity of that):

```
(10.1)   ∂²E/∂z² + (ω²/c²) E = −μ₀ ω² P
```

Insert `E = ½Ê(z)e^{i(k_p z − ωt)} + c.c.` and `P = ½ P̂(z) e^{i(k_p z − ωt)} + c.c.`, use
`k_p = ω/c` (vacuum), and drop `∂²Ê/∂z²` against `2k_p|∂Ê/∂z|` (SVEA). With the constitutive
relation `P̂ = ε₀ χ_p Ê` this gives the **probe propagation equation**:

```
(10.2)   dÊ_p/dz = + i (k_p/2) χ_p(z) Ê_p(z)      [V/m per m]
         equivalently  dΩ_p/dz = + i (k_p/2) χ_p(z) Ω_p(z)      [rad/s per m]
```

**Sign audit (mandatory, this is the single most bug-prone line in the module).** With
`Im χ_p > 0` (absorption, spec 00 symbol table), Eq. (10.2) gives
`|Ê_p(z)|² = |Ê_p(0)|² exp(−k_p Im χ_p z)`, i.e. `T = exp(−k_p Im χ_p L)` — **exactly** spec 05
§2.f and exactly `rydsim.eit.transmission()` (`src/rydsim/eit.py:212`). A `−i` in (10.2) would
produce gain and is caught by benchmark **10/P-1**.

**Sources.** The equation form and its factor are **VERIFIED**: Ogden, Whittaker, Keaveney,
Wrathmall, Adams & Potvliege, *Quasi-simultons in thermal atomic vapors*, Phys. Rev. Lett. **123**,
243604 (2019), arXiv:1909.07161, **Eq. (6)** quoted verbatim from the fetched full text
(ar5iv, 2026-08-11):

> `[∂/∂z + (1/c)∂/∂t] ℰ_α = (ik/2ε₀) 𝒫_α,  α = p,c`

Setting `∂/∂t → 0` (CW steady state) and `𝒫_α = ε₀ χ_α ℰ_α` reproduces (10.2) **including the
factor ½ and the `+i`**, for *both* fields, in a thermal alkali ladder. This is a
propagation-equation-level match, not an abstract-level citation.

**Assumptions folded into (10.2), each with its fence:**

| # | Assumption | Fence / criterion | Status |
|---|---|---|---|
| A1 | SVEA: `\|∂²Ê/∂z²\| ≪ 2k_p\|∂Ê/∂z\|` | equivalent to `α_p ≪ 4k_p`, i.e. `OD ≪ 4k_p L ≈ 1.6×10⁹` for a 5 cm Rb cell | never binding — VERIFIED by inspection |
| A2 | `\|χ\| ≪ 1`, so `n_refr ≈ 1 + χ/2`, no Fresnel reflection at the vapor boundary | `\|χ\| ~ 10⁻⁶` at 25 °C (spec 05 §2.f) | VERIFIED (spec 05) |
| A3 | CW steady state, no `(1/c)∂_t` | scan rate `≪ Γ_EIT²/2π` (spec 06 §4.7); transit time `w₀/⟨v⊥⟩ = 4.71 µs` ≫ light transit `L/c = 0.17 ns` | LITERATURE-RECALL, ratio computed |
| A4 | Scalar field; polarization preserved | absorbed in effective dipoles, spec 06 §7.1 | declared limitation |
| A5 | No diffraction between radial shells | `L ≪ 2z_R`; `z_R = 4.0 m` at `w₀ = 1 mm`, 780 nm (spec 05 §2.g) | VERIFIED (spec 05) |
| A6 | Local (no atomic transport) response | §2.6, Eq. (10.14) | **DERIVED-IN-SPEC, benchmark 10/P-7** |

**Why `χ_p` carries a `z` argument at all.** `χ_p` is a *local* material property evaluated with the
*local* field amplitudes and local thermodynamic state:

```
(10.3)   χ_p(z) = χ_p[ Ω_p(z), Ω_c(z), Ω_RF(z); N(z), T(z), {γ}(z) ]  ⊗ ∫dv f(v)(…)
```

There are exactly **four** channels by which `z` enters, and no others:

1. `Ω_p(z)` — the probe attenuates itself. *Vanishes identically in strict linear response*, because
   the weak-probe `χ_p` of spec 06 §2.4 is independent of `Ω_p` (it is `2N℘²σ_eg/(ε₀ħΩ_p)` with
   `σ_eg ∝ Ω_p`). **This is the load-bearing fact of §2.3.**
2. `Ω_c(z)` — coupling depletion (§2.4). Also `O(Ω_p²)`.
3. `N(z), T(z)` — density/temperature gradients (cold spots, stem, window heating). Externally
   imposed; not generated by the optics.
4. Non-local transport: an atom at `z` carries coherence created upstream (§2.6).

---

### 2.2 The coupling propagation equation and its sign

The coupling propagates toward `−z`:
`E_c(z,t) = ½ Ê_c(z) e^{i(−k_c z − ω_c t)} + c.c.` The same SVEA reduction gives

```
(10.4)   dÊ_c/dz = − i (k_c/2) χ_c(z) Ê_c(z)
         equivalently, in the coupling's own propagation coordinate s_c = L − z:
         dÊ_c/ds_c = + i (k_c/2) χ_c Ê_c        — identical in form to (10.2)
```

with the co-propagating case recovered by `s_c → z`. **Never write a bare sign**: expose
`geometry ∈ {"counter","co"}` exactly as spec 06 §4.8(iii) already requires for the Doppler term.

The coupling susceptibility, by exact analogy with spec 06 §2.4 (`χ_p = 2N℘_ge² σ_eg/(ε₀ħΩ_p)`):

```
(10.5)   χ_c = 2 N ℘_er² σ_re / (ε₀ ħ Ω_c)
```

with `σ_re = ⟨r|σ|e⟩` the rotating-frame coherence on the driven `e–r` transition and `℘_er` its
dipole matrix element [C·m]. **DERIVED-IN-SPEC**; the derivation is the same three lines as spec
06's, with `ρ_re^lab = σ_re e^{−iω_c t}`.

---

### 2.3 The reduction theorem — *the* falsification anchor

> **Theorem (10.R).** Assume
> **(i)** strict linear response: `χ_p` independent of `Ω_p`;
> **(ii)** undepleted coupling: `Ω_c(z) = Ω_c` for all `z ∈ [0, L]`;
> **(iii)** uniform medium: `N`, `T` and all rate parameters independent of `z`;
> **(iv)** local response (no atomic transport; see §2.6).
> Then `χ_p(z) ≡ χ_p` is a constant, (10.2) integrates in closed form, and
>
> ```
> (10.6)   Ê_p(L) = Ê_p(0) · exp( i k_p χ_p L / 2 )
>          T = exp( − k_p Im χ_p L ) = exp(−OD),   Δφ = (k_p L/2) Re χ_p
> ```
>
> **exactly, for every L and every optical depth.** There is no `OD ≪ 1` in the hypothesis.

**This is not a RydSim claim.** It is Eq. (36) of the fetched primary source — Finkelstein, Bali,
Firstenberg & Novikova, *A practical guide to electromagnetically induced transparency in atomic
vapor*, New J. Phys. **25**, 035001 (2023), arXiv:2205.10959 (ar5iv full text fetched 2026-08-11),
quoted verbatim:

> `E_out(δ) = ∫d²k⊥ E_in(δ, k⃗⊥) e^{i k_z L ∫d²v w(v⃗) χ_p(δ, k⊥, v)}` — Eq. (36)

i.e. the Doppler-averaged susceptibility appears **once**, multiplied by the full length `L`, inside
a single exponential. That is (10.6) with the transverse decomposition of §2.9 made explicit.
**VERIFIED.**

**Corollary 10.R.1 — the OD ceiling in the shipped code is operational, not physical.**
`rydsim.experiment.LadderConfig.max_optical_depth = 5.0` (`src/rydsim/experiment.py:102`, raised at
`:317`) refuses above `OD = 5`. The docstring at `:92–101` already states the correct reason —
*"In the weak-probe limit Beer-Lambert with the EIT-suppressed chi is exact at any OD … moderate OD
is VALID physics — merely a poor operating point"* — and **that docstring is right**. The gate
exists because `T < 0.7 %` makes the transduction slope numerically dead and the derived NEF
diverges to a number that looks like a result (audit CRIT-2), **not** because Beer–Lambert fails.
The spec-06 §7.2 phrasing "*breaks down for optically thick cells (α ℓ ≳ 1 with the coupling also
attenuated)*" is imprecise: the parenthesis is the actual condition; `αℓ ≳ 1` alone is not
sufficient for breakdown. See §7 amendment **A-2**.

**Corollary 10.R.2 — condition (iv) is not independent.** In strict linear response,
`σ_eg(z,v) ∝ Ω_p(z)`, so the coherence *does* vary with `z` even under (i)–(iii). The transport
term does not vanish; it produces a complex shift of every coherence decay rate — see Eq. (10.13).
Condition (iv) is therefore a genuine, quantified approximation, not a corollary of (i)–(iii). This
is corrected here relative to a naive reading of "uniform fields ⇒ local response".

**Falsification test (benchmark 10/P-1, RELEASE-GATING).** With (i)–(iii) imposed by construction,
the `z`-stepped solver must reproduce (10.6) to machine precision. **Measured this session** (RK4,
`L = 5 cm`, `χ = 10⁻⁶(0.3 + 1.0i)`, `OD = 0.4026`): relative error in `Ê_p(L)`
`4.04×10⁻⁶` (1 step) → `3.48×10⁻¹⁰` (10 steps) → `1.36×10⁻¹⁵` (1000 steps). Fitted convergence
order **4.07** (RK4 expectation 4). A solver that cannot pass P-1 has a sign, factor-of-2, or
`k_p`-convention bug and must not be used at any OD.

**What actually breaks, ordered by magnitude.** Sections 2.4, 2.7, 2.8 show that channels (1), (2)
and radiation trapping are *all* driven by one quantity — the **probe photon flux `Φ_p`** — and are
therefore fenced by a single gate (§4.1). That unification is the main design result of this
document.

---

### 2.4 Coupling depletion — the term the current model omits entirely

#### 2.4.1 It is `O(Ω_p²)`, i.e. the same order as probe saturation

In the weak-probe perturbative hierarchy of spec 06 §2.4 (`σ_gg = 1 + O(Ω_p²)`;
`σ_eg, σ_rg = O(Ω_p)`), the coherence `σ_re` between two *excited* states is **second order** in
`Ω_p`. Hence, from (10.5):

```
(10.7)   χ_c = O(Ω_p²)  →  χ_c ≡ 0 identically in the strict linear-response limit
```

**This is the structural reason the shipped model is self-consistent.** The very approximation that
makes `χ_p` probe-independent (and therefore makes Beer–Lambert exact, Theorem 10.R) makes the
coupling absorption vanish. **Normative consequence:** *a model that retains coupling depletion
must also retain probe saturation, and vice versa — they are the same `O(Ω_p²)` correction.* Any
implementation that adds one without the other is inconsistent at the order it claims to work to.

**A sharpening of the code's stated rationale.** `experiment.py:95` says the coupling is undepleted
"because the intermediate state is thermally empty". The *thermal* population of 5P₃/₂ is
`exp(−hc/λ_p k_BT)` = **2.0×10⁻²⁷ at 300 K, 9.5×10⁻²¹ at 400 K, 9.6×10⁻¹⁷ at 500 K** (computed this
session) — true, but not the operative fact. What actually matters is the *optically driven*
population, `ρ_ee ≈ 1.2×10⁻²` in the resonant velocity class at `Ω_p/2π = 0.68 MHz`. The correct
fence is the photon-flux bound below, not the Boltzmann factor.

#### 2.4.2 The exact photon-number (Manley–Rowe) bound

In CW steady state, count photons. In the closed 3-level ladder:

- every `g→e` excitation consumes exactly one **probe** photon;
- every net `e→r` transfer consumes exactly one **coupling** photon (stimulated `r→e` returns one);
- in steady state the net `e→r` flux equals the loss rate of `|r⟩`, i.e. `N ρ_rr (Γ_r + γ_t + …)`;
- the probe absorption rate density is `N ρ_ee Γ_e + N ρ_rr (Γ_r + γ_t + …)`.

Therefore, **exactly**:

```
(10.8)   |dΦ_c/dz|  =  N ρ_rr (Γ_r + γ_t + …)  ≤  |dΦ_p/dz| = α_p Φ_p
```

with equality when the two-photon (EIT-shelving) channel dominates, i.e. near two-photon resonance
where the atoms sit in the dark state. Integrating over the cell and converting to intensity
(`I_k = ħω_k Φ_k`, `ω_c/ω_p = λ_p/λ_c`):

```
(10.9)   |ΔI_c| / I_c  ≤  (λ_p/λ_c) · (I_p(0)/I_c) · (1 − T)

(10.10)  |ΔΩ_c| / Ω_c  ≤  ½ (λ_p/λ_c) · (I_p(0)/I_c) · (1 − T)        [since Ω ∝ √I]
```

**DERIVED-IN-SPEC** (elementary photon bookkeeping; every step is an equality except the final
inequality, which is the one-photon-channel branch). Its falsification test is **10/P-3**: the
solved BVP's integrated coupling loss must satisfy (10.8) as an identity against
`N∫ρ_rr(Γ_r+γ_t)dz`, and must never exceed the right-hand side.

Note the bound is **independent of the optical depth and of `N`**: `ΔΦ_c ≤ Φ_p(0)(1−T) ≤ Φ_p(0)`.
Coupling depletion is governed by the **photon-flux ratio of the two beams**, not by how thick the
cell is. This is the single most important quantitative statement in §2.4.

#### 2.4.3 When does it matter? — numbers

Evaluated this session for Rb-87, `λ_p/λ_c = 1.625503`, `w₀ = 1 mm` for both beams:

| Configuration | `\|ΔΩ_c/Ω_c\|` bound |
|---|---|
| Standard: `P_p = 1 µW`, `P_c = 30 mW`, full absorption (`T→0`) | **2.71×10⁻⁵** |
| Hot cell: `P_p = 50 µW`, `P_c = 10 mW`, `T = 0.1` | **3.66×10⁻³** |
| At `Ω_p = Ω_c` with `℘_er/℘_ge = 4.92×10⁻³` (5P₃/₂→50D) | **1.97×10⁻⁵** |

In Rabi/dipole form — the version that shows *why* it is small — using
`I_p/I_c = (Ω_p/Ω_c)²(℘_er/℘_ge)²`:

```
(10.11)  |ΔΩ_c|/Ω_c  ≤  ½ (λ_p/λ_c) (Ω_p/Ω_c)² (℘_er/℘_ge)² (1 − T)
```

The Rydberg coupling dipole is **~200× smaller** than the D2 probe dipole, so reaching a comparable
Rabi frequency costs `(℘_ge/℘_er)² ≈ 4×10⁴` times the intensity. **Coupling depletion is small
because the coupling beam is intrinsically intense, not because the medium is thin.**

**Threshold statement.** Requiring `|ΔΩ_c|/Ω_c ≤ η_Ωc` at the spec-05 probe gate
`I_p ≤ 0.01 I_sat` (`I_sat,far-π = 2.50399(73) mW/cm² = 25.04 W/m²`, Steck, VERIFIED via spec 05
§2.g) gives, for `η_Ωc = 1 %` and `T → 0`:

```
(10.12)  I_c ≥ ½ (λ_p/λ_c) I_p(0) / η_Ωc = 20.35 W/m² = 0.81 I_sat
                                          ≈ 32 µW at w₀ = 1 mm
```

Published Rydberg-EIT experiments run 10–700 mW of coupling. **Coupling depletion is therefore
negligible by 3–4 orders of magnitude in every realistic configuration, and this spec's honest
conclusion is that the term the current model omits is genuinely omissible — with the number to
prove it.** The regimes where it is *not*: (a) coupling powers below ~100 µW at mm waists;
(b) a second strong field populating `|e⟩` (repump, or a second probe); (c) incoherent `|e⟩`
population from radiation trapping (§2.8) — *not* covered by (10.8)'s coherent bookkeeping unless
the trapped photons are also counted, which they are, since they originate from the same `Φ_p`;
(d) micro-cells where the coupling is not the intense beam.

#### 2.4.4 Ratio to probe saturation

Dividing (10.10) by the saturation parameter `s_sat = I_p/I_sat`:

```
(ΔΩ_c/Ω_c) / s_sat = ½ (λ_p/λ_c) (1 − T) I_sat / I_c = 1.07×10⁻³
```

for `P_c = 30 mW`, `w₀ = 1 mm` (`I_c = 1.91×10⁴ W/m² = 763 I_sat`). **Probe saturation bites
~1000× harder than coupling depletion.** Enforcing the probe gate automatically enforces the
coupling gate. This is the first leg of the unification in §4.1.

---

### 2.5 The counter-propagating geometry: a two-point boundary-value problem

With `χ_c ≠ 0`, Eqs. (10.2) and (10.4) form a coupled system whose data are split between the two
faces of the cell:

```
        dΩ_p/dz = + i (k_p/2) χ_p[Ω_p(z), Ω_c(z)] Ω_p(z)     Ω_p(0) = Ω_p^in    (probe enters at z = 0)
        dΩ_c/dz = − i (k_c/2) χ_c[Ω_p(z), Ω_c(z)] Ω_c(z)     Ω_c(L) = Ω_c^in    (coupling enters at z = L)
```

This is a **two-point BVP, not an initial-value problem.** It cannot be marched: `Ω_c(0)` is
unknown, and `Ω_p(L)` — the observable — is unknown. The standard remedies are shooting on
`Ω_c(0)` (or on `Ω_p(L)`) with a scalar root-find, or a relaxation/collocation sweep. The
counter-propagating-fields → two-point-BVP → shooting-method structure for a **cascade (ladder)**
atomic ensemble is stated in H.-H. Jen, *Theory of Light-Matter Interactions in Cascade and Diamond
Type Atomic Ensembles*, PhD thesis, Georgia Institute of Technology (2010), arXiv:1106.2082
(ar5iv fetched 2026-08-11) — **VERIFIED at the structural level only**; the counter-propagating
fields there are signal/idler, not probe/coupling, so the *structure* is corroborated, not the
application. The BVP formulation for probe/coupling in a Rydberg cell is **DERIVED-IN-SPEC**.

**When does the boundary actually matter?** Exactly when the coupling is measurably depleted:
if `|ΔΩ_c|/Ω_c` from (10.10) is below the accuracy target `η_Ωc`, then `Ω_c(z) ≡ Ω_c^in` to that
accuracy, the coupling equation decouples, and the BVP **degenerates into an IVP** which the
existing single-`z` code already solves exactly (Theorem 10.R). Substituting the operating numbers
of §2.4.3, the BVP is genuinely required only when `I_c ≲ 0.8 I_sat/η_Ωc·(I_p/0.01 I_sat)` — i.e.
sub-100-µW coupling. **Normative:** the implementation must *compute* (10.10) and take the IVP
branch only when it passes; it must never assume it.

**A geometric result that favours counter-propagation.** The coupling enters at `z = L`, which is
the probe's **exit** face. The coupling is therefore *least* depleted precisely in the layer that
sets the transmitted probe amplitude, while the most-depleted coupling sits at `z ≈ 0` where the
probe is strong and the EIT feature contributes least to the exit signal. In co-propagation the
ordering reverses and the depletion accumulates *into* the signal-generating layer.
**DERIVED-IN-SPEC; falsification test 10/P-6:** at identical total depletion, the counter-prop
transmitted spectrum must be perturbed less than the co-prop one. If the test comes out the other
way, this paragraph is wrong and must be struck, not softened.

**Two further two-point structures, for the record.** (a) The atoms themselves obey a two-point
condition: those with `v_z > 0` enter the illuminated region at `z = 0`, those with `v_z < 0` at
`z = L`. This is subdominant to the *transverse* transit channel by the ratio of traversal times —
`L/σ_v = 295 µs` axially versus `w₀/⟨v⊥⟩ = 4.71 µs` transversely, a factor **62.7** (Rb-87, 300 K,
`w₀ = 1 mm`, `L = 5 cm`) — so the spec-06 §2.2 measure-and-replace channel already dominates it and
the axial boundary condition may be dropped **provided that ratio is computed and reported, not
assumed.** (b) In a hot cell the RF field is itself a standing wave with its own boundary problem;
that is spec 05/07 territory and out of scope here.

---

### 2.6 Doppler and propagation: how the two integrals compose, and why they do not commute

#### 2.6.1 The exact problem

The correct object is the **phase-space density matrix** `ϱ(z, v, t)` obeying a Boltzmann–Bloch
transport equation. **VERIFIED** against Firstenberg, Shuker, Ron & Davidson, *Colloquium: Coherent
diffusion of polaritons in atomic media*, Rev. Mod. Phys. **85**, 941 (2013), arXiv:1207.6748
(ar5iv fetched 2026-08-11), **Eq. (14)**, quoted verbatim:

> `(∂_t + v⃗·∂_r⃗) ϱ̃_{ss'} + (∂_t ϱ̃_{ss'})_{col.} = Σ_i (∂_t ρ_{ss'}^i) δ(r⃗−r⃗_i(t)) δ(v⃗−v⃗_i(t))`

Specialized to CW steady state, 1-D axial motion, no velocity-changing collisions (evacuated cell —
spec 05 §7.3), with the transverse transit channel already inside the Liouvillian:

```
(10.13a)  v ∂_z ϱ(z,v) = 𝓛[Ω_p(z), Ω_c(z); v] ϱ(z,v)
(10.13b)  P̂_p(z) = 2 N ℘_ge ∫dv f(v) ϱ_eg(z,v)     ⇒  χ_p(z) = 2N℘²/(ε₀ħΩ_p(z)) ∫dv f(v) ϱ_eg(z,v)
```

The polarization source form (10.13b) is **VERIFIED** against Ogden *et al.* PRL 123, 243604,
**Eq. (8)**, quoted verbatim: `𝐏(z,t) = 𝒩 ∫_{-∞}^{∞} f(v_z) Tr[𝐝 ρ(z,t;v_z)] dv_z`.

**The local approximation used by every RydSim path to date** is: drop `v ∂_z ϱ`, so that
`ϱ(z,v) = ϱ_ss[Ω(z); v]` is the *local* steady state and `χ_p(z) = ⟨χ_v⟩(Ω(z))`.

#### 2.6.2 They do not commute — the exact first-order statement

Write the velocity average `𝒜[·] = ∫dv f(v)[·]` and the `z`-propagation map `𝒫_L[·]`. Because the
propagator generated by `𝓛(v)` **depends on `v`**, `𝒜∘𝒫_L ≠ 𝒫_L∘𝒜` in general. The local
approximation is precisely the assertion that they commute.

The size of the error is computable in closed form for the linear-response case. Write the
linearized coherence equation as `v ∂_z σ(z,v) = M(v) σ(z,v) + b(v) Ω_p(z)` with `M` a constant
(in `z`) matrix. With `Ω_p(z) = Ω_p(0)e^{μz}`, `μ = i k_p χ_p/2`, the **exact** solution is
`σ = A e^{μz}` with

```
(10.13c)  A = −(M − vμ)⁻¹ b Ω_p(0)        vs.  A_local = −M⁻¹ b Ω_p(0)
```

i.e. **ballistic transport shifts every coherence decay rate by `−vμ`**. Splitting
`μ = i k_p Reχ/2 − α_p/2`:

- the *imaginary* part shifts the detuning by `v k_p Reχ/2` — a relative correction `Reχ/2 ~ 5×10⁻⁷`
  to the Doppler shift `k_p v`. **Negligible always** (this is the medium-modified wavevector, and
  `|χ| ~ 10⁻⁶`).
- the *real* part shifts the decay rate by `−v α_p/2`. This is the whole effect.

Define the **non-locality parameter**

```
(10.14)  ε_transp(v) ≡ |v| α_p / (2 γ_min)  =  |v| / (γ_min L_grad),    L_grad ≡ 2/α_p
                     =  |v| · OD / (2 γ_min L)     for a uniform cell
```

with `γ_min` the *slowest* coherence decay rate carrying the observable — for the EIT/AT feature
that is `γ_gr` (transit-dominated, `2π×39.8 kHz` for Rb-87 at 300 K, `w₀ = 1 mm`, spec 05 §2.e /
ruling R-3), **not** `γ_ge`.

**Numbers (computed this session, Rb-87 300 K, `L = 5 cm`, `γ_gr = γ_t = 2.50×10⁵ rad/s`,
`v_c = 20 m/s` = the class range spanned by a `±2π×20 MHz` probe scan):**

| Optical depth | `α_p` [m⁻¹] | `L_grad` | `ε_transp` |
|---|---|---|---|
| 1 | 20 | 100 mm | 8.0×10⁻⁴ |
| 10 | 200 | 10 mm | 8.0×10⁻³ |
| 100 | 2000 | 1 mm | **8.0×10⁻²** |

Equivalently `ε_transp ≈ 8×10⁻⁴ × OD` for that cell. A 200 µm MEMS cell at the same OD is ~12×
worse (`ε ≈ 10⁻² × OD`), because shrinking `L` raises `α_p` faster than shrinking `w₀` raises
`γ_t`.

**The cost of getting it right.** The local approximation replaces one scalar ODE in `z` by a
system of `N_v × N_levels²` coupled ODEs in `z` (one trajectory per velocity class, with
`N_v ~ 10⁴–10⁵` per spec 05 §2.d / ruling R-2), **and** turns the atomic problem into its own
two-point BVP (`v>0` classes specified at `z=0`, `v<0` at `z=L`). That is a `~10⁶`-dimensional BVP
in place of a scalar quadrature — the reason no production Rydberg-EIT code does it, and the reason
the local approximation must be *fenced*, not *fixed*, at first release.

**Partial cancellation — a claim that must be measured, not assumed.** The shift `−vα_p/2` is
**odd in `v`**, and at line centre the velocity-resolved response is even in `v`. The first-order
term therefore cancels in the symmetric Maxwell average, leaving `O(ε_transp²)` at the EIT peak and
`O(ε_transp)` on the Doppler wings. **DERIVED-IN-SPEC — benchmark 10/P-7 exists specifically to
falsify it.** If the measured error at the peak scales linearly in `ε`, this paragraph is wrong.

**Normative fence:** the local-response path may be used when `ε_transp ≤ 0.05` evaluated at the
largest `|v|` carrying ≥1 % of the spectral weight; above that the result must be flagged
`transport_uncontrolled` and, at `ε_transp > 0.2`, refused. `ε_transp` is **always computed and
reported** as part of the spec-00 §4-audit "convergence records" block.

---

### 2.7 Probe saturation with depth: the approximation that improves as you go in

#### 2.7.1 The two independent conditions

The weak-probe expansion of spec 06 §2.4 has **two** small parameters, with different physics:

```
(10.15a)  one-photon saturation:  s_sat = I_p/I_sat = 2Ω_p²/Γ_e²          (Doppler background)
(10.15b)  dark-state depletion:   ρ_rr ≈ Ω_p²/(Ω_p² + Ω_c²)               (the EIT feature)
```

(10.15b) is exact for the ideal dark state `|D⟩ ∝ Ω_c|g⟩ − Ω_p|r⟩`: the ground-state population is
*coherently* transferred to `|r⟩` in the ratio `Ω_p²/Ω_c²`. LITERATURE-RECALL (textbook dark-state
result); it is the physical content of the shipped gate `Ω_p < 0.01·min(Γ_e, Ω_c)` (spec 06 §4.6,
integrity-audit refusal #21). The combined validity condition is

```
(10.16)  Ω_p(z) ≪ min( Γ_e , Ω_c(z) )     for all z ∈ [0, L]
```

`Ω_p ≪ Ω_c` is **VERIFIED** as the stated linear-response condition of the NJP 2023 guide
(fetched: *"this condition applies when Ωp≪Ωc"*).

Numerically, `s_sat = 0.01` ⇔ `Ω_p = Γ_e/14.1`; `s_sat = 0.001` ⇔ `Ω_p = Γ_e/44.7` (computed this
session). **Note a tension with spec 05 §2.f**, which writes "`I_p < 0.01·I_sat` … In Rabi terms
`Ω_p ≲ Γ_e/7`": `Γ_e/7` corresponds to `s_sat = 2/49 = 0.041`, not 0.01. See §7 amendment **A-3**.

#### 2.7.2 It self-heals — with an important caveat

`Ω_p(z) = Ω_p(0) e^{−α_p z/2}`, so **both** parameters in (10.15) decrease monotonically into the
cell. The weak-probe approximation is therefore **worst at `z = 0` and best at `z = L`**, and the
entrance condition (10.16) at `z = 0` is *sufficient* for the whole cell (given an undepleted
coupling). The residual `O(s_sat)` error is confined to the first `~2/α_p = L_grad` of the cell and
its contribution to the transmitted amplitude is suppressed by the subsequent attenuation.

**The caveat that must be stated, because it is exactly where the metrology lives.** The
attenuation is strongly detuning-dependent: at the EIT resonance the probe is *transmitted* — that
is the entire point of the scheme — so `α_p` is small there and the probe stays at its entrance
value all the way through. The "it gets better with depth" argument is therefore **weakest precisely
at the operating point** (the EIT peak / the transduction-slope flank) and strongest on the Doppler
wings, which carry no signal. Quantitatively, from the shipped engine (Rb-87, `Ω_c/2π = 5 MHz`,
`γ_t/2π = 39.8 kHz`, 5 cm cell) the EIT-peak OD is **0.50 / 1.87 / 10.15** at **300 / 313 / 333 K**
against a coupling-off line-centre OD of **1.12 / 4.16 / 22.6** — the EIT feature is attenuated
about **2.2× less** than the background. Any claim that saturation is "burned off by depth" must
therefore be evaluated **on the EIT resonance**, not on the line-centre OD.

**Falsification test 10/P-8:** with a saturable (non-perturbative) `χ_p(Ω_p)`, the local `s_sat(z)`
must be monotone decreasing, and `T(s_sat(0))` must approach the linear-response `T` with error
`O(s_sat(0))` — measured slope 1.0 ± 0.15 on a log-log fit over `s_sat(0) ∈ [10⁻³, 10⁻¹]`.

---

### 2.8 Radiation trapping and reabsorption — a criterion, not an assertion

Spec 06 §7.4 asserts "no radiation trapping"; spec 05 §7 does not mention it. That assertion needs
a number, and the number is not obviously safe at hot-cell densities.

**Why it is a live effect here.** Radiation trapping in an optically thick alkali vapor
*measurably* raises the effective decay rate of a ground-state coherence: Matsko, Novikova, Scully
& Welch, *Radiation trapping in coherent media*, Phys. Rev. Lett. **87**, 133601 (2001),
arXiv:quant-ph/0101147 — abstract fetched 2026-08-11, quoted verbatim: *"the effective decay rate
of Zeeman coherence … increases significantly with the atomic density. We explain this phenomenon as
the result of radiation trapping. Our study shows that radiation trapping must be taken into account
to fully understand many electromagnetically induced transparency experiments with optically thick
media."* **VERIFIED (abstract level).** The mechanism transfers directly to a Rydberg ladder: a
spontaneously scattered probe photon, reabsorbed by an atom in the dark state `|D⟩`, projects that
atom and destroys its `g–r` coherence — an addition to `γ_gr`, **not** a change to `Γ_e` (a real
emission event occurs at rate `Γ_e` whatever the photon subsequently does, so `γ_ge = Γ_e/2 + …`
is untouched).

**How thick the vapor is to its own fluorescence.** The Doppler-limited peak cross-section is

```
(10.17)  σ_D,peak = √(π ln 2) · (Γ_e/2π)/Δν_D · σ₀,     σ₀ = 3λ²/2π
         √(π ln 2) = 1.475665 (exact closed form)
```

**DERIVED-IN-SPEC** (ratio of unit-area Gaussian and Lorentzian peak heights). Rb-87 D2 at 300 K:
`σ_D,peak/σ₀ = 0.017509`, `σ_D,peak = 5.089×10⁻¹⁵ m²`.

Two independent confirmations, both this session: (a) reproducing spec 05's benchmark B9a
(natural Rb, 75 mm, 25.0 °C, ⁸⁵Rb `F=3` dip) from (10.17) alone gives **OD = 1.323** against the
spec's **1.298** — 1.9 %, the residual being the neglected wings of the other hyperfine lines;
(b) the shipped engine's own `α_p/N` at Rb 300 K line centre with the coupling off is
**5.129×10⁻¹⁵ m²** against (10.17)'s **5.089×10⁻¹⁵ m²** — **0.8 %**. (b) is an *absolute*
cross-check of the susceptibility chain that is independent of the spec-06 B-2 sum rule, and is
promoted to benchmark **10/P-10**.

**The model-free bound.** In steady state an atom cannot absorb trapped photons faster than the
ensemble emits them per atom, and the ensemble emission rate density is bounded by the probe
absorption rate density `α_p Φ_p`. Hence

```
(10.18)  γ_trap  ≤  α_p Φ_p / N  =  σ_eff Φ_p  =  (α_p/N) · I_p/(ħω_p)
         with the actual value  γ_trap = (1 − g_esc) · σ_eff Φ_p
```

**DERIVED-IN-SPEC.** Two properties make (10.18) the right fence: it needs **no escape-factor
theory** (`g_esc ≥ 0` suffices), and it is built from quantities the engine already computes
(`α_p` from `χ`, `N` from `rydsim.cell`, `Φ_p` from beam geometry). It is also **independent of
`N`** — more atoms means more emitters *and* more absorbers — so the knob is **probe photon flux**.

**Numbers (Rb-87, 300 K, `w₀ = 1 mm`, `σ_eff = 5.129×10⁻¹⁵ m²`, against `γ_t = 2π×39.8 kHz`):**

| Probe power | `I₀` [W/m²] | `γ_trap` bound | as % of `γ_gr` |
|---|---|---|---|
| 0.39 µW (`= 0.01 I_sat,far-π`) | 0.2546 | `2π×0.82 kHz` | **2.1 %** |
| 1 µW | 0.637 | `2π×2.04 kHz` | 5.1 % |
| 10 µW | 6.37 | `2π×20.4 kHz` | 51 % |
| 100 µW | 63.7 | `2π×204 kHz` | **513 %** |

**Conclusion, and it is the second leg of the unification:** *enforcing the spec-05 probe-saturation
gate `I_p ≤ 0.01 I_sat` bounds radiation trapping to ≤ 2.1 % of the transit-limited `γ_gr`.* Above
~10 µW at mm waists, trapping can no longer be dismissed and the escape factor must be modelled or
the output refused.

**MISSING.** The closed-form Holstein escape factor `g_esc(k₀R)` for a Doppler-broadened line in a
cylinder is **not sourced in this session.** The one hard datum obtained is a *lifetime-enhancement*
factor from an alkali-cell measurement — `g₁ = 1.6` (i.e. a 60 % lengthening of the Cs 6²P₃/₂
lifetime) at attenuation parameter `αr ≈ 0.66` at 23 °C — from arXiv:1912.10089 (ar5iv fetched
2026-08-11), which cites the closed forms to **A. Molisch, B. P. Oehry, W. Schupita & G. Magerl,
JQSRT 49, 361–370 (1993)** and ultimately **T. Holstein, Phys. Rev. 72, 1212 (1947) / 83, 1159
(1951)**. *Resolution:* fetch Molisch 1993 (or Holstein 1951) and transcribe `g_esc` with its
geometry and validity range; until then only the bound (10.18) may be used, and no `g_esc`-dependent
number may ship. **Note the convention trap** on this row: the fetched `g₁ = 1.6` is
`τ_eff/τ = 1/g_esc`, the reciprocal of the escape factor — transcribing it as `g_esc` would invert
the physics.

**UNVERIFIED sub-claim** (fenced by the bound, so it cannot corrupt a shipped number): that a
reabsorption event destroys the `g–r` coherence with **unit** probability. It is a projective
measurement of the atom's internal state, so unit probability is the natural assumption and it is
conservative (it maximizes `γ_trap`), but it is not sourced.

---

### 2.9 Transverse structure and dispersive self-lensing

Under A5 (no inter-shell diffraction), each radial shell propagates independently under (10.2) with
its own `Ω_p(r), Ω_c(r)`, and the measured transmission is the spec-05 §2.g power-weighted average
over `s_radial = 2r²/w₀²` — the **`z`-propagation must be performed inside the radial quadrature,
never after it**, because `exp(⟨·⟩) ≠ ⟨exp(·)⟩` and the discrepancy grows with OD. The
transverse-wavevector form of (10.6) — NJP 2023 Eq. (36), fetched — is the same statement.

**Self-lensing fence.** The medium is a graded-index lens of accumulated differential phase

```
(10.19)  Δφ_lens = (k_p L/2) · [ Re χ_p(r=0) − Re χ_p(r=w₀) ]
```

Near the EIT flank `|Re χ| ~ |Im χ|`, so `Δφ_lens ~ (OD/2)·(fractional variation of χ across the
beam)`. At `OD = 50` with a 20 % radial variation of `χ`, `Δφ_lens ≈ 5 rad` — the paraxial-shell
model has failed. **Normative:** compute (10.19); flag above 0.3 rad, refuse above 1 rad. Spec 05
§7.6 already declares "*Lensing by the dispersive medium at very high OD is not modeled*" —
(10.19) is the missing criterion for that declaration. **DERIVED-IN-SPEC**; a full diffraction
calculation is out of scope (§6).

---

## 3. Constants and parameters

**No new empirical constant is introduced by this document.** The table records the derived
quantities and the borrowed ones a propagation implementation touches, with sources.

| Quantity | Symbol | Value | Unit | Source | Confidence |
|---|---|---|---|---|---|
| Vacuum probe wavenumber | `k_p` | `2π/λ_p` (**no extra 2 or 4π**) | rad/m | spec 05 §2.f verbatim | VERIFIED |
| Propagation coefficient | — | `dÊ/dz = +i(k/2)χÊ` | — | Ogden PRL 123, 243604 Eq. (6), fetched | **VERIFIED** |
| Doppler-averaged source | — | `P = 𝒩∫f(v_z)Tr[dρ(z;v_z)]dv_z` | C/m² | Ogden PRL 123, 243604 Eq. (8), fetched | **VERIFIED** |
| Resonant absorption coefficient | `α₀` | `k_p N μ₁₃²/(2ħε₀γ₁₃)` | m⁻¹ | NJP 25, 035001 Eq. (12), fetched | **VERIFIED** |
| Optical-depth convention | `OD` | `2α₀L` ≡ `k_p Im χ L` (identity, §5 P-10b) | — | NJP 25, 035001, fetched | **VERIFIED** |
| Closed-form thin-cell solution | — | `E_out = E_in e^{ik_zL∫d²v w(v)χ_p}` | — | NJP 25, 035001 Eq. (36), fetched | **VERIFIED** |
| Transport equation | — | `(∂_t + v·∂_r)ϱ̃ + collisions = source` | — | RMP 85, 941 Eq. (14), fetched | **VERIFIED** |
| Linear-response condition | — | `Ω_p ≪ Ω_c` | — | NJP 25, 035001, fetched | **VERIFIED** |
| Wavelength ratio (Rb fixture) | `λ_p/λ_c` | 1.625503 (780.241209686 / 480.0 nm) | — | spec 00 lock #10 / R-15 | VERIFIED (fixture only) |
| Rb-87 D2 resonant cross-section | `σ₀` | `3λ²/2π` = 2.9067×10⁻¹³ | m² | spec 05 §2.f / Steck | VERIFIED |
| Doppler peak / resonant ratio | — | `√(π ln2)·(Γ_e/2π)/Δν_D`; `√(π ln2)` = 1.475665 | — | Eq. (10.17) | **DERIVED-IN-SPEC** (2 checks, §2.8) |
| Rb-87 D2 Doppler peak σ, 300 K | `σ_D,peak` | 5.089×10⁻¹⁵ (analytic) / 5.129×10⁻¹⁵ (engine) | m² | Eq. (10.17) vs `rydsim` run | **DERIVED**, 0.8 % closure |
| `I_sat`, far-detuned π, Rb-87 D2 | `I_sat` | 2.50399(73) mW/cm² = 25.04 W/m² | W/m² | Steck via spec 05 §2.g | VERIFIED |
| Transit rate, Rb-87 300 K, `w₀`=1 mm | `γ_t` | `2π×39.79 kHz` (`√(2ln2)⟨v⊥⟩/w₀`) | rad/s | spec 05 §2.e / ruling R-3 | VERIFIED (recomputed) |
| Rb-87 300 K speeds | `v_p/σ_v/⟨v⊥⟩` | 239.6 / 169.4 / 212.3 | m/s | spec 05 §2.c | VERIFIED (recomputed) |
| Axial/transverse traversal ratio | — | 62.7 (`L=5 cm`, `w₀=1 mm`, 300 K) | — | §2.5 | **DERIVED-IN-SPEC** |
| Thermal 5P₃/₂ fraction | — | 2.0×10⁻²⁷ / 9.5×10⁻²¹ / 9.6×10⁻¹⁷ at 300/400/500 K | — | §2.4.1 | **DERIVED-IN-SPEC** |
| Coupling-depletion bound | — | Eq. (10.10); 2.71×10⁻⁵ at 1 µW / 30 mW | — | §2.4.2–3 | **DERIVED-IN-SPEC** |
| Depletion/saturation ratio | — | `½(λ_p/λ_c)(1−T)I_sat/I_c` = 1.07×10⁻³ | — | §2.4.4 | **DERIVED-IN-SPEC** |
| Trapping bound | `γ_trap` | Eq. (10.18); `2π×0.82 kHz` at `I_p = 0.01 I_sat` | rad/s | §2.8 | **DERIVED-IN-SPEC** |
| Holstein escape factor | `g_esc` | closed form | — | Molisch JQSRT 49, 361 (1993); Holstein PR 83, 1159 (1951) | **MISSING** (unfetched) |
| Engine OD, Rb 5 cm, `Ω_c/2π`=5 MHz | — | EIT peak 0.50 / 1.87 / 10.15 at 300/313/333 K | — | `rydsim` run this session | DERIVED (measured) |

---

## 4. Numerical method and named pitfalls

### 4.1 The unified gate (normative)

Sections 2.4, 2.7 and 2.8 all reduce to the **probe photon flux**. One gate therefore fences three
physics channels:

```
GATE-P (probe flux):   s_sat(0) = I_p(0)/I_sat ≤ 0.01
   ⇒ probe saturation error                  ≤ O(1 %)                (§2.7)
   ⇒ coupling depletion |ΔΩ_c|/Ω_c           ≤ 1.07×10⁻³ × s_sat = 1.1×10⁻⁵   for I_c ≥ 763 I_sat  (§2.4.4)
   ⇒ radiation trapping γ_trap/γ_gr          ≤ 2.1 %                 (§2.8)

GATE-C (coupling intensity):  I_c ≥ ½(λ_p/λ_c)·I_p(0)/η_Ωc          (Eq. 10.12; 32 µW at w₀=1 mm, η=1 %)
GATE-T (transport locality):  ε_transp ≤ 0.05                        (Eq. 10.14; flag 0.05, refuse 0.2)
GATE-L (self-lensing):        |Δφ_lens| ≤ 0.3 rad                     (Eq. 10.19; refuse at 1 rad)
```

All four are **computed and reported**, never assumed, and appear in the spec-00 §4-audit
provenance block alongside the velocity-grid convergence record. **`max_optical_depth` is not in
this list** — see §7 amendment A-1.

### 4.2 Integrator

- **Constant-`χ` fast path (Theorem 10.R).** When GATE-C passes with margin and the medium is
  uniform, take the closed form (10.6). Cost: one velocity average. This is what the code does
  today, and it is *exact*, not "thin-medium".
- **`z`-stepped path.** RK4 on (10.2) with the spec-05 §2.f step control `k_p Im χ Δz ≤ 0.05` and
  `|ΔΩ_p|/|Ω_p| ≤ 2 %` per step. Measured order **4.07**; 10 steps over a 5 cm cell already reaches
  `3.5×10⁻¹⁰` relative. Convergence recorded by step halving (`|ΔT| < 10⁻⁴`), as data, not prose.
- **BVP path.** Shooting on `Ω_c(0)` with a scalar secant root-find; the residual is monotone in the
  shot value for `|ΔΩ_c|/Ω_c ≪ 1`, so 3–5 iterations suffice. Seed from the IVP solution.
  Relaxation/collocation is the fallback if the shooting Jacobian becomes ill-conditioned (it will,
  if depletion ever approaches unity — at which point GATE-C has already refused).
- **Velocity average inside the `z` loop.** Re-solve the velocity integral at each `z` level (spec
  05 §2.f says so explicitly). Reusing a single `⟨χ⟩` across `z` is only valid when Theorem 10.R
  applies — in which case the `z` loop is unnecessary anyway.

### 4.3 Named pitfalls

1. **Sign of `i` in (10.2).** `−i` gives gain. Caught by 10/P-1.
2. **`k_p` vs `k_p/2` vs `2k_p`.** The propagation coefficient is `k_p/2`; the *intensity*
   coefficient is `k_p Im χ`. Mixing them is a silent factor of 2 in every OD. Caught by 10/P-1
   and by the OD-convention identity 10/P-10b.
3. **`k` in the medium vs in vacuum.** `k_p = 2π/λ_p` (vacuum) — spec 05 §2.f verbatim. Using
   `n_refr k_p` double-counts `χ` and is a `10⁻⁶` error masquerading as rigor.
4. **Marching the BVP.** `Ω_c(0)` is not data. An IVP march that "just uses `Ω_c^in` at `z=0`"
   silently solves the **co-propagating** problem. Caught by 10/P-5 and 10/P-6.
5. **Averaging before propagating.** Radial (and any parameter) averaging must be *outside* the
   `z` integration: `⟨exp(−OD(r))⟩ ≠ exp(−⟨OD(r)⟩)`, and the gap grows with OD. Spec 05 §4.8's
   "peak-intensity-only simulation is forbidden" is the same pitfall one level up.
6. **Assuming the local approximation.** `ε_transp` (10.14) is `8×10⁻⁴ × OD` for a 5 cm cell and
   ~12× worse in a MEMS cell. Compute it.
7. **`γ_min` in (10.14).** Use the *slowest* rate carrying the observable (`γ_gr`, transit-limited),
   not `γ_ge`. Using `γ_ge` understates `ε_transp` by ~76×.
8. **Escape-factor reciprocal.** Published "trapping factors" are often `τ_eff/τ = 1/g_esc > 1`
   (the fetched Cs value `g₁ = 1.6` is of this kind). Transcribing one as `g_esc` inverts the
   physics. See §2.8.
9. **`s` collision.** Saturation parameter vs spec-05 radial variable vs line strength vs PSD.
   `s_sat` / `s_radial` in code, never bare `s`.
10. **`exp(−OD)` underflow.** Guard at `OD > 700` and return `0.0` explicitly (spec 05 §4.7).
    Do not let a `0.0` transmission propagate into a division that produces a finite-looking NEF —
    that is audit CRIT-2 in a new costume.
11. **Photon-flux bookkeeping across isotopes/hyperfine.** (10.8) counts photons per *transition*.
    In a natural-abundance cell the probe absorption is spread over both isotopes and four ground
    levels, but the coupling addresses only the sensed one. The bound (10.10) is still valid (it is
    an upper bound) but becomes loose by the branching fraction — do not tighten it silently.

---

## 5. Validation benchmarks

Common fixture unless stated: Rb-87, `λ_p = 780.241209686 nm`, `λ_c = 480.0 nm` (declared fixture,
ruling R-15), `Γ_e/2π = 6.0666 MHz`, `T = 300 K`, `L = 5 cm`, `w₀ = 1 mm`, `Ω_c/2π = 5 MHz`,
`γ_t/2π = 39.8 kHz`, counter-propagating.

| ID | Quantity | Setup | Expected | Tolerance | Source / type | Confidence |
|---|---|---|---|---|---|---|
| **10/P-1** | `z`-solver vs closed form (10.6) — **RELEASE-GATING** | (i)–(iii) imposed; `OD ∈ {0.1, 1, 5, 50}` | identical | rel ≤ 1e-12 in `T` | Theorem 10.R; measured 1.36e-15 @1000 RK4 steps | **VERIFIED** (identity + measured) |
| 10/P-1b | same at extreme depth | `OD = 200` (`T = 10⁻⁸⁷`) | compare `ln T`, not `T` | rel ≤ 1e-12 in `OD` | underflow-safe restatement of P-1 | VERIFIED (identity) |
| 10/P-2 | integrator order | halve `Δz` ×4 from 1 step | fitted order 4 | 4.0 ± 0.2 | RK4; measured **4.07** this session | VERIFIED (measured) |
| 10/P-3 | photon-flux closure (10.8) | BVP solved with `χ_c` retained | `\|ΔΦ_c\| = N∫ρ_rr(Γ_r+γ_t)dz` **and** `≤ \|ΔΦ_p\|` | equality rel ≤ 1e-6; inequality exact | Manley–Rowe, Eq. (10.8) | DERIVED-IN-SPEC |
| 10/P-4 | coupling-depletion bound (10.10) | `P_p = 1 µW`, `P_c = 30 mW`, `T→0` | solved `\|ΔΩ_c/Ω_c\| ≤ 2.71e-5`, and ≥ half of it when the 2-photon channel dominates | bound never violated; within 2× when dominant | Eq. (10.10), computed this session | DERIVED-IN-SPEC |
| 10/P-5 | BVP vs IVP degeneracy | GATE-C passing by 100× | spectra agree | ≤ the P-4 bound | §2.5 | DERIVED-IN-SPEC |
| 10/P-6 | counter- vs co-prop asymmetry | identical total depletion, both geometries | counter-prop transmitted spectrum perturbed **less** | sign test (boolean) | §2.5 geometric argument | **DERIVED-IN-SPEC — may falsify §2.5** |
| 10/P-7 | transport (locality) error | `v`-resolved transport BVP vs local, `OD ∈ {1, 10, 100}` | error at EIT peak `∝ ε²`, on wings `∝ ε`; magnitude ≈ 8e-2 at `OD = 100` | fitted exponents 2.0 ± 0.3 / 1.0 ± 0.2 | Eq. (10.14) | **DERIVED-IN-SPEC — may falsify §2.6** |
| 10/P-8 | probe-saturation self-healing | saturable `χ_p`, `s_sat(0) ∈ [1e-3, 1e-1]` | `s_sat(z)` monotone ↓; `T` error `∝ s_sat(0)` | slope 1.0 ± 0.15 (log-log) | §2.7 | DERIVED-IN-SPEC |
| 10/P-9 | radiation-trapping bound (10.18) | `I_p = 0.01 I_sat`, 300 K | `γ_trap ≤ 5.13e3 rad/s = 2.1 % of γ_gr`; fence fires above 0.1 `γ_gr` | bound exact; fence boolean | Eq. (10.18), computed this session | DERIVED-IN-SPEC |
| 10/P-10 | absolute-χ cross-check via `σ_eff` | engine `α_p/N`, coupling off, line centre, 300 K | 5.089e-15 m² from Eq. (10.17) | rel ≤ 3 % (**measured 0.8 %**) | independent of the spec-06 B-2 sum rule | **VERIFIED** (two-path closure) |
| 10/P-10b | OD-convention identity | 2-level, on resonance | `k_p Im χ L ≡ 2α₀L` with `α₀ = k_pNμ²/(2ħε₀γ₁₃)` | exact algebra, rel ≤ 1e-14 | NJP 25, 035001 Eq. (12), **fetched** | **VERIFIED** |
| 10/P-11 | Doppler-σ vs spec-05 B9a | natural Rb, 75 mm, 25.0 °C, ⁸⁵Rb F=3 | OD = 1.298 (spec 05 B9a) from Eq. (10.17) + `S₃₄` | ≤ 3 % (**measured 1.9 %**) | cross-spec closure | DERIVED (measured) |
| 10/P-12 | radial-then-propagate ordering | `OD_peak = 5`, Gauss–Laguerre 12 nodes | `⟨exp(−OD(r))⟩ ≠ exp(−⟨OD⟩)`; the two must differ by ≥ 5 % | difference detected (boolean) | pitfall 4.3.5 | DERIVED-IN-SPEC |
| 10/P-13 | self-lensing fence (10.19) | `OD = 50`, 20 % radial `χ` variation | `Δφ_lens ≈ 5 rad` → refusal fires | boolean | Eq. (10.19) | DERIVED-IN-SPEC |
| 10/P-14 | axial vs transverse traversal | `L = 5 cm`, `w₀ = 1 mm`, 300 K | ratio 62.7 | ≤ 1 % | §2.5 | DERIVED (recomputed) |
| 10/P-15 | published-regime coverage | Rb-87 thermal cell | engine reaches `OD = 0.42 → 5.0` over 27 → 65 °C | qualitative (2 booleans) | Su, Liou, Lin & Chen, Opt. Express **30**, 1499 (2022), **fetched** | **VERIFIED** (their stated range) |

`10/P-1` and `10/P-10b` are the convention locks — run first, fail loudly. `10/P-6` and `10/P-7`
exist to **refute** §2.5 and §2.6 respectively; a failure there is a spec edit, not a code fix.

---

## 6. Known limitations and model breakdown

1. **CW steady state only.** No pulse propagation, adiabatons, dark-state polaritons, slow-light
   storage, or scan-rate transients. The group delay `τ_g = (L/2c)·ω dReχ/dω` is computable from
   (10.6) but is *not* validated here.
2. **No four-wave mixing or backward-generated fields.** The NJP 2023 guide states (fetched)
   *"increasing OD results in enhancement of non-linear light-matter interactions, such as four-wave
   mixing."* In a counter-propagating ladder the FWM phase-matching condition and the known
   Rb 5S–5P–5D/nD cascade emissions (e.g. 5.23 µm / 420 nm) are entirely outside this model. **This
   is the least-fenced limitation in the document** — no criterion is offered because none was
   sourced. Open question O-4.
3. **Radiation-trapping escape factor MISSING** (§2.8). Only the bound ships.
4. **Rydberg–Rydberg interactions, ionization, plasma fields and optical bistability.** At high `N`
   and high `Ω_c` the medium becomes *intrinsically* nonlinear and the propagation problem acquires
   multiple solutions. Charge-induced optical bistability in thermal Rydberg vapor is documented
   (Phys. Rev. A **94**, 063820 (2016); Carr, Ritter, Wade, Adams & Weatherill, Phys. Rev. Lett.
   **111**, 113901 (2013) — **citations located this session, papers not fetched,
   LITERATURE-RECALL**). RydSim's single-valued `χ` cannot represent a hysteretic medium; a
   bistability sweep is out of scope, and any high-`N`/high-`Ω_c` result must carry the caveat.
5. **Velocity-changing collisions absent.** Evacuated cells only (spec 05 §7.3). A buffer gas turns
   the ballistic transport of §2.6 into diffusion and changes `ε_transp` qualitatively — refuse,
   do not extrapolate (integrity-audit refusal #16).
6. **Scalar/paraxial shells; no diffraction.** Valid for `L ≪ 2z_R`; self-lensing fenced by (10.19)
   but not modelled.
7. **1-D axial propagation only.** RF standing waves, cell etalon and internal-field ≠
   incident-field effects (spec 06 §2.8 items 6–7) are a cell/EM-module concern and are *not*
   fixed by this document. At high OD they are the dominant metrology systematic, per Holloway
   JAP 121, 233106 (2017).
8. **Single sensed isotope/hyperfine channel in the photon bookkeeping** (pitfall 4.3.11).
9. **The transport correction is fenced, not solved.** The exact `v`-resolved BVP is specified
   (§2.6) but its cost (`~10⁶`-dimensional BVP) is prohibitive at first release. `ε_transp` is a
   *bound on the error*, not a *correction to the answer*.

---

## 7. Proposed amendments to specs 00, 05 and 06 (stated openly, per the brief)

**A-1 (to spec 00, new ruling R-29 — the OD ceiling is operational).**
`max_optical_depth` refuses on a *numerical-conditioning* criterion, and Theorem 10.R shows the
underlying physics is exact at any OD. The refusal message at `experiment.py:317` should say so:
the correct remedy list is "raise the probe power *within* GATE-P, shorten the cell, lower the
temperature, or improve the detection chain" — **not** "implement the z-propagation solver", since
the `z`-solver returns the identical number. Proposed replacement gates: **GATE-P / C / T / L**
(§4.1) *plus* a separate, explicitly-labelled `min_transmitted_power` conditioning gate. This
narrows what the refusal claims while keeping every existing refusal in force.

**A-2 (to spec 06 §7.2 — precision of the thin-medium limitation).**
Current text: *"Breaks down for optically thick cells (α ℓ ≳ 1 with the coupling also attenuated)."*
The parenthesis is the operative condition; `αℓ ≳ 1` alone is not sufficient. Proposed:
*"Exact in strict linear response with an undepleted, uniform coupling (Theorem 10.R), at any αℓ.
Breaks down through probe saturation at the entrance (§2.7), coupling depletion (§2.4), medium
gradients, atomic transport (§2.6), or radiation trapping (§2.8) — each with its own criterion, none
of which is 'OD > 1'."*

**A-3 (to spec 05 §2.f and to integrity-audit refusal #18).**
(a) §2.f writes *"I_p < 0.01·I_sat … In Rabi terms Ω_p ≲ Γ_e/7"*. With `s_sat = 2Ω_p²/Γ_e²`,
`s_sat = 0.01` ⇔ `Ω_p = Γ_e/14.1`; `Γ_e/7` corresponds to `s_sat = 0.041`. One of the two numbers
should move; this spec recommends keeping `I_p < 0.01 I_sat` (it is the sourced form) and correcting
the Rabi restatement to `Γ_e/14`.
(b) Integrity-audit refusal #18 raises `ThickCellError` for `OD > 0.1` through the analytic path.
Per Theorem 10.R that threshold is over-strict by ~2 orders of magnitude and would refuse the
regime the published experiments occupy (`OD = 0.42–5.0`, Opt. Express 30, 1499 (2022), fetched).
Proposed: replace the OD trigger with the GATE-P/C/T/L set, keeping the *refusal* (never a silent
degradation) and keeping `OD > 0.1` as a **warning-level flag** so nothing currently caught becomes
uncaught. **This amendment loosens a fence and must not be adopted without 10/P-1 passing.**

---

## 8. Sources

**Fetched and quoted this session (2026-08-11) — VERIFIED:**

- R. Finkelstein, S. Bali, O. Firstenberg & I. Novikova, *A practical guide to electromagnetically
  induced transparency in atomic vapor*, **New J. Phys. 25, 035001 (2023)**, arXiv:2205.10959
  (ar5iv full text). Taken: `α₀ = k_pNμ₁₃²/(2ħε₀γ₁₃)` Eq. (12); `OD = 2α₀L`; `χ_p` Eq. (11);
  the closed-form transmitted field Eq. (36); the linear-response condition `Ω_p ≪ Ω_c`; the
  Doppler-averaged `χ_ensemble = ∫χ_p(δ−kv_z)w(v_z)dv_z`; transit broadening `Γ_tt = v_th/w₀`;
  the statement that FWM grows with OD.
- T. P. Ogden, K. A. Whittaker, J. Keaveney, S. A. Wrathmall, C. S. Adams & R. M. Potvliege,
  *Quasi-simultons in thermal atomic vapors*, **Phys. Rev. Lett. 123, 243604 (2019)**,
  arXiv:1909.07161 (ar5iv). Taken: propagation Eq. (6)
  `[∂_z + (1/c)∂_t]ℰ_α = (ik/2ε₀)𝒫_α, α = p,c`; Doppler-averaged polarization Eq. (8); the fact
  that **both** fields are propagated self-consistently in a thermal ladder.
- O. Firstenberg, M. Shuker, A. Ron & N. Davidson, *Colloquium: Coherent diffusion of polaritons in
  atomic media*, **Rev. Mod. Phys. 85, 941 (2013)**, arXiv:1207.6748 (ar5iv). Taken: the
  phase-space transport equation, Eq. (14).
- A. B. Matsko, I. Novikova, M. O. Scully & G. R. Welch, *Radiation trapping in coherent media*,
  **Phys. Rev. Lett. 87, 133601 (2001)**, arXiv:quant-ph/0101147 (abstract). Taken: trapping raises
  the effective ground-coherence decay with density and *"must be taken into account to fully
  understand many EIT experiments with optically thick media."*
- H.-J. Su, J.-Y. Liou, I-C. Lin & Y.-H. Chen, *Optimizing the Rydberg EIT spectrum in a thermal
  vapor*, **Opt. Express 30, 1499 (2022)**, arXiv:2111.13408 (abstract + ar5iv). Taken: the
  operating range `OD = 0.42 (27 °C) → 5.0 (65 °C)`, `T = exp[−α Im(Γ_eρ_eg/Ω_p)]` (their Eq. 5,
  a thin-medium exponential applied at OD 5), 13 % peak EIT height.
- H.-H. Jen, *Theory of Light-Matter Interactions in Cascade and Diamond Type Atomic Ensembles*,
  PhD thesis, Georgia Tech (2010), arXiv:1106.2082 (ar5iv). Taken: counter-propagating fields in a
  cascade ensemble → two-point BVP → shooting method (**structural corroboration only**; the fields
  there are signal/idler).
- arXiv:1912.10089 (Cs 5²D₅/₂ lifetime, ar5iv). Taken: Holstein factor `g₁ = 1.6` (a 60 % lifetime
  lengthening) at attenuation parameter `αr ≈ 0.66`, and the pointer to the closed forms in
  Molisch *et al.*, JQSRT **49**, 361 (1993).

**Located but NOT fetched — LITERATURE-RECALL / MISSING:**

- T. Holstein, Phys. Rev. **72**, 1212 (1947); **83**, 1159 (1951) — escape factors. **MISSING.**
- A. Molisch, B. P. Oehry, W. Schupita & G. Magerl, JQSRT **49**, 361 (1993) — the closed-form
  `g_esc(αr)` for cylinder/sphere. **MISSING** (citation itself VERIFIED via arXiv:1912.10089).
- M. Fleischhauer, A. Imamoglu & J. P. Marangos, Rev. Mod. Phys. **77**, 633 (2005) — citation
  confirmed (volume/page/DOI) via APS and ADS this session; **full text not fetched**, so no
  equation from it is used or quoted here.
- Carr, Ritter, Wade, Adams & Weatherill, Phys. Rev. Lett. **111**, 113901 (2013); Phys. Rev. A
  **94**, 063820 (2016) — Rydberg-vapor optical bistability. Citations located; papers not fetched.
- Holloway *et al.*, J. Appl. Phys. **121**, 233106 (2017) — cell/internal-field systematics
  (spec-00 §6 gap 6, still paywalled).

**RydSim-internal measurements this session** (all reproducible from the shipped tree): RK4
convergence table for 10/P-1/P-2; `σ_eff = 5.129×10⁻¹⁵ m²` from `rydsim.eit.chi_si` at Rb 300 K
line centre with `Ω_c = 0`; EIT-peak vs coupling-off OD at 300/313/333 K; all §2.4/§2.7/§2.8
numeric tables.

---

*GreyNOC · RydSim spec 10 (theory) · authored 2026-08-11, network available ·
house rule: reproducible or it didn't happen.*

---

## Provenance of this draft section
### Sources FETCHED this session
- arXiv:2205.10959 / New J. Phys. 25, 035001 (2023) — Finkelstein, Bali, Firstenberg, Novikova, 'A practical guide to EIT in atomic vapor'. FETCHED via ar5iv full text. Took: Eq. (12) alpha_0 = k_p N mu_13^2/(2 hbar eps0 gamma_13); the convention OD = 2 alpha_0 L (which I verified is algebraically IDENTICAL to RydSim's OD = k_p Im chi L); Eq. (11) chi_p; Eq. (9)-(10) rho_21, rho_31; Eq. (36) E_out = int d2k_perp E_in exp(i k_z L int d2v w(v) chi_p) — the single-exponential closed form that proves Beer-Lambert is EXACT at any OD in strict linear response; the linear-response condition Omega_p << Omega_c; chi_ensemble = int chi_p(delta - k v_z) w(v_z) dv_z; transit broadening Gamma_tt = v_th/w0; the statement that increasing OD enhances four-wave mixing.
- arXiv:1909.07161 / Phys. Rev. Lett. 123, 243604 (2019) — Ogden, Whittaker, Keaveney, Wrathmall, Adams, Potvliege, 'Quasi-simultons in thermal atomic vapors'. FETCHED via ar5iv. Took: Eq. (6) [d/dz + (1/c)d/dt] E_alpha = (i k / 2 eps0) P_alpha for alpha = p,c — the propagation equation for BOTH fields, which fixes the +i and the factor 1/2 in RydSim's dE/dz = i(k/2) chi E; Eq. (8) P(z,t) = N int f(v_z) Tr[d rho(z,t;v_z)] dv_z (Doppler-averaged source); confirmation that both probe and coupling are propagated self-consistently in a thermal ladder medium; their parameters N = 2.0e15 cm^-3, 2 um cell.
- arXiv:1207.6748 / Rev. Mod. Phys. 85, 941 (2013) — Firstenberg, Shuker, Ron, Davidson, 'Colloquium: Coherent diffusion of polaritons in atomic media'. FETCHED via ar5iv. Took: Eq. (14), the phase-space transport equation (d_t + v.d_r) rho~_ss' + (d_t rho~)_col = source — the exact equation whose neglected v.grad term is the non-commutation of the velocity average with the spatial integration.
- arXiv:quant-ph/0101147 / Phys. Rev. Lett. 87, 133601 (2001) — Matsko, Novikova, Scully, Welch, 'Radiation trapping in coherent media'. FETCHED (abstract). Took the verbatim claim that the effective decay rate of ground-state (Zeeman) coherence rises significantly with atomic density due to radiation trapping, and that trapping 'must be taken into account to fully understand many EIT experiments with optically thick media' — the primary justification for treating radiation trapping as an addition to gamma_gr rather than a modification of Gamma_e.
- arXiv:2111.13408 / Opt. Express 30, 1499 (2022) — Su, Liou, Lin, Chen, 'Optimizing the Rydberg EIT spectrum in a thermal vapor'. FETCHED (abstract page + ar5iv). Took: the experimental operating range OD (their alpha) = 0.42 at 27 C to 5.0 at 65 C in Rb-87; their Eq. (5) T = exp[-alpha Im(Gamma_e rho_eg / Omega_p)] — i.e. a thin-medium exponential applied at OD up to 5 in published work; 13% peak EIT height. This anchors benchmark 10/P-15 (the regime RydSim currently refuses).
- arXiv:1106.2082 — H.-H. Jen, 'Theory of Light-Matter Interactions in Cascade and Diamond Type Atomic Ensembles', PhD thesis, Georgia Tech (2010). FETCHED via ar5iv. Took: the structural statement that counter-propagating fields in a CASCADE (ladder) ensemble carry a boundary condition at each end and are solved as a two-point BVP by shooting. Used ONLY as structural corroboration — the counter-propagating fields there are signal/idler, not probe/coupling, and I say so in the spec.
- arXiv:1912.10089 (Cs 5^2D_5/2 lifetime measurement). FETCHED via ar5iv. Took: the measured Holstein trapping factor g_1 = 1.6 (a 60% lengthening of the Cs 6P_3/2 lifetime) at attenuation parameter alpha*r ~ 0.66 at 23 C — evidence that trapping is a live effect at ordinary alkali-cell opacities — and the pointer that the closed-form escape factors live in Molisch, Oehry, Schupita, Magerl, JQSRT 49, 361 (1993). Also the source of the convention warning that published 'trapping factors' are tau_eff/tau = 1/g_esc, the reciprocal of the escape factor.
- In-repo normative specs read in full and used as binding constraints: docs/spec/00-conventions.md (20 locks, rulings R-1..R-28), docs/spec/00-integrity-audit.md (risk register + refusal list), docs/spec/05-vapor-cell-physics.md (esp. 2.d/2.e/2.f/2.g), docs/spec/06-optical-bloch-eit.md (esp. 2.4, 4.4, 7.2), src/rydsim/eit.py, src/rydsim/experiment.py (max_optical_depth gate at :102/:317).
- RydSim engine runs executed this session (reproducible from the shipped tree): rydsim.eit.chi_si/spectrum at Rb-87, Omega_c/2pi = 5 MHz, 5 cm cell -> EIT-peak OD = 0.50/1.87/10.15 and coupling-off line-centre OD = 1.12/4.16/22.6 at 300/313/333 K; effective per-atom cross-section alpha_p/N = 5.129e-15 m^2 at 300 K line centre. Independent numerical checks: RK4 vs closed-form exp(i k chi L/2) relative error 4.04e-6 (1 step) / 3.48e-10 (10) / 1.36e-15 (1000), fitted order 4.07; sqrt(pi ln2) = 1.475665; sigma_D,peak = 5.089e-15 m^2; spec-05 B9a reproduction 1.323 vs 1.298 (1.9%); coupling-depletion bounds 2.71e-5 / 3.66e-3 / 1.97e-5; depletion-to-saturation ratio 1.07e-3; radiation-trapping bounds 2pi x 0.82/2.04/20.4/204 kHz at 0.39/1/10/100 uW; thermal 5P fraction 2.0e-27/9.5e-21/9.6e-17 at 300/400/500 K; axial/transverse traversal ratio 62.7; epsilon_transp = 8.0e-4 x OD for the 5 cm fixture.

### UNVERIFIED / recall-only
- MISSING — the closed-form Holstein escape factor g_esc(k0 R) for a Doppler-broadened line in a cylinder/slab. Molisch, Oehry, Schupita & Magerl, JQSRT 49, 361 (1993) and Holstein, Phys. Rev. 83, 1159 (1951) were located but NOT fetched (paywalled). Consequence: the spec ships ONLY the model-free bound gamma_trap <= alpha_p Phi_p / N, which needs no escape-factor theory, and forbids any g_esc-dependent number from shipping. Resolution: fetch either paper and transcribe g_esc with its geometry and validity range, watching the reciprocal-convention trap.
- DERIVED-IN-SPEC, not sourced — Eq. (10.8)/(10.10), the Manley-Rowe photon-flux bound on coupling depletion |dOmega_c/Omega_c| <= 0.5 (lambda_p/lambda_c)(I_p(0)/I_c)(1-T). Every step is an equality except the final one-photon-branch inequality. No published statement of this bound was found this session. Falsified by benchmark 10/P-3 if the solved BVP's integrated coupling loss ever exceeds it.
- DERIVED-IN-SPEC — the claim in Sec 2.5 that COUNTER-propagation is geometrically favourable (fresh coupling sits at the probe's exit face, so the signal-generating layer sees the least-depleted coupling, whereas co-propagation accumulates depletion into it). Benchmark 10/P-6 exists specifically to refute it; if the test comes out the other way the paragraph must be struck, not softened.
- DERIVED-IN-SPEC — the claim in Sec 2.6 that the first-order transport correction is ODD in v and therefore cancels in the symmetric Maxwell average at line centre, leaving O(eps^2) at the EIT peak and O(eps) on the Doppler wings. Benchmark 10/P-7 measures the exponents; a linear-in-eps peak error falsifies it.
- DERIVED-IN-SPEC — Eq. (10.13c), that ballistic axial transport shifts every coherence decay rate by -v*mu with mu = i k_p chi / 2, and hence the non-locality parameter eps_transp = |v| alpha_p / (2 gamma_min) = |v| OD / (2 gamma_min L). The algebra is exact for the linear-response matrix problem as posed; what is unverified is that the reduced linear system is the correct one to pose (it drops the population sector).
- UNVERIFIED sub-claim — that a reabsorbed trapped resonance photon destroys the g-r (dark-state) coherence with UNIT probability. It is a projective measurement of the internal state so unity is natural and conservative (it maximises gamma_trap), but no source was found. Fenced: it only ever tightens the bound, so it cannot corrupt a shipped number.
- DERIVED-IN-SPEC — Eq. (10.17), sigma_D,peak/sigma_0 = sqrt(pi ln 2) (Gamma/2pi)/Delta_nu_D. Elementary (ratio of unit-area Gaussian and Lorentzian peak heights) but not quoted from a source. Self-checked twice: against the shipped engine (0.8%) and against spec-05 benchmark B9a (1.9%).
- DERIVED-IN-SPEC — Eq. (10.19), the self-lensing criterion Delta_phi_lens = (k_p L/2)[Re chi(0) - Re chi(w0)] with flag at 0.3 rad and refusal at 1 rad. The thresholds are engineering judgement, not a sourced diffraction calculation; spec 05 Sec 7.6 declares the limitation but offers no criterion, and this fills the gap with an admittedly heuristic one.
- LITERATURE-RECALL, papers NOT fetched — Fleischhauer, Imamoglu & Marangos, Rev. Mod. Phys. 77, 633 (2005): citation (volume/page/DOI) confirmed via APS and ADS this session, but the full text was NOT retrieved, so NO equation from it is used or quoted anywhere in the spec. The brief suggested it as a starting point; I decline to dress the citation up as verification.
- LITERATURE-RECALL, papers NOT fetched — Rydberg-vapor optical bistability: Carr, Ritter, Wade, Adams & Weatherill, PRL 111, 113901 (2013) and Phys. Rev. A 94, 063820 (2016). Citations located this session via search only. Used solely to name a limitation (Sec 6.4), never to support a number.
- LITERATURE-RECALL — the dark-state population relation rho_rr = Omega_p^2/(Omega_p^2 + Omega_c^2) (Eq. 10.15b). Textbook result, not re-checked against a primary source this session; it is the physical content of the already-shipped gate Omega_p < 0.01 min(Gamma_e, Omega_c).
- NOT ADDRESSED — four-wave mixing and backward-generated fields in the counter-propagating ladder at high OD. The NJP 2023 guide (fetched) states FWM grows with OD, but no phase-matching criterion was derived or sourced. This is explicitly named in Sec 6.2 as the least-fenced limitation in the document; no criterion is offered because none was found.

### Open questions
- O-1 (blocks a shipped number). The closed-form Holstein escape factor g_esc for a Doppler-broadened line in a cylinder is MISSING. Fetch Molisch, Oehry, Schupita & Magerl, JQSRT 49, 361 (1993) or Holstein, Phys. Rev. 83, 1159 (1951), transcribe g_esc with its geometry and validity range, and note the reciprocal-convention trap (published 'trapping factors' such as the fetched g_1 = 1.6 are tau_eff/tau = 1/g_esc). Until then only the model-free bound Eq. (10.18) may ship and no g_esc-dependent number may appear in a finding.
- O-2. Is there ANY published measurement of coupling-beam absorption through a Rydberg-EIT vapor cell? None was found this session. Such a datum would turn benchmark 10/P-4 from a self-check into an external validation, and would test the Manley-Rowe bound Eq. (10.10) against experiment rather than against the solver.
- O-3. Does a reabsorbed trapped resonance photon destroy the g-r dark-state coherence with unit probability? Assumed yes (projective, and conservative). A model calculation or a source would let gamma_trap become an estimate rather than a bound.
- O-4 (the least-fenced limitation in the document). Four-wave mixing and backward-generated fields in the counter-propagating ladder at high OD. The NJP 2023 guide states FWM grows with OD; no phase-matching criterion is derived or sourced here. Needs: the phase-matching condition for the counter-propagating Rb 5S-5P-nD ladder, the known cascade emissions, and a threshold OD. Until then the spec offers no criterion at all and says so.
- O-5 (requires a spec-00 ruling; LOOSENS a fence, so it must not be adopted before 10/P-1 passes). Integrity-audit refusal #18 raises ThickCellError for OD > 0.1 through the analytic path. Theorem 10.R shows that threshold is over-strict by ~2 orders of magnitude and would refuse the regime published experiments occupy (OD = 0.42-5.0). Proposed replacement: the GATE-P/C/T/L set of Sec 4.1, keeping the refusal semantics and demoting OD > 0.1 to a warning-level flag so nothing currently caught becomes uncaught.
- O-6 (requires a spec-00 ruling). Should max_optical_depth be reclassified? It is a numerical-conditioning gate, not a physics gate, and its refusal message currently advises implementing the z-propagation solver — which would return the identical number. Proposed: split it into an explicit min_transmitted_power conditioning gate plus GATE-P/C/T/L, and rewrite the message.
- O-7 (numerical inconsistency inside spec 05). Sec 2.f states 'I_p < 0.01 I_sat ... In Rabi terms Omega_p <~ Gamma_e/7'. With s_sat = 2 Omega_p^2/Gamma_e^2 these are inconsistent: s_sat = 0.01 gives Omega_p = Gamma_e/14.1, while Gamma_e/7 corresponds to s_sat = 0.041. One of the two must move; this spec recommends keeping the sourced intensity form and correcting the Rabi restatement.
- O-8. Cost and feasibility of the exact velocity-resolved axial transport BVP (Sec 2.6): ~N_v x N_levels^2 ~ 1e6 coupled ODEs in z with split boundary data (v>0 at z=0, v<0 at z=L). Needs a prototype to measure whether a reduced-order or moment-closure treatment can deliver the O(eps^2) correction at acceptable cost, or whether eps_transp remains a fence forever.
- O-9. Self-lensing (Eq. 10.19): the 0.3 rad flag / 1 rad refusal thresholds are engineering judgement. A proper split-step diffraction calculation on the shell decomposition would replace them with a measured criterion, and would also test assumption A5 (no inter-shell diffraction) at high OD rather than only at low.
- O-10. Photon bookkeeping in a natural-abundance cell (pitfall 4.3.11): the probe absorption is spread over two isotopes and four ground levels while the coupling addresses only the sensed channel, so Eq. (10.10) is valid but loose by the branching fraction. Quantifying that fraction would tighten GATE-C; doing it carelessly would break the bound's status as a bound.

### Proposed benchmarks

| id | quantity | expected | tol | source | conf |
|---|---|---|---|---|---|
| 10/P-1 | z-stepped propagation solver vs the closed-form thin-cell solution T = exp(-k_p Im chi L), with strict linear response, undepleted uniform coupling and uniform medium imposed by construction; OD in {0.1, 1, 5, 50} | identical (relative difference 0); measured 1.36e-15 with 1000 RK4 steps, 3.48e-10 with 10 steps, at L = 5 cm, chi = 1e-6(0.3+1.0i) | relative <= 1e-12 in T | Reduction Theorem 10.R; independently VERIFIED against Finkelstein/Bali/Firstenberg/Novikova NJP 25, 035001 (2023) Eq. (36) E_out = E_in exp(i k_z L int d2v w(v) chi_p), fetched this session; numerics measured this session | VERIFIED (identity + measured). RELEASE-GATING: a solver failing this has a sign, factor-of-2 or k_p-convention bug and must not be used at ANY optical depth. |
| 10/P-1b | same as P-1 at extreme depth, compared in ln T rather than T to avoid underflow | OD = 200 reproduced exactly (T = 1e-87) | relative <= 1e-12 in OD | underflow-safe restatement of P-1; spec 05 Sec 4.7 guards exp(-OD) for OD > 700 | VERIFIED (identity) |
| 10/P-2 | convergence order of the z-integrator (RK4) by step halving | order 4; measured 4.07 from the pair (1 step -> 4.044e-6, 10 steps -> 3.476e-10) | fitted order 4.0 +/- 0.2 | RK4 theory; measured this session | VERIFIED (measured) |
| 10/P-3 | Manley-Rowe / photon-number closure of the coupled BVP: integrated coupling photon loss vs the two-photon transfer rate, and against the probe photon loss | \|Delta Phi_c\| = N Integral rho_rr (Gamma_r + gamma_t) dz exactly, AND \|Delta Phi_c\| <= \|Delta Phi_p\| never violated | equality relative <= 1e-6; inequality must hold exactly (a violation is a solver bug, not a tolerance question) | Eq. (10.8), derived in spec from steady-state photon bookkeeping in the closed 3-level ladder | DERIVED-IN-SPEC |
| 10/P-4 | coupling-depletion bound \|Delta Omega_c / Omega_c\| <= 0.5 (lambda_p/lambda_c)(I_p(0)/I_c)(1-T) | 2.71e-5 for P_p = 1 uW, P_c = 30 mW, w0 = 1 mm, T -> 0 (Rb-87, lambda_p/lambda_c = 1.625503); 3.66e-3 for 50 uW / 10 mW / T = 0.1; 1.97e-5 at Omega_p = Omega_c with dipole ratio 4.917e-3 | the solved BVP depletion must never exceed the bound, and must reach at least half of it when the two-photon (EIT-shelving) channel dominates | Eq. (10.10); numbers computed this session | DERIVED-IN-SPEC |
| 10/P-5 | BVP-to-IVP degeneracy: full two-point shooting solution vs the uniform-coupling initial-value solution when GATE-C passes by 100x | identical spectra | difference <= the P-4 bound on \|Delta Omega_c/Omega_c\| | Sec 2.5; falsifies an IVP march that silently solves the co-propagating problem | DERIVED-IN-SPEC |
| 10/P-6 | counter- vs co-propagating depletion asymmetry at identical total coupling depletion | the counter-propagating transmitted spectrum is perturbed LESS than the co-propagating one (fresh coupling enters at the probe's exit face, i.e. at the signal-generating layer) | boolean sign test | Sec 2.5 geometric argument, derived in spec | DERIVED-IN-SPEC — this benchmark exists to REFUTE Sec 2.5; a failure is a spec edit, not a code fix |
| 10/P-7 | error of the local-response approximation vs the velocity-resolved axial transport BVP, at OD in {1, 10, 100} | error scales as eps_transp^2 at the EIT peak and as eps_transp on the Doppler wings, with eps_transp = \|v\| OD/(2 gamma_min L) = 8.0e-4 x OD for the 5 cm Rb-87 fixture (so ~8e-2 at OD 100) | fitted exponents 2.0 +/- 0.3 (peak) and 1.0 +/- 0.2 (wings); magnitude within 2x of the predicted eps | Eq. (10.14) from the transport equation, Firstenberg et al. RMP 85, 941 Eq. (14) (fetched); scaling derived in spec | DERIVED-IN-SPEC — exists to REFUTE the odd-in-v cancellation claim of Sec 2.6 |
| 10/P-8 | probe-saturation self-healing with depth: monotonicity of s_sat(z) and the approach of the saturable-chi transmission to the linear-response transmission | s_sat(z) = s_sat(0) exp(-alpha_p z) monotone decreasing; T error linear in s_sat(0) over s_sat(0) in [1e-3, 1e-1] | log-log slope 1.0 +/- 0.15 | Sec 2.7, Eqs. (10.15)-(10.16); Omega_p << Omega_c VERIFIED from NJP 25, 035001 (fetched) | DERIVED-IN-SPEC |
| 10/P-9 | radiation-trapping bound gamma_trap <= alpha_p I_p /(N hbar omega_p) = sigma_eff Phi_p, and the fence that fires above 0.1 gamma_gr | 5.13e3 rad/s = 2pi x 0.82 kHz = 2.1% of gamma_t at I_p = 0.01 I_sat (0.39 uW, w0 = 1 mm, Rb-87 300 K); 2pi x 204 kHz = 513% of gamma_t at 100 uW | bound exact (it is an inequality, not a fit); fence firing is a boolean | Eq. (10.18), derived in spec; motivated by Matsko/Novikova/Scully/Welch PRL 87, 133601 (2001), abstract fetched | DERIVED-IN-SPEC (bound); the escape factor that would convert it to an equality is MISSING |
| 10/P-10 | absolute susceptibility-chain cross-check via the effective per-atom cross-section sigma_eff = alpha_p/N at Rb-87 300 K line centre with the coupling off | 5.089e-15 m^2 from sigma_D,peak = sqrt(pi ln2)(Gamma_e/2pi)/Delta_nu_D x sigma_0; engine returned 5.129e-15 m^2 | relative <= 3% (MEASURED 0.8%) | Eq. (10.17) derived in spec vs a live rydsim.eit run this session | VERIFIED (two-path closure). Independent of the spec-06 B-2 sum rule, so it is a genuinely second check on the absolute chi chain. |
| 10/P-10b | optical-depth convention identity: RydSim's OD = k_p Im chi L against the published OD = 2 alpha_0 L with alpha_0 = k_p N mu_13^2/(2 hbar eps0 gamma_13) | algebraically identical (both equal k_p N mu^2 L /(hbar eps0 gamma_13) for the two-level on-resonance case) | relative <= 1e-14 | Finkelstein/Bali/Firstenberg/Novikova NJP 25, 035001 (2023) Eq. (12) and the definition OD = 2 alpha_0 L, FETCHED this session | VERIFIED. Convention lock: run first, fail loudly. |
| 10/P-11 | cross-spec closure of the Doppler-limited peak cross-section against spec 05 benchmark B9a (natural Rb, 75.0 mm, 25.0 C, Rb-85 F=3 dip) | OD = 1.298 (spec 05 B9a); reproduced 1.323 from Eq. (10.17) x S_34 x p_F alone | relative <= 3% (MEASURED 1.9%; residual is the neglected wings of the other hyperfine lines) | spec 05 Sec 2.f/Sec 6 B9a vs Eq. (10.17); computed this session | DERIVED (measured) |
| 10/P-12 | ordering of radial averaging and z-propagation: <exp(-OD(r))> vs exp(-<OD(r)>) at OD_peak = 5 with 12-node Gauss-Laguerre | the two differ by >= 5% — i.e. the test must DETECT that propagation has to happen inside the radial quadrature | boolean (difference detected) | pitfall Sec 4.3.5; same class as spec 05 Sec 4.8's forbidden peak-intensity-only simulation | DERIVED-IN-SPEC |
| 10/P-13 | dispersive self-lensing fence Delta_phi_lens = (k_p L/2)[Re chi(r=0) - Re chi(r=w0)] | ~5 rad at OD = 50 with a 20% radial variation of chi, so the 1 rad refusal fires | boolean (refusal fires) | Eq. (10.19), derived in spec; fills the criterion gap left by spec 05 Sec 7.6 | DERIVED-IN-SPEC (thresholds are engineering judgement, declared as such) |
| 10/P-14 | axial vs transverse traversal-time ratio, which justifies dropping the axial atomic boundary condition in favour of the spec-06 transit channel | 62.7 (L/sigma_v = 295.1 us against w0/<v_perp> = 4.71 us; Rb-87, 300 K, w0 = 1 mm, L = 5 cm) | relative <= 1% | Sec 2.5; recomputed this session from spec 05 Sec 2.c speeds | DERIVED (recomputed) |
| 10/P-15 | coverage of the published operating regime that the engine currently refuses | the thick-cell path returns a graded result across OD = 0.42 (27 C) to 5.0 (65 C) in Rb-87, matching the range reported experimentally | qualitative (2 booleans: no refusal fires; T monotone in OD) | Su, Liou, Lin & Chen, Opt. Express 30, 1499 (2022) / arXiv:2111.13408, FETCHED — their stated alpha range and their Eq. (5) thin-medium exponential applied at OD 5 | VERIFIED (their stated range); RydSim reproduction is the thing under test |
