# 02 — Radial Wavefunctions by Numerov Integration with Alkali Model Potentials

**RydSim physics specification. Species: Rb-85, Rb-87, Cs-133. Python 3.11 + numpy/scipy only.**

> **Verification status (2026-08-10, network AVAILABLE).** The Marinescu–Sadeghpour–Dalgarno
> parameter tables below were verified against **two independent machine-readable transcriptions**:
> ARC master (`arc/alkali_atom_data.py`, github.com/nikolasibalic/ARC-Alkali-Rydberg-Calculator) and
> pairinteraction's `ryd-numerov` (`src/ryd_numerov/elements/{rubidium,cesium}.py`,
> github.com/pairinteraction/ryd-numerov). They agree digit-for-digit on every entry **except one**
> (§3.3). The original PRA table was not directly consulted (paywalled). The Kaulakys (1995) formulas
> were transcribed from the **full paper text held locally in this repo** (`kaulakys_text.txt`,
> arXiv:physics/9610018). Every equation in this document was additionally **executed numerically
> this session** (harness: scratchpad `verify_radial_02*.py`, since ported to
> `tests/test_radial.py` + `tests/test_dipoles.py`, not to the `tests/test_spec02_benchmarks.py`
> name used in earlier drafts — §6): Numerov vs. exact hydrogen, Gordon-exact vs. direct
> integration, Kaulakys vs. Gordon, model-potential vs. Coulomb-approximation for Rb and Cs at
> n = 20–60. Measured agreement numbers quoted below are from those runs.

---

## 1. Scope

Defines the computation of bound-state valence-electron radial wavefunctions `R_nlj(r)` and radial
matrix elements `<n l j | r^k | n' l' j'>` (k = 1 default) for Rb and Cs Rydberg *and* low-lying
states, given state energies (equivalently effective quantum numbers ν = n − δ_nlj) from doc 01.

- **Method A (primary):** Numerov integration of the radial equation in the x = √r coordinate with
  the l-dependent Marinescu–Sadeghpour–Dalgarno (MSD94) model potential + spin–orbit term.
- **Method B (cross-check, model-parameter-free):** Coulomb approximation / single-channel QDT —
  the Whittaker function `W_{ν,l+1/2}(2r/ν)` with Seaton normalization, evaluated stably.
- **Method C (cross-check, fully analytic):** Kaulakys (1995) quasiclassical Anger-function formula.
- **Exact hydrogenic limit:** Gordon's formula for testing the machinery at δ → 0.

House rule application: **all three methods run on every requested matrix element; the spread is
reported as the numerical-uncertainty estimate** (`RadialMEResult.spread_rel`). No single-method
number ships.

Out of scope: energies/quantum defects (doc 01), angular factors (doc 03), continuum states,
two-valence-electron species, quadratic Zeeman.

Units: **atomic units throughout** (length a₀, energy Hartree, ħ = e = mₑ = 1) unless stated.
Conversions via `rydsim.constants` (CODATA/scipy).

---

## 2. Equations

### 2.1 Radial equation and the MSD94 model potential

For `u(r) = r·R(r)` (so ∫₀^∞ u² dr = 1):

```
u''(r) = [ 2μ (V_lj(r) − E) + l(l+1)/r² ] u(r)                                        (2.1)
```

- `μ` — reduced mass in units of mₑ. This doc's original ion-core form `μ = (M − mₑ)/M` is
  **superseded by ruling R-10** (doc 00, binding): the single code-level definition is
  `μ = 1/(1 + mₑ/M) = R_M/R_∞`, shared with doc 01's energies. M = atomic mass. The two differ by
  O((mₑ/M)²) = 4.2e-11 (measured, Rb-85) — below every tolerance here, but one definition, not two.
  (Rb-85: μ = 1 − 6.46e-6 under either; effect on MEs ≤ 1e-5 relative — include it, but it is below
  every tolerance in §6.)
- `E = −μ/(2ν²)` Hartree with ν = n − δ_nlj from doc 01 (measured energies; **E is an input, never
  an eigenvalue solved for here** — that is what makes inward integration + truncation correct).

The MSD94 model potential (Marinescu, Sadeghpour & Dalgarno, PRA **49**, 982 (1994)), for l ≤ 3:

```
V_l(r) = − Z_l(r)/r − (α_c / 2r⁴) [ 1 − exp(−(r/r_c[l])⁶) ]                            (2.2)
Z_l(r) = 1 + (z−1) e^{−a₁ r} − r (a₃ + a₄ r) e^{−a₂ r}                                 (2.3)
```

- `z` — nuclear charge (Rb: 37, Cs: 55). `Z_l(r) → z` as r → 0, `Z_l(r) → 1` as r → ∞.
- `α_c` — static dipole polarizability of the ionic core (a₀³).
- `a₁..a₄, r_c` — l-dependent fitted parameters (§3). **For l ≥ 4 use the pure Coulomb potential
  V(r) = −1/r** (ARC convention; the l ≥ 3 parameter row exists but high-l states never probe the
  core — measured difference on MEs is below 1e-5).

**Spin–orbit term (include it, l > 0):**

```
V_so(r) = (α² / 2r³) · [ j(j+1) − l(l+1) − s(s+1) ] / 2 ,   s = 1/2                    (2.4)
```

α = fine-structure constant (CODATA via scipy). This is the hydrogenic-limit form
(α²/2)(1/r)(dV/dr)L·S with V ≈ −1/r; identical to ARC's implementation (verified against source).
For l = 0 the bracket vanishes identically. **Measured effect on Rb 50S→50P₃/₂ radial ME: 2.7e-11
relative** — at high n the entire j-dependence enters through δ_nlj; the term matters (marginally)
only for low-n P states of Cs. It is kept because it is free and makes the low-n wavefunction
j-dependence physical. Its r⁻³ divergence is inside the inner cutoff (§4.1) and harmless.

### 2.2 The √r substitution — the equation actually integrated

Substitute `x = √r`, `X(x) = R(r)·r^{3/4} = u(r)·r^{−1/4}`. Then (2.1) becomes exactly

```
X''(x) = g(x) X(x)
g(x)   = 8 μ x² [ V_lj(x²) − E ] + (2l + 1/2)(2l + 3/2) / x²                           (2.5)
```

Derivation check: (2l+1/2)(2l+3/2) = 4l(l+1) + 3/4; the 3/4 is the extra term from the coordinate
change. Verified (i) algebraically, (ii) against ARC's `kfun` (identical up to sign convention
y'' = −k y), and (iii) empirically: hydrogen R(1s→2p) reproduced to 4e-12 (§6, B1).

*Why √r:* the local de Broglie wavelength under a Coulomb potential scales as √r at small r, so a
uniform x-grid gives ~constant points-per-oscillation everywhere (Zimmerman, Littman, Kash &
Kleppner, PRA **20**, 2251 (1979)). Max local wavenumber in x is k_x ≈ √(8 Z_l(0)) ≈ 17 (Rb) /
21 (Cs) per √a₀, reached only at the inner cutoff.

### 2.3 Normalization and matrix elements on the x-grid

With uniform spacing h in x (measure: dr = 2x dx):

```
Norm:   2 ∫ X²(x) x² dx = 1            ⇔  ∫ u² dr = ∫ R² r² dr = 1                     (2.6)
ME:     <1| r^k |2> = 2 ∫ X₁(x) X₂(x) x^{2k+2} dx                                      (2.7)
```

Trapezoidal quadrature on the uniform x-grid is sufficient (error floor measured at ≤1.5e-8
relative for n = 50, dominated by the outer cutoff, not the quadrature — §4.5).

### 2.4 Method B — Coulomb approximation / QDT (independent of §3 parameters)

The QDT orbital for effective quantum number ν, valid outside the core (Seaton, MNRAS **118**, 504
(1958); Bates & Damgaard, Phil. Trans. R. Soc. A **242**, 101 (1949)):

```
u_ν,l(r) = N W_{ν, l+1/2}(2r/ν) ,      N = [ ν² Γ(ν+l+1) Γ(ν−l) ]^{−1/2}               (2.8)
```

`W_{κ,μ}(ζ)` = Whittaker-W. In scipy terms:

```
W_{ν,l+1/2}(ζ) = e^{−ζ/2} ζ^{l+1} U(l + 1 − ν, 2l + 2, ζ) ,   U = scipy.special.hyperu (2.9)
```

**CRITICAL, MEASURED PITFALL — do not skip:** `scipy.special.hyperu` loses precision
catastrophically as ν grows (relative error vs. analytic hydrogen u_nl, classical region, l = 0):

| ν | 10 | 15 | 20 | 25 | 28 | 30 | 35 | 40 |
|---|----|----|----|----|----|----|----|----|
| rel. err (original session harness) | 5e-13 | 7e-11 | 1.5e-8 | 5.7e-6 | 1.5e-4 | 8.2e-4 | 1.7e-1 | 4.3e+1 |
| rel. err (**scipy 1.17.1**, shipped) | 6.2e-13 | 1.5e-10 | **3.5e-8** | **5.5e-6** | 1.9e-4 | 1.8e-3 | 0.49 | 93 |

The second row is the binding one: hyperu accuracy is **scipy-version-dependent**, so the table is
regenerated on the *installed* scipy by `rydsim.radial.hyperu_hydrogen_error(ν, l)` and asserted at
test time (`tests/test_radial.py::test_hyperu_fence_r4_binding_form`, audit R4). The two rows differ
by up to ~2× because they were measured on different scipy builds — do not treat either as a
constant of nature; re-measure before moving the §7.3 fence.

Therefore the **production evaluation of Method B is numerical**: run the *same* Numerov machinery
(2.5)–(2.7) with `V(r) = −1/r` exactly (no model potential, no SO), E = −μ/(2ν²), same grid and
divergence guard, unit-normalized numerically. This is mathematically the same solution as (2.8)
— identical ODE, identical decaying boundary condition — evaluated stably. The closed form
(2.8)/(2.9) with `hyperu` **must still be implemented** and used as a pointwise cross-check for
ν ≤ 20 (benchmark B12) — that is what makes Method B genuinely Numerov-code-independent at low ν
while remaining stable at high ν.

**Normalization equivalence is an INTEGER-ν statement (corrected 2026-08-10).** At integer ν the
Seaton analytic normalization and the numerical unit norm agree because ∫u² dr for (2.8) equals 1
up to the (negligible, truncated) core region: verified for integer ν against analytic hydrogen to
≤2e-13 (ν ≤ 10). That measurement is the *only* support the claim ever had, and it does **not**
extend to the QDT case Method B actually ships. At **non-integer ν the Whittaker function is
irregular at the origin** (W ~ r^{−l} as r → 0, §4.4 pitfall 6), so the numerically unit-normalized
Numerov solution and the analytically normalized closed form differ by a **cutoff-dependent scale**:

| measured abs(scale) − 1, ν = 10.5, r_inner = 1e-4 a₀ | l = 0 | l = 1 | l = 2 |
|---|---|---|---|
| offset | 7.3e-6 | 6.6e-4 | 3.0e-3 |

Moving `r_inner` 1e-4 → 1 a₀ moves the l = 0 offset to **3.7e-5**, which is what identifies the
offset as a cutoff artifact rather than an error in either method. The **shape** — which is what the
cross-check exists to test — agrees to **3e-11** either way. Consequently a non-integer-ν
cross-check must compare shape after a fitted scale and pin the norm offset *separately*, never fold
one into the other (§6 B12/B12b). Method B's own numbers are unaffected: it is unit-normalized
numerically, and the Seaton prefactor N never enters the shipped path.

Method B is *independent of the §3 model-potential parameters* — its only inputs are ν and l. The
A−B spread isolates exactly the core-model contribution: **measured A−B ≤ 4e-6 relative for
n = 50–60 S,P,D,F MEs; 4.2e-5 at n = 20** (§6, B8).

### 2.5 Method C — Kaulakys (1995) semiclassical formula

Source: B. Kaulakys, J. Phys. B **28**, 4963 (1995) — full text in `kaulakys_text.txt`; equation
numbers below are the paper's. For a dipole transition ν,l → ν′,l′ = l±1 (Z = 1 core seen by the
Rydberg electron; ν, ν′ from doc 01):

```
s   = ν′ − ν                                                                    (def. of s)
ν_c³ = 2 (ν ν′)² / (ν + ν′)                                                        (eq. 19)
e   = √[ 1 − ( (l + l′ + 1) / (2 ν_c) )² ]        (orbit eccentricity)             (eq. 17)
J_{−s}(w)  = (1/π) ∫₀^π cos(s ξ + w sin ξ) dξ      (Anger function)                (eq. 24)
J′_{−s}(w) = dJ_{−s}/dw = −(1/π) ∫₀^π sin ξ · sin(s ξ + w sin ξ) dξ

D_p^± = (1/s) [ J′_{−s}(e s) ± √(e^{−2} − 1) ( J_{−s}(e s) − sin(πs)/(πs) ) ]      (eq. 21)
D_r^± = D_p^± + (1 − e) sin(πs)/(πs)                                               (eq. 23)

R_{ν l}^{ν′ l±1} = (−1)^{Δn} · [ ν_c⁵ / (ν ν′)^{3/2} ] · D_r^±(e, s)               (eq. 22)
```

Upper sign: l′ = l+1; lower sign: l′ = l−1. Symmetry `D^±(e,−s) = D^∓(e,s)` (paper eq. 23′) makes
the formula valid for either ordering; negative s needs no special handling in the quadrature.
Limits (used both as code branches and as tests):

```
s → 0:  R → (−1)^{Δn+1} (3/2) ν² e   — exact for hydrogen n′ = n (paper eqs. 30, 31)
```

Branch to the s→0 form when |s| < 1e-4. Compare *magnitudes* across methods (sign conventions of
R_nl differ between methods; only |ME| is observable after doc 03's angular chain).

**Measured accuracy vs. exact hydrogen (Gordon):** 4.4e-5 (ν=50, Δν=1), 9.9e-5 (ν=50, Δν=2),
1.2e-4 (ν=30, Δν=1), 9.8e-4 (ν=10, Δν=1) — consistent with the paper's "up to some percents even
for low states" claim, much better at high ν. Measured vs. Method A for Rb/Cs n≈50 pairs:
1e-6 … 2e-3 (worst case is the cancellation-suppressed 50D→51F ME, §6 B9).

### 2.6 Exact hydrogenic limit — Gordon's formula (integrator test at δ → 0)

Gordon (1929); Bethe & Salpeter, *QM of One- and Two-Electron Atoms* (1957) §63. For
`R(n′,l−1; n,l) = <n′,l−1| r |n,l>` in a₀, Z = 1, with `n_r = n−l−1`, `n′_r = n′−l`,
`X = −4 n n′/(n−n′)²`:

```
R = (−1)^{n′−l} / (4 (2l−1)!)
    · √[ (n+l)! (n′+l−1)! / ( (n−l−1)! (n′−l)! ) ]
    · (4 n n′)^{l+1} (n−n′)^{n+n′−2l−2} / (n+n′)^{n+n′}
    · [ ₂F₁(−n_r, −n′_r; 2l; X) − ((n−n′)/(n+n′))² ₂F₁(−n_r−2, −n′_r; 2l; X) ]        (2.10)
```

Both ₂F₁ are terminating polynomials. **Reference implementation: exact rational arithmetic**
(`fractions.Fraction` for the bracket, log-gamma for the prefactor, magnitudes combined in log
space) — this was validated against direct `quad` integration of analytic R_nl to ≤9e-10 for all
tested (n ≤ 11) and is exact by construction. A float fast path via `scipy.special.hyp2f1` was
measured accurate to ≤6e-14 for n ≤ 70, Δn ≤ 20, l = 1 (terminating case is stable in scipy), but
any implementation using it MUST assert agreement with the rational path in its own tests.
**Invalid at n = n′** (0⁰ and (n−n′)^(...) degenerate): use the closed form

```
<n,l−1| r |n,l> = (3/2) n √(n² − l²)         (same-n, exact, all n)                    (2.11)
```

Reference values this formula chain produces (all reproduced by ≥2 independent routes this
session): see §6 table.

### 2.7 Expected magnitudes (smell tests)

- Hydrogen same-n: R(nS→nP) = (3/2) n √(n²−1) ≈ 1.5 n² a₀ — the canonical `~n*²` scaling with
  coefficient **3/2**.
- Alkalis, nS→nP (Δδ = δ_S − δ_P ≈ 0.48 Rb / 0.49 Cs suppresses the same-ν coefficient):
  measured `|R| / (ν ν′)`: **Rb 50S₁/₂→50P₃/₂: 1.131**, →50P₁/₂: 1.149, 60S→60P₃/₂: 1.130;
  **Cs 50S₁/₂→50P₃/₂: 1.130**; Rb 50D₅/₂→51P₃/₂: 1.366; Cs 50D₅/₂→51P₃/₂: 1.489.
  Rule of thumb: nS→nP₃/₂ radial ME ≈ **1.13 ν*² a₀** (Rb and Cs both), not 1.5 ν*².
- Δν-changing MEs fall fast: Rb 50S→51P₃/₂ is 316 a₀ vs. 2511 a₀ for 50S→50P₃/₂.
- Low-n anchor: Rb 5S→5P₃/₂ model potential gives 5.57 a₀ (see §7 for the known +8% bias).

---

## 3. Constants / parameter tables

### 3.1 MSD94 model-potential parameters — Rubidium (z = 37, both Rb-85 and Rb-87)

`α_c = 9.0760 a₀³`

| l | a₁ | a₂ | a₃ | a₄ | r_c |
|---|-----|-----|-----|-----|-----|
| 0 | 3.69628474 | 1.64915255 | −9.86069196 | 0.19579987 | 1.66242117 |
| 1 | 4.44088978 | 1.92828831 | −16.79597770 | −0.81633314 † | 1.50195124 |
| 2 | 3.78717363 | 1.57027864 | −11.65588970 | 0.52942835 | 4.86851938 |
| ≥3 | 2.39848933 | 1.76810544 | −12.07106780 | 0.77256589 | 4.79831327 |

Source: Marinescu, Sadeghpour & Dalgarno, PRA **49**, 982 (1994), Table I — via two independent
transcriptions (ARC master; pairinteraction/ryd-numerov master), fetched and diffed 2026-08-10.
Confidence: **VERIFIED (secondary-source cross-transcription)** — see †/§3.3 for the one exception.

### 3.2 MSD94 model-potential parameters — Cesium (z = 55)

`α_c = 15.6440 a₀³`

| l | a₁ | a₂ | a₃ | a₄ | r_c |
|---|-----|-----|-----|-----|-----|
| 0 | 3.49546309 | 1.47533800 | −9.72143084 | 0.02629242 | 1.92046930 |
| 1 | 4.69366096 | 1.71398344 | −24.65624280 | −0.09543125 | 2.13383095 |
| 2 | 4.32466196 | 1.61365288 | −6.70128850 | −0.74095193 | 0.93007296 |
| ≥3 | 3.01048361 | 1.40000001 | −3.20036138 | 0.00034538 | 1.99969677 |

Source & confidence: as §3.1 (both transcriptions agree digit-for-digit on every Cs entry).

### 3.3 † The one transcription discrepancy — Rb a₄(l=1)

ARC master has `−0.8163314` (7 significant digits); ryd-numerov has `−0.81633314` (8 digits, matching
the digit count of every other table entry). The two differ by **2.13e-6 relative** (this paragraph
previously said 2.5e-6; recomputed 2026-08-10 as 1.74e-6 / 0.81633314 = 2.1315e-6 — the figure moves,
the conclusion does not). **RydSim adopts −0.81633314** (ryd-numerov reading, consistent formatting)
and records the alternative.

**Impact bound (CORRECTED 2026-08-10).** The figures previously printed here — "δa₄ = 8e-8 perturbs
Z₁(r) by < 4e-8" — were arithmetically wrong, by 21.8× and 6.3× respectively, and visibly
inconsistent with the *relative* statement in the same paragraph (a ~2e-6 relative difference on
a₄ ≈ −0.816 is δa₄ ≈ 1.7e-6, not 8e-8 — that mismatch is what exposed the error). The re-derived
bound is:

```
|δa₄| = |0.81633314 − 0.8163314| = 1.74e-6            (2.13e-6 relative)
δZ₁(r) = |δa₄| r² e^{−a₂(l=1) r},  maximal at r = 2/a₂ = 1.04 a₀ (r² e^{−a₂ r} = 0.145586)
⇒ max |δZ₁| = 2.53e-7
```

This is shipped verbatim as `rydsim.radial.A4_L1_NOTE` and both figures are regenerated from the two
transcriptions at test time (`tests/test_radial.py::test_a4_l1_impact_bound_is_reproducible`):

> Rb a4(l=1): adopted -0.81633314 (ryd-numerov); ARC reads -0.8163314; last digit UNVERIFIED,
> |delta a4| = 1.74e-6 perturbs Z_1(r) by <= 2.53e-7 (max at r = 1.04 a0) (spec 02 §3.3 / audit R20)

The physical conclusion is unchanged, and is now stated on the quantities that actually move rather
than on Z₁ alone: adopting the ARC reading instead shifts the Rb 50S→50P consensus ME by **6.5e-14**
relative and the reproduced Rb P-series quantum defect (`model_potential_defect` — the one quantity
in this module that *is* sensitive to the tables, §6 note) by **4.7e-8** — both far below every
tolerance in §6. Anyone with PRA 49, 982 Table I in hand should close this out and update this
paragraph. Confidence of this single digit: **UNVERIFIED (two secondary sources disagree)** —
numerically irrelevant, flagged for honesty.

### 3.4 Other constants

| Constant | Value | Source | Confidence |
|---|---|---|---|
| α (fine structure) | `scipy.constants.alpha` (CODATA) | CODATA via scipy | VERIFIED |
| μ (reduced mass ratio) | (M − mₑ)/M; M from doc 01 (CODATA/AME) | computed | VERIFIED |
| ν = n − δ_nlj | doc 01 (measured quantum defects) | doc 01 | see doc 01 |

The harness runs used Rydberg–Ritz coefficients recalled from Li et al., PRA **67**, 052502 (2003)
(Rb) and Goy et al./Lorenzen–Niemax (Cs) *for tolerance-setting only* — tagged LITERATURE-RECALL;
production values are exclusively doc 01's.

---

## 4. Numerical method + pitfalls

### 4.1 Grid

- **Inner cutoff** `r_i = α_c^{1/3}` (Rb: 2.0856 a₀, Cs: 2.5006 a₀). Justification: setting
  α_c/(2r⁴) ≈ Z_eff/r gives r ≈ (α_c/2)^{1/3}; inside this radius the −α_c/2r⁴ term dominates the
  Coulomb term, the one-electron model potential is unphysical (it can bind spurious deep states),
  and the multi-electron core begins anyway. ARC uses the identical cutoff. **Measured
  sensitivity:** moving r_i over 1→20 a₀ changes the Rb 50S→50P ME by ≤1.4e-4 (2.0856→1.0: 9e-7).
- **Outer start** `r_o = 2n(n+15) a₀`. The outer classical turning point is r₂ ≈ 2ν²; 2n(n+15)
  = 2n² + 30n sits ~30n a₀ beyond it, deep in the exponential tail. **Measured (H, 50s→51p):**
  outer 2n(n+5) → 4.0e-2 rel. error; 2n(n+10) → 6.5e-5; **2n(n+15) → 1.4e-8**; 2n(n+25) →
  converged reference. Do not economize below +15.
- **Step** `h = 0.001 √a₀` in x (default). Grid size: n = 50 → ~79,000 points (x ∈ [1.444, 80.62]
  for Rb). Max phase per step: k_x·h ≈ √(8 Z_l(r_i))·h < 0.02 rad. **Measured:** results at
  h = 0.004…0.0005 identical to 1.4e-8 at n = 50; at n ≤ 5, h = 0.01 still gives 4.3e-10. h may be
  relaxed to `min(0.001·(n/50), 0.004)` for n > 50 if profiling demands; must stay ≤ 0.004.

### 4.2 Integration direction, seeds, guards

- **Integrate inward** (x_out → x_in): the physical solution decays outward, so inward stepping is
  the stable direction; the admixed unphysical solution decays away from the seed. Use
  `rydsim.numerov.numerov_inward` (recurrence `f_k y_k = (12 − 10 f_{k+1}) y_{k+1} − f_{k+2} y_{k+2}`,
  `f = 1 − h²g/12`; O(h⁶) local / O(h⁴) global — order verified, §6 B7).
- **Seeds:** `X(x_N) = 1e-10`, `X(x_{N−1}) = 1.2e-10`. Seed choice is irrelevant after
  normalization (ARC uses 0.01/0.01); the growing-inward solution swamps seed error within a few
  hundred steps. Do not seed exactly (0, 0).
- **Overflow guard:** linear ODE ⇒ rescale the accumulated solution whenever |X| exceeds 1e100
  (already implemented in `numerov_inward(rescale_threshold=...)`).
- **Divergence guard (inner truncation)** — because E is fixed from experiment, not an eigenvalue
  of the model potential, the inward solution eventually diverges near/inside the inner turning
  point. ARC-style cut, on `|u| = |X|·√x`: sweeping inward, track the running max of |u|; after 50
  consecutive non-increasing steps, freeze the max ("checkpoint"); the first later point where |u|
  exceeds that max is the divergence point — zero everything inside it, then normalize. States with
  ν < l + 1 or extreme defect mismatch can truncate early; `RadialSolution.r_cut` must expose where.

### 4.3 Matrix elements

Shared uniform x-grid (same h, same x_in) ⇒ elementwise product over the common index range,
trapezoid with weight x^{2k+2}, factor 2 (eq. 2.7). Never interpolate between different grids;
generate both states on the same grid. For |Δn| ≳ 10 the ME is cancellation-suppressed —
report the three-method spread (it will honestly widen; e.g. Rb 50D→51F, ME = 13 a₀: A−K = 2e-3).

### 4.4 Method-specific pitfalls

| # | Pitfall | Rule |
|---|---|---|
| 1 | `hyperu` precision collapse (table §2.4) | never call hyperu-Whittaker for **ν > 20** (`rydsim.radial.WHITTAKER_NU_MAX`; `whittaker_u` raises `IntegrityError`) — fence moved 25 → 20 by integrator ruling, §7.3; use pure-Coulomb Numerov as Method B production path |
| 1b | `hyperu` returns NaN for non-integer ν ≳ 11 *inside* the classical region (scipy 1.17.1) | `whittaker_u` refuses (`IntegrityError`) if **any** requested sample is non-finite — an all-NaN slice would let a `np.allclose`-style B12 pass and silently disable the only Numerov-independent cross-check |
| 2 | Gordon at n = n′ | closed form (2.11); the general formula divides by zero |
| 3 | Gordon float cancellation risk | reference path = exact `Fraction` arithmetic; scipy `hyp2f1` fast path allowed only with a test pinning it to the rational path (measured ≤6e-14, n ≤ 70, Δn ≤ 20) |
| 4 | Kaulakys s → 0 | branch to R = (3/2)ν²e for \|s\| < 1e-4 (removes 0/0 in D_p, sinc terms) |
| 5 | Kaulakys Anger quadrature | `scipy.integrate.quad` on [0, π], limit ≥ 400; integrand is smooth, cost trivial |
| 6 | Whittaker ME integrand diverges as r^{−l} at r → 0 for non-integer ν | irrelevant on the truncated grid; never integrate (2.8) from r = 0 |
| 7 | non-uniform grid fed to Numerov | forbidden — `numerov_inward` raises; the x-grid must be `x_in + h·arange(N)` |
| 8 | trusting one method | `radial_matrix_element_consensus` is the only public ME entry point; single-method calls are private |

### 4.5 Error budget (measured, n = 50 Rb, defaults)

| Source | Relative size |
|---|---|
| Numerov truncation (h = 0.001) | < 1e-10 |
| outer cutoff 2n(n+15) | ~1e-8 |
| trapezoid quadrature | ≤1e-8 |
| inner cutoff / divergence cut | ≤1e-6 (ME), ≤1.4e-4 if r_i moved to 20 a₀ |
| model-potential physics (A−B spread) | ~2e-6 (n=50), ~4e-5 (n=20) |
| semiclassical (A−K spread) | 1e-6 … 2e-3 |
| **dominant real uncertainty** | **input ν (doc 01 quantum defects), not this doc's numerics** |

---

## 5. Recommended Python API (`rydsim/radial.py`)

```python
@dataclass(frozen=True)
class ModelPotentialParams:
    """MSD94 parameters for one species. All atomic units. source/confidence per row in doc 02 §3."""
    z: int                                  # nuclear charge (Rb 37, Cs 55)
    alpha_c: float                          # core polarizability [a0^3]
    a1: tuple[float, float, float, float]   # index = min(l, 3)
    a2: tuple[float, float, float, float]
    a3: tuple[float, float, float, float]
    a4: tuple[float, float, float, float]
    r_c: tuple[float, float, float, float]
    source: str                             # provenance tag, e.g. "MSD94 via ARC+ryd-numerov 2026-08-10"

RB_MODEL_POTENTIAL: ModelPotentialParams    # §3.1 (shared by Rb-85/Rb-87)
CS_MODEL_POTENTIAL: ModelPotentialParams    # §3.2

def effective_charge(p: ModelPotentialParams, l: int, r: np.ndarray) -> np.ndarray:
    """Z_l(r), eq. (2.3). Vectorized over r. l>=4 returns ones (pure Coulomb regime)."""

def model_potential(p: ModelPotentialParams, l: int, j: float, r: np.ndarray,
                    *, include_so: bool = True) -> np.ndarray:
    """V_lj(r) = eq.(2.2) + eq.(2.4). Hartree. l>=4 -> -1/r + SO. r in a0, r > 0."""

@dataclass(frozen=True)
class RadialSolution:
    """Normalized radial state on the uniform x = sqrt(r) grid."""
    x: np.ndarray        # uniform grid [sqrt(a0)], increasing
    X: np.ndarray        # X(x) = R(r) r^{3/4}; 2*trapz(X^2 x^2 dx) == 1; zeroed inside r_cut
    n: int; l: int; j: float; nu: float
    r_cut: float         # inner truncation radius chosen by the divergence guard [a0]
    method: str          # "model_potential" | "coulomb"
    @property
    def r(self) -> np.ndarray: ...          # x**2
    @property
    def u(self) -> np.ndarray: ...          # r*R = X*sqrt(x)

def radial_wavefunction(species: AtomParams, n: int, l: int, j: float, *,
                        nu: float | None = None,      # default: species.effective_n(n, l, j)
                        h: float = 1e-3,
                        r_inner: float | None = None, # default alpha_c**(1/3)
                        r_outer: float | None = None, # default 2n(n+15)
                        include_so: bool = True,
                        method: str = "model_potential") -> RadialSolution:
    """Method A (or B with method="coulomb": V=-1/r, ignores model params & SO).
    Contracts: raises if nu <= l (no bound QDT orbital); warns if divergence guard
    truncated above 1.05*r_inner_turning_point; deterministic for fixed inputs."""

def radial_matrix_element(a: RadialSolution, b: RadialSolution, k: int = 1) -> float:
    """2 * trapz(Xa Xb x^{2k+2}) over the shared grid, eq. (2.7).
    Requires identical h and x[0]; raises otherwise. Sign is convention-laden; document."""

def radial_me_kaulakys(nu1: float, l1: int, nu2: float, l2: int) -> float:
    """|R| by §2.5 (k=1 only). Requires |l1-l2| == 1. Branches to (3/2)nu^2 e for |s|<1e-4."""

def radial_me_gordon(n1: int, l1: int, n2: int, l2: int) -> float:
    """Exact hydrogen <n1 l1|r|n2 l2>, |l1-l2|==1, via (2.10) in exact rational arithmetic,
    or (2.11) when n1==n2. Reference-grade; used only in tests and the delta->0 limit."""

def whittaker_u(nu: float, l: int, r: np.ndarray) -> np.ndarray:
    """Seaton-normalized QDT orbital u(r), eqs. (2.8)-(2.9) via scipy.special.hyperu.
    VALIDATION INSTRUMENT ONLY — no production caller (Method B is coulomb_wavefunction).
    Contract: raises IntegrityError if nu > WHITTAKER_NU_MAX = 20.0 (documented precision
    collapse, doc 02 §2.4/§7.3) or if ANY returned sample is non-finite (§4.4 pitfall 1b)."""

WHITTAKER_NU_MAX: float = 20.0   # normative fence, doc 02 §7.3 (was 25 before 2026-08-10)

def hyperu_hydrogen_error(nu: int, l: int = 0, n_samples: int = 300) -> float:
    """One row of the §2.4 table on the INSTALLED scipy: max relative deviation of the
    hyperu-Whittaker orbital from analytic hydrogen over [0.2, 1.8] nu^2. Integer nu only.
    The fence is re-measured with this, never assumed (audit R4)."""

def coulomb_wavefunction(nu: float, l: int, *, mu_mass: float = 1.0, h: float = 1e-3,
                         r_inner: float = 1e-4,
                         r_outer: float | None = None) -> RadialSolution:
    """Species-independent Method-B engine (V = -1/r, arbitrary real nu > l): the hydrogen
    benchmarks B1-B7/B13-B14 and the B12 cross-check run through this + radial_matrix_element.
    r_inner = 1e-4 a0 is required to reach B1/B7's stated measurements (§6 note); alkali
    Method B goes through radial_wavefunction(method="coulomb") and keeps r_i = alpha_c^(1/3)."""

def model_potential_defect(species: AtomParams, n: int, l: int, j: float, *,
                           h: float = 4e-3, r_min: float = 1e-6, include_so: bool = True,
                           params: ModelPotentialParams | None = None) -> float:
    """Solve the MSD94 potential as a genuine EIGENVALUE problem and return the predicted
    quantum defect. This — not the A-vs-B spread — is what guards the §3 tables (§6 note)."""

@dataclass(frozen=True)
class RadialMEResult:
    value: float          # Method A (model potential Numerov), the quoted number
    per_method: dict[str, float]   # {"model_potential":…, "coulomb":…, "kaulakys":…}
    spread_rel: float     # max pairwise |diff|/|value| -> the numerical uncertainty estimate
    spread_abs: float

def radial_matrix_element_consensus(species: AtomParams,
                                    state1: tuple[int, int, float],
                                    state2: tuple[int, int, float],
                                    k: int = 1, **grid_kwargs) -> RadialMEResult:
    """THE public entry point. Runs A, B and (k==1, both l<=... ) C; asserts spread_rel
    against per-regime ceilings (doc 02 §6) unless check=False; returns all three."""
```

Implementation notes: reuse `rydsim.numerov.numerov_inward` unchanged; keep the divergence guard in
`radial.py` (it is physics, not integration); vectorize potentials with numpy on the whole grid
before the Numerov loop; cache `RadialSolution`s keyed on `(species, n, l, j, h)` for ME matrices
(Stark maps in doc 07 request thousands of MEs).

---

## 6. Validation benchmarks (→ `tests/test_radial.py`; B15 in `tests/test_dipoles.py`)

All MEs in a₀, magnitudes. "Measured" = this session's harness, spec-default grid unless noted.

| ID | Quantity | Expected | Tolerance | Source | Confidence |
|----|----------|----------|-----------|--------|------------|
| B1 | H Numerov R(1s→2p) | 1.2902662020 (=128√6/243) | rel ≤ 1e-7 (measured 4e-12; 4.7e-12 as shipped — note ‡) | Gordon/B&S §63; 3 independent routes | VERIFIED |
| B2 | H Numerov R(2s→3p) | 3.0648154066 | rel ≤ 1e-7 (measured 8e-13) | exact-rational Gordon ≡ quad | VERIFIED |
| B3 | H Numerov R(2p→3d) | 4.7479916115 | rel ≤ 1e-7 | same | VERIFIED |
| B4 | H Numerov R(10s→11p) | 40.4352023233 | rel ≤ 1e-7 | same | VERIFIED |
| B5 | H Numerov \|R(50s→50p)\| | 3749.2499249850 | rel ≤ 1e-6 (measured 5.5e-10) | closed form (3/2)n√(n²−l²) | VERIFIED |
| B6 | H Numerov R(50s→51p) | 851.4038694455 | rel ≤ 1e-6 (measured 1.5e-8) | exact-rational Gordon | VERIFIED |
| B7 | Numerov global order: err(h=0.01)/err(h=0.005) on B1 | 16 (h⁴) | ∈ [8, 32] (measured 16.2; 16.29 as shipped — note ‡) | classic Numerov analysis | VERIFIED |
| B8 | Rb 50S₁/₂→50P₃/₂: A vs B spread | ≤ 1e-4 rel | measured 2.0e-6; also 51P/49P/60S pairs ≤ 4.2e-6; n=20 pair 4.2e-5 | this-session cross-method | VERIFIED |
| B9 | Rb/Cs n≈50 pairs: A vs Kaulakys spread | ≤ 5e-3 rel (≤ 1e-2 for MEs < 50 a₀) | measured 1e-6…2e-3 (worst: 50D₅/₂→51F₇/₂) | this-session cross-method | VERIFIED |
| B10 | Rb nS₁/₂→nP₃/₂ scaling: \|R\|/(ν ν′), n ∈ {40…70} | 1.13 | ∈ [1.10, 1.16] (measured 1.1312 @50, 1.1295 @60) | computed; coefficient LITERATURE-CONSISTENT with 1.5ν² × defect suppression | VERIFIED (as computed invariant) |
| B11 | Cs 50S₁/₂→50P₃/₂: \|R\|/(ν ν′); A−B | 1.1304; ≤1e-4 | measured 1.1304; 2.8e-6 | this-session | VERIFIED (defects: LITERATURE-RECALL) |
| B12 | hyperu-Whittaker vs Method B, pointwise u(r), classical region, **integer ν ≤ 20** | equal | rel ≤ 1e-6 (measured on scipy 1.17.1: 2.6e-11 @ ν=10, 2.4e-10 @ 15, 5.0e-8 @ 20; the "≤1.5e-8 @ ν=20" printed here before 2026-08-10 does **not** reproduce — the *tolerance* is unchanged, only the check value) | Numerov-independent closed form, §2.4 | VERIFIED |
| B12b | same at **non-integer ν** — the QDT case Method B actually ships: shape compared after a fitted scale, Seaton-vs-unit-norm offset pinned *separately* | shape equal; offset (ν = 10.5, r_inner = 1e-4 a₀) 7.3e-6 (l=0), 6.6e-4 (l=1), 3.0e-3 (l=2) | shape rel ≤ 1e-6 (measured ≤3.0e-11); offset reproduced to ±25 % | §2.4 — W irregular at origin ⇒ cutoff-dependent scale (r_inner → 1 a₀ moves the l=0 offset to 3.7e-5) | VERIFIED |
| B13 | Kaulakys vs Gordon, hydrogen ν=50, Δν=1,2 | equal | rel ≤ 2e-4 (measured 4.4e-5, 9.9e-5) | §2.5 vs §2.6 | VERIFIED |
| B14 | outer-cutoff adequacy: B6 at 2n(n+15) vs 2n(n+25) | equal | rel ≤ 5e-8 (measured 1.3e-8) | grid study | VERIFIED |
| B15 | Rb 5S₁/₂→5P₃/₂ model-potential radial ME | 5.57 | ∈ [5.45, 5.70] (regression band; computed 5.569 with ν from measured term energies) | this-session; **known +≈8% bias vs. experiment-derived ≈5.18** (Steck→doc 03 chain) | computed VERIFIED; bias LITERATURE-RECALL |
| B16 | Rb 50S₁/₂→50P₁/₂ vs 50P₃/₂ ME ratio | 1.0158 | ± 0.002 (measured 2550.6/2510.9) | j-dependence via δ; this-session | VERIFIED (defects LITERATURE-RECALL) |

pytest structure: B1–B7, B12–B14 are absolute (hydrogen/analytic — no recalled data); B8–B11, B16
are *invariants of the shipped code + doc 01 data* (re-derive ν from doc 01 at test time; if doc 01
values move within their error bars these stay inside tolerance); B15 is a bias-documentation
regression (asserts the bias is present AND stable, so nobody mistakes model-potential D-line
dipoles for accurate ones). **Where they live:** `tests/test_radial.py` (B1–B14, B16) and
`tests/test_dipoles.py` (B15, with the D-line closure); the `tests/test_spec02_benchmarks.py` file
named above and in the header note was never created — this pointer is corrected, not the plan.

**Note ‡ — B1–B6 run through the engine, and that requires `r_inner = 1e-4 a₀`.** B1–B6 are labelled
"H **Numerov** R(…)" and are now implemented that way: `coulomb_wavefunction` → the same scaled
equation (2.5), divergence guard, norm (2.6) and ME weight (2.7) that every shipped alkali number
goes through → `radial_matrix_element`, with exact-rational Gordon (§2.6) as truth. Likewise B7
(global order) and B12 (Whittaker cross-check). This is not covered by the A-vs-B spread: Methods A
and B share the identical ODE solver, grid, guard and quadrature, so a systematic error in the
scaled equation cancels exactly in A − B.

The stated measurements only reproduce at an inner cutoff of `r_inner ≈ 1e-4 a₀`, which is now the
`coulomb_wavefunction` default (it changed from 1e-2 on 2026-08-10; it is a pure hydrogen-benchmark
knob with no production caller — the alkali grids still start at r_i = α_c^{1/3}, §4.1). Measured
|rel| vs. exact Gordon at h = 1e-3:

| r_inner | B1 | B2 | B3 | B4 | B5 | B6 | B7 ratio |
|---|---|---|---|---|---|---|---|
| 1e-2 (old default) | **6.57e-7** | 8.2e-8 | 1.0e-11 | 6.3e-10 | 5.5e-10 | 1.9e-9 | **1.02** |
| 1e-3 | 6.6e-10 | 9.4e-11 | 1.3e-11 | 9.2e-11 | 5.6e-10 | 1.3e-9 | — |
| **1e-4 (default)** | **4.7e-12** | 2.9e-12 | 2.7e-11 | 2.7e-11 | 5.5e-10 | 1.9e-9 | **16.29** |
| 1e-5 | 6.7e-13 | 1.6e-11 | 5.4e-8 | 5.2e-11 | 5.5e-10 | 1.9e-9 | — |

At the old 1e-2 default B1 measures 6.57e-7 — 6.6× **outside** B1's own 1e-7 tolerance — and B7's
convergence ratio collapses to 1.02, i.e. the h⁴ order test had no teeth. Below 1e-4 the l = 2 row
(B3) degrades because the centrifugal term (2l+1/2)(2l+3/2)/x² drives h²g/12 toward 1 near the inner
edge, where the Numerov auxiliary f = 1 − h²g/12 loses meaning. **No §6 tolerance was changed to
accommodate any of this, and none may be** — the fix was the grid knob, not the bar.

**Note — what actually guards the §3 parameter tables.** It is **not** the A-vs-B spread (B8).
Everywhere else in this doc the energy is an *input* from doc 01's measured defects, so the model
potential only shapes the wavefunction inside the divergence-guard truncation radius: doubling Rb
a₃(l=0) moves the 50S→50P consensus ME by 4e-7 relative (A−B spread 2.02e-6 → 2.40e-6, 40× inside
the B8 ceiling) and flipping the sign of the UNVERIFIED Rb a₄(l=1) moves it by 6e-8 — the suite
stays green under both. The quantity MSD94 was *fitted* to is the eigenvalue, so that is what is
asserted: `rydsim.radial.model_potential_defect` solves the model potential as a genuine bound-state
problem and `tests/test_radial.py::test_msd94_reproduces_measured_quantum_defects` requires it to
reproduce doc 01's measured quantum defects (l-centroids at n = 12, guard band 1.2e-2; measured
residuals Rb S +3.25e-4, P +2.18e-3, D +9.98e-4, F +8.47e-4; Cs S +5.64e-4, P +5.38e-3, D +1.32e-3,
F −1.27e-4), with a companion test that pins deliberate parameter corruptions *outside* that band.
This supersedes the mitigation recorded in `00-integrity-audit.md` R20 — see that document's
parallel change.

---

## 7. Known limitations / where the model breaks down

1. **Low-n dipoles are ~5–10% wrong.** The MSD94 potential is fitted to *energies*; it does not
   guarantee dipole accuracy where the core overlap matters. Measured here: Rb 5S→5P₃/₂ = 5.57 a₀
   vs. ≈5.18 a₀ derived from the measured D2 lifetime (Steck chain, doc 03) — +7.6%. RydSim MUST
   take D-line and other low-n dipoles from experiment (docs 03/04), never from this machinery.
   A core-polarization correction to the dipole operator (r → r[1 − α_c/r³(1 − e^{−(r/r_c)³})]-type
   dressing) exists in the dimer literature and shrinks this bias; exact form UNVERIFIED here —
   optional future refinement, default OFF.
2. **Non-eigenvalue truncation.** Because E is the measured energy, the model-potential solution
   diverges at small r and is truncated (§4.2). Wavefunctions are unreliable inside r ≈ r_cut;
   any observable weighted toward r ≲ 5 a₀ (contact terms, hyperfine constants) is out of scope.
3. **Whittaker path ν ≤ 20 only** via scipy.special.hyperu (measured collapse, §2.4); production
   Method B is the pure-Coulomb Numerov equivalent.

   > **INTEGRATOR RULING (2026-08-10): the fence moves 25 → 20 — STRICTER.** Amends this item and
   > `00-integrity-audit.md` refusal #6, both of which read ν > 25; refusal #6 additionally
   > mis-states the exception as `ValueError`. Shipped as `rydsim.radial.WHITTAKER_NU_MAX = 20.0`;
   > `whittaker_u` raises **`IntegrityError`** (the house-rule type for a refusal-to-guess), not
   > `ValueError`.
   >
   > *Justification — the fence must sit where the method meets its OWN contract.* Benchmark B12
   > demands `rel ≤ 1e-6` pointwise. Measured on the installed scipy 1.17.1 via
   > `hyperu_hydrogen_error` (§2.4, second row):
   >
   > | ν | 12 | 20 | 25 | 28 | 30 | 35 | 40 |
   > |---|----|----|----|----|----|----|----|
   > | rel. err | 7.4e-12 | **3.5e-8** | **5.5e-6** | 1.9e-4 | 1.8e-3 | 0.49 | 93 |
   >
   > The band 20 < ν ≤ 25 measures up to 5.7e-6 — **5.7× the tolerance B12 asserts** (worst over
   > l = 0..2 at ν = 25: 5.48e-6, 4.71e-6, 5.72e-6; 3.5e-8 at ν = 20) — and
   > it rises to 0.49 by ν = 35. A `UserWarning` was judged insufficient: this is a number a
   > cross-check will *trust*, and a validation instrument running outside its claimed accuracy is
   > exactly the "plausible but wrong" hazard the house rule exists to stop.
   >
   > *The fence costs no physics.* `whittaker_u` is a **validation instrument only** — nothing on
   > the production Rydberg path calls it (grep-verified: Method B in production is the pure-Coulomb
   > Numerov `coulomb_wavefunction`; the sole callers are the B12 and audit-R4 tests). For
   > deliberate out-of-contract work — the R4 error table itself — `_whittaker_u_unguarded` remains
   > available in-module, private and explicitly named.
   >
   > *Re-measure, do not assume.* hyperu accuracy is scipy-version-dependent; the fence's location
   > is asserted against the installed scipy at test time (audit R4 binding form), including a lower
   > guard that fires if a future scipy makes ν = 25 better than 1e-7 (i.e. the fence has become
   > over-conservative and should be revisited rather than silently kept).

4. **Kaulakys degrades for small MEs and low ν**: cancellation-suppressed elements (e.g. nD→(n+1)F,
   \|R\| ~ 10 a₀) show 2e-3 method spread; ν ≲ 10 shows ~1e-3. It also assumes k = 1 (dipole) only.
5. **Orthogonality is not exact.** Truncation + fixed-E integration break strict ⟨ν,l|ν′,l⟩ = δ
   orthogonality; do not build unitary transformations from these states without re-orthogonalizing.
6. **High-l (l ≥ 4) treated as hydrogenic** (plus SO): correct to the extent core penetration is
   negligible; polarization-induced defects of high-l states come from doc 01's δ values, not the
   potential.
7. **No relativistic wavefunction corrections** beyond the SO potential term (no Darwin, no mass–
   velocity in the ODE): their energy effect is inside the measured defects; their wavefunction
   effect at Rydberg n is below all §6 tolerances but unquantified at n = 5 (folded into item 1).
8. **Fine structure of the same-(n,l) pair enters only through ν(j)** at high n (SO potential term
   measured at 2.7e-11 on MEs): the j-resolved accuracy is exactly the accuracy of doc 01's δ_lj.
9. **Rb-85 vs Rb-87**: same model potential; only μ and doc 01 defect sets differ (ME effect
   ≤ 1e-5). Cs-133 fully covered by §3.2.

---

*GreyNOC · RydSim spec 02 · 2026-08-10 · methods verified this session; harness →
`tests/test_radial.py` (+ `tests/test_dipoles.py` for B15). Amended 2026-08-10 post-audit: §2.4
integer-ν scoping of the norm equivalence, §3.3 corrected a₄(l=1) impact bound, §7.3 integrator
ruling moving the hyperu fence 25 → 20, §6 B12/B12b split + r_inner and MSD94-guard notes. No §6
tolerance was changed.*
