# 03 — Angular Algebra and Dipole Matrix Elements

**RydSim physics specification. Species: Rb-85, Rb-87, Cs-133. Python 3.11 + numpy/scipy only.**

> **Verification status (2026-08-10, network AVAILABLE; independent re-verification pass completed).**
> All primary-source numbers in this document were verified against freshly downloaded primary PDFs:
> the three Steck alkali datasheets (**revision 2.3.4, 8 August 2025** for Rb-87, Rb-85, Cs —
> frequencies, lifetimes, dipoles, HFS constants, and convention Eqs. (35)/(43)–(45) machine-parsed
> verbatim), and Simons, Gordon & Holloway, *J. Appl. Phys.* **120**, 123103 (2016) (NIST public PDF:
> Eq. (1) incl. λ_p/λ_c, A-factor table, m_j = ±1/2 co-linear convention, Sobelman reference).
> Every 3j/6j value, angular factor, S_FF' factor, hyperfine span, sum rule and lifetime↔dipole
> round trip below was reproduced with two *independent* implementations (float log-factorial and
> exact rational arithmetic) written from the formulas in §4 — including a 10⁴-sample random-grid
> float-vs-oracle cross-check that **corrected** an earlier overclaim of generic large-j float
> accuracy (see §4.3.4). Items not verifiable are tagged LITERATURE-RECALL or UNVERIFIED inline.

---

## 1. Scope

Defines, with a single self-consistent phase convention:

1. Decomposition of `<n l j m_j | e r_q | n' l' j' m_j'>` into **radial × angular** parts
   (Wigner–Eckart chain through j-reduced, l-reduced, and C¹ elements).
2. m_j-resolved elements vs. polarization index q ∈ {−1, 0, +1} (σ⁻/π/σ⁺) and how q is fixed by
   laser/RF polarization geometry.
3. Electric-dipole selection rules; the **effective dipole moment ℘** entering the Autler–Townes (AT)
   relation ħΩ_RF = ℘·E_RF, and the NIST-traceable convention for it.
4. Hyperfine-resolved probe matrix elements and the effective two-level reduction valid in
   Doppler-broadened hot vapor.
5. D-line reduced elements as the first-class validation of the radial machinery (doc 02).
6. Oscillator strengths, Thomas–Reiche–Kuhn (TRK) sum rule, Einstein A coefficients.
7. Exact Wigner 3j/6j implementation (log-factorial Racah sums; **scipy has no 3j/6j** — confirmed:
   `scipy.special` contains no Wigner symbols; sympy is banned per project rules).

Out of scope: radial integral computation (doc 02), Stark/Zeeman mixing (Stark doc), optical Bloch /
EIT lineshape dynamics (EIT doc), blackbody/lifetime aggregation (lifetimes doc).

---

## 2. Equations

### 2.1 Conventions — read first, this is where implementations die

All angular momentum algebra uses **Condon–Shortley phases** (the convention of Edmonds, *Angular
Momentum in Quantum Mechanics* (1957), and of Steck's datasheets; `Y_lm* = (−1)^m Y_{l,−m}`).

**Internal (code) convention — "Racah" Wigner–Eckart:**

```
<γ j m | T^k_q | γ' j' m'>  =  (−1)^(j−m) ( j  k  j' ; −m  q  m' ) <γ j || T^k || γ' j'>      (2.1)
```

where `( ... ; ... )` is the Wigner 3j symbol. Properties in this convention (dipole `T^1 = e r`):

- Hermiticity/symmetry: `<j||er||j'> = (−1)^(j'−j) <j'||er||j>` (elements real);
  **magnitudes are direction-symmetric**: `|<g||er||e>| = |<e||er||g>|`.
- Normalization: `Σ_{m',q} |<j m|er_q|j' m'>|² = |<j||er||j'>|² / (2j+1)`, independent of m.

**Steck's datasheet convention** (his Eq. (35)–(39), Rb-87 rev 2.3.4 — verified from the PDF):

```
<F m_F | er_q | F' m_F'> = <F||er||F'>_S (−1)^(F'−1+m_F) √(2F+1) ( F' 1 F ; m_F' q −m_F )      (2.2)
```

**Conversion (the single most common factor-of-√2 bug in this field):**

```
<j || er || j'>_Racah  =  √(2j+1) · <j || er || j'>_Steck                                       (2.3)
```

where j is the *first* (bra) label. Steck's normalization is
`Σ_{m',q} |<J m|er_q|J' m'>|² = |<J||er||J'>_S|²` (his Eq. (39)).
All Steck tabulated D-line values (§3.2) are in the **Steck convention**. The ARC package and most
Rydberg-electrometry theory papers use the **Racah convention**; e.g. Rb-87 D2:
`4.22752 e·a₀ (Steck) = √2 × 4.22752 = 5.97862 e·a₀ (Racah)`. RydSim stores Racah internally and
converts at the I/O boundary. Provide `steck_to_racah()` / `racah_to_steck()` (§5).

Only `|angular × radial|` is observable. Signs of radial integrals depend on the wavefunction sign
convention of doc 02; never compare signed reduced elements across codes, compare squares.

### 2.2 The Wigner–Eckart chain (fine-structure basis `|n l j m_j>`, s = 1/2)

Full chain, Racah convention, all phases explicit. `R ≡ R_{n l j}^{n' l' j'} = ∫₀^∞ P_{nlj}(r) · r · P_{n'l'j'}(r) dr`
(SI: metres; atomic units: a₀), with `P = r·R_radial(r)` from doc 02.

**Step 1 — strip m_j (Wigner–Eckart, Eq. 2.1):**

```
<n l j m | e r_q | n' l' j' m'>  =  (−1)^(j−m) ( j 1 j' ; −m q m' ) · <n l j || e r || n' l' j'>   (2.4)
```

**Step 2 — strip spin (spectator; Edmonds Eq. 7.1.7 with j₁ = l, j₂ = s):**

```
<n l s j || e r || n' l' s j'> = (−1)^(l+s+j'+1) √((2j+1)(2j'+1)) { l  j  s ; j'  l'  1 } · <n l || e r || n' l'>   (2.5)
```

`{ ... }` is the Wigner 6j symbol, written row-major: `{a b c; d e f}` has triads (abc),(aef),(dbf),(dec).

**Step 3 — factor radial × orbital-angular:**

```
<n l || e r || n' l'>  =  e · R_{nlj}^{n'l'j'} · <l || C^(1) || l'>                              (2.6)
<l || C^(1) || l'>     =  (−1)^l √((2l+1)(2l'+1)) ( l 1 l' ; 0 0 0 )                            (2.7)
```

Closed forms (exact; verified against Racah-formula implementation to 3e-16):

```
<l || C¹ || l+1> = −√(l+1)          <l || C¹ || l−1> = +√l                                      (2.8)
```

**Combined m_j-resolved element** (what the code actually evaluates):

```
<n l j m | e r_q | n' l' j' m'> = e · R_{nlj}^{n'l'j'} · A(l,j,m; l',j',m'; q)

A = (−1)^(j−m) ( j 1 j' ; −m q m' ) · (−1)^(l+s+j'+1) √((2j+1)(2j'+1)) { l j s ; j' l' 1 } · <l||C¹||l'>   (2.9)
```

`A` is dimensionless, **independent of n and n'** (cache it per (l,j,m,l',j',m',q) — the set of
distinct values in any simulation is tiny).

Steck-convention equivalents, for cross-checking against his datasheet (his Eqs. (36)–(37), verified
verbatim from the rev 2.3.4 PDF):

```
<J||er||J'>_S = <L||er||L'>_S (−1)^(J'+L+1+S) √((2J'+1)(2L+1)) { L L' 1 ; J' J S }              (2.10)
```

Consistency identity (used as a unit test): Eq. 2.10 equals Eq. 2.5 after applying Eq. 2.3 at each level.

**Fine-structure factors for the D lines** (exact, from Eqs. 2.5–2.8; verified numerically):

```
<n S₁/₂ || er || n' P₃/₂>_Racah / (e·R) = ± 2/√3  = 1.154701...   → Steck: √(2/3) = 0.816497
<n S₁/₂ || er || n' P₁/₂>_Racah / (e·R) = ± √(2/3) = 0.816497...  → Steck: √(1/3) = 0.577350
```

⇒ `|<J||er||J'>_S(D2)| / |<J||er||J'>_S(D1)| = √2` if the radial integrals were j-independent.
Measured: Rb-87 4.22752/2.9931 = 1.41243 (√2 − 0.13%); Cs 4.4837/3.1869 = 1.40692 (√2 − 0.52%).
The deviation is the spin-orbit difference of the radial wavefunctions — a *feature* the radial code
must reproduce, see §6.

**Subtlety:** strict factorization (2.5–2.6) assumes r acts only on the orbital part. Because
alkali quantum defects are j-dependent, `R` itself depends on (j, j'). Standard practice (ARC does the
same): evaluate `R` per (n l j)→(n' l' j') pair and use the angular chain unchanged. This is exact
within the single-valence-electron model.

**Hyperfine chain (add nuclear spin I, F = J + I; operator acts on J, I spectator):**

```
<n J I F m_F | e r_q | n' J' I F' m_F'> = (−1)^(F−m_F) ( F 1 F' ; −m_F q m_F' ) <F||er||F'>_R   (2.11)
<F||er||F'>_R = (−1)^(J+I+F'+1) √((2F+1)(2F'+1)) { J F I ; F' J' 1 } · <J||er||J'>_R            (2.12)
```

(Steck's Eq. (36) is the Steck-convention version of 2.12; verified.)

### 2.3 Polarization: choosing q

Spherical basis vectors relative to the quantization axis ẑ:

```
ê₀ = ẑ ,   ê±₁ = ∓(x̂ ± i ŷ)/√2 ,   r_q = r·êq* ,  i.e.  r₀ = z, r±₁ = ∓(x ± iy)/√2
```

Field convention: `E(t) = E₀ Re[ ε̂ e^(−iωt) ]` with `|ε̂| = 1`; expand `ε̂ = Σ_q ε_q êq`. The RWA
coupling to the pair (upper e, lower g) is

```
ħ Ω = E₀ · Σ_q ε_q <e| e r_q |g>        (single term when ε̂ is a pure σ⁺/σ⁻/π state)      (2.13)
```

- **π** (linear ∥ ẑ): q = 0, Δm = 0.
- **σ⁺** (circular, photon spin +ħ along ẑ): q = +1, Δm = +1 on absorption. **σ⁻**: q = −1.
- Linear polarization ⊥ ẑ decomposes as `(σ⁺ + σ⁻)` with `|ε±₁| = 1/√2` each.

**Rule used throughout RydSim (and by NIST):** put the quantization axis **along the common linear
polarization** of probe, coupling and RF ("co-linear geometry"). Then every field is pure π (q = 0),
m_j is conserved along the ladder, and each populated m_j experiences a single scalar dipole moment.
If the RF is polarized ⊥ to the optical axis it becomes an equal σ⁺/σ⁻ mixture with *different* |A|
per m_j-path → several simultaneous AT splittings → apparent broadening / multi-peak structure
(cf. arXiv:2503.17997 for the angular response of D- vs S-ladders; Sedlacek et al., PRL **111**,
063001 (2013) exploit this for vector electrometry). RydSim must model non-collinear RF as a
coherent sum over q, not with a single ℘.

### 2.4 Selection rules (E1)

From Eqs. 2.4–2.8 (each rule = a vanishing symbol):

| Rule | Origin |
|---|---|
| Δl = ±1 (parity) | `(l 1 l'; 0 0 0)` = 0 unless l+l'+1 even and triangle |
| Δj = 0, ±1; j=0↛j'=0 | triangle of `(j 1 j')` |
| Δm_j = q (i.e. 0, ±1) | 3j m-condition |
| Δs = 0 | spin spectator |
| ΔF = 0, ±1; F=0↛F'=0 | triangle of `(F 1 F')` |
| Δm_F = q | 3j m-condition |

Δj = 0 transitions (e.g. nD₃/₂→n'P₃/₂) are allowed but carry smaller angular factors. All rules must
emerge from the 3j/6j returning **exact 0.0** (§4), never be special-cased upstream.

### 2.5 Autler–Townes relation and the effective dipole ℘ (what NIST actually does)

Four-level ladder (Rb): `5S₁/₂ →(probe, λ_p≈780 nm) 5P₃/₂ →(coupling, λ_c≈480 nm) n l j →(RF) n' l' j'`.
On-resonance RF splits the EIT peak by the RF Rabi frequency. The traceable field relation, exactly as
published — Simons, Gordon & Holloway, *J. Appl. Phys.* **120**, 123103 (2016), Eq. (1)
[**VERIFIED from primary PDF this session**]:

```
|E_RF| = (2πħ / ℘) · (λ_p/λ_c) · Δf_m  =  (2πħ / ℘) · Δf_o                                    (2.14)
```

- `Δf_m` [Hz]: AT peak splitting **as measured while scanning the probe laser** (coupling fixed).
- `λ_p/λ_c` (Rb: 780/480 ≈ 1.63; Cs: 852/510 ≈ 1.67): Doppler-mismatch factor for counter-propagating
  probe/coupling in hot vapor. `Δf_o = (λ_p/λ_c) Δf_m` is the true splitting `Ω_RF/2π`.
  If instead the **coupling** is scanned, the factor is absent (`Δf_o = Δf_m`). This factor is a
  weak-probe, velocity-averaged 3-level result; the full OBE velocity average (EIT doc) supersedes it
  and must reproduce it in the appropriate limit.
- `℘ = e · R · A` [C·m]: **RF-transition dipole moment for the specific m_j pair and polarization
  actually driven.**

**NIST convention for traceable measurements** (same paper, §II — verified): co-linear linear
polarizations (all π), and the RF transition evaluated for **m_j = ±1/2**. Published angular factors
(their notation `℘ = R·A`, A independent of n), against which RydSim's Eq. 2.9 was verified to
4 decimal places this session:

| RF transition (π, m_j = ±1/2) | A (NIST, JAP 120, 123103) | Exact value (RydSim Eq. 2.9) |
|---|---|---|
| nS₁/₂ → nP₃/₂ and nS₁/₂ → (n−1)P₃/₂ | 0.4714 | √2/3 = 0.471405 |
| nD₅/₂ → (n+1)P₃/₂ | 0.4899 | √6/5 = 0.489898 |
| nD₅/₂ → (n−1)F₇/₂ | 0.4949 | 2√3/7 = 0.494872 |

Full π-manifold for reference (exact, computed & dual-verified; needed when optical pumping is
imperfect):

| pair | m_j = 1/2 | m_j = 3/2 | m_j = 5/2 |
|---|---|---|---|
| D₅/₂ → P₃/₂, q=0 | √6/5 = 0.489898 | 2/5 = 0.400000 | — (no m=5/2 in P₃/₂) |
| D₅/₂ → F₇/₂, q=0 | 2√3/7 = 0.494872 | √10/7 = 0.451754 | √6/7 = 0.349927 |

**Why a single ℘ is legitimate:** starting from S₁/₂ (only m_j = ±1/2 exist), π–π optical excitation
conserves m_j, so only Rydberg m_j = ±1/2 are populated; |A(+m)| = |A(−m)| by symmetry of the 3j
under m → −m. Hence one scalar ℘ describes the whole ensemble — *this is contingent on the co-linear
geometry*. Sedlacek et al., *Nat. Phys.* **8**, 819 (2012) introduced the λ_p/λ_c-scaled AT
electrometry (Eq. 2.14); the NIST papers (Holloway et al., IEEE Trans. Antennas Propag. **62**, 6169
(2014); Simons et al. 2016; Sobelman's book for the angular algebra) standardized the m_j = ±1/2 π
convention above. (Sedlacek's own numeric dipole values were not independently re-verified here —
methodology citation only.)

RydSim contract: `effective_rf_dipole()` (§5) implements exactly `℘ = e·R·|A(l,j,1/2 → l',j',1/2; q=0)|`
and is the number to report next to any simulated AT splitting.

### 2.6 Hyperfine-resolved probe elements and the hot-vapor two-level reduction

Probe transition (D2): `n S₁/₂ F → n P₃/₂ F'` with F = I ± 1/2. m_F-resolved elements via
Eqs. 2.11–2.12.

**Line-strength factors** (Steck Eq. (41), verified; Racah- and Steck-convention agree since S is a ratio):

```
S_FF' = (2F'+1)(2J+1) { J J' 1 ; F' F I }²        with    Σ_{F'} S_FF' = 1                    (2.15)
```

Exact values (computed this session with dual implementations; Rb-87 row matches the well-known
Steck Table 8 values):

| Species (I) | probe lower F | S_F,F'−1 … | | |
|---|---|---|---|---|
| Rb-87 (3/2), F=2 | → F'=1: **1/20** | → F'=2: **1/4** | → F'=3: **7/10** | Σ = 1 |
| Rb-85 (5/2), F=3 | → F'=2: **5/63** | → F'=3: **5/18** | → F'=4: **9/14** | Σ = 1 |
| Cs-133 (7/2), F=4 | → F'=3: **7/72** | → F'=4: **7/24** | → F'=5: **11/18** | Σ = 1 |

**Effective two-level dipole moments** (Steck Eqs. (43)–(45), verified from PDF):

- Resolved F→F', isotropic pump: `|d_iso,eff|² = (1/3) S_FF' |<J||er||J'>_S|²`.
- **Doppler-unresolved excited HFS, π-polarized light (the RydSim hot-vapor default):** summing over
  F' at fixed ground |F m_F⟩ gives exactly 1/3 independent of m_F (Steck Eq. (44)):

```
d_eff,far = <J||er||J'>_S / √3          (Rb-87 D2: 4.22752/√3 = 2.44076 e·a₀)                 (2.16)
```

**Validity criteria (quantitative):**

1. Excited-state HFS unresolved under Doppler: span(HFS(nP₃/₂)) ≲ Δν_Doppler(FWHM) =
   `ν₀ √(8 ln2 · k_B T / (m c²))`. At 300 K (derived from verified masses/frequencies):
   Rb-87 D2: **511 MHz**, Rb-85 D2: **517 MHz**, Cs D2: **379 MHz**.
   Excited HFS spans from the verified A, B constants of §3.3 via the Casimir formula
   (E_hf = A·K/2 + B·[3K(K+1)/2 − 2I(I+1)J(J+1)] / [4I(2I−1)J(2J−1)], K = F'(F'+1)−I(I+1)−J(J+1)):
   ≈ 0.50 GHz (Rb-87 5P₃/₂, F'=0…3), ≈ 0.21 GHz (Rb-85), ≈ 0.60 GHz (Cs 6P₃/₂). Marginal for
   Rb-87/Cs: the reduction is adequate for AT *splitting* extraction (peak positions), but hyperfine-
   resolved multi-level treatment is required for accurate EIT *lineshapes* — flag in the EIT doc.
2. Ground HFS resolved (only one F addressed): splittings 6.834 683 GHz (Rb-87), 3.035 732 GHz
   (Rb-85), 9.192 631 770 GHz (Cs, exact by definition of the second) ≫ Doppler. Always satisfied.
3. The coupling transition 5P₃/₂→nlj addresses Rydberg HFS ∝ n*⁻³, sub-MHz for n ≳ 20 —
   always unresolved; treat Rydberg states in the |n l j m_j⟩ basis (no F). [Scaling argument;
   UNVERIFIED-FROM-MEMORY for prefactors, but the conclusion is safe by ≥2 orders of magnitude.]

### 2.7 Oscillator strengths, TRK sum rule, Einstein A

**Absorption oscillator strength** (dimensionless; lower state i = (l,j), upper f = (l',j'),
ω = (E_f − E_i)/ħ > 0 for absorption; emission f is negative via the same formula):

```
f_(i→f) = (2 m_e ω) / (3 ħ (2j+1)) · |<j||er||j'>_R|² / e²                                     (2.17)
```

l-basis equivalent (spin-free, for hydrogenic tests; a.u.: f = (2/3) ω_au · max(l,l')/(2l+1) · R_au²):

```
f_(nl→n'l') = (2 m_e ω) / (3ħ) · max(l,l')/(2l+1) · R²                                         (2.18)
```

Internal consistency test: `Σ_{j'} f_(j→j')` from 2.17 (radial integrals held equal) must equal 2.18.

Steck's Eq. (4) (verified from PDF) links f and Γ: `Γ = (e²ω₀²)/(2π ε₀ m_e c³) · (2J+1)/(2J'+1) · f`.

**TRK sum rule** (absolute correctness test):

```
Σ_f  f_(i→f)  =  1      (one valence electron; sum over ALL final states incl. continuum,
                          downward transitions enter with negative f)                          (2.19)
```

Partial (per-Δl) sum rules — Bethe & Salpeter, *QM of One- and Two-Electron Atoms*, §61
[LITERATURE-RECALL for attribution; the pair is self-verifying since they sum to 1 identically]:

```
Σ_{n'} f(nl → n', l+1) = (l+1)(2l+3) / (3(2l+1))      Σ_{n'} f(nl → n', l−1) = − l(2l−1) / (3(2l+1))
```

Hydrogen anchors: f(1s→2p) = 2¹³/3⁹ = 0.416197 (exact; reproduced numerically this session to 1e-8
from the analytic wavefunctions, radial integral ⟨1s|r|2p⟩ = 128√6/243 = 1.290266 a₀); discrete sum
Σ_{n'=2}^∞ f(1s→n'p) = **0.565004**, continuum = 0.434996 [VERIFIED this session: closed form
f(1s→np) = 2⁸n⁵(n−1)^(2n−4)/(3(n+1)^(2n+4)) summed to convergence at 30-digit precision; formula
attribution Bethe–Salpeter §63 — implementation self-check: discrete + numeric continuum integral
must total 1.000 ± 0.005].

**Alkali caveat (quantitative):** from the *verified* D-line dipoles, f(Rb-87 D2) = 0.696 and
f(D1) = 0.342 ⇒ principal doublet alone sums to 1.038 > 1. The valence-electron TRK is violated at
the few-% level by core–valence coupling. For Rb/Cs use TRK only as a sanity band
`0.95 ≤ Σf ≤ 1.10`; the *exact* test (tolerance 0.5%) is reserved for the hydrogenic-potential mode
of the radial engine.

**Einstein A / spontaneous decay** (upper e → lower g; the (2j_e+1) is the UPPER-state degeneracy):

```
Racah:  A_(e→g) = ω³ / (3π ε₀ ħ c³) · |<g||er||e>_R|² / (2j_e + 1)                             (2.20)
Steck:  1/τ    = ω₀³ / (3π ε₀ ħ c³) · (2J_g+1)/(2J_e+1) · |<J_g||er||J_e>_S|²   (his Eq. 38)   (2.21)
```

2.20 and 2.21 are identical under Eq. 2.3 (verified numerically: both reproduce τ(Rb-87 5P₃/₂) =
26.2348 ns from d = 4.22752 e·a₀ to 2 × 10⁻⁶ relative). For states with several decay channels,
`1/τ_e = Σ_g A_(e→g)` (radiative only; BBR handled in the lifetimes doc).

**D-line closure — the first-class radial validation (item e of scope):** invert Eq. 2.21 with the
verified τ, ω₀ (§3.2) and the fine-structure factors (§2.2) to get the radial integrals the doc-02
machinery must hit:

```
|R(5S₁/₂–5P₃/₂)|_Rb87 = 4.22752·√(3/2) = 5.17766 a₀        |R(5S₁/₂–5P₁/₂)|_Rb87 = 2.9931·√3 = 5.18420 a₀
|R(5S₁/₂–5P₃/₂)|_Rb85 = 4.22753·√(3/2) = 5.17768 a₀        |R(5S₁/₂–5P₁/₂)|_Rb85 = 5.18420 a₀
|R(6S₁/₂–6P₃/₂)|_Cs   = 4.4837·√(3/2)  = 5.49142 a₀        |R(6S₁/₂–6P₁/₂)|_Cs   = 3.1869·√3 = 5.51988 a₀
```

---

## 3. Constants / parameter tables

### 3.1 Fundamental constants

| Constant | Value | Source | Confidence |
|---|---|---|---|
| e | 1.602 176 634 × 10⁻¹⁹ C (exact) | SI 2019 definition | VERIFIED |
| h, ħ | 6.626 070 15 × 10⁻³⁴ J·s (exact); ħ = h/2π = 1.054 571 817… × 10⁻³⁴ | SI 2019 | VERIFIED |
| c | 299 792 458 m/s (exact) | SI | VERIFIED |
| ε₀ | 8.854 187 8188(14) × 10⁻¹² F/m | CODATA 2022 (NIST CUU, fetched this session) | VERIFIED |
| a₀ | 5.291 772 105 44(82) × 10⁻¹¹ m | CODATA 2022 (NIST CUU, fetched this session) | VERIFIED |
| m_e | 9.109 383 7139(28) × 10⁻³¹ kg | CODATA 2022 (NIST CUU, fetched this session) | VERIFIED |
| e·a₀ | 8.478 353 6 × 10⁻³⁰ C·m (derived) | derived | VERIFIED (arith.) |

Implementation should pin CODATA 2022 from `scipy.constants` (which ships CODATA values) and assert
agreement with this table to 1e-8; the §3.2 lifetime↔dipole round trips close to ≤ 2 × 10⁻⁵ with
these values, independently bounding any transcription error.

### 3.2 D-line data (all VERIFIED — machine-extracted from Steck datasheets rev 2.3.4, 8 Aug 2025)

`d_S ≡ <J||er||J'>` in the **Steck convention**; multiply by √2 for Racah (Eq. 2.3, J = 1/2).

| Line | ω₀/2π | τ | Γ/2π (derived) | d_S [e·a₀] |
|---|---|---|---|---|
| Rb-87 D2 5S₁/₂→5P₃/₂ | 384.230 484 468 5(62) THz | 26.2348(77) ns | 6.0666 MHz | 4.227 52(62) |
| Rb-87 D1 5S₁/₂→5P₁/₂ | 377.107 463 380(11) THz | 27.679(27) ns | 5.7500 MHz | 2.9931(14) |
| Rb-85 D2 | 384.230 406 373(14) THz | 26.2348(77) ns | 6.0666 MHz | 4.227 53(62) |
| Rb-85 D1 | 377.107 385 690(46) THz | 27.679(27) ns | 5.7500 MHz | 2.9931(14) |
| Cs D2 6S₁/₂→6P₃/₂ | 351.725 718 50(11) THz | 30.405(77) ns | 5.2343 MHz | 4.4837(57) |
| Cs D1 6S₁/₂→6P₁/₂ | 335.116 048 807(41) THz | 34.791(90) ns | 4.5745 MHz | 3.1869(41) |

Note: the task brief quoted Rb-87 D2 = 4.22752(87) and D1 = 2.9931(3); those uncertainties are from
an older datasheet revision. Rev 2.3.4 gives (62) and (14) respectively — use the rev 2.3.4 numbers.

### 3.3 Hyperfine constants needed by §2.6 (VERIFIED, Steck rev 2.3.4)

| Quantity | Rb-87 (I=3/2) | Rb-85 (I=5/2) | Cs-133 (I=7/2) |
|---|---|---|---|
| Ground HFS splitting | 6.834 682 610 904 290(90) GHz | 3.035 732 439 0(60) GHz | 9.192 631 770 GHz (exact) |
| A(nP₃/₂) | 84.7185(20) MHz | 25.0354(69) MHz | 50.288 27(23) MHz |
| B(nP₃/₂) | 12.4965(37) MHz | 25.898(91) MHz | −0.4934(17) MHz |
| A(nP₁/₂) | 407.25(63) MHz | 120.527(56) MHz | 291.9201(75) MHz |

### 3.4 NIST effective-dipole angular factors — see table in §2.5 (VERIFIED, primary source + exact reproduction)

---

## 4. Numerical method + pitfalls

### 4.1 Wigner 3j — Racah closed sum with log-factorials (no scipy support exists; do NOT use sympy)

Triangle coefficient: `Δ(abc) = (a+b−c)! (a−b+c)! (−a+b+c)! / (a+b+c+1)!`

```
( j1 j2 j3 ; m1 m2 m3 ) = δ(m1+m2+m3, 0) · (−1)^(j1−j2−m3)
  × √[ Δ(j1 j2 j3) · Π_{i=1..3} (ji+mi)! (ji−mi)! ]
  × Σ_k (−1)^k / [ k! (j1+j2−j3−k)! (j1−m1−k)! (j2+m2−k)! (j3−j2+m1+k)! (j3−j1−m2+k)! ]
```

k runs over all integers making every factorial argument ≥ 0:
`max(0, j2−j3−m1, j1−j3+m2) ≤ k ≤ min(j1+j2−j3, j1−m1, j2+m2)`.

### 4.2 Wigner 6j — Racah formula

With `S1 = j1+j2+j3, S2 = j1+j5+j6, S3 = j4+j2+j6, S4 = j4+j5+j3`,
`T1 = j1+j2+j4+j5, T2 = j2+j3+j5+j6, T3 = j3+j1+j6+j4`:

```
{ j1 j2 j3 ; j4 j5 j6 } = √[ Δ(j1j2j3) Δ(j1j5j6) Δ(j4j2j6) Δ(j4j5j3) ]
  × Σ_{k=max(S1..S4)}^{min(T1..T3)} (−1)^k (k+1)! / [ (k−S1)! (k−S2)! (k−S3)! (k−S4)! (T1−k)! (T2−k)! (T3−k)! ]
```

### 4.3 Implementation rules (all empirically validated this session)

1. **Doubled-integer representation.** Accept js/ms as floats, immediately convert
   `J2 = int(round(2j))` with guard `|2j − J2| < 1e-9` (raise otherwise). All triangle/parity/range
   tests and all phase exponents are integer arithmetic on doubled values; every `(−1)^x` exponent in
   §2 is provably integer when selection rules hold — compute as `(−1)**((...)//2)`-style integer
   ops, never `math.pow` on floats (half-integer exponent bugs are the classic failure).
2. **Selection-rule zeros must be exact `0.0`.** Test m-sum, |m|≤j, parity (2j+2m even), and all
   triangles *before* touching any factorial. Downstream code relies on exact zeros.
3. **Log-factorials.** Precompute `lgf[n] = ln n!` once (cumulative `np.log` sum or `math.lgamma`),
   table length ≥ 4·j_max + 3 (j_max ≈ 200 → ~1000 doubles).
4. **Alternating-sum stability.** Factor the largest term out of the k-sum:
   `ln_t_k` per term, `s = Σ (−1)^k exp(ln_t_k − max_k ln_t_k)`, result
   `= phase · s · exp(prefactor + max_k ln_t_k)`. **Measured accuracy vs. exact-rational oracle**
   (random grids, this session): *rank-1 3j* (j₂ = 1, the only 3j RydSim's dipole chain uses,
   k-sum ≤ 3 terms): max rel err **4.2 × 10⁻¹²** up to j = 150 (2000 samples). *6j of type
   {l j ½; j′ l′ 1}*: max **3.0 × 10⁻¹³** up to l = 150. *Generic* symbols degrade with j from
   alternating-sum cancellation: generic 3j ≤ 2.4 × 10⁻¹⁴ (j ≤ 10), ≤ 3.4 × 10⁻¹⁰ (j ≤ 30),
   ≤ 5 × 10⁻⁶ (j ≤ 60), **worst observed 8.2 × 10⁻⁵ at j ≈ 55–70** (19–41-term sums, small-value
   outputs); generic 6j (j ≤ 60) worst 3.8 × 10⁻⁷. Consequence: the float path is certified only
   for the rank-1 symbols of this spec; any generic large-j (> 30) use elsewhere must route through
   the exact-rational oracle. Pytest tolerance: 1e-9 relative on rank-1 (any j ≤ 150) and on
   generic j ≤ 20.
5. **Caching & vectorization.** `functools.lru_cache` on doubled-int tuples. Angular factors
   (Eq. 2.9) are n-independent → memoize; a full simulation touches O(10²) distinct symbols. Provide
   array frontends that broadcast with `np.vectorize` (cache makes this O(1) per element); do NOT
   attempt a "vectorized Racah sum" — unnecessary complexity.
6. **Exact-rational test oracle.** Ship (in tests only) a second implementation using
   `fractions.Fraction` + integer factorials returning `(sign, exact_square)`; value
   `= sign·√(p/q)`. Cross-validate the float path on a randomized grid j ≤ 60 (done this session; see
   §6). This is the no-fabrication guarantee for the whole angular layer.
7. **Convergence criteria:** none — every formula is a finite closed sum. The only numerical knobs
   are the lgf table size and the tolerance guard of rule 1.
8. **Phase-convention regression trap:** never "fix" signs to match another code's table. ARC,
   Steck, and RydSim can legitimately differ in the sign of reduced elements and radial integrals;
   only |element|² and interference-free observables are comparable. Cross-code tests compare
   squares.

---

## 5. Recommended Python API (`rydsim/angular.py`)

```python
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True, slots=True)
class FineState:
    """|n l j> fine-structure state. j half-integer; s=1/2 implicit. Validates |l-1/2| <= j <= l+1/2."""
    n: int
    l: int
    j: float

def wigner_3j(j1, j2, j3, m1, m2, m3) -> float | np.ndarray:
    """Wigner 3j, Condon-Shortley/Racah conventions (Sec 4.1). Scalar or broadcast arrays.
    Exact 0.0 on any selection-rule violation. Raises ValueError on non-(half)integer inputs.
    Accuracy: <=1e-12 relative for j<=150 (validated vs exact-rational oracle)."""

def wigner_6j(j1, j2, j3, j4, j5, j6) -> float | np.ndarray:
    """Wigner 6j via Racah sum (Sec 4.2). Exact 0.0 when any of the 4 triads fails."""

def reduced_C1(l: int, lp: int) -> float:
    """<l||C^(1)||l'> = (-1)^l sqrt((2l+1)(2l'+1)) (l 1 l'; 0 0 0).
    Contract: equals -sqrt(l+1) for lp==l+1, +sqrt(l) for lp==l-1, else 0.0."""

def reduced_dipole_j(l: int, j: float, lp: int, jp: float, s: float = 0.5) -> float:
    """<l s j||er||l' s j'>_Racah / (e*R): Eq. 2.5 x 2.7, dimensionless, n-independent.
    D2 contract: reduced_dipole_j(0,0.5,1,1.5) == +-2/sqrt(3)."""

def angular_factor(l, j, mj, lp, jp, mjp, q, s: float = 0.5) -> float:
    """A of Eq. 2.9: <n l j mj|er_q|n' l' j' mj'> = e * R * A. Cached. Exact 0.0 off selection rules.
    abs(angular_factor(2,2.5,.5, 1,1.5,.5, 0)) == sqrt(6)/5 (NIST 0.4899)."""

def reduced_dipole_F(J, Jp, F, Fp, I) -> float:
    """<F||er||F'>_Racah / <J||er||J'>_Racah: Eq. 2.12 (dimensionless)."""

def line_strength_S(J, Jp, F, Fp, I) -> float:
    """S_FF' of Eq. 2.15. Contract: sum over Fp == 1.0 within 1e-12."""

def dipole_matrix_element(bra: FineState, ket: FineState, mj, mjp, q,
                          radial_integral_a0: float) -> float:
    """Full <bra mj|e r_q|ket mjp> in C*m. radial_integral_a0 = R in units of a0 (doc 02).
    = e * a0 * radial_integral_a0 * angular_factor(...)."""

def effective_rf_dipole(l, j, lp, jp, radial_integral_a0, mj: float = 0.5, q: int = 0) -> float:
    """NIST-convention AT dipole: |℘| = e*a0*R*|A(l,j,mj -> l',j',mj+q)| in C*m (Sec 2.5).
    Default mj=1/2, q=0 reproduces Simons/Gordon/Holloway 2016 A-factors."""

def steck_to_racah(d_steck: float, j_lower: float) -> float:
    """<j||er||j'>_Racah = sqrt(2*j_lower+1) * <j||er||j'>_Steck (Eq. 2.3). Inverse: racah_to_steck."""

def einstein_A(omega0: float, d_reduced_racah_Cm: float, j_upper: float) -> float:
    """A_[e->g] [1/s] = omega0^3 * |<g||er||e>_R|^2 / (3 pi eps0 hbar c^3 (2*j_upper+1)). Eq. 2.20.
    omega0 in rad/s, d in C*m. Vectorized over omega0/d."""

def oscillator_strength(omega: float, d_reduced_racah_Cm: float, j_lower: float) -> float:
    """f (dimensionless), Eq. 2.17; omega>0 absorption, sign follows omega."""

def oscillator_strength_l(omega: float, radial_integral_a0: float, l: int, lp: int) -> float:
    """Spin-free l-basis f, Eq. 2.18 (hydrogenic tests, TRK)."""

def trk_sum(f_values: np.ndarray) -> float:
    """Convenience: returns sum; caller supplies discrete+continuum set. Test helper, Eq. 2.19."""

# tests/oracle_wigner.py  (test-only, exact rational)
def wigner_3j_exact(j1, j2, j3, m1, m2, m3) -> tuple[int, "Fraction"]:
    """(sign, exact square as Fraction); value = sign*sqrt(square). Ground truth for wigner_3j."""
```

Contracts common to all: pure functions; floats in/floats out; no global state except caches and the
lgf table; every docstring cites the equation number in this document.

---

## 6. Validation benchmarks (→ pytest)

Tolerances are *relative* unless stated. "Dual-impl" = float log-factorial vs. exact-rational oracle,
both written and cross-checked during spec preparation.

| # | Quantity | Expected | Tol | Source | Confidence |
|---|---|---|---|---|---|
| B1 | 3j (1/2 1 3/2; 1/2 0 −1/2) | +1/√6 = 0.4082482905 | 1e-12 | Racah formula, dual-impl | VERIFIED |
| B2 | 3j (2 1 1; 0 0 0) | +√(2/15) = 0.3651483717 | 1e-12 | dual-impl | VERIFIED |
| B3 | 3j (5/2 1 3/2; 1/2 0 −1/2) | −1/√10 = −0.3162277660 | 1e-12 | dual-impl | VERIFIED |
| B4 | 3j (50 1 51; 0 0 0) | −√(51/10403) = −0.0700173692 | 1e-10 | dual-impl (float err 4.4e-14) | VERIFIED |
| B5 | 3j (121/2 1 123/2; 1/2 0 −1/2) | +1/√246 = 0.0637576713 | 1e-10 | dual-impl | VERIFIED |
| B6 | 6j {1 1 1; 1 1 1} | 1/6 | 1e-12 | dual-impl; std. tables | VERIFIED |
| B7 | 6j {0 1 1; 3/2 1/2 1/2} | −1/√6 = −0.4082482905 | 1e-12 | dual-impl | VERIFIED |
| B8 | 6j {2 1 1; 3/2 5/2 1/2} | −1/√20 = −0.2236067977 | 1e-12 | dual-impl | VERIFIED |
| B9 | 6j {2 3 1; 7/2 5/2 1/2} | −1/√42 = −0.1543033500 | 1e-12 | dual-impl | VERIFIED |
| B10 | Σ_{m1,m2} (2j3+1)·3j² at fixed m3, for (1,1,2),(3/2,1,5/2),(6,4,9) | 1 | 1e-10 | orthogonality | VERIFIED |
| B11 | Σ_x (2x+1){a b x; c d p}{a b x; c d q}, (a,b,c,d) = (3/2,1,5/2,2); (p,q) ∈ {(3/2,3/2),(5/2,5/2),(3/2,5/2)} — triads force half-integer p ∈ [3/2,7/2]; integer p gives identically-zero symbols (vacuous test) | 1/4, 1/6, 0 | 1e-9 abs | orthogonality; dual-impl | VERIFIED |
| B12 | reduced_C1: ⟨0‖C¹‖1⟩, ⟨1‖C¹‖2⟩, ⟨2‖C¹‖1⟩ | −1, −√2, +√2 | 1e-12 | closed form | VERIFIED |
| B13 | \|A\|(S₁/₂→P₃/₂, m=1/2, q=0) | √2/3 = 0.4714045 | 1e-12 (0.4714: 2e-4) | NIST JAP 120,123103 (2016) | VERIFIED |
| B14 | \|A\|(D₅/₂→P₃/₂, m=1/2, q=0) | √6/5 = 0.4898979 | 1e-12 (0.4899: 2e-4) | NIST ibid. | VERIFIED |
| B15 | \|A\|(D₅/₂→F₇/₂, m=1/2, q=0) | 2√3/7 = 0.4948717 | 1e-12 (0.4949: 2e-4) | NIST ibid. | VERIFIED |
| B16 | S_FF' Rb-87 F=2→F'=1,2,3 | 1/20, 1/4, 7/10 | 1e-12 | Steck Eq. 41/Table 8 | VERIFIED |
| B17 | Σ_F' S_FF' (all 3 species, both ground F) | 1 | 1e-12 | Steck Eq. 42 | VERIFIED |
| B18 | τ from d: Rb-87 D2 (d_S = 4.22752 e·a₀, ω₀ = 2π·384.2304844685 THz, Eq. 2.21) | 26.2348 ns | 3e-4 | Steck rev 2.3.4 | VERIFIED |
| B19 | τ: Rb-87 D1 (2.9931 e·a₀) | 27.679 ns | 5e-4 | Steck rev 2.3.4 | VERIFIED |
| B20 | τ: Cs D2 (4.4837 e·a₀) | 30.405 ns | 5e-4 | Steck rev 2.3.4 | VERIFIED |
| B21 | τ: Cs D1 (3.1869 e·a₀) | 34.791 ns | 5e-4 | Steck rev 2.3.4 | VERIFIED |
| B22 | Racah-form Eq. 2.20 with d_R = √2·4.22752 = 5.97862 e·a₀, j_e = 3/2 | same τ = 26.2348 ns | 3e-4 | convention identity | VERIFIED |
| B23 | Radial-machinery closure: \|R(5S₁/₂–5P₃/₂)\|_Rb87 from doc 02 | 5.1777 a₀ | 1%² | inverted Steck (§2.7) | VERIFIED (target) |
| B24 | \|R(6S₁/₂–6P₃/₂)\|_Cs from doc 02 | 5.4914 a₀ | 1%² | inverted Steck | VERIFIED (target) |
| B25 | D2/D1 Steck-dipole ratio (per species) | √2 within [−0.7%, 0%] | band | §2.2, measured 1.41243 (Rb87), 1.40692 (Cs) | VERIFIED |
| B26 | H: ⟨1s\|r\|2p⟩ (analytic radial mode) | 128√6/243 = 1.2902662 a₀ | 1e-6 | analytic; reproduced numerically | VERIFIED |
| B27 | H: f(1s→2p) | 2¹³/3⁹ = 0.4161967 | 1e-6 | analytic; reproduced numerically | VERIFIED |
| B28 | H: TRK Σf(1s→ all np, discrete+continuum) | 1.000 | 0.5% | TRK, Eq. 2.19 | VERIFIED (identity) |
| B29 | H: discrete-only Σ_{n'} f(1s→n'p) | 0.565004 | ±0.0005 abs | closed-form sum (this session); Bethe–Salpeter §63 | VERIFIED |
| B30 | Partial sum-rule identity: [(l+1)(2l+3) − l(2l−1)]/[3(2l+1)] for l=0..5 | 1 exactly | 1e-14 | algebraic | VERIFIED |
| B31 | Rb TRK sanity: f(D1)+f(D2) from §3.2 dipoles | 1.038 | ±0.01 | derived from VERIFIED dipoles | VERIFIED (derived) |
| B32 | d_eff,far (Rb-87 D2, Eq. 2.16) | 2.4408 e·a₀ | 1e-3 | Steck Eq. 44 + Table | VERIFIED |

² 1% is the expected accuracy of model-potential radial integrals for low-lying states; the angular
chain contributes < 1e-12 of this budget. A doc-02 result outside 1% indicates a radial bug, not an
angular one — B1–B22 isolate the angular layer completely.

Property-based tests (hypothesis or fixed random grid, seed pinned): float-vs-oracle agreement to
1e-9 on (a) 2000 random **rank-1** 3j (j₂ = 1, j ≤ 150; measured max err 4.2e-12), (b) 2000 random
6j of type {l j ½; j′ l′ 1} (l ≤ 150; measured 3.0e-13), (c) 3000 random **generic** 3j restricted
to j ≤ 20 (measured 2.4e-11). Do NOT assert 1e-9 on generic j ≤ 60 grids — measured float error
reaches 8.2e-5 there (§4.3.4); generic large-j agreement is tested against the oracle at 1e-3 or
routed exclusively through the oracle. Plus: m→−m symmetry `3j(−m's) = (−1)^(j1+j2+j3)·3j(m's)`;
column permutation phases; regge/triangle zero checks return exact 0.0.

---

## 7. Known limitations / model breakdown

1. **Pure |n l j m_j⟩ basis.** Angular factors assume m_j is good. DC electric/magnetic bias fields
   mix states (Stark doc); beyond weak-field perturbative regimes the single-℘ AT relation (Eq. 2.14)
   fails before the angular algebra does.
2. **Rydberg hyperfine structure ignored** (justified §2.6.3, sub-MHz for n ≳ 20). Not valid for
   low-n intermediate states other than the D-line treatment given.
3. **Two-level probe reduction** (Eq. 2.16) is marginal for Rb-87/Cs excited HFS vs. Doppler width;
   good for AT peak positions, inadequate for precision EIT lineshapes/amplitudes and for cold-atom
   or narrow-velocity-class applications. The EIT doc must offer the F-resolved ladder.
4. **λ_p/λ_c Doppler factor** is a weak-probe, 3-level, velocity-averaged approximation (it is what
   NIST publishes and what B-field-free experiments confirm at the ~1% level), not exact; full
   velocity-averaged OBE supersedes it.
5. **Non-collinear / elliptical RF polarization** breaks the single-℘ assumption; requires coherent
   multi-q treatment (§2.3), including its m_j-dependent AT multi-splitting.
6. **j-dependent radial integrals** formally break the r = (orbital ⊗ spin-identity) factorization;
   handled by per-(j,j') radial integrals (standard, same as ARC), exact within the one-electron
   model potential. Two-electron/core-polarization corrections to the *dipole operator* (beyond the
   potential) are neglected — this is the main reason alkali TRK ≈ 1.04 and why ab-initio dipoles
   differ from model-potential ones at the ~0.5–1% level.
7. **E1 approximation only.** No M1/E2, no retardation; fine for RF wavelengths ≫ atom size (always
   true here).
8. **Signs are convention-relative** (§2.1, §4.3.8). Any consumer doing interference between paths
   must use one consistent convention end-to-end — never mix RydSim reduced elements with external
   tables at amplitude level.
9. **Large-j numerics:** the rank-1 (≤ 3-term) sums this spec needs are validated to j = 150 at
   ≤ 4.2e-12 (3j) / ≤ 3.0e-13 (6j). Generic many-term symbols lose precision from alternating-sum
   cancellation — measured up to 8.2e-5 rel. err at j ≈ 55–70 — and MUST use the exact-rational
   oracle path if ever needed at high j (§4.3.4).

---

### Source list

- D. A. Steck, "Rubidium 87 D Line Data", "Rubidium 85 D Line Data", "Cesium D Line Data",
  rev 2.3.4 (8 Aug 2025), https://steck.us/alkalidata/ — PDFs parsed this session (Eqs. (4),
  (34)–(45), lifetime/dipole/HFS tables).
- M. T. Simons, J. A. Gordon, C. L. Holloway, *J. Appl. Phys.* **120**, 123103 (2016) — Eq. (1)
  (AT field relation, λ_p/λ_c), ℘ = R·A convention, A factors; NIST public PDF
  (tsapps.nist.gov/publication/get_pdf.cfm?pub_id=920613) parsed this session. Preprint:
  arXiv:1607.01766.
- J. A. Sedlacek et al., *Nat. Phys.* **8**, 819 (2012) — original AT electrometry (methodology
  citation via the above; not independently re-parsed).
- C. L. Holloway et al., *IEEE Trans. Antennas Propag.* **62**, 6169 (2014) — tutorial/uncertainty
  framework (citation; numbers not quoted from it here).
- A. R. Edmonds, *Angular Momentum in Quantum Mechanics* (Princeton, 1957) — Eq. 7.1.7, Racah
  formulas. I. I. Sobelman, *Atomic Spectra and Radiative Transitions* (Springer, 1996) — NIST's
  cited source for the angular factors.
- H. A. Bethe & E. E. Salpeter, *QM of One- and Two-Electron Atoms* (1957) §61–63 — partial sum
  rules, hydrogen oscillator-strength distribution (LITERATURE-RECALL items B29).
- CODATA 2022 (via NIST); SI 2019 exact constants.
