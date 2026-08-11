# 10 — Optically Thick Propagation of the Probe and Coupling Through a Vapor Cell

**RydSim physics specification, document 10. Status: NORMATIVE for `rydsim.propagate`.**
Subordinate to `docs/spec/00-conventions.md` (20 locks, rulings R-1…R-28) and
`docs/spec/00-integrity-audit.md` (§3 refusal list) in every particular. Where this document
believes an existing lock, ruling or refusal must be **amended**, it says so in §3 and states the
amendment as a proposal for spec 00's owner to adopt — it never diverges silently.

**Provenance of this document.** It is an *adjudication* of four independently authored draft
sections (`docs/spec/drafts/10-*.md`, ~301 kB, authored 2026-08-11 with network available). Every
substantive disagreement between those drafts is recorded and ruled on in §2. Every equation and
number below is attributed to the draft that produced it and carries that draft's confidence tag,
re-graded here where the adjudication found the tag too generous. Confidence vocabulary:

| Tag | Meaning |
|---|---|
| **VERIFIED** | primary source fetched and quoted during draft authoring (2026-08-11), or an exact algebraic identity |
| **VERIFIED-ARC** | matches a reputable secondary implementation; primary unfetched |
| **VERIFIED-COMPUTED** | reproduced numerically during draft authoring from VERIFIED inputs, script named |
| **SELF-MEASURED** | produced by running the shipped RydSim tree during draft authoring, harness **not yet in repo** — an unreproducible assertion until ported (integrity-audit R4). **Never gates a release.** |
| **DERIVED-IN-SPEC** | algebra performed in a draft; the falsifying numerical check is stated |
| **LITERATURE-RECALL** | standard result recalled, not re-checked |
| **UNVERIFIED** | memory only; must be self-checked as described |
| **MISSING** | declared absent; the resolution path is named |

House rule in force: *reproducible or it didn't happen.* A spec that admits a gap is worth more
than one that fills it with plausible text.

---

## 1. Scope

### 1.1 What this document owns

The **coupled steady-state propagation problem** for a CW Rydberg-EIT vapor cell: the reduction of
Maxwell's equations to the paraxial slowly-varying-envelope form for probe and coupling; the exact
conditions under which the shipped thin-medium answer `T = exp(−k_p Im χ L)` is not an
approximation at all; coupling depletion and its photon-number bound; the counter-propagating
two-point boundary-value problem; how the velocity average and the spatial integration compose;
probe saturation with depth; radiation trapping; the composition of radial and longitudinal
averaging; and the density ceiling above which the medium is no longer the medium the model
describes.

Ownership is **not moved.** Spec 00 §4 assigns "propagation" to **05 / `rydsim.vapor`**. This
document is the theory and numerics that spec 05 §2.f sketches in six lines. The implementing code
module is `rydsim.propagate` (ruling **R10-2**), owned by spec 05 jointly with this document.

### 1.2 What this document does not own

Susceptibility itself (**06**), velocity grids and quadrature convergence (**05 §2.d / 06 §4.4**,
ruling R-2), transit rates (**05 §2.e**, ruling R-3), density and vapor pressure (**05 §2.a**),
beam geometry and radial averaging (**05 §2.g**), cell-wall screening (**05 §2.h**, ruling R-7),
the superheterodyne readout chain (**08**), and RF field structure inside the cell (spec 06 §2.8
items 6–7 — a cell/EM module that does not exist).

Out of scope entirely: pulsed/transient propagation, dark-state polaritons, slow light and storage,
four-wave mixing and backward-generated fields, optical bistability, Rydberg–Rydberg interactions
and ionization, buffer-gas cells.

### 1.3 New empirical content

**None.** This document introduces no new fitted or measured constant. Every number is a
fundamental constant from `scipy.constants` (lock #15), a quantity owned by specs 01–06, an
algebraic identity, or a numerical measurement whose harness is named and whose porting is a
release blocker. A propagation spec that needs new fitted parameters is a propagation spec that is
hiding a model.

---

### 1.4 THE STRATEGIC ANSWER — does propagation change the trade STRUCTURE or only the SCALE?

This is stated first because it reorders the work. The answer has three parts and they are not the
same answer.

**(a) In the strict weak-probe (linear-response) limit, propagation changes NOTHING — not the
scale, not the structure. It is an exact no-op.**

When `χ` is independent of `Ω_p`, `Ω_c` is undepleted and the medium is uniform, the propagation
ODE has constant coefficients and integrates in closed form to `T = exp(−k_p Im χ L)` — bit-for-bit
the shipped `rydsim.eit.transmission()`, at **every** optical depth, with no `OD ≪ 1` anywhere in
the hypothesis. This is not a RydSim claim: it is Eq. (36) of Finkelstein, Bali, Firstenberg &
Novikova, *New J. Phys.* **25**, 035001 (2023), fetched and quoted (VERIFIED). It was measured
independently by two drafts: relative difference **2.8×10⁻¹⁴** at OD = 10.0558 (RK4, 20 001 steps)
and **1.36×10⁻¹⁵** at 1000 RK4 steps on a synthetic χ. Consequently a thick-cell solver that keeps
the weak-probe `χ` is a re-derivation of `exp(−OD)`.

**(b) At finite `Ω_p`, propagation changes the STRUCTURE of the `(N, L, Ω_p)` trade.** Three
mechanisms, all sourced:

* `OD` stops being `∝ N·L`. The front of the cell bleaches and the back does not, so `OD_eff` is
  sublinear in both `L` and `N`. **Any trade exponent measured against `L` or `N` at finite `Ω_p`
  in the thin-cell model is wrong by a z-dependent factor.**
* A probe-intensity optimum exists that the weak-probe model **cannot produce at all** — at weak
  probe `χ` has no `I_p` dependence, so the peak-height-vs-`I_p` maximum does not exist in the
  model. Su, Liou, Lin & Chen, *Opt. Express* **30**, 1499 (2022) (VERIFIED, fetched):
  *"For any given temperature, the peak height … reaches the maximum value with the optimum `I_p`.
  The optimum `I_p` becomes stronger with a higher vapor temperature."* A two-dimensional optimum
  whose location moves along one axis as the other is varied is a structural feature.
* Direct published evidence that the `N`-scaling itself moves: Su's fitted optical depth rises by
  **11.90×** from 27 °C to 65 °C where the spec-05 weak-probe model predicts **30.19×** — a ratio
  that is independent of cell length and isotopic enrichment, i.e. a length-free, enrichment-free
  statement of the discrepancy (benchmark 10/B-45).

**(c) The `Ω_c` trade and every frequency observable are untouched.** `Ω_c` does not appear in the
probe propagation equation, and coupling depletion is bounded at `2.7×10⁻⁵` in realistic
configurations (§4.5). Frequency observables are invariant *exactly*: since
`T = exp(−k_p L Im χ)` and `exp` is strictly monotone, `dT/dΔ_p = 0 ⟺ d(Im χ)/dΔ_p = 0`, so AT peak
positions, the peak separation, the `λ_c/λ_p` compression factor and the inverted field
`E = ħΩ_RF/℘` are propagation-invariant to machine precision.

#### 1.4.1 Consequence for `findings/d3-trade-law-v2`

**Ruling R10-24. `findings/d3-trade-law-v2-6f7f20848dca40a1` is NOT retracted by this document, and
is NOT vindicated by it either. It is flagged REGIME-LIMITED pending benchmark 10/B-30.**

The reasoning, stated plainly:

1. D3-v2 measures an **`Ω_c`** trade (NEF U-shaped in `Ω_c`; `α < 0.15` above the optimum) at
   `L = 2 mm`, `OD ≪ 5`, weak probe. Per (c), propagation does not enter that trade. On the physics,
   the finding is **expected to survive**.
2. "Expected" is not "shown". The declared threat is *not* propagation — it is the weak-probe gate.
   `LadderConfig` ships `omega_probe = 2π·100 kHz` against `omega_coupling = 2π·5 MHz`, i.e.
   `Ω_p/min(Γ_e, Ω_c) = 0.02` — **twice** integrity-audit refusal #21's ceiling of 0.01, and that
   refusal **is not implemented anywhere in `src/rydsim`** (verified by two independent drafts).
   D3-v2's runs may therefore sit outside the linear-response regime whose exactness clause (a)
   depends on. That is a defect in the *thin-cell* path, uncovered by writing this spec.
3. **Decision rule (normative, benchmark 10/B-30).** Re-run the D3 `Ω_c` sweep through the
   propagation solver at `Ω_p/Γ_e ∈ {→0, 0.1, 0.5, 1.0}`. At `Ω_p → 0` the exponent must be
   **bit-identical** to the thin-cell value (a consequence of (a); a discrepancy is a solver bug).
   Then: `|Δα| < 0.02` ⇒ **scale only**, the finding stands unqualified and the no-lab thesis is
   vindicated for this trade. `|Δα| > 0.05` ⇒ **structural**, the finding is retracted a second
   time and re-derived. `0.02 ≤ |Δα| ≤ 0.05` ⇒ the finding ships with a declared regime bound.
4. Until 10/B-30 runs, every citation of D3-v2 must carry: *"measured in the thin-cell weak-probe
   regime; the `(N, L, Ω_p)` trade is known to be regime-limited (spec 10 §1.4b) and the
   configuration's compliance with the weak-probe gate is unverified (spec 10 R10-9)."*

**The honest summary the program should quote:** *propagation changes the absolute scale and leaves
the `Ω_c` trade structure and all frequency observables intact; it changes the trade structure in
density, length and probe power, where a previously published exponent must not be extrapolated.
The no-lab thesis survives for the trade RydSim actually published, and fails for the trade it has
not yet measured.*

### 1.5 The scope statement this forces

Thick-cell propagation makes the published corpus *computable in principle*. It makes it
*physically defensible* only as the **strong-probe, z-coupled solve**. The weak-probe z-solver
closes none of the gap. Measured against RydSim's own weak-probe gate, the published corpus runs
**200–600× outside it** (Sedlacek 2012: `s₀ = 1.96`, 300×; Jing 2020: `s₀ = 2.37`, 588×; Su 2022:
`s₀ = 48.9`; multi-dress 2026: `s₀ = 16.0`, 283×). Every flagship vapour-cell electrometry
experiment runs at OD of order 1–17 **and** with a saturating probe **simultaneously**; the two
failures are not independent and cannot be patched one at a time.

**Therefore the deliverable of spec 10 is the strong-probe z-coupled solver, and "thick cell" alone
is a sub-case of it that the engine already answers exactly.** (Ruling R10-1.)

---

## 2. Ruling register — the adjudication

Each entry: **what conflicts, where, and the binding decision with its reason.** Implementations
follow the ruling, not the original draft text. Draft labels: **D1** = `…-1-scope-b`,
**D2** = `…-theory-go`, **D3** = `…-4-numeric`, **D4** = `…-6-validat`.

**R10-1. Is the deliverable "thick cell" or "strong probe"?**
D1 §2.2 argues the weak-probe z-solver is provably a no-op and recommends re-scoping. D2, D3 and
D4 all independently derive the same reduction theorem but title and organise around "thick
propagation". **Ruling: D1's re-scoping is adopted.** The document keeps its title (spec numbering
stability) but the deliverable is the **strong-probe z-coupled solve**; the weak-probe branch is a
closed form, not a computation, and must be *stamped* as such (R10-13). Reason: all four drafts
agree the physics; only D1 drew the work-ordering consequence, and it is correct.

**R10-2. Module name and ownership.**
D1/D3 name the module `rydsim.propagate`; D2/D3/D4 note spec 00 §4 assigns "propagation" to
`rydsim.vapor`. **Ruling: the code module is `rydsim.propagate`; ownership stays with spec 05.**
Spec 00 §4's "propagation" row is amended to read `05 + 10 / rydsim.propagate` (§3 amendment A-4).
Reason: `rydsim.vapor` is not a shipped single file (spec 00 §7 shipped-tree note), and a
seven-hundred-line BVP solver does not belong inside a vapour-pressure module. This is a naming
decision, not an ownership move.

**R10-3. Is the `OD → 0` reduction test sufficient to validate the module?**
D1 (D-4) says the reduction test is **degenerate** — satisfied by construction at every OD — and
must not be shipped as evidence the solver works. D2 (10/P-1) and D3 (R-THIN) call it
release-gating and "the single most important check". D4 §6.1 resolves it structurally by splitting
**Structural (S)** from **Discriminating (D)** rows. **Ruling: all four are right about different
things, and D4's taxonomy is binding.** The reduction test is:
* **GATING and necessary** — it is a *convention lock* that catches sign errors, `k/2` vs `k` vs
  `2k` factor errors, `k_p` vs `k_c`, and vec-ordering/plumbing bugs. A solver that fails it must
  not be used at **any** optical depth.
* **Not sufficient, and never to be cited as physics validation.** It cannot distinguish the new
  solver from `eit.transmission()`.
* **Stated in the strong form (D3's R-THIN), not the weak one:** the agreement is required at
  **every** OD in {1e-6, 1e-3, 0.1, 1, 5, 14.216, 20, 50, 500}, not merely as `OD → 0`, because in
  that branch the integrand is a constant and every consistent quadrature integrates a constant
  exactly. There is no discretisation error to hide behind.
* **The discriminating reduction is D4's P-2** — the `O(x²)` coefficient `−s/(2(1+s)³)` at finite
  saturation — which a solver that silently calls the thin path below a threshold **fails**.
Benchmarks 10/B-01 (structural, GATING), 10/B-05 (discriminating, GATING).

**R10-4. Tolerance on the reduction test.**
D2/D3 require `≤ 1e-12` relative in `ln T`; D4 requires `≤ 1e-6` in `T` under the spec-05 step rule.
**Ruling: both, for different branches.** Branch **S0** (closed form, χ z-constant) → `≤ 1e-12`
relative in `ln T`, because the quadrature is exact. Branch **S/N** (z-stepped, step rule
`|ΔOD| ≤ 0.05`) → `≤ 1e-6` relative in `ln T`. Reason: a single tolerance either lets a stepped
integrator off (1e-12 is unattainable at 20 steps) or falsely fails the exact branch.

**R10-5. Where does the validity boundary sit? — the four-way OD conflict.**
Four numbers are in the tree simultaneously: integrity-audit refusal #18 (`OD > 0.1` →
`ThickCellError`); spec 05 §2.f (`OD_peak ≤ 0.1` for the single exponential);
`LadderConfig.max_optical_depth = 5.0` (shipped); and D1's proposed `OD ≤ 100`. D2 and D3
independently argue OD is the **wrong variable** entirely. D4 proposes a density ceiling plus a
convergence-based refusal.
**Ruling: OD is retired as a physics fence.** It is a derived quantity that conflates density,
length and lineshape, and Theorem 10.R shows the underlying physics is exact at any OD. The
replacement is a set of fences on the *causes*, each computed at runtime and reported:
`GATE-P` (probe saturation / linear response), `GATE-C` (coupling depletion),
`GATE-T` (transport locality), `GATE-L` (self-lensing), `GATE-D` (density/collisional validity),
`GATE-R` (representability: transmitted power vs the caller's stated detector floor), and
`GATE-G` (beam geometry, `2z_R/L`). Full definitions in §9. `OD > 0.1` survives **only as a
warning-level flag**, so nothing currently caught becomes uncaught. `max_optical_depth` is
retained as a *user-settable numerical-conditioning* knob, explicitly relabelled as such, and is
**not** a physics gate (§3 amendment A-1). Reason: three of the four drafts converged on this
independently, and the fourth (D1) supplies the density fence that the other three lacked.

**R10-6. Where does the DENSITY ceiling sit? 1×10¹² or 3.5×10¹² cm⁻³?**
D1 §4.1 sets `N_max = 3.5×10¹² cm⁻³` (equivalently OD = 100 for a 5 cm Rb cell) on three anchors:
charge-induced bistability observed by Weller *et al.*, *Phys. Rev. A* **94**, 063820 (2016) at
Rb total ≈ **2.5×10¹²** cm⁻³; the UNVERIFIED Fermi-broadening surrogate becoming the dominant
Rydberg linewidth; and spec 05 §7.2's D2 self-broadening coefficient being MISSING above
**1×10¹²** cm⁻³. D4 §6.10 item 1 sets the ceiling at **1×10¹²** cm⁻³ on the last of those anchors.
**Ruling: D4's `1×10¹² cm⁻³` is binding; D1's `3.5×10¹²` is rejected.** Reason, and it is D1's own
arithmetic: **D1 set its ceiling above two of its own three anchors.** A ceiling that sits 1.4×
above the density at which the breakdown it fences against has been *observed*, and 3.5× above the
density at which its own broadening coefficient leaves its validated band, is not a ceiling. The
binding number is the **lowest** sourced anchor. Consequences: for a 5 cm natural-Rb cell the
usable window is `34 °C ≤ T ≤ ~75 °C` spanning `OD ≈ 1 → ~28`; Su 2022's 65 °C row
(`N = 5.0×10¹¹ cm⁻³`) is **inside** the fence; D1's `OD = 100` band is **outside** it. The fence
is expressed in **density**, never in OD (R10-5). Raising it requires a spec edit citing the
Durham-grade D2 coefficient (MISSING item M-4) — lock #20.

**R10-7. Does coupling depletion matter?**
D1 (F-2, R-P2) treats it as a live term the solver must carry and refuse above 1 %. D2 §2.4 proves
it is `O(Ω_p²)` — *identically zero in strict linear response* — and bounds it by Manley–Rowe
photon counting at `2.71×10⁻⁵` for 1 µW probe / 30 mW coupling, "negligible by 3–4 orders of
magnitude in every realistic configuration". D3 §4.3.4 makes it a computed gate with a mandatory
one-sweep certificate. D4 (P-17) says measure it, never assume it.
**Ruling: all four compose, and D2's structural statement is elevated to normative.**
*A model that retains coupling depletion must also retain probe saturation, and vice versa — they
are the same `O(Ω_p²)` correction. Any implementation that adds one without the other is
inconsistent at the order it claims to work to.* The implementation is D3's: compute the a-priori
bound `|δOD| ≤ OD_p·s_c·OD_c/2`, take the forward-only branch only when it passes `ε_gate = 1e-5`,
**and always take exactly one relaxation sweep as a measured certificate** — a bound that is ever
violated is a wrong bound. D1's 1 % figure is retained as the refusal threshold (spec 05 §2.f
already requires it). D2's "genuinely omissible" is recorded as a **measured expectation, never an
assumption**.

**R10-8. Velocity grid: freeze it, or re-converge it at every z?**
Direct conflict. D3's Rule **V-FREEZE** requires the velocity node set be built once for the widest
dressed structure over the whole parameter box and then **frozen**, on the measured grounds that a
grid rebuilt per parameter value jitters the integrand and floors an otherwise spectrally
convergent interpolant at **4.1×10⁻⁷** instead of **2.3×10⁻¹¹**. D4 §6.9 item 1 requires that
"every z level re-solves the velocity integral … at every z, not only at z = 0 — a solver that
converges the grid once at the entrance and reuses the node set is under-resolved at the exit,
where the EIT feature has narrowed by √OD".
**Ruling: freeze the node set; re-evaluate the integral at every z. D4's stated *reason* is
rejected.** The `1/√OD` narrowing (D4 A6, verified over six octaves) is a narrowing of the
*transmitted* window in **detuning**, produced by the exponential acting on a fixed `Im χ(δ)` — it
is **not** a narrowing of the velocity-space structure of `χ` itself, whose width is set by
`γ_min/|k_c − k_p|` and is z-independent at fixed `Ω_c`. So there is nothing at the exit that the
entrance grid fails to resolve. What *is* binding from D4 is the sizing rule: **the frozen grid
must be sized for the narrowest feature anywhere in the parameter box** (smallest `γ_min`, largest
`Ω_c`, largest `Ω_RF`), not at the entrance values. Convergence is demonstrated once, on the frozen
grid, by node-disjoint refinement over the whole box. This preserves R-2 in full (the grid is still
the uniform/composite resonance-refined grid with halving convergence; Gauss–Hermite remains
forbidden) and adds the freeze as spec 00 amendment A-3.

**R10-9. The probe gate: `Ω_p < 0.01·min(Γ_e, Ω_c)` vs `s_sat ≤ 0.01` vs `Ω_p ≲ Γ_e/7`.**
Integrity-audit refusal #21 and spec 06 §4.6 give `Ω_p < 0.01·min(Γ_e, |Ω_c|)`. Spec 05 §2.f gives
`I_p < 0.01 I_sat` and restates it as `Ω_p ≲ Γ_e/7`. D2's GATE-P gives `s_sat = 2Ω_p²/Γ_e² ≤ 0.01`,
i.e. `Ω_p ≤ Γ_e/14.14`. D4 uses `s₀ = 2Ω_p²/Γ_e²` throughout.
**Ruling: these are TWO physically distinct conditions and both are enforced; neither replaces the
other. No fence is loosened.**
* **(i) Linear-response gate (the χ linearisation), audit #21, UNCHANGED:**
  `Ω_p < 0.01·min(Γ_e, |Ω_c|)`. This gates the *analytic weak-probe path* and is the condition
  under which Theorem 10.R's hypothesis (i) holds. D2's `s_sat ≤ 0.01` corresponds to
  `Ω_p = 0.0707 Γ_e`, which is **7.07× looser** than audit #21; adopting it as a replacement would
  loosen a fence, and this document does not loosen fences. **Rejected as a replacement.**
* **(ii) Ancillary-neglect gate (radiation trapping, coupling depletion), D2's GATE-P, ADOPTED as a
  separate gate on *all* branches:** `s_sat = I_p/I_sat = 2Ω_p²/Γ_e² ≤ 0.01`. It bounds radiation
  trapping to ≤ 2.1 % of `γ_gr` and coupling depletion to `1.1×10⁻⁵`. It does **not** license the
  linear-response χ.
* **(iii) Spec 05 §2.f's `Γ_e/7` is a numerical error** (`Γ_e/7` ⇒ `s_sat = 2/49 = 0.041`, not
  0.01) and is corrected to `Γ_e/14.14` (§3 amendment A-2). The sourced intensity form
  `I_p < 0.01 I_sat` is kept; the Rabi restatement moves.
* **Convention lock on (ii):** `s_sat = 2Ω_p²/Γ_e²` holds only when `Ω_p` and `I_sat` are built
  from the **same** dipole (spec 00 §2, `I_sat` row: "must be paired with the matching d"). The
  solver must assert the identity from its own two-level steady state at `Ω_c = 0`, never type it.

**R10-10. The shipped OD estimator is wrong — by how much?**
D1 (D-1, HIGH) measures the shipped `experiment.superhet_transfer` OD chain as **2.40× (Rb-87) to
2.67× (Cs)** too large: `eit.dipole_from_linewidth` returns the **cycling** dipole
`⟨J‖er‖J'⟩/√2` where spec 00 §6 gap 7 mandates `d_eff,far = ⟨J‖er‖J'⟩/√3` (factor 1.5 in `d²`),
and no ground-hyperfine fraction `p_F` is applied (a further 1.60–1.78×). Its worked fixture:
nat-Rb, 75 mm, 25 °C, `Ω_c = 0`, Rb-87 F=2 → shipped **1.3874**, D1-corrected **0.578**, spec 05
B9b **0.481**. D4 independently re-implemented spec 05 §2.f from scratch and reproduced **every**
B9 row to < 0.15 % (0.4805 for that fixture). D2's 10/P-10 reports the shipped engine's
`α_p/N = 5.129×10⁻¹⁵ m²` against an analytic `σ_D,peak = 5.089×10⁻¹⁵ m²` — 0.8 % — and calls it "an
*absolute* cross-check of the susceptibility chain … independent of the spec-06 B-2 sum rule".
**Ruling, three parts:**
1. **The defect is confirmed** — two drafts triangulate it from opposite directions. Shipped
   1.3874 against the independently reproduced normative 0.4805 is a factor **2.89**, not 2.40.
   D1's "corrected" 0.578 is itself still **1.20×** above the normative model, so D1's correction
   is *partial*. Neither draft's factor is adopted as a number; the **benchmark settles it**
   (10/B-35, GATING).
2. **D2's 10/P-10 is re-graded from "independent absolute cross-check" to "internal-consistency
   check of the Doppler-broadening factor".** It is **not** independent of the dipole convention:
   `σ₀ = 3λ²/2π` is the closed *cycling* transition cross-section, the same convention
   `dipole_from_linewidth(degeneracy_factor=1.0)` uses. The 0.8 % agreement confirms the
   Gaussian/Lorentzian peak-height ratio and the `k_p Im χ` plumbing; it is silent on the very
   error D1 found. Dressing it as an independent absolute check would have concealed the defect.
   This re-grade is the single most consequential sourcing correction in this adjudication.
3. **Consequence for the shipped gate.** `max_optical_depth = 5.0` therefore binds at a *true* OD
   between **1.7 and 2.1** — i.e. at or just above the shot-noise-limited optimum `OD* = 2`
   (R10-11). The engine has been refusing at almost exactly the operating point it should be
   recommending. Fixing the estimator is a **prerequisite** to any re-derivation of the ceiling.

**R10-11. The optical-depth optimum.**
D1 (§5.2 Grade C) and D4 (A4/P-7) derive the same result independently: with
`κ_E ∝ P_in e^{−OD}·OD`, `NEF ∝ e^{OD/2}/OD` (shot-noise-limited) ⇒ **`OD* = 2` exactly**
(numeric argmin 1.999996); `NEF ∝ e^{OD}/OD` (detector-NEP-limited) ⇒ **`OD* = 1` exactly**;
`NEF ∝ 1/OD` (RIN-limited) ⇒ **no interior optimum**. **Ruling: adopted verbatim, GATING
(10/B-29).** No conflict; recorded because it is the strongest no-lab-legitimate claim in the
document — the optimum's *location* is an integer independent of every uncalibrated absolute
parameter. D4's penalty ratios (`NEF(2)/NEF(0.1) = 0.1293`, `/NEF(1) = 0.8244`, `/NEF(5) = 0.5578`)
are adopted: the thin-cell regime spec 05 currently mandates is **7.7× worse in sensitivity than
the optimum**.

**R10-12. `T` or `ln T` as the primary return?**
D3 §4.2.1 argues for integrating `u = ln Ω` and returning `ln T`; spec 05 §4.7 guards
`exp(−OD)` at `OD > 700` returning `0.0`. D4 §6.9 item 3 agrees to work in log space.
**Ruling: the solver's primary return is `ln T` (= −OD) and the accumulated phase `φ`; `T` is a
separate accessor that sets `underflow=True` above the measured float64 subnormal onset
`OD = 708.3964185322641`.** Spec 05's `OD > 700 → 0.0` is retained as the **display** rule only.
Reason: an absolute tolerance on `ln T` is a *relative* tolerance on `T` uniformly across the scan
— including where `T ≈ 1` in the wings and where `T ≈ 10⁻⁷` at line centre — and a bare `0.0`
propagating into `κ = dP/dE` is audit CRIT-2 in a new costume.

**R10-13. A no-op must be stamped as a no-op.**
D1's R-P12 requires that when the weak-probe branch is taken the result carry
`path="beer_lambert_exact"`. **Ruling: adopted and generalised.** Every propagation result carries
`branch ∈ {"S0_closed_form", "S_quadrature", "N_nonlinear"}`. Reason: "a no-op dressed as a
computation is the 'plausible but wrong' hazard in its purest form", and a caller who believes the
z-solver did work it did not do will over-trust the number.

**R10-14. Radial × longitudinal ordering.**
D1 §3.3a, D2 §2.9 (pitfall 4.3.5) and D4 D3/P-18 all state that `⟨exp(−OD(r))⟩ ≠ exp(−⟨OD(r)⟩)`
and that the z-propagation must happen **inside** the radial quadrature. **Ruling: adopted;
implemented order is radial-average-of-propagated-shells.** D1's exact identity is adopted as a
GATING unit check: for equal waists the probe-power-weighted mean of `Ω_c²` is exactly
`Ω_c0²/2`, so **on-axis evaluation overstates `Ω_c,eff` by 29.3 % and the EIT width
(`∝ Ω_c²/2γ_ge`) by a factor 2**; general case `1/(1 + (w₀p/w₀c)²)`. Frequency observables are
exempt (`Ω_RF` is radially uniform on the scale of an optical beam by 10²–10³).

**R10-15. Benchmark ID collision.**
D2 uses `10/P-1…P-15`; D4 uses `P-1…P-20`; they are **different benchmarks under the same names**
(e.g. D2's `10/P-7` is the transport-locality error, D4's `P-7` is the shot-noise OD optimum).
D1 uses `S-1…S-12`, D3 uses `10/N-1…N-13`. **Ruling: one namespace, `10/B-nn`.** §8 carries the
full provenance map from each draft ID to the unified ID. Reason: two live benchmark registries
with colliding names is exactly how a green test gets cited for the wrong claim.

**R10-16. Radiation trapping — found by one draft only.**
Only D2 §2.8 treats it. Spec 06 §7.4 asserts "no radiation trapping" without a number.
**Ruling: D2's treatment is adopted in full, and its consequence is escalated.** Trapping adds to
`γ_gr` (not to `Γ_e`: a real emission event occurs at rate `Γ_e` whatever the photon subsequently
does). The model-free bound `γ_trap ≤ α_p Φ_p/N = σ_eff Φ_p` requires no escape-factor theory and
is built from quantities the engine already computes. **The escalation:** D2's own table gives
`γ_trap/γ_gr` = 2.1 % at `I_p = 0.01 I_sat`, 51 % at 10 µW and **513 % at 100 µW** (Rb-87, 300 K,
`w₀ = 1 mm`). The published corpus runs at `I_p/I_sat ≈ 0.16` (Mohapatra) to `≈ 17.6` (Su
0.044 W/cm²), i.e. **radiation trapping is unbounded-in-practice over the entire regime this
module exists to reach**, and only a *bound* is available because the Holstein escape factor is
MISSING. This is a second unbounded systematic alongside the RF internal-field one (R10-17), and
it was invisible to three of the four drafts.

**R10-17. The dominant unmodelled systematic is not optical.**
D1 §3.1: every corpus configuration violates the published cell-geometry criterion `D/λ_rf < 0.1`
(Fan *et al.*, *Phys. Rev. Applied* **4**, 044015 (2015), VERIFIED, fetched) by **2× to 262×**, and
the size of the induced `|E_int/E_inc|` error is **MISSING**. **Ruling: adopted unchanged, and it
binds every field output of this module.** `E = ħΩ_RF/℘` returns the field **at the atoms**, never
the incident field; `EFieldResult` carries `field_reference: Literal["at_atoms"]` and the computed
`D/λ_rf`. Thick-cell propagation does not touch this and must not be described as if it does.

**R10-18. Beam geometry — half the corpus violates the 1-D model's own precondition.**
D1 §3.3b computes `2z_R/L` for every corpus beam: E1 coupling **0.44**, E2 probe **0.17**,
E6 probe **0.44**, E6 coupling **0.22** — four of seven **fail** the collimated criterion. At
`2z_R/L = 0.44` the on-axis intensity at the cell exit is 0.16 of the waist value: a 6× drop in
`I_c`, 2.5× in `Ω_c`, across the cell. **Ruling: adopted as a hard refusal (GATE-G).**
`2z_R/L ≥ 10` → valid; `1 ≤ 2z_R/L < 10` → warn, with the exit/waist intensity ratio attached;
`2z_R/L < 1` → **refuse**. A 1-D z-integration inherits the violation rather than fixing it. The
fix is a paraxial 2-D `(r, z)` split-step solver and it is named as future work, not smuggled in.

**R10-19. Which weak-probe OD does the Jing fixture have?**
D1 §2.3 reports shipped **15.909** → D1-corrected **5.97**. D4 §6.5 C1 computes **4.639** from
Jing's own stated density with the spec 05 §2.f Cs D2 model. The two differ by **29 %** from the
same published inputs. **Ruling: D4's 4.639 is normative; D1's 5.97 is superseded.** Reason: D4's
number comes from a from-scratch re-implementation that reproduced *every* spec-05 B9 row to
< 0.15 %, whereas D1's is a two-factor patch on the shipped estimator that R10-10 shows is still
1.20× high on the Rb fixture. The 29 % residual is recorded as a live discrepancy that benchmark
10/B-35 must close before either number is quoted.

**R10-20. Jing's cell length.**
The ar5iv rendering reads "55-cm-long"; spec 08 §3.2 records 5 cm as VERIFIED from arXiv v1.
D4 adjudicates numerically: at 55 cm the predicted detector power is ≤ 1e-10 µW against a published
~10 µW — excluded by ~11 orders of magnitude. **Ruling: 5 cm stands.** This is a *fixture
adjudication*, not a solver benchmark, and is therefore recorded in §5 rather than in §8 (D4's
P-11b is dropped as a benchmark, R10-23).

**R10-21. `s` and `α` symbol collisions, and one the drafts did not catch.**
D2 flags `s_sat` vs spec 05's radial `s = 2r²/w₀²` vs spec 03's `S_FF'` vs PSD `S_x`; D3 flags
`L` (cell length vs Liouvillian). **Ruling: both adopted into spec 00 §3** (§3 amendment A-5), plus
a third the drafts did not register: **`α` is used in the propagation literature as the symbol for
optical depth itself.** Su *et al.* 2022's "`α` varied from 0.42 to 5.0" is an **OD**, not an
absorption coefficient; spec 00 §3 already lists five meanings for `α` and this is a sixth. Code
names: `s_sat`, `s_radial`, `absorption_coeff_probe`, `optical_depth`. Never bare `s`, never bare
`α`. A transcription of Su's `α` as an absorption coefficient would be wrong by a factor `L`.

**R10-22. Model-problem benchmarks must be labelled as such.**
D4's A2/A3 (the `−s/(2(1+s)³)` coefficient and the Lambert-W implicit law) are derived for a
**homogeneously broadened, single-velocity-class, `Ω_c = 0` saturable absorber** — not for the
RydSim Doppler ladder. **Ruling: they are GATING as *solver* unit tests and are explicitly labelled
MODEL-PROBLEM; they are not physics benchmarks of the ladder and may never be cited as validating
the Rydberg-EIT model.** Their discriminating power is nonetheless the highest in the corpus (a
factor 15 in `T` at `s = 10`, 58.5 %/93.2 % Beer–Lambert error at `s = 1`/`s = 10`) and requires no
literature at all — which is exactly why they are worth having.

**R10-23. Benchmark drops.**
D1's **S-12** (Alcock/Steck Cs density vs Jing's printed `4.89e10 cm⁻³`) is **dropped as a
duplicate** of spec 05 B3e, which already gates it; the cross-check is cited in the fixture table.
D4's **P-11b** (Jing 55 cm exclusion) is **dropped as a benchmark** and becomes a fixture
adjudication note (R10-20) — it tests a reading of a paper, not the solver. No other row is
dropped for being unsourced; every remaining row carries a source or is an explicit identity. Rows
that are structurally specified but numerically unmeasured (D4's P-18 magnitude, D3's 10/N-7b 2-D
interpolant rate) are **retained with the magnitude declared MISSING** and gate only on the
structural inequality, never on a number nobody measured.

**R10-24.** See §1.4.1 — the `d3-trade-law-v2` ruling.

---

## 3. Convention enforcement, and amendments proposed to specs 00 / 05 / 06

### 3.1 Convention checks performed against spec 00, and the fixes applied

| # | Lock / ruling | Check | Result |
|---|---|---|---|
| 1 | #1 SI internal | every equation and constant below | PASS — display units (cm⁻³, °C, mW/cm²) appear only in prose and fixture tables |
| 2 | #2 angular rates | `Ω`, `Γ`, `γ`, `Δ`, `δ` all rad/s; `_hz` only at boundaries | PASS |
| 3 | #3 amplitude not RMS | `I = ε₀c\|Ê\|²/2`; `s_sat = 2Ω_p²/Γ_e²` | PASS, **with a fix**: the `s_sat` identity is valid only when `Ω_p` and `I_sat` use the same dipole. Made explicit (R10-9) and asserted from the solver's own two-level steady state, never typed |
| 4 | #4 `Ω = d·ℰ/ħ`, full Rabi | all four drafts | PASS |
| 5 | #5 `Δ = ω_field − ω_atom`, `(γ − iΔ)` | inherited from spec 06 §2.4 | PASS |
| 6 | §2 `α` row, `T = exp(−k_p Im χ L)` intensity | `dÊ/dz = +i(k/2)χÊ` ⇒ `\|Ê\|² ∝ exp(−k Im χ z)` | PASS — sign and factor audited in §4.1; VERIFIED against Ogden *et al.* PRL **123**, 243604 Eq. (6) |
| 7 | R-2 velocity quadrature | V-FREEZE vs re-converge | **FIXED** by R10-8; Gauss–Hermite remains forbidden, halving convergence retained |
| 8 | R-3 transit prefactor | `γ_t = √(2 ln 2)·v⊥/w₀` used for `γ_min` in `ε_transp` | PASS — D2 correctly uses spec 05's value (2π·39.8 kHz), not spec 06's demoted estimator |
| 9 | R-22 never import published Ω | Sedlacek 2π·6 MHz, Jing 2π·5.7 MHz, Su 2π·30 MHz | PASS — used only to *characterise regimes*; `Ω` recomputed from `(d, ℰ)` at fixture build |
| 10 | R-26 no second `/h` on C₆ | D1 §3.5 vdW estimates | PASS |
| 11 | R-28 species defaults | Cs (Jing) fixture must assert `species_defaults_in_use() == []` | PASS — mandated in §8 execution rules |
| 12 | #12 one-sided PSDs | the `NEF(OD)` scalings of R10-11 | PASS |
| 13 | §3 collision register | `s`, `L`, `α` | **EXTENDED** — see A-5 |

**Deviations found and corrected:**
* **F-1.** Spec 05 §2.f's `Ω_p ≲ Γ_e/7` is arithmetically inconsistent with its own
  `I_p < 0.01 I_sat` (`Γ_e/7 ⇒ s_sat = 0.041`). Corrected to `Γ_e/14.14` (A-2).
* **F-2.** D2's GATE-P (`s_sat ≤ 0.01`) would have loosened integrity-audit refusal #21 by 7.07×.
  **Rejected as a replacement, kept as a separate gate** (R10-9). No fence is loosened by this
  document except by an explicit numbered amendment.
* **F-3.** D2's 10/P-10 was tagged as an *independent* absolute check of the χ chain; it is not
  independent of the dipole convention. Re-graded (R10-10.2).
* **F-4.** D1's density ceiling sat above two of its own three anchors. Corrected (R10-6).
* **F-5.** Spec 00 §2 has no row for the **amplitude** attenuation coefficient `α/2`, which is the
  coefficient that actually appears in the z-ODE. Added (A-6).

### 3.2 Amendments proposed to spec 00 / 05 / 06 (stated, not enacted)

None of these is applied here; spec 00's owner adjudicates. Each states whether it **tightens** or
**loosens**, because a loosening must not be adopted casually.

**A-1 (to spec 00, new ruling R-29 — replaces the OD fence; NET TIGHTENING).**
Integrity-audit refusal #18 (`OD > 0.1 → ThickCellError`), spec 05 §2.f's `OD_peak ≤ 0.1`, and
`LadderConfig.max_optical_depth = 5.0` are three mutually inconsistent numbers for one fence, and
all three fence the wrong variable. Replace with the computed gate set **GATE-P / C / T / L / D /
R / G** (§9). `OD > 0.1` is demoted to a **warning-level flag** (loosening) while six new
computed refusals are added (tightening), including a density ceiling and a representability floor
that the current tree lacks entirely. `max_optical_depth` is retained and **relabelled** as a
user-settable numerical-conditioning knob, not physics. **Must not be adopted before 10/B-01
passes.**

**A-2 (to spec 05 §2.f; EDITORIAL).** `"In Rabi terms Ω_p ≲ Γ_e/7"` → `"Ω_p ≲ Γ_e/14.14"`. The
sourced intensity form `I_p < 0.01 I_sat` is authoritative; the Rabi restatement was wrong by √4.1.

**A-3 (to spec 00 R-2 — extends, does not relax).** Append: *"When the same velocity average is
evaluated repeatedly across a parameter sweep (Ω_c or Ω_RF along z, an LO scan, an interpolation
node set), the grid is built once for the narrowest dressed feature over the whole parameter box
and then held fixed. A grid rebuilt per parameter value satisfies R-2 pointwise but jitters the
parameter dependence; measured, this floors an otherwise spectrally convergent interpolant at
4.1×10⁻⁷ instead of 2.3×10⁻¹¹."*

**A-4 (to spec 00 §4 ownership map; EDITORIAL).** The "propagation" entry reads
`05 + 10 / rydsim.propagate`. Ownership of parameter values stays with spec 05.

**A-5 (to spec 00 §3 collision register; EDITORIAL).** Add `s` (saturation parameter / radial
variable / line strength / PSD → `s_sat`, `s_radial`, `S_FF'`, `S_x`); `L` (cell length /
Liouvillian → `cell_length_m`, `liouvillian`); and extend the `α` row with **"optical depth, in the
propagation literature (Su 2022, Häupl 2025)"** — a sixth meaning and a live transcription trap.

**A-6 (to spec 00 §2 symbol table; EDITORIAL, closes a real gap).** New row:
`α_amp ≡ α/2` [m⁻¹] — *"amplitude attenuation coefficient; the coefficient appearing in
`dΩ/dz = i(k/2)χΩ`. Never interchangeable with the intensity coefficient `α = k_p Im χ` — see
spec 10 §6.3 pitfall P4."*

**A-7 (to spec 06 §7.2 — precision, NOT a relaxation).** Current text: *"Breaks down for optically
thick cells (α ℓ ≳ 1 with the coupling also attenuated)."* The parenthesis is the operative
condition; `αℓ ≳ 1` alone is not sufficient for breakdown. Proposed: *"Exact in strict linear
response with an undepleted, uniform coupling (spec 10 Theorem 10.R), at any αℓ. Breaks down
through probe saturation at the entrance, coupling depletion, medium gradients, atomic transport,
or radiation trapping — each with its own criterion, none of which is 'OD > 1'."*

**A-8 (to spec 05 B9c; TIGHTENING by restatement).** The apparent ⁸⁷Rb F=2 dip position drifts with
temperature as the Doppler width reweights the hyperfine blend: **−2.4231 GHz at 16.5 °C, −2.4241
at 25.0 °C, −2.4267 at 51 °C, −2.4280 at 65 °C** — a 4.9 MHz span against B9c's ±3 MHz tolerance.
Restate B9c **per temperature** (preferred) or widen to ±6 MHz with the drift documented. Grading a
high-OD spectrum against the 25 °C positions at ±3 MHz is a false failure waiting to happen.

**A-9 (to integrity-audit §3, new refusals).** Refusal #18's `ThickCellError` **does not exist
anywhere in `src/rydsim`** and no `I_p > 0.01 I_sat` gate exists on the physics path (measured by
two drafts independently, `grep`/AST level). Either implement item 18 as amended by A-1, or amend
the audit. **Both texts must not be left standing.** This document recommends A-1 plus a shipped
`rydsim.propagate.ThickCellError(IntegrityError)`.

---

## 4. Equations

### 4.0 Conventions inherited (deviations are bugs)

Real probe field `E_p(z,t) = ½ Ê_p(z) e^{i(k_p z − ω_p t)} + c.c.`, `Ê` the **peak amplitude**
(lock #3); `I = ε₀c|Ê|²/2`; `Ω_k = ℘_k Ê_k/ħ` (lock #4); `Δ = ω_field − ω_atom` and `(γ − iΔ)`
denominators (lock #5); `k_p = 2π/λ_p` is the **vacuum** wavenumber — "no extra 2 or 4π"
(spec 05 §2.f, verbatim); `χ` dimensionless with `Im χ > 0 ⇔ absorption`;
`α_p ≡ k_p Im χ_p` is the **intensity** absorption coefficient [m⁻¹]; `OD ≡ ∫₀^L α_p dz`.

**New symbols introduced by this document** (all SI, all with mandated code names):

| Symbol | Meaning | Unit | Code name |
|---|---|---|---|
| `z`, `L` | axial coordinate (probe → +z); cell optical path length | m | `z_m`, `cell_length_m` |
| `s_c ≡ L − z` | distance travelled by the counter-propagating coupling | m | `s_coupling_m` |
| `χ_p`, `χ_c` | susceptibility at the probe / coupling frequency | — | `chi_probe`, `chi_coupling` |
| `α_p`, `α_c` | **intensity** absorption coefficients, `α_k = k_k Im χ_k` | m⁻¹ | `absorption_coeff_probe/_coupling` |
| `α_amp` | **amplitude** attenuation coefficient `= α/2` | m⁻¹ | `alpha_amp` |
| `Φ_k` | photon flux `I_k/(ħω_k)` | m⁻² s⁻¹ | `photon_flux_probe/_coupling` |
| `s_sat` | probe saturation parameter `I_p/I_sat = 2Ω_p²/Γ_e²` | — | `s_sat` (**never bare `s`**) |
| `u_p ≡ ln Ω_p` | complex log-amplitude (the integration variable) | — | `log_rabi_probe` |
| `L_grad ≡ 2/α_p` | field-gradient scale | m | `grad_length_m` |
| `ε_transp` | non-locality parameter, Eq. (10.14) | — | `transport_epsilon` |
| `γ_trap` | extra `g–r` dephasing from reabsorbed resonance photons | rad/s | `gamma_trap` |
| `g_esc` | Holstein escape factor (**MISSING closed form**) | — | `holstein_escape_factor` |
| `Δφ_lens` | accumulated differential (self-lensing) phase | rad | `lensing_phase_rad` |
| `η_Ωc` | tolerated fractional error in `Ω_c` | — | `omega_c_tolerance` |

### 4.1 The probe propagation equation, and the sign audit

From the scalar wave equation with `P̂ = ε₀ χ_p Ê`, paraxial + SVEA:

```
(10.1)   dÊ_p/dz = + i (k_p/2) χ_p(z) Ê_p(z)          [V/m per m]
         equivalently  dΩ_p/dz = + i (k_p/2) χ_p(z) Ω_p(z)      [rad/s per m]
```

**Sign audit (the single most bug-prone line in the module).** With `Im χ_p > 0` (absorption),
(10.1) gives `|Ê_p(z)|² = |Ê_p(0)|² exp(−k_p Im χ_p z)`, i.e. `T = exp(−k_p Im χ_p L)` — exactly
spec 05 §2.f and exactly `rydsim.eit.transmission()` (`src/rydsim/eit.py:212`). A `−i` produces
gain and is caught by 10/B-01.

**Source. VERIFIED** — Ogden, Whittaker, Keaveney, Wrathmall, Adams & Potvliege,
*Quasi-simultons in thermal atomic vapors*, **Phys. Rev. Lett. 123, 243604 (2019)**,
arXiv:1909.07161, **Eq. (6)**, quoted verbatim from the fetched ar5iv full text:
`[∂/∂z + (1/c)∂/∂t] ℰ_α = (ik/2ε₀) 𝒫_α, α = p,c`. Setting `∂/∂t → 0` and `𝒫_α = ε₀χ_αℰ_α`
reproduces (10.1) **including the factor ½ and the `+i`, for both fields, in a thermal alkali
ladder** — a propagation-equation-level match, not an abstract-level citation.

**Assumptions folded into (10.1), each with its fence:**

| # | Assumption | Fence | Status |
|---|---|---|---|
| A1 | SVEA | `α_p ≪ 4k_p` ⇒ `OD ≪ 1.6×10⁹` for 5 cm Rb | never binding — VERIFIED by inspection |
| A2 | `\|χ\| ≪ 1`, no Fresnel reflection at the vapour boundary | `\|χ\| ~ 10⁻⁶` at 25 °C | VERIFIED (spec 05 §2.f) |
| A3 | CW steady state | transit time `w₀/⟨v⊥⟩ = 4.71 µs` ≫ light transit `L/c = 0.17 ns` | LITERATURE-RECALL; ratio computed |
| A4 | scalar field, polarization preserved | absorbed into effective dipoles | declared limitation (spec 06 §7.1) |
| A5 | no diffraction between radial shells | **GATE-G**, `2z_R/L ≥ 10` | **binding — fails for 4 of 7 corpus beams** (R10-18) |
| A6 | local (no atomic transport) response | **GATE-T**, `ε_transp ≤ 0.05` | DERIVED-IN-SPEC, 10/B-13 |

**The four channels — and only four — by which `z` enters `χ_p`:** (1) `Ω_p(z)`, which *vanishes
identically in strict linear response* because the weak-probe `χ_p` is independent of `Ω_p`;
(2) `Ω_c(z)`, coupling depletion, also `O(Ω_p²)`; (3) externally imposed `N(z), T(z)` gradients;
(4) non-local transport. Channel (1) is the load-bearing fact of §4.3.

### 4.2 The coupling propagation equation

The coupling propagates toward `−z` (`E_c ∝ e^{i(−k_c z − ω_c t)}`):

```
(10.2)   dÊ_c/dz = − i (k_c/2) χ_c(z) Ê_c(z)
         in the coupling's own coordinate s_c = L − z:  dÊ_c/ds_c = + i (k_c/2) χ_c Ê_c
```

Co-propagation is recovered by `s_c → z`. **Never write a bare sign**: expose
`geometry ∈ {"counter","co"}`, exactly as spec 06 §4.8(iii) already requires for the Doppler term.

```
(10.3)   χ_c = 2 N ℘_er² σ_re / (ε₀ ħ Ω_c)
```

with `σ_re = ⟨r|σ|e⟩` the rotating-frame coherence on the driven `e–r` transition and `℘_er` its
dipole [C·m]. **DERIVED-IN-SPEC** (D2), by exact analogy with spec 06 §2.4's
`χ_p = 2N℘_ge²σ_eg/(ε₀ħΩ_p)`, with `ρ_re^lab = σ_re e^{−iω_c t}`.
**Flagged risk:** (10.3) carries a conjugation hazard — `σ_re` versus `σ_er*` — that no draft
sourced. The numerical self-check that catches it is 10/B-07: a wrong conjugation flips the sign of
`Im χ_c` and manufactures coupling **gain**, which the Manley–Rowe closure detects exactly.

### 4.3 Theorem 10.R — the reduction theorem, and the falsification anchor

> **Theorem 10.R.** Assume **(i)** strict linear response (`χ_p` independent of `Ω_p`);
> **(ii)** undepleted coupling (`Ω_c(z) = Ω_c` ∀ z); **(iii)** uniform medium (`N`, `T`, all rates
> z-independent); **(iv)** local response (no atomic transport). Then `χ_p(z) ≡ χ_p` is constant,
> (10.1) integrates in closed form, and
>
> ```
> (10.4)   Ê_p(L) = Ê_p(0)·exp( i k_p χ_p L / 2 )
>          T = exp(−k_p Im χ_p L) = exp(−OD),      Δφ = (k_p L/2)·Re χ_p
> ```
>
> **exactly, for every `L` and every optical depth.** There is no `OD ≪ 1` in the hypothesis.

**Source. VERIFIED** — Finkelstein, Bali, Firstenberg & Novikova, *A practical guide to
electromagnetically induced transparency in atomic vapor*, **New J. Phys. 25, 035001 (2023)**,
arXiv:2205.10959, **Eq. (36)**, quoted verbatim from the fetched ar5iv full text:
`E_out(δ) = ∫d²k⊥ E_in(δ, k⃗⊥) e^{i k_z L ∫d²v w(v⃗) χ_p(δ, k⊥, v)}` — the Doppler-averaged
susceptibility appears **once**, multiplied by the **full** length `L`, inside a **single**
exponential.

**Corollary 10.R.1.** The shipped `max_optical_depth = 5.0` refusal is a *numerical-conditioning*
gate, not a physics gate. The docstring at `experiment.py:92–101` already says so and **is right**.
The refusal exists because `T < 0.7 %` makes the transduction slope numerically dead and the
derived NEF diverges to a number that looks like a result (audit CRIT-2) — **not** because
Beer–Lambert fails.

**Corollary 10.R.2 — condition (iv) is not implied by (i)–(iii).** In strict linear response
`σ_eg(z,v) ∝ Ω_p(z)`, so the coherence *does* vary with `z` even under (i)–(iii). The transport
term does not vanish (§4.6). Condition (iv) is a genuine, quantified approximation, not a
corollary. **This corrects a naive reading of "uniform fields ⇒ local response"** and is D2's.

**Falsification. Measured independently by two drafts.** D1: RK4, 20 001 steps, Rb-87, 60 °C,
`L = 5 cm`, `OD = 10.0558` → RK4 `4.2937676186e-05`, closed form `4.2937676186e-05`, relative
difference **2.8×10⁻¹⁴**. D2: `L = 5 cm`, `χ = 10⁻⁶(0.3 + 1.0i)`, `OD = 0.4026` → relative error
`4.04×10⁻⁶` (1 step), `3.48×10⁻¹⁰` (10 steps), `1.36×10⁻¹⁵` (1000 steps); fitted order **4.07**.
A solver that cannot pass 10/B-01 has a sign, factor-of-2 or `k_p`-convention bug and **must not be
used at any optical depth**.

### 4.4 What actually breaks — the unification

Sections 4.5, 4.7 and 4.8 show that coupling depletion, probe saturation and radiation trapping are
**all driven by one quantity, the probe photon flux `Φ_p`**, and are therefore fenced by a single
gate. That unification is the main design result of the theory drafts and is why the fence set of
§9 is small.

### 4.5 Coupling depletion, and the Manley–Rowe bound

**It is `O(Ω_p²)`.** In the weak-probe hierarchy (`σ_gg = 1 + O(Ω_p²)`, `σ_eg, σ_rg = O(Ω_p)`),
the coherence `σ_re` between two *excited* states is second order in `Ω_p`. Hence from (10.3):

```
(10.5)   χ_c = O(Ω_p²)   →   χ_c ≡ 0 identically in strict linear response
```

**This is the structural reason the shipped model is self-consistent** (D2): the very approximation
that makes `χ_p` probe-independent — and Beer–Lambert exact — makes coupling absorption vanish.
**Normative (R10-7):** a model that retains coupling depletion must also retain probe saturation.

*A sharpening of the code's stated rationale.* `experiment.py:95` says the coupling is undepleted
"because the intermediate state is thermally empty". The thermal 5P₃/₂ fraction is
**2.0×10⁻²⁷ / 9.5×10⁻²¹ / 9.6×10⁻¹⁷ at 300 / 400 / 500 K** — true, but not the operative fact. What
matters is the *optically driven* population (`ρ_ee ≈ 1.2×10⁻²` in the resonant velocity class at
`Ω_p/2π = 0.68 MHz`; measured `ρ_ee = 9.4927e-10 / 9.4901e-6 / 9.2464e-4` at
`Ω_p/2π = 1 kHz / 100 kHz / 1 MHz`, clean `Ω_p²` scaling). The correct fence is the photon-flux
bound, not the Boltzmann factor.

**The exact photon-number bound.** In CW steady state, every `g→e` excitation consumes one probe
photon; every net `e→r` transfer consumes one coupling photon; in steady state the net `e→r` flux
equals the loss rate of `|r⟩`. Therefore, exactly:

```
(10.6)   |dΦ_c/dz| = N ρ_rr (Γ_r + γ_t + …)  ≤  |dΦ_p/dz| = α_p Φ_p

(10.7)   |ΔI_c|/I_c ≤ (λ_p/λ_c)·(I_p(0)/I_c)·(1 − T)
(10.8)   |ΔΩ_c|/Ω_c ≤ ½ (λ_p/λ_c)·(I_p(0)/I_c)·(1 − T)          [since Ω ∝ √I]
(10.9)   |ΔΩ_c|/Ω_c ≤ ½ (λ_p/λ_c)(Ω_p/Ω_c)²(℘_er/℘_ge)²(1 − T)
```

**DERIVED-IN-SPEC** (D2; every step an equality except the final one-photon-channel inequality).
No published statement of this bound was found. Its falsification test is 10/B-07.

**The bound is independent of optical depth and of `N`:** `ΔΦ_c ≤ Φ_p(0)(1−T) ≤ Φ_p(0)`. Coupling
depletion is governed by the **photon-flux ratio of the two beams**, not by how thick the cell is.
Evaluated for Rb-87, `λ_p/λ_c = 1.625503`, `w₀ = 1 mm` both beams:

| Configuration | `\|ΔΩ_c/Ω_c\|` bound |
|---|---|
| `P_p = 1 µW`, `P_c = 30 mW`, `T → 0` | **2.71×10⁻⁵** |
| `P_p = 50 µW`, `P_c = 10 mW`, `T = 0.1` | **3.66×10⁻³** |
| `Ω_p = Ω_c` with `℘_er/℘_ge = 4.92×10⁻³` (5P₃/₂→50D) | **1.97×10⁻⁵** |

The Rydberg coupling dipole is ~200× smaller than the D2 probe dipole, so a comparable Rabi
frequency costs ~4×10⁴ times the intensity. **Coupling depletion is small because the coupling beam
is intrinsically intense, not because the medium is thin.** Requiring `|ΔΩ_c|/Ω_c ≤ η_Ωc = 1 %` at
the spec-05 probe gate gives `I_c ≥ ½(λ_p/λ_c) I_p(0)/η_Ωc = 20.35 W/m² = 0.81 I_sat ≈ 32 µW` at
`w₀ = 1 mm` — against published couplings of 10–700 mW. Dividing (10.8) by `s_sat`:
**probe saturation bites ~1000× harder than coupling depletion** (ratio `1.07×10⁻³` at
`P_c = 30 mW`), so enforcing the probe gate automatically enforces the coupling gate.

Regimes where it is **not** negligible, and which the gate must still *compute*: coupling powers
below ~100 µW at mm waists; a second strong field populating `|e⟩`; incoherent `|e⟩` population
from radiation trapping; micro-cells where the coupling is not the intense beam.

### 4.6 Counter-propagation: a two-point BVP, and how often it binds

```
dΩ_p/dz = + i (k_p/2) χ_p[Ω_p, Ω_c] Ω_p ,   Ω_p(0) = Ω_p^in     (probe enters at z = 0)
dΩ_c/dz = − i (k_c/2) χ_c[Ω_p, Ω_c] Ω_c ,   Ω_c(L) = Ω_c^in     (coupling enters at z = L)
```

**Counter-propagation alone does not create the BVP; depletion does.** If `χ_c` is negligible,
`Ω_c(z) ≡ Ω_c^in` is known, (10.2) drops out, and (10.1) marches forward from `z = 0`. The
implementation must **compute** (10.8) and take the IVP branch only when it passes — never assume
it. Structural corroboration for the counter-propagating-cascade → two-point-BVP → shooting chain:
H.-H. Jen, PhD thesis, Georgia Tech (2010), arXiv:1106.2082 (**VERIFIED at the structural level
only** — the counter-propagating fields there are signal/idler, not probe/coupling). CoOMBE, the
reference open-source Maxwell–Bloch suite, restricts to *"one or two laser fields
(co)-propagating"* (VERIFIED from its README): **the counter-propagating case is genuinely
unsupported by standard tooling**, and the schemes of §6.2 are standard numerical practice applied
here, not methods taken from an EIT paper. Declared as such.

**A geometric claim that must be measured, not assumed (DERIVED-IN-SPEC, D2).** The coupling enters
at `z = L`, the probe's *exit* face, so the signal-generating layer sees the least-depleted
coupling; in co-propagation the ordering reverses. **Benchmark 10/B-11 exists to refute this. If it
comes out the other way the paragraph is struck, not softened.**

**Axial atomic boundary conditions are droppable — with the ratio reported.** Atoms with `v_z > 0`
enter at `z = 0`, those with `v_z < 0` at `z = L`. This is subdominant to the transverse transit
channel by the traversal-time ratio `L/σ_v = 295 µs` vs `w₀/⟨v⊥⟩ = 4.71 µs` = **62.7** (Rb-87,
300 K, `w₀ = 1 mm`, `L = 5 cm`), so the spec-06 §2.2 measure-and-replace channel dominates —
**provided that ratio is computed and reported, not assumed** (10/B-21).

### 4.7 Doppler × propagation: they do not commute, and by how much

The exact object is the phase-space density matrix `ϱ(z,v)` obeying a Boltzmann–Bloch transport
equation. **VERIFIED** against Firstenberg, Shuker, Ron & Davidson, *Colloquium: Coherent diffusion
of polaritons in atomic media*, **Rev. Mod. Phys. 85, 941 (2013)**, arXiv:1207.6748, **Eq. (14)**,
fetched and quoted. Specialised to CW steady state, 1-D axial motion, no velocity-changing
collisions:

```
(10.10a)  v ∂_z ϱ(z,v) = 𝓛[Ω_p(z), Ω_c(z); v] ϱ(z,v)
(10.10b)  χ_p(z) = 2N℘²/(ε₀ħΩ_p(z)) ∫dv f(v) ϱ_eg(z,v)
```

(10.10b) is **VERIFIED** against Ogden *et al.* PRL 123, 243604 **Eq. (8)**, fetched:
`𝐏(z,t) = 𝒩 ∫ f(v_z) Tr[𝐝 ρ(z,t;v_z)] dv_z`.

**The local approximation** — used by every RydSim path to date — drops `v ∂_z ϱ`. Because the
propagator generated by `𝓛(v)` depends on `v`, the velocity average `𝒜` and the z-propagation map
`𝒫_L` do **not** commute; the local approximation is precisely the assertion that they do. For the
linear-response case the error is closed-form: writing `v ∂_z σ = M(v)σ + b(v)Ω_p(z)` with
`Ω_p(z) = Ω_p(0)e^{μz}`, `μ = i k_p χ_p/2`, the exact solution is

```
(10.11)   A = −(M − vμ)⁻¹ b Ω_p(0)     vs.    A_local = −M⁻¹ b Ω_p(0)
```

i.e. **ballistic transport shifts every coherence decay rate by `−vμ`**. The imaginary part shifts
the detuning by `v k_p Re χ/2` — a relative correction `Re χ/2 ~ 5×10⁻⁷` to `k_p v`, negligible
always. The real part shifts the decay rate by `−v α_p/2`; that is the whole effect. Define

```
(10.12)   ε_transp(v) ≡ |v| α_p / (2 γ_min) = |v| / (γ_min L_grad) = |v|·OD/(2 γ_min L)
```

with `γ_min` the **slowest** coherence decay rate carrying the observable — for EIT/AT that is
`γ_gr` (transit-dominated, `2π×39.8 kHz`, spec 05 §2.e / R-3), **not** `γ_ge`. Using `γ_ge`
understates `ε_transp` by ~76×.

| OD | `α_p` [m⁻¹] | `L_grad` | `ε_transp` |
|---|---|---|---|
| 1 | 20 | 100 mm | 8.0×10⁻⁴ |
| 10 | 200 | 10 mm | 8.0×10⁻³ |
| 100 | 2000 | 1 mm | **8.0×10⁻²** |

i.e. `ε_transp ≈ 8×10⁻⁴ × OD` for a 5 cm Rb cell at 300 K with `v_c = 20 m/s`; a 200 µm MEMS cell
is ~12× worse. **Normative fence (GATE-T):** the local path may be used when `ε_transp ≤ 0.05`
evaluated at the largest `|v|` carrying ≥ 1 % of the spectral weight; above that flag
`transport_uncontrolled`; above **0.2** refuse. Note this fence and the density fence of R10-6
converge independently on the same neighbourhood — at the density ceiling `ε_transp` is already
flagged.

**A partial-cancellation claim that must be measured, not assumed (DERIVED-IN-SPEC, D2).** The
shift `−vα_p/2` is **odd in `v`** and at line centre the velocity-resolved response is even, so the
first-order term should cancel in the symmetric Maxwell average, leaving `O(ε²)` at the EIT peak
and `O(ε)` on the wings. **Benchmark 10/B-13 exists to falsify it.** If the measured peak error
scales linearly in `ε`, this paragraph is wrong.

**Cost of getting it right:** the exact treatment replaces one scalar ODE by `N_v × N_levels²`
coupled ODEs in `z` (`N_v ~ 10⁴–10⁵` per R-2) **and** turns the atomic problem into its own
two-point BVP (`v>0` at `z=0`, `v<0` at `z=L`) — a `~10⁶`-dimensional BVP in place of a scalar
quadrature. That is why the local approximation is **fenced, not fixed**, at first release, and why
`ε_transp` is a *bound on the error*, not a *correction to the answer*.

### 4.8 Probe saturation with depth — and why "it self-heals" is only half true

Two independent small parameters, with different physics:

```
(10.13a)  one-photon saturation:  s_sat = I_p/I_sat = 2Ω_p²/Γ_e²       (Doppler background)
(10.13b)  dark-state depletion:   ρ_rr ≈ Ω_p²/(Ω_p² + Ω_c²)            (the EIT feature)
(10.14)   combined:  Ω_p(z) ≪ min( Γ_e , Ω_c(z) )   for all z ∈ [0, L]
```

(10.13b) is the ideal dark-state result for `|D⟩ ∝ Ω_c|g⟩ − Ω_p|r⟩` (**LITERATURE-RECALL**,
textbook); it is the physical content of audit refusal #21. `Ω_p ≪ Ω_c` is **VERIFIED** as the
stated linear-response condition of the NJP 2023 guide (fetched: *"this condition applies when
Ωp≪Ωc"*).

**It self-heals — with the caveat that matters.** `Ω_p(z) = Ω_p(0)e^{−α_p z/2}`, so both parameters
decrease monotonically into the cell: the weak-probe approximation is **worst at `z = 0`, best at
`z = L`**, and the entrance condition is *sufficient* for the whole cell given an undepleted
coupling. **But the attenuation is strongly detuning-dependent, and at the EIT resonance the probe
is *transmitted* — that is the entire point of the scheme — so `α_p` is small there and the probe
stays near its entrance value all the way through.** The "it gets better with depth" argument is
therefore **weakest precisely at the operating point** and strongest on the Doppler wings, which
carry no signal. Measured on the shipped engine (Rb-87, `Ω_c/2π = 5 MHz`, 5 cm): EIT-peak OD
**0.50 / 1.87 / 10.15** at 300 / 313 / 333 K against coupling-off line-centre OD
**1.12 / 4.16 / 22.6** — the EIT feature is attenuated about **2.2× less**. Any claim that
saturation is burned off by depth must be evaluated **on the EIT resonance**, not on the
line-centre OD.

### 4.9 Radiation trapping — a bound, not an estimate

Spec 06 §7.4 asserts "no radiation trapping" without a number, and the number is not safe at the
densities this module unlocks. **VERIFIED (abstract level)** — Matsko, Novikova, Scully & Welch,
*Radiation trapping in coherent media*, **Phys. Rev. Lett. 87, 133601 (2001)**, fetched:
*"the effective decay rate of Zeeman coherence … increases significantly with the atomic density …
radiation trapping must be taken into account to fully understand many electromagnetically induced
transparency experiments with optically thick media."* The mechanism transfers to a Rydberg ladder:
a scattered probe photon reabsorbed by an atom in `|D⟩` projects that atom and destroys its `g–r`
coherence — **an addition to `γ_gr`, not a change to `Γ_e`** (a real emission event occurs at rate
`Γ_e` whatever the photon subsequently does, so `γ_ge = Γ_e/2 + …` is untouched).

**How thick the vapour is to its own fluorescence:**

```
(10.15)   σ_D,peak = √(π ln 2) · (Γ_e/2π)/Δν_D · σ₀,     σ₀ = 3λ²/2π,   √(π ln 2) = 1.475665
```

**DERIVED-IN-SPEC** (ratio of unit-area Gaussian and Lorentzian peak heights). Rb-87 D2 at 300 K:
`σ_D,peak/σ₀ = 0.017509`, `σ_D,peak = 5.089×10⁻¹⁵ m²`. Two closures: the shipped engine's `α_p/N`
gives `5.129×10⁻¹⁵ m²` (**0.8 %**) and spec 05's B9a reproduces to **1.9 %**. *(Re-graded per
R10-10.2: the 0.8 % closure is an internal-consistency check of the Doppler-broadening factor, not
an independent check of the dipole convention — both sides use the cycling `σ₀ = 3λ²/2π`.)*

**The model-free bound.** In steady state an atom cannot absorb trapped photons faster than the
ensemble emits them per atom, and the ensemble emission rate density is bounded by `α_p Φ_p`:

```
(10.16)   γ_trap ≤ α_p Φ_p / N = σ_eff Φ_p = (α_p/N)·I_p/(ħω_p)
          actual value:  γ_trap = (1 − g_esc)·σ_eff Φ_p          [g_esc MISSING]
```

**DERIVED-IN-SPEC.** It needs **no escape-factor theory** (`g_esc ≥ 0` suffices) and is
**independent of `N`** — more atoms means more emitters *and* more absorbers — so the knob is
probe photon flux. Rb-87, 300 K, `w₀ = 1 mm`, against `γ_gr = γ_t = 2π×39.8 kHz`:

| Probe power | `I₀` [W/m²] | `γ_trap` bound | as % of `γ_gr` |
|---|---|---|---|
| 0.39 µW (`= 0.01 I_sat`) | 0.2546 | `2π×0.82 kHz` | **2.1 %** |
| 1 µW | 0.637 | `2π×2.04 kHz` | 5.1 % |
| 10 µW | 6.37 | `2π×20.4 kHz` | 51 % |
| 100 µW | 63.7 | `2π×204 kHz` | **513 %** |

**Enforcing `s_sat ≤ 0.01` bounds trapping to ≤ 2.1 % of the transit-limited `γ_gr`. Above ~10 µW
at mm waists it can no longer be dismissed, and the corpus runs at `I_p/I_sat ≈ 0.16` (Mohapatra)
to `≈ 17.6` (Su): over the entire regime this module exists to reach, radiation trapping is
bounded only, never estimated** (R10-16). **MISSING:** the closed-form Holstein escape factor
`g_esc(k₀R)` for a Doppler-broadened line in a cylinder — Molisch, Oehry, Schupita & Magerl,
**JQSRT 49, 361 (1993)** and Holstein, **Phys. Rev. 83, 1159 (1951)** were located but not fetched.
Until then only (10.16) may be used and **no `g_esc`-dependent number may ship**.
**Named convention trap:** published "trapping factors" (e.g. the fetched Cs `g₁ = 1.6`) are
`τ_eff/τ = 1/g_esc`, the **reciprocal** of the escape factor. Transcribing one as `g_esc` inverts
the physics.
**UNVERIFIED sub-claim, fenced:** that a reabsorption event destroys the `g–r` coherence with
**unit** probability. Natural (it is a projective measurement) and conservative (it maximises
`γ_trap`), but unsourced — it can only tighten the bound, so it cannot corrupt a shipped number.

### 4.10 Transverse structure, ordering, and self-lensing

Under A5 each radial shell propagates independently with its own `Ω_p(r), Ω_c(r)`, and the measured
transmission is the spec-05 §2.g power-weighted average over `s_radial = 2r²/w₀²`. **The
z-propagation must be performed *inside* the radial quadrature, never after it**, because
`⟨exp(−OD(r))⟩ ≠ exp(−⟨OD(r)⟩)` and the gap grows with OD (R10-14; the transverse-wavevector form
of Eq. (10.4) — NJP 2023 Eq. (36) — is the same statement).

**Exact radial identity (VERIFIED, analytic, numerically confirmed at 24-node Gauss–Laguerre).**
For equal waists, the probe-power-weighted mean of `Ω_c²` is

```
(10.17)   ⟨Ω_c²⟩/Ω_c0² = ∫₀^∞ e^{−s}e^{−s} ds / ∫₀^∞ e^{−s} ds = 1/2
          ⇒ Ω_c,eff = Ω_c0/√2 = 0.70711·Ω_c0 ;  general: 1/(1 + (w₀p/w₀c)²)
```

**On-axis evaluation overstates `Ω_c,eff` by 29.3 % and the EIT width (`∝ Ω_c²/2γ_ge`) by a factor
2**, hence — since the AT resolvability threshold is `Δf_AT ≳ Γ_EIT^obs/2π` — a **√2 error in the
minimum resolvable field**. Spec 05 §4.8 already forbids peak-intensity-only comparison; this
document raises it to a refusal (§9, R-P4) because thick-cell propagation multiplies the error:
each shell has a different OD, so radial and longitudinal averaging do not commute.

**Self-lensing fence (DERIVED-IN-SPEC; thresholds are declared engineering judgement).**

```
(10.18)   Δφ_lens = (k_p L/2)·[ Re χ_p(r=0) − Re χ_p(r=w₀) ]
```

Near the EIT flank `|Re χ| ~ |Im χ|`, so `Δφ_lens ~ (OD/2)·(fractional radial variation of χ)`. At
`OD = 50` with 20 % radial variation, `Δφ_lens ≈ 5 rad` — the paraxial-shell model has failed.
**GATE-L: flag above 0.3 rad, refuse above 1 rad.** Spec 05 §7.6 declares the limitation; (10.18)
is the criterion it lacked.

### 4.11 The density trap — why "optically thick" and "dilute" are one knob pulled two ways

This module exists so RydSim can raise OD, which is raised by raising `N` (or `L`). But every
collisional term scales with `N`, and "dilute, non-interacting atoms" is spec 06 §7.4's own
premise. Inputs: self-broadening `Γ_self = β·N` (FWHM; `γ_ge += β·N/2`), `β/2π = 1.03×10⁻⁷ Hz·cm³`
Rb D2 / `1.16×10⁻⁷` Cs D2 (**VERIFIED in spec 04 §3.5 per R-6; NOT re-fetched**); Rydberg–ground
Fermi shift `−9.9×10⁻⁸ Hz·cm³·N[cm⁻³]` with broadening `≈ 0.5·|shift|` (**UNVERIFIED, 2× flag**,
audit R11). Natural Rb, Rb-87 sensed, `Ω_c/2π = 5 MHz`, weak probe, line centre:

| Target OD | `T` (5 cm) | `N_total` (cm⁻³) | `Γ_self` FWHM | as % of `Γ_e` | Fermi broad. as % of default `deph_r` |
|---|---|---|---|---|---|
| 0.1 | 12.1 °C | 3.12×10⁹ | 0.3 kHz | 0.005 % | 0.2 % |
| 1 | 33.9 °C | 3.23×10¹⁰ | 3.3 kHz | 0.05 % | 1.6 % |
| **10** | **59.9 °C** | **3.36×10¹¹** | **34.6 kHz** | **0.57 %** | **16.6 %** |
| **~28 (ceiling)** | **~75 °C** | **1.0×10¹²** | **≈103 kHz** | **≈1.7 %** | **≈50 %** |
| 100 | 91.8 °C | 3.51×10¹² | 361 kHz | 5.96 % | 174 % |
| 1000 | 130.6 °C | 3.69×10¹³ | 3.80 MHz | 62.6 % | 1824 % |

Reference crossings (Rb): `Γ_self` reaches `γ_t/2π = 39.8 kHz` at `N = 3.86×10¹¹ cm⁻³`;
`Γ_self = 0.1 Γ_e` at `5.89×10¹² cm⁻³`; Fermi shift reaches 100 kHz at `1.01×10¹² cm⁻³`.
Cs, 5 cm (Jing geometry): OD 1 at 2.0 °C, OD 10 at 23.4 °C, OD 100 at 49.1 °C.

**Normative reading (as ruled in R10-6):**

1. **`N ≤ 3.4×10¹¹ cm⁻³` (OD ≲ 10, ≤ 60 °C at 5 cm): sound.** Every collisional term is ≤ 17 % of
   an already-modelled rate, and all of them *are* modelled — but they are **not switched on** in
   `LadderConfig` today. They must be.
2. **`3.4×10¹¹ < N ≤ 1×10¹² cm⁻³`: computable with mandatory flags.** The Rydberg–ground
   broadening reaches ~50 % of the default `deph_r`, and **it is the term carrying the 2×
   UNVERIFIED uncertainty**. Any output here inherits that 2× on the EIT/AT linewidth, hence on the
   resolvability threshold, hence on `E_min`.
3. **`N > 1×10¹² cm⁻³`: REFUSE (GATE-D).** Three independent sourced reasons: spec 05 §7.2 declares
   the Durham-grade **D2** self-broadening coefficient MISSING above `10¹² cm⁻³` (Weller's 0.1 %
   validation to `3×10¹⁴ cm⁻³` is **D1**), so `β` itself leaves its validated band; the dominant
   Rydberg linewidth becomes the UNVERIFIED Fermi surrogate, so the answer is an extrapolation of a
   guess; and charge-induced optical bistability is *observed* at Rb total ≈ **2.5×10¹² cm⁻³**
   (Weller, Urvoy, Rico, Löw & Kübler, **Phys. Rev. A 94, 063820 (2016)**, fetched: 2 % Rydberg
   fraction, `N_ion ≤ 1×10¹⁰ cm⁻³` = 27 % of `N_r`, dipole–dipole ruled out in favour of
   charge-induced) while RydSim has **no ionization, free-charge or bistability physics at all**.
4. **The trap in one line:** for a 5 cm Rb cell the entire usable window is
   **34 °C ≤ T ≤ ~75 °C**, spanning `OD ≈ 1 → 28`. Below it the cell is thin and the solver is a
   no-op; above it the atoms are not the atoms the model describes. Every design campaign must
   report where in that window it sat.

**Rydberg–Rydberg vdW is *not* the binding constraint at weak probe** (computed with `ρ_rr = 10⁻³`,
`f_vel = Γ_e/Δν_D = 0.0119`, `Δν_vdW = (C₆/h)/r_nn⁶`, R-26): 0.01–0.04 kHz at OD ≈ 10, 0.58–4.3 kHz
at OD ≈ 100 — two to three orders below the ground-perturber terms. **But at the corpus's actual
Rydberg fraction (Weller's 2 %, 20× larger), `r_nn` falls 2.7× and `Δν_vdW` rises by 2.7⁶ ≈ 390×,
into the MHz** — and `ρ_rr` is exactly what the strong-probe solver will start producing. Hence
GATE-D's companion refusal R-P8 **computes** `N_r = N·f_vel·ρ_rr` and never assumes it.
Confidence: order-of-magnitude only (C₆ fit LITERATURE-RECALL, audit R11).

**Foreign-gas velocity-changing collisions may legitimately be neglected — and this is now a
sourced neglect.** Lei, Eckel, Norrgard, Prajapati, Artusio-Glimpse, Simons & Holloway,
arXiv:2408.16669 → **Phys. Rev. Applied 23, 034028 (2025)** (fetched): *"roughly 0.02 mbar of
contaminant gas would be required to add roughly 1 MHz of additional broadening"*, and cells are
*"generally evacuated to well below 10⁻⁵ mbar prior to sealing"* ⇒ ≲ 1 kHz, negligible. It may
**not** be neglected for buffer-gas cells, which remain refused outright (audit refusal #16).
**MISSING: alkali–alkali VCC**, which scales with `N` and therefore rises exactly where the density
trap bites; that paper covers foreign gases only.

---

## 5. Constants, parameters and fixtures

**No new empirical constant is introduced by this document.** This table records the derived and
borrowed quantities a propagation implementation touches.

| Quantity | Symbol | Value | Unit | Source | Confidence |
|---|---|---|---|---|---|
| Vacuum probe wavenumber | `k_p` | `2π/λ_p` (**no extra 2 or 4π**) | rad/m | spec 05 §2.f verbatim | VERIFIED |
| Propagation coefficient | — | `dÊ/dz = +i(k/2)χÊ` | — | Ogden PRL **123**, 243604 Eq. (6), fetched | **VERIFIED** |
| Doppler-averaged source | — | `P = 𝒩∫f(v_z)Tr[dρ(z;v_z)]dv_z` | C/m² | Ogden Eq. (8), fetched | **VERIFIED** |
| Resonant absorption coefficient | `α₀` | `k_p N μ₁₃²/(2ħε₀γ₁₃)` | m⁻¹ | NJP **25**, 035001 Eq. (12), fetched | **VERIFIED** |
| OD convention identity | `OD` | `2α₀L ≡ k_p Im χ L` | — | NJP **25**, 035001, fetched | **VERIFIED** |
| Closed-form solution | — | `E_out = E_in e^{ik_zL∫d²v w(v)χ_p}` | — | NJP **25**, 035001 Eq. (36), fetched | **VERIFIED** |
| Transport equation | — | `(∂_t + v·∂_r)ϱ̃ + collisions = source` | — | RMP **85**, 941 Eq. (14), fetched | **VERIFIED** |
| Linear-response condition | — | `Ω_p ≪ Ω_c` | — | NJP **25**, 035001, fetched | **VERIFIED** |
| Cell-geometry criterion | — | `D/λ_rf < 0.1` | — | Fan *et al.*, PRApplied **4**, 044015 (2015), fetched | **VERIFIED** |
| Pyrex RF absorption @ 12.6 GHz | — | 0.066 % per mm | — | ibid., fetched | **VERIFIED** |
| Bistability density (Rb, total) | — | ≈2.5×10¹² | cm⁻³ | PRA **94**, 063820 (2016), fetched | **VERIFIED** |
| Foreign-gas VCC tolerance | — | ~0.01–0.02 mbar for 1 MHz | — | PRApplied **23**, 034028 (2025), fetched | **VERIFIED** |
| Wavelength ratio (Rb fixture) | `λ_p/λ_c` | 1.625503 (780.241209686/480.0 nm) | — | lock #10 / R-15 | VERIFIED (fixture only) |
| Rb-87 D2 resonant cross-section | `σ₀` | `3λ²/2π` = 2.9067×10⁻¹³ | m² | spec 05 §2.f / Steck | VERIFIED |
| Doppler peak/resonant ratio | — | `√(π ln2)(Γ_e/2π)/Δν_D`; `√(π ln2)` = 1.475665 | — | Eq. (10.15) | **DERIVED-IN-SPEC** (2 closures) |
| Rb-87 D2 Doppler peak σ, 300 K | `σ_D,peak` | 5.089×10⁻¹⁵ (analytic) / 5.129×10⁻¹⁵ (engine) | m² | Eq. (10.15) vs live run | **DERIVED**, 0.8 % closure (see R10-10.2) |
| `I_sat`, far-detuned π, Rb-87 D2 | `I_sat` | 2.50399(73) mW/cm² = 25.04 W/m² | W/m² | Steck via spec 05 §2.g | VERIFIED |
| Transit rate, Rb-87 300 K, `w₀`=1 mm | `γ_t` | `2π×39.79 kHz` | rad/s | spec 05 §2.e / R-3 | VERIFIED (recomputed) |
| Rb-87 300 K speeds `v_p/σ_v/⟨v⊥⟩` | — | 239.6 / 169.4 / 212.3 | m/s | spec 05 §2.c | VERIFIED (recomputed) |
| Rb D2 self-broadening | `β/2π` | 1.03×10⁻⁷ (theory), 1.10(17)×10⁻⁷ (meas.) | Hz·cm³ | spec 04 §3.5 / R-6 | VERIFIED-in-repo, **not re-fetched** |
| Rydberg–ground Fermi shift | — | −9.9×10⁻⁸ | Hz·cm³ | spec 04 §2.3.5 | **UNVERIFIED, 2× flag** |
| Rydberg–ground broadening surrogate | — | `≈ 0.5·\|shift\|` | — | spec 04 §2.3.5 | **UNVERIFIED, 2× flag — dominant term in band 2** |
| Coupling-depletion bound | — | Eq. (10.8); 2.71×10⁻⁵ at 1 µW / 30 mW | — | §4.5 | **DERIVED-IN-SPEC** |
| Depletion/saturation ratio | — | `½(λ_p/λ_c)(1−T)I_sat/I_c` = 1.07×10⁻³ | — | §4.5 | **DERIVED-IN-SPEC** |
| Trapping bound | `γ_trap` | Eq. (10.16); `2π×0.82 kHz` at `s_sat = 0.01` | rad/s | §4.9 | **DERIVED-IN-SPEC** |
| Holstein escape factor | `g_esc` | closed form | — | Molisch JQSRT **49**, 361 (1993); Holstein PR **83**, 1159 (1951) | **MISSING** (unfetched) |
| Transport parameter, 5 cm Rb | `ε_transp` | `8.0×10⁻⁴ × OD` | — | Eq. (10.12) | **DERIVED-IN-SPEC** |
| Axial/transverse traversal ratio | — | 62.7 (`L`=5 cm, `w₀`=1 mm, 300 K) | — | §4.6 | **DERIVED-IN-SPEC** |
| Radial `⟨Ω_c²⟩` identity | — | `1/2` equal waists; `1/(1+(w₀p/w₀c)²)` general | — | Eq. (10.17) | VERIFIED (identity + numeric) |
| Per-step OD (fixed-step branch) | `η` | 0.05 | — | spec 05 §2.f; corroborated by §6.1 error law | VERIFIED (spec-internal) + SELF-MEASURED |
| RK4 amplitude error law | — | `\|Δy/y\| ≈ OD·ΔOD⁴/3840`, order 4.00 | — | §6.1 | **SELF-MEASURED** |
| z-convergence acceptance | `ε_z` | 1e-4, **absolute in OD** | — | mirrors spec 06 §4.4 | NORMATIVE (this spec) |
| Relaxation acceptance / ceiling / cap | `ε_relax`, `q`, `m_max` | 1e-4; damp above 0.5, raise above 1.0 ×2; 20 | — | §6.2 | NORMATIVE |
| Forward-only gate | `ε_gate` | 1e-5 (one decade inside `ε_z`) | — | §6.2 | NORMATIVE |
| Chebyshev nodes in `Ω_c` | `M_c` | 16 → 2.473×10⁻⁹ (frozen grid) | — | §6.3 | **SELF-MEASURED** |
| Interpolation error budget | — | ≤ 1e-6 (`ε_z`/100) | — | §6.3 | NORMATIVE |
| float64 `exp(−OD)` subnormal / zero | — | 708.3964185322641 / ≈745.2 | — | measured | **SELF-MEASURED** |
| Probe-scan absorption dynamic range | — | 4.40×10³ (Rb-87, 300 K, ±1.5 GHz) | — | §6.1 | **SELF-MEASURED** |
| Loop sensitivity `\|∂ln Im χ̄_p/∂ln Ω_c\|` | `s_c` | 0.515 / 0.825 / 0.986 at `Ω_c/2π` = 2.5/5/10 MHz | — | §6.2 | **SELF-MEASURED** |

**Fixtures (published; every row fetched during draft authoring).**

| Fixture | Cell / beams | `Ω_p/Γ_e` | `s_sat` | weak-probe OD | Confidence |
|---|---|---|---|---|---|
| Mohapatra 2007 (Rb) | 75 mm; probe 1 µW, `w₀`=0.4 mm; coupling ≤200 mW, `w₀`=0.8 mm | — | ≈0.16 | 1.298 (⁸⁵Rb F=3, 25 °C) | VERIFIED (ar5iv quant-ph/0612200) |
| Sedlacek 2012 (Rb) | 10 cm glass, **7.5 cm effective**; `w_p`=750 µm, `w_c`=100 µm | 0.989 | **1.956** | **0.4805** (⁸⁷Rb F=2, 25 °C) | VERIFIED (ar5iv 1205.4461) |
| Jing 2020 (Cs) | **5 cm** (R10-20), `N₀`=4.89×10¹⁰ cm⁻³; probe 120±4 µW, 1/e² dia 1.70 mm; coupling 34±1 mW, 2.00 mm | 1.089 | **2.372** | **4.639** (R10-19) | VERIFIED (ar5iv 1902.11063) |
| Jing transmitted power | ~10 µW from 120 µW ⇒ `T`=0.0833, `OD_eff`=2.485 | — | — | — | VERIFIED (published pair) |
| Su 2022 (Rb) | Thorlabs GC25075-RB, 27–65 °C; `Ω_p/2π`=30 MHz, `Ω_c/2π`=0.38 MHz | 4.945 | **48.91** | 0.566 (27 °C) → 17.10 (65 °C) | VERIFIED (ar5iv 2111.13408) |
| Su cell length | Ø25.4 mm × **71.8 mm** borosilicate | — | — | — | **LITERATURE-RECALL** (vendor snippet; direct fetch failed) |
| Siddons reference | 7.5 cm natural Rb; 16.5/25.0/25.4/36.6 °C; `I/I_sat`=0.002; **rms error better than 0.2 %** | — | — | — | VERIFIED (arXiv:0805.1139) |
| Densities | `N`(Rb,298.15 K)=1.2918e16 m⁻³; `N`(Rb,323.15 K)=1.4672e17; `N`(Cs,298.15 K)=4.8941e16 | — | — | — | VERIFIED-COMPUTED (= spec 05 B3a/c/e; Cs matches Jing's printed 4.89e10 cm⁻³) |

**MISSING, deliberately not specified here (owned elsewhere):** `℘_er` and the coupling-frequency
prefactor (spec 03 / `rydsim.dipoles`); the `Ω_RF(z)` profile inside the cell (cell-EM module, spec
06 §2.8 items 6–7); `N(T)` coefficients (spec 05 §2.a).

---

## 6. Numerical method and named pitfalls

### 6.0 Branch structure — selected by computed gates, never by assumption

| Branch | Condition (all computed at runtime) | Method | Cost |
|---|---|---|---|
| **S0** closed form | linear-response gate holds **and** forward-only gate G holds | `ln T = −k_p L Im χ̄_p`, one velocity average | `O(N_Δ N_v)` |
| **S** quadrature | linear-response gate holds, coupling depletes | Gauss–Legendre in z + relaxation | `O(M N_Δ N_v + N_it N_z N_Δ)` |
| **N** nonlinear | probe above the linear-response gate | adaptive embedded RK on `ln Ω_p` + relaxation | S × `N_z` |

Only in branch **N** is (10.1) genuinely nonlinear. **Branch S0 must stamp `branch="S0_closed_form"`
(R10-13).** Verified on the shipped code: `rydsim.eit.chi_ladder` returns **bit-identical** output
for `Ω_p/2π = 1 Hz` and `10 MHz` (max relative difference exactly `0.0`, SELF-MEASURED) — `Ω_p`
cancels between `σ_eg` and the χ prefactor, so in branches S0/S the propagation equation is a
**linear scalar ODE** whose exact solution is the exponential of a *quadrature*. **The numerical
task in S0/S is quadrature in z, not integration of an ODE** (the scalar Magnus series truncates
identically at first order).

### 6.1 Discretisation in z — integrate the logarithm

Define `u_p ≡ ln Ω_p` (complex). Then `du_p/dz = i(k_p/2)χ̄_p(z)`, and
`OD = −2 Re[u_p(L) − u_p(0)]`, `φ = Im[u_p(L) − u_p(0)]`. Three properties the amplitude
formulation lacks: no underflow (`OD = 750` is representable, `exp(−750)` is not); an absolute
tolerance on `Re u_p` **is** a tolerance on OD, which **is** a uniform relative tolerance on `T`
across the scan; and the reduction of §4.3 is exact by construction, because a constant integrand
is integrated exactly by every consistent quadrature rule.

* **Branches S0/S:** Gauss–Legendre in z, order `n`. `χ̄_p(z)` is analytic in the smooth profile
  `Ω_c(z)`, so GL converges spectrally; `n = 8–16` typically suffices — orders of magnitude fewer
  velocity averages than marching.
* **Branch N:** `scipy.integrate.solve_ivp(method="DOP853", rtol=1e-10, atol=1e-12)` on `u_p`
  (VERIFIED from the fetched SciPy reference: *"Explicit Runge-Kutta method of order 8"*, error
  controlled as `atol + rtol*abs(y)`; **the defaults `rtol=1e-3`/`atol=1e-6` are far too loose and
  must be overridden**).
* **Fixed-step / marching fallback step rule.** State the step in local absorption lengths:
  `Δz ≤ η/α_p(z)`, i.e. per-step `ΔOD ≤ η`. Measured RK4 global relative amplitude error at
  `OD_total = 20`: `ΔOD` = 1.00/0.40/0.20/0.10/0.04/0.02 → 7.949e-3 / 1.576e-4 / 9.058e-6 /
  5.430e-7 / 1.356e-8 / 8.403e-10, fitted order **4.00**, closed form `|Δy/y| ≈ OD·ΔOD⁴/3840`
  (better than 10 % for `ΔOD ≤ 0.2`). Inverting, `η ≤ (3840 ε_amp/OD_total)^{1/4}`.
  **Spec 05 §2.f's existing `|ΔOD| ≤ 0.05` is corroborated, not replaced**: it delivers
  `ε_amp = 1.6e-7` at OD = 100. Keep 0.05 as the default `η`; expose the formula. Spec 05's
  companion `|ΔΩ_p|/|Ω_p| ≤ 2 %` is the same constraint restated (`ΔOD ≤ 0.04`) and is retained as
  a redundant guard. **Canary:** a *fixed* 16-step RK4 is wrong by **209 %** at OD = 20 — which is
  why the step rule is normative rather than advisory (10/B-01c).

**Per-detuning z-grids (normative).** Measured absorption dynamic range across a probe scan
(Rb-87, 300 K, ±1.5 GHz, 601 detunings, 4.01e5 velocity nodes): peak `1.010e-9` at
`Δ_p/2π = −5.0 MHz` to `2.297e-13` in the wings — a factor **4.40×10³**, of which the EIT feature
contributes only ×2.55. **The Doppler profile, not the transparency window, is the dominant
driver.** A z-grid adequate in the wings is 4400× too coarse at line centre, so **each `Δ_p` gets
its own GL order or its own adaptive step sequence, sized from that detuning's own `α`.**
Independent corroboration (VERIFIED, fetched): Häupl *et al.*, arXiv:2410.19916 (NJP 2025) —
*"the current model only allows the calculation of the propagation-corrected absorption for a
single value of the detuning Δ at a time."*
**Do not call this stiffness.** For a single detuning the scalar equation has exactly one rate,
varying slowly through `Ω_c(z)`; explicit methods are optimal. Genuine stiffness can appear only in
branch N near the saturation knee — switch to `Radau`/`LSODA` there, and **on a measured symptom**
(DOP853 step rejections > 30 % of attempts), never pre-emptively.

**Rule Z-CONV (normative).** Recompute the whole spectrum on a refined z-rule that (i) at least
**halves** the effective step (GL `n → 2n`, or adaptive tolerance ÷ 100) **and** (ii) is
**node-disjoint** from the coarse rule. Accept iff
`max over Δ_p of |OD_fine − OD_coarse| ≤ ε_z = 1e-4` — an *absolute* criterion on OD, hence a
*relative* one on `T` uniformly, and still meaningful where `T` underflows. Failure raises
`IntegrityError` carrying the measured magnitude. `(converged, max_delta_od, rule,
n_nodes_coarse/fine)` ships in provenance (audit §4 item 6).
*Node-disjointness is a requirement, not a nicety:* a nested refinement that reuses coarse nodes
can agree with its parent to high precision while both alias the same structure — the identical
failure mode that made Gauss–Hermite unusable (R-2 / audit R2: *"'doubling nodes' can falsely
converge on an under-resolved dip"*). That `GL_n` and `GL_2n` share no nodes is **UNVERIFIED as a
theorem**; the implementation must not rely on it — assert at runtime that no refined node lies
within `1e-12·L` of a coarse node, else refine to `GL_{2n+1}`.

### 6.2 The counter-propagating BVP

**Scheme A (default): wave relaxation, undepleted coupling as the zeroth iterate.**

```
m = 0:   Ω_c^(0)(z) ≡ Ω_c^in                      (the current engine's model)
repeat:  forward  solve (10.1) with Ω_c^(m)   →  Ω_p^(m+1)(z)
         backward solve (10.2) with Ω_p^(m+1) →  Ω̃_c^(m+1)(z)
         damp     Ω_c^(m+1) = (1−θ_m)Ω_c^(m) + θ_m Ω̃_c^(m+1)
```

Gauss–Seidel wave relaxation, chosen because the zeroth iterate **is** the physical answer in the
regime where the loop is weak, so the iteration starts inside the basin and each sweep is a
physically interpretable correction. **Convergence is judged in the observable, not the field** — a
field residual can be small while the OD it produces is not:
`r_m = max_Δ |OD^(m+1) − OD^(m)|`, `q_m = r_m/r_{m−1}`.
**Accept** when `r_m ≤ ε_relax = 1e-4` **and** `q_m < 1` — never on `r_m` alone, because a stalled
iteration produces a small increment while sitting far from the fixed point. **Damp** `θ_m = 1`
while `q_m ≤ 0.5`, else `θ_m = 1/(1+q_m)`; Anderson(2) is permitted only after A/B measurement.
**Refuse** if `q_m ≥ 1` twice consecutively or `m > 20`: raise `IntegrityError` naming
`(r_m, q_m, m, OD_p, OD_c)` and **do not return the last iterate** — a diverging relaxation here
means the physical feedback loop has gain ≥ 1, i.e. the medium is bistable or self-focusing at
these parameters, which is a physics statement the caller must see. The full trace `[(m,r_m,q_m)]`
ships in provenance. Expected contraction is fast (`|s_c| ≲ 1` measured), **which is a prediction
`q_m` tests, not an assumption the code may make.**

**Gate G (forward-only, a-priori bound + mandatory measured certificate).**
`|δOD| ≤ OD_p·s_c·(1 − e^{−OD_c/2}) ≤ OD_p·s_c·OD_c/2`; take the forward-only path iff
`OD_p·s_c·OD_c/2 ≤ ε_gate = 1e-5`. **Whenever G passes, still take exactly one relaxation sweep and
record `r_1`.** One backward pass is negligible against the forward scan and converts "I proved a
bound" into "I measured the thing the bound was about". If `r_1 > ε_gate`, **raise** and fall
through to Scheme A: *a bound that is ever violated is a wrong bound* (10/B-08).

**Scheme B (fallback): collocation.** When Scheme A refuses, do not give up and do not loosen
`ε_relax`. Recast as a real 4-vector `y = (Re u_p, Im u_p, Re u_c, Im u_c)` and hand to
`scipy.integrate.solve_bvp` — *"a 4th order collocation algorithm with the control of residuals"*
using *"a damped Newton method with an affine-invariant criterion function"* (VERIFIED, fetched).
Normative settings: `tol = 1e-8` (**not** the default `1e-3`, three orders looser than `ε_z`),
`max_nodes ≥ 1e4`, seeded with Scheme A's last iterate. **Status handling is a refusal, not a
warning:** status 1 (max mesh nodes) and status 2 (singular Jacobian) both raise. Status 0 still
goes through Z-CONV. Cost warning: each Newton step is `O(n_mesh)` velocity averages — without the
interpolant of §6.3 this path is unusable, which is why Scheme A is the default.

### 6.3 The inner loop — Chebyshev interpolation in `Ω_c`, on a frozen velocity grid

Measured unit cost on the shipped `rydsim.eit` path: one Doppler-averaged spectrum over 201
detunings on a 42 281-node grid takes **0.442 s** (~52 ns per detuning×node; hardware-dependent,
anchor only). Naive nesting (Doppler loop inside the z loop, as CoOMBE does — VERIFIED from
`mbe.f90`) costs `N_it·N_z·N_Δ·N_v·c₀` ≈ **265 s per spectrum per LO point**, i.e. ≈3 hours for a
40-point superheterodyne transfer curve. **This is the real reason the engine currently refuses.**

`χ̄(Δ_p; z)` depends on `z` *only* through `(Ω_c(z), Ω_RF(z))` — plus `|Ω_p(z)|²` in branch N. So
build once a tensor Chebyshev interpolant of the **complex velocity-averaged response `S̄`** (never
`T` or `OD`, §6.4-P7) on Chebyshev–Gauss nodes spanning `[Ω_c(0), Ω_c(L)]` widened 10 %.

> **Rule V-FREEZE (normative; R10-8).** All `M` interpolant nodes are evaluated on **one and the
> same velocity grid**, built once for the **narrowest dressed feature over the whole parameter
> box** (smallest `γ_min`, largest `Ω_c`, largest `Ω_RF`) and then **frozen**. Maxwell weights are
> renormalised **once**, at freeze time.

**Measured convergence** (Rb-87 300 K, 201 detunings ±2π·30 MHz, `Ω_c/2π ∈ [1.5, 5.0]` MHz):

| `M` | frozen grid (V-FREEZE) | grid rebuilt per node |
|---|---|---|
| 8 | 3.296e-5 | 3.293e-5 |
| 12 | **2.804e-7** | 2.364e-6 |
| 16 | **2.473e-9** | 6.365e-7 |
| 20 | **2.343e-11** | — |
| 24 | — | 4.126e-7 (**floored**) |

Two readings, both load-bearing: frozen converges **spectrally**; rebuilt **floors at ~4e-7 and
stops improving** — the limiter is grid jitter, not polynomial degree, and that floor is only 400×
inside `ε_z`, i.e. indistinguishable from "converged" to a careless refinement test. **Normative
default `M_c = 16`.** Interpolating in `Ω_c²` was measured and is not better; **use `Ω_c`**. For a
z-varying `Ω_RF`, a tensor `M_c × M_RF = 16 × 16`; the 2-D spectral rate is **LITERATURE-RECALL and
must be measured before it is relied on** (10/B-24b).
**Accounting rule:** interpolation error is not free error. Require `err_interp ≤ ε_z/100 = 1e-6`,
**measured** at ≥ 5 points drawn inside the box and never coinciding with the Chebyshev nodes;
report it in provenance. Resulting speed-up: `N_it·N_z/(M_c·M_RF)` — 600 → 16 velocity averages,
**37.5×**, at 2.5e-9 interpolation error (265 s → 7.1 s on the measured unit cost).

**A caching trick that must NOT be imported.** Häupl *et al.* reduce Doppler cost by the
substitution `v′_z = Δ/k − v_z`, so `χ(v′_z)` is computed once and reused across `Δ` (VERIFIED,
fetched). **Unavailable to RydSim.** It requires the response to depend on `(Δ, v)` only through
`Δ − kv`. A mismatched ladder depends on **two** independent combinations,
`a = Δ_p − k_p v` and `b = Δ_p + Δ_c + (k_c − k_p)v`, whose Jacobian determinant is `k_c ≠ 0`; the
map is a bijection and the response is irreducibly two-argument. (The trick reappears in the
degenerate case `k_c = k_p`, which is why it is correct for the single-field D-line problem Häupl
*et al.* solve.)

### 6.4 Named pitfalls — symptom and detector

Entries marked **always-on** run in production, not only in CI.

**P1 — Silent under-resolution (in z, and in the `Ω_c` interpolant).** *Symptom:* a smooth,
plausible spectrum whose OD is wrong by a few percent; refinement appears to converge because it
shares nodes with the coarse rule, or because the interpolant floored on grid jitter (measured
4.1e-7) rather than on degree. *Prior-art warning (VERIFIED, fetched):* Häupl *et al.* —
*"the value of N required for the model to converge must be determined manually."* **A
manually-chosen slice count is precisely the hazard; RydSim must not ship one.**
*Detector (always-on):* Z-CONV with node-disjoint refinement, **plus** the off-node interpolant
check. Both **raise**. Mutation-tested (10/B-23).

**P2 — Exponential underflow at high OD.** *Symptom:* `T = 0.0`, every downstream quantity becomes
0/0 or ∞ and "looks like a result" — exactly audit CRIT-2's 5.4e9 nV/cm/√Hz. *Measured float64
thresholds:* subnormal below `OD = 708.3964185322641`, flush-to-zero for `OD ≳ 745.2`.
*Fix (normative):* primary return is `ln T`; `T` is an accessor flagging `underflow=True`
(R10-12). *Physics rule that must accompany it:* at `OD ≳ 30` the transmitted probe is below any
realistic detector floor — the honest output is "no signal, OD = X", never a denormal (GATE-R).

**P3 — Sign error in the counter-propagating term.** *Symptom:* the coupling *gains* across the
cell; EIT contrast grows with cell length; the relaxation diverges for no physical reason.
*Correct sign:* with `dΩ_c/dz = −i(k_c/2)χ̄_cΩ_c`, `d|Ω_c|²/dz = +k_c Im(χ̄_c)|Ω_c|² > 0` for an
absorptive medium, so `|Ω_c|` is **largest at `z = L`**, its entry face. Copying the probe's `+i`
manufactures gain. *Detector (always-on):* assert `|Ω_c(z)|` monotone non-decreasing in `z` and
`|Ω_c(0)|/|Ω_c(L)| = exp(−OD_c/2)` to 1e-10 (10/B-12).

**P4 — Intensity vs amplitude: the factor-2 trap (lock #3).** *Symptom:* OD wrong by exactly 2× or
0.5×; catastrophic in a superhet run because both `P` and `dP/dE` inherit it. The three quantities
that must never be interchanged: `α = k_p Im χ̄` (**intensity**); `α_amp = α/2` (**amplitude**, the
coefficient in (10.1)); `Im n_refr ≈ Im χ̄/2`. *Independent primary corroboration (VERIFIED,
fetched):* Häupl *et al.* Eq. (11) writes `dI/dz = (4π/λ)·Im{√(1+χ)}·I`, i.e. `2k·Im n_refr` — two
factors of 2 that cancel against `Im n_refr = Im χ/2` to give exactly `k·Im χ`. Writing
`k·Im n_refr` for intensity is 2× low; `2k·Im χ` is 2× high. *Detector:* 10/B-04.

**P5 — Velocity-grid jitter under a moving `Ω_c`.** *Detector:* V-FREEZE plus the mutation test
that asserts the frozen-grid error at `M = 16` is ≤ 1e-8 **and** the rebuilt-grid error is ≥ 1e-7 —
i.e. **the rule is proven to matter, not merely asserted** (10/B-24).

**P6 — One shared z-grid across the whole probe scan.** *Symptom:* line-centre OD systematically
low, wings fine — the error hides exactly where `T` is smallest. *Detector:* 10/B-26.

**P7 — Interpolating `T` or `OD` instead of `χ̄`.** *Rule:* interpolate the complex response;
form `OD` and `T` only after the z-quadrature. *Detector:* type-level — the cache API accepts and
returns complex response arrays only.

**P8 — Renormalising Maxwell weights inside the z loop.** Wasted work, and if the grid ever changes
mid-loop it introduces a step-to-step discontinuity in `χ̄` that destroys the z-quadrature's
smoothness assumption. *Rule:* renormalise once, at freeze time.

**P9 — A relaxation that "converges" because the update never took effect.** *Symptom:* `r_1 ≈ 0`
on every fixture, including ones engineered to deplete strongly — the backward sweep writes into a
copy, or the forward sweep reads a stale `Ω_c`. *Detector:* a **positive control** — a fixture with
`OD_c ≈ 0.3` must show `r_1` comfortably above `ε_relax` before the iteration converges. **The test
asserts the correction is nonzero before asserting it is small** (10/B-10).

**P10 — Unit drift between `Ω`, `ℰ` and `I` along z.** `Ω ∝ ℰ ∝ √I` (locks #3/#4). A solver that
steps in `I` but reports `Ω`, or applies `exp(−ΔOD)` to an amplitude, is off by the square.
*Rule:* the state variable is `ln Ω` (complex) and nothing else; `I` and `ℰ` are derived at the API
boundary by one conversion each.

**P11 — Photon-flux bookkeeping across isotopes/hyperfine.** (10.6) counts photons per
*transition*. In a natural-abundance cell the probe absorption spreads over both isotopes and four
ground levels while the coupling addresses only the sensed one. The bound stays valid but becomes
loose by the branching fraction — **do not tighten it silently.**

**P12 — Averaging before propagating.** Radial (and any parameter) averaging must be *outside* the
z integration: `⟨exp(−OD(r))⟩ ≠ exp(−⟨OD(r)⟩)`, and the gap grows with OD (10/B-18). Spec 05
§4.8's "peak-intensity-only simulation is forbidden" is the same pitfall one level up.

**P13 — Black-region peak extraction.** At `OD ≳ 10` the transmission trace has a flat zero floor
hundreds of MHz wide (measured: 50 °C, 75 mm → `T_min = 6.7e-7`). Dip *positions* must be recovered
by fitting the wings of `OD(ν) = −ln T`, **never** by `argmin T`.

**P14 — The transparency-width √2 convention trap.** `2√(ln2/(a₂L))` (half-peak-transmission) vs
`2√(2ln2/(a₂L))` (half-depth) — a factor √2 apart, and **both appear in the literature**. RydSim
defines the width on the baseline-subtracted signal and **stamps the convention**. This is exactly
the class of bug lock #12's ±3 dB traps exist to prevent.

---

## 7. Python API

### 7.1 New module `rydsim.propagate`

```python
# rydsim/propagate.py
from dataclasses import dataclass, field
from typing import Literal
import numpy as np
from numpy.typing import NDArray

from .provenance import IntegrityError

class ThickCellError(IntegrityError):
    """Raised when a propagation gate refuses (integrity-audit refusal #18, as amended by
    spec 10 §3.2 A-1). Carries the offending gate name, its value and its threshold."""

Branch = Literal["S0_closed_form", "S_quadrature", "N_nonlinear"]

@dataclass(frozen=True, slots=True)
class PropagationGates:
    """Every gate is COMPUTED at runtime and reported; none is assumed (spec 10 §9)."""
    s_sat_entrance: float           # GATE-P  (i) linear response, (ii) ancillary neglect
    omega_p_over_min: float         #         Omega_p / min(Gamma_e, |Omega_c|)  -- audit #21
    coupling_depletion: float       # GATE-C  |dOmega_c|/Omega_c, MEASURED not bounded
    coupling_depletion_bound: float #         the Manley-Rowe a-priori bound, Eq. (10.8)
    epsilon_transport: float        # GATE-T  Eq. (10.12)
    lensing_phase_rad: float        # GATE-L  Eq. (10.18)
    number_density_m3: float        # GATE-D  ground density of the sensed element
    rydberg_density_m3: float       #         COMPUTED N*f_vel*rho_rr, never assumed
    transmitted_power_w: float      # GATE-R  vs detector_floor_w
    two_zr_over_l: tuple[float, float]   # GATE-G  (probe, coupling)
    d_over_lambda_rf: float | None  #         cell dimension / RF wavelength (Fan criterion)
    gamma_trap_over_gamma_gr: float #         radiation-trapping BOUND, Eq. (10.16)

@dataclass(frozen=True, slots=True)
class PropagationResult:
    ln_transmission: NDArray[np.float64]   # PRIMARY return (= -OD); never T (R10-12)
    phase_rad: NDArray[np.float64]         # accumulated dispersive phase, Eq. (10.4)
    branch: Branch                         # R10-13: a no-op must be stamped as a no-op
    gates: PropagationGates
    converged: bool
    convergence: dict                      # max_delta_od, rule, n_nodes_coarse/fine,
                                           # relaxation trace [(m, r_m, q_m)], err_interp
    validity_flags: list[str]              # warn-level items, empty list if none
    underflow: bool                        # ln_transmission < -708.3964185322641 anywhere
    provenance: dict                       # audit §4 items 1-12 + spec 10 §7.3 additions

    def transmission(self) -> NDArray[np.float64]:
        """T = exp(ln_transmission). Sets .underflow; guarded per spec 05 §4.7."""
```

**Entry points:**

```python
def propagate_probe(
    chi_fn,                       # (Delta_p, Omega_c, Omega_p) -> velocity-averaged complex S-bar
    delta_p: NDArray[np.float64],
    *, cell_length_m: float, lambda_probe_m: float, lambda_coupling_m: float,
    omega_p_in: float, omega_c_in: float,
    geometry: Literal["counter", "co"] = "counter",   # never a bare sign (spec 06 §4.8 iii)
    number_density_m3: float, gamma_e: float, i_sat_w_m2: float, dipole_ge_cm: float,
    detector_floor_w: float | None = None,
    radial: "RadialQuadrature | None" = None,
    force_branch: Branch | None = None,   # tests only; never a production path
) -> PropagationResult:
    """Solve (10.1)/(10.2) with the branch selected by COMPUTED gates (spec 10 §6.0).
    Raises ThickCellError on any gate refusal (§9), IntegrityError on non-convergence."""

def propagate_shells(...) -> PropagationResult:
    """z-propagation INSIDE the spec 05 §2.g Gauss-Laguerre radial quadrature (R10-14).
    Each shell propagates at its own (Omega_p(r), Omega_c(r)); shells do not exchange energy."""

def optical_depth_weak(...) -> NDArray[np.float64]:
    """The NORMATIVE weak-probe OD: d_eff,far = <J||er||J'>/sqrt(3) (spec 00 §6 gap 7) and the
    ground-hyperfine fraction p_F (spec 05 §2.b). Replaces the ad-hoc chain in
    experiment.superhet_transfer (R10-10). Reproduces spec 05 B9 to +-1 % (10/B-17)."""

def gate_report(...) -> PropagationGates:
    """Compute every gate WITHOUT solving. Cheap; used by the designer/search layer to
    reject infeasible points before paying for a solve."""
```

### 7.2 What changes in existing modules, and how backward compatibility is preserved

| Existing symbol | Change | Back-compat |
|---|---|---|
| `rydsim.eit.transmission(chi, length, lambda_probe)` | **signature and semantics unchanged.** Docstring narrowed: it is the *exact* branch-S0 answer under Theorem 10.R, not a "thin-medium approximation" | **full** — it remains the S0 kernel and 10/B-01 asserts `propagate_probe` reproduces it bit-for-bit |
| `rydsim.eit.chi_si`, `chi_ladder`, `doppler_average`, `resonance_refined_vgrid` | unchanged; `doppler_average` gains an optional `v_grid=` pass-through already present, used by V-FREEZE | **full** |
| `rydsim.eit.dipole_from_linewidth` | unchanged, but **must not be used to build an OD**: it returns the *cycling* dipole. `optical_depth_weak` is the OD path (R10-10) | full; a deprecation note, not a removal |
| `LadderConfig.max_optical_depth` | **retained**, relabelled in-place as a *numerical-conditioning* knob, not physics (A-1). Default raised only after 10/B-35 fixes the estimator | **full** — existing configs keep working and keep refusing |
| `LadderConfig` | **new optional fields**, all defaulting to today's behaviour: `detector_floor_w: float \| None = None`, `enable_collisional: bool = False`, `propagation: Literal["auto","s0","off"] = "auto"` | **full** — `propagation="off"` is exactly the current code path |
| `experiment.superhet_transfer` | OD routed through `optical_depth_weak`; branch chosen by `propagation=`; the existing `IntegrityError` gate is retained and joined by GATE-R | **behaviour-compatible for `propagation="off"`**; the OD *number* changes (it was wrong — R10-10), which is a **fix, and a release-note item** |
| `spectroscopy`, `superhet`, `objective`, `designer` | unchanged; they consume `PropagationResult.transmission()` | full |

**The compatibility contract, stated as a test (10/B-27):** for any configuration satisfying the
linear-response gate with an undepleted coupling, `propagate_probe(...).ln_transmission` and
`log(eit.transmission(chi_si(...), L, λ_p))` agree to `≤ 1e-12` relative. **The already-validated
thin-cell chain is therefore a complete oracle for the new solver over a whole branch** — the
cheapest high-value regression available, and the reason R10-3 makes it gating.

---

## 8. Validation benchmarks

**Namespace.** One namespace, `10/B-nn` (R10-15). The four drafts proposed **73 rows** between
them; after de-duplication and the two drops of R10-23 this table carries **55**:
**41 GATING** (must pass for release) and **14 DIAGNOSTIC** (failure triggers review, never a
tolerance edit). The gating count is high because most gating rows are cheap exact identities and
convention locks; only 10/B-05, 10/B-06/06b, 10/B-07, 10/B-08, 10/B-14, 10/B-18, 10/B-30, 10/B-35
and 10/B-39 are expensive or physically discriminating.

**Four gating rows are gating-on-port.** 10/B-02, 10/B-24, 10/B-25 and 10/B-26 rest on
**SELF-MEASURED** numbers whose harnesses are not in the repo. Per audit R4 and the rule below they
**cannot gate a release until ported** to `tests/test_spec10_numerics.py`; they are marked G
because that porting is itself a release blocker, not because an unported measurement may gate.
Effective gating count at first release is therefore **37**, rising to 41 on porting.

**Class column.** **S** = *structural* — tests the z-integrator, step control and reduction limits;
a thin-cell solver passes it by construction, so it is a consistency gate and never evidence of
physics. **D** = *discriminating* — physics the thin path cannot produce; a solver that silently
reduces to old behaviour **fails**. **M** = *model problem* — an idealised homogeneously broadened
saturable absorber, a solver unit test only, never a validation of the Rydberg-EIT model (R10-22).

**Binding rules.** No expected value or tolerance may be edited to make a run pass; changes require
a spec edit with rationale (audit §3 item 38). UNVERIFIED- and SELF-MEASURED-confidence rows never
gate a release (audit §3 item 36 / R4) — a **SELF-MEASURED** row is a release blocker on its own
harness until ported to `tests/`. Every `raise` benchmark carries a **mutation test** proving the
raise fires: *a refusal that never fires is not a refusal.*

### 8.1 Tier A — reduction, exactness and the z-integrator

| ID | Gate | Class | Quantity / setup | Expected | Tolerance | Source | Confidence |
|---|---|---|---|---|---|---|---|
| **10/B-01** | **G** | S | **THE REDUCTION TEST.** Branch S0 (linear-response gate satisfied, coupling declared undepleted, uniform medium): solver `ln T` and `φ` vs `rydsim.eit.doppler_average → chi_si → transmission`, at **every** OD ∈ {1e-6, 1e-3, 0.1, 1, 5, 14.216, 20, 50, 500} — **not merely as OD → 0** | identical: `ln T = −k_p L Im χ̄`, `φ = ½k_p L Re χ̄` | **rel ≤ 1e-12 in `ln T`** (branch S0, exact); **rel ≤ 1e-6** on the z-stepped path under `\|ΔOD\| ≤ 0.05` (R10-4) | Theorem 10.R; NJP **25**, 035001 Eq. (36) fetched; measured 2.8e-14 (D1) and 1.36e-15 (D2) | VERIFIED (identity + measured). **Run first, fail loudly; gates every other row in the file.** |
| 10/B-01b | G | S | Same at extreme depth, compared in `ln T` | `OD = 200` reproduced (`T = 1e-87`) | rel ≤ 1e-12 in OD | underflow-safe restatement | VERIFIED (identity) |
| 10/B-01c | G | S | **Step-control necessity canary:** fixed 16-step RK4 at OD = 20 | rel error ≈ **2.09 (209 %)** — must be **detected and refused**, never returned | boolean (refusal fires) | measured (D4) | VERIFIED-COMPUTED |
| 10/B-02 | G | S | z-integrator order by step halving, OD ∈ {5, 14.216, 20}; and the RK4 law at OD = 20, `ΔOD` ∈ {0.4, 0.2, 0.1, 0.04} | order **4.00–4.07**; errors 1.576e-4 / 9.058e-6 / 5.430e-7 / 1.356e-8 | order ∈ [3.8, 4.2]; constants ≤ 15 % | measured (D2, D3, D4) | SELF-MEASURED — **port before release** |
| 10/B-03 | G | S | **OD-convention identity:** RydSim `OD = k_p Im χ L` vs published `OD = 2α₀L`, `α₀ = k_pNμ²/(2ħε₀γ₁₃)`, two-level on resonance | algebraically identical | rel ≤ 1e-14 | NJP **25**, 035001 Eq. (12), **fetched** | **VERIFIED. Convention lock — run first.** |
| 10/B-04 | G | S | **Exponent isolation (factor-2 / sign):** constant `χ̄ = iχ″`, then constant real `χ̄ = χ′` | `ln T = −k_p χ″ L`; and `ln T = 0` with `φ = ½k_p L χ′` | rel ≤ 1e-14 | §6.4-P4; spec 05 §2.f; Häupl Eq. (11) corroboration | VERIFIED (identity) |
| **10/B-05** | **G** | **D**, M | **THE DISCRIMINATING REDUCTION.** `(T_prop − T_thin)/OD_lin²` at `s_sat` = 0.25/1/3/10, `OD_lin = 1e-4`, with `T_thin = exp(−x/(1+s))` | **−0.06400 / −0.06250 / −0.023438 / −0.0037566**, i.e. exactly `−s/(2(1+s)³)` | rel ≤ 1e-3 | derived by series inversion of 10/B-06's implicit law; verified at 4 values of `s` (D4) | VERIFIED-COMPUTED. **This is the row that catches silent reduction to the old behaviour** — a solver "correct by accident" as OD → 0 (e.g. one that calls the thin path below a threshold) fails, because the ratio at finite `x` will not track the coefficient |
| 10/B-06 | G | D, M | Saturable absorber, `α₀L = 3`: `T` vs `ln T + s(T−1) = −α₀L` (Lambert-W) at `s` = 1e-6/0.1/1/10 | 0.0497871157 / 0.0547229392 / 0.1200282390 / 0.7312939746 | rel ≤ 1e-9 | derived + RK4-verified to 1.5e-12 (D4) | VERIFIED-COMPUTED (law); LITERATURE-RECALL (the name "LambertW model") |
| 10/B-06b | G | D, M | Discrimination margin of 10/B-06 vs Beer–Lambert at `s` = 1 and 10 | **58.5 %** and **93.2 %** | measured margin must exceed 10 % (else the solver is thin) | measured (D4) | VERIFIED-COMPUTED |
| 10/B-14 | G | D | Probe-saturation self-healing: monotonicity of `s_sat(z)` and approach of the saturable-χ `T` to linear response, `s_sat(0) ∈ [1e-3, 1e-1]`. **Evaluated on the EIT resonance, not on line centre** (§4.8) | `s_sat(z)` monotone ↓; `T` error `∝ s_sat(0)` | log–log slope 1.0 ± 0.15 | §4.8; `Ω_p ≪ Ω_c` VERIFIED from NJP 2023 | DERIVED-IN-SPEC |
| 10/B-27 | G | S | Branch S vs branch N cross-solver inside the overlap region | identical OD | ≤ 1e-9 absolute in OD | method-A-vs-B, mirroring 06/B-1 | VERIFIED (by construction) |

### 8.2 Tier B — coupling, geometry and transport

| ID | Gate | Class | Quantity / setup | Expected | Tolerance | Source | Confidence |
|---|---|---|---|---|---|---|---|
| 10/B-07 | G | D | **Photon-flux (Manley–Rowe) closure**, BVP solved with `χ_c` retained | `\|ΔΦ_c\| = N∫ρ_rr(Γ_r+γ_t)dz` **and** `≤ \|ΔΦ_p\|` | equality rel ≤ 1e-6; **inequality exact — a violation is a solver bug, not a tolerance question** | Eq. (10.6) | DERIVED-IN-SPEC. Also the detector for the `σ_re`-vs-`σ_er*` conjugation hazard of (10.3) |
| 10/B-08 | G | D | **Coupling-depletion bound + measured certificate.** ≥ 5 fixtures, `OD_p ∈ [0.1, 30]`; and `P_p = 1 µW`, `P_c = 30 mW`, `T→0` | bound `2.71e-5`; measured `\|δOD\|` from one sweep ≤ `OD_p·s_c·OD_c/2` on **every** fixture; ≥ half the bound when the two-photon channel dominates | one-sided; **any violation FAILS** (a violated bound is a wrong bound) | Eqs. (10.8)–(10.9); §6.2 gate G | DERIVED-IN-SPEC |
| 10/B-09 | G | S | Coupling depletion `1 − P_c(L)/P_c(0)` **reported** for every published fixture | < 1 % (else warn and require depleted-coupling mode) | 1 % absolute | spec 05 §2.f verbatim | VERIFIED (policy) |
| 10/B-10 | G | S | **Relaxation positive control then convergence.** Fixture engineered to `OD_c ≈ 0.3`; and BVP-vs-IVP degeneracy with GATE-C passing by 100× | `r_1 > ε_relax` **first** (the correction is nonzero), *then* `r_m ≤ 1e-6` within 5 sweeps with `q_m ≤ 0.5`; BVP and IVP spectra agree | as stated; degeneracy ≤ the 10/B-08 bound | §6.2, §6.4-P9 | NORMATIVE. **Asserts the correction is nonzero before asserting it is small** |
| 10/B-11 | D | D | Counter- vs co-propagating asymmetry at identical total depletion | counter-prop transmitted spectrum perturbed **less** | boolean sign test | §4.6 geometric argument | DERIVED-IN-SPEC — **exists to REFUTE §4.6. A failure is a spec edit, not a code fix.** |
| 10/B-12 | G | S | Counter-propagating sign, `Im χ̄_c > 0` | `\|Ω_c(z)\|` monotone ↑ in z; `\|Ω_c(0)\|/\|Ω_c(L)\| = e^{−OD_c/2}` | monotonicity always-on (boolean); ratio ≤ 1e-10 | §6.4-P3 | VERIFIED (derivation) |
| 10/B-13 | D | D | Local-response error vs the velocity-resolved transport BVP, OD ∈ {1, 10, 100} | error `∝ ε_transp²` at the EIT peak, `∝ ε_transp` on the wings; magnitude ≈ 8e-2 at OD = 100 | fitted exponents 2.0 ± 0.3 / 1.0 ± 0.2; magnitude within 2× | Eq. (10.12); RMP **85**, 941 Eq. (14) fetched | DERIVED-IN-SPEC — **exists to REFUTE §4.7's odd-in-`v` cancellation claim** |
| 10/B-15 | G | S | **Radiation-trapping bound** and its fence, `s_sat = 0.01`, 300 K | `γ_trap ≤ 5.13e3 rad/s = 2π×0.82 kHz = 2.1 % of γ_gr`; fence fires above `0.1 γ_gr` | bound exact (an inequality, not a fit); fence boolean | Eq. (10.16); PRL **87**, 133601 fetched | DERIVED-IN-SPEC (bound); the escape factor that would make it an equality is **MISSING** |
| 10/B-18 | G | D | **Radial × axial ordering.** `⟨exp(−OD(r))⟩` vs `exp(−⟨OD(r)⟩)` at `OD_peak = 5`, 12-node Gauss–Laguerre; **and** at Sedlacek's waist ratio `w₀p²/w₀c² = 56.25`, `s_sat = 1.956` | the two must **differ** — the test detects that propagation has to happen inside the radial quadrature; implemented order is radial-average-of-propagated-shells | must differ by > 1 % (magnitude at the Sedlacek ratio **MISSING** — measure once the solver exists) | §4.10; spec 05 §2.g | VERIFIED (structure) / MISSING (magnitude) |
| 10/B-19 | G | S | Radial `⟨Ω_c²⟩` identity, equal waists, 24-node Gauss–Laguerre | `⟨Ω_c²⟩/Ω_c0² = 1/2` exactly (`Ω_c,eff = 0.70711×` on-axis); general `1/(1+(w₀p/w₀c)²)` — measured 0.800 at ratio 0.5, 0.200 at 2.0 | ≤ 1e-9 | Eq. (10.17) | VERIFIED (identity + numeric) |
| 10/B-20 | G | S | Self-lensing fence, `OD = 50` with 20 % radial `χ` variation | `Δφ_lens ≈ 5 rad` ⇒ the 1 rad refusal fires | boolean | Eq. (10.18) | DERIVED-IN-SPEC (thresholds are declared engineering judgement) |
| 10/B-21 | D | S | Axial vs transverse traversal ratio, `L = 5 cm`, `w₀ = 1 mm`, 300 K | **62.7** (295.1 µs vs 4.71 µs) | rel ≤ 1 % | §4.6 | DERIVED (recomputed) |
| 10/B-31 | G | S | **Corpus geometry gate.** `D/λ_rf` and `2z_R/L` computed and stamped for every spec-09 corpus entry | `D_long/λ_rf` = 4.75 (E1) / 4.26, 26.2 (E2) / 1.16 (E3) / 2.67 (E6) / 0.17, 6.67 (E7) — **all violate `<0.1` by 2×–262×**; `2z_R/L` = 15.1, **0.44** (E1 probe, coupling) / **0.17** (E2 probe) / 107, 247 (E3) / **0.44, 0.22** (E6) | must match to 1 %; `2z_R/L < 1` **refuses** | Fan *et al.* PRApplied **4**, 044015 fetched; Gaussian optics; spec 09 §3.5 | VERIFIED. Two transverse dimensions (E6, E7) are inferred from cell type — margins are far too large for that to change the conclusion, but those two must not be quoted to two figures |

### 8.3 Tier C — absolute anchors against the spec-05 Doppler reference cell

This is the one place the corpus has an experimentally validated absolute model. **What Siddons,
Adams, Ge & Hughes, *J. Phys. B* **41**, 155004 (2008) actually publishes (VERIFIED, fetched):** a
7.5 cm natural-Rb cell, 16.5/25.0/25.4/36.6 °C, weak probe `I/I_sat = 0.002`, transmissions
"ranging from 5 to 95 %", and **"an rms error better than 0.2 % for the D₂ line at 16.5 °C"**.
**What it does NOT publish:** tabulated transmission minima. **Spec 05's B9 table is therefore the
project's own model output, correctly tagged V-computed — it is not a transcription of Siddons
numbers, and any release note claiming otherwise would be a fabrication.** Siddons' contribution is
the *method's* 0.2 % rms experimental validation, which is what licenses these rows as absolute
anchors. All rows below were **independently re-implemented from scratch** during draft authoring
(Voigt sum over both isotopes, `S_FF'` from `rydsim.angular`, Steck rev 2.3.4 hyperfine from
`rydsim.atom`, densities from `rydsim.cell`, **no call to any RydSim propagation routine**) and
reproduce spec 05 to < 0.15 %.

| ID | Gate | Class | Quantity / setup | Expected | Tolerance | Source | Confidence |
|---|---|---|---|---|---|---|---|
| 10/B-17a | G | S | Nat-Rb, 75.0 mm, 25.0 °C, weak probe: OD / `T_min`, ⁸⁵Rb F=3 | 1.2982 / 0.273031 | ±1 % (physical band ±6 %) | spec 05 B9a; recomputed | VERIFIED-COMPUTED |
| 10/B-17b | G | S | same cell: ⁸⁷Rb F=2 | 0.4805 / 0.618489 | ±1 % | spec 05 B9b; recomputed | VERIFIED-COMPUTED |
| 10/B-17c | G | S | same cell: ⁸⁵Rb F=2 / ⁸⁷Rb F=1 OD | 0.9558 / 0.3124 (`T_min` 0.384523 / 0.731713) | ±1 % | spec 05 §2.f; recomputed | VERIFIED-COMPUTED |
| 10/B-17d | G | S | Nat-Rb 75.0 mm, 16.5 °C: `T_min` (OD), ⁸⁵Rb F=3 | 0.594036 (0.5208) | ±1 % | spec 05 B9d + Siddons | VERIFIED-COMPUTED (model) / VERIFIED (0.2 % rms claim) |
| 10/B-17e | G | S | dip positions vs the ⁸⁷Rb D2 centroid at 25.0 °C | −2.4241 / −1.2883 / +1.6195 / +4.0942 GHz | ±3 MHz **at 25 °C only** | spec 05 B9c; recomputed | VERIFIED-COMPUTED |
| 10/B-17f | D | S | dip-position temperature drift (⁸⁷Rb F=2), 16.5 → 65 °C | −2.4231 → −2.4280 GHz (**4.9 MHz span** vs B9c's ±3 MHz) | ±1 MHz | recomputed; motivates amendment A-8 | VERIFIED-COMPUTED |
| **10/B-17g** | **G** | S | **Nat-Rb 75.0 mm, 50.0 °C: OD / `T_min`, ⁸⁵Rb F=3 — 2.8× above the engine's present refusal ceiling** | **14.2160 / 6.69992e-7** | OD ±1 %; `T` rel ≤ 1e-3 | spec 05 §2.f; recomputed | VERIFIED-COMPUTED. The gating high-OD stability row; by Theorem 10.R its correct answer is exactly `exp(−14.216)` |
| **10/B-35** | **G** | **D** | **OD-ESTIMATOR ADJUDICATION (R10-10).** Nat-Rb, 75 mm, 25.0 °C, `Ω_c = 0`, ⁸⁷Rb F=2, via `propagate.optical_depth_weak` | must reproduce **0.4805** (= 10/B-17b). Recorded for the record: shipped `superhet_transfer` chain gives **1.3874** (factor **2.89**); D1's partial correction gives 0.578 (still **1.20×** high) | ±1 % against 10/B-17b; the `d_eff,far`/cycling factor `2/3` in `d²` must reproduce to ≤ 1e-4 and `p_F` exactly | spec 00 §6 gap 7; spec 05 §2.b/§2.f | VERIFIED-COMPUTED. **Closes the 2.40× / 2.67× / 2.89× spread between the drafts. No OD number is quoted until this passes.** |

### 8.4 Tier D — published thick-cell rows, and what is actually checkable

| ID | Gate | Class | Quantity / setup | Expected | Tolerance | Source | Confidence |
|---|---|---|---|---|---|---|---|
| **10/B-39** | **G** | **D** | **JING DISCRIMINATOR (bracket direction).** Cs, 5 cm, `N₀ = 4.89e10 cm⁻³`, `Ω_p = 2π·5.7 MHz`, `Ω_c = 2π·0.97 MHz` | `T_solver ≥ 3 × T_weak-probe`, `T_weak = 0.00966` (OD 4.639) — the thin path predicts 1.16 µW against a published ~10 µW, **8.6× too low** | strict inequality | Jing *et al.*, *Nat. Phys.* **16**, 911 (2020), ar5iv:1902.11063 fetched; OD computed | VERIFIED (inputs) / VERIFIED-COMPUTED (bound). **The only published end-to-end thick-cell transmission pair in the flagship record** |
| 10/B-40 | D | D | Jing absolute: transmitted probe power from 120 µW incident | ~10 µW (`T = 0.0833`, `OD_eff = 2.485`) | factor-2 window [5, 20] µW | ibid., verbatim published pair | VERIFIED (published). Diagnostic only: the pair bundles uncoated-borosilicate window loss (~8–15 %), EIT/AT transparency at the operating point, and hyperfine optical pumping |
| 10/B-42 | D | D | Sedlacek 2012 max probe-transmission increase; 7.5 cm effective, OD 0.4805, `s_sat` 1.956, `w_p/w_c` = 7.5 (area dilution **1.78 %**) | ~4.5 % | ORDER (factor 3) | ar5iv:1205.4461 fetched | VERIFIED (published, figure-caption level); amplitude model is tens-of-% by spec 06 §7.1. **Nothing here may be tuned to Sedlacek's printed `℘` — R-22 records it as an unresolved tension** |
| 10/B-43 | D | D | Mohapatra 2007 probe-transmission change: n = 45 (`Ω_c = 2π·3.5 MHz`) vs n = 80 (`Ω_c = 2π·1.5 MHz`), same 75 mm cell | 5 % and 1 % → **ratio 5.0** | absolutes ORDER (factor 3); **ratio factor 2** | ar5iv:quant-ph/0612200 fetched | VERIFIED (published). Upgrades spec 06 B-12 from 3 booleans to a quantitative ratio largely free of amplitude-model error |
| 10/B-44 | D | S | Mohapatra EIT linewidth band across their power/transition range | 22–44 MHz | simulated width must fall inside | ibid., verbatim | VERIFIED |
| **10/B-22** | **D** | **D** | **Su 2022 EIT peak height vs optical depth** — the only quantitative published OD scan found. Rb, `L = 71.8 mm`(?), `Ω_p/2π = 30 MHz`, `Ω_c/2π = 0.38 MHz`, `B ≈ 0`, 27→65 °C | max peak height **13 %** at an **interior** optimum near 51 °C; EIT linewidth ≈ 10 MHz; engine returns a graded result across their `α = 0.42 → 5.0` | height ORDER (factor 2); **the optimum must be interior, not at the hottest point** | *Opt. Express* **30**, 1499 (2022), ar5iv:2111.13408 fetched | VERIFIED (published); **cell length LITERATURE-RECALL (vendor snippet, direct fetch failed)** |
| **10/B-45** | **D** | **D** | **Su fitted-α tension — length- and enrichment-INDEPENDENT.** `α(65 °C)/α(27 °C)` | published **11.90**; spec-05 weak-probe model **30.19** (absolutes 0.42 vs 0.566; 5.0 vs 17.10) | the **2.54×** discrepancy must be **EXPLAINED** — saturation-along-z vs cold-spot calibration — never fitted away | ibid. + in-session model | VERIFIED (published pair) / VERIFIED-COMPUTED (model). Both hypotheses are live: the spec-05 density model runs at 11 %/K, so a ~10 K cold-spot offset at 65 °C with none at 27 °C reproduces the gap. **RydSim's job is to discriminate, not to assume** |

**And the sentence that justifies this module:** Su's *published optimum operating point* — 51 °C,
13 % peak height — sits at a computed weak-probe optical depth of **5.52**, i.e. just above the
depth at which `rydsim.experiment` presently refuses to answer. A ceiling that excludes the
literature optimum is a capability gap, not a safety fence.

### 8.5 Tier E — optima, fences, convergence and the strategic test

| ID | Gate | Class | Quantity / setup | Expected | Tolerance | Source | Confidence |
|---|---|---|---|---|---|---|---|
| **10/B-29** | **G** | S | **Optical-depth optimum by dominant noise term** (analytic `NEF(OD)`) | `OD* = 2` **exactly** (shot noise, numeric argmin 1.999996); `OD* = 1` **exactly** (detector NEP); **no interior optimum** (RIN, `NEF ∝ 1/OD`). Penalties `NEF(2)/NEF(0.1) = 0.1293`, `/NEF(1) = 0.8244`, `/NEF(5) = 0.5578` | `OD*` rel ≤ 1e-6; ratios ±1 % | closed forms derived independently by D1 and D4, both numerically confirmed | VERIFIED-COMPUTED. **The strongest no-lab-legitimate claim in the document** — the optimum's location is an integer, independent of every uncalibrated absolute parameter |
| 10/B-37 | G | S | EIT-contrast-optimal background depth `OD_bg* = ln r/(1 − 1/r)`, `r` ∈ {2, 5, 10, 100} | 1.38629 / 2.01180 / 2.55843 / 4.65169; crosses 5.0 at `r = 143.3`, and `→ 1` as `r → 1⁺` | rel ≤ 1e-3 | derived + verified (2e6-point scan) | VERIFIED-COMPUTED. The optimum is **always** above spec 05's `OD ≤ 0.1` gate |
| 10/B-38 | G | S | Transparency-window FWHM ratio per doubling of OD | **√2 = 1.414214** (measured 1.41421–1.41422 over six octaves) | ±1 % | derived + verified; **RMP 77, 633 (2005) deliberately NOT relied upon (not fetched)** | VERIFIED-COMPUTED. Must stamp the width convention — `2√(ln2/a₂L)` vs `2√(2ln2/a₂L)` differ by √2 (§6.4-P14) |
| **10/B-48** | **G** | S | **AT position invariance vs fitted-splitting drift.** Doppler Rb, `Ω_RF/2π = 20 MHz`, OD ∈ {0.1, 1, 5, 20} | **true** stationary points of `T(Δ_p)` coincide with those of `Im χ` at every OD (since `exp` is strictly monotone), so the true splitting is propagation-invariant; the **Lorentzian-fitted** splitting is not | positions rel ≤ 1e-9 (**gating**); fitted-splitting drift **reported as data, not asserted** (diagnostic) | §1.4(c) theorem; spec 06 §2.7 extraction algorithm | VERIFIED (theorem); drift is a measurement. **The one place OD leaks into a frequency observable — an algorithm systematic, not a physics one** |
| **10/B-30** | **G** | **D** | **THE STRATEGIC TEST (R10-24).** D3 `Ω_c` sweep re-run through the propagation solver at `Ω_p/Γ_e ∈ {→0, 0.1, 0.5, 1.0}` | `α(Ω_p→0)` **bit-identical** to the thin-cell value (a consequence of 10/B-01; a discrepancy is a solver bug). `\|Δα\|` at finite `Ω_p` is the measured strategic result | **`\|Δα\| < 0.02` ⇒ scale only, `d3-trade-law-v2` stands. `> 0.05` ⇒ FAIL-open: structural, the finding is retracted and re-derived. In between ⇒ ships with a declared regime bound** | §1.4.1; baseline `findings/d3-trade-law-v2-6f7f20848dca40a1` (NEF U-shaped in `Ω_c`, `α < 0.15` above the optimum) | self-check; **the decision rule is NORMATIVE, the outcome is genuinely unknown until run.** Run it **before** the solver is considered done — a FAIL changes what the solver is for |
| **10/B-34** | **G** | S | **Density ceiling fires (GATE-D).** Rb, `L = 5 cm`, T swept 20 → 140 °C | `ThickCellError` for `N_total > 1.0e12 cm⁻³` (≈75 °C at 5 cm); mandatory validity flag (not a raise) for `3.4e11 < N ≤ 1e12`; clean pass below. Anchor table `T` = 33.9 / 59.9 / 91.8 / 130.6 °C for OD = 1 / 10 / 100 / 1000 | exact threshold, no tolerance | §4.11, R10-6 | coefficients VERIFIED-in-repo; **the ceiling VALUE is a declared normative judgement (lock #20)** |
| **10/B-46** | **G** | S | **Every fence fires, and is mutation-tested.** GATE-P/C/T/L/D/R/G; Gauss–Hermite as sole velocity quadrature raises (R-2); non-converged grid returns `converged: False` **as data**; `ThickCellError` exists | all fire | boolean, each with a mutation test | audit §3 items 17–19, 21, 36, 38; R-2 | VERIFIED (policy). **Measured spec-vs-code divergence: `ThickCellError` does not exist in `src/rydsim`, and no `I_p > 0.01 I_sat` gate exists anywhere on the physics path.** Both drafts confirmed this independently by grep/AST |
| 10/B-23 | G | S | **Z-CONV fires.** `GL_n` vs node-disjoint `GL_2n`, plus a deliberately under-resolved mutant | `max_Δ \|ΔOD\| ≤ ε_z = 1e-4`; mutant **raises** | ε_z = 1e-4 absolute in OD | §6.1 | NORMATIVE |
| 10/B-24 | G | S | **`Ω_c` interpolant, and that V-FREEZE matters.** `M = 16`, Rb-87 fixture | frozen grid ≤ 1e-8 (**measured 2.473e-9**) **and** rebuilt grid ≥ 1e-7 (**measured 6.365e-7**) | as stated | §6.3 | SELF-MEASURED — **port before release.** The rule is *proven to matter*, not merely asserted |
| 10/B-24b | D | S | 2-D `(Ω_c, Ω_RF)` tensor interpolant, `M_c = M_RF = 16` | ≤ 1e-6 | `err_interp` budget | §6.3 | **UNMEASURED — must be measured, never assumed.** Blocks any z-varying `Ω_RF` run |
| 10/B-25 | G | S | High-OD representation, `OD = 750` requested | returns `ln T = −750`, `underflow=True`; **never a bare `0.0`** | exact | §6.4-P2 | SELF-MEASURED (thresholds) |
| 10/B-26 | G | S | Per-detuning z-grids required: a shared grid sized in the wings vs per-detuning | shared grid fails Z-CONV by ≫ `ε_z`, given the measured 4.40e3 dynamic range | QUALITATIVE predicate | §6.1 | SELF-MEASURED |
| 10/B-28 | G | S | **Provenance completeness.** Any propagation run | record carries every §7.3 item: `branch`, `converged`, `max_delta_od`, rule, node counts, relaxation trace `[(m,r_m,q_m)]`, `err_interp`, `OD_p`, `OD_c`, `s_c`, all seven gate values, `underflow`, `s_sat` at entrance and exit, the `d_eff,far`/`p_F` factors applied, `N_total`, `N_r`, `Δν_vdW`, `Γ_self`, `Δν_Fermi` and whether they were enabled, `2z_R/L` per beam, `D/λ_rf` | all present | audit §4 items 1–12 | NORMATIVE |
| 10/B-32 | D | D | RF-inhomogeneity z-weighting: linear `E_RF(z)` ramp ±10 %, OD ∈ {0, 5} | the inverted field shifts by the `exp(−OD(z))`-weighted mean (front-of-cell weighting), **not** the unweighted mean | rel ≤ 2 % vs the analytic weight | §4 / spec 06 §2.8 item 7 | derived; self-checking. **No corpus entry publishes an `E_RF` map** |
| 10/B-33 | D | S | Screening decoupling: `T(f)` with and without a z-resolved `P_c(z)` | identical while coupling depletion < 1 % | rel ≤ 1e-6 | ruling R-7; §4.5 | VERIFIED (by construction, conditional on 10/B-09) |
| 10/B-47 | D | S | Kramers–Kronig closure of the `(Re χ, Im χ)` the solver accumulates, over the scanned band | consistent | ≤ 1 % (band-truncation-limited) | standard KK | LITERATURE-RECALL. Catches a solver that propagates `\|Ω_p\|` but drops the phase |

### 8.6 Draft-ID provenance map

| Unified | D1 (`scope`) | D2 (`theory`) | D3 (`numeric`) | D4 (`validat`) |
|---|---|---|---|---|
| 10/B-01, 01b, 01c | S-1 | 10/P-1, 1b | 10/N-1 | P-1, P-1b |
| 10/B-02 | — | 10/P-2 | 10/N-6 | P-4 |
| 10/B-03, 04 | — | 10/P-10b | 10/N-3 | — |
| 10/B-05 | S-2 | — | 10/N-2 | P-2 |
| 10/B-06, 06b | — | — | — | P-3, P-3b |
| 10/B-07, 08, 09 | (R-P2) | 10/P-3, P-4 | 10/N-10 | P-17 |
| 10/B-10, 11, 12 | — | 10/P-5, P-6 | 10/N-4, N-9 | — |
| 10/B-13 | — | 10/P-7 | — | — |
| 10/B-14, 15 | — | 10/P-8, P-9 | — | — |
| 10/B-17a–g, 35 | S-11 | 10/P-10, P-11 | — | P-5a–f, P-6 |
| 10/B-18, 19 | S-5 | 10/P-12 | — | P-18 |
| 10/B-20, 21 | — | 10/P-13, P-14 | — | — |
| 10/B-22, 45 | — | 10/P-15 | — | P-14, P-15 |
| 10/B-23–28 | (R-P5) | — | 10/N-5, 7, 7b, 8, 11, 12, 13 | P-20 |
| 10/B-29, 37, 38 | S-4 | — | — | P-7, P-8, P-9 |
| 10/B-30 | **S-6** | — | — | — |
| 10/B-31, 32, 33, 34 | S-7, S-8, S-9, S-10 | — | — | — |
| 10/B-39, 40, 42, 43, 44 | — | — | — | P-10, P-11, P-12, P-13, P-13b |
| 10/B-46 | (R-P1…P-12) | — | — | P-16, P-16b |
| 10/B-47, 48 | S-3 | — | — | P-19 |
| **dropped** | **S-12** (duplicate of spec 05 B3e) | — | — | **P-11b** (a fixture adjudication, not a solver test — R10-20) |

---

## 9. Refuse to guess — the conditions under which `rydsim.propagate` raises

All raise `rydsim.propagate.ThickCellError(IntegrityError)` carrying the gate name, the physics
reason, the offending value and the threshold. Each is a **computed** quantity, never an
assumption, and each is **mutation-tested** (10/B-46): a refusal that never fires is not a refusal.

| ID | Gate | Condition | Why refusing beats answering |
|---|---|---|---|
| **R-P1** | GATE-G | `2z_R/L < 1` for **either** beam | the 1-D model's own precondition is violated — the beam changes radius by > √2 inside the cell. Warn for `1 ≤ 2z_R/L < 10` with the exit/waist intensity ratio attached. **Fires today for E1-coupling, E2-probe, E6-probe, E6-coupling** |
| **R-P2** | GATE-C | measured coupling depletion over the cell **> 1 %** | §4.5's premise fails; `Ω_c(z)` and `τ_s,eff(z)` both become z-dependent and the ladder is not the modelled one |
| **R-P3** | GATE-P (i) | `Ω_p ≥ 0.01·min(Γ_e, \|Ω_c\|)` on the **analytic linear-response** path | integrity-audit refusal #21, **currently unimplemented**. The z-coupled Lindblad path is the only route above it — and it, not the weak-probe path, is what the corpus needs |
| **R-P3b** | GATE-P (ii) | `s_sat(0) = 2Ω_p²/Γ_e² > 0.01` **with radiation trapping and coupling depletion left unmodelled** | above it `γ_trap` is no longer bounded at 2.1 % of `γ_gr` (it reaches 513 % at 100 µW) and the Holstein escape factor is MISSING — the neglect is no longer justified by a number |
| **R-P4** | — | on-axis-only evaluation requested for any **amplitude** observable (contrast, `κ_E`, NEF, EIT width) at OD > 0.1 | radial and longitudinal averaging do not commute; the error is 29.3 % in `Ω_c,eff` and **2× in EIT width**. Frequency observables are exempt (§1.4c) |
| **R-P5** | — | radial quadrature unconverged (12 vs 32 Gauss–Laguerre differ by > 1e-5), **or** Z-CONV `> ε_z = 1e-4`, **or** `err_interp > 1e-6`, **or** velocity-grid halving `> 1e-4` | convergence ships as **data** (`converged: bool` + magnitude), never as a docstring (audit §4 item 6). Refuse when the *answer* is not converged, not when the OD is large |
| **R-P6** | GATE-D | `N_total > 1.0×10¹² cm⁻³` (R10-6) | three sourced coincidences: spec 05 §7.2's D2 self-broadening coefficient is MISSING above it; the dominant Rydberg linewidth becomes the UNVERIFIED Fermi surrogate; and charge-induced bistability is *observed* at ≈2.5×10¹² cm⁻³ while RydSim has no ionization or bistability physics |
| **R-P7** | GATE-D | `3.4×10¹¹ < N ≤ 1×10¹² cm⁻³` **without** the collisional terms enabled and their **2× uncertainty propagated** | in this band `Γ_self` reaches ~1.7 % of `Γ_e` and Fermi broadening ~50 % of the default `deph_r`; running with them off is a silently wrong linewidth |
| **R-P8** | GATE-D | **computed** `N_r = N·f_vel·ρ_rr` implies `Δν_vdW > 0.1 MHz` **or** `N_r > 1×10⁹ cm⁻³` | spec 04's own vdW warning threshold; above it the absent Rydberg–Rydberg and ionization physics is no longer a rounding error. `N_r` and `ρ_rr` must be **computed from the steady state**, never assumed |
| **R-P9** | GATE-T | `ε_transp > 0.2` (flag above 0.05) | the local-response approximation is uncontrolled; the exact treatment is a ~10⁶-dimensional BVP and is not implemented |
| **R-P10** | GATE-L | `\|Δφ_lens\| > 1 rad` (flag above 0.3) | the paraxial-shell model has failed; the medium is a graded-index lens and shells no longer propagate independently |
| **R-P11** | GATE-R | transmitted power below the caller's stated `detector_floor_w`, **or** `OD > 708.3964` requested through the bare-`T` accessor | **this is what audit CRIT-2 was actually about**: the NEF diverged because the *signal* was dead, not because OD crossed 5. The honest output is "no signal, OD = X", never a denormal that looks like a result |
| **R-P12** | — | any **absolute field** output without `field_reference="at_atoms"` and the computed `D/λ_rf` in the result object | every corpus entry violates Fan *et al.*'s `D/λ_rf < 0.1` by 2×–262× and the induced `\|E_int/E_inc\|` is **MISSING**; a bare "E = … V/m" is a claim the model cannot support |
| **R-P13** | — | buffer-gas cells, wall-coated cells | audit refusal #16, unchanged and inherited: transit and collision models are invalid and must not be extrapolated. A buffer gas also turns the ballistic transport of §4.7 into diffusion, changing `ε_transp` qualitatively |
| **R-P14** | — | vapour temperature outside 298–550 K **and** an absolute (not relative) OD claim | audit refusal #17 escalated: warn-only extrapolation is acceptable for a relative sweep, not for an absolute optical depth that gates a refusal |
| **R-P15** | — | Gauss–Hermite as the sole velocity quadrature for any EIT/AT spectrum | ruling R-2, unchanged and inherited at every z level |
| **R-P16** | — | relaxation `q_m ≥ 1` twice consecutively, or `m > 20`; `solve_bvp` status 1 or 2 | a diverging relaxation means the physical feedback loop has gain ≥ 1 — the medium is bistable or self-focusing. **Do not return the last iterate**; that is a physics statement the caller must see |
| **R-P17** | — | the z-solver returning a result while the propagation path was never exercised — it must return the closed form and **stamp `branch="S0_closed_form"`** | R10-13: a no-op dressed as a computation is the "plausible but wrong" hazard in its purest form |

**Warn-only (in `validity_flags`, never silent):** `1 ≤ 2z_R/L < 10`; `0.1 < OD`; `ε_transp > 0.05`;
`\|Δφ_lens\| > 0.3 rad`; `D/λ_rf ≥ 0.1` (**always true today** — the flag is the honest default, not
an exception); `N > 3.4×10¹¹ cm⁻³`; screening-uncalibrated; Rb-85 ±40 MHz systematic;
vapour-T extrapolation.

---

## 10. Sourcing grade — what was fetched, what is recall, and what could ship a wrong number

### 10.1 Consolidated confidence table

| Claim / equation | Grade | Basis |
|---|---|---|
| `dÊ/dz = +i(k/2)χÊ`, both fields, thermal ladder | **VERIFIED** | Ogden *et al.* PRL **123**, 243604 Eq. (6), full text fetched and quoted |
| Doppler-averaged polarization source | **VERIFIED** | Ogden Eq. (8), fetched |
| Theorem 10.R (single exponential, exact at any OD) | **VERIFIED** | NJP **25**, 035001 Eq. (36), fetched |
| `α₀ = k_pNμ²/(2ħε₀γ₁₃)` and `OD = 2α₀L` | **VERIFIED** | NJP **25**, 035001 Eq. (12), fetched |
| Linear-response condition `Ω_p ≪ Ω_c` | **VERIFIED** | NJP **25**, 035001, fetched |
| Phase-space transport equation | **VERIFIED** | RMP **85**, 941 Eq. (14), fetched |
| Radiation trapping raises ground-coherence decay with density | **VERIFIED (abstract)** | PRL **87**, 133601, fetched |
| `D/λ_rf < 0.1` cell criterion; Pyrex 0.066 %/mm at 12.6 GHz | **VERIFIED** | PRApplied **4**, 044015, full text fetched |
| Bistability density, Rydberg fraction, ion density, ionization cross-section | **VERIFIED** | PRA **94**, 063820 (2016), fetched |
| Foreign-gas VCC tolerance (0.01–0.02 mbar for 1 MHz; ≤1e-5 mbar seals) | **VERIFIED** | PRApplied **23**, 034028 (2025), fetched |
| Published fixtures (Sedlacek, Jing, Mohapatra, Su, Siddons parameters) | **VERIFIED** | ar5iv full texts fetched, quoted verbatim |
| SciPy `solve_ivp` / `solve_bvp` method descriptions and defaults | **VERIFIED** | SciPy reference fetched |
| CoOMBE co-propagating-only restriction; `mbe.f90` fixed-step integrator | **VERIFIED** | README + source fetched |
| Häupl *et al.* `dI/dz` factor-2 corroboration; manual slice count; per-detuning limitation; `v′_z` substitution | **VERIFIED** | arXiv:2410.19916 PDF fetched |
| Spec 05 B9 rows (10/B-17a–g) | **VERIFIED-COMPUTED** | from-scratch re-implementation reproducing spec 05 to < 0.15 %, no propagation routine called |
| `OD*` = 2 / 1 / none; `OD_bg* = ln r/(1−1/r)`; √2 window narrowing; `−s/(2(1+s)³)`; Lambert-W law | **VERIFIED-COMPUTED** | derived and numerically confirmed, twice for `OD*` (two independent drafts) |
| `⟨Ω_c²⟩ = Ω_c0²/2` and the general waist-ratio form | **VERIFIED** | analytic identity + 24-node Gauss–Laguerre |
| Reduction exactness measurements (2.8e-14, 1.36e-15, order 4.07) | **VERIFIED-COMPUTED** | two independent drafts, both reproducible from the shipped tree |

### 10.2 Equations that will produce a SHIPPED NUMBER but rest only on recall or in-spec derivation

**This is the section the project grades adversarially. Each row states the numerical self-check
that would catch an error in it.**

| # | Claim | Grade | Ships a number? | The check that would falsify it |
|---|---|---|---|---|
| **U-1** | **Rydberg–ground collisional broadening `≈ 0.5·\|shift\|`, and the Fermi shift coefficient `−9.9×10⁻⁸ Hz·cm³`** | **UNVERIFIED, 2× flag** (spec 04 §2.3.5, audit R11) | **YES — and it is the DOMINANT Rydberg linewidth term across the upper half of the usable density band** (≈50 % of `deph_r` at the ceiling, 174 % at `3.5e12 cm⁻³`) | Measured density-dependent hot-cell EIT widths (spec 04's own stated self-check). **Until then every output above `N = 3.4×10¹¹ cm⁻³` inherits a 2× uncertainty on the EIT/AT linewidth, hence on the resolvability threshold, hence on `E_min`** — and must print it. **This is the single largest recall-borne risk in the module** (MISSING item M-7) |
| **U-2** | Rb/Cs D2 self-broadening `β/2π = 1.03e-7 / 1.16e-7 Hz·cm³` | VERIFIED-in-repo (R-6), **NOT re-fetched**; and **MISSING** as a *validated* coefficient above `10¹² cm⁻³` | **YES** — the entire density-trap table scales linearly on it, and GATE-D's own threshold is derived from its validity boundary | Spec 04's S5 formula check reproduces `1.03e-7`; the regime boundary is fenced by R-P6 rather than trusted. A wrong `β` moves the fence, not a shipped spectrum |
| **U-3** | `χ_c = 2N℘_er²σ_re/(ε₀ħΩ_c)`, Eq. (10.3) | **DERIVED-IN-SPEC**, unsourced; carries a `σ_re` vs `σ_er*` conjugation hazard | YES, whenever branch S/N runs | **10/B-07**: a wrong conjugation flips `Im χ_c` and manufactures coupling **gain**, which the Manley–Rowe closure detects exactly; **10/B-12** asserts `\|Ω_c\|` monotone non-decreasing in z |
| **U-4** | Manley–Rowe depletion bound, Eqs. (10.6)–(10.9) | **DERIVED-IN-SPEC**; no published statement found | YES — it selects the IVP branch | **10/B-08**: the solved BVP depletion must never exceed the bound on any of ≥5 fixtures, and must reach ≥ half of it when the two-photon channel dominates. A violated bound is a wrong bound |
| **U-5** | Radiation-trapping bound `γ_trap ≤ σ_eff Φ_p`, Eq. (10.16) | **DERIVED-IN-SPEC**; the sub-claim that reabsorption destroys the `g–r` coherence with **unit** probability is **UNVERIFIED** | Only as a **fence**, never as a value — `g_esc` is MISSING so no trapping-corrected number may ship | The unit-probability assumption only ever *tightens* the bound, so it cannot corrupt a number. **10/B-15** asserts the bound and the fence. **The exposure is that above `s_sat ≈ 0.01` there is no estimate at all, only a bound that the corpus violates by 16–1760×** |
| **U-6** | `ε_transp = \|v\|α_p/(2γ_min)`, Eq. (10.12), and the odd-in-`v` cancellation | **DERIVED-IN-SPEC**; the algebra is exact for the reduced linear problem, but that the reduced problem is the right one to pose (it drops the population sector) is unverified | Only as a fence | **10/B-13** measures the exponents (2.0 ± 0.3 at the peak, 1.0 ± 0.2 on the wings). A linear-in-`ε` peak error falsifies §4.7 |
| **U-7** | Counter-propagation is geometrically favourable (§4.6) | **DERIVED-IN-SPEC** | No — it justifies a design preference | **10/B-11** is a boolean sign test that exists to refute it. If it fails, the paragraph is struck |
| **U-8** | `σ_D,peak = √(π ln2)(Γ_e/2π)/Δν_D · σ₀`, Eq. (10.15) | **DERIVED-IN-SPEC** | Yes, in the trapping bound | Two closures: 0.8 % against the shipped engine, 1.9 % against spec 05 B9a. **Note (R10-10.2): the 0.8 % closure is NOT independent of the dipole convention** — both sides use the cycling `σ₀ = 3λ²/2π` |
| **U-9** | Self-lensing thresholds 0.3 / 1.0 rad, Eq. (10.18) | **DERIVED-IN-SPEC**, thresholds are declared engineering judgement | Only as a fence | A split-step diffraction calculation on the shell decomposition would replace judgement with measurement. Until then the threshold is a lock-#20 spec-edit item |
| **U-10** | `ρ_rr = Ω_p²/(Ω_p²+Ω_c²)` dark-state depletion | **LITERATURE-RECALL** (textbook) | Only through the gate it motivates | It is the physical content of an already-shipped gate; **10/B-14** measures the saturation scaling directly |
| **U-11** | Inhomogeneous saturation law `α = α₀/√(1+s₀)` | **LITERATURE-RECALL**, no primary fetched | **NO — bracketing only, never grading** | It is a *frozen* law (uses the input intensity), so a correct z-resolved solver must predict **more** absorption than it does. That asymmetry is itself the check |
| **U-12** | Su *et al.* cell length 71.8 mm | **LITERATURE-RECALL** (vendor snippet; direct fetch failed) | Only in 10/B-22's **absolute** OD | **10/B-45 is chosen precisely to be length- and enrichment-independent.** Only the 65/27 ratio is graded tightly |
| **U-13** | Chebyshev `M_c = 16 → 2.5e-9`; RK4 law constant 3840; dynamic range 4.40e3; `s_c` values; `ρ_ee` scaling; float64 underflow thresholds; unit cost 0.442 s | **SELF-MEASURED**, harnesses **not in repo** | They set normative defaults | **Release blocker** until ported (audit R4). Each has an in-place mutation test specified (10/B-24 proves V-FREEZE matters; 10/B-02 refits the RK4 order) |
| **U-14** | 2-D `(Ω_c, Ω_RF)` tensor interpolant inherits the 1-D spectral rate | **UNVERIFIED / UNMEASURED** | Would, if a z-varying `Ω_RF` were run | **10/B-24b must be measured before any z-varying `Ω_RF` run.** Blocked until then |
| **U-15** | `GL_n` and `GL_2n` share no nodes | **UNVERIFIED as a theorem** | No | Replaced by a runtime assertion (no refined node within `1e-12·L` of a coarse node, else `GL_{2n+1}`). The code does not rely on the claim |
| **U-16** | Relaxation/collocation schemes for the counter-propagating BVP | **LITERATURE-RECALL** (standard numerical practice), **applied here, not taken from an EIT paper** — no published counter-propagating Rydberg-ladder Maxwell–Bloch treatment was found | Yes, in branch S/N | Contraction is **not** claimed as a theorem: `q_m` is monitored every sweep and divergence **raises** (R-P16). 10/B-10's positive control proves the correction is nonzero before proving it is small |

### 10.3 The honest bottom line on sourcing

The **structure** of this document is well sourced: the propagation equation, the reduction
theorem, the transport equation, the polarization source, the cell-geometry criterion, the
bistability density and every published fixture were fetched and quoted. The **fences** are largely
derived in-spec and each carries a named falsifier. The **one place recall reaches a shipped
number** is U-1, the Rydberg–ground collisional broadening surrogate — and it does so exactly in
the density band this module exists to reach. That is stated in §11 as the module's principal known
limitation rather than buried in a table.

---

## 11. Known limitations, MISSING items, and what would close them

1. **The transport correction is fenced, not solved.** The exact velocity-resolved axial BVP is
   specified (§4.7) but costs ~10⁶ coupled ODEs with split boundary data. `ε_transp` is a **bound
   on the error**, not a correction to the answer.
2. **Radiation trapping ships as a bound only.** `g_esc` is **MISSING**. Over the entire
   strong-probe regime the corpus occupies (`I_p/I_sat` 0.16 → 17.6) the bound exceeds `γ_gr` by
   16×–1760×, so no trapping-corrected amplitude may ship. **This is the second unbounded
   systematic in the program** (R10-16).
3. **The RF internal field is unmodelled and its error is MISSING.** Every corpus configuration
   violates `D/λ_rf < 0.1` by 2×–262×. `E = ħΩ_RF/℘` returns the field **at the atoms**, never the
   incident field. **This is the largest unbounded systematic in the program** (R10-17).
4. **Four of seven corpus beams violate the 1-D model's own precondition** (`2z_R/L < 1`). A
   paraxial 2-D `(r, z)` split-step solver is the fix and is out of scope. E1/E2/E6 amplitude
   observables are refused, not degraded.
5. **The Rydberg–ground broadening surrogate is UNVERIFIED with a 2× flag and becomes the dominant
   Rydberg linewidth in the upper half of the usable density band.** See U-1 / M-7.
6. **No four-wave mixing or backward-generated fields.** The NJP 2023 guide (fetched) states FWM
   grows with OD; **no phase-matching criterion is derived or sourced here.** This is the
   least-fenced limitation in the document, and no criterion is offered because none was found.
7. **No ionization, free-charge or bistability physics at all.** RydSim's single-valued `χ` cannot
   represent a hysteretic medium. GATE-D fences it by density; it does not model it.
8. **CW steady state only.** No pulse propagation, adiabatons, dark-state polaritons, slow-light
   storage or scan-rate transients. The group delay `τ_g = (L/2c)·ω dReχ/dω` is computable from
   Eq. (10.4) but is not validated here.
9. **Single sensed isotope/hyperfine channel in the photon bookkeeping** (§6.4-P11).
10. **`Ω_RF(z)` is an input, not a solved field.** If a cell-EM module ever supplies a z-profile,
    the 2-D interpolant is required and 10/B-24b must be measured first.
11. **SELF-MEASURED rows are not yet reproducible in-repo** and must not gate a release until their
    harnesses are ported to `tests/test_spec10_numerics.py` (audit R4). This includes the headline
    `M = 16 → 2.5e-9` figure and the RK4 error-law constant.
12. **Structural risk this corpus deliberately carries:** Tier A is entirely self-validating, Tiers
    B/C are model-internal (Siddons validates the *method* to 0.2 %, not these specific numbers),
    and only Tier D touches measured Rydberg-EIT reality — where it is thin and amplitude-dominated.
    **RydSim must not claim thick-cell Rydberg-EIT amplitude accuracy better than a factor of 2 on
    the strength of this corpus.** Frequency observables inherit spec 06's much stronger footing;
    amplitudes do not.

| # | MISSING | Blocks | What would close it |
|---|---|---|---|
| M-1 | quantitative `\|E_int/E_inc\|` for the declared cell geometries | §11.3 — the largest gap in the program | full-wave FDTD/FEM of each cell with measured permittivity, or a per-cell measurement in the Richardson *et al.* (arXiv:2604.11785) style |
| M-2 | Holloway *et al.*, *J. Appl. Phys.* **121**, 233106 (2017) numeric systematics budget | §11.3; spec 00 §6 gap 6; audit refusal #35 | fetch the paper (paywalled). **Must not be filled from memory.** Now on this module's critical path, not just spec 09's |
| M-3 | closed-form Holstein escape factor `g_esc(k₀R)` | §11.2 | Molisch *et al.*, JQSRT **49**, 361 (1993) or Holstein, PR **83**, 1159 (1951) — watching the reciprocal-convention trap |
| M-4 | Durham-grade Rb **D2** self-broadening above `10¹² cm⁻³` | GATE-D's own coefficient (R-P6) | the Durham D2 follow-up to Weller 2011; R-6 adopts the Table-I value, which is not the high-density validation |
| M-5 | Penning/ionization rate law and free-charge field model for thermal Rydberg vapour | §11.7 | a primary with **rates**, not Weller 2016's fitted cross-section bound |
| M-6 | alkali–alkali velocity-changing-collision cross-section | §4.11 | primary source; the fetched Lei 2024 covers foreign gases only |
| M-7 | validated Rydberg–ground collisional **broadening** coefficient (today `0.5·\|shift\|`, 2×) | **U-1 — the dominant linewidth in the upper band** | measured density-dependent hot-cell EIT widths (spec 04's own stated self-check) |
| M-8 | a published measurement of coupling-beam absorption through a Rydberg-EIT cell | would turn 10/B-08 from a self-check into external validation | none found this session |
| M-9 | a published Rydberg-EIT spectrum with fully stated absolute axes | the whole amplitude chain | Jing's 120 µW → ~10 µW pair is the closest thing in the record, and it is a single point. **No row in this corpus is a figure digitisation** — a short table of stated numbers beats a long table of traced pixels |
| M-10 | Sedlacek 2016 (PRL **116**, 133201) adsorbate DC field magnitudes | DC/sub-kHz glass-cell claims (already refused) | a readable copy; the citation is VERIFIED from three indexes, the numbers would not extract |

---

## 12. Open questions no section could resolve

1. **Does the strong-probe solver need an ionization fence *before* it ships?** The density that
   buys OD ≈ 100 is essentially the density at which charge-induced bistability is observed, and at
   a realistic 2 % Rydberg fraction the vdW dephasing estimate rises ~390×. GATE-D fences by
   density today; whether that suffices once `ρ_rr` is genuinely computed is unknown.
2. **Is the `3.4×10¹¹ < N ≤ 1×10¹² cm⁻³` band publishable at all before M-7 closes?** The dominant
   Rydberg linewidth there is a 2×-uncertain guess. The alternative is dropping the ceiling to
   `N = 3.4×10¹¹ cm⁻³` (OD ≈ 10) until it does — which would exclude Su's 65 °C row.
3. **Is Su's fitted-α discrepancy (11.90 vs 30.19) saturation-along-z or cold-spot calibration?**
   Both reproduce it (the density model runs at 11 %/K, so a ~10 K cold-spot offset at 65 °C with
   none at 27 °C would do it). A concrete post-solver experiment, not a matter of opinion
   (10/B-45).
4. **Should audit refusal #21 be implemented as written — which would refuse the shipped default
   `LadderConfig` — or should the defaults move first?** The default sits at exactly 2× the gate.
   Recommendation: implement the gate, move the default, then re-run 10/B-30.
5. **Is a paraxial 2-D `(r, z)` split-step solver in scope, or do E1/E2/E6 get permanently
   downgraded to QUALITATIVE grading on every amplitude observable?**
6. **What is the phase-matching criterion for four-wave mixing in the counter-propagating
   Rb 5S–5P–nD ladder at high OD?** No criterion is offered because none was found (§11.6).
7. **Does a reabsorbed trapped photon destroy the `g–r` coherence with unit probability?** Assumed
   yes (projective, conservative). A source or a model calculation would turn `γ_trap` from a
   bound into an estimate — which is what §11.2 needs.
8. **Can a reduced-order or moment-closure treatment deliver the `O(ε²)` transport correction at
   acceptable cost, or does `ε_transp` remain a fence forever?** Needs a prototype to measure.
9. **Who ports the SELF-MEASURED harnesses, and does marking this document normative block on it?**
   Audit R4 says an unported session measurement is an unreproducible assertion.
10. **Should 10/B-30 (the strategic test) run before or after the solver is considered done?**
    Recommendation: **before**, on a stub — a FAIL changes what the solver is *for*.
11. **Does anyone publish EIT contrast vs optical depth beyond Su 2022?** One paper is a thin
    evidential base for a headline capability claim. A second independent OD-scan dataset would
    materially strengthen Tier D.

---

## 13. Implementation order (normative)

1. **`optical_depth_weak` + 10/B-17a–g + 10/B-35.** Fix the OD estimator first: every gate
   threshold downstream is calibrated in OD, and the shipped chain is off by ~2.9× (R10-10).
2. **`gate_report` + all seven gates + 10/B-46 mutation tests.** The fences before the physics.
   `ThickCellError` and the `s_sat`/linear-response gates do not exist today and must.
3. **Branch S0 + 10/B-01 / 01b / 01c / 03 / 04.** The convention locks. Nothing else is trusted
   until these pass; a solver failing 10/B-01 must not be used at any optical depth.
4. **10/B-30 on a stub.** Answer the strategic question before building for it.
5. **Branch N (saturable χ) + 10/B-05 / 06 / 06b / 14.** The discriminating physics — this is the
   actual deliverable (R10-1).
6. **Branch S (relaxation BVP) + 10/B-07 / 08 / 09 / 10 / 11 / 12.**
7. **Radial composition + 10/B-18 / 19 / 20; interpolant + V-FREEZE + 10/B-24.**
8. **Published rows 10/B-39 / 40 / 42 / 43 / 44 / 22 / 45** — last, and diagnostic.

---

*GreyNOC · RydSim spec 10 · adjudicated 2026-08-11 from four independent draft sections
(`docs/spec/drafts/10-*.md`) · 24 rulings, 55 benchmarks (41 gating / 14 diagnostic; 37 gating
at first release), 18 refusal conditions, 9 proposed amendments to specs 00/05/06 · subordinate to
spec 00 (locks 1–20, rulings R-1…R-28) and the integrity audit ·
house rule: reproducible or it didn't happen.*
