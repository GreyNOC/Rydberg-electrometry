# 10 §4 — Numerical method: the z-solver, its cost, and how it fails

**Status of this section.** Normative for the implementation of `rydsim.propagate` (the module
that spec 00 §4 assigns to **05 `rydsim.vapor`** — "propagation" is in that row's ownership
list). It is subordinate to `00-conventions.md`; where I believe an existing ruling needs
amending I say so explicitly in §4.12 rather than diverging silently.

**Confidence tags used here.** `VERIFIED` = primary source fetched and quoted this session.
`VERIFIED-ARC` = matches a reputable secondary implementation. `LITERATURE-RECALL`.
`UNVERIFIED`. Plus one tag this section needs and defines:

> **`SELF-MEASURED`** — produced this session by executing the shipped RydSim code or a
> short script whose text is reproduced in §4.13. Per integrity-audit **R4**, a
> session-measured number whose harness is not in the repo is *an unreproducible assertion*.
> Every `SELF-MEASURED` row below is therefore also a **release blocker until its harness is
> ported to `tests/test_spec10_numerics.py`.** They are not VERIFIED and must not be tagged so.

---

## 4.1 What is actually being solved

Slowly-varying-envelope, scalar, paraxial, one transverse shell (spec 05 §7.6). Probe along
+ẑ, coupling counter-propagating along −ẑ (spec 00 lock #10 default geometry).

```
(P)   dΩ_p/dz  =  + i (k_p/2) · χ̄_p(z) · Ω_p(z)          z ∈ [0, L],  Ω_p(0) = Ω_p^in
(C)   dΩ_c/dz  =  − i (k_c/2) · χ̄_c(z) · Ω_c(z)          z ∈ [0, L],  Ω_c(L) = Ω_c^in
(R)   Ω_RF(z)  =  supplied profile (not solved here — see §4.11 item 4)
```

| Symbol | Meaning | SI unit |
|---|---|---|
| z, L | position along the probe axis; cell optical path length | m |
| Ω_p(z), Ω_c(z) | **complex** probe / coupling Rabi amplitudes, `Ω = d·ℰ/ħ` (spec 00 lock #4) | rad/s |
| Ω_RF(z) | RF Rabi amplitude profile inside the cell | rad/s |
| k_p, k_c | probe / coupling vacuum wavenumbers 2π/λ | rad/m |
| χ̄_p, χ̄_c | **velocity-averaged** susceptibilities at the probe / coupling frequency | — (dimensionless) |
| α_p ≡ k_p Im χ̄_p | **intensity** absorption coefficient (spec 00 §2, α row) | m⁻¹ |
| α_p/2 | **amplitude** attenuation coefficient — the coefficient appearing in (P) | m⁻¹ |
| OD(Δ_p) ≡ ∫₀^L α_p dz | optical depth (intensity) | — |
| T(Δ_p) = exp(−OD) | intensity transmission | — |
| φ(Δ_p) = (1/2)∫₀^L k_p Re χ̄_p dz | dispersive phase | rad |
| N | number density of the sensed isotope | m⁻³ |
| v, f1D(v) | axial velocity; 1-D Maxwell pdf (spec 05 §2.c) | m/s; s/m |
| Δ_p, Δ_c, Δ_RF | detunings, `Δ = ω_field − ω_atom` (spec 00 lock #5) | rad/s |

The signs in (P) reproduce spec 05 §2.f exactly: `Ê(L) = Ê(0)·exp(i k_p L χ/2)` ⇒
`|Ê|² ∝ exp(−k_p Im χ L)`. The sign in (C) is derived and trapped in §4.8-P3.

**The structural fact that drives every design decision below.** In the weak-probe branch the
continued-fraction response of spec 06 §2.4 carries **no Ω_p dependence at all** — Ω_p cancels
between σ_eg and the χ prefactor. Verified on the shipped code: `rydsim.eit.chi_ladder` returns
**bit-identical** output for Ω_p/2π = 1 Hz and Ω_p/2π = 10 MHz (max relative difference exactly
`0.0`, SELF-MEASURED). Therefore:

* **(P) is a *linear scalar* ODE.** Its exact solution is the exponential of a *quadrature*:
  `ln Ω_p(L) = ln Ω_p(0) + i(k_p/2)∫₀^L χ̄_p(z) dz`. No ODE solver is required — the Magnus
  series for a scalar generator truncates identically at first order because a scalar commutes
  with itself at different z. **The numerical task is quadrature in z, not integration of an ODE.**
* z-dependence enters χ̄_p **only** through Ω_c(z) and Ω_RF(z).
* If additionally the coupling is undepleted, χ̄_p is a *constant* and (P) is solved in closed
  form — which is the thin-cell answer the engine already ships, exact at **any** OD, not just
  small OD (§4.9 and the amendment in §4.12).

The solver therefore has three branches, selected by *computed gates*, never by assumption:

| Branch | Condition (all computed at runtime) | Method | Cost |
|---|---|---|---|
| **S0** closed form | weak-probe gate holds **and** forward-only gate G (§4.5) holds | `ln T = −k_p L Im χ̄_p`, one velocity average | O(N_Δ N_v) |
| **S** quadrature | weak-probe gate holds, coupling depletes | Gauss–Legendre in z + relaxation (§4.4, §4.5) | O(M N_Δ N_v + N_it N_z N_Δ) |
| **N** nonlinear | probe above the weak-probe gate (spec 06 §4.6: Ω_p ≥ 0.01·min(Γ_e,\|Ω_c\|)) | adaptive embedded RK on ln Ω_p (§4.3) + relaxation | as S × N_z |

Branch **N** is the only branch in which (P) is genuinely nonlinear, because only there does
χ̄_p depend on \|Ω_p(z)\|². Everything expensive in this section exists for branches S and N.

---

## 4.2 Discretisation in z

### 4.2.1 Normative scheme: exponential (log-amplitude), not Runge–Kutta on the amplitude

**Integrate the logarithm.** Define `u_p(z) ≡ ln Ω_p(z)` (complex; Re u = ln amplitude,
Im u = phase). Then (P) becomes

```
du_p/dz = i (k_p/2) χ̄_p(z)          →      u_p(L) = u_p(0) + i (k_p/2) ∫₀^L χ̄_p dz
```

and the observables are `OD = −2·Re[u_p(L) − u_p(0)]`, `φ = Im[u_p(L) − u_p(0)]`.
Three properties, each of which the naive amplitude formulation lacks:

1. **No underflow.** `u_p` stays O(10²) where `Ω_p` underflows (§4.8-P2). OD = 750 is
   representable; `exp(−750)` is not.
2. **Error control means what you want it to mean.** An absolute tolerance on `Re u_p` *is* a
   tolerance on OD, which *is* a relative tolerance on T, uniformly across the scan — including
   in the wings where T ≈ 1 and at line centre where T ≈ 10⁻⁶.
3. **The OD → 0 reduction is exact by construction** (§4.9), because a constant integrand is
   integrated exactly by every consistent quadrature rule.

**In branches S0/S, use Gauss–Legendre quadrature in z**, order n, over [0, L]. The integrand
χ̄_p(z) is smooth (it is an analytic function of the smooth profile Ω_c(z)), so GL converges
spectrally in n and typically n = 8–16 suffices — orders of magnitude fewer velocity averages
than a marching scheme. **In branch N** (χ̄_p depends on Ω_p(z)) fall back to an initial-value
integrator on u_p: `scipy.integrate.solve_ivp(method="DOP853", rtol=1e-10, atol=1e-12)`
("Explicit Runge-Kutta method of order 8", error controlled as `atol + rtol*abs(y)`; defaults
rtol 1e-3 / atol 1e-6 are **far too loose** and must be overridden — VERIFIED, SciPy
`solve_ivp` reference fetched this session).

**Comparison to prior art.** CoOMBE (Potvliege & Wrathmall, *Comput. Phys. Commun.* 306, 2025;
source fetched this session) marches Maxwell–Bloch in z on a **fixed uniform step**
`z_step = (zmax − zmin)/n_z_steps` with a mid-point rule or a "4th-order Runge-Kutta formula"
(VERIFIED, quoted from `mbe.f90` `mbe_propagate_2`), with **no adaptive step-size halving**,
and with the Doppler loop nested *inside* the z loop. That is a correct and honest choice for a
general MBE integrator; RydSim can do better in branches S0/S because it exploits the
weak-probe linearity CoOMBE cannot assume.

### 4.2.2 The step criterion in terms of the local absorption length 1/α

For branch N (and for any fixed-step marching implementation), state the step in units of the
**local absorption length** `ℓ_abs(z) ≡ 1/α_p(z)`:

```
Δz(z)  ≤  η / α_p(z)   ≡   η · ℓ_abs(z) ,      i.e.   per-step optical depth  ΔOD ≤ η
```

**Choosing η, from measurement rather than folklore.** Classical RK4 applied to the amplitude
equation `dy/dz = −(α/2)y` over a total amplitude exponent X = OD/2 has global relative
amplitude error (SELF-MEASURED, script §4.13-A, total OD = 20):

| per-step ΔOD | steps | measured rel. err. in \|y(L)\| | law `OD·ΔOD⁴/3840` |
|---|---|---|---|
| 1.00 | 20 | 7.949e-3 | 5.21e-3 |
| 0.40 | 50 | 1.576e-4 | 1.333e-4 |
| 0.20 | 100 | 9.058e-6 | 8.333e-6 |
| 0.10 | 200 | 5.430e-7 | 5.208e-7 |
| 0.04 | 500 | 1.356e-8 | 1.333e-8 |
| 0.02 | 1000 | 8.403e-10 | 8.333e-10 |

Fitted order 4.00 (successive ratios 4.06, 4.00 in log₂); the closed form
**|Δy/y| ≈ OD·ΔOD⁴/3840** is accurate to better than 10 % for ΔOD ≤ 0.2 and approaches the
measurement from below. Inverting:

```
η = ΔOD_max  ≤  ( 3840 · ε_amp / OD_total )^(1/4)
```

with ε_amp the target relative amplitude error. Worked: ε = 1e-6 gives η = 0.140 at OD = 10,
**η = 0.0787 at OD = 100**, η = 0.0443 at OD = 1000. **Spec 05 §2.f's existing rule
`|ΔOD| ≤ 0.05` is hereby corroborated rather than replaced**: it delivers
ε_amp = OD·1.63e-9, i.e. 1.6e-7 at OD = 100 and 1.1e-6 at OD = 700 — adequate across the whole
declared range. Keep 0.05 as the default η; expose the formula above so a tighter run can ask
for more. Spec 05's companion rule `|ΔΩ_p|/|Ω_p| ≤ 2 %` is the same constraint restated
(ΔOD ≤ 0.04) and is retained as a redundant guard.

### 4.2.3 How the step adapts across the probe scan (this is *not* stiffness)

**Measured dynamic range.** On the shipped `rydsim.eit` path (Rb-87, 300 K, counter-prop
780.241/480.0 nm, Ω_c/2π = 5 MHz, γ_gr/2π = 51.5 kHz, 601 detunings over ±1.5 GHz,
4.01e5 velocity nodes) the velocity-averaged absorption Re S̄ spans
**1.010e-9 (peak, at Δ_p/2π = −5.0 MHz) to 2.297e-13 (wing, ±1.5 GHz) — a factor 4.40e3**
(SELF-MEASURED, script §4.13-D). The EIT feature itself contributes only ×2.55 at these
parameters; **the Doppler profile, not the transparency window, is the dominant driver.**

Consequences, both normative:

* **A z-grid adequate in the wings is 4400× too coarse at line centre.** Under the ΔOD ≤ η rule
  a shared grid must be sized for the *largest* α over the scan, wasting ~4.4e3× the work in
  the wings. Therefore: **the z-grid is per-detuning.** Each Δ_p gets its own GL order (branch
  S) or its own adaptive step sequence (branch N), sized from that detuning's own α.
  Independent corroboration that this is the real structure of the problem: Häupl *et al.*,
  arXiv:2410.19916 (NJP 2025), state of their own propagation model — *"the current model only
  allows the calculation of the propagation-corrected absorption for a single value of the
  detuning Δ at a time. This is because for a given longitudinal position z inside the medium,
  the intensity will change, depending on the detuning Δ."* (VERIFIED, quoted from the fetched
  PDF).
* **Do not call this stiffness, and do not reach for an implicit solver because of it.** For a
  *single* detuning the scalar equation has exactly one rate α(z), varying slowly through
  Ω_c(z). It is not stiff; explicit methods are optimal and implicit ones buy nothing. What
  exists is *heterogeneity across the parameter scan*, cured by per-detuning grids. The one
  place genuine stiffness can appear is **branch N near the saturation knee**, where χ̄_p(|Ω_p|²)
  turns over: there, and only there, switch to `method="Radau"` (implicit Radau IIA order 5) or
  `"LSODA"` (Adams/BDF with automatic stiffness detection) — and switch **on a measured
  symptom** (DOP853 step rejections exceeding 30 % of attempts), never pre-emptively.
  (Method descriptions VERIFIED from the SciPy `solve_ivp` reference this session.)

### 4.2.4 Convergence acceptance — raise, never return

Mirrors spec 06 §4.4, which requires *halving the step AND widening the domain*. In z the
domain is fixed at L, so the second, node-disjointness leg replaces widening:

> **Rule Z-CONV (normative).** Recompute the whole spectrum on a refined z-rule that
> (i) at least **halves** the effective step (GL order n → 2n, or adaptive tolerance ÷ 100),
> **and** (ii) is **node-disjoint** from the coarse rule. Accept iff
> ```
> max over the scanned Δ_p of | OD_fine(Δ_p) − OD_coarse(Δ_p) |  ≤  ε_z = 1e-4
> ```
> — an **absolute** criterion on OD, which is a **relative** criterion on T uniformly across
> the scan and remains meaningful where T underflows. Failure raises
> `rydsim.provenance.IntegrityError` carrying the measured magnitude. The pair
> `(converged: bool, max_delta_od: float, rule: str, n_nodes_coarse/fine: int)` ships in
> provenance (integrity-audit §4 item 6, "z-step halving result — as data").

*Why node-disjointness is a requirement and not a nicety.* A nested refinement that reuses the
coarse nodes can agree with its parent to high precision while both are aliasing the same
structure in Ω_c(z) — the identical failure mode that made Gauss–Hermite unusable for the
velocity average (ruling R-2 / audit R2: *"Spec 09's own 'doubling nodes' criterion can falsely
converge on an under-resolved dip"*). GL_n and GL_2n are believed to share no nodes (0 is a root
of P_m only for odd m, and 2n is even) — but **that is stated here as UNVERIFIED as a theorem**,
so the implementation must not rely on it: assert at runtime that no refined node lies within
1e-12·L of a coarse node, and if any does, refine to GL_{2n+1} instead.

---

## 4.3 The two-point boundary value problem from counter-propagation

### 4.3.1 Where the BVP comes from — and where it does not

`Ω_p` is specified at z = 0; `Ω_c` is specified at **z = L**. That is a two-point BVP.
Two clarifications that determine how often the expensive path is taken:

* **Counter-propagation alone does not create the BVP; depletion does.** If χ̄_c is negligible,
  Ω_c(z) ≡ Ω_c^in is *known*, (C) drops out, and (P) marches forward from z = 0. The topology is
  a BVP only when the coupling is attenuated by atoms whose state depends on Ω_p — i.e. only
  through the probe→coupling→probe feedback loop.
* **For a co-propagating coupling both conditions sit at z = 0** and the coupled system is an
  IVP, forward-marchable in one pass with no iteration. RydSim's default geometry is counter
  (lock #10), so the BVP is the *default* topology and must be handled, not assumed away.
  Note that CoOMBE — the reference open-source MBE suite — restricts its Maxwell–Bloch module to
  *"one or two laser fields (co)-propagating in an atomic vapour"* (VERIFIED, quoted from the
  fetched README). The counter-propagating case is genuinely extra work that standard tooling
  does not do for you.
* The coupling problem is **local** in z (no nonlocality): velocity-changing collisions and
  atomic transport, which would couple different z, are out of scope (spec 05 §7).

### 4.3.2 Scheme A (default): wave relaxation, undepleted coupling as the zeroth iterate

```
m = 0:      Ω_c^(0)(z) ≡ Ω_c^in                      (undepleted — the current engine's model)
repeat:
   forward   solve (P) on [0, L] with Ω_c^(m)(·)   →  Ω_p^(m+1)(z)      [§4.2 machinery]
   backward  solve (C) on [L, 0] with Ω_p^(m+1)(·) →  Ω̃_c^(m+1)(z)
   damp      Ω_c^(m+1) = (1 − θ_m)·Ω_c^(m) + θ_m·Ω̃_c^(m+1)
```

This is Gauss–Seidel wave relaxation. It is the natural scheme here precisely because the
zeroth iterate is *the physical answer in the regime where the loop is weak*, so the iteration
starts inside the basin and each sweep is a physically interpretable correction.

**Convergence criterion (normative).** Define the residual in the *observable*, not in the
field — a field residual can be small while the OD it produces is not:

```
r_m  =  max over scanned Δ_p of | OD^(m+1)(Δ_p) − OD^(m)(Δ_p) |          [absolute, in OD]
q_m  =  r_m / r_{m−1}                                                     [contraction ratio]
```

* **Accept** when `r_m ≤ ε_relax = 1e-4` (same units and same number as ε_z, so the two error
  budgets are commensurate) **and** `q_m < 1` — never on r_m alone, because a stalled iteration
  can produce a small increment while sitting far from the fixed point.
* **Damping.** θ_m = 1 while q_m ≤ 0.5. For 0.5 < q_m < 1 use θ_m = 1/(1 + q_m); Anderson(2)
  acceleration is permitted as an option and must be A/B-tested against plain damping before
  becoming a default.
* **Refuse.** If `q_m ≥ 1` on two consecutive iterations, or `m > m_max = 20`, raise
  `IntegrityError` naming (r_m, q_m, m, OD_p, OD_c). **Do not return the last iterate.** A
  diverging relaxation in this problem means the physical feedback loop has gain ≥ 1, i.e. the
  medium is bistable/self-focusing at these parameters — a physics statement the caller must
  see, not a number to smooth over.
* The full trace `[(m, r_m, q_m)]` ships in provenance.

*Expected contraction.* The loop gain is the product of two logarithmic sensitivities times the
two optical depths. The probe-side factor is measurable and small: on the Rb-87 fixture,
`d ln(Re S̄)/d ln Ω_c` at the EIT peak is **−0.515 (Ω_c/2π = 2.5 MHz), −0.825 (5 MHz),
−0.986 (10 MHz)** — i.e. |s_c| ≲ 1 (SELF-MEASURED, script §4.13-C). The coupling-side factor is
smaller still because χ̄_c is O(Ω_p²) (§4.3.4). Contraction is therefore expected to be fast in
every regime RydSim is meant to operate in — **which is a prediction that `q_m` tests, not an
assumption the code may make.**

### 4.3.3 Scheme B (fallback): collocation

When Scheme A refuses, do not give up and do not loosen ε_relax. Recast the system as a real
4-vector `y(z) = (Re u_p, Im u_p, Re u_c, Im u_c)` with `u = ln Ω`, boundary conditions
`y₀(0) = ln|Ω_p^in|`, `y₁(0) = arg Ω_p^in`, `y₂(L) = ln|Ω_c^in|`, `y₃(L) = arg Ω_c^in`, and hand
it to `scipy.integrate.solve_bvp` — *"a 4th order collocation algorithm with the control of
residuals"* using *"a damped Newton method with an affine-invariant criterion function"*,
driving `norm(r/(1 + abs(f))) < tol` per mesh interval with `r = y' − f(x,y)` (VERIFIED, SciPy
`solve_bvp` reference fetched this session; refs Kierzenka & Shampine 2001; Shampine, Muir & Xu
2006; Ascher, Mattheij & Russell 1995).

Normative settings and handling:

* `tol = 1e-8` (**not** the default `tol = 1e-3`, which is three orders looser than ε_z),
  `max_nodes` ≥ 10⁴.
* Seed with Scheme A's last iterate — a good initial mesh is what makes damped Newton converge.
* **Status handling is a refusal, not a warning.** Status 1 (*"maximum number of mesh nodes is
  exceeded"*) and status 2 (*"singular Jacobian encountered"*) both raise `IntegrityError`.
  Status 0 still goes through Rule Z-CONV.
* **Cost warning.** `solve_bvp` calls a *vectorised* RHS over the entire mesh each Newton
  iteration, so every Newton step is O(n_mesh) velocity averages. Without the interpolant of
  §4.4 this path is not merely slow, it is unusable. That is why Scheme A is the default and
  Scheme B the fallback, not the reverse.

### 4.3.4 When forward-only integration is provably adequate

This gate is what stops the expensive path being taken needlessly. It is a **computed bound
plus a measured certificate** — never a formula alone.

**Ingredients, all already available from machinery the solver has built:**

* `OD_c ≡ k_c ∫₀^L Im χ̄_c dz`, evaluated on the zeroth iterate. χ̄_c is the susceptibility at the
  coupling frequency; it is `O(Ω_p²)` because ground-state atoms are transparent at λ_c and the
  intermediate state is nearly empty. Measured on `rydsim.lindblad` (Rb-87 parameters,
  Ω_c/2π = 5 MHz, Γ_e/2π = 6.0666 MHz, Γ_r/2π = 3 kHz, transit + dephasing 2π·50 kHz,
  all fields resonant, v = 0):
  ρ_ee = **9.4927e-10** at Ω_p/2π = 1 kHz, **9.4901e-6** at 100 kHz, **9.2464e-4** at 1 MHz —
  clean Ω_p² scaling (SELF-MEASURED, script §4.13-B). Combined with the small 5P→nD dipole this
  makes OD_c minuscule in weak probe. **The solver must still compute it. Never assume it.**
* `s_c ≡ max over z and Δ_p of |∂ ln Im χ̄_p / ∂ ln Ω_c|` — free, by finite differencing the
  Ω_c-interpolant of §4.4 that has already been built. (Fixture value ≲ 1, above.)

**Gate G (a-priori).** The neglected change in the observable is bounded by

```
| δOD |  ≤  OD_p · s_c · ( 1 − e^{−OD_c/2} )   ≤   OD_p · s_c · OD_c / 2
```

Take the forward-only path iff `OD_p · s_c · OD_c / 2  ≤  ε_gate = 1e-5` (a decade inside ε_z,
so the neglect can never be the dominant error).

**Certificate (mandatory, and this is the part that makes G honest).** Whenever G passes,
**still take exactly one relaxation sweep** and record `r_1`. One sweep costs one backward pass
— negligible against the forward scan — and it converts "I proved a bound" into "I measured the
thing the bound was about". If `r_1 > ε_gate`, G was wrong for these parameters: **raise**, and
fall through to Scheme A. A bound that is ever violated is a wrong bound (benchmark 10/N-10).

---

## 4.4 The inner loop: avoiding a full velocity average at every z

### 4.4.1 The cost that has to be beaten

The velocity average is the expensive object and R-2 forbids making it cheap by coarsening
(Gauss–Hermite is banned for anything with EIT/AT structure). **SELF-MEASURED unit cost** on
the shipped `rydsim.eit` path, single core, this session: one Doppler-averaged spectrum over
201 probe detunings on a 42 281-node resonance-refined grid takes **0.442 s** — about
**52 ns per (detuning × velocity node)**. This figure is hardware-dependent and is used only as
an order-of-magnitude anchor for the complexity argument below; it is not a portable constant.

Naive nesting (Doppler loop inside the z loop, as CoOMBE does — VERIFIED from `mbe.f90`) costs

```
C_naive  =  N_it · N_z · N_Δ · N_v · c₀
```

With N_it = 3, N_z = 200, that is 600 × 0.442 s ≈ **265 s per spectrum, per LO point.** For a
superheterodyne transfer curve with N_E = 40 LO points (`superhet_transfer`), ≈ 3 hours.
Unacceptable, and the reason the current engine refuses instead.

### 4.4.2 What may be cached, and the one rule that makes it work

**The invariance to exploit.** χ̄(Δ_p; z) depends on z *only* through the local field amplitudes
(Ω_c(z), Ω_RF(z)) — plus |Ω_p(z)|² in branch N. It does not depend on z in any other way. So
build, once, a **tensor Chebyshev interpolant of the velocity-averaged spectrum in those
parameters**, and evaluate the interpolant at every z step:

```
S̄(Δ_p ; Ω_c)  ≈  Σ_{j<M} a_j(Δ_p) · T_j( 2(Ω_c − Ω_c^min)/(Ω_c^max − Ω_c^min) − 1 )
```

on Chebyshev–Gauss nodes spanning `[Ω_c(0), Ω_c(L)]` widened by 10 %, i.e.
`[Ω_c^in·e^{−OD_c/2}·0.9, Ω_c^in·1.1]`. Interpolate the **complex response S̄**, never T or OD
(§4.8-P7).

> **Rule V-FREEZE (normative, and this is the whole trick).** All M nodes of the interpolant
> **must be evaluated on one and the same velocity grid**, built once for the *widest* dressed
> structure over the parameter box (largest Ω_c and largest Ω_RF), and then **frozen**. A grid
> that is rebuilt per node — as `rydsim.eit.resonance_refined_vgrid` does, since its window
> widths depend on Ω_c — jitters the integrand and destroys the interpolant's convergence.

**SELF-MEASURED convergence** (script §4.13-E; Rb-87 300 K, counter-prop 780.241/480.0 nm,
201 detunings over ±2π·30 MHz, Ω_c/2π ∈ [1.5, 5.0] MHz, 17 random test points, error =
max |interp − direct| / max|direct|):

| M (Chebyshev nodes in Ω_c) | **frozen grid (Rule V-FREEZE)** | grid rebuilt per node |
|---|---|---|
| 6 | 3.766e-4 | 3.765e-4 |
| 8 | 3.296e-5 | 3.293e-5 |
| 12 | **2.804e-7** | 2.364e-6 |
| 16 | **2.473e-9** | 6.365e-7 |
| 20 | **2.343e-11** | — |
| 24 | — | 4.126e-7 (floored) |

Two readings, both load-bearing. (i) With the grid frozen the interpolant converges
**spectrally** — 2.5e-9 at M = 16, five decades inside ε_z. (ii) Without it the error **floors
at ~4e-7 and stops improving**: adding nodes buys nothing because the limiter is grid jitter,
not polynomial degree. That floor is only 400× inside ε_z and would be indistinguishable from
"converged" to a careless refinement test — exactly the silent-under-resolution signature
(§4.8-P1). Interpolating in Ω_c² instead of Ω_c was also measured (3.6e-5 / 3.6e-7 / 3.1e-9 at
M = 8/12/16) and is not better; **use Ω_c**.

**Normative default: M = 16 in Ω_c**, giving 2.5e-9 ≪ ε_z. When Ω_RF also varies along z, use a
tensor product M_c × M_RF = 16 × 16 = 256 velocity averages; a 2-D Chebyshev interpolant on a
smooth analytic function inherits the same spectral rate per axis (LITERATURE-RECALL for the
tensor case — the 1-D rate above is SELF-MEASURED; the 2-D rate must be measured before it is
relied on, benchmark 10/N-7b).

### 4.4.3 What may *not* be cached — the convolution trick does not transfer

Häupl *et al.* (arXiv:2410.19916) reduce their Doppler cost with a shift of variable:
*"The substitution v′_z = Δ/k − v_z reduces the computational burden, as χ(v′_z) must only be
computed once and can be reused for multiple values of Δ."* (VERIFIED, quoted from the fetched
PDF.) **This trick is unavailable to RydSim and must not be imported.** Two lines:

It is valid when the response depends on (Δ, v) only through the single combination Δ − kv, so
that the velocity average is a convolution in Δ. A mismatched ladder response depends on **two**
independent combinations, the one-photon and two-photon detunings,

```
a(Δ_p, v) = Δ_p − k_p v ,       b(Δ_p, v) = Δ_p + Δ_c + (k_c − k_p) v
```

whose Jacobian `∂(a,b)/∂(Δ_p,v)` has determinant **k_c ≠ 0**. The map (Δ_p, v) → (a, b) is
therefore a bijection and the response is an irreducibly two-argument function; no 1-D
convolution kernel in Δ_p exists. (The trick *does* reappear in the degenerate case k_c = k_p,
which is why it is correct in the single-field D-line problem Häupl *et al.* solve.)

What *is* legitimately cached besides §4.4.2: the frozen velocity grid and its renormalised
Maxwell weights (built once per (T, m, geometry, parameter box)); the coherence-decay assembly
γ_ij extracted from the collapse set (spec 06 §2.2); and, in branch S0, the single velocity
average itself.

### 4.4.4 Resulting complexity, and the error it buys

```
C_cached  =  M_c·M_RF · N_Δ · N_v · c₀        (build the interpolant, once)
          +  N_it · N_z · N_Δ · M_c·M_RF · c₁ (evaluate; c₁ = one polynomial eval, ~ns)
```

The second term is negligible against the first because `c₁ ≪ N_v·c₀`. The speed-up over
naive nesting is therefore ≈ `N_it · N_z / (M_c·M_RF)`. Concretely, with N_it = 3, N_z = 200,
M_c = 16, M_RF = 1: **600 → 16 velocity averages, a 37.5× reduction, at an interpolation error
of 2.5e-9** — i.e. 265 s → 7.1 s per spectrum on the measured unit cost. The saving is
*larger*, not smaller, for the expensive collocation fallback of §4.3.3, whose Newton sweeps
would otherwise each cost n_mesh velocity averages.

**Accounting rule (normative).** The interpolation error is not free error: it enters the same
budget as ε_z. Require `err_interp ≤ ε_z/100 = 1e-6`, measured (not assumed) by evaluating the
interpolant against a direct velocity average at ≥ 5 points drawn *inside* the box and never
coinciding with the Chebyshev nodes. Report `err_interp` in provenance.

---

## 4.5 Reduction to the existing thin-cell answer — the free regression test

This is the single most important check in the section, and it is stronger than "as OD → 0".

> **Reduction R-THIN (normative).** In branch S0 — weak-probe gate satisfied *and* coupling
> declared undepleted — the propagation solver **must return, at every OD**:
> ```
> ln T(Δ_p)  =  − k_p · L · Im χ̄(Δ_p)      and      φ(Δ_p) = (1/2)·k_p·L·Re χ̄(Δ_p)
> ```
> agreeing with the already-validated chain
> `rydsim.eit.doppler_average → chi_si → transmission` to **≤ 1e-12 relative in ln T**,
> for **every** OD in {1e-6, 1e-3, 0.1, 1, 5, 50, 500}, not merely as OD → 0.

*Why the strong form is the right one.* In that branch χ̄ carries no z-dependence whatsoever
(§4.1), so the exact solution of (P) is the exponential of a constant and **any consistent
quadrature integrates a constant exactly**. There is no discretisation error to hide behind at
any OD. A discrepancy is therefore never "the thick solver is doing more physics" — it is a bug
in the exponent (factor 2, sign, k_p vs k_c), in the velocity grid, or in the plumbing.
This makes the existing, independently validated thin-cell code a *complete* oracle for the new
solver over a whole branch, which is the cheapest high-value regression test available.

**The weak form, for the depleted case (benchmark 10/N-2).** With depletion enabled, the
difference from the thin answer must vanish **quadratically** in density N: OD_c ∝ N gives
δχ̄_p ∝ N, and δOD = OD_p·(δχ̄_p/χ̄_p) ∝ N². Sweep N over a decade and require the fitted
log–log slope of |ln T_thick − ln T_thin| versus N to be **≥ 1.9**. A slope of 1 means the
solver is applying a first-order correction where none exists (a leaked sign or an
Ω_c update applied to the probe); a slope of 0 means depletion is not wired in at all.

---

## 4.6 Numerical pitfalls: name, symptom, detector

Each is a named failure with an observable symptom and a test that fires on it. "Detector"
entries marked **always-on** run in production, not only in CI.

**P1 — Silent under-resolution (in z, and in the Ω_c interpolant).**
*Symptom:* a smooth, plausible spectrum whose OD is wrong by a few percent; refinement appears
to "converge" because the refined rule shares nodes with the coarse one, or because the
interpolant has floored on grid jitter (measured floor 4.1e-7, §4.4.2) rather than on degree.
*Prior art warning:* Häupl *et al.* — *"We noticed that for 10 slices our simulation converges
to an acceptable level of accuracy … Note that the required number of slices N will vary
depending on the absorption α, cell length L and atomic species … Currently, the value of N
required for the model to converge must be determined manually."* (VERIFIED, fetched). A
manually-chosen slice count is precisely the hazard; RydSim must not ship one.
*Detector (always-on):* Rule Z-CONV with a node-disjoint refinement (§4.2.4) **and** the
off-node interpolant check of §4.4.4. Both **raise**. Mutation test: deliberately under-resolve
and assert the raise fires (10/N-5).

**P2 — Exponential underflow at high OD.**
*Symptom:* T returned as `0.0`; every downstream quantity (transduction slope κ = dP/dE, NEF)
becomes 0/0 or ∞ and "looks like a result" — the exact failure that produced audit CRIT-2's
5.4e9 nV/cm/√Hz.
*Measured float64 thresholds (SELF-MEASURED, script §4.13-F):* `exp(−OD)` becomes **subnormal
below OD = 708.3964** (`ln(tiny) = −708.3964185322641`), progressively losing relative
precision, and flushes to **exactly 0.0 for OD ≳ 745.2** (measured `exp(−745) = 5e-324`,
`exp(−746) = 0.0`).
*Fix (normative):* the solver's primary return is **`ln T` (i.e. −OD)**, computed in log space
throughout (§4.2.1); `T` is a separate accessor that flags `underflow=True` when OD > 708.4.
Spec 05 §4.7's existing "guard exp(−OD) for OD > 700 → return 0.0 explicitly" is retained as
the *display* rule and is consistent with the measurement.
*Physics rule that must accompany it:* at OD ≳ 30 the transmitted probe is below any realistic
detector's noise floor. The honest output is "no signal, OD = X", **not** a denormal. See the
amendment proposal in §4.12.

**P3 — Sign error in the counter-propagating term.**
*Symptom:* the coupling *gains* as it crosses the cell; EIT contrast grows with cell length;
the relaxation of §4.3.2 diverges (q_m > 1) for no physical reason.
*Derivation of the correct sign:* with `dΩ_c/dz = −i(k_c/2)χ̄_c Ω_c`,
`d|Ω_c|²/dz = +k_c Im(χ̄_c)|Ω_c|² > 0` for an absorptive medium, so |Ω_c| is **largest at
z = L (the coupling's entry face)** and smallest at z = 0. Copying the probe's `+i` sign
inverts this and manufactures gain.
*Detector (always-on):* assert `|Ω_c(z)|` is monotone non-decreasing in z and
`|Ω_c(0)|/|Ω_c(L)| = exp(−OD_c/2)` to 1e-10 (10/N-4).

**P4 — Intensity vs amplitude: the factor-2 trap (spec 00 lock #3's field-vs-intensity rule).**
*Symptom:* OD wrong by exactly 2× or 0.5×; in a superhet run this is catastrophic because both
P and dP/dE inherit it.
*The three quantities that must never be interchanged:* `α = k_p Im χ̄` is the **intensity**
absorption coefficient (spec 00 §2 α row; spec 05 §2.f: *"T = exp(−k_p Im χ L) ← exact factor
convention: k_p = 2π/λ_p, no extra 2 or 4π"*); `α/2` is the **amplitude** attenuation
coefficient and is what appears in (P); `Im n_refr ≈ Im χ̄/2` is the imaginary refractive index.
*Independent primary corroboration (VERIFIED, fetched):* Häupl *et al.* Eq. (11) writes the
intensity equation as `dI/dz = (4π/λ)·Im{√(1+χ)}·I`, i.e. **2k·Im n_refr** — two factors of 2
that cancel against `Im n_refr = Im χ/2` to give exactly `k·Im χ`. Writing `k·Im n_refr` for
intensity is 2× low; writing `2k·Im χ` is 2× high.
*Detector:* the closed-form check 10/N-3 — propagate a *constant, purely imaginary* χ̄ and
assert `ln T = −k_p Im χ̄ L` to 1e-14 relative, and a *purely real* χ̄ and assert `ln T = 0`
with `φ = k_p L Re χ̄/2`. This isolates the exponent from all physics.

**P5 — Velocity-grid jitter under a moving Ω_c.** *Symptom:* the Ω_c interpolant stops
converging at ~4e-7 (measured) while looking converged. *Detector:* Rule V-FREEZE plus the
mutation test 10/N-7 that asserts the frozen-grid error at M = 16 is ≤ 1e-8 *and* the
rebuilt-grid error is ≥ 1e-7 — i.e. the rule is proven to matter, not merely asserted.

**P6 — One shared z-grid across the whole probe scan.** *Symptom:* line-centre OD systematically
low, wings fine; the error hides because it is largest exactly where T is smallest. *Detector:*
per-detuning grids (§4.2.3) plus 10/N-11, which asserts that a shared grid sized in the wings
fails Z-CONV by ≫ ε_z given the measured 4.4e3 dynamic range.

**P7 — Interpolating T or OD instead of χ̄.** *Symptom:* small errors near line centre become
large errors in the exponent; the interpolant is fitted to a function with an exponential
dynamic range instead of a smooth analytic one. *Rule:* interpolate the complex response S̄ (or
χ̄); form OD and T only after the z-quadrature. *Detector:* type-level — the cache API accepts
and returns complex response arrays only.

**P8 — Renormalising the Maxwell weights inside the z loop.** Spec 05 §2.d requires the velocity
weights be renormalised to Σw = 1 (killing ±4σ truncation, ~6e-5). Doing this *per z step* on a
frozen grid is wasted work and, worse, if the grid ever changes mid-loop it introduces a
step-to-step discontinuity in χ̄ that destroys the z-quadrature's smoothness assumption.
*Rule:* renormalise once, at grid-freeze time.

**P9 — A relaxation that "converges" because the update never took effect.** *Symptom:*
r_1 ≈ 0 on every fixture, including ones engineered to deplete strongly — the backward sweep is
writing into a copy, or the forward sweep is reading the stale Ω_c. *Detector:* 10/N-9's
positive control — a fixture with OD_c ≈ 0.3 must show r_1 comfortably above ε_relax before the
iteration converges, i.e. **the test asserts the correction is nonzero before asserting it is
small.**

**P10 — Unit drift between Ω, ℰ and I along z.** Spec 00 lock #4 and lock #3: Ω ∝ ℰ ∝ √I. A
solver that steps in I but reports Ω, or that applies `exp(−ΔOD)` to an amplitude, is off by the
square. *Rule:* the state variable is `ln Ω` (complex) and nothing else; I and ℰ are derived at
the API boundary by one conversion each.

---

## 4.7 Parameters and tolerances

| # | Quantity | Symbol | Value | Unit | Source | Confidence |
|---|---|---|---|---|---|---|
| 1 | per-step optical depth (branch N / fixed-step) | η = ΔOD_max | 0.05 | — | spec 05 §2.f; corroborated by the §4.2.2 error law | VERIFIED (spec-internal) + SELF-MEASURED |
| 2 | RK4 error law coefficient, \|Δy/y\| = OD·ΔOD⁴/C | C | 3840 | — | §4.2.2 measurement, order 4.00 | SELF-MEASURED |
| 3 | z-convergence acceptance (absolute in OD) | ε_z | 1e-4 | — | mirrors spec 06 §4.4's 1e-4 velocity rule | NORMATIVE (this spec) |
| 4 | relaxation acceptance (absolute in OD) | ε_relax | 1e-4 | — | commensurate with ε_z | NORMATIVE |
| 5 | relaxation contraction ceiling | q_max | 0.5 (damp above), 1.0 (raise above, ×2 consecutive) | — | §4.3.2 | NORMATIVE |
| 6 | max relaxation sweeps | m_max | 20 | — | §4.3.2 | NORMATIVE |
| 7 | forward-only gate | ε_gate | 1e-5 | — | one decade inside ε_z | NORMATIVE |
| 8 | Chebyshev nodes in Ω_c | M_c | 16 (→ 2.473e-9) | — | §4.4.2 measurement | SELF-MEASURED |
| 9 | interpolation error budget | err_interp | ≤ 1e-6 | — | ε_z/100 | NORMATIVE |
| 10 | `solve_ivp` settings (branch N) | rtol/atol | 1e-10 / 1e-12 on ln Ω; DOP853; error = atol + rtol·\|y\| | — | SciPy `solve_ivp` reference, fetched | VERIFIED |
| 11 | `solve_bvp` settings (Scheme B) | tol / max_nodes | 1e-8 / ≥1e4 (defaults 1e-3 / 1000 rejected) | — | SciPy `solve_bvp` reference, fetched | VERIFIED |
| 12 | float64 `exp(−OD)` subnormal onset | OD_sub | 708.3964185322641 | — | `ln(np.finfo(float).tiny)` | SELF-MEASURED |
| 13 | float64 `exp(−OD)` flush-to-zero | OD_zero | ≈745.2 (`exp(−745)=5e-324`, `exp(−746)=0.0`) | — | measured | SELF-MEASURED |
| 14 | probe-scan absorption dynamic range (Rb-87, 300 K, ±1.5 GHz) | α_max/α_wing | 4.40e3 | — | §4.2.3 measurement | SELF-MEASURED |
| 15 | probe-side loop sensitivity \|∂ln Im χ̄_p/∂ln Ω_c\| at the EIT peak | s_c | 0.515 / 0.825 / 0.986 at Ω_c/2π = 2.5 / 5 / 10 MHz | — | §4.3.2 measurement | SELF-MEASURED |
| 16 | intermediate-state population (weak probe, resonant, v = 0) | ρ_ee | 9.4927e-10 / 9.4901e-6 / 9.2464e-4 at Ω_p/2π = 1 kHz / 100 kHz / 1 MHz | — | `rydsim.lindblad`, §4.3.4 | SELF-MEASURED |
| 17 | unit cost of one velocity-averaged spectrum | c₀·N_v | 0.442 s / 201 detunings / 42 281 nodes ⇒ ≈52 ns per (Δ, v) | s | shipped `rydsim.eit`, this session | SELF-MEASURED (hardware-dependent; anchor only) |

MISSING (owned elsewhere, deliberately not specified here): the value of χ̄_c's dipole and the
coupling-frequency prefactor (spec 03 / `rydsim.dipoles`); the Ω_RF(z) profile inside the cell
(spec 05/07 cell-EM, spec 06 §2.8 items 6–7); the density N(T) (spec 05 §2.a).

---

## 4.8 Validation benchmarks (→ `tests/test_spec10_numerics.py`)

Tolerance semantics per spec 09 §7. Every `raise` benchmark must also carry a **mutation test**
proving the raise fires — a refusal that never fires is not a refusal.

| ID | Quantity | Setup | Expected | Tolerance | Source / type | Confidence |
|---|---|---|---|---|---|---|
| **10/N-1** | **R-THIN: solver vs. the shipped thin-cell chain, branch S0** | weak probe, undepleted coupling; OD ∈ {1e-6, 1e-3, 0.1, 1, 5, 50, 500} | identical `ln T`, `φ` at **every** OD | ≤ 1e-12 rel on ln T | §4.5; exact because χ̄ is z-constant | VERIFIED (by construction) — **the free regression test** |
| 10/N-2 | depleted-case reduction order in density | depletion on, sweep N over a decade | log–log slope of \|ln T_thick − ln T_thin\| vs N | slope ≥ 1.9 | §4.5 weak form | derived; self-checked |
| 10/N-3 | exponent identity (factor-2 / sign isolation) | constant χ̄ = i·χ″ (then χ̄ = χ′ real) | `ln T = −k_p χ″ L`; `ln T = 0`, `φ = k_p L χ′/2` | ≤ 1e-14 rel | §4.6-P4; spec 05 §2.f | VERIFIED (identity) |
| 10/N-4 | counter-propagating sign | depletion on, Im χ̄_c > 0 | \|Ω_c(z)\| monotone ↑ in z; \|Ω_c(0)\|/\|Ω_c(L)\| = e^{−OD_c/2} | monotonicity: QUALITATIVE (always-on); ratio ≤ 1e-10 | §4.6-P3 | VERIFIED (derivation) |
| 10/N-5 | Z-CONV fires | GL_n vs node-disjoint GL_2n; plus a deliberately under-resolved mutant | max_Δ \|ΔOD\| ≤ 1e-4; mutant raises `IntegrityError` | ε_z = 1e-4 | §4.2.4 | NORMATIVE |
| 10/N-6 | RK4 step-size law (branch N) | OD = 20, ΔOD ∈ {0.4, 0.2, 0.1, 0.04} | 1.576e-4 / 9.058e-6 / 5.430e-7 / 1.356e-8; fitted order 4.00 | order 4.00 ± 0.05; constants ≤ 15 % | §4.2.2 | SELF-MEASURED — port before release (R4) |
| 10/N-7 | Ω_c interpolant, **and that V-FREEZE matters** | M = 16, Rb-87 fixture of §4.4.2 | frozen grid ≤ 1e-8 (measured 2.473e-9) **and** rebuilt grid ≥ 1e-7 (measured 6.365e-7) | as stated | §4.4.2 | SELF-MEASURED — port before release |
| 10/N-7b | 2-D (Ω_c, Ω_RF) tensor interpolant | M_c = M_RF = 16 | ≤ 1e-6 | err_interp budget | §4.4.2 | **UNMEASURED — must be measured, not assumed** |
| 10/N-8 | high-OD representation | OD = 750 requested | returns `ln T = −750`, `underflow=True`; **never** a bare `0.0` | exact | §4.6-P2 | SELF-MEASURED (thresholds) |
| 10/N-9 | relaxation: correction is nonzero, then small | fixture with OD_c ≈ 0.3 | r_1 > ε_relax (positive control) **then** r_m ≤ 1e-6 within 5 sweeps, q_m ≤ 0.5 | as stated | §4.3.2, §4.6-P9 | NORMATIVE |
| 10/N-10 | forward-only gate G is an upper bound | ≥ 5 fixtures, OD_p ∈ [0.1, 30] | measured \|δOD\| from one sweep ≤ `OD_p·s_c·OD_c/2` on **every** fixture | one-sided; any violation FAILS | §4.3.4 | NORMATIVE (a violated bound is a wrong bound) |
| 10/N-11 | per-detuning z-grids required | shared grid sized in the wings vs per-detuning | shared grid fails Z-CONV by ≫ ε_z | QUALITATIVE predicate | §4.2.3; measured range 4.40e3 | SELF-MEASURED |
| 10/N-12 | branch S vs branch N cross-solver | inside the weak-probe overlap | identical OD | ≤ 1e-9 absolute in OD | method-A-vs-B, mirroring 06/B-1 | VERIFIED (by construction) |
| 10/N-13 | provenance completeness | any thick run | record carries (converged, max_delta_od, rule, n_nodes, relaxation trace [(m, r_m, q_m)], err_interp, OD_p, OD_c, s_c, branch, underflow) | all present | integrity-audit §4 items 5–8 | NORMATIVE |

pytest notes: 10/N-1 and 10/N-3 are the convention locks — run first, fail loudly, and gate
every other benchmark in the file. 10/N-7 and 10/N-11 are slow; mark `@pytest.mark.slow` and
gate releases on them.

---

## 4.9 Known limitations of this numerical scheme

1. **Scalar, single transverse shell.** Radial structure is handled by spec 05 §2.g's
   Gauss–Laguerre shell average *outside* this solver; shells do not exchange energy, so
   dispersive self-lensing and self-focusing at very high OD are absent (spec 05 §7.6). The
   relaxation of §4.3.2 diverging (q_m ≥ 1) is the *symptom* by which this solver detects that
   it is being asked a question outside its model — hence the refusal rather than a return.
2. **Steady state in time.** z-propagation is a spatial steady-state problem; nothing here
   relaxes spec 06 §4.7's CW restriction. Pulse propagation, slow light and storage need the
   time-dependent MBE path, which this section does not specify.
3. **The velocity average is 1-D and collisionless.** Velocity-changing collisions would couple
   different z through atomic transport and would break the locality assumed in §4.3.1,
   invalidating the two-point BVP formulation itself (not just its solution). Buffer-gas cells
   are already refused (integrity-audit §3 item 16); that refusal must be re-asserted on this
   path.
4. **Ω_RF(z) is an input, not a solved field.** RF absorption by the vapour is negligible, but
   the in-cell RF profile (standing waves, cell etalon — spec 06 §2.8 items 6–7) is owned by the
   cell-EM module. If that module ever supplies a z-profile, the 2-D interpolant of §4.4.2 is
   required and 10/N-7b must be measured first.
5. **`SELF-MEASURED` rows are not yet reproducible in-repo.** Per integrity-audit **R4**, every
   number in §4.7 tagged SELF-MEASURED rests on a script that is currently in a scratchpad, not
   in `tests/`. Until §4.13's harness is ported, those numbers are unreproducible assertions and
   **must not gate a release**. This includes the headline M = 16 → 2.5e-9 interpolation figure.
6. **The 2-D interpolant convergence rate is asserted from the 1-D measurement.** Tagged
   UNVERIFIED in §4.4.2 and benchmarked as 10/N-7b precisely so it cannot be quietly assumed.
7. **The GL node-disjointness claim is UNVERIFIED as a theorem** and is replaced by a runtime
   assertion (§4.2.4). If someone later proves it, the assertion becomes free; until then it
   stays.
8. **Contraction of Scheme A is a prediction, not a guarantee.** The measured sensitivities
   (§4.3.2) make fast convergence overwhelmingly likely in RydSim's operating regimes, but no
   contraction theorem is claimed for the full nonlinear map. That is exactly why q_m is
   monitored and why divergence raises.

---

## 4.10 Amendments this section proposes to spec 00 (stated, not taken)

Per the brief's instruction not to diverge silently. **None of these is applied here; spec 00's
owner adjudicates.**

**A-1 (substantive) — the optical-depth fence is on the wrong variable, and the corpus carries
three different numbers for it.** Today: integrity-audit §3 refusal #18 says *"Optically thick
(OD > 0.1) … through the analytic thin-cell path — `ThickCellError`"*; spec 05 §2.f says
*"Optically thin (use single-exponential with χ evaluated once): OD_peak ≤ 0.1"*; and
`rydsim.experiment.LadderConfig.max_optical_depth = 5.0` ships with a comment that states the
opposite physics — *"In the weak-probe limit Beer-Lambert with the EIT-suppressed chi is exact
at any OD (chi is probe-independent; the coupling is undepleted …), so moderate OD is VALID
physics — merely a poor operating point."* §4.1 and 10/N-1 establish that the code comment is
right: in branch S0 the thin-cell answer is **exact at any OD**, not an approximation, so the
OD > 0.1 fence refuses a correct answer while OD ≤ 5.0 admits an incorrect one whenever the
probe saturates or the coupling depletes — neither of which OD measures.
*Proposed replacement:* fence on the **three causes**, each computed:
(i) the weak-probe gate (spec 06 §4.6, Ω_p vs 0.01·min(Γ_e, |Ω_c|));
(ii) the coupling-depletion gate G (§4.3.4, `OD_p·s_c·OD_c/2 ≤ ε_gate`) plus its one-sweep
certificate; and
(iii) a **representability/observability** floor — refuse to report a transduction slope or NEF
when the transmitted power is below the caller's stated detector floor, and refuse to return a
bare T when OD > 708.4 (§4.6-P2). The last one is what audit CRIT-2 was actually about: the NEF
diverged because the *signal* was dead, not because OD crossed 5.
*Consequence if adopted:* RydSim can speak at OD 1–100 — the regime Sedlacek 2012, Jing 2020 and
the NIST metrology work occupy — while refusing more sharply than it does now in the cases that
are genuinely unanswerable.

**A-2 (editorial, closes a real gap) — add the amplitude attenuation coefficient to spec 00 §2.**
The symbol table defines `α = k_p Im χ` as the intensity coefficient and pins
`T = exp(−k_p Im χ L)` as intensity transmission, but the z-ODE needs `α/2` and the table has no
row for it. Propose a row: `α_amp ≡ α/2` [m⁻¹], "amplitude attenuation coefficient; the
coefficient in dΩ/dz = i(k/2)χΩ; never interchangeable with α — see 10 §4.6-P4". Also add
`α_amp` to the §3 collision register under α.

**A-3 (extends R-2, does not relax it) — freeze the velocity grid across a parameter sweep.**
R-2 mandates the uniform/composite grid with halving-convergence. Propose appending: *"When the
same velocity average is evaluated repeatedly across a parameter sweep (Ω_c or Ω_RF along z, an
LO scan, an interpolation node set), the grid must be built once for the widest dressed
structure over the whole parameter box and then held fixed. A grid rebuilt per parameter value
satisfies R-2 pointwise but jitters the parameter dependence; measured, this floors an otherwise
spectrally-convergent interpolant at 4.1e-7 instead of 2.3e-11."* (10 §4.4.2, benchmark 10/N-7.)

---

## 4.11 What I could not source — and what would close it

* **MISSING: a published numerical treatment of *counter-propagating* Maxwell–Bloch with
  coupling depletion in a Rydberg ladder.** Searched this session; found only co-propagating
  treatments. CoOMBE — the reference open-source MBE suite — explicitly restricts to
  *"(co)-propagating"* fields (VERIFIED from its README), and Häupl *et al.* solve a single-field
  problem. The relaxation and collocation schemes of §4.3 are therefore **standard numerical
  practice applied to this problem by me, not a method taken from an EIT paper.** They are
  tagged as such. *What would close it:* the CoOMBE `user_manual.pdf` (§4.13 lists it as
  outstanding) and a targeted literature search in the slow-light / stationary-light community,
  where counter-propagating control fields are standard.
* **MISSING: any published statement of an *automatic* z-convergence criterion for vapour-cell
  propagation.** The one primary source found states the opposite — Häupl *et al.*: *"the value
  of N required for the model to converge must be determined manually."* Rule Z-CONV is
  therefore RydSim's own, not an imported standard.
* **UNVERIFIED: the CoOMBE z-integrator details** are read from the *source* (`mbe.f90`, fetched
  this session) rather than from the paper's numerical-methods section, which the abstract page
  does not contain. The quotes are accurate to the file fetched; the paper-level description was
  not obtained.

---

## 4.12 Sources

**Fetched and quoted this session (VERIFIED):**

* R. M. Potvliege & S. A. Wrathmall, *CoOMBE: A suite of open-source programs for the
  integration of the optical Bloch equations and Maxwell-Bloch equations*, Comput. Phys. Commun.
  **306** (2025); arXiv:2406.19144. README (co-propagating restriction) and `mbe.f90`
  (`mbe_propagate_2`: fixed uniform `z_step = (zmax − zmin)/n_z_steps`, mid-point or 4th-order
  Runge–Kutta start-up, no adaptive halving, Doppler `vmesh` loop nested inside the z loop,
  density matrix recomputed each z step) — github.com/durham-qlm/CoOMBE.
* D. R. Häupl, C. R. Higgins, D. Pizzey, J. D. Briscoe, S. A. Wrathmall, I. G. Hughes, R. Löw,
  N. Y. Joly, *Modelling spectra of hot alkali vapour in the saturation regime*, arXiv:2410.19916
  (New J. Phys. 2025). Full text extracted from the fetched PDF: §2.5 propagation (10 equal
  slices over a 2 mm cell; Eq. (11) `dI/dz = (4π/λ)Im{√(1+χ[I(z),Δ,n])}I(z)`; *"for 10 slices
  our simulation converges to an acceptable level of accuracy"*; *"the value of N required for
  the model to converge must be determined manually"*; *"the current model only allows the
  calculation of the propagation-corrected absorption for a single value of the detuning Δ at a
  time"*), and §2.4's convolution substitution *"v′_z = Δ/k − v_z … χ(v′_z) must only be computed
  once and can be reused for multiple values of Δ"*.
* SciPy reference, `scipy.integrate.solve_bvp` — *"4th order collocation algorithm with the
  control of residuals"*, *"damped Newton method with an affine-invariant criterion function"*,
  `norm(r/(1 + abs(f))) < tol` with `r = y' − f(x,y)`, defaults `tol=1e-3`, `max_nodes=1000`,
  status codes 0/1/2; refs Kierzenka & Shampine (2001), Shampine, Muir & Xu (2006), Ascher,
  Mattheij & Russell (1995).
* SciPy reference, `scipy.integrate.solve_ivp` — DOP853 (*"Explicit Runge-Kutta method of order
  8"*), Radau (*"implicit Runge-Kutta method of the Radau IIA family of order 5"*), BDF, LSODA
  (*"automatic stiffness detection and switching"*); local error controlled as
  `atol + rtol*abs(y)`; defaults `rtol=1e-3`, `atol=1e-6`.

**Binding in-project documents (normative, read this session):** `00-conventions.md` (locks
#1–#5, #10, #12, #18; rulings R-2, R-21, R-22); `00-integrity-audit.md` (§3 refusals 18–23; §4
provenance items 5–8; risk rows R2, R4); `05-vapor-cell-physics.md` §2.d, §2.f, §2.g, §4;
`06-optical-bloch-eit.md` §2.4, §4.4, §4.6, §7.2; `src/rydsim/eit.py`;
`src/rydsim/experiment.py` (`max_optical_depth`, `superhet_transfer`); `src/rydsim/lindblad.py`.

**Standard practice, recalled not re-checked (LITERATURE-RECALL):** wave/Gauss–Seidel relaxation
for coupled counter-propagating field equations; Anderson acceleration for fixed-point
iterations; exponential integrators and the scalar Magnus truncation; Chebyshev interpolation's
spectral convergence for analytic functions on an interval.

---

## 4.13 Reproduction harness (must be ported to `tests/test_spec10_numerics.py` before release)

All `SELF-MEASURED` numbers above come from these scripts, run this session against the shipped
tree at `src/rydsim`. Per integrity-audit **R4** they are unreproducible assertions until ported.

* **A — RK4 step law (§4.2.2, 10/N-6).** Integrate `dy/dz = −a y` with classical RK4 to
  total amplitude exponent X = 10 (OD = 20) at n ∈ {20, 50, 100, 200, 500, 1000}; compare
  |y(L)| to `exp(−10)`.
* **B — ρ_ee vs Ω_p (§4.3.4).** `rydsim.lindblad.LadderSystem(omegas=[2π·Ω_p, 2π·5e6],
  deltas=[0,0], decays=[0, 2π·6.0666e6, 2π·3e3], dephasings=[0,0,2π·50e3], transit=2π·50e3)`;
  `.steady_state()[1,1].real` at Ω_p/2π ∈ {1e3, 1e5, 1e6} Hz.
* **C — loop sensitivity s_c (§4.3.2).** Central difference of
  `ln Re[doppler_average(...)]` in `ln Ω_c` at h = 1e-3, Δ_p = 0, on a frozen grid.
* **D — dynamic range (§4.2.3).** `doppler_average` over 601 detunings, ±2π·1.5 GHz, frozen
  4.01e5-node grid; ratio of max Re S̄ to the ±1.5 GHz wing value.
* **E — Chebyshev interpolant (§4.4.2, 10/N-7).** Chebyshev–Gauss nodes in Ω_c over
  [0.3, 1.0]×2π·5 MHz, M ∈ {4…24}, 201 detunings ±2π·30 MHz, 17 uniform-random test points
  (seed 0); run twice — once with `v_grid` supplied from
  `resonance_refined_vgrid` at the largest Ω_c (frozen, 42 281 nodes) and once letting
  `doppler_average` rebuild it per node.
* **F — float64 underflow (§4.6-P2).** `np.log(np.finfo(float).tiny)`; `np.exp(-745.0)`,
  `np.exp(-746.0)`.

---

*GreyNOC · RydSim spec 10 §4 (numerics) · house rule: reproducible or it didn't happen.*


---

## Provenance of this draft section
### Sources FETCHED this session
- CoOMBE README (github.com/durham-qlm/CoOMBE, fetched this session) — VERIFIED that the Maxwell-Bloch module is restricted to 'one or two laser fields (co)-propagating in an atomic vapour', i.e. the reference open-source MBE suite does NOT solve the counter-propagating case. Used in §4.3.1 and §4.11 to establish that RydSim's counter-propagating BVP is genuinely unsupported by standard tooling.
- CoOMBE mbe.f90 source (raw.githubusercontent.com/durham-qlm/CoOMBE/main/mbe.f90, fetched this session) — VERIFIED z-integrator details: routine mbe_propagate_2; fixed uniform step z_step = (zmax - zmin)/n_z_steps; mid-point formula or '4th-order Runge-Kutta formula' for start-up; NO adaptive step-size halving; Doppler velocity loop (vmesh, fMvweight) nested INSIDE the z loop; density matrix recomputed at each z step. Used in §4.2.1 as prior art and in §4.4.1 as the naive-nesting cost model.
- Haupl, Higgins, Pizzey, Briscoe, Wrathmall, Hughes, Low, Joly, 'Modelling spectra of hot alkali vapour in the saturation regime', arXiv:2410.19916 (New J. Phys. 2025) — full PDF fetched and text-extracted this session. VERIFIED quotes taken: (a) §2.5 'We divided the cell into 10 equal slices, along the laser propagation axis, such that we can consider ten vapour cells each of length 0.2 mm'; (b) Eq. (11) dI/dz = (4pi/lambda) Im{sqrt(1+chi[I(z),Delta,n])} I(z) — used in §4.6-P4 as independent corroboration of the intensity-vs-amplitude factor-2 bookkeeping; (c) 'We noticed that for 10 slices our simulation converges to an acceptable level of accuracy... the value of N required for the model to converge must be determined manually' — used in §4.6-P1 as the named prior-art instance of the silent-under-resolution hazard; (d) 'the current model only allows the calculation of the propagation-corrected absorption for a single value of the detuning Delta at a time' — used in §4.2.3 to justify per-detuning z-grids; (e) §2.4 'The substitution v'_z = Delta/k - v_z reduces the computational burden, as chi(v'_z) must only be computed once and can be reused for multiple values of Delta' — used in §4.4.3 as the caching trick that is provably NOT transferable to a mismatched ladder.
- SciPy reference, scipy.integrate.solve_bvp (docs.scipy.org, fetched this session) — VERIFIED: '4th order collocation algorithm with the control of residuals'; 'damped Newton method with an affine-invariant criterion function'; residual r = y' - f(x,y) with norm(r/(1+abs(f))) < tol per mesh interval; default tol=0.001, max_nodes=1000; status codes 0 (converged), 1 (max mesh nodes exceeded), 2 (singular Jacobian); refs Kierzenka & Shampine 2001, Shampine/Muir/Xu 2006, Ascher/Mattheij/Russell 1995. Used to specify Scheme B in §4.3.3 including the rejection of the default tol.
- SciPy reference, scipy.integrate.solve_ivp (docs.scipy.org, fetched this session) — VERIFIED: DOP853 = 'Explicit Runge-Kutta method of order 8'; Radau = 'implicit Runge-Kutta method of the Radau IIA family of order 5'; LSODA = 'Adams/BDF method with automatic stiffness detection and switching'; local error controlled as 'atol + rtol * abs(y)'; defaults rtol=1e-3, atol=1e-6; guidance 'Explicit Runge-Kutta methods should be used for non-stiff problems and implicit methods for stiff problems'. Used in §4.2.1 and §4.2.3.
- In-project normative documents read in full this session: docs/spec/00-conventions.md (20 locks + R-1..R-28), docs/spec/00-integrity-audit.md (risk register, refusal list §3, provenance requirements §4), docs/spec/05-vapor-cell-physics.md (§2.d velocity quadrature, §2.f propagation incl. the existing dOmega_p/dz = i(k_p/2)chi Omega_p equation and the |dOD| <= 0.05 step rule, §4 pitfalls), docs/spec/06-optical-bloch-eit.md (§2.4 susceptibility chain, §4.4 halve-AND-widen convergence rule, §4.6 weak-probe gate, §7.2 thin-medium limitation), docs/spec/09-validation-corpus.md (grading classes TIGHT/ORDER/QUALITATIVE).
- Shipped RydSim source read this session: src/rydsim/eit.py (chi_ladder, resonance_refined_vgrid, doppler_average, chi_si, transmission), src/rydsim/experiment.py (LadderConfig.max_optical_depth = 5.0 and its comment 'Beer-Lambert with the EIT-suppressed chi is exact at any OD', superhet_transfer and its IntegrityError gate), src/rydsim/lindblad.py (LadderSystem API). Used to establish the branch structure of §4.1 and the amendment proposal A-1.

### UNVERIFIED / recall-only
- SELF-MEASURED (this session, shipped code, harness NOT yet in repo — integrity-audit R4 blocker): RK4 global relative amplitude error at OD=20 for per-step dOD in {1.00, 0.40, 0.20, 0.10, 0.04, 0.02} = {7.949e-3, 1.576e-4, 9.058e-6, 5.430e-7, 1.356e-8, 8.403e-10}, fitted order 4.00, and the derived closed form |dy/y| ~ OD*dOD^4/3840. Reproducible from script §4.13-A but not yet a test.
- SELF-MEASURED (harness not in repo): Chebyshev-in-Omega_c interpolation error of the Doppler-averaged response — frozen velocity grid 3.766e-4 / 3.296e-5 / 2.804e-7 / 2.473e-9 / 2.343e-11 at M = 6/8/12/16/20, versus rebuilt-per-node grid which FLOORS at 6.365e-7 (M=16) / 4.126e-7 (M=24). This is the headline number behind the M=16 default and Rule V-FREEZE; it must be ported to tests before it can gate anything.
- SELF-MEASURED (harness not in repo): probe-scan absorption dynamic range 4.40e3 (max Re S-bar 1.010e-9 at Delta_p/2pi = -5.0 MHz vs 2.297e-13 at +/-1.5 GHz), Rb-87 300 K, Omega_c/2pi = 5 MHz, 601 detunings, 4.01e5 velocity nodes. Drives the per-detuning z-grid rule.
- SELF-MEASURED (harness not in repo): loop sensitivity dln(Re S-bar)/dln(Omega_c) at the EIT peak = -0.515 / -0.825 / -0.986 at Omega_c/2pi = 2.5 / 5 / 10 MHz. Used only as an expectation for relaxation contraction; the code must still MEASURE q_m.
- SELF-MEASURED (harness not in repo): rho_ee = 9.4927e-10 / 9.4901e-6 / 9.2464e-4 at Omega_p/2pi = 1 kHz / 100 kHz / 1 MHz (rydsim.lindblad, resonant, v=0). Supports but does not substitute for the computed coupling-depletion gate.
- SELF-MEASURED (harness not in repo): float64 exp(-OD) subnormal onset at OD = 708.3964185322641 and flush-to-zero at OD ~ 745.2 (exp(-745) = 5e-324, exp(-746) = 0.0).
- SELF-MEASURED, hardware-dependent, anchor only: 0.442 s per velocity-averaged spectrum at 201 detunings x 42281 velocity nodes on the shipped rydsim.eit path (~52 ns per detuning-velocity pair). All complexity figures in §4.4.4 (265 s -> 7.1 s, 37.5x) are scaled from this single machine measurement.
- UNVERIFIED / UNMEASURED: the 2-D tensor Chebyshev interpolant in (Omega_c, Omega_RF) is ASSUMED to inherit the measured 1-D spectral rate per axis. Only the 1-D rate was measured. Benchmark 10/N-7b exists precisely so this is not quietly assumed.
- UNVERIFIED as a theorem: that Gauss-Legendre node sets GL_n and GL_2n share no nodes. The section does not rely on it — a runtime assertion (no refined node within 1e-12*L of a coarse node, else refine to GL_{2n+1}) replaces the claim.
- LITERATURE-RECALL, not re-checked: wave/Gauss-Seidel relaxation for counter-propagating coupled field equations; Anderson acceleration; the scalar-generator Magnus truncation; spectral convergence of Chebyshev interpolation for analytic functions. These are standard numerical practice applied by me to this problem, NOT methods taken from a specific EIT/Maxwell-Bloch paper — no such paper was found (see open questions).
- MISSING: any published numerical treatment of counter-propagating Maxwell-Bloch with coupling depletion in a Rydberg ladder. Searched this session; CoOMBE is explicitly co-propagating-only and Haupl et al. solve a single-field problem. The BVP schemes of §4.3 are therefore unsourced-by-necessity and tagged as such in-document.
- MISSING: the CoOMBE user_manual.pdf and the paper's numerical-methods section. The z-integrator facts quoted are read from the Fortran source, not from the published method description; the abstract landing page carried no numerical detail.
- NORMATIVE-BY-FIAT (this spec, not sourced): eps_z = 1e-4, eps_relax = 1e-4, eps_gate = 1e-5, q_max = 0.5, m_max = 20, err_interp <= 1e-6, solve_bvp tol = 1e-8, solve_ivp rtol/atol = 1e-10/1e-12. These are chosen for commensurability with spec 06 §4.4's existing 1e-4 velocity rule; they are design decisions, not measurements.

### Open questions
- ADJUDICATION NEEDED (amendment A-1, substantive): the corpus carries three mutually inconsistent optical-depth fences — integrity-audit §3 refusal #18 says OD > 0.1 must raise ThickCellError; spec 05 §2.f says OD_peak <= 0.1 for the single-exponential path; and rydsim.experiment.LadderConfig.max_optical_depth ships at 5.0 with a comment asserting the thin-cell answer is 'exact at any OD'. §4.1 and benchmark 10/N-1 establish the code comment is correct in the weak-probe undepleted-coupling branch. Should the fence move off OD entirely and onto the three computed causes (weak-probe gate; coupling-depletion gate G; signal-representability/observability floor)? This is the single decision that determines whether RydSim can speak at the OD 1-100 of Sedlacek 2012 / Jing 2020 / NIST.
- ADJUDICATION NEEDED (amendment A-2, editorial but load-bearing): spec 00 §2 defines alpha = k_p Im chi as the INTENSITY coefficient but has no row for the amplitude coefficient alpha/2 that the z-ODE actually uses. Add a row (alpha_amp) and a §3 collision-register entry, or accept that every implementer re-derives the factor 2?
- ADJUDICATION NEEDED (amendment A-3, extends R-2): should R-2 be appended with the frozen-grid requirement for parameter sweeps? Measured impact: a grid rebuilt per Omega_c value satisfies R-2 pointwise yet floors an otherwise spectrally-convergent interpolant at 4.1e-7 instead of 2.3e-11. Without this clause an implementer can satisfy every existing rule and still lose four decades.
- UNRESOLVED LITERATURE GAP: no published numerical treatment of counter-propagating Maxwell-Bloch with coupling depletion in a Rydberg ladder was found this session. CoOMBE is explicitly co-propagating-only; Haupl et al. solve a single field. Should someone search the slow-light / stationary-light literature (where counter-propagating control fields are standard) before §4.3's schemes ship as unsourced-by-necessity? The CoOMBE user_manual.pdf is also still unfetched and may describe a counter-propagating extension.
- R4 PORTING OBLIGATION: every SELF-MEASURED number in §4.7 (the RK4 error law, the M=16 -> 2.5e-9 interpolation figure, the 4.40e3 dynamic range, s_c, rho_ee, the underflow thresholds, the 0.442 s unit cost) currently rests on scratchpad scripts. Integrity-audit R4 says such numbers are unreproducible assertions. Who ports §4.13's harness to tests/test_spec10_numerics.py, and does spec 10 ship with those rows marked non-gating until then?
- OPEN DESIGN QUESTION: the 0.442 s unit cost and therefore the whole 265 s -> 7.1 s complexity argument come from ONE machine, single core, this session. Should the interpolation-node default M_c be set by a cost/accuracy optimisation at runtime (given the caller's eps_z) rather than fixed at 16? The measured convergence table makes this straightforward but it is a design decision I did not take.
- OPEN PHYSICS-ADJACENT QUESTION: Scheme A diverging (q_m >= 1) is specified to RAISE, on the reasoning that loop gain >= 1 means the medium is bistable/self-focusing and therefore outside the scalar single-shell model (§4.9 item 1). Is that the right reading, or is there a parameter regime where divergence is purely a numerical artefact of Gauss-Seidel ordering and Anderson acceleration would recover a legitimate fixed point? I found no source either way; the conservative refusal is what I specified.
- OPEN: the RF field profile Omega_RF(z) is treated here as a supplied input (§4.9 item 4). If the cell-EM module ever supplies a real standing-wave profile, the 2-D interpolant becomes mandatory and 10/N-7b must be measured first. Who owns that handoff — spec 05 (cell) or spec 07 (screening/interface)?

### Proposed benchmarks

| id | quantity | expected | tol | source | conf |
|---|---|---|---|---|---|
| 10/N-1 | R-THIN reduction: thick-solver ln T and phase vs the shipped thin-cell chain (eit.doppler_average -> chi_si -> transmission), branch S0 (weak-probe gate holds, coupling undepleted), evaluated at OD in {1e-6, 1e-3, 0.1, 1, 5, 50, 500} | identical at EVERY OD, not merely as OD -> 0: ln T = -k_p L Im chi_bar and phi = (1/2) k_p L Re chi_bar | <= 1e-12 relative in ln T | spec 10 §4.1/§4.5; exact because chi_bar carries no z-dependence in that branch, so any consistent quadrature integrates a constant exactly. Anchored on the already-validated rydsim.eit chain and on the LadderConfig comment 'Beer-Lambert with the EIT-suppressed chi is exact at any OD'. | VERIFIED (by construction) - this is the free regression test against code already validated |
| 10/N-2 | Density scaling of the depleted-coupling correction: log-log slope of \|ln T_thick - ln T_thin\| versus number density N over one decade, depletion enabled | slope 2 (OD_c ~ N and delta OD = OD_p * delta chi/chi ~ N^2) | fitted slope >= 1.9 | spec 10 §4.5 weak form; derived. Slope 1 indicates a first-order correction where none exists; slope 0 indicates depletion is not wired in. | derived, self-checked (not literature) |
| 10/N-3 | Exponent identity, isolating the factor-2 / sign traps: propagate a constant purely-imaginary chi_bar = i*chi'' and then a constant purely-real chi_bar = chi' | ln T = -k_p chi'' L ; and for real chi': ln T = 0 with phi = k_p L chi'/2 | <= 1e-14 relative | spec 00 lock #3 (fields vs intensity), spec 00 §2 alpha row, spec 05 §2.f ('no extra 2 or 4pi'); corroborated by Haupl et al. arXiv:2410.19916 Eq. (11) dI/dz = (4pi/lambda) Im{sqrt(1+chi)} I, in which the two factors of 2 cancel | VERIFIED (identity; corroborating primary source fetched this session) |
| 10/N-4 | Sign of the counter-propagating coupling term: monotonicity of \|Omega_c(z)\| and the entry/exit amplitude ratio, with depletion on and Im chi_c > 0 | \|Omega_c(z)\| monotone non-decreasing in z (largest at z = L, the coupling entry face); \|Omega_c(0)\|/\|Omega_c(L)\| = exp(-OD_c/2) | monotonicity QUALITATIVE and always-on; ratio <= 1e-10 relative | spec 10 §4.6-P3 derivation: dOmega_c/dz = -i(k_c/2) chi_c Omega_c gives d\|Omega_c\|^2/dz = +k_c Im(chi_c)\|Omega_c\|^2 > 0. Copying the probe's +i sign manufactures unphysical gain. | VERIFIED (derivation) |
| 10/N-5 | Rule Z-CONV: z-quadrature convergence under a node-disjoint refinement (GL_n -> GL_2n, or adaptive tol / 100), max over scanned Delta_p of \|OD_fine - OD_coarse\|; plus a deliberately under-resolved mutant | converged run <= 1e-4; mutant raises rydsim.provenance.IntegrityError | eps_z = 1e-4 ABSOLUTE in OD (equivalently relative in T, uniformly across the scan and meaningful where T underflows) | spec 10 §4.2.4, mirroring spec 06 §4.4's halve-AND-widen rule (domain fixed at L, so node-disjointness replaces widening). Motivated by audit R2's warning that a nested doubling criterion can falsely converge. | NORMATIVE (this spec); the mutation arm makes the refusal falsifiable |
| 10/N-6 | RK4 fixed-step law in branch N: global relative amplitude error at total OD = 20 for per-step dOD in {0.40, 0.20, 0.10, 0.04} | 1.576e-4 / 9.058e-6 / 5.430e-7 / 1.356e-8; fitted convergence order 4.00; closed form \|dy/y\| = OD*dOD^4/3840 | order 4.00 +/- 0.05; each constant within 15% of the closed form | spec 10 §4.2.2, measured this session (script §4.13-A). Corroborates spec 05 §2.f's existing \|dOD\| <= 0.05 step rule, which this law shows delivers 1.6e-7 at OD = 100 and 1.1e-6 at OD = 700. | SELF-MEASURED this session - harness NOT yet in repo; per integrity-audit R4 this cannot gate a release until ported |
| 10/N-7 | Chebyshev-in-Omega_c interpolant of the velocity-averaged response at M = 16, run BOTH with a frozen velocity grid (Rule V-FREEZE) and with the grid rebuilt per node | frozen grid 2.473e-9 (converging spectrally: 2.804e-7 at M=12, 2.343e-11 at M=20); rebuilt grid 6.365e-7 and FLOORED (4.126e-7 at M=24) | frozen <= 1e-8 AND rebuilt >= 1e-7 - the test must prove the freeze rule matters, not merely assert it | spec 10 §4.4.2, measured this session (script §4.13-E) on the shipped rydsim.eit path; Rb-87 300 K counter-prop 780.241/480.0 nm, 201 detunings +/-2pi*30 MHz, Omega_c/2pi in [1.5, 5.0] MHz, 42281-node frozen grid, 17 random test points (seed 0) | SELF-MEASURED this session - harness NOT yet in repo (R4 blocker); this is the headline number behind the M=16 default |
| 10/N-7b | 2-D tensor Chebyshev interpolant in (Omega_c, Omega_RF) at M_c = M_RF = 16 | <= 1e-6 (err_interp budget = eps_z/100) | 1e-6 | spec 10 §4.4.2. The per-axis spectral rate is ASSUMED from the measured 1-D case; this benchmark exists so the assumption cannot be quietly adopted. | UNMEASURED - must be measured before the 2-D cache is used in production |
| 10/N-8 | High-OD representation: request OD = 750 | returns ln T = -750 with underflow=True; NEVER a bare 0.0 for T | exact; float64 anchors exp(-745) = 5e-324, exp(-746) = 0.0, subnormal onset at OD = 708.3964185322641 | spec 10 §4.6-P2; measured float64 thresholds (script §4.13-F). Consistent with spec 05 §4.7's existing 'guard exp(-OD) for OD > 700, return 0.0 explicitly' as the DISPLAY rule. Prevents the audit CRIT-2 failure mode where a dead signal produced a finite-looking NEF. | SELF-MEASURED (thresholds); NORMATIVE (the log-space return contract) |
| 10/N-9 | Relaxation behaviour on a fixture with OD_c ~ 0.3: first-sweep residual r_1 (positive control) then convergence r_m and contraction q_m | r_1 > eps_relax (proving the correction is nonzero and actually applied), THEN r_m <= 1e-6 within 5 sweeps with q_m <= 0.5 | eps_relax = 1e-4 absolute in OD for acceptance; r_m <= 1e-6 within m <= 5; q_m <= 0.5 | spec 10 §4.3.2 and pitfall §4.6-P9 (a relaxation that 'converges' because the update never took effect). The positive control is what distinguishes a working iteration from a no-op. | NORMATIVE (this spec) |
| 10/N-10 | Forward-only gate G is a genuine upper bound: measured \|delta OD\| from one relaxation sweep versus the a-priori bound OD_p * s_c * OD_c / 2, on >= 5 fixtures spanning OD_p in [0.1, 30] | measured \|delta OD\| <= bound on EVERY fixture | one-sided; any single violation is a FAIL (a violated bound is a wrong bound, not a tolerance question) | spec 10 §4.3.4. s_c is measured by finite-differencing the Omega_c interpolant already built; fixture sensitivity \|dln chi_bar_p/dln Omega_c\| measured at 0.515-0.986. | NORMATIVE (this spec); the mandatory one-sweep certificate converts the bound into a measurement |
| 10/N-11 | Per-detuning z-grids are required: a single shared z-grid sized in the probe-scan wings, checked against Rule Z-CONV | shared grid fails Z-CONV by much more than eps_z, given the measured probe-scan absorption dynamic range of 4.40e3 | QUALITATIVE predicate (shared-grid max \|delta OD\| > 100 * eps_z) | spec 10 §4.2.3; dynamic range measured this session (script §4.13-D). Corroborated by Haupl et al.: 'the current model only allows the calculation of the propagation-corrected absorption for a single value of the detuning Delta at a time.' | SELF-MEASURED (dynamic range) + VERIFIED (the corroborating quote) |
| 10/N-12 | Cross-solver agreement: branch S (Gauss-Legendre z-quadrature on ln Omega_p) versus branch N (DOP853 on ln Omega_p) inside the weak-probe overlap region | identical OD | <= 1e-9 absolute in OD | spec 10 §4.1/§4.2.1; method-A-vs-B self-check, mirroring spec 06 benchmark B-1's analytic-vs-Lindblad lock | VERIFIED (by construction) |
| 10/N-13 | Provenance completeness of any optically-thick run | record carries converged, max_delta_od, quadrature rule + node counts (coarse and fine), the full relaxation trace [(m, r_m, q_m)], err_interp, OD_p, OD_c, s_c, solver branch, and underflow flag | all fields present; empty list only where genuinely empty | spec 00 integrity-audit §4 items 5-8 ('Convergence records: ... z-step halving result - as data (converged: bool + magnitudes), not docstrings') | NORMATIVE (existing corpus requirement, applied to this module) |
