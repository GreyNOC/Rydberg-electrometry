# 00 — Normative Conventions, Symbol Registry, and Conflict Rulings

**RydSim physics specification, document 00. Status: NORMATIVE — this document overrides
every other spec (01–09) wherever they disagree.** Produced by the chief-theorist consistency
review of 2026-08-10 over the full spec set. Every conflict found between documents is recorded
in §5 with a binding ruling; the implementation follows the ruling, not the original text.
Every ruling that changes a number was re-verified numerically during this review.

Precedence order: **00 > the owning spec (per the ownership map, §4) > any other spec's
convenience copy.** A spec's restatement of another spec's quantity is never authoritative.

---

## 1. Global convention locks (the non-negotiables)

1. **Units: SI internally, everywhere.** The single exception is `rydsim.radial` (spec 02),
   which computes in Hartree atomic units internally and converts at its API boundary
   (radial integrals returned in a₀; the caller multiplies by a₀ to get metres).
   Display/IO units (cm⁻³, V/cm, MHz/(V/cm)², nV/cm/√Hz, torr) exist **only** in I/O helpers
   and reports — never inside physics code.
2. **Angular frequencies internally.** Every rate, Rabi frequency, detuning, and linewidth
   inside the code is angular, rad/s (`Omega_*`, `Delta_*`, `Gamma_*`, `gamma_*`,
   `delta_*`). Hz appears only at API boundaries with the `_hz`/`freq_`/`f_` suffix;
   the conversion is exactly 2π and happens at exactly one site per boundary.
3. **Field convention.** Real field `E(t) = ℰ cos(ωt)` (equivalently ½ℰe^{−iωt} + c.c.);
   **ℰ is the peak amplitude, never RMS.** Intensity `I = ε₀c ℰ²/2`; plane-wave flux
   `S = ℰ²/(2η₀)`. All sensitivity figures (NEF, E_min) are **amplitude** spectral densities.
   Any literature value suspected of an RMS convention (e.g. the √2 in Jing's printed
   transduction formula) is converted before use — see ruling R-22.
4. **Rabi frequency.** `Ω = d·ℰ/ħ` [rad/s], with d the transition dipole moment [C·m] of the
   specific driven pair and ℰ the peak amplitude. Ω is the **full** Rabi frequency: the RWA
   Hamiltonian carries `−ħΩ/2` on off-diagonals; the on-resonance AT peak separation equals
   Ω (rad/s), i.e. Ω/2π in Hz. Never define Ω from the half-amplitude ℰ/2.
5. **Detuning sign.** `Δ = ω_field − ω_atom` (blue positive). Cumulative ladder detunings
   δ₁ = Δ_p, δ₂ = Δ_p + Δ_c, δ₃ = Δ_p + Δ_c + Δ_RF. Weak-probe denominators are
   `(γ − iΔ)`; the literature's `(γ + iΔ)` form is the opposite sign convention — do not mix
   (spec 06 §2.4).
6. **Hamiltonian.** Exactly spec 06 §2.1: `H = −(ħ/2)·[[0, Ωp, …], …]` with 2×cumulative
   detunings on the diagonal, −Ω/2 off-diagonal. Deviations are bugs, caught by benchmark 06/B-1.
7. **Decay-rate bookkeeping (Γ vs γ, FWHM vs HWHM).** `Γ` = population decay rate = 1/τ
   [rad/s ≡ s⁻¹]; for a two-level line Γ is also the Lorentzian **FWHM in angular units**
   (Γ/2π = FWHM in Hz; Rb D2: 2π·6.0666 MHz). `γ_ij` = coherence decay rates
   (HWHM-type): `γ_ij = (Γ_i + Γ_j)/2 + Σ pure dephasings`. Every quoted width is a FWHM
   unless explicitly labelled HWHM. γ_ij are never hand-assembled in production code — they
   are extracted from the collapse-operator set (spec 06 §2.2).
8. **Laser linewidth ↔ dephasing.** Laser linewidths are quoted as Lorentzian **FWHM Δν in
   Hz**. White-frequency-noise equivalent: one-sided S_ν = Δν/π [Hz²/Hz]. The pure-dephasing
   rate contributed to a coherence carrying one photon of that laser is `γ_laser = π·Δν`
   [rad/s]; the Lindblad operator is `√(2γ_laser)·(projector sum)` (specs 04 §2.3.4 / 06 §2.2
   agree). Ladder two-photon coherence: rates **add**
   (`γ_gr,laser = π(Δν_p + Δν_c + 2c√(Δν_p Δν_c))`, correlation c ∈ [−1,1], default 0);
   common-mode noise does NOT cancel in a ladder.
9. **Angular algebra.** Condon–Shortley phases; **Racah convention Wigner–Eckart internally**
   (spec 03 Eq. 2.1). Steck-datasheet reduced elements convert via
   `⟨j‖er‖j'⟩_Racah = √(2j+1)·⟨j‖er‖j'⟩_Steck` (j = bra label). Signed reduced elements and
   radial-integral signs are convention-relative: only |element|² is comparable across codes.
   Spec 03 is the single implementation; no other module reimplements 3j/6j or phases.
10. **Autler–Townes / Doppler factor.** Counter-propagating hot-vapor ladder, **probe scanned**:
    observed splitting `Δf_meas = (λ_c/λ_p)·Ω_RF/2π` (compressed, λ_c/λ_p ≈ 0.615 Rb /
    0.598 Cs); field recovery `E = (h·Δf_AT)/℘` with `Δf_AT = (λ_p/λ_c)·Δf_meas`.
    **Coupling scanned**: factor 1. The factor is computed from the actual state-dependent
    wavelengths at runtime, never hard-coded; benchmark 09/A11 adjudicates it numerically
    (ruling R-1).
11. **Effective RF dipole (NIST convention).** `℘ = e·a₀·R·|A(l,j,m_j=1/2 → l',j',1/2; q=0)|`
    (co-linear π geometry, m_j = ±1/2), per spec 03 §2.5. Non-collinear/elliptical RF requires
    the coherent multi-q sum, never a single ℘.
12. **PSDs and demodulation.** All PSDs are **one-sided**. ENBW bookkeeping per spec 08 §2.7
    (boxcar t ↔ 1/(2t)). `E_min(t) = NEF/√t` (phase-known single quadrature, SNR = 1).
    The four ±3 dB traps (one/two-sided, amplitude/RMS, RBW/ENBW, log-average 2.51 dB) are
    explicit code parameters.
13. **SQL convention.** Default **ħ-convention (Meyer-consistent)**:
    `NEF_SQL = (ħ/μ_RF)/√(N_eff·τ)` continuous; `(ħ/μ_RF)/(τ√(N_eff·R))` pulsed. The
    `convention={"hbar","h"}` flag reproduces printed h-convention results; every SQL output
    is stamped with (N_eff, τ, convention, mode) — ruling R-8.
14. **Stark/polarizability.** `ΔE = −(1/2)αE²`; α > 0 ⇔ level shifts down.
    `α(v,m_j) = α₀ + α₂·[3m_j² − j(j+1)]/[j(2j−1)]` (α₂ ≡ 0 for j ≤ 1/2). AC:
    `ΔE = −(1/4)α(ω)ℰ²`. Unit factors derived at runtime from `scipy.constants`;
    1 a.u. → α/h = 2.488318e-4 Hz/(V/cm)² is a check value, never a hard-coded constant.
15. **Fundamental constants.** Single source: `scipy.constants` (CODATA 2022 vintage as
    shipped); vintage recorded in provenance output. No fundamental constant is ever typed
    by hand in library code; printed values in specs are check values (ruling R-9).
16. **Steck datasheets.** Revision **2.3.4 (8 Aug 2025)** is canonical for all D-line,
    hyperfine, mass, and vapor-pressure data. Older-revision values (Cs 6P₃/₂ 30.473 ns,
    Rb dipole uncertainty (87), …) are superseded (ruling R-25).
17. **Liouvillian.** Column-major vectorization (`order='F'`), trace-row replacement scaled by
    s = max|L|, per spec 06 §2.3. Steady-state acceptance gates: |Tr σ − 1| ≤ 1e-10,
    ‖σ − σ†‖_max ≤ 1e-10, min eig ≥ −1e-10 (ruling R-18).
18. **Velocity averaging.** 1-D Doppler with `f1D(v)`, σ_v = √(k_BT/m). For any spectrum
    containing EIT/AT structure the quadrature is the **uniform or composite grid** of spec 05
    §2.d / 06 §4.4 with mandatory halving-convergence; **Gauss–Hermite is forbidden as the
    sole method** for EIT (permitted only for smooth EIT-free Voigt absorption) — ruling R-2.
19. **Transit.** Measure-and-replace Lindblad channel toward the ground state at rate
    `γ_t = √(2 ln 2)·v⊥/w₀` [rad/s] per transverse-velocity class (thermal shortcut:
    v⊥ → ⟨v⊥⟩ = √(πk_BT/2m)); w₀ is always the 1/e² **intensity** radius. FWHM equivalent
    `Δν_tt = γ_t/π` — ruling R-3.
20. **No fabrication.** Every constant carries source + confidence tag; UNVERIFIED items never
    gate a release; benchmark tolerances are changed only by editing spec 09 with rationale.

---

## 2. Master symbol table

One row per symbol as used across specs 01–09. "Owner" = the spec whose definition is
normative. Units are the internal (SI) units; display units in parentheses.

| Symbol | Meaning | SI unit (internal) | Chosen convention (normative) | Owner |
|---|---|---|---|---|
| n | principal quantum number | — | integer ≥ n_min per series | 01 |
| l, s, j, m_j | orbital / spin / total electronic AM and projection | — | s = 1/2; j = l ± 1/2; Condon–Shortley phases | 01/03 |
| I, F, m_F | nuclear spin; total AM incl. nucleus | — | F = J + I. **F is never the electric field** (see E) | 01/03 |
| δ_lj(n) | quantum defect | — | Ritz Form A (Rb all; Cs P) or Form B fixed-point (Cs S, D, F, G) per spec 01 §2.2 — forms not interchangeable | 01 |
| n* (n_eff, ν) | effective principal quantum number | — | n − δ_lj(n), from spec 01 machinery (incl. Form B where assigned) | 01 |
| R_∞, R_M | Rydberg constant; mass-corrected | m⁻¹ | R_M = R_∞/(1 + m_e/M_atom) — **atom-mass convention** (ruling R-10); c·R_∞ = 3 289 841.960 **GHz** (ruling R-12) | 01 |
| E_I | ionization energy | J (quoted MHz via /h) | referenced to ground **hyperfine centroid**: E_I(centroid) = E_I(from F) **+** ΔE_hfs(F) (sign — ruling R-11) | 01 |
| E_nlj, E_b | level / binding energy | J | E = E_I − hcR_M/n*²; E_b = hcR_M/n*² > 0; intervals from binding energies directly | 01 |
| ΔE_hfs(F) | hyperfine shift from FS centroid | J | Casimir formula spec 01 Eq. (1.7), denominator 4I(2I−1)J(2J−1) | 01 |
| A, B, C (hfs) | hyperfine constants | Hz (as X/h) | Steck rev 2.3.4 values; Rydberg scaling A(nS) = A_S·n*⁻³ | 01 |
| M, m | atomic mass; electron mass | kg (u at I/O) | AME/Steck rev 2.3.4 digits | 01 |
| μ_mass | electron reduced-mass ratio | — | **single definition** μ = 1/(1 + m_e/M_atom) = R_M/R_∞ (ruling R-10) | 01 |
| R (= R_{nlj}^{n'l'j'}) | radial dipole integral ∫P r P' dr | m (a₀ at radial API) | P = r·R_radial; sign convention-laden, only |R| comparable; per-(j,j') pair | 02 |
| x, X(x), h_x | Numerov grid √r, scaled wavefn, step | a₀^½ (a.u.) | x = √r, X = R·r^{3/4}, uniform h_x = 0.001 default. Step symbol is **h_x**, never bare h (Planck clash) | 02 |
| α_c | core polarizability (model potential) | a₀³ (a.u.) | Rb 9.0760, Cs 15.6440 (MSD94) | 02 |
| V_l(r), V_so | model potential, spin-orbit term | Hartree (a.u.) | MSD94 + spec 02 Eq. (2.4); l ≥ 4 pure Coulomb | 02 |
| A(l,j,m;…;q) | dimensionless angular factor | — | spec 03 Eq. (2.9), Racah chain; n-independent, cached | 03 |
| ⟨j‖er‖j'⟩ | reduced dipole element | C·m (e·a₀ at I/O) | Racah convention internal; Steck→Racah ×√(2j+1) | 03 |
| d, ℘, μ_RF | transition dipole moment (specific pair + polarization) | C·m | one physical quantity, three legacy symbols; code name `dipole_Cm` / `mu_rf`. NIST effective RF dipole per lock #11 | 03 |
| q | spherical polarization index | — | q ∈ {−1,0,+1} = σ⁻/π/σ⁺; r_q per spec 03 §2.3; Δm = q | 03 |
| S_FF' | hyperfine line strength | — | (2F'+1)(2J+1)·6j²; Σ_F' S_FF' = 1; convention-free ratio | 03 |
| d_eff,far | hot-vapor two-level probe dipole | C·m | ⟨J‖er‖J'⟩_Steck/√3 (π, unresolved excited HFS); used for BOTH Ω_p and the χ prefactor | 03 |
| f | oscillator strength | — | spec 03 Eq. (2.17); TRK Σf = 1 hydrogenic; alkali sanity band 0.95–1.10 | 03 |
| A_{e→g} | Einstein A coefficient | s⁻¹ | ω³|⟨g‖er‖e⟩_R|²/(3πε₀ħc³(2j_e+1)); (2j_e+1) = **upper** degeneracy | 03/04 |
| τ, τ₀, τ_eff | lifetimes (radiative; with BBR) | s (ns at I/O) | 1/τ₀ = ΣA; τ₀ = τ_s·n_eff^δ fits valid 15 ≤ n ≤ 80 | 04 |
| Γ_e, Γ_r, Γ_r' | population decay rates of e, r, r' | rad/s | lock #7; Γ_r includes BBR when stated | 04 |
| n̄(ω,T) | Planck occupation | — | 1/(e^{ħω/k_BT} − 1); via expm1 | 04 |
| Γ_BBR, W_BBR_ion | BBR depopulation; direct BBR photoionization | s⁻¹ | Beterov full sum + Eq. (14)/(27) fits with verbatim coefficients | 04 |
| γ_ge, γ_gr, γ_er | coherence decay rates | rad/s | assembled from collapse set only (lock #7) | 06 |
| γ_t | transit reset rate | rad/s | lock #19 (ruling R-3) | 05 |
| β (self-broadening) | resonant self-broadening coefficient | Hz·m³ internal (Hz·cm³ at I/O) | γ_ge += β·N/2; Rb D2 β/2π = 1.03e-7 Hz·cm³ theory, 1.10(17)e-7 meas (ruling R-6) | 04 |
| C₆ | van der Waals coefficient | stored as C₆/h [Hz·m⁶] (GHz·µm⁶ I/O) | Δν_vdW = (C₆/h)/r⁶ — no second /h (ruling R-26) | 04 |
| a_s | e⁻–ground scattering length (Fermi shift) | m (a₀ I/O) | Δν = (ħ a_s/m_e)·N [Hz] | 04 |
| P_v(T), N | vapor pressure; number density | Pa; **m⁻³** | Alcock/Steck two-branch log₁₀ fit; N = P_v/(k_BT); density symbol is N (n reserved) — ruling R-21 | 05 |
| η_iso, p_F | isotope fraction; ground-F population | — | Steck abundances; p_F = (2F+1)/Σ(2F+1) | 05 |
| v_p, σ_v, u | most-probable speed; 1-D σ; (u ≡ v_p) | m/s | v_p = √(2k_BT/m); σ_v = v_p/√2; spec 06's u = v_p | 05 |
| Δν_D | Doppler FWHM | Hz | (2/λ)√(2 ln2 k_BT/m) | 05 |
| k_p, k_c | probe/coupling wavenumbers | m⁻¹ (rad/m) | 2π/λ; Δ_p(v) = Δ_p − k_p v; counter-prop: Δ_c(v) = Δ_c + k_c v; δ₂(v) = Δ_p+Δ_c+(k_c−k_p)v | 05/06 |
| λ_p, λ_c | probe/coupling vacuum wavelengths | m | λ_c computed from spec 01 energies at runtime; 480.0 nm (Rb) / 509.4 nm (Cs) only inside declared benchmark fixtures (ruling R-15) | 01 |
| χ | probe electric susceptibility | — | χ = i(N d_ge²/ε₀ħ)/D(Δ_p), (γ − iΔ) denominators; Doppler-averaged over ρ_eg(v) | 06 |
| α (absorption) | absorption coefficient | m⁻¹ | α = k_p·Im χ; **T = exp(−k_p Im χ L)** is INTENSITY transmission; OD = k_p Im χ L | 05 |
| σ₀ | resonant cross-section | m² | 2ωd²/(cε₀ħΓ) = 3λ²/2π (closed transition) | 05 |
| I_sat | saturation intensity | W/m² | Steck; must be paired with the matching d (cycling vs iso vs π-detuned) | 01/05 |
| w₀ | beam radius | m | 1/e² **intensity** radius (= 1/e field radius) always | 05 |
| I₀, ℰ₀ | peak intensity, peak field | W/m², V/m | I₀ = 2P/(πw₀²); ℰ₀ = √(2I₀/ε₀c) | 05 |
| Ω_p, Ω_c, Ω_RF (Ω_L) | Rabi frequencies (probe/coupling/RF; LO) | rad/s | lock #4; always recomputed from (d, ℰ) — never quoted independently (ruling R-22) | 06 |
| Γ_EIT | EIT transparency FWHM | rad/s | ≈ 2γ_gr + Ω_c²/(2γ_ge) sanity form only; hot-cell value from velocity integral | 06 |
| σ (density matrix) | rotating-frame density matrix | — | σ_eg = probe coherence; vec order='F' | 06 |
| L, L₀, L₁ | Liouvillian; linearization pieces | rad/s | spec 06 §2.3 / 08 §2.3; affine in Δ's and v | 06/08 |
| E (field), F (spec 07) | electric field amplitude | V/m (V/cm I/O) | code symbol `E_field` / `E_Vm`; spec 07's F is the same quantity — the letter F in code is reserved for hyperfine | 07 |
| α₀, α₂, α(v,m_j) | scalar/tensor/total polarizability | C²m²/J (a.u., MHz/(V/cm)² I/O) | lock #14; sign anchors: α₀(Rb nS/nD) > 0, α₀(Cs nD₅/₂) < 0, α(Rb nD₅/₂, m_j=1/2) = α₀ − 0.8α₂ can be < 0 (ruling R-19) | 07 |
| γ (hyperpolarizability) | E⁴ Stark coefficient | SI | diagnostic only, no sourced values, never a finding | 07 |
| F_IT, F_ion, F₀ | Inglis–Teller, classical ionization, a.u. field | V/m | F₀/(3n⁵); F₀/(16n⁴); F₀ = 5.1422068e11 V/m | 07 |
| T(f), S_geo, τ_s,eff, κ_ph, β_scr, T_DC | cell screening transfer function + parameters | — / s / (s·mW)⁻¹ | unified single form, ruling R-7; spec 05 owns parameter values/calibration | 05 |
| E_LO, E_sig, u | LO/signal field amplitudes; ratio | V/m; — | superhet envelope E_env ≈ E_LO + E_sig cos(δt+φ); u = E_sig/E_LO | 08 |
| δ (beat) | beat/Fourier angular frequency | rad/s | signal offset from LO; image at −δ inherent | 08 |
| κ, κ_E, H(δ) | transduction slope; transfer function | W/(V/m); W·s/rad | κ = dP/dE at E_LO (computed); H(δ) via resolvent; complex Doppler average | 08 |
| S_P, S_ν, RIN, NEP | power/frequency-noise PSDs; RIN; detector NEP | W²/Hz; Hz²/Hz; 1/Hz; W/√Hz | one-sided everywhere (lock #12) | 08 |
| NEF | noise-equivalent field | (V/m)/√Hz (nV/cm/√Hz I/O) | amplitude convention; NEF_x = √S_P,x/|κ_E(δ)| except SQL (already field-referred) | 08 |
| N_eff, τ (coh), T₂ | effective atom number; coherence time | — ; s | N_eff = n₀·V_int·f_vel·f_state, all four computed and printed; τ = 1/Γ₂ of the signal-accumulating coherence | 08 |
| T_eq, NF, G | noise temperature, figure, antenna gain | K, dB, — | T_eq = Gλ²NEF²/(8πη₀k_B), G = 1.64 reference dipole, amplitude convention | 08 |
| E_1dB, SFDR, DR | compression point, spur-free / linear dynamic range | V/m, dB | spec 08 §2.5; sweep result authoritative over cubic fit | 08 |
| ENBW | equivalent noise bandwidth | Hz | one-sided; boxcar t ↔ 1/(2t) | 08 |
| g_J, μ_B | Landé factor; Bohr magneton | — ; J/T | Zeeman module (new — ruling R-17): Δf = (μ_B B/h)(g_J' m_J' − g_J m_J) | Zeeman |
| kB, h, ħ, c, e, ε₀, a₀, E_h, η₀ | fundamental constants | SI | scipy.constants CODATA 2022 only (lock #15) | 00 |

**Numeric check values** (derived, never hard-coded): 1 a.u. polarizability → α/h =
2.488318e-4 Hz/(V/cm)²; e·a₀/h = 1.2795448 MHz/(V/cm); e·a₀ = 8.4783536e-30 C·m;
λ_p/λ_c(Rb, 780.241/480.0) = 1.62550; 1 torr = 101325/760 Pa exactly.

---

## 3. Symbol-collision register (same letter, different meanings — disambiguation is mandatory in code names)

| Letter | Collides as | Rule |
|---|---|---|
| α | polarizability (07) / absorption coefficient (05,06) / fine-structure constant (02) / Jing's photon fraction (09) / Beterov BBR fit coefficient A..D (04) | code names: `alpha_pol`, `absorption_coeff`, `alpha_fs`, never bare `alpha` |
| Γ vs γ | population rate vs coherence rate | lock #7; never interchange |
| F | hyperfine quantum number vs electric field (spec 07) | field is `E_field` in code; F reserved for hyperfine |
| h | Planck constant vs Numerov step (02) vs hour | step is `h_x`; Planck from scipy only |
| n | principal quantum number vs number density vs refractive index | density is `N` (`number_density_m3`); refractive index `n_refr` |
| N | number density (m⁻³) vs atom number (08) | `N` = density; `N_eff` = atom number |
| S | line strength S_FF' / screening S(f) / PSD S_x / TRK completeness (07) / flux | full names in code |
| μ | reduced mass (02) vs dipole moment (08) vs quantum defect symbol μ_L (04) | `mu_mass`, `mu_rf` (= `dipole_Cm`), defects are δ everywhere (04's μ_L ≡ δ_l) |
| δ | quantum defect (01) / detuning (06) / beat frequency (08) / Beterov lifetime exponent (04) | context + code names `delta_qd`, `delta_det`, `delta_beat`, `beterov_delta` |
| T | temperature / transmission / transfer function / coherence time T₂ | `T_K`, `transmission`, `T_screen`, `T2_s` |
| u | radial u(r) = rR (02) vs envelope ratio E_sig/E_LO (08) vs most-probable speed (06) | context-local only; never crosses an API |
| A | Einstein A / angular factor A / hyperfine constant A / Beterov fit A / area | full names in code |

---

## 4. Module ownership map (authoritative; fixes cross-reference errors — ruling R-13)

| Quantity | Owning spec / module |
|---|---|
| Quantum defects, energies, n*, E_I, D-line + hyperfine data, λ_c values | **01** `rydsim.atom` |
| Radial wavefunctions, radial matrix elements ⟨r^k⟩ | **02** `rydsim.radial` |
| 3j/6j, phases, reduced/effective dipoles, oscillator strengths, Einstein A formula | **03** `rydsim.angular` |
| Lifetimes, BBR, dephasing budget, collision/self-broadening coefficients | **04** `rydsim.lifetimes` |
| Vapor pressure/density, Doppler machinery + velocity grids, transit, propagation, beam geometry, **screening parameters** | **05** `rydsim.vapor` |
| OBE/Lindblad solver, EIT/AT lineshapes, χ, AT extraction, field inversion | **06** `rydsim.obe` (NOT "doc 04" as written in spec 05) |
| Stark maps, polarizabilities, DC/LF sensing, screening **interface form** | **07** `rydsim.stark` |
| Superheterodyne, noise budget, SQL, receiver metrics, demodulation | **08** `rydsim.superhet` |
| Zeeman shifts (g_J, tuning law) | **Zeeman module (new)** `rydsim.zeeman` — ruling R-17 |
| Benchmarks, grading, validation report | **09** `rydsim.validation` |

Known misreferences corrected by this map: spec 05 twice calls the OBE solver "doc 04";
spec 09 §3.3 attributes quantum-defect values to "spec 03" (they live in 01) and §C8
attributes the vapor-pressure fit to "spec 02" (it lives in 05).

---

## 5. Resolved conflicts — the ruling register

Each entry: what conflicts, where, and the binding ruling. Implementations follow the ruling.

**R-1. Doppler AT-scaling direction (λ_c/λ_p vs λ_p/λ_c).**
Specs 03/05/06/09 all derive: probe-scan observed splitting = (λ_c/λ_p)·Ω_RF/2π (compressed);
Sedlacek arXiv v1 prose claims the inverse (×1.625). **Ruling:** the compression direction is
normative (Holloway 2014 Eq. (12), Simons 2016 Eq. (1), verified derivations in 05/06/09).
The factor is never hard-coded; benchmark 09/A11 must reproduce it from the full velocity
average, and the Sedlacek prose is recorded as a documented literature tension only.

**R-2. Velocity quadrature — Gauss–Hermite.**
Spec 04 §4.4 recommends GH (41 nodes) and spec 09 §4.2 mandates GH (≥80 nodes, "cheaper");
specs 05 §2.d and 06 §4.4 forbid GH for EIT with quantified reasons (narrow ~1 m/s features
need ~3e5 GH nodes; <100 nodes silently biases AT peaks). **Ruling:** specs 05/06 win. Any
spectrum containing EIT/AT structure uses the uniform or composite grid with the
halving-convergence criterion (also superseding 09's "±4σ_v, ≥401 points" uniform-grid rule,
which is ~10× too coarse against 05's step rule Δv ≤ min(Γ_e/k_p, Γ_EIT/(k_c−k_p))/4).
GH (20–40 nodes) is permitted solely for EIT-free Doppler absorption. Spec 04's and 09's
execution rules are amended accordingly.

**R-3. Transit-rate prefactor.**
Specs 04/05 (consistent): γ_t = √(2 ln2)·v⊥/w₀ (⇒ γ_t/2π = 39.8 kHz for Rb, 300 K, w₀ = 1 mm).
Spec 06 §2.2 estimator: γ_t ≈ ū_2D/(2w₀) (⇒ 16.9 kHz — factor 2.35 lower). **Ruling:**
γ_t = √(2 ln2)·v⊥/w₀ is normative (per-v⊥-class; thermal shortcut uses ⟨v⊥⟩ = √(πk_BT/2m)).
Spec 06's estimator is demoted to an order-of-magnitude docstring note; its API correctly
takes γ_t as an explicit input, which must be fed from spec 05's formula.

**R-4. Rydberg decay routing.**
Spec 04 §2.4 mandates branch-to-e plus a **sink state**; spec 06 §2.2 defaults to routing all
Γ_r to g (closed system) with 'cascade' as an option. **Ruling:** for weak-probe χ and all
EIT/AT/superhet production paths, `decay_route='to_ground'` (spec 06) is the default — the
routing choice perturbs χ at O((Ω_p/Ω_c)²). The 04 sink-state model is the accurate-mode
option for population-transport questions. The two must agree on weak-probe χ to ≤1e-3
relative (benchmark 06/B-14), which is a self-check, not an assumption.

**R-5. α₀(Rb 50S₁/₂) magnitude.**
Spec 04 §3.5 quotes "~6e2 MHz/(V/cm)²" (UNVERIFIED); spec 07 (O'Sullivan & Stoicheff 1985 /
Yerokhin 2016, VERIFIED) gives **50.5 MHz/(V/cm)²** (2.03e11 a.u.). The 04 value is ~12× too
large (it corresponds to n* ≈ 70, not 50). **Ruling:** spec 07 value. Consequence: spec 04's
stray-field dephasing examples rescale — 10 mV/cm on Rb 50S gives ≈ 2.5 kHz (not ~30 kHz),
and the "typical cell" budget row "Stark inhom ~30 kHz" becomes ~2.5 kHz. The §5.1 default
tables in 04 are corrected at implementation time from spec 07's α values.

**R-6. Rb D2 self-broadening coefficient.**
Spec 05 §7.2 declares the D2 coefficient MISSING; spec 04 §3.5 has it VERIFIED from Weller
2011 Table I (theory 1.03e-7 Hz·cm³, measured 1.10(17)e-7, Kondo 2006; Cs D2 1.16e-7 /
1.15(23)e-7). **Ruling:** adopt spec 04's verified values; spec 05's MISSING tag is stale.
Spec 04 owns collision coefficients (ownership map §4).

**R-7. Screening transfer function form.**
Spec 05: S(f) = S_geo·(i2πfτ)^β/(1 + (i2πfτ)^β) (high-pass, |S(∞)| = S_geo, S(0) = 0, stretch
β, photo-activated τ_eff). Spec 07 Eq. (7.14): T(f) = T_DC + (1 − T_DC)·(i2πfτ)/(1 + i2πfτ)
(DC leak, |T(∞)| = 1, no S_geo, no β, no photo-activation). **Ruling:** unified normative form

    T(f) = S_geo · [ T_DC + (1 − T_DC) · (i2πf·τ_s,eff)^β / (1 + (i2πf·τ_s,eff)^β) ]
    1/τ_s,eff = 1/τ_s,dark + κ_ph·P_c

with defaults T_DC = 0, β = 1. Spec 07's (7.14) is the (S_geo = 1, β = 1) special case; spec
05 owns all parameter values and calibration; every output using T(f) logs
(τ_s,dark, κ_ph, P_c, τ_s,eff, S_geo, β, T_DC, calibration status).

**R-8. SQL prefactor convention (ħ vs h vs √2ħ/2).**
Sci. Adv. 2024 prints h/(μ√(N T₂)); Jing v1 prints (√2ħ/2μ)/√(N_a τ_c); Meyer 2020 uses ħ.
**Ruling:** RydSim default is the ħ-convention (lock #13) with an explicit `convention` flag;
published SQLs are reproduced under their own printed convention in benchmarks (08/B6–B8);
no τ-fudging to absorb a 2π.

**R-9. CODATA vintage.**
Specs 04/05/08 print CODATA-2018 values (ε₀ = 8.8541878128e-12); specs 03/06/07 print CODATA
2022 (8.8541878188e-12). **Ruling:** scipy.constants (CODATA 2022) is the sole source; all
printed values are check values at their stated vintage; provenance records the vintage.
Differences (≤1e-9 relative) are below every physics tolerance.

**R-10. Reduced-mass convention.**
Spec 01 mandates the atom-mass convention R_M = R_∞/(1 + m_e/M_atom); spec 02 uses
μ = (M − m_e)/M (ion-core convention). Difference O((m_e/M)²) ≈ 4e-11 — below all tolerances,
but two definitions is one too many. **Ruling:** single code-level definition
μ_mass = 1/(1 + m_e/M_atom) = R_M/R_∞, used by both the energy and radial modules.

**R-11. Sign in spec 01 Eq. (1.8).**
As printed, E_I(centroid) = E_I(from F) − ΔE_hfs(F) contradicts the worked numbers and the
adopted (verified) E_I values: with ΔE_hfs(F=1, Rb-87) = −4271.68 MHz, the printed formula
would ADD 4271.68 MHz. **Ruling:** the correct relation (which the worked arithmetic and
benchmark AS-04 already use) is
`E_I(centroid) = E_I(from F) + ΔE_hfs(F)` (a level below the centroid has ΔE_hfs < 0 and a
larger measured ionization interval). Spec 01's tabulated E_I values are unaffected (they
were computed correctly); only the printed equation carries the sign error.

**R-12. Unit typo, spec 01 §2.1.**
"c·R_∞ = 3 289 841.960 250 THz" — the value is **GHz** (3.2898e15 Hz). Ruling: GHz.
Any dimensional-analysis test on this line uses GHz.

**R-13. Cross-reference/ownership errors.**
Spec 05 calls the OBE solver "doc 04" (twice); spec 09 attributes quantum defects to
"spec 03" and the vapor-pressure model to "spec 02". **Ruling:** the ownership map (§4) is
authoritative; the misreferences are editorial and carry no physics change.

**R-14. Cs nD₅/₂ quantum defect δ₀.**
Spec 04 Table 3.4: 2.4663091 (Goy recall, Form-A context); spec 01: 2.4663144(6)
(Deiglmayr 2016, **Form B**, VERIFIED). **Ruling:** spec 01 normative (as spec 04 itself
states); all convenience copies of defects in specs 02/04/07/09 are non-normative and must
not be duplicated into code. n_eff for the Beterov fits is computed with spec 01's machinery
(including Form-B iteration for Cs S/D₅/₂); the resulting deviation from Beterov's own n_eff
is ≪ the fits' 5% accuracy.

**R-15. Nominal coupling wavelengths.**
480.0 / 509.4 nm (spec 05), 510 nm (specs 03/09 Cs ratio), ranges 479–484 / 508–512 nm
(specs 04/06). **Ruling:** λ_c is state-dependent and always computed from spec 01 energies
at runtime; the fixed numbers 480.0 nm (Rb) and 509.4 nm (Cs) are permitted only inside
declared benchmark fixtures, and every Doppler-ratio factor is computed from the actual
wavelengths in use (λ_p/λ_c = 1.62550 for the Rb 780.241/480.0 fixture).

**R-16. Correlated laser noise — API gap.**
Spec 04 §2.3.4 requires the mutual-coherence coefficient c; spec 06's `LadderParams` has no
such field (independent projector-sum operators give γ_gr = γ_p + γ_c only). **Ruling:**
extend spec 06's API with `noise_correlation: float = 0.0`; implement the cross term as an
additional pure-dephasing rate 2c·√(γ_p γ_c) on every level above both transitions (r, r'),
so that γ_gr,laser = π(Δν_p + Δν_c + 2c√(Δν_p Δν_c)) exactly as spec 04 specifies. Unit test
= spec 04 S7/B15 extended with c ≠ 0.

**R-17. Zeeman physics has no owner.**
Spec 09 benchmarks C9, E7.1–E7.2 require Zeeman tuning (μ_B, g_J, Δf = (μ_B B/h)(g_J'm_J' −
g_J m_J)); spec 07 explicitly excludes Zeeman ("future spec") and spec 05 §7.7 points to a
misnumbered doc. **Ruling:** create `rydsim.zeeman` (linear Zeeman of fine-structure
Rydberg states, Landé g_J = 3/2 + [s(s+1) − l(l+1)]/(2j(j+1)), stretched-state tuning law
above) as a prerequisite for C9/E7; μ_B/h = 1.399625 MHz/G check value. Quadratic Zeeman
and E×B remain out of scope.

**R-18. Steady-state acceptance tolerances.**
Spec 06: |Tr−1| < 1e-12, Herm < 1e-10, eig > −1e-10; spec 09: |Tr−1| ≤ 1e-10, eig ≥ −1e-10.
**Ruling:** release gates use 1e-10 across all three checks (spec 09); the tighter 1e-12
trace figure is an internal aspiration, not a gate.

**R-19. "Scalar polarizability −8.6 GHz/(V/cm)²" for Rb 100D₅/₂ (Meyer/E5.1) vs
"α₀(Rb nD) > 0" (spec 07).**
Not a sign contradiction, but a decomposition trap: for m_j = 1/2, α(v, m_j) = α₀ − 0.8·α₂,
and Rb nD₅/₂ has α₂ ≈ 1.9·α₀ > 0, so the **m_j = 1/2 total α is negative** while the scalar
α₀ is positive. **Ruling:** benchmark E5.1 targets α(v, m_j = 1/2) (the m_j-resolved value),
computed via spec 07 Eq. (7.1); spec 07's sign anchors (α₀ > 0 for Rb nS/nD, α₀ < 0 for Cs
nD₅/₂) stand and are tested on α₀, never on α(m_j).

**R-20. Older Steck revisions.**
Cs 6P₃/₂ = 30.473 ns and Rb-87 D2 dipole uncertainty (87) circulating in earlier drafts and
the task brief come from pre-2.3.4 datasheets. **Ruling:** rev 2.3.4 values everywhere
(Cs 30.405(77) ns, Γ/2π = 5.234(13) MHz; Rb-87 4.22752(62) e·a₀); superseded values may
appear only in "tension documented" notes.

**R-21. Number-density units across the 04↔05↔06 interface.**
Spec 04 quotes coefficients in Hz·cm³ and densities in cm⁻³; specs 05/06 are SI (m⁻³).
**Ruling:** internal unit is m⁻³ (and Hz·m³ for β, C₆/h in Hz·m⁶); cm-based values convert
at data-load time with the conversion recorded next to the source value. Mixed-unit
arithmetic in physics code is a release blocker.

**R-22. Literature Rabi/field conventions (Jing).**
Spec 09 Eq. (2.5) records Jing's P_s = (√2·d_RF·α·P̄/(ħΓ))·E_s (a √2 that is their field/RMS
convention artifact), and spec 08 flags Jing's Ω_L = 7.9 MHz vs E_LO = 3.0 mV/cm being
mutually inconsistent (μE/ħ = 2π·5.54 MHz). **Ruling:** RydSim never imports literature Rabi
frequencies or prefactors — Ω is always recomputed from (d, ℰ) under lock #3/#4. Benchmarks
C6a/C6b test only the convention-free relations Ω_L* = Γ/√3 and S_max = 3√3χ₀/(8Γ).

**R-23. Transit refill target state.**
Spec 05: relax toward "thermal ground state" ρ_thermal; spec 06: measure-and-replace to
|g⟩⟨g|. **Ruling:** identical in the reduced 3/4-level model (single ground state). In any
multilevel extension, the refill target is the degeneracy-weighted thermal ground manifold
(p_F weights from spec 05 §2.b).

**R-24. EIT-width closed forms.**
Spec 04 quotes FWHM_EIT ≈ 2γ_gr + Ω_c²/Γ_e; spec 06 the more general 2γ_gr + Ω_c²/(2γ_ge)
(equal when γ_ge = Γ_e/2). **Ruling:** spec 06's form is the sanity formula; both are
non-quantitative in Doppler media — hot-cell linewidths come only from the velocity integral.

**R-25. = R-20 (kept for numbering stability; see R-20).**

**R-26. vdW dephasing double-/h.**
Spec 04 writes Δν_vdW ≈ |C₆|/(h·r_nn⁶) while quoting C₆ in GHz·µm⁶ (already an energy/h).
**Ruling:** C₆ is stored as C₆/h [Hz·m⁶]; Δν_vdW = (C₆/h)/r⁶ with no additional /h.

---

## 6. Interface gaps (closed or explicitly assigned)

1. **Zeeman module** — assigned (R-17). Blocks 09/C9, E7 until implemented.
2. **Correlated-laser-noise parameter** in the OBE API — assigned (R-16).
3. **Low-n intermediate levels** (Rb 5D/6P, Cs 7P) — spec 01 declares MISSING; take from
   NIST ASD as data if/when needed; never from the Ritz expansion below n_hard.
4. **Rb D2 self-broadening** — closed by R-6 (spec 04's verified Weller/Kondo values).
5. **Borosilicate τ_s,dark** — remains a per-cell calibration parameter (spec 05); glass-cell
   kHz-band NEF outputs must carry "parameter-dependent" caveats.
6. **Holloway JAP 2017 numeric systematics budget** (09/E9.2+) — paywalled; must be fetched
   before release; never filled from memory.
7. **Effective dipole for hot-vapor χ and Ω_p** — closed: d_eff,far = ⟨J‖er‖J'⟩_Steck/√3
   (spec 03 Eq. 2.16) is used for both, consistently (spec 05's worked numbers already do).
8. **SQL N_eff/f_vel recipe** — owned by spec 08 §2.4.5 (computed from the velocity-resolved
   response weight, never a folklore fraction).

---

## 7. Module dependency graph — implementation order (normative)

Build strictly in this order; each module's tests may use only earlier modules plus declared
fixtures.

1. **`rydsim.constants`** (this doc) — scipy CODATA passthrough, unit helpers, provenance.
   Depends on: nothing.
2. **`rydsim.angular`** (spec 03) — 3j/6j, Wigner–Eckart chain, S_FF', conversion helpers,
   exact-rational test oracle. Depends on: constants only.
3. **`rydsim.atom`** (spec 01) — species data, quantum defects (Form A/B), energies, E_I,
   D-line/hyperfine data, transition frequencies, λ_c. Depends on: constants.
4. **`rydsim.radial`** (spec 02) — Numerov + MSD94, Method B/C cross-checks, consensus
   matrix elements. Depends on: atom (n*), constants.
5. **`rydsim.dipoles`** (specs 02+03 integration layer) — full dipole matrix elements,
   effective RF dipole ℘, D-line closure checks. Depends on: angular, radial, atom.
6. **`rydsim.lifetimes`** (spec 04) — Einstein-A sums, Beterov fits, BBR, dephasing budget,
   collision coefficients. Depends on: dipoles, atom, constants.
7. **`rydsim.vapor`** (spec 05) — vapor pressure/density, Doppler machinery + velocity grids,
   transit rates, beam geometry, propagation, screening T(f). Depends on: atom (masses,
   D-line), angular (S_FF' for the Voigt reference), constants.
8. **`rydsim.zeeman`** (new, R-17) — g_J and linear Zeeman tuning. Depends on: atom, angular.
9. **`rydsim.obe`** (spec 06) — Hamiltonians, collapse sets, Liouvillian, steady state,
   weak-probe χ, Doppler-averaged spectra, AT extraction, field inversion. Depends on:
   dipoles, lifetimes, vapor.
10. **`rydsim.stark`** (spec 07) — perturbative α, Stark maps, readout/bias formulas,
    screening interface. Depends on: atom, radial, angular (and vapor for T(f) parameters).
11. **`rydsim.superhet`** (spec 08) — transduction slope, H(δ)/IBW, noise budget, SQL,
    receiver metrics, demodulation. Depends on: obe, dipoles, lifetimes, vapor (and stark
    for the biased-quadratic LF readout mode).
12. **`rydsim.validation`** (spec 09) — benchmark registry, grading, report. Depends on: all.

Circularity note: the spec 07 ↔ 08 NEF pipeline (07 Eq. 7.15 consumes 08's δν_min) is a
*data-flow* loop at analysis time, not an import cycle — `stark` exposes responsivity,
`superhet` exposes δν_min, and the pipeline is assembled in `validation`/application code.

---

## 8. Enforcement

Convention locks are enforced by existing benchmarks: 06/B-1 (Ω/2 and Hamiltonian), 06/B-2
(χ prefactor via σ₀ = 3λ²/2π), 06/B-7 ('F' ordering), 06/B-9+09/A11 (Doppler factor,
adjudicates R-1), 06/B-14 (R-4), 04/S7 + R-16 test (laser-noise factors), 05/B6/B15
(λ ratio and drag sign), 08/B4 (shot-noise unit identity), 08/B6–B8 (SQL conventions),
07/RS-07-01/02 (unit chain), 09 report rules (one-sided NEF grading). New tests required by
this document: (i) transit prefactor lock γ_t·w₀/v⊥ = √(2 ln 2) (R-3); (ii) unified T(f)
reduces to both parents' forms (R-7); (iii) correlated-noise wiring (R-16); (iv) Zeeman
tuning-law check against μ_B/h (R-17); (v) α₀(Rb 50S) = 50.5 MHz/(V/cm)² replaces the 04
budget figure (R-5).

---

*GreyNOC · RydSim spec 00 · consistency review 2026-08-10 · house rule: reproducible or it
didn't happen.*
