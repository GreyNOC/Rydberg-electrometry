# 00 — Scientific-Integrity Audit of Specs 01–09

**Role:** adversarial integrity audit. **Date:** 2026-08-10. **Network:** AVAILABLE during audit.
**House rule applied:** *no fabrication — reproducible or it didn't happen.*
**Scope:** every numeric constant table in `docs/spec/01…09` — quantum defects, model-potential
parameters, vapor-pressure coefficients, dipole moments, lifetimes, polarizabilities, sensitivities —
assessed for (a) sourcing, (b) plausibility, (c) correctness of the confidence tag; plus the refusal
rules and provenance metadata the implementation must enforce.

---

## 1. Audit method and what was independently re-verified this session

The audit did not take the specs' own "VERIFIED" tags at face value. Independent spot-checks were run
against primary and secondary sources **this session** (all fetches 2026-08-10):

| # | Check | Result |
|---|---|---|
| 1 | Steck datasheet revision "2.3.4, 8 Aug 2025" (specs 01, 03–06, 09) | **CONFIRMED** from steck.us index *and* from inside the Cs PDF itself |
| 2 | Steck Cs values: τ(6P₃/₂) = 30.405(77) ns, Γ = 32.889(84)×10⁶ s⁻¹, Γ/2π = 5.234(13) MHz, ⟨J‖er‖J′⟩ = 4.4837(57) e·a₀, τ(6P₁/₂) = 34.791(90) ns, ν(D2) = 351.725 718 50(11) THz, I_sat 1.1049(28) / 2.7119(69) mW/cm² | **CONFIRMED** — extracted directly from the primary PDF (saved copy, pypdf) |
| 3 | Steck/Alcock Cs vapor coefficients: solid 2.881 + 4.711 − 3999/T; liquid 2.881 + 4.165 − 3830/T (spec 05) | **CONFIRMED** from the Cs PDF |
| 4 | Beterov 2009 lifetime fits (all 10 τ_s/δ rows) and BBR A,B,C,D rows (Rb/Cs nS), Eq. 16 "<5 % for 15<n<80" claim (specs 04, 09) | **CONFIRMED verbatim** from ar5iv arXiv:0810.0339 |
| 5 | O'Sullivan & Stoicheff Rb nS fit 2.202(28)×10⁻⁹ n*⁶ + 5.53(13)×10⁻¹¹ n*⁷ MHz/(V/cm)² (spec 07 Eq. 7.6) | **CONFIRMED** via ADS record of PRA 31, 2718 |
| 6 | MSD94 model-potential tables, Rb + Cs, all 40 entries (spec 02 §3.1–3.2) | **CONFIRMED digit-for-digit** vs ARC `alkali_atom_data.py` (fetched); ARC a₄(l=1) = −0.8163314 confirmed |
| 7 | The flagged a₄(l=1) transcription discrepancy (spec 02 §3.3) | **CONFIRMED REAL**: ryd-numerov `elements/rubidium.py` (fetched via gh api) carries −0.81633314; ARC carries −0.8163314. The spec's disclosure is accurate, not decorative |
| 8 | Rb quantum defects (Li 2003 digits incl. G series) as transcribed in ARC (spec 01 §3.4) | **CONFIRMED** — ARC carries exactly the spec's digits (secondary source; primaries still unfetched, see register) |
| 9 | Deiglmayr Cs E_I = 31 406.467 732 5(14) cm⁻¹ (spec 01 §3.3) | **CONFIRMED** from arXiv:1601.08005 abstract |
| 10 | Bai 2023 Cs nF defects: δ₀(F₅/₂) = 0.033 415 37(70), δ₂ = −0.2014(16); δ₀(F₇/₂) = 0.033 564 6(13), δ₂ = −0.2052(29), n = 45–50 (spec 01 §3.5) | **CONFIRMED verbatim** from arXiv:2304.07974 abstract |
| 11 | Mack 2011 Rb-87 E_I(from 5S₁/₂ F=1)/h = 1 010.029 164 6(3) THz (spec 01 §3.3) | **CONFIRMED** from arXiv:1103.6221 abstract |
| 12 | Jing 2020 sensitivity 55 nV cm⁻¹ Hz⁻¹ᐟ² (specs 08, 09) | **CONFIRMED** from arXiv:1902.11063 abstract |
| 13 | Yerokhin polarizability rows (Rb 30S/35S/45S/50S, Rb 35D₅/₂, Cs 50S, Cs 39D₅/₂ incl. the negative α₀) **and** the "a₀⁵" tensor-table label anomaly (spec 07 §3.2) | **CONFIRMED** from ar5iv arXiv:1608.04515 — including the label anomaly the spec warns about |
| 14 | Kaulakys formula chain, Eqs. (19), (21)–(24), (30)–(31) (spec 02 §2.5) | **CONFIRMED** against the full paper text held in-repo (`kaulakys_text.txt`) |
| 15 | Internal arithmetic spot-recomputations: c·R_M(Rb-87), δ_pol(l=4) both species, Cs F=4 hyperfine shift +4.021 776 4 GHz, ground-splitting identities, e·a₀/h, a.u.→MHz/(V/cm)² factor, F_IT(30), α₀(50S) from the O'S&S fit, Beterov 50S lifetime, T_eq for both cited receivers, shot-noise 7.479 pW/√Hz, Fermi-shift coefficient −9.9×10⁻⁸ Hz·cm³ from a_s = −16.1 a₀, transit prefactor 0.3748, beam numbers I₀/Ê₀/Ω_p | **ALL REPRODUCE** |

**Verdict on the fabrication question:** no hallucinated table was found. Every high-risk table that
could be checked, checked out — including two deliberately awkward disclosures (the a₄ two-source
discrepancy and the Yerokhin unit-label anomaly) that a confabulated document would not contain.
The confidence-tag discipline is genuine. The remaining risk is concentrated in (i) values tagged
honestly as recall/estimates that nevertheless leak into derived numbers, (ii) session-measured
numerics whose harness scripts are not in the repo, and (iii) two cross-spec contradictions found by
this audit. Those are the register.

---

## 2. RISK REGISTER

Risk levels: **HIGH** = wrong value or rule would corrupt shipped numbers and is not currently
fenced by an automatic check; **MED** = wrong value would be caught by an existing benchmark or is
bounded/declared, but needs the named self-check wired in; **LOW** = verified this session against a
primary source, or pure algebra/self-checking.

| # | Constant / table | Spec file | Risk | Why | Concrete self-check the implementation MUST run |
|---|---|---|---|---|---|
| R1 | α₀(Rb 50S) "~6×10² MHz/(V/cm)²" placeholder used in the dephasing budget | 04 §2.3.6, §3.5, §5.1 | **HIGH** | The verified value (spec 07, Yerokhin + O'S&S, re-confirmed this session) is **50.5 MHz/(V/cm)²** — the spec-04 placeholder is **~12× too large**, and the derived budget numbers inherit it: "10 mV/cm → ~30 kHz" is actually ~2.5 kHz; the §5.1 defaults row "Stark inhom. ~30 kHz" is wrong by the same factor. The UNVERIFIED tag exists but the derived numbers are printed untagged. | `build_budget()` must call the spec-07 engine for α — the 6×10² literal must never appear in code. Regression: assert budget-Stark(Rb 50S, 10 mV/cm) = ½·α₀·E²/h with α₀ from `rydsim.stark`, and cross-check α₀(50S)/h = 50.5 MHz/(V/cm)² ± 3 % (RS-07-05). Spec 04 §5.1 table needs an erratum note. |
| R2 | Corpus velocity-quadrature rule: "Gauss–Hermite ≥ 80 nodes" | 09 §4.2 vs 05 §2.d, 06 §4.4 | **HIGH** | Direct cross-spec contradiction: specs 05/06 **forbid** GH as the primary scheme for EIT (narrowest velocity feature ~1 m/s vs σ_v ~170 m/s needs ~10⁵-node-equivalent resolution; GH with <100 nodes silently biases AT peaks). Spec 09's own "doubling nodes" criterion can falsely converge on an under-resolved dip — corrupting exactly the benchmark (A11, λ-ratio adjudication) the corpus exists to settle. | Every corpus velocity average runs BOTH the spec-05 uniform/composite grid and the GH grid; require agreement < 10⁻⁴ in T(Δp) or FAIL the run (not the benchmark). A11 specifically must be executed on the uniform grid with step ≤ δv/10 per spec 06 §4.4. Spec 09 §4.2 must be amended to defer to spec 05 §2.d. |
| R3 | Cs E_I erratum row: "PRA 112, 049902(E) (2025)", +0.47 MHz, and "Shen et al., 941 542 216.431(4) MHz" | 01 §3.3, §7.3 | **HIGH** (as citation), LOW (numerically) | Unverifiable this session; the Shen citation is admitted to be known "only via the erratum summary — UNVERIFIED". High-specificity digits with no retrievable source is precisely the hallucination signature this audit screens for. The spec correctly does NOT apply the correction, which fences the number — but only as long as no one "helpfully" encodes it. | The erratum value must not exist in code or data files; only the 2016 value ships. Provenance note carries the erratum flag as text. Release gate: before any claim of Cs absolute accuracy < 0.5 MHz, fetch PRA 112, 049902(E) and the Shen paper or strike the row. AS-08/08b/08c (UV-line closure < 150 kHz) remain the operative guard. |
| R4 | Spec-02 session-measured numerics: `hyperu` precision-collapse table, grid-convergence numbers (2n(n+15), h = 0.001), B8–B16 "measured" spreads | 02 §2.4, §4.1, §6 | **HIGH** | All rest on scratchpad scripts (`verify_radial_02*.py`) that are **not in the repo**. Until ported, every "measured 4×10⁻¹²/2.0×10⁻⁶/…" is an unreproducible assertion. Worse: `hyperu` accuracy is scipy-version-dependent — the ν ≤ 25 cutoff could be wrong (either direction) on the installed scipy. | Port the harness as `tests/test_spec02_benchmarks.py` before any release. The hyperu-vs-analytic-hydrogen error table must be **regenerated at test time** on the installed scipy: assert error(ν=20) < 10⁻⁶ and assert the collapse (error(ν=35) > 10⁻²) actually occurs where claimed; `whittaker_u` raises for ν > 25 regardless. B7 (h⁴ order) and B14 (outer-cutoff) re-measure the grid claims. |
| R5 | μ_RF(Cs 47D₅/₂→48P₃/₂) = 1443.450 e·a₀ (radial 2946.512 e·a₀) — the Jing benchmark fixture | 08 §3.2 | **HIGH** | LITERATURE-RECALL from an unnamed secondary survey; it anchors B17/B19/B20 — the sensitivity ("money") gates. Internal consistency (2946.512 × 0.4899 ≈ 1443.5) shows only that the angular factor was applied, not that the radial number is right. | Never use the literal: compute μ_RF from spec-02 consensus radial ME × spec-03 angular factor at fixture-build time; assert agreement with 1443.45 e·a₀ to 2 % (else the *fixture*, not the code, is flagged). Same rule for Sedlacek's 1.37×10⁻²⁶ C·m (C5e) and Tu's 1218 e·a₀ (C5f). |
| R6 | VERIFIED-ARC-only quantum-defect rows: Rb nP₁/₂, nP₃/₂, nF₅/₂, nF₇/₂, nG; Cs nD₃/₂ (Lorenzen–Niemax), Cs nG₇/₂ (Weber–Sansonetti); α_d(Rb⁺) = 9.076, α_d(Cs⁺) = 15.644 | 01 §3.4–3.5, 02 §3 | MED | Secondary transcription (ARC data file) — audit re-confirmed ARC carries exactly these digits, but the primary PRA papers remain unfetched (paywalled). A transcription error in ARC would propagate. Spec's own §7.9 already mandates primary retrieval before sub-MHz absolute claims. | Keep the C4a–C4f microwave-interval benchmarks (±5–100 MHz) and AS-09 (polarization-formula vs measured G defect, ≤ 5 %) as the external tripwires — a 10⁻⁴ δ₀ error fails C4 by ≫10 MHz. Release blocker for "sub-MHz absolute" claims until Li 2003 / Han 2006 / L-N 1984 / W-S 1987 primaries are fetched. |
| R7 | Rydberg HFS coefficients: A_S(Rb-85 nS) ≈ 4.96 GHz (±30 %), A_S(Cs nS) ≈ 17.1 GHz (±40 %, ground-scaling estimate that overpredicts the measured Rb-87 case by 33 %) | 01 §2.4, §3.8 | MED | Honest estimates driving a normative modeling cutoff (HFS included only for nS n < 30). If the Cs coefficient is badly wrong, the cutoff error bound (<0.5 MHz) is wrong too. | Code must carry the ±30–40 % uncertainty on A(n) outputs and propagate it into the declared HFS-neglect bound. Fetch Sassmannshausen/Merkt/Deiglmayr PRA 87, 032519 (2013) before any hyperfine-resolved Rydberg claim. AS-15 tests only the Rb-87 coefficient (the verified one) — do not extend it to Cs without the paper. |
| R8 | Rb-85 ionization limit 1 010 024 700(7) MHz (Sanguinetti) vs Steck/Lee-1978 (41 MHz ≈ 6σ tension) | 01 §3.3, §7.2 | MED | The Sanguinetti digits are tagged VERIFIED but sit in the paper body (abstract shows no numbers — not re-confirmable this session), and the documented 6σ tension means the Rb-85 **absolute** optical scale carries a ±40 MHz systematic either way. Intervals are immune (E_I cancels). | Every Rb-85 absolute-frequency output must carry the declared ±40 MHz systematic in its uncertainty field (not a footnote). Assert in tests that Rb-85 interval predictions (C4c, C4d) are computed via Eq. (1.9) so E_I genuinely cancels. |
| R9 | Beterov Table VII Rb nD₃/₂ anomaly claim (n = 50–60 rows deviate 5–6 % from their own fit) | 04 §6 | MED | Specific claim about the source table, not re-verified this session. If the anomaly is actually a spec-side transcription slip, the widened 10 % D₃/₂ tolerance hides a real 5 % engine error. | When porting benchmarks, re-extract Table VII digits from arXiv:0810.0339 (page-level) and record them in the test file with a comment. Keep D-state validation on the nD₅/₂ column as specified; the D₃/₂ 10 % band must be documented as anomaly-driven, not physics-driven. |
| R10 | Quantum-defect convenience copies drift: spec 04 §3.4 lists Cs nD₅/₂ δ₀ = 2.4663091 (Goy) while spec 01 normative is 2.466 314 4 (Deiglmayr); similar per-source drift for other Cs rows | 04 §3.4 vs 01 §3.5 | MED | Duplicated constants are how normative values fork. The 5×10⁻⁶ drift is harmless today; the *pattern* is the hazard. | Single source of truth: `rydsim.atom` dataclasses populated only from spec 01; any other module importing defect literals is a lint failure (grep-based CI check for the digit strings). Spec 04's table stays documentation-only. |
| R11 | Collisional/interaction coefficients: C6 fit (Singer 2005: 11.97, −0.8486, 3.385×10⁻³), a_s(e⁻–Rb) = −16.1 a₀, Rydberg–ground broadening = |shift|·0.5 (×2 uncertainty) | 04 §2.3.5, §3.5 | MED | All LITERATURE-RECALL / order-of-magnitude; they gate warnings (vdW > 0.1 MHz) and hot-cell budget terms, not headline numbers. The Fermi-shift coefficient −9.9×10⁻⁸ Hz·cm³ was re-derived from a_s this audit and is internally consistent. | vdW/collision outputs are flagged `order-of-magnitude`; simulator must emit the spec-04 warning when they exceed 0.1 MHz. Self-check vs measured density-dependent hot-cell widths before any hot-cell (>100 °C) sensitivity claim. |
| R12 | Screening parameters: borosilicate τ_s (MISSING/recall), κ_ph = 1.7 s⁻¹/mW (condition-dependent lower bound), Jau & Carter cutoffs 64→770 Hz | 05 §2.h, 07 §2.9 | MED | Phenomenological model with per-cell parameters; the specs say so plainly, and the Jau & Carter anchor numbers could not be re-fetched this session. Any sub-kHz NEF prediction quoted without calibration status is meaningless — the spec's own words. | `screening_factor()` refuses default-τ_s use for borosilicate absolute claims (raises unless `calibrated=True` or explicitly `estimate_ok=True`); every S(f)-dependent output carries (τ_s,dark, κ_ph, P_c, τ_s,eff, S_geo, β, calibration status). B16 pins the arithmetic only. |
| R13 | Rb D2 self-broadening for hot-cell absolute absorption (Weller-grade); Rb D2 β verified only at Weller-Table level | 05 §7.2, 04 §3.5 | MED | Spec 05 declares the Durham-grade D2 coefficient MISSING for the ≥10¹² cm⁻³ regime. Supporting hot cells without it silently degrades B9-class OD spectra. | Hard warning above n = 10¹² cm⁻³; hot-cell support gated on adding the D2 coefficient with a primary source. S5 (β formula reproduces 1.03×10⁻⁷) remains the formula check. |
| R14 | Technical-noise defaults: RIN 10⁻¹⁴ /Hz, corner 100 kHz, NEP 5 pW/√Hz, η = 0.85, 100 kHz linewidths; lock-in/Hann ENBW rows | 08 §3.5, §2.7 | MED | All UNVERIFIED-FROM-MEMORY and correctly labeled config inputs — the risk is a user reading B19's ×3 window as physics. ENBW table rows (SRS/Harris/Keysight conventions) are recall. | NoiseInputs defaults print an UNVERIFIED banner into every report; B19/B20 remain order-of-magnitude gates whose failure triggers review, never tolerance-widening. The I/Q-vs-SA 0.5 dB self-agreement test (§2.7) is the internal check on the ENBW bookkeeping. |
| R15 | Jing fixture open items: Ω_L = 7.9 MHz vs E_LO = 3.0 mV/cm inconsistency; v1-only beam/cell parameters; 90 dB DR; Tu noise recombination 7.95 ≠ 10.0 | 08 §3.2–3.3, §7.9 | MED | Honestly flagged, unresolved. Benchmarks depending on them (B17, B19, E3.x) must inherit the wide windows already specified — the risk is future tightening without resolving the source. | RydSim always recomputes Ω from (μ, E) — never ingests a published Ω_L. Tolerance changes to B17/B19/E3.x require a spec-09 edit citing the published PDF (per spec 09 §7 rule 5). |
| R16 | 2026-vintage corpus targets: E6 (arXiv:2506.10541 — IBW 54.6 MHz @ 140.4 nV cm⁻¹ Hz⁻¹ᐟ², record 76.8 MHz/222.6), E7 (Comms. Phys.), E8 (npj QM 13.5 nV cm⁻¹ Hz⁻¹ᐟ² @ 100 kHz); also PRResearch 6, 023138 citation | 09 §3.5, 07 §2.5/2.9 | MED | Post-knowledge-cutoff papers; claimed fetched during authoring but not re-verifiable by this audit. Graded ×1.5/ORDER/QUALITATIVE, which bounds the damage. | Keep grades frozen; a failing E6–E8 benchmark is investigated against a re-fetched paper, never resolved by editing the expected value from memory (spec 09 §7 rule 5 + E9.2 rule generalized). |
| R17 | Transit-broadening prefactor: specs 04/05 use γ_t = 1.177·v⊥/w (matched to FWHM 0.3748·v/w); spec 06 estimator uses ū/(2w₀) = 0.5·v/w | 06 §2.2 vs 04 §2.3.3, 05 §2.e | MED | Factor-2.4 spread between sibling specs for the same physical rate. Spec 06 tags its version UNVERIFIED and takes γ_t as an input — but two "default" conventions in one codebase is how silent 2× linewidth errors happen. | One transit module (`rydsim.vapor.transit_rate_rad_s`, spec 05 convention) is the only source; spec 06's estimator delegates to it. Unit test asserts the 04/05 numbers (106 kHz @ 0.75 mm, 79.6 kHz @ 1 mm, Rb-87 300 K) from the single implementation. |
| R18 | λ-ratio direction: Sedlacek v1 prose (×1.625) vs Holloway Eq. 12 / NIST convention (×0.615 compression, recover with ×1.625) | 09 §2.2/A11, 03 §2.5, 05 §2.d, 06 §2.6 | MED | The four specs are mutually consistent (measured probe-scan splitting is *compressed*; multiply by λp/λc to recover Ω) and match the audited Simons-2016 convention — but the documented literature tension means a (λp/λc)² ≈ 2.64 systematic if ever wired backwards. | A11 (numerical adjudication from the velocity-averaged OBE, 5 %) is mandatory and release-gating; `invert_field` exposes `scan=` and applies the factor exactly once (B15/B6-spec05 sign tests). Until A11 passes, no field inversion ships. |
| R19 | Steck rows for Rb-85/Rb-87 (spec 03 §3.2, 01 §3.6) | 01, 03, 04, 05 | LOW | Cs datasheet re-verified from the primary PDF this session (every row checked matched); Rb sheets not independently re-parsed by this audit, but the isotope-shift closure (AS-10: 78.0955 MHz internal consistency) and the D1/D2 √2-ratio check (B25) hold on the printed digits, and the revision string is real. | AS-07/AS-07b/AS-10, B18–B22 lifetime↔dipole round trips (≤ 2×10⁻⁵ closure) — already specified; they would expose any transcription slip at the printed-digit level. |
| R20 | MSD94 potential tables + a₄(l=1) choice −0.81633314 | 02 §3.1–3.3 | LOW | Both transcriptions re-fetched and diffed this audit; the single discrepant digit is real, disclosed, and bounded (< 4×10⁻⁸ effect on Z₁, far below all tolerances). | Keep the §3.3 note; B8 (A-vs-B spread ≤ 10⁻⁴) catches any *material* potential error because Method B is potential-independent. Anyone with PRA 49, 982 Table I closes the item. |
| R21 | Beterov fit tables (τ_s, δ; A–D) | 04 §3.1–3.2, 09 §3.4 | LOW | Re-verified verbatim this session from arXiv:0810.0339. Residual: v4-includes-erratum is presumed (APS page 403'd) — declared in-spec. | S2/S3 (fit-vs-sum 10–20 %) plus B2–B9 pin both the tables and the engine; erratum presumption noted in provenance. |
| R22 | Vapor-pressure model (both species, both phases) + worked densities | 05 §2.a, §3.1 | LOW | Cs coefficients re-verified from the primary PDF; Rb coefficients match the same Alcock family and Steck's quoted 3.92(20)×10⁻⁷ torr closure (B1). Worked densities re-derivable. | B1–B3 numeric reproduction (±0.5–1 %); the ±5 % model band and the 11 %/K T-sensitivity warning must appear in every density-dependent output. |
| R23 | Polarizability sources: Yerokhin tables, O'S&S fits, unit-conversion chain, H exact values | 07 §2–3 | LOW | All re-verified this session (incl. the a₀⁵ label anomaly and the exact 9/2 a.u. H result); unit chain re-derived. Note the Rb nD closed-form fit is properly declared MISSING (paywalled) — the spec resisted inventing one. | RS-07-01…18 as specified; §4.3 rule (no hard-coded conversion factors in library code) enforced by grep-lint. |
| R24 | Angular-algebra tables: 3j/6j benchmark values, S_FF′ fractions, NIST A-factors, fine-structure ratios | 03 §2, §6; 05 §3.4 | LOW | Pure algebra with dual-implementation (float vs exact-rational) verification; NIST factors are exact radicals (√2/3, √6/5, 2√3/7) matching the published 4-decimal values. The generic large-j float-accuracy *limitation* (8.2×10⁻⁵ at j≈55–70) is itself a disclosed measured result. | Ship the exact-rational oracle in tests (§4.3.6); enforce the rank-1-only certification rule for j > 30. |
| R25 | Kaulakys semiclassical formulas | 02 §2.5 | LOW | Verified against the full paper text in-repo this audit (eqs. 19, 21–24, 30–31 match, including the (1−e)·sin(πs)/(πs) term and the (3/2)ν²e limit). | B13 (Kaulakys vs Gordon ≤ 2×10⁻⁴) plus the s→0 branch test. |
| R26 | Corpus cross-reference pointers: §3.3 says defects live in "docs/spec/03" (they live in 01); C8 attributes the vapor model to "spec 02" (it is 05) | 09 §3.3, §6 C8 | LOW | Pure documentation defects, but they route implementers to the wrong normative tables. | Fix the two pointers in spec 09; CI doc-link check. |

---

## 3. Where the code MUST refuse to produce a number (never guess)

Consolidated from the specs plus audit findings. Each entry is a *raise/refuse*, not a warning,
unless stated. The unifying rule: **when the normative data source is absent, the answer is an
exception carrying the reason — never a silently degraded number.**

**Energies / structure (spec 01)**
1. Quantum-defect energies below `n_min_hard` (Rb: 8, Cs: 12) — `ValueError`. Below `n_min_mhz`
   (19/25) — mandatory warning attached to the result.
2. Low-lying intermediate levels not in the data tables (Rb 5D, 6P; Cs 7P, …) — MISSING: refuse to
   synthesize from Eq. (1.1); they must be added as NIST-ASD data rows first.
3. Rydberg hyperfine A for series with no sourced coefficient (nP₃/₂, nD, nF at any n): return
   exactly 0 **with the documented bound attached**; never a scaled guess. Cs nS A-values must carry
   the ±40 % tag; refuse "hyperfine-resolved" output modes for those series.
4. The Cs erratum E_I value (941 542 216.33 MHz) must not be reachable through any API (R3).
5. Any Rb-85 absolute optical frequency without the ±40 MHz E_I systematic in its uncertainty (R8).

**Radial machinery (spec 02)**
6. `whittaker_u` for ν > 25 — `ValueError` (hyperu collapse; re-measure per R4).
7. Gordon formula at n = n′ — general formula must raise; only the closed form (2.11) is valid.
8. Radial matrix elements from a single method — the only public entry point is
   `radial_matrix_element_consensus`; single-method calls stay private. No consensus (spread above
   the per-regime ceiling) ⇒ raise, don't average.
9. `radial_wavefunction` with ν ≤ l — raise (no bound QDT orbital).
10. Non-uniform grid into `numerov_inward` — raise.
11. Observables weighted at r ≲ r_cut (contact/hyperfine-type integrals) — out of scope, refuse.

**Angular algebra (spec 03)**
12. Non-(half-)integer j/m inputs — raise (no rounding).
13. Generic (non-rank-1) Wigner symbols at j > 30 through the float path — route to the
    exact-rational oracle or refuse; the float path is certified only for the rank-1 chain.

**Lifetimes / dephasing (spec 04)**
14. Beterov fit paths outside 15 ≤ n ≤ 80 — `ValueError`; the sum path is the only alternative.
15. Converting inhomogeneous budget terms (Doppler, vdW nn-distribution, DC-Stark P(E), beam
    profile) into Lindblad rates in accurate mode — forbidden by construction (§2.4 table is
    normative); in particular adding Δk·σ_v to γ_gr is a named, forbidden error.
16. Buffer-gas cells — transit and collision models invalid: refuse, don't extrapolate.

**Vapor / propagation (spec 05)**
17. Vapor density outside 298–550 K validity — warn + flag extrapolation in the result; never bare.
18. Optically thick (OD > 0.1) or strong-probe (I > 0.01 I_sat) conditions through the analytic
    thin-cell path — `ThickCellError`; only the z-propagation path may answer.
19. Gauss–Hermite as the sole velocity quadrature for any EIT/AT spectrum — raise (and R2: corpus
    runs must cross-validate quadratures).
20. Screening-dependent low-frequency field claims with uncalibrated τ_s (borosilicate default) —
    refuse absolute output; estimates only behind an explicit flag, always with the §2.h parameter
    tuple attached (R12).

**OBE / EIT / inversion (spec 06)**
21. Weak-probe analytic χ outside Ω_p < 0.01·min(Γ_e, Ω_c) — refuse the fast path.
22. Singular/disconnected Liouvillian (cond > 1/eps) — `SteadyStateError` with a physics message;
    `steady_state` never returns an unchecked σ (trace/Hermiticity/positivity gates always on).
23. `invert_field` when the AT doublet is unresolved (`resolved == False`) — raise; the full-model
    fit is the only sub-threshold route. Systematics report entries may read `not_evaluated` but may
    never be silently omitted.

**Stark / polarizability (spec 07)**
24. Perturbative α for n < 10 — `EngineValidityError` (continuum omission; H-1s misses ~19 %).
25. α through a quasi-degeneracy — `DegeneracyError`; high-l (l ≥ 4) targets must use the manifold
    diagonalization, never Eq. (7.1).
26. Dynamic α within the resonance guard band (|ω − ω_k| < 10·max(Γ, Ω)) — `ResonanceError`.
27. Truncated dynamic-α sums in the crossover without the TRK completeness bound (S < 0.98 ⇒ widen
    or switch to ponderomotive; never present the bare truncated sum).
28. Rb nD closed-form polarizability fit — MISSING by declaration: anchors + sum-over-states only;
    no invented fit may be added without a primary source.
29. Hyperpolarizability γ — diagnostic field only; must never appear in a finding.
30. Fields above F_ion (or beyond the Inglis–Teller validity for perturbative results) — refuse or
    hand off to the diagonalization with the validity cap attached.

**Superhet / sensitivity (spec 08)**
31. SQL numbers without the full stamp (N_eff, τ, convention ∈ {ħ, h}, mode, rep rate) — refuse to
    format the output.
32. Published Ω_L values as inputs — never ingested; Ω is always recomputed from (μ, E) (R15).
33. NEF totals when NEF_at and NEF_shot are within 3× — must carry the correlation-caveat flag
    (r = −0.78 measured in the literature; uncorrelated sum strained).
34. E_LO* — computed by optimization, never the hard-coded 3.0 mV/cm.

**Corpus / reporting (spec 09)**
35. E9.2+ (NIST JAP 2017 numeric systematic budget) — placeholder must stay empty until the paper
    is fetched; "must not be filled from memory" is a hard rule this audit re-affirms.
36. UNVERIFIED-confidence benchmarks can never gate a release; VERIFIED benchmark failures are red.
37. A simulated NEF below the SQL for the stated atom number — automatic FAIL (unphysical), and
    >10× better than published triggers mandatory review; both are report-level rules.
38. Benchmark tolerance/expected-value changes require a spec edit with rationale — never a
    code-side constant change.

---

## 4. Required provenance metadata (every simulator output)

Every public result object / report / finding artifact must carry, machine-readable:

1. **Code identity:** RydSim version, git commit hash, dirty-tree flag.
2. **Constants provenance:** scipy version + CODATA vintage actually loaded (asserted at import
   against the spec-01/03 check values); a dump handle to the full constant table with per-row
   `source` and `confidence` tags (`rydsim.provenance` per spec 01 §5).
3. **Configuration hash:** frozen-config hash of every physical input (spec 09 §4.1); no RNG
   anywhere in corpus paths.
4. **Data lineage / taint:** the **minimum confidence class encountered** on the computation path
   (VERIFIED > VERIFIED-ARC > LITERATURE-RECALL > UNVERIFIED > MISSING-blocked). Any output whose
   floor is below VERIFIED prints the offending rows. This makes "confident number, unsourced
   input" structurally impossible to ship unnoticed.
5. **Method + cross-method spread:** for radial MEs the per-method dict and `spread_rel` (spec 02);
   for α the perturbative-vs-map agreement; for steady states the solve-vs-expm check status. The
   spread ships as the numerical uncertainty, per house rule.
6. **Convergence records:** velocity-grid scheme + halving result, n-window/l_max widening result,
   Δn truncation result, z-step halving result — as data (`converged: bool` + magnitudes), not
   docstrings.
7. **Convention stamps:** angular-frequency vs Hz fields (naming rule), dipole convention
   (Steck vs Racah) on every dipole, detuning sign convention, Doppler-ratio factor applied
   (1.0 or λp/λc) + scan axis, SQL convention/mode/N_eff/τ, PSD sidedness, ENBW of the filter.
8. **Validity flags:** every validity gate crossed in warn-mode (vapor-T extrapolation, n_min_mhz,
   vdW > 0.1 MHz, screening-uncalibrated, BBR T_bbr ≠ T_cell) — enumerated in the output, empty
   list if none.
9. **Cell/screening tuple** whenever S(f) touched the result: (τ_s,dark, κ_ph, P_c, τ_s,eff,
   S_geo, β, calibration status) — spec 05 §2.h verbatim.
10. **Uncertainty decomposition:** statistical (propagated σ from radial spreads / fit covariances)
    separated from declared systematics (Rb-85 E_I ±40 MHz; Cs erratum +0.47 MHz note; model-bias
    items like the +8 % low-n dipole bias).
11. **Caveat block:** findings-grade outputs append the spec 09 §8 cannot-claim list verbatim, and
    the report is sorted worst-first with LITERATURE-RECALL/UNVERIFIED sources surfaced
    (spec 09 §5 `report_markdown` contract).
12. **Timestamp + network status** of the run, mirroring the spec-authoring convention.

---

## 5. Summary judgment

The spec suite is **not** a hallucinated-table document: 15/15 independent spot-checks against
primary or independent secondary sources passed, including the adversarial ones (disclosed
discrepancies that fabricated documents don't contain). The tag discipline (VERIFIED /
VERIFIED-ARC / LITERATURE-RECALL / UNVERIFIED / MISSING) is real and mostly correctly applied.

The audit found **two substantive cross-spec defects** that need spec edits — the ×12 stale
polarizability placeholder contaminating spec 04's printed dephasing budget (R1), and the corpus
velocity-quadrature rule that contradicts the EIT specs and could bias the corpus's own
adjudication benchmark (R2) — plus one **fenced but unverifiable citation** carrying
high-specificity digits (R3), one **reproducibility gap** (spec 02's session-measured numerics,
R4), and one **recall-sourced fixture anchor** under the money benchmarks (R5). All five have
concrete self-checks specified above; none require guessing to fix.

*GreyNOC · RydSim integrity audit 00 · 2026-08-10 · reproducible or it didn't happen.*
