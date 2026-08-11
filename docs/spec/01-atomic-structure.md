# Spec 01 — Atomic Structure and Energy Levels (Rb-85, Rb-87, Cs-133)

**Seat:** atomic-structure specialist. **Module:** `rydsim.atom`.
**Depended on by:** spec 02 (radial wavefunctions — needs n*, E), spec 03 (angular algebra), spec 04 (lifetimes), spec 06 (EIT — needs probe-line data), spec 07 (Stark — needs level energies), spec 09 (validation corpus — C4/C5 benchmarks).
**Network status:** WebSearch/WebFetch were AVAILABLE during authoring (2026-08-10). The primary PDFs of Mack et al. PRA 83, 052515 (2011) [arXiv:1103.6221], Deiglmayr et al. PRA 93, 013424 (2016) [arXiv:1601.08005], Sanguinetti et al. J. Phys. B 42, 165004 (2009) [arXiv:0905.0571], and all three Steck datasheets (**revision 2.3.4, 8 August 2025**) were retrieved and their tables transcribed directly. Quantum-defect digits not available from those PDFs were cross-checked verbatim against the ARC v3 source data file (`arc/alkali_atom_data.py`, github.com/nikolasibalic/ARC-Alkali-Rydberg-Calculator), which carries per-series citations; rows resting only on that secondary source are tagged so. Every benchmark number in §6 was *recomputed from the constants in this document* during authoring and closed within the stated tolerance.

---

## 1. Scope

Normative specification for:

- (a) Rydberg–Ritz energies `E_nlj` with the mass-corrected Rydberg constant `R_M`, per isotope.
- (b) Quantum-defect Ritz expansions with complete coefficient tables: Rb-85/Rb-87 (S1/2, P1/2, P3/2, D3/2, D5/2, F5/2, F7/2, G, and l ≥ 5 via the core-polarization formula) and Cs-133 (S1/2, P1/2, P3/2, D3/2, D5/2, F5/2, F7/2, G7/2).
- (c) Ionization limits per isotope, with uncertainties and hyperfine-reference bookkeeping.
- (d) Atomic masses, nuclear spins, natural abundances.
- (e) D-line (probe-transition) data: Rb 5S1/2–5P1/2/5P3/2, Cs 6S1/2–6P1/2/6P3/2 — vacuum frequencies/wavelengths, lifetimes, natural linewidths, oscillator strengths, saturation intensities, reduced dipole matrix elements.
- (f) Hyperfine structure: A/B(/C) constants of ground and first-excited states; exact F-level shift formula; Rydberg-state HFS scaling laws and the normative modeling cutoff.
- (g) Effective quantum number n*, validity window of the Ritz expansion, low-n fallback policy.

Out of scope: radial wavefunctions and matrix elements (spec 02), Stark/Zeeman shifts (spec 07/future), lifetimes and BBR (spec 04), pressure shifts (spec 05), fine structure of hydrogenic high-l manifolds (limitation §7.6).

---

## 2. Equations

### 2.1 Rydberg–Ritz energy (NORMATIVE)

For a Rydberg fine-structure level |n l j⟩ of an alkali species with (spinless-core) ionization limit `E_I`:

```
E_nlj = E_I − h·c·R_M / (n − δ_lj(n))²                    [J]      (1.1)

R_M   = R_∞ / (1 + m_e/M_atom)                            [1/m]    (1.2)
```

- `E_I` [J]: first ionization energy measured from the **hyperfine centroid** of the ground state (see §2.4 — this convention is mandatory; mixing it up injects a 4.27 GHz error for Rb-87).
- `R_∞` [1/m]: Rydberg constant, CODATA via `scipy.constants` (2018 value 10 973 731.568 160(21) m⁻¹; use whatever the installed scipy ships and record it in provenance).
- `m_e/M_atom`: electron-to-atom mass ratio; masses in §3.2. (The "atom" mass convention — not the ion-core mass `M − m_e` — is what Mack and Deiglmayr use in (1.2); the difference shifts R_M by δR/R ≈ (m_e/M)² ≈ 4·10⁻¹¹, i.e. ~0.13 MHz·(n*/30)⁻² on binding energies — below every tolerance here, but pick the atom convention and stay with it.)
- `n − δ_lj(n) ≡ n*` is the effective principal quantum number. Binding energy: `E_b = h·c·R_M/n*²`.
- Unit conversions (exact): 1 cm⁻¹ = 29 979.2458 MHz; `c·R_∞` = 3 289 841.960 250 THz·(scipy value).

Reference R_M values (compute from (1.2); these are cross-checks, not inputs):

```
c·R_M(Rb-87) = 3 289 821.194 55 GHz   [Mack 2011 quotes 3 289 821.194 66(2) GHz; Δ = 0.11 MHz from
                                       CODATA-vintage + mass differences — agreement required ≤ 0.5 MHz]
c·R_M(Rb-85) = 3 289 820.706 08 GHz
R_M(Cs-133)  = 109 736.862 732 cm⁻¹   [Deiglmayr 2016 quotes 109 736.862 733 9(6) cm⁻¹; Δ = 1.7·10⁻⁶ cm⁻¹]
```

### 2.2 Ritz expansion of the quantum defect (NORMATIVE)

Two functional forms appear in the source literature; the coefficients are **not interchangeable** between forms.

**Form A — "modified Ritz" (δ0 in the correction denominators):**

```
δ_lj(n) = δ0 + δ2/(n−δ0)² + δ4/(n−δ0)⁴ + δ6/(n−δ0)⁶ + δ8/(n−δ0)⁸        (1.3)
```

**Form B — self-consistent Ritz (δ(n) in the denominators, solve by iteration):**

```
δ_lj(n) = δ0 + δ2/(n−δ_lj(n))² + δ4/(n−δ_lj(n))⁴ + …                     (1.4)
```

Assignment of forms (from the papers themselves):
- All Rb series (Li 2003, Han 2006, Mack 2011): **Form A** with {δ0, δ2} only. Mack Eq. (2) is explicitly Form A; higher orders "did not lead to improved results" for n ≥ 19.
- Cs nP1/2, nP3/2 (Deiglmayr Table II): **Form A** (their Eq. (5)); they verified Form B refit changes nothing within uncertainty for n ≥ 27.
- Cs nS1/2, nD5/2 (Deiglmayr Table IV): **Form B** (their Eq. (4), fitted down to n = 11/9 with δ4..δ8). Evaluate by fixed-point iteration (§4.2). Using Form A on these coefficients errs by 49 MHz at 12S, < 1 kHz at 47D (measured during authoring — benchmark AS-11).
- Cs nD3/2, nF, nG (Lorenzen–Niemax / Weber–Sansonetti / Bai coefficient sets): treat as **Form B** (extended-Ritz convention of those papers); at n ≥ 20 the A/B difference for these small-δ2 series is < 10 kHz.

### 2.3 High-l states: measured G defects, polarization formula for l ≥ 5

For nonpenetrating states the defect is dominated by core polarization. With the hydrogenic expectation value (a.u.; Bethe–Salpeter standard result, self-checkable against the repo Numerov integrator, spec 02):

```
⟨n l | r⁻⁴ | n l⟩ = [3n² − l(l+1)] / [2 n⁵ (l−½) l (l+½) (l+1) (l+3/2)]      (1.5)

ΔE_pol = −(α_d/2)·⟨r⁻⁴⟩            ⇒
δ_pol(n,l) = (α_d/4)·[3n² − l(l+1)] / [n² (l−½) l (l+½) (l+1) (l+3/2)]
           → (3/4)·α_d / [(l−½) l (l+½) (l+1) (l+3/2)]   (n ≫ l)             (1.6)
```

with α_d the static dipole polarizability of the ion core (a.u.): α_d(Rb⁺) = 9.076, α_d(Cs⁺) = 15.644 [Marinescu et al., PRA 49, 982 (1994), as adopted in ARC; Confidence: VERIFIED-as-adopted]. Self-check built into (1.6): evaluated at l = 4 it must reproduce the *measured* G defects to ≤ 5 % (Rb: 0.003 93 vs measured 0.003 999 → 1.8 %; Cs: 0.006 77 vs measured 0.007 04 → 3.8 % — both verified during authoring; benchmark AS-09). **Normative:** l = 4 uses the measured coefficients (§3.4/§3.5); l ≥ 5 uses (1.6); j-independence assumed for l ≥ 5 (fine structure of these states is hydrogenic-small, §7.6).

### 2.4 Hyperfine structure (NORMATIVE where modeled)

F-level shift relative to the fine-structure level ("centroid"), K ≡ F(F+1) − I(I+1) − J(J+1):

```
ΔE_hfs(F) = (A/2)·K + B · [ (3/2)K(K+1) − 2 I(I+1) J(J+1) ] / [ 4 I(2I−1) J(2J−1) ]    (1.7)
```

(B-term only for I, J > 1/2; the magnetic-octupole C term is retained only for Cs 6P3/2 where it is measured, 0.56 kHz — negligible, keep for provenance.) Ground-state splittings implied by (1.7), used as exact internal checks:

```
Rb-87 (I=3/2): ν(F=2)−ν(F=1) = 2A = 6 834 682 610.904 290(90) Hz
Rb-85 (I=5/2): ν(F=3)−ν(F=2) = 3A = 3 035 732 439.0(60) Hz
Cs    (I=7/2): ν(F=4)−ν(F=3) = 4A = 9 192 631 770 Hz (exact, defines the SI second)
```

**Hyperfine reference of E_I.** Measured ionization frequencies are quoted from a specific hyperfine level; the spec stores E_I from the **5S1/2 / 6S1/2 hyperfine centroid**:

```
E_I(centroid) = E_I(from F) − ΔE_hfs(F)                                          (1.8)
e.g. Rb-87:  E_I(centroid)/h = 1 010 029 164.6(3) MHz − (−5A/4)·(1/h)... :
             ΔE_hfs(F=1) = −(5/4)A ⇒ E_I(centroid)/h = 1 010 029 164.6 − 4 271.68 = 1 010 024 892.9(3) MHz
```

**Rydberg-state HFS scaling (for the modeling-cutoff decision):**

```
A(nS1/2) = A_S · n*⁻³      A_S(Rb-87 nS) = 16.75(45) GHz   [Mack 2011 / Li 2003: 2A·n*³ = 33.5(9) GHz; VERIFIED]
                            A_S(Rb-85 nS) ≈ (A_5S(85)/A_5S(87))·16.75 GHz = 4.96 GHz   [DERIVED-ESTIMATE, ±30 %]
                            A_S(Cs nS)    ≈ 17.1 GHz   [ground-state-scaling estimate ONLY — the same estimate
                            overpredicts the measured Rb-87 coefficient by 33 %, so assign ±40 %; UNVERIFIED —
                            fetch Sassmannshausen/Merkt/Deiglmayr PRA 87, 032519 (2013) before relying on it]
A(Cs np1/2) = 3.85(32) GHz · n*⁻³   [derived from Deiglmayr's measured 27p1/2 F=3–4 interval 1.2(1) MHz = 4A; VERIFIED]
```

**Normative modeling scope:** (i) ground S1/2 — full F resolution, always; (ii) 5P/6P excited states — full F′ resolution with A, B from §3.7 (hot-cell EIT probe spectra are hyperfine-resolved within the Doppler profile); (iii) Rydberg states — include HFS only for nS1/2 with n < 30 (where A > 0.4 MHz, comparable to EIT linewidths); for nS n ≥ 30, nP, nD, nF at all n treat HFS = 0 and carry a stated error bound A_S·n*⁻³ (< 0.5 MHz for Rb nS at n ≥ 30, < 60 kHz for Cs 27P-class states, smaller for higher l).

### 2.5 Transition frequencies

```
ν(n₁l₁j₁ → n₂l₂j₂) = c·R_M · [ 1/(n₁*)² − 1/(n₂*)² ]        [Hz]   (Rydberg–Rydberg; E_I cancels)   (1.9)
ν(probe, F → F′)   = ν_centroid + ΔE_hfs(F′)/h − ΔE_hfs(F)/h        (D lines; ν_centroid from §3.6)  (1.10)
```

Level ordering note: because δ_P − δ_D ≈ 1.30 (Rb) / 1.09 (Cs) > 1, the (n+1)P3/2 level lies **below** nD5/2; the RF/mm-wave transition nD5/2 → (n+1)P3/2 is downward. Report ν > 0 with an explicit `sign` field (§5).

Closure check performed for (1.10) (becomes benchmark AS-07): Steck centroid 384.230 484 468 5 THz + ΔE_hfs(5P3/2, F′=3) − ΔE_hfs(5S1/2, F=2) = 384.230 484 468 5 − 0.002 563 006 + 0.000 193 741 THz = **384.228 115 20 THz**, equal to the independently comb-measured 5S1/2(F=2)→5P3/2(F=3) frequency 384.228 115 2 THz used by Mack. Two sources, one formula, exact agreement.

---

## 3. Constants / parameter tables

Confidence legend: **VERIFIED** = read from the primary source (PDF/datasheet) during authoring. **VERIFIED-ARC** = digits confirmed verbatim in ARC v3 `alkali_atom_data.py` with the stated citation; primary paper not retrieved (paywalled). **LITERATURE-RECALL** = from memory, plausible, self-check indicated. **MISSING** = no sourced value; do not invent.

### 3.1 Fundamental constants

All from `scipy.constants` (CODATA as shipped) — the repo's `rydsim/constants.py` already enforces this. No fundamental constant may be typed by hand. Record `scipy.version` + CODATA vintage in provenance output (`rydsim.provenance`).

### 3.2 Masses, nuclear spin, abundance

| Quantity | Rb-85 | Rb-87 | Cs-133 | Source | Confidence |
|---|---|---|---|---|---|
| Mass [u] | 84.911 789 7379 | 86.909 180 5310 | 132.905 451 9610 | AME/NIST Atomic Weights as adopted in ARC; cross-check Steck 2.3.4 (Bradley et al. PRL 83, 4510 (1999)): 84.911 789 732(14), 86.909 180 520(15), 132.905 451 931(27) — agree ≤ 2·10⁻⁸ u | VERIFIED (both routes) |
| Nuclear spin I | 5/2 | 3/2 | 7/2 | Steck 2.3.4 | VERIFIED |
| Natural abundance | 72.17(2) % | 27.83(2) % | 100 % | Steck 2.3.4 (CRC) | VERIFIED |
| Stability | stable | β⁻, τ = 4.88·10¹⁰ yr | stable | Steck 2.3.4 | VERIFIED |

Mass choice shifts R_M by < 1·10⁻¹⁰ relative between the two routes — irrelevant; adopt the ARC/AME digits.

### 3.3 Ionization limits (ground-state hyperfine **centroid** convention, Eq. 1.8)

| Isotope | E_I/h [MHz] | E_I/(hc) [cm⁻¹] | Source | Confidence |
|---|---|---|---|---|
| Rb-85 | 1 010 024 700(7) | 33 690.797 52(23) | Sanguinetti, Majeed, Jones, Varcoe, J. Phys. B 42, 165004 (2009), Method-3 fit (comb-calibrated, centroid-referenced) | VERIFIED |
| Rb-87 | 1 010 024 892.9(3) | 33 690.803 95(1) | Mack et al., PRA 83, 052515 (2011): E_I(from 5S1/2 F=1)/h = 1 010 029 164.6(3) MHz = 33 690.946 44(1) cm⁻¹, converted per (1.8) with A_5S exact-grade | VERIFIED |
| Cs-133 | 941 542 215.86(4) | 31 406.467 732 5(14) | Deiglmayr, Herburger, Saßmannshausen, Jansen, Schmutz, Merkt, PRA 93, 013424 (2016) | VERIFIED |

Auxiliary verified anchors:
- Rb-87: E_i(from 5P3/2(F=3))/h = 625.794 214 8(3) THz (Mack Table II) — used by absolute benchmarks AS-04..06.
- Cs erratum: PRA 112, 049902(E) (2025) revises E_I(Cs)/h to **941 542 216.33(3)_stat(10)_syst MHz** (+0.47 MHz), prompted by a higher-precision result of Shen et al., 941 542 216.431(4) MHz (citation known only via the erratum summary — UNVERIFIED as an independent reference). **Policy:** keep the 2016 value as normative because it is the value self-consistent with the 2016 quantum-defect fit (§3.5) — the pair reproduces the measured UV lines to < 80 kHz (AS-08). The +0.47 MHz is carried as a declared systematic on Cs *absolute* energies; it cancels exactly in all Rydberg–Rydberg intervals.
- Known tension (Rb-85): Steck 2.3.4 quotes E_I(85Rb) = 33 690.798 90(20) cm⁻¹ = 1 010 024 741(6) MHz from Lee, Helmcke, Hall, Stoicheff, Opt. Lett. 3, 141 (1978) — **41 MHz (≈6σ) above Sanguinetti**. Adopted: Sanguinetti (modern, comb-calibrated). Consequence: Rb-85 absolute optical predictions carry a ±40 MHz systematic risk; Rb-85 microwave intervals are unaffected (E_I cancels). Flagged, not resolved. Isotope-shift sanity: E_I(87) − E_I(85) = 192.9 MHz vs normal-mass-shift estimate ≈ 150 MHz — right order and sign.

### 3.4 Quantum defects — Rb-85 and Rb-87 (Form A, δ = δ0 + δ2/(n−δ0)²)

Rb-85 (also the default for series unmeasured in Rb-87):

| Series | δ0 | δ2 | Source | Confidence |
|---|---|---|---|---|
| nS1/2 | 3.131 180 4(10) | 0.178 4(6) | Li, Mourachko, Noel, Gallagher, PRA 67, 052502 (2003) | VERIFIED (quoted in Mack Table I) |
| nP1/2 | 2.654 884 9(10) | 0.290 0(6) | Li et al. 2003 | VERIFIED-ARC (digits); uncertainties LITERATURE-RECALL |
| nP3/2 | 2.641 673 7(10) | 0.295 0(7) | Li et al. 2003 | VERIFIED-ARC (digits); uncertainties LITERATURE-RECALL |
| nD3/2 | 1.348 091 7(4) | −0.602 9(3) | Li et al. 2003 | VERIFIED (quoted in Mack Table I) |
| nD5/2 | 1.346 465 7(3) | −0.596 0(2) | Li et al. 2003 | VERIFIED (quoted in Mack Table I) |
| nF5/2 | 0.016 519 2(9) | −0.085(9) | Han, Jamil, Norum, Tanner, Gallagher, PRA 74, 054502 (2006) | VERIFIED-ARC (digits); uncertainties LITERATURE-RECALL |
| nF7/2 | 0.016 543 7(7) | −0.086(7) | Han et al. 2006 | VERIFIED-ARC (digits); uncertainties LITERATURE-RECALL |
| nG (l=4) | 0.003 999 0 | −0.020 2 | Raithel-group 2020 measurement as adopted by ARC (`[#Raithel2020]`); primary citation not independently retrieved | VERIFIED-ARC only |
| l ≥ 5 | Eq. (1.6) with α_d = 9.076 | — | Marinescu 1994 via ARC | VERIFIED-ARC; self-check AS-09 |

Rb-87 — isotope-specific replacements where measured (Mack et al. 2011, Table I; all others fall back to the Rb-85 row above — the residual isotope dependence of δ is below 3·10⁻⁶, i.e. ≤ 0.2 MHz at n = 50):

| Series | δ0 | δ2 | Source | Confidence |
|---|---|---|---|---|
| nS1/2 | 3.131 180 7(8) | 0.178 7(2) | Mack 2011 (fine-structure level, HFS-corrected) | VERIFIED |
| nD3/2 | 1.348 094 8(11) | −0.605 4(4) | Mack 2011 | VERIFIED |
| nD5/2 | 1.346 462 2(11) | −0.594 0(4) | Mack 2011 | VERIFIED |
| nG (l=4) | 0.004 05 | 0 | Afrousheh et al. (PRA 74, 062712 (2006) — journal ref LITERATURE-RECALL) as adopted by ARC | VERIFIED-ARC only |

Note: Mack's nD δ0 differ from Li's by ~3σ; the underlying *frequencies* agree — it is an analysis/n-range difference (Mack §IV). Using the wrong isotope's D defects moves a 50D level by ≲ 0.2 MHz: below all §6 tolerances.

### 3.5 Quantum defects — Cs-133

| Series | δ0 | δ2 | δ4 | δ6 | δ8 | Form | Source | Confidence |
|---|---|---|---|---|---|---|---|---|
| nS1/2 | 4.049 353 2(4) | 0.239 1(5) | 0.06(10) | 11(7) | −209(150) | B | Deiglmayr 2016 Table IV (global fit incl. Weber–Sansonetti 11 ≤ n ≤ 31) | VERIFIED |
| nP1/2 | 3.591 587 1(3) | 0.362 73(16) | — | — | — | A | Deiglmayr 2016 Table II (n = 27–74) | VERIFIED |
| nP3/2 | 3.559 067 6(3) | 0.374 69(14) | — | — | — | A | Deiglmayr 2016 Table II | VERIFIED |
| nD3/2 | 2.475 456 2 | 0.009 320 | −0.434 98 | −0.763 58 | −18.006 1 | B | Lorenzen & Niemax, Z. Phys. A 315, 127 (1984) | VERIFIED-ARC (digits); uncertainties MISSING |
| nD5/2 | 2.466 314 4(6) | 0.013 81(15) | −0.392(12) | −1.9(3) | — | B | Deiglmayr 2016 Table IV (incl. Weber–Sansonetti 9 ≤ n ≤ 36) | VERIFIED |
| nF5/2 | 0.033 415 37(70) | −0.201 4(16) | — | — | — | B | Bai, Jiao, Song, Fan, Zhao, Jia, Raithel, PRA 108, 022804 (2023) [arXiv:2304.07974], microwave (n+2)D5/2→nF, n = 45–50 | VERIFIED |
| nF7/2 | 0.033 564 6(13) | −0.205 2(29) | — | — | — | B | Bai et al. 2023 | VERIFIED |
| nG7/2 (l=4) | 0.007 038 65 | −0.049 252 | 0.012 91 | — | — | B | Weber & Sansonetti, PRA 35, 4650 (1987) | VERIFIED-ARC (digits); uncertainties MISSING |
| l ≥ 5 | Eq. (1.6) with α_d = 15.644 | — | — | — | — | — | Marinescu 1994 via ARC | VERIFIED-ARC; self-check AS-09 |

Legacy alternates (for reproduction of older papers only, do NOT mix into the normative set): Weber–Sansonetti 1987 F5/2 δ0 = 0.033 414 24, δ2 = −0.198 674, δ4 = 0.289 53, δ6 = −0.260 1 (VERIFIED-ARC); ARC itself sets F7/2 ≡ F5/2, which is wrong by Δδ0 = 1.49·10⁻⁴ → ~9 MHz at n = 47 (this is the "~10 MHz" ARC's own comment admits; Bai 2023 resolves it — we adopt Bai). W–S ionization limit 31 406.467 66(15) cm⁻¹ (superseded).

### 3.6 D-line optical data (Steck datasheets, all revision 2.3.4, 8 Aug 2025 — canonical; every row VERIFIED from the PDFs)

| Quantity | Rb-85 D2 | Rb-85 D1 | Rb-87 D2 | Rb-87 D1 | Cs D2 | Cs D1 |
|---|---|---|---|---|---|---|
| Transition | 5S1/2→5P3/2 | 5S1/2→5P1/2 | 5S1/2→5P3/2 | 5S1/2→5P1/2 | 6S1/2→6P3/2 | 6S1/2→6P1/2 |
| ν₀ (vacuum, centroid) [THz] | 384.230 406 373(14) | 377.107 385 690(46) | 384.230 484 468 5(62) | 377.107 463 380(11) | 351.725 718 50(11) | 335.116 048 807(41) |
| λ_vac [nm] | 780.241 368 271(27) | 794.979 014 933(96) | 780.241 209 686(13) | 794.978 851 156(23) | 852.347 275 82(27) | 894.592 959 86(10) |
| Lifetime τ [ns] | 26.234 8(77) | 27.679(27) | 26.234 8(77) | 27.679(27) | 30.405(77) | 34.791(90) |
| Γ = 1/τ [10⁶ s⁻¹] | 38.117(11) | 36.129(35) | 38.117(11) | 36.129(35) | 32.889(84) | 28.743(75) |
| Γ/2π [MHz] | 6.066 6(18) | 5.750 0(56) | 6.066 6(18) | 5.750 0(56) | 5.234(13) | 4.575(12) |
| Osc. strength f | 0.695 77(20) | 0.342 31(33) | 0.695 77(20) | 0.342 31(33) | 0.716 4(18) | 0.344 86(90) |
| ⟨J‖er‖J′⟩ [e·a₀] | 4.227 53(62) | 2.993 1(14) | 4.227 52(62) | 2.993 1(14) | 4.483 7(57) | 3.186 9(41) |
| I_sat cycling σ± [mW/cm²] | 1.669 32(49) (F=3→F′=4) | — | 1.669 33(49) (F=2→F′=3) | — | 1.104 9(28) (F=4→F′=5) | — |
| I_sat iso,eff [mW/cm²] | 3.895 1(11) | — | 3.577 1(10) | — | 2.711 9(69) | — |
| I_sat det,eff π [mW/cm²] | 2.503 99(73) | 4.487 6(43) | 2.503 99(73) | 4.487 6(43) | 1.657 3(42) | 2.505 5(65) |
| Isotope shift ν(87)−ν(85) [MHz] | 78.095(12) | 77.583(12) | — | — | — | — |

Convention notes (Steck's): Γ is FWHM angular natural linewidth = A-coefficient; I_sat = c·ε₀·Γ²·ħ²/(4|d|²) with d the dipole moment relevant to the stated driving configuration; the "cycling" row is the stretched-state σ± value; Rb lifetimes are isotope-independent at current accuracy. Internal consistency check (AS-10): 384 230 484.468 5 − 384 230 406.373 = 78.095 5 MHz = the quoted isotope shift. Cs Γ_D2 = 2π·5.234 MHz matches corpus benchmark C5d; ⟨6S‖er‖6P3/2⟩ = 4.4837 matches C5b.

### 3.7 Hyperfine constants, ground + first excited states (Steck 2.3.4; all VERIFIED)

| Constant | Rb-85 | Rb-87 | Cs-133 |
|---|---|---|---|
| A(nS1/2 ground) | h·1.011 910 813 0(20) GHz | h·3.417 341 305 452 145(45) GHz | h·2.298 157 942 5 GHz (exact) |
| A(nP1/2) | h·120.527(56) MHz | h·407.25(63) MHz | h·291.920 1(75) MHz |
| A(nP3/2) | h·25.035 4(69) MHz | h·84.718 5(20) MHz | h·50.288 27(23) MHz |
| B(nP3/2) | h·25.898(91) MHz | h·12.496 5(37) MHz | h·−0.493 4(17) MHz |
| C(nP3/2) | — | — | h·0.560(70) kHz |

(n = 5 for Rb, n = 6 for Cs.) Derived, for implementers' orientation (from (1.7) + §3.6, verified during authoring): Rb-87 5S1/2(F=2)→5P3/2(F′=3) = 384.228 115 20 THz; Cs 6S1/2(F=4) sits +4.021 776 4 GHz above centroid (equals the value Deiglmayr uses) and 6S1/2(F=4)→6P3/2(F′=5) = 351.721 960 6 THz.

### 3.8 Rydberg-state hyperfine scaling

See §2.4: A_S(Rb-87 nS)·n*³ = 16.75(45) GHz [VERIFIED]; Cs nP1/2 coefficient 3.85(32) GHz [VERIFIED-derived]; Rb-85 nS and Cs nS coefficients are estimates (±30–40 %) pending Sassmannshausen 2013 — tagged UNVERIFIED. Rydberg nP3/2/nD/nF hyperfine: no sourced coefficients; treat as 0 with error bound smaller than the corresponding S/P1/2 value at the same n* (contact interaction dominance) — MISSING, declared.

---

## 4. Numerical method + pitfalls

### 4.1 Evaluation

1. Compute R_M from (1.2) at import time per species; assert against the §2.1 cross-check values (tolerance 0.5 MHz on c·R_M) — this catches mass/CODATA regressions.
2. Defects: Form A is closed-form. Form B: fixed-point iteration `d ← f(d)` starting at δ0; contraction factor ~2δ2/n³ ≪ 1, so 4 iterations reach double-precision fixed point for n ≥ 9; iterate to |Δd| < 1e−12 (max 30 iterations, raise if not converged).
3. Energies via (1.1) in Hz (float64: binding ~10¹²–10¹⁵ Hz with eps·E ≈ mHz–Hz — MHz targets have ≥ 6 orders of headroom).
4. Intervals via (1.9) **directly from binding energies**, never as a difference of two absolute optical frequencies computed through E_I (avoids re-adding then cancelling 10¹⁵ Hz; with (1.9) the cancellation error is < 1 Hz at n = 50).
5. Vectorize over n (numpy broadcasting); l, j select the coefficient row.

### 4.2 Validity window of the Ritz expansion (item g)

| Species | MHz-grade (≲ 2 MHz absolute) | 100-MHz-grade | Hard floor (raise below) | Basis |
|---|---|---|---|---|
| Rb-85/87 | n ≥ 19 (S, D: Mack/Li fit range; P: n ≥ 32 measured, interpolation to n ≥ 19 consistent with Sanguinetti data at ≤ 4 MHz) | n ≥ 8 | n = 8 | Mack residuals < 1 MHz for 19–65; ARC `minQuantumDefectN = 8` |
| Cs | n ≥ 25 (P: fit range 27–74; S/D5/2: global fit incl. n ≥ 9 data, Form B mandatory) | n ≥ 12 | n = 12 | Deiglmayr residuals < 110 kHz for 27–74; ARC `minQuantumDefectN = 12` |

Below the hard floor the single-channel Ritz form breaks (core penetration, series perturbers, exchange): the low-lying levels needed by RydSim (Rb 5S/5P, Cs 6S/6P) are **data** (§3.6/3.7), never computed from (1.1). If intermediate levels (Rb 5D, 6P, Cs 7P, …) are ever needed, take them from NIST ASD term tables as data (NOT fetched in this pass — MISSING).

### 4.3 Pitfalls (each is a regression test)

1. **Hyperfine reference of E_I** — Mack's number is from 5S1/2(F=1): forgetting (1.8) misplaces every Rb-87 absolute level by 4.27 GHz. (AS-04 catches this: error would be ×2800 the tolerance.)
2. **Form A vs Form B** for Cs S/D5/2 at low n: 49 MHz at 12S, 3.6 MHz at 15S, < 10 kHz at n ≥ 40. (AS-11.)
3. **R_∞ instead of R_M**: fractional error 6.3·10⁻⁶ (Rb) / 4.1·10⁻⁶ (Cs) → 8.3 MHz on a 50-level binding, ~130 MHz at n = 19. (AS-01 tolerance excludes it.)
4. **cm⁻¹↔MHz**: use exactly 29 979.2458 MHz/cm⁻¹; a 299 792 458-typo (factor 10) or refractive-index-contaminated "air" wavenumbers are the classic failure. All λ in this spec are **vacuum**.
5. **Interval sign**: nD5/2 lies above (n+1)P3/2 (§2.5); publishing signed detunings with the wrong convention flips AT sideband asymmetries downstream (spec 06 consumes this).
6. **Isotope mixing**: Rb-85 defects with Rb-87 R_M (or E_I) produces ~100–200 MHz absolute errors — always resolve species by dataclass instance, never by element name.
7. **ARC's F7/2 ≡ F5/2 shortcut for Cs**: ~9 MHz error at n ≈ 47 — we adopt Bai 2023 instead; do not "fix" the table back to ARC.
8. **δ2-sign typo for Rb D states** (δ2 < 0): flips the n-dependence; caught by AS-02/AS-03 spanning n = 28→53.

### 4.4 Convergence / accuracy budget (absolute optical, n ≈ 50)

E_I uncertainty: 0.3 MHz (Rb-87) / 7 MHz (Rb-85, + 40 MHz systematic risk §3.3) / 0.04 MHz (+0.47 MHz erratum offset, Cs). Defect-fit propagation 2R_M·σ(δ)/n*³: ≲ 0.1 MHz (Rb S/D), ≲ 0.06 MHz (Cs P). Intervals: defect terms only — ≲ 2 MHz at the C4 benchmark points. These budgets set the §6 tolerances; anything failing them is a code bug, not physics.

---

## 5. Recommended Python API (`rydsim/atom.py`)

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Mapping
import numpy as np

RitzForm = Literal["modified", "iterated"]   # Form A / Form B of spec 01 §2.2

@dataclass(frozen=True)
class RitzSeries:
    """Quantum-defect Ritz coefficients for one (l, j) series.

    d0..d8: expansion coefficients per spec 01 §3.4/§3.5.
    form:  'modified' -> Eq. (1.3); 'iterated' -> Eq. (1.4) fixed-point.
    n_min_mhz / n_min_hard: validity floors per spec 01 §4.2.
    source, confidence: provenance strings (mandatory, non-empty).
    """
    l: int
    j: float
    d0: float
    d2: float = 0.0
    d4: float = 0.0
    d6: float = 0.0
    d8: float = 0.0
    form: RitzForm = "modified"
    n_min_mhz: int = 19
    n_min_hard: int = 8
    source: str = ""
    confidence: str = ""

@dataclass(frozen=True)
class HyperfineConstants:
    """A, B, C in Hz (energy/h) for one fine-structure level; Eq. (1.7)."""
    A_hz: float
    B_hz: float = 0.0
    C_hz: float = 0.0
    source: str = ""

@dataclass(frozen=True)
class DLine:
    """One D-line dataset (Steck 2.3.4). Frequencies in Hz, vacuum; SI throughout."""
    name: Literal["D1", "D2"]
    nu0_hz: float                  # hyperfine-centroid transition frequency
    nu0_unc_hz: float
    lambda_vac_m: float
    tau_s: float
    gamma_rad_s: float             # 1/tau, angular FWHM
    f_osc: float
    d_reduced_ea0: float           # <J||er||J'>
    isat_cycling_w_m2: float | None
    isat_iso_w_m2: float | None
    isat_detuned_pi_w_m2: float
    upper_hfs: HyperfineConstants  # A,B(,C) of the 5P/6P level
    source: str = "Steck rev 2.3.4 (2025-08-08)"

@dataclass(frozen=True)
class Species:
    """Immutable per-isotope data bundle. Instances: RB85, RB87, CS133."""
    name: Literal["Rb85", "Rb87", "Cs133"]
    mass_u: float
    nuclear_spin: float
    abundance: float
    e_ion_hz: float                # ionization energy/h from ground hyperfine CENTROID
    e_ion_unc_hz: float
    ground_n: int                  # 5 (Rb), 6 (Cs)
    ground_hfs: HyperfineConstants
    series: Mapping[tuple[int, float], RitzSeries]   # key (l, j)
    alpha_core_au: float           # for Eq. (1.6), l >= 5
    d1: DLine
    d2: DLine

def rydberg_constant_hz(sp: Species) -> float:
    """c*R_M in Hz via Eq. (1.2) from scipy CODATA + sp.mass_u.
    Contract: asserts agreement with the spec 01 §2.1 anchors to 0.5 MHz."""

def quantum_defect(sp: Species, n: np.ndarray | int, l: int, j: float) -> np.ndarray:
    """delta_lj(n). Uses the series' declared form; l>=5 -> Eq. (1.6).
    Raises ValueError below n_min_hard; warns (warnings.warn) below n_min_mhz."""

def n_star(sp: Species, n, l: int, j: float) -> np.ndarray:
    """n - delta_lj(n)."""

def energy_hz(sp: Species, n, l: int, j: float, *, ref: Literal["ionization", "ground_centroid"] = "ionization") -> np.ndarray:
    """E/h of |n l j>. ref='ionization': negative binding energy -c*R_M/n*^2.
    ref='ground_centroid': e_ion_hz - c*R_M/n*^2 (absolute optical scale)."""

def transition_hz(sp: Species, n1, l1: int, j1: float, n2, l2: int, j2: float) -> np.ndarray:
    """Signed E2-E1 in Hz via Eq. (1.9) (binding-energy difference; E_I never enters).
    Positive => state 2 lies above state 1."""

def hyperfine_shift_hz(hfs: HyperfineConstants, I: float, J: float, F: float) -> float:
    """Eq. (1.7). Exact-rational K handling; B term skipped when I<=1/2 or J<=1/2."""

def rydberg_hfs_A_hz(sp: Species, n, l: int, j: float) -> np.ndarray:
    """A(n) = A_S * n*^-3 for (l, j) = (0, 1/2) [+ Cs (1, 1/2)]; 0 with a
    documented bound otherwise. Confidence tags per spec 01 §3.8 in the docstring."""

def probe_transition_hz(sp: Species, line: Literal["D1", "D2"], F: int, Fp: int) -> float:
    """Eq. (1.10): centroid + upper HFS shift - lower HFS shift. Validates F/Fp ranges."""
```

Data population rule: every `RitzSeries`/`DLine`/`HyperfineConstants` instance is constructed with `source=` and `confidence=` exactly as in §3 tables; `rydsim.provenance` must be able to dump the full table with tags (house no-fabrication rule).

---

## 6. Validation benchmarks (→ `tests/test_spec01_benchmarks.py`)

IDs AS-xx; overlap with corpus spec 09 noted. "calc" = value recomputed from this spec's constants during authoring.

| ID | Quantity | Expected | Tol | Source | Confidence |
|---|---|---|---|---|---|
| AS-01 | c·R_M(Rb-87) from (1.2) | 3 289 821 194.66 MHz | ±0.5 MHz | Mack 2011 (calc: …194.55) | VERIFIED |
| AS-01b | R_M(Cs) | 109 736.862 733 9 cm⁻¹ | ±5·10⁻⁶ cm⁻¹ | Deiglmayr 2016 (calc: Δ=1.7·10⁻⁶) | VERIFIED |
| AS-02 | ν(Rb-85 53D5/2→54P3/2) | 14.233 GHz | ±5 MHz | Sedlacek 2012 meas. (their calc 14.232; our calc 14.2317) | VERIFIED (= C4a) |
| AS-02b | ν(Rb-85 50D5/2→51P3/2) | 17.04 GHz | ±10 MHz | Holloway 2014 (calc 17.0415) | VERIFIED (= C4c) |
| AS-03 | ν(Rb-85 28D5/2→29P3/2) | 104.77 GHz | ±50 MHz | Holloway 2014 (calc 104.755) | VERIFIED (= C4d) |
| AS-03b | ν(Rb-87 39D5/2→40P3/2) | 36.9 GHz | ±100 MHz | Tu 2024 (calc 36.895) | VERIFIED (= C4e) |
| AS-03c | ν(Cs 47D5/2→48P3/2) | 6.94 GHz | ±10 MHz | Jing 2020 (calc 6.9452) | VERIFIED (= C4b) |
| AS-04 | ν(Rb-87 5P3/2(F=3)→19S1/2), absolute, via §3.3 anchor | 612.728 838 1 THz | ±1.5 MHz | Mack 2011 Table III (calc +0.9 MHz) | VERIFIED |
| AS-05 | same, n = 20 | 614.232 154 2 THz | ±1.5 MHz | Mack 2011 (calc −0.5 MHz) | VERIFIED |
| AS-06 | same, n = 21 | 615.490 168 7 THz | ±1.5 MHz | Mack 2011 (calc +0.1 MHz) | VERIFIED |
| AS-07 | Rb-87 5S1/2(F=2)→5P3/2(F′=3) via (1.10) | 384.228 115 2 THz | ±0.1 MHz | Steck 2.3.4 + (1.7) vs Mack's comb value | VERIFIED |
| AS-07b | Rb-87 D2 centroid (data integrity) | 384.230 484 468 5 THz | ±10 kHz | Steck 2.3.4 [Ye et al. 1996] | VERIFIED |
| AS-08 | Cs 6S1/2(centroid)→27P1/2 wavenumber | 31 206.189 769 8 cm⁻¹ | ±100 kHz (±3.3·10⁻⁶ cm⁻¹) | Deiglmayr Table I (calc +3.9 kHz) | VERIFIED |
| AS-08b | Cs 6S1/2(centroid)→47P3/2 | 31 348.316 589 8 cm⁻¹ | ±150 kHz | Deiglmayr Table I (calc +79 kHz) | VERIFIED |
| AS-08c | Cs 6S1/2(centroid)→74P3/2 | 31 384.351 900 1 cm⁻¹ | ±150 kHz | Deiglmayr Table I (calc +17 kHz) | VERIFIED |
| AS-09 | δ_pol(l=4, n→∞) (1.6) vs measured G defects | Rb: 0.003 999 / Cs: 0.007 039 | ≤ 5 % rel | §3.4/§3.5 (calc 1.8 % / 3.8 %) | VERIFIED (method) |
| AS-10 | ν(Rb-87 D2) − ν(Rb-85 D2) from §3.6 rows | 78.095 MHz | ±0.024 MHz | Steck 2.3.4 isotope shift row (internal consistency) | VERIFIED |
| AS-11 | Cs 12S1/2: Form B − Form A energy difference | 49.2 MHz | ±5 MHz (documents §4.3-2; asserts Form B is in use) | this spec (analytic/numeric) | VERIFIED (method) |
| AS-12 | Ground hyperfine splittings via (1.7) | 6 834.682 610 904 MHz (Rb-87), 3 035.732 439 0 MHz (Rb-85), 9 192.631 770 MHz (Cs, exact) | ±1 kHz / ±1 kHz / exact | Steck A-constants; Cs defines SI second | VERIFIED |
| AS-13 | Cs 6S1/2(F=4) shift above centroid | +4.021 776 4 GHz | ±1 kHz | (1.7) vs Deiglmayr's quoted correction | VERIFIED |
| AS-14 | Hydrogen limit: δ≡0, M→∞ ⇒ E_b(n)=hcR_∞/n² | exact | 1·10⁻¹² rel | analytic | VERIFIED (analytic) |
| AS-15 | A(30S1/2, Rb-87) = 16.75 GHz·n*⁻³ | 0.864 MHz | ±30 % | Mack/Li scaling §3.8 (n* = 26.87) | VERIFIED (coefficient) |

pytest contract: each row is one parametrized case; AS-x tests must recompute from `rydsim.atom` public API only (no literal reuse of "calc" values); tolerance breaches are release blockers per house rules.

---

## 7. Known limitations / where the model breaks down

1. **Low n.** Single-channel Ritz fails below n_hard (§4.2): series perturbers and core penetration are not modeled; low-lying levels are data-only. No NIST ASD term values were fetched in this pass — intermediate states (Rb 5D/6P, Cs 7P, …) are MISSING until spec 02/06 needs them.
2. **Rb-85 absolute energy scale.** The 41 MHz Lee-1978-vs-Sanguinetti-2009 E_I tension (§3.3) caps Rb-85 absolute optical accuracy at ~40 MHz until an independent modern measurement is adopted. Microwave intervals are immune.
3. **Cs absolute scale.** 2016-consistent set adopted; +0.47 MHz erratum offset declared, not applied (§3.3). Applying the erratum E_I without the erratum's (unretrieved) refitted defects would *worsen* UV-line reproduction — do not cherry-pick.
4. **Rb-85 nP3/2 δ0 tension.** Sanguinetti's Method-3 δ0 = 2.641 57(2) sits ~5σ from Li's 2.641 673 7(10) (different parametrization/fit correlations with E_I). Normative: Li 2003 (used by essentially all electrometry literature). Effect at n = 50: ≲ 2 MHz on intervals — inside the C4 tolerances but visible if tolerances ever tighten.
5. **Quantum defects are fitted constants, not QDT.** Energy dependence beyond the fitted δ2..δ8 window (extrapolation to n > 80 or n below fit ranges) is uncontrolled; expect drift at the few-hundred-kHz level at n = 100+ (Mack observed lines to n = 180 but did not fit them).
6. **High-l fine structure and j-dependence (l ≥ 5)** ignored: hydrogenic FS ~ α²R_M/n³·[1/(j+½) − 3/(4n)] ~ ≲ 100 kHz at n = 50 — below tolerances, unmodeled. G-series j-splitting for Rb is also unresolved in the adopted data.
7. **Rydberg hyperfine** modeled only as §2.4/§3.8 scaling for S (and Cs P1/2) series; Cs nS coefficient is an estimate (±40 %) pending Sassmannshausen 2013; Rydberg D/F hyperfine has no sourced coefficients (bounded, declared MISSING).
8. **No external fields, no interactions.** Stark/Zeeman/BBR/pressure/Rydberg–Rydberg shifts are other specs' territory; energies here are isolated-atom, zero-field values.
9. **Secondary-source rows.** Every row tagged VERIFIED-ARC (Rb P/F/G digits, Cs D3/2/G, α_d values) rests on the ARC data file's transcription of the primary papers; the C4 frequency benchmarks are the designated external self-check for exactly these rows (a transcription error ≥ 10⁻⁴ in δ0 fails them by ≫ 10 MHz). Fetch Li 2003 / Han 2006 / Lorenzen–Niemax 1984 / Weber–Sansonetti 1987 primaries before any release that claims sub-MHz absolute Rydberg accuracy.
