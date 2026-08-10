# Rydberg Electrometry — Bulletin

**Plain-language field brief.** Everything important about Rydberg-atom RF
sensing, in the order you'd want to learn it. Literature numbers are current
through mid-2026 and come from [Study Report.txt](../Study%20Report.txt);
engineering rules come from the normative specs in [docs/spec/](spec/).

---

## 1. The one-paragraph version

Take an atom. Kick its outermost electron into a huge orbit (principal quantum
number **n ≈ 30–100**). That bloated atom is now absurdly sensitive to electric
fields — so sensitive that a radio wave hitting it visibly reshapes the color
of light it absorbs. Shine two lasers through a glass cell of warm rubidium
vapor, watch the transmitted light, and you have an **antenna-free RF receiver
whose calibration comes from Planck's constant instead of from a calibration
lab.** No metal, no cryogenics, DC to terahertz, in a sensor smaller than the
wavelength it measures.

---

## 2. Why anyone cares (the honest version)

**Atoms win at:**

| Advantage | Why |
|---|---|
| **Self-calibration** | Field strength comes out of a measured frequency and a *calculable* atomic dipole moment. SI-traceable by construction. |
| **Span** | One physical sensor covers what would take a rack of antennas — HF to 0.61+ THz demonstrated. |
| **Sub-wavelength size** | Sensing volume can be far smaller than λ. Near-field imaging without perturbing the field. |
| **Stealth** | All-dielectric. No metal aperture, near-zero scattering cross-section — nothing for a counter-detector to see or couple into. |
| **No self-perturbation** | A metal probe distorts the field it measures. A glass cell doesn't. |

**Antennas still win at:** raw sensitivity in a single band, dynamic range,
size/weight/power, cost, and maturity. The best vapor cells (~10 nV/cm/√Hz) are
within roughly 10× of a mainstream engineered receiver front end — impressive,
not yet dominant.

> **Rule of thumb:** atoms are chosen for *span, traceability, size, and
> stealth* — not because they out-sensitive a good antenna in its own band.

---

## 3. How it actually works — four steps

1. **Probe laser** (780 nm in Rb) drives ground → first excited state
   (5S₁/₂ → 5P₃/₂). You measure how much of this light gets through the cell.
2. **Coupling laser** (480 nm) drives first excited → Rydberg state
   (5P₃/₂ → nD or nS).
3. **EIT** — with both lasers on resonance, quantum interference punches a
   narrow *transparency window* in the probe absorption. Probe transmission is
   now a live, all-optical readout of the Rydberg level. **This is the trick
   that made the whole field possible** (Mohapatra 2007).
4. **Autler–Townes** — an RF field resonant with a Rydberg→Rydberg transition
   dresses the levels and **splits the single EIT peak into two**. The gap is
   proportional to field strength.

```
        no RF                  with RF
      ┌──▲──┐               ┌─▲───▲─┐
  ────┘  │  └────       ────┘ │   │ └────      splitting Δf = d·E / h
       EIT peak            AT doublet
```

**The measurement in one line:**

> **E = (h / d) · Δf_AT** — measure a frequency splitting, divide by a computed
> atomic dipole moment, get volts per meter. No reference field required.

---

## 4. The scaling laws (why n matters so much)

| Property | Scales as | Consequence |
|---|---|---|
| Orbital radius | **n²** | Enormous electron cloud |
| Binding energy | n⁻² | Levels crowd near ionization |
| Level spacing | n⁻³ | Rydberg↔Rydberg transitions land in MHz–THz — i.e. *the RF spectrum* |
| Rydberg↔Rydberg dipole | **n²** | Thousands of atomic units → extreme RF coupling |
| Polarizability | **n⁷** | Brutal sensitivity to DC/slow fields |
| Radiative lifetime | n³ | Long-lived states, narrow lines |

Two of these do the heavy lifting: **n² dipole** (why RF drives the atom so
hard) and **n⁷ polarizability** (why DC fields shift levels so far).

---

## 5. The four detection regimes — pick one

| Regime | When to use | Trade |
|---|---|---|
| **Resonant AT** | Strong fields (≳ mV/cm) | Self-calibrated and accurate; least sensitive |
| **Amplitude** | Weak resonant fields, splitting unresolved — read peak height/width | More sensitive, less directly traceable |
| **AC Stark / off-resonant** | Carrier far from any Rydberg resonance | Continuous frequency coverage; reduced sensitivity |
| **Superheterodyne** | The modern default | Best sensitivity + phase recovery; least self-calibrated |

**Superhet in one sentence:** add a strong local-oscillator RF field, let the
weak signal beat against it, and the probe transmission oscillates at the
difference frequency — which you read with ordinary lock-in/DSP hardware.
Jing 2020 got ~2 orders of magnitude out of this and everyone has built on it
since.

**The tension you can't escape:** the most sensitive modes are the least
self-calibrated. Metrology-grade and receiver-grade are *different operating
points of the same cell*.

---

## 6. Scoreboard — state of the art, mid-2026

| Metric | Best published | Who |
|---|---|---|
| Microwave sensitivity | **~10 nV/cm/√Hz** (2.6× above the standard quantum limit) | Sci. Adv. SQL paper; Princeton time-separated fields |
| Continuous-band sensitivity | ≤65 nV/cm/√Hz across **1–40 GHz** | Zeeman-tuned superhet (Comms. Phys. 2026) |
| kHz-band sensitivity | 13.5 nV/cm/√Hz @ 100 kHz | Sapphire-cell self-dressing (npj QM 2026) |
| Instantaneous bandwidth | **54.6 MHz** (at 140 nV/cm/√Hz) | Multi-dress-state (arXiv:2506.10541) |
| Frequency reach | ~DC to **0.61+ THz** | Multi-band / high-ℓ OAM receivers |
| Traceability | SI-traceable fields *and* volts | NIST (Holloway et al.) |

**Reference point:** a typical GPS front end sits around the 1 nV/cm/√Hz
equivalent level. So the atoms are close, not ahead.

---

## 7. The papers that built the field

| Year | Work | Why it matters |
|---|---|---|
| 1999 | Osterwalder & Merkt | Rydberg states proposed as E-field sensors — the origin |
| **2007** | **Mohapatra, Jackson & Adams** | **EIT readout of Rydberg states in hot vapor — the enabling trick** |
| 2010 | Kübler / Pfau (micro-cells); Gordon & Holloway | Miniaturization; the SI-traceable probe concept |
| **2012** | **Sedlacek et al., Nat. Phys.** | **The landmark: microwave electrometry via AT splitting at room temperature. Everything cites this** |
| 2014 | Holloway et al. | Broadband self-calibrated probe; NIST becomes the metrology backbone |
| 2015 | Fan et al. (tutorial) | "Radio to terahertz" — establishes the reach, gives the working equations |
| 2017 | Kumar / Shaffer | Optical homodyne — first big sensitivity architecture beyond bare EIT |
| 2019–20 | NIST + Army Research Lab | Real receivers: AM/FM, PSK, phase, angle-of-arrival, streaming video |
| **2020** | **Jing et al., Nat. Phys.** | **Atomic superheterodyne, 55 nV/cm/√Hz + phase. The architecture that now dominates** |
| 2020 | Jau & Carter | Sub-1-kHz sensing; confronts wall-adsorption screening |
| 2022 | Holloway et al. | Traceable **voltage**, not just fields |
| 2024 | Romalis group (Princeton) | Ramsey/time-separated fields → 10 nV/cm/√Hz at 10 GHz |
| **2024** | **Nature Reviews Physics review** | **Read this first if you read only one thing** |

---

## 8. What's still broken (the open problems)

1. **Sensitivity ↔ bandwidth.** EIT coherence dynamics couple gain to response
   time. Every bandwidth extension so far is paid for in sensitivity. *This is
   the defining race.*
2. **Comb-like native response.** Sensitivity peaks only at discrete
   Rydberg–Rydberg resonances. Zeeman/Stark/dressing tuning fills the gaps, at
   the cost of hardware and calibration burden.
3. **Dynamic range.** Atoms saturate. Linear-range characterization is recent;
   behavior under strong co-site interferers is *largely unpublished*.
4. **Low-frequency screening.** Free charges and alkali adsorbed on cell walls
   screen kHz-and-below fields. Sapphire and resistivity engineering help;
   nothing closes it.
5. **Laser SWaP.** Two stabilized lasers, one at an awkward blue wavelength, is
   *the* deployment bottleneck. Chip-scale photonics is the declared path
   (explicitly DARPA's thesis) and hasn't matched lab performance yet.
6. **Environmental drift.** Stray DC fields, cell-wall effects, temperature,
   vapor density — all shift lines.
7. **Traceability vs sensitivity.** See §5. You cannot have both maxed at once.

---

## 9. Where the money and the pressure are

- **Metrology (mature)** — NIST Boulder: traceable field standards, antenna
  calibration, EMC, near-field imaging, traceable voltage.
- **Comms (advanced prototyping)** — atomic receivers demodulating
  AM/FM/PSK/QAM, multi-band in one cell; framed for 6G in the 2026
  "Rydberg Atomic Quantum Radio" survey.
- **Defense (funded, accelerating)** — DARPA 2026 SBIR for low-SWaP ruggedized
  wideband receivers (up to $5M, chip-scale photonics emphasis); India's DRDO
  national prototype; quantum-radar receiver concepts.
- **Commercial** — Rydberg Technologies (Ann Arbor, MI — Raithel/Anderson
  lineage) sells SI-traceable atomic probes; Infleqtion and others ship cells
  and photonics.

**Groups to know:** NIST Boulder · Army Research Lab (Meyer, Cox, Kunz) ·
Shaffer lineage · Adams (Durham) · Pfau/Kübler (Stuttgart) ·
Raithel/Anderson (Michigan) · Shanxi University (superhet inventors) ·
Romalis (Princeton) · DRDO/IIT Delhi.

---

## 10. Bulletins from our own bench (RydSim)

Hard-won rules from building the simulator. These are the things that silently
give wrong answers.

**Convention locks — mix these up and every number is wrong:**

- **Angular units internally** (rad/s). Hz only at API boundaries, converted at
  exactly one site.
- **Detuning sign:** Δ = ω_field − ω_atom (blue positive). Weak-probe
  denominators are **(γ − iΔ)** — much of the literature uses the opposite sign.
- **Ω = d·ℰ/ħ with ℰ the *peak* amplitude, never RMS.** The RWA Hamiltonian
  carries −ħΩ/2 off-diagonal. On-resonance AT separation equals Ω.
- **Γ vs γ:** Γ = population decay (also the Lorentzian FWHM in angular units);
  γ_ij = coherence decay (HWHM-type). Never interchange.
- **PSDs are one-sided; NEF is an amplitude spectral density.** There are four
  independent ±3 dB traps here (one/two-sided, amplitude/RMS, RBW/ENBW,
  log-average 2.51 dB) — all of them are explicit parameters in our code.

**Physics gotchas that bit us for real:**

- 🐛 **Doppler quadrature aliasing.** Coarse Gauss–Hermite velocity quadrature
  *silently* biases AT peaks — EIT features are ~1 m/s wide and would need
  ~3×10⁵ GH nodes. Uniform/composite grids with a mandatory halving-convergence
  gate only. **Caught as a real bug; now locked by regression test.**
- 🐛 **Transit channel factor 2.** The measure-and-replace Kraus set must
  include |g⟩⟨g|, or ground coherences decay at half the correct rate.
  **Also a real caught bug.**
- ⚠️ **The λc/λp mismatch factor (0.6152 for Rb 780/480) must *emerge* from the
  velocity average** — it is a validation observable, not an input. Probe-scanned
  splitting is *compressed* by this factor; coupling-scanned is factor 1.
- ⚠️ **Never import a literature Rabi frequency.** Recompute Ω from (d, ℰ).
  Published Ω and E values are sometimes mutually inconsistent by their own
  convention artifacts.
- ⚠️ **Polarizability sign traps.** α₀(Rb nD) > 0, but the m_j = 1/2 *total*
  α = α₀ − 0.8·α₂ can be negative. Test sign anchors on α₀, never on α(m_j).

**House rule, non-negotiable:** *reproducible or it didn't happen.* When a
result isn't numerically converged the engine raises `IntegrityError` rather
than returning a number. Every finding ships with its uncertainty budget,
constants provenance, and caveats.

---

## 11. Cheat sheet

```
Level energies      E_n = −hcR / (n − δ_ℓ)²          δ_ℓ = quantum defect
Field from AT       E   = (h / d) · Δf_AT
Rabi frequency      Ω   = d·ℰ / ħ          (ℰ = peak amplitude)
Quadratic Stark     ΔE  = −½ α E²                    α ∝ n⁷
Doppler AT factor   Δf_measured = (λ_c/λ_p) · Ω_RF/2π    (probe scanned)
Resonant σ          σ₀  = 3λ² / 2π
Shot-limited field  E_min ≈ (h/d)·Γ_EIT /(SNR·√T)
```

**Glossary:** **AT** Autler–Townes (RF-induced EIT doublet) · **EIT**
electromagnetically induced transparency · **IBW** instantaneous bandwidth ·
**LO** local oscillator · **NEF** noise-equivalent field · **SQL** standard
quantum limit (atom projection noise) · **quantum defect** the correction to
the hydrogenic level formula from alkali core penetration · **vapor cell**
sealed glass/sapphire cell of alkali vapor at ~room temperature.

---

## 12. Read next, in this order

1. **Nature Reviews Physics 2024** — the map of the territory.
2. **Sedlacek 2012** + **Mohapatra 2007** — the two founding papers. Short.
3. **Fan 2015 (J. Phys. B tutorial)** — scope and working equations.
4. **Jing 2020** — superheterodyne, the architecture everything builds on.
5. **Meyer 2020 (J. Phys. B 53, 034001)** — the skeptical wideband assessment.
   Calibrates the hype. Do not skip.
6. Frontier singles as needed: SQL paper (Sci. Adv.), Zeeman 1–40 GHz
   (Comms. Phys. 2026), multi-dress bandwidth (arXiv:2506.10541), sapphire kHz
   cells (npj QM 2026), Princeton time-separated fields (arXiv:2406.05106).

**Hands-on tooling:** **ARC** (Alkali Rydberg Calculator, open-source Python) —
dipole moments, polarizabilities, Stark maps, transition frequencies. The
standard for designing a sensing scheme on paper. And **RydSim** (this repo)
for the full EIT/AT → superhet → noise-budget chain with provenance.

---

*GreyNOC · RydSim · companion to [MISSION.md](MISSION.md),
[Study Report.txt](../Study%20Report.txt), and the normative specs in
[docs/spec/](spec/) · reproducible or it didn't happen.*
