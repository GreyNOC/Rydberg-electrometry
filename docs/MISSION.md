# RydSim Mission

**GreyNOC Rydberg Electrometry Program — the next era, and what we test**

---

## 1. Where the technology is

One decade took Rydberg electrometry from a lab curiosity (Sedlacek 2012) to
a metrology-grade, receiver-grade instrument class. As of mid-2026 the
published frontier stands at:

| Axis | Frontier | Meaning |
|---|---|---|
| Sensitivity | ~10 nV/cm/√Hz, 2.6× above the standard quantum limit | Nearly out of technical noise to remove |
| Continuous coverage | ≤65 nV/cm/√Hz across 1–40 GHz (Zeeman-tuned) | The comb-like response is falling |
| Instantaneous bandwidth | >100 MHz at 147 nV/cm/√Hz (arXiv:2607.25309, Jul 2026) | Superseded 54.6 MHz twice; see correction below |
| Low frequency | 13.5 nV/cm/√Hz @ 100 kHz (sapphire cells) | Screening problem cracked, not closed |
| Reach | ~DC to 0.61+ THz in one physical sensor class | One aperture, the whole spectrum |

*(Sources: Study Report §4–5; Sci. Adv. SQL paper, Comms. Phys. 2026, arXiv:2506.10541, npj QM 2026.)*

## 2. The next era

The last era proved the physics. The next era is **the engineering fight to
field it** — and it will be decided on four fronts:

1. **The quantum-limited wideband receiver.** Sensitivity is within 3× of
   the SQL; bandwidth is not. The defining race is holding near-SQL
   sensitivity while pushing instantaneous bandwidth from ~50 MHz toward
   the GHz class. **This program previously asserted here that "every
   published scheme so far pays for bandwidth in sensitivity". That claim is
   FALSIFIED** and is retained only as the correction below. Whoever quantifies and then beats that trade owns the era.
2. **Chip-scale deployment.** Two stabilized lasers (one at an awkward blue
   wavelength) are the SWaP bottleneck. DARPA's 2026 SBIR thesis is explicit:
   photonic integration to a ruggedized, low-SWaP receiver. Lab performance
   on-chip has not been matched yet — the gap is the opportunity.
3. **The contested spectrum.** An all-dielectric receiver with no metal
   aperture, absolute self-calibration, and HF-to-THz span is a new class of
   EW/spectrum-awareness asset — and a new blind spot for everyone defending
   against one. Its real limits (dynamic range, co-site interference,
   LO leakage, saturation behavior) are barely published. That silence is
   where the operational advantage sits.
4. **Traceability as a weapon of rigor.** The same cell is both a receiver
   and an SI-traceable standard. The next era merges the two operating
   points: fielded sensors that carry their own calibration, measurement
   chains traceable to Planck's constant instead of to a calibration lab.

## 3. Why GreyNOC is here

Spectrum awareness, EW threat modeling, wireless security research, and
metrology discipline are the program's stated relevance (Study Report §10).
We are not building hardware yet: we are building the **simulation-and-
calibration capability** that lets us understand, predict, and stress this
technology class before and while it fields — the same posture as our
detection-validation labs: model it, drive it, measure it, report only what
reproduces.

## 4. Our goals with testing

RydSim exists to turn the open questions of §2 into **numbered, reproducible
findings**. The standing test campaigns:

### T1 — Validate the model against the published record
Reproduce the canonical results (Sedlacek 2012 AT electrometry, Jing 2020
superhet at 55 nV/cm/√Hz, the 10 nV/cm/√Hz SQL benchmark, NIST traceability
scalings) within stated tolerances, from first-principles physics — quantum
defects → wavefunctions → dipoles → OBE/EIT → noise. *No finding ships from
an unvalidated model.*

### T2 — Map the sensitivity–bandwidth frontier
Sweep LO Rabi frequency, coupling power, beam geometry, density, and readout
bandwidth to chart NEF vs IBW as a surface, locate the published operating
points on it, and quantify the trade law the literature reports only as
folklore. Deliverable: the trade-off curve with uncertainty bands.

### T3 — Characterize receiver figures of merit nobody publishes
Dynamic range and 1 dB compression from atomic saturation, spurious-free
dynamic range, equivalent noise temperature vs a conventional front end,
LO-leakage observability, and behavior under strong co-site interferers.
Deliverable: an EW-grade datasheet for an idealized atomic receiver.

### T4 — Quantify the metrology/receiver tension
The most sensitive modes are the least self-calibrated (Study Report §7.7).
Sweep the same cell between AT-traceable and superhet operating points and
put numbers on what traceability costs in sensitivity — and where the dual-
use optimum sits.

### T5 — Stress the low-frequency story
Fold the phenomenological screening model (sapphire vs pyrex, adsorbate
dynamics) into Stark-readout NEF predictions for kHz-and-below sensing;
bound what cell engineering can and cannot buy.

### T6 — Model-honesty regression
Every campaign runs against the integrity gate: cross-method disagreement
reported as uncertainty, `UNVERIFIED` constants surfaced in every report,
and an explicit list of what the simulation *cannot* claim (ion physics,
high-density interactions, cell-wall microphysics, laser technical noise
realism). We publish the caveats with the same prominence as the results.

## 5. Definition of success

- The validation suite reproduces the published record and says exactly how
  well, benchmark by benchmark, graded honestly.
- Each campaign T2–T5 yields at least one finding that is **not** directly
  stated in the literature — a quantified trade curve, a bound, a predicted
  operating point — carried by a full uncertainty budget and reproducible
  from its config hash by anyone with this repo.
- Every claim survives the house rule: **reproducible or it didn't happen.**

---

*GreyNOC · RydSim mission · 2026-08-10 · companion to [COLLABORATION.md](COLLABORATION.md)*


---

## Correction — 2026-08-11 (literature audit)

A web-verified audit of the sensitivity–bandwidth question (24 agents, every
load-bearing claim adversarially refuted) overturned three statements this
document and `Study Report.txt` had been making. They are corrected here rather
than quietly edited, because the program's own record is subject to the same
"reproducible or it didn't happen" rule as its findings.

**1. "Every bandwidth extension pays in sensitivity" is FALSIFIED.**
Two primary sources contradict it. Manchaiah et al. (NIST, arXiv:2509.20632)
state: *"Using optical homodyne detection, we demonstrate that sensitivity is
preserved while achieving a response bandwidth of 8 MHz"* — and the preceding
sentence names the belief as folklore ("is thought to compromise sensitivity").
Separately, arXiv:2607.25309 improves on the same group's earlier result on
**both axes at once** (147 nV/cm/√Hz at >100 MHz, versus 222.6 at 76.8 MHz).

**2. The bandwidth frontier row was stale by two revisions, and was an
interior point.** We quoted 140.4 nV/cm/√Hz at 54.6 MHz from arXiv:2506.10541.
That paper is now at v6 (222.6 nV/cm/√Hz at 76.8 MHz), and 140.4/54.6 was never
its headline — it is one point of an internal Δ_c sweep. We were quoting an
interior operating point as the state of the art.

**3. There is no measured frontier at all.** The four papers that define the
sensitivity axis (Sedlacek 2012, Jing 2020, Sci. Adv. 2024, Commun. Phys. 2026)
**state no instantaneous bandwidth whatsoever**. Every paper that states an IBW
comes from the bandwidth-engineering literature. The "trade-off" is therefore
assembled by placing one lab's sensitivity beside another lab's bandwidth, and
has never been measured as a trade within a single apparatus.

Consequence for §2 of this document: the sensitivity–bandwidth fight is still
the defining engineering problem, but it should be framed as **an open question
rather than an established law** — and the first real measurement of the trade
is a cheaper and more valuable target than beating either axis.
