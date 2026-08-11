# 04 — Lifetimes, Decay Channels and the Dephasing Budget

RydSim physics specification, document 04. Species: Rb-85, Rb-87, Cs-133.
Python 3.11 + numpy/scipy, first-principles (no ARC, no qutip).

**Network status of this revision:** WebSearch/WebFetch were AVAILABLE when this
document was written (2026-08-10). The Beterov et al. tables (radiative-lifetime
fits, BBR fits, effective-lifetime tables) were extracted directly from
arXiv:0810.0339v4 (6 Nov 2009); the BBR-ionization formula and coefficients from
arXiv:0807.2535; the self-broadening table from arXiv:1107.3092v2. Items that
could not be re-verified online are tagged LITERATURE-RECALL or UNVERIFIED in
the tables below. **arXiv:0810.0339v4 postdates the published Erratum
[Phys. Rev. A 80, 059902 (2009)] and is presumed to incorporate it; the APS
erratum page itself returned HTTP 403 and could not be checked directly.**

**Second verification pass (2026-08-10, same day):** all primary sources were
re-parsed from the source PDFs. Verified verbatim: Beterov Tables I, II, VII,
VIII (arXiv:0810.0339v4 pages 4–6, 10–12); NJP Eq. (27) + Tables 1–2
(arXiv:0807.2535v4); Weller Table I (arXiv:1107.3092v2); Steck datasheets
**revision 2.3.4 (8 Aug 2025)** for Rb-87, Rb-85, Cs. One correction resulted:
the Cs 6P3/2 lifetime/linewidth in the previous revision (30.473(39) ns /
5.2227(66) MHz) came from an older Steck revision; the current datasheet gives
**30.405(77) ns / Gamma = 32.889(84)e6 s^-1 / Gamma/2pi = 5.234(13) MHz**
(sections 2.3.1, 3.4, benchmark B1b updated). Note the unresolved ~1.2%
tension with Patterson et al., PRA 91, 012506 (2015): 30.462(46) ns.

---

## 1. Scope

This document specifies:

- (a) Radiative lifetime `tau_0` of a Rydberg state `|n L J>` as a sum of
  Einstein A coefficients; truncation rules; the fitted scaling
  `tau_0 = tau_s * n_eff^delta` for Rb and Cs nS/nP/nD.
- (b) Blackbody (BBR) depopulation at finite T: full sum, the Beterov analytic
  approximation, BBR redistribution to nearby states, BBR photoionization.
- (c) The term-by-term dephasing budget of ladder (cascade) Rydberg EIT in a
  thermal vapor cell (Rb 780/480 nm, Cs 852/509 nm), with formulas and typical
  magnitudes.
- (d) The homogeneous/inhomogeneous classification of every term and the
  corresponding entry point into the optical Bloch equations (Lindblad rate vs
  statistical average). This classification is normative: implementations MUST
  NOT convert inhomogeneous terms into Lindblad rates in "accurate" mode.
- (e) Default parameter sets "typical cell" and "good cell".

Dependencies on sibling specs:

- Radial matrix elements `R(nL -> n'L')` and model-potential/Numerov machinery:
  wavefunction/matrix-element spec (doc 02/03 of this series).
- Quantum defects and energy levels: level-structure spec. Values quoted here
  (Table 3.4) are convenience copies for `n_eff`; the level-structure spec is
  authoritative.
- Stark shifts/polarizabilities: Stark spec. Only the scaling and the entry
  into the dephasing budget are given here.

Notation: `Gamma` denotes rates in s^-1 (angular frequency units, i.e. the
Lorentzian FWHM in rad/s equals the population decay rate for a two-level
line); `Gamma/2pi` is in Hz. `gamma_ij` denotes the decay rate of coherence
`rho_ij` in s^-1. `n_eff = n - mu_L(n)` is the effective principal quantum
number. SI units unless explicitly marked a.u.

---

## 2. Equations

### 2.1 Einstein A coefficients and radiative lifetime

Spontaneous rate for `|n L J> -> |n' L' J'>` (E1, unpolarized initial state,
summed over final m'):

```
A(nLJ -> n'L'J') = (omega^3 * e^2) / (3 * pi * eps0 * hbar * c^3)
                   * (2J' + 1) * sixj(L, J, S; J', L', 1)^2
                   * L_max * R(nL -> n'L')^2                     [s^-1]
```

- `omega = (E_nLJ - E_n'L'J') / hbar > 0` — transition angular frequency [rad/s].
- `e, eps0, hbar, c` — SI constants (CODATA 2018, exact where SI-defined).
- `S = 1/2`; `sixj(...)` — Wigner 6-j symbol `{L J S; J' L' 1}`.
- `L_max = max(L, L')`.
- `R(nL -> n'L') = ∫ P_nL(r) r P_n'L'(r) dr` — radial integral [m], from the
  matrix-element spec (`P = r * R_radial`). In atomic units multiply by `a0`.

Consistency check (fine structure summed): `sum_{J'} A` must reduce to the
L-basis rate `(omega^3 e^2 / 3 pi eps0 hbar c^3) * (L_max/(2L+1)) * R^2`.
Implement this as a unit test — it catches 6-j and degeneracy-factor bugs.

Radiative (0 K) lifetime:

```
1 / tau_0 = Gamma_0 = sum_{E_n'L'J' < E_nLJ}  A(nLJ -> n'L'J')      (Beterov Eq. 4)
```

Fitted scaling (Beterov et al. 2009, Eq. 15; coefficients in Table 3.1):

```
tau_0 = tau_s * n_eff^delta        [ns]
```

Valid for approximately `15 <= n <= 80` (fit built on quasiclassical
calculations over that range; do not extrapolate below n ~ 15).

### 2.2 BBR depopulation

Planck occupation per mode:

```
nbar(omega, T) = 1 / (exp(hbar*omega / (kB*T)) - 1)
```

BBR-stimulated rate on each dipole-allowed transition (both up and down):

```
W(nL -> n'L') = A(nL -> n'L') * nbar(omega_nn', T)
Gamma_BBR = sum_{n'L'} W(nL -> n'L')                              (Beterov Eq. 5)
```

where for upward transitions the "A coefficient" is computed with the same
formula evaluated at `omega = |omega_nn'|` using the matrix element of the
transition (absorption is stimulated only; no spontaneous term). Effective
lifetime:

```
1 / tau_eff(T) = Gamma_0 + Gamma_BBR                              (Beterov Eq. 6)
```

Beterov analytic approximation (Eq. 14; coefficients A, B, C, D in Table 3.2):

```
Gamma_BBR_fit = (A / n_eff^D) * 2.14e10 / ( exp( 315780 * B / (n_eff^C * T) ) - 1 )   [s^-1]
```

(equivalently `21.4 ns^-1` as printed in their Eq. 16; `T` in kelvin. The
constant 315780 K is their rounding of `E_h/kB = 315775.02 K`; keep 315780
inside the fit — it is part of the fit.) Combined analytic effective lifetime
(their Eq. 16):

```
tau_eff_fit(T) = [ 1/(tau_s * n_eff^delta)  +  Gamma_BBR_fit * 1e-9 ]^-1    [ns]
```

Paper-stated accuracy: better than 5% against their numerical results for
`15 < n < 80`. See §6 for the D3/2 caveat found during verification.

High-T/low-omega analytic limit (Farley & Wing 1981) — use as an independent
cross-check only, expected within ~+30%/−20% of the full sum for n ~ 40–70 at
300 K:

```
Gamma_BBR_FW = (4/3) * alpha_fs^3 * (kB*T/E_h) / n_eff^2   [atomic units of rate]
             = 2.034e7 * (T/300 K) / n_eff^2               [s^-1]
```

BBR redistribution: the BBR-stimulated transfer is dominated by
`n -> n±1, n±2` transitions into neighboring `n'L±1` states (for nS: nearby
n'P). These states are dark to the EIT lasers. For the OBE they are a
*population loss* channel from `|r>` (see §2.4); they are NOT a pure dephasing.

BBR photoionization (direct), Beterov et al., New J. Phys. 11, 013052 (2009),
Eq. (27), extracted from arXiv:0807.2535:

```
W_BBR_ion = A_L * (11500 * T / n_eff^(7/3))
            * [ cos^2(Delta_plus + pi/6) + cos^2(Delta_minus - pi/6) ]
            * ln( 1 / (1 - exp( -157890 / (T * n_eff^2) )) )        [s^-1]
```

- `Delta_plus = pi * (mu_L - mu_{L+1})`, `Delta_minus = pi * (mu_{L-1} - mu_L)`
  (quantum-defect differences, Table 3.3). For nS keep only the first
  (L' = L+1) term.
- `A_L` — empirical scaling coefficient (Table 3.3). With `A_L = 1` accuracy is
  ~50%; with tabulated `A_L`, better.
- `T` in kelvin. 157890 = 315780/2 (Rydberg in K, their rounding).

Worked value: Rb 50S1/2 at 300 K (`n_eff = 46.8688`, `mu_S - mu_P = 0.490134`,
`A_S = 1`): `W_BBR_ion ≈ 1.5e2 s^-1` — about 1% of the total depopulation rate
(1/65.2 us = 1.53e4 s^-1). BBR ionization is therefore negligible for the EIT
linewidth but is THE seed for free-ion production in the cell (feeds the
charge-field term, §2.3.8). The additional indirect channels of that paper
(SFI of BBR-populated high states, mixed terms) are not modeled; the direct
term underestimates total ionization by a factor ~1.3–2 (paper's own
comparison).

### 2.3 Dephasing budget for ladder Rydberg EIT

Three-level ladder: `|g> = 5S1/2 (Rb) / 6S1/2 (Cs)`, `|e> = 5P3/2 / 6P3/2`,
`|r> = nS1/2 or nD_J`. Probe `Omega_p` on g–e (780.241 nm / 852.347 nm),
coupling `Omega_c` on e–r (~479–484 nm / ~508–512 nm), counter-propagating.

#### 2.3.1 Intermediate-state natural linewidth

```
Gamma_e = 1/tau_e
Rb 5P3/2:  tau_e = 26.2348(77) ns   ->  Gamma_e/2pi = 6.0666(18) MHz   [VERIFIED, Steck rev 2.3.4]
Cs 6P3/2:  tau_e = 30.405(77) ns    ->  Gamma_e/2pi = 5.234(13) MHz    [VERIFIED, Steck rev 2.3.4;
                                        cf. Patterson 2015: 30.462(46) ns, ~1.2% tension]
```

Enters as Lindblad decay `|e> -> |g>` (see §2.4 for hyperfine-leak handling).
Its effect on the EIT *two-photon* linewidth is indirect: in the weak-probe,
homogeneous, `Omega_c << Gamma_e` limit the transparency-window FWHM is

```
FWHM_EIT (rad/s) ≈ 2*gamma_gr + Omega_c^2 / Gamma_e        [Fleischhauer RMP 77, 633 (2005)]
```

where `gamma_gr` is the total ground–Rydberg coherence decay rate. In a
Doppler-broadened medium this closed form is NOT quantitative; the simulator
must obtain the lineshape from the velocity integral (§2.4). Use the closed
form only for sanity checks.

#### 2.3.2 Rydberg-state total decay

```
gamma_r_pop = Gamma_0(n,L,J) + Gamma_BBR(n,L,J,T)  ( + W_BBR_ion, optional)
```

Example Rb 50S1/2, 300 K: `Gamma_r/2pi = 1/(2*pi*65.18 us) = 2.44 kHz`
(radiative part 1.13 kHz, BBR part 1.32 kHz). Contribution to `gamma_gr` is
`Gamma_r/2` — of order kHz, i.e. negligible against transit and laser terms,
but it sets the population loss during pulse propagation and MUST be kept in
the Lindblad part.

#### 2.3.3 Transit-time broadening

Convention chosen (state it in code docstrings): an atom with transverse speed
`v` crossing a Gaussian beam of 1/e^2 **intensity** radius `w` through the
axis sees the field amplitude envelope `exp(-(v t)^2 / w^2)`; its spectrum is
Gaussian with

```
FWHM_omega = 2*sqrt(2*ln 2) * v / w        =>        Delta_nu_transit = sqrt(2*ln 2)/pi * v / w
                                                                     = 0.3748 * v / w   [Hz]
```

(Derivation: |FT of exp(-a t^2)|^2 = exp(-omega^2/(2a)) with a = v^2/w^2.
This is the standard Gaussian-beam transit result, cf. Demtröder, *Laser
Spectroscopy*, transit-time broadening section — prefactor re-derived here, so
the formula is self-validating.) Thermal average over the 2-D transverse
Maxwell distribution `P(v) = (m v/kB T) exp(-m v^2 / 2 kB T)`:

```
v_bar_2D = sqrt(pi * kB * T / (2 * m))
Delta_nu_transit ≈ 0.3748 * v_bar_2D / w        [Hz, FWHM; convention accuracy ~±20%]
```

Numbers: Rb-87, 300 K: `v_bar_2D = 212.3 m/s`; `w = 0.75 mm` gives
`Delta_nu_transit ≈ 106 kHz`. Cs-133, 300 K: `v_bar_2D = 171.7 m/s`; same beam
gives 86 kHz. Scale as `sqrt(T)/w`.

Competing conventions in the literature (document, do not mix): (i) rate
`gamma_t = v_bar_2D / w` used as a Lindblad-style reset rate (no 0.3748
factor); (ii) most-probable speed `sqrt(2 kB T/m)` instead of `v_bar_2D`
(+13%); (iii) 1/e amplitude radius instead of 1/e^2 intensity radius (identical
numerically — they are the same length; note it to avoid double conversion).
Fast mode uses the reset model with `gamma_t = pi * Delta_nu_transit` matched
so the linear-response FWHM equals the formula above; accurate mode does a
Monte-Carlo/quadrature average over impact parameter and speed (§4).

#### 2.3.4 Laser linewidths and mutual coherence

Phase-diffusion (Wiener) model for each laser: Lorentzian line of FWHM
`Delta_nu_i` corresponds to white frequency noise giving coherence decay
`gamma_i = pi * Delta_nu_i` [s^-1] per photon of that laser in the coherence.
In the rotating frame the laser frequency noises `delta_p(t)`, `delta_c(t)`
enter H as `-delta_p |e><e| - (delta_p + delta_c) |r><r|` (ladder!). For
independent lasers this yields

```
gamma_ge += pi*Delta_nu_p
gamma_er += pi*Delta_nu_c
gamma_gr += pi*(Delta_nu_p + Delta_nu_c)
```

**Ladder-scheme warning (classic error):** in a ladder the two-photon
resonance involves the SUM `omega_p + omega_c`. Common-mode frequency noise
therefore ADDS on `rho_gr` (for perfectly correlated noise the variance is
larger than for independent lasers), unlike a Lambda scheme where the
difference cancels. Locking both lasers to one cavity helps only because it
reduces each laser's absolute noise; it does not cancel anything in the
two-photon coherence. Noise cancellation is possible only if one field is
derived such that its frequency noise is anti-correlated with the other
(difference-frequency/comb schemes). The API therefore takes a mutual PSD
correlation coefficient `c in [-1, 1]`:

```
gamma_gr_laser = pi * ( Delta_nu_p + Delta_nu_c + 2*c*sqrt(Delta_nu_p*Delta_nu_c) )
```

with `c = 0` for independent lasers (default), `c > 0` for common-mode
(worse), `c < 0` only for engineered anti-correlation.

Non-Lorentzian (1/f) laser noise makes the "linewidth" ill-defined; then the
white-noise Lindblad model overestimates or underestimates wings — accurate
mode may sample slow frequency offsets as an inhomogeneous term (§2.4).

#### 2.3.5 Collisional broadening

(i) Resonant self-broadening of the probe line (impact regime, valid for
detunings below the Weisskopf frequency ~2pi*4 GHz):

```
Gamma_self = beta * N_g          (adds to the FWHM of the g-e line;
                                  coherence rate gamma_ge += beta*N_g/2)
beta_D2/2pi (theory, Lewis 1980):  sqrt(2) * Gamma_e * (lambda/2pi)^3 * ... =
    Rb D2: 1.03e-7 Hz cm^3;  measured 1.10(17)e-7  [Kondo et al., PRA 73, 062504 (2006)]
    Cs D2: 1.16e-7 Hz cm^3;  measured 1.15(23)e-7  [Akulshin et al., JETP Lett. 36, 303 (1982)]
    (Rb D1: theory 0.73e-7, measured 0.69(4)e-7 — Weller et al. 2011)
```

Closed theory form (Weller et al. Eq. 2–3): `beta_1 = 2 d1^2/(9 hbar eps0)
= 2pi * Gamma_1 (lambda_1/2pi)^3`, `beta_2 = sqrt(2) * 2 d2^2/(9 hbar eps0)
= 2pi * sqrt(2) * Gamma_2 (lambda_2/2pi)^3` — implement the formula and check
it reproduces 1.03e-7 for Rb D2 (self-test). Magnitude: `N_g = 1e10 cm^-3`
(Rb, ~25 °C) gives 1.1 kHz — negligible; `N_g = 2e13 cm^-3` (Rb, ~130 °C)
gives ~2.2 MHz — dominant. Affects `gamma_ge` (and hence EIT contrast/width
via the intermediate state), NOT directly `gamma_gr`.

(ii) Rydberg–ground collisions (Fermi pseudopotential + polarization
interaction; Fermi 1934, Omont, J. Phys. France 38, 1343 (1977)). Mean-field
line shift of the Rydberg level:

```
Delta_nu_Fermi = (hbar * a_s / m_e) * N_g   [Hz]  = -9.9e-8 Hz cm^3 * N_g[cm^-3]  for Rb
```

with `a_s(e^- – Rb(5S), triplet) = -16.1 a0` [LITERATURE-RECALL, Bahrim &
Thumm]; broadening of the same order as the shift (perturber-density
fluctuations + polarization scattering). Rb at 1e10 cm^-3: ~1 kHz
(negligible); at 2e13 cm^-3: shift ≈ -2 MHz and comparable broadening —
dominant in hot cells, and n-independent for n >~ 30 (the Rydberg orbit
contains many perturbers). Treat the shift as deterministic (recalibrate line
center) and the broadening as pure dephasing on `gamma_gr`, coefficient
configurable; default broadening = |shift| * 0.5 with a 2x uncertainty flag
[UNVERIFIED — self-check: compare simulated hot-cell EIT width vs measured
density-dependent widths].

(iii) Rydberg–Rydberg van der Waals dephasing:

```
V(r) = -C6 / r^6,     C6 ∝ n^11
|C6|(Rb 60S1/2) ≈ 140 GHz um^6  [LITERATURE-RECALL, Singer et al., J. Phys. B 38, S295 (2005);
                                 self-check: |C6| = n^11*(c0 + c1 n + c2 n^2) a.u.,
                                 (c0, c1, c2) = (11.97, -0.8486, 3.385e-3) for Rb nS1/2;
                                 1 a.u. C6 = 1.4448e-19 GHz um^6]
```

Estimate the dephasing as the nearest-neighbor shift at Rydberg density
`N_r`: `r_nn ≈ (3/(4 pi N_r))^(1/3)`, `Delta_nu_vdW ≈ |C6| / (h r_nn^6)`
(scales as `N_r^2 n^11`). Rb 50S, `N_r = 1e8 cm^-3` (`r_nn = 13.4 um`,
|C6|(50S) ≈ 140*(50/60)^11 ≈ 19 GHz um^6): ~3 kHz. At `N_r = 1e9`: ~0.6 MHz.
The simulator must expose `N_r` (steady-state Rydberg density from the OBE
solution * ground density) and warn when `Delta_nu_vdW > 0.1 MHz` — this is
the dominant systematic when the coupling/probe powers are turned up.
Inhomogeneous (random nearest-neighbor distances): enters as a statistical
average, not a Lindblad rate, in accurate mode.

#### 2.3.6 DC-field and charge-induced inhomogeneous broadening

Quadratic Stark: `Delta_nu = -(1/2) alpha_0 E^2 / h`, `alpha_0 ∝ n_eff^7`.
Order of magnitude `alpha_0(Rb 50S) ~ 6e2 MHz/(V/cm)^2` [UNVERIFIED — take the
authoritative value/scaling from the Stark spec; self-check against n^7
scaling]. A stray/inhomogeneous field of rms `E_rms` gives an inhomogeneous
width `~ (1/2) alpha_0 <E^2> / h`; 10 mV/cm on Rb 50S: ~30 kHz; 50 mV/cm:
~0.75 MHz. Single ion at distance r produces `E = 1.44e-9/r^2 V·m` →
0.14 V/cm at 10 um, 1.4 mV/cm at 100 um: keep ion density below ~1e6 cm^-3
(mean spacing >~ 100 um). Ion production is seeded by W_BBR_ion (§2.2),
photoionization by the beams, and Penning collisions; the simulator models a
user-supplied stationary field distribution `P(E)` (default: delta at
E_stray plus optional Holtsmark for ion density `N_ion`) and averages —
inhomogeneous, never a Lindblad rate.

#### 2.3.7 Power broadening

Emerges from the coherent dynamics — DO NOT add as a rate. For sanity checks:
probe saturation broadens the one-photon line to
`Gamma_e' = Gamma_e sqrt(1 + 2 Omega_p^2/Gamma_e^2)`; the EIT window grows as
`Omega_c^2/Gamma_e` (homogeneous, §2.3.1). Example: `Omega_c/2pi = 3 MHz`
(Rb): `Omega_c^2/Gamma_e / 2pi = 1.5 MHz` — usually the largest controllable
term. Beam-profile inhomogeneity of `Omega_{p,c}(r)` across the Gaussian beam
is an inhomogeneous average (radial quadrature), significant when quoting
absolute contrast.

#### 2.3.8 Doppler and residual (mismatch) Doppler

1-D velocity along beam axis: `f(v) = exp(-v^2/2 sigma_v^2)/sqrt(2 pi) sigma_v`,
`sigma_v = sqrt(kB T/m)` (169.4 m/s for Rb-87 at 300 K → one-photon Doppler
FWHM 511 MHz at 780 nm). Counter-propagating ladder: detunings per velocity
class `delta_p -> delta_p - k_p v`, `delta_c -> delta_c + k_c v` (signs per
geometry), residual two-photon wavevector `Delta k = k_c - k_p > 0` for
Rb/Cs (inverted-wavelength ladder). The naive residual width
`Delta k * sigma_v` (~136 MHz sigma for Rb) is NOT observed: velocity
selection by the probe Lorentzian collapses the effective contribution to
`~ (Delta k/k_p) * Gamma_e'` ~ 0.625 * 6.1 ≈ 3.8 MHz (Rb), 0.67 * 5.2 ≈
3.5 MHz (Cs). This arises automatically from the velocity integral
[Gea-Banacloche et al., PRA 51, 576 (1995)]; it is the main reason hot-cell
Rydberg EIT lines are a few MHz. MUST be treated by explicit velocity-class
integration; adding `Delta k sigma_v` (or any Doppler width) to `gamma_gr` is
the canonical modeling error and is forbidden in this codebase.

### 2.4 Homogeneous vs inhomogeneous — normative OBE mapping

Master equation per parameter class `xi` (velocity, position/intensity, field,
nn-distance): `drho/dt = -i/hbar [H(xi), rho] + L_hom[rho]`, then observables
are `<O> = ∫ P(xi) Tr[O rho_ss(xi)] dxi`.

| # | Term | Type | OBE entry |
|---|------|------|-----------|
| 1 | Gamma_e spontaneous | Homogeneous | Lindblad collapse `sqrt(Gamma_e_closed)|g><e|`; hyperfine leak fraction to sink/repump model |
| 2 | Gamma_r = Gamma_0 + Gamma_BBR | Homogeneous | Collapse `sqrt(b_e Gamma_r)|e><r|` for the small branch back to `|e>` (b_e = A(r->e)/Gamma_r), remainder `sqrt((1-b_e)Gamma_r)|s><r|` to a sink state `|s>` (BBR-populated dark states + other cascades) |
| 3 | W_BBR_ion | Homogeneous | Loss `|ion><r|`; feeds N_ion for §2.3.6 |
| 4 | Laser phase noise (white) | Homogeneous | Dephasing: `gamma_ge += pi Dnu_p`, `gamma_er += pi Dnu_c`, `gamma_gr += pi(Dnu_p+Dnu_c+2c sqrt(Dnu_p Dnu_c))`; via Lindblads `sqrt(2 pi Dnu_p)(P_e+P_r)`, `sqrt(2 pi Dnu_c) P_r` (verify the resulting gamma_ij matrix in a unit test — factor-2 conventions are the #1 bug source) |
| 5 | Slow laser drift (1/f) | Inhomogeneous | Average over two-photon detuning offset distribution |
| 6 | Transit (fast mode) | Pseudo-homogeneous | Reset: `drho/dt += gamma_t (rho_0 - rho)`, `rho_0 = |g><g|`, `gamma_t = pi * Delta_nu_transit` |
| 7 | Transit (accurate) | Inhomogeneous | MC/quadrature over chord + speed; time-dependent Omega(t) envelopes |
| 8 | Self-broadening (g–e) | Homogeneous | `gamma_ge += beta N_g / 2`; add collisional shift to delta_p |
| 9 | Rydberg–ground collisions | Homogeneous (impact) | `gamma_gr += Gamma_Ryd-g/2`; deterministic Fermi shift on delta_2ph |
| 10 | Rydberg–Rydberg vdW | Inhomogeneous (quasi-static) | Average over nn-distance distribution; fast mode: warn threshold only |
| 11 | DC Stark / ions | Inhomogeneous | Average over P(E); shift `-(1/2) alpha E^2` on `|r>` |
| 12 | Doppler + mismatch | Inhomogeneous | Velocity integral with `k_p`, `k_c` signs; Gauss–Hermite quadrature |
| 13 | Beam profile Omega(r) | Inhomogeneous | Radial average of steady-state observable |
| 14 | Power broadening | — | Emerges from H; never added as a rate |

Homogeneous rates add on coherences:
`gamma_ij = (Gamma_i_pop + Gamma_j_pop)/2 + sum(pure dephasings touching ij)`.

---

## 3. Constants and parameter tables

Confidence: VERIFIED = checked against the cited source during writing of this
revision; LITERATURE-RECALL = standard value recalled with high confidence,
not re-checked today; UNVERIFIED = must be validated by the self-checks named.

### 3.1 Radiative-lifetime fit `tau_0 = tau_s * n_eff^delta` [ns] (0 K)

Source: Beterov, Ryabtsev, Tretyakov, Entin, PRA 79, 052504 (2009) + Erratum
PRA 80, 059902 (2009); values extracted from arXiv:0810.0339v4 Table II.

| Species | Series | tau_s [ns] | delta | Source | Confidence |
|---|---|---|---|---|---|
| Rb | nS1/2 | 1.368  | 3.0008 | Beterov 2009 Table II | VERIFIED |
| Rb | nP1/2 | 2.4360 | 2.9989 | Beterov 2009 Table II | VERIFIED |
| Rb | nP3/2 | 2.2214 | 3.0026 | Beterov 2009 Table II | VERIFIED |
| Rb | nD3/2 | 1.0761 | 2.9898 | Beterov 2009 Table II | VERIFIED |
| Rb | nD5/2 | 1.0687 | 2.9897 | Beterov 2009 Table II | VERIFIED |
| Cs | nS1/2 | 1.2926 | 3.0005 | Beterov 2009 Table II | VERIFIED |
| Cs | nP1/2 | 2.9921 | 2.9892 | Beterov 2009 Table II | VERIFIED |
| Cs | nP3/2 | 3.2849 | 2.9875 | Beterov 2009 Table II | VERIFIED |
| Cs | nD3/2 | 0.6580 | 2.9944 | Beterov 2009 Table II | VERIFIED |
| Cs | nD5/2 | 0.6681 | 2.9941 | Beterov 2009 Table II | VERIFIED |

Identical fits apply to Rb-85 and Rb-87 (quantum defects differ negligibly at
this precision). Range `15 <= n <= 80`.

### 3.2 BBR analytic-fit coefficients (Eq. Gamma_BBR_fit, §2.2)

Source: same paper, Table I (arXiv:0810.0339v4).

| Species | Series | A | B | C | D | Confidence |
|---|---|---|---|---|---|---|
| Rb | nS1/2 | 0.134 | 0.251 | 2.567 | 4.426 | VERIFIED |
| Rb | nP1/2 | 0.053 | 0.128 | 2.183 | 3.989 | VERIFIED |
| Rb | nP3/2 | 0.046 | 0.109 | 2.085 | 3.901 | VERIFIED |
| Rb | nD3/2 | 0.033 | 0.084 | 1.912 | 3.716 | VERIFIED |
| Rb | nD5/2 | 0.032 | 0.082 | 1.898 | 3.703 | VERIFIED |
| Cs | nS1/2 | 0.123 | 0.231 | 2.517 | 4.375 | VERIFIED |
| Cs | nP1/2 | 0.041 | 0.072 | 1.693 | 3.607 | VERIFIED |
| Cs | nP3/2 | 0.038 | 0.056 | 1.552 | 3.505 | VERIFIED |
| Cs | nD3/2 | 0.038 | 0.076 | 1.790 | 3.656 | VERIFIED |
| Cs | nD5/2 | 0.036 | 0.073 | 1.770 | 3.636 | VERIFIED |

### 3.3 BBR photoionization (Beterov NJP 11, 013052 (2009))

| Quantity | Value | Source | Confidence |
|---|---|---|---|
| Rb: mu_S - mu_P | 0.490134 | NJP Table 1 (arXiv:0807.2535) | VERIFIED |
| Rb: mu_P - mu_D | 1.29456 | NJP Table 1 | VERIFIED |
| Rb: mu_D - mu_F | 1.34636 | NJP Table 1 | VERIFIED |
| Cs: mu_S - mu_P | 0.458701 | NJP Table 1 | VERIFIED |
| Cs: mu_P - mu_D | 1.12661 | NJP Table 1 | VERIFIED |
| Cs: mu_D - mu_F | 2.43295 | NJP Table 1 | VERIFIED |
| A_S, A_P, A_D (Rb) | 1, 1, 0.6 | NJP Table 2 | VERIFIED |
| A_S, A_P, A_D (Cs) | 0.85, 1.1, 0.35 | NJP Table 2 | VERIFIED |

### 3.4 Intermediate states, quantum defects, atomic data

| Quantity | Value | Source | Confidence |
|---|---|---|---|
| Rb-87 5P3/2 lifetime | 26.2348(77) ns | Steck, Rubidium 87 D Line Data, rev 2.3.4 (parsed from PDF) | VERIFIED |
| Rb-87 5P3/2 Gamma/2pi | 6.0666(18) MHz (Gamma = 38.117(11)e6 s^-1) | Steck Rb87 rev 2.3.4 | VERIFIED |
| Rb-85 5P3/2 (same values) | 26.2348(77) ns / 6.0666(18) MHz | Steck, Rubidium 85 D Line Data, rev 2.3.4 (parsed from PDF) | VERIFIED |
| Cs 6P3/2 lifetime | 30.405(77) ns | Steck, Cesium D Line Data, rev 2.3.4 (parsed from PDF); cf. 30.462(46) ns (Patterson et al., PRA 91, 012506 (2015)) — ~1.2% unresolved tension | VERIFIED (datasheet value) |
| Cs 6P3/2 Gamma/2pi | 5.234(13) MHz (Gamma = 32.889(84)e6 s^-1) | Steck Cs rev 2.3.4 | VERIFIED |
| Rb 5P1/2 (D1) lifetime / Gamma/2pi | 27.679(27) ns / 5.7500(56) MHz (Gamma = 36.129(35)e6 s^-1) | Steck Rb87 rev 2.3.4 | VERIFIED |
| Rb-87 D2 wavelength (vac) | 780.241209686(13) nm | Steck Rb87 rev 2.3.4 | VERIFIED |
| Rb-85 D2 wavelength (vac) | 780.241368271(27) nm | Steck Rb85 rev 2.3.4 | VERIFIED |
| Cs D2 wavelength (vac) | 852.34727582(27) nm | Steck Cs rev 2.3.4 | VERIFIED |
| Rb coupling wavelength | 479–484 nm (n-dep.) | level-structure spec | (derived) |
| Cs coupling wavelength | 508–512 nm (n-dep.) | level-structure spec | (derived) |
| Rb nS1/2 delta_0 | 3.1311804 | Li et al., PRA 67, 052502 (2003); Mack et al., PRA 83, 052515 (2011) | LITERATURE-RECALL (authoritative copy: level-structure spec) |
| Rb nP1/2 / nP3/2 delta_0 | 2.6548849 / 2.6416737 | Li et al. 2003 | LITERATURE-RECALL |
| Rb nD3/2 / nD5/2 delta_0 | 1.3480917 / 1.3464657 | Li et al. 2003 | LITERATURE-RECALL |
| Cs nS1/2 delta_0 | 4.0493532 | Goy et al., PRA 26, 2733 (1982); Weber & Sansonetti, PRA 35, 4650 (1987) | LITERATURE-RECALL |
| Cs nP1/2 / nP3/2 delta_0 | 3.5915871 / 3.5590676 | Goy et al. 1982 | LITERATURE-RECALL |
| Cs nD3/2 / nD5/2 delta_0 | 2.4754562 / 2.4663091 | Goy et al. 1982 | LITERATURE-RECALL |
| delta_2 corrections | see level-structure spec | — | MISSING here by design |
| m(Rb-87) | 86.909180520(15) u | Steck Rb87 rev 2.3.4 (parsed from PDF) | VERIFIED |
| m(Rb-85) | 84.911789732(14) u | Steck Rb85 rev 2.3.4 | VERIFIED |
| m(Cs-133) | 132.905451931(27) u | Steck Cs rev 2.3.4 | VERIFIED |
| kB | 1.380649e-23 J/K (exact) | SI 2019 | VERIFIED (definition) |
| E_h/kB | 315775.02 K | CODATA 2018 | LITERATURE-RECALL |
| 1 a.u. rate | 4.1341373e16 s^-1 | CODATA 2018 | LITERATURE-RECALL |
| alpha_fs | 7.2973525693e-3 | CODATA 2018 | LITERATURE-RECALL |

Note: `n_eff` in the Beterov fits uses `mu_L(n) = delta_0 + delta_2/(n-delta_0)^2`;
at n = 50 the delta_2 term shifts mu by <1e-3 — negligible for the fits but use
the full expression for consistency with the level-structure spec.

### 3.5 Collisional / interaction coefficients

| Quantity | Value | Source | Confidence |
|---|---|---|---|
| beta_D2/2pi (Rb), theory | 1.03e-7 Hz cm^3 | Lewis, Phys. Rep. 58, 1 (1980) via Weller et al., J. Phys. B 44, 195006 (2011) Table I | VERIFIED |
| beta_D2/2pi (Rb), measured | 1.10(17)e-7 Hz cm^3 | Kondo et al., PRA 73, 062504 (2006), via Weller Table I | VERIFIED |
| beta_D1/2pi (Rb), measured | 0.69(4)e-7 Hz cm^3 | Weller et al. 2011 | VERIFIED |
| beta_D2/2pi (Cs), theory / measured | 1.16e-7 / 1.15(23)e-7 Hz cm^3 | Weller Table I; Akulshin et al., JETP Lett. 36, 303 (1982) | VERIFIED |
| a_s(e−Rb 5S, triplet) | −16.1 a0 | Bahrim & Thumm, PRA (2000/2001) | LITERATURE-RECALL; self-check: Fermi-shift coefficient −9.9e-8 Hz cm^3 vs measured hot-cell shifts |
| C6 fit, Rb nS1/2 | n^11(11.97 − 0.8486 n + 3.385e-3 n^2) a.u. | Singer et al., J. Phys. B 38, S295 (2005) | LITERATURE-RECALL; self-check: |C6|(60S) ≈ 140 GHz um^6, sign = repulsive for Rb nS pairs |
| alpha_0(Rb 50S1/2) | ~6e2 MHz/(V/cm)^2, ∝ n_eff^7 | Stark spec (authoritative) | UNVERIFIED here |
| N_g(Rb, 25 °C / 100 °C / 130 °C) | ~1e10 / ~5e12 / ~2e13 cm^-3 | Alcock et al. 1984 vapor-pressure model (via Steck) | LITERATURE-RECALL; implement the Alcock model, do not hardcode |

---

## 4. Numerical method and pitfalls

### 4.1 Radiative sum

1. Enumerate final states `n'L'J'` with `E < E_nLJ`, `L' = L ± 1`, `J' = J, J±1`
   (respecting |L'-1/2| <= J' <= L'+1/2). Lowest `n'` per series from the
   species ground configuration (Rb: n' >= 5 for P; Cs: n' >= 6).
2. The sum converges rapidly: the `omega^3` factor makes the *lowest* states
   dominant (e.g. Rb nS: ~70–80% of Gamma_0 from 5P, 6P). Truncation rule:
   include ALL lower states below, and stop when the running tail contributes
   `< 1e-4` relative for three consecutive n' — typically n' <= n is enough
   for tau_0 (there are no higher states in the 0 K sum).
3. Pitfall: the accuracy bottleneck is `R(nL -> n'L')` for n' ~ 5–7 where
   quasiclassical and Coulomb approximations are poor. Use the model-potential
   Numerov elements (matrix-element spec). Expected residual error on tau_0:
   ~few %, dominated by these low-n' elements (this is why Beterov's nP fits
   disagree with de Oliveira experiments at the several-% level).
4. Vectorize over n' arrays; precompute quantum defects; cache radial
   integrals keyed by (species, n, L, j, n', L', j').

### 4.2 BBR sum

1. Include both downward (A*(nbar)) and upward (A(|omega|)*nbar) transitions.
   Upward states: `n' <= n + 30` captures >99.9% at 300 K for n >= 30
   (transfer peaks at n' = n ± 1, 2); apply the same 1e-4 tail criterion.
2. Use `numpy.expm1` for `1/(exp(x)-1)`; for optical downward transitions
   `x = hbar omega/kB T ~ 60` — `exp(-x)` underflow is fine, but never compute
   `1/(exp(x)-1)` via `exp(x)` overflow path for x > 700 (guard).
3. Quasi-degenerate pairs (fine structure) — compute omega from the actual
   level energies, not from n_eff differences, or the nbar factor is wrong for
   the smallest omegas where nbar ~ kB T/hbar omega is large.
4. Continuum (BBR ionization): use §2.2 Eq. (27); do NOT try to extend the
   discrete sum to the continuum by brute force in v1.

### 4.3 Self-checks (mandatory tests)

- S1: `sum_{J'} A` reduction identity (§2.1) to 1e-12.
- S2: tau_0(full sum) vs `tau_s n_eff^delta` within 10% for 20 <= n <= 70
  (S, P series; D5/2), see D3/2 caveat §6.
- S3: Gamma_BBR(full sum) vs Beterov analytic fit within 20% for
  20 <= n <= 70; tau_eff within 10%.
- S4: Farley–Wing limit ratio `Gamma_BBR / Gamma_BBR_FW` in [0.7, 1.2] for
  40 <= n <= 70 at T = 300 K.
- S5: beta_D2 formula reproduces 1.03e-7 Hz cm^3 (Rb), 1.16e-7 (Cs) to 3%.
- S6: computed tau(5P3/2) from the same A machinery = 26.23 ns (Rb) /
  30.41 ns (Cs) within 3% — ties the Rydberg machinery to Steck.
- S7: Lindblad dephasing wiring: with only laser noise on, steady-state probe
  line FWHM = Gamma_e/2pi + Delta_nu_p (weak probe, no Doppler), and EIT dip
  FWHM -> 2*gamma_gr/2pi as Omega_c -> 0.

### 4.4 Velocity / inhomogeneous integration

- Doppler: Gauss–Hermite quadrature over v with `sigma_v = sqrt(kB T/m)`;
  41 nodes gives <1e-3 relative error on EIT lineshapes for Rb at 300 K
  (validate by doubling nodes; the integrand has structure at
  `v ~ Gamma_e/k_p ~ 4.7 m/s` — check node coverage there, this is the classic
  under-resolution pitfall; if needed, use adaptive scipy.integrate.quad in a
  band |v| < 5 Gamma_e'/k_p plus GH tails).
- Beam profile: 8–16 point radial Gauss–Legendre over the Gaussian intensity
  weight.
- Transit accurate mode: 2-D quadrature over (impact parameter b, speed v);
  time-dependent OBE integration over the chord; 100–300 trajectories
  suffice for 1% observables (converge-test).
- Stark: user P(E); default delta-function; Holtsmark option
  `P(E) ~ Holtsmark(E; N_ion)`.
- Steady state: solve `L rho = 0` with trace constraint via LU on the
  (dim^2) Liouvillian (dim = 3 or 4 with sink — tiny); never time-evolve to
  steady state in fast mode.

---

## 5. Recommended Python API

```python
# rydsim/lifetimes.py — numpy-vectorized; all rates s^-1, times s, T kelvin.

from dataclasses import dataclass, field
from typing import Literal
import numpy as np

Species = Literal["Rb85", "Rb87", "Cs133"]
Series  = Literal["S1/2", "P1/2", "P3/2", "D3/2", "D5/2"]

@dataclass(frozen=True)
class RydState:
    species: Species
    n: int
    L: int
    J: float          # 0.5, 1.5, 2.5

def n_eff(species: Species, n: np.ndarray | int, L: int, J: float) -> np.ndarray:
    """n - mu_L(n); mu from level-structure spec (delta_0 + delta_2 term)."""

def einstein_A(upper: RydState, lower: RydState, *, radial: "RadialProvider") -> float:
    """Spontaneous rate, Eq. 2.1. Raises if not E1-allowed. Unit test: 6j reduction S1."""

def radiative_lifetime(state: RydState, *, method: Literal["sum", "fit"] = "sum",
                       radial: "RadialProvider | None" = None,
                       tail_rtol: float = 1e-4) -> float:
    """tau_0 [s]. 'sum': Eq. 2.1/§4.1 (requires radial). 'fit': Beterov Table II
    (valid 15<=n<=80; ValueError outside). Vectorized companion:
    radiative_lifetime_fit(species, series, n_array)."""

def bbr_depopulation_rate(state: RydState, T: float, *,
                          method: Literal["sum", "beterov_fit", "farley_wing"] = "sum",
                          radial: "RadialProvider | None" = None) -> float:
    """Gamma_BBR [s^-1], §2.2. 'farley_wing' is the cross-check limit only."""

def bbr_ionization_rate(state: RydState, T: float) -> float:
    """Direct BBR photoionization, NJP Eq. 27 with Table-2 A_L. Accuracy ~50%
    (worse: excludes indirect SFI/mixed channels, factor ~1.3-2 low)."""

def effective_lifetime(state: RydState, T: float, **kw) -> float:
    """1/(Gamma_0 + Gamma_BBR). Benchmarked against Beterov Tables VII/VIII."""

# --- dephasing budget -------------------------------------------------------

@dataclass
class CellParams:
    T_cell_K: float = 300.0
    N_ground_cm3: float | None = None       # None -> Alcock vapor model(T)
    beam_waist_w_m: float = 0.75e-3         # 1/e^2 intensity radius, probe/coupling matched
    E_stray_Vcm: float = 0.010
    N_ion_cm3: float = 0.0
    N_rydberg_cm3: float = 0.0              # 0 -> self-consistent from OBE solution

@dataclass
class LaserParams:
    fwhm_probe_Hz: float = 2.0e5
    fwhm_coupling_Hz: float = 2.0e5
    noise_correlation: float = 0.0          # c in [-1,1], §2.3.4; ladder: c>0 hurts

@dataclass
class DephasingBudget:
    """All rates s^-1 (angular). Homogeneous entries feed Lindblad/gamma_ij;
    inhomogeneous entries carry distributions, not rates."""
    Gamma_e: float
    Gamma_r_rad: float
    Gamma_r_bbr: float
    W_bbr_ion: float
    gamma_transit: float                    # reset rate, fast mode
    gamma_laser_ge: float
    gamma_laser_er: float
    gamma_laser_gr: float
    gamma_self_ge: float                    # beta*N_g/2
    gamma_rydgnd_gr: float
    shift_fermi_Hz: float
    inhomo: dict = field(default_factory=dict)  # {"doppler": sigma_v, "stark": P(E),
                                                #  "vdw": P(r_nn), "beam": (w, quad),
                                                #  "transit_mc": sampler}
    def gamma_ij(self) -> dict[tuple[str, str], float]:
        """Coherence decay matrix per §2.4 (explicitly unit-tested, S7)."""
    def warnings(self) -> list[str]:
        """e.g. vdW dephasing > 100 kHz at requested powers/density."""

def build_budget(state: RydState, cell: CellParams, lasers: LaserParams,
                 Omega_p: float, Omega_c: float) -> DephasingBudget: ...

def transit_fwhm_Hz(T: float, mass_kg: float, w_m: float) -> float:
    """0.3748 * sqrt(pi kB T / 2 m) / w  — convention of §2.3.3 (documented)."""

def self_broadening_fwhm_Hz(species: Species, line: Literal["D1", "D2"],
                            N_cm3: np.ndarray) -> np.ndarray:
    """(beta/2pi)*N; beta from closed form (S5) with measured-value override."""

def default_cell(kind: Literal["typical", "good"]) -> tuple[CellParams, LaserParams]: ...
```

Contracts: every function validates 15 <= n <= 80 for fit paths; every fit
path cross-checks against the sum path in the test suite, never at runtime;
all rate outputs are angular (s^-1) and every docstring states the /2pi
conversion explicitly.

### 5.1 Simulator defaults

| Parameter | "typical" cell | "good" cell |
|---|---|---|
| T_cell | 300 K | 300 K |
| w (1/e^2 radius) | 0.75 mm | 1.0 mm |
| Probe/coupling FWHM | 200 kHz / 200 kHz | 10 kHz / 10 kHz (cavity-locked) |
| noise_correlation c | 0 | 0 |
| E_stray | 10 mV/cm | 2 mV/cm |
| N_ion | 0 | 0 |
| Omega_c/2pi | 3 MHz | 1 MHz |
| Omega_p/2pi | 1 MHz | 0.3 MHz |
| Resulting budget (Rb 50S): transit | 106 kHz | 79 kHz |
| laser gamma_gr/pi (FWHM-equiv.) | 400 kHz | 20 kHz |
| Stark inhom. | ~30 kHz | ~1 kHz |
| Gamma_r/2pi | 2.4 kHz | 2.4 kHz |
| power term Omega_c^2/Gamma_e/2pi | 1.5 MHz | 0.16 MHz |
| Predicted EIT FWHM (velocity-integrated) | ~3–5 MHz | ~0.5–1 MHz |

Default-parameter values are engineering choices (UNVERIFIED as "typical" in
any statistical sense); the resulting typical-cell EIT width must land in the
measured 2–10 MHz band (benchmark B10).

---

## 6. Validation benchmarks (pytest)

Tolerances: "fit" rows test the analytic formulas exactly as specified
(tight); "sum" rows test the first-principles machinery (loose, limited by
low-n' matrix elements).

| ID | Quantity | Expected | Tol | Source | Confidence |
|----|----------|----------|-----|--------|------------|
| B1 | Rb 5P3/2 lifetime from A-sum machinery | 26.2348 ns | 3% | Steck Rb87 datasheet | VERIFIED |
| B1b | Cs 6P3/2 lifetime from A-sum machinery | 30.405 ns | 3% (covers the 30.462 ns Patterson value) | Steck Cs datasheet rev 2.3.4 | VERIFIED |
| B2 | Rb 50S1/2 tau_0 (0 K), fit path | 141.26 us (fit) vs 141.31 us (table) | 0.5% fit; 10% sum | Beterov Table VII / Table II | VERIFIED |
| B3 | Rb 50S1/2 tau_eff (300 K) | 65.18 us | 2% analytic (Eq. 16 gives 64.4 us); 10% sum | Beterov Table VII | VERIFIED |
| B4 | Rb 50S1/2 tau_eff (77 K) | 109.87 us | 10% sum | Beterov Table VII | VERIFIED |
| B5 | Rb 60S1/2 tau_0 / tau_eff(300 K) | 252.44 / 103.53 us | 10% sum | Beterov Table VII | VERIFIED |
| B6 | Rb 50P3/2 tau_0 / tau_eff(300 K) | 239.23 / 84.74 us | 10% sum | Beterov Table VII | VERIFIED |
| B7 | Rb 50D5/2 tau_0 / tau_eff(300 K) | 118.21 / 65.35 us | 10% sum | Beterov Table VII | VERIFIED |
| B8 | Cs 50S1/2 tau_0 / tau_eff(300 K) | 125.64 / 60.41 us | 10% sum; analytic path: 125.68 / 59.8 us at 1% | Beterov Table VIII | VERIFIED |
| B8b | Cs 50D5/2 tau_0 / tau_eff(300 K) | 70.12 / 46.61 us | 10% sum | Beterov Table VIII | VERIFIED |
| B9 | Rb 50S BBR rate at 300 K (table-derived) | 8.27e3 s^-1 | 15% sum; fit gives 8.45e3 (2%) | 1/65.176us − 1/141.31us, Beterov | VERIFIED |
| B9b | Farley–Wing ratio, Rb 40–70S, 300 K | Gamma_BBR/Gamma_FW in [0.7, 1.2] | band | Farley & Wing PRA 23, 2397 (1981) | LITERATURE-RECALL |
| B10 | Hot-cell Rydberg EIT linewidth, typical params | 2–10 MHz band; canonical 2 MHz (Mohapatra) | band test | Mohapatra, Jackson, Adams, PRL 98, 113003 (2007) | VERIFIED |
| B11 | Rb D2 self-broadening beta/2pi (formula) | 1.03e-7 Hz cm^3 | 3% | Lewis 1980 / Weller 2011 Table I | VERIFIED |
| B11b | vs measured | 1.10(17)e-7 Hz cm^3 | 1 sigma | Kondo et al. PRA 73, 062504 (2006) | VERIFIED |
| B12 | Transit FWHM, Rb-87, 300 K, w = 0.75 mm | 106 kHz | 20% (convention) | §2.3.3 derivation | derivation (self-validating) |
| B13 | Direct BBR ionization, Rb 50S, 300 K | ~1.5e2 s^-1 | 50% | Beterov NJP Eq. 27 (worked here) | VERIFIED formula, derived value |
| B14 | 6j-reduction identity S1 | exact | 1e-12 | algebra | — |
| B15 | Ladder laser-noise wiring S7 | FWHM = Gamma_e/2pi + Dnu_p (probe line); EIT dip -> 2 gamma_gr | 2% | phase-diffusion model | derivation |

**Verification finding (source-table anomaly):** Beterov Table VII Rb nD3/2
entries at n = 50, 55, 60 (126.53, 168.53, 218.98 us at 0 K) deviate by 5–6%
from both their own Table II fit (119.2, 159.5, 208.4 us) and from the
neighboring nD5/2 column, while n = 40, 45 agree to <0.5%. Benchmarks
therefore use the nD5/2 column for D-state validation and apply a 10%
tolerance to Rb nD3/2 at n >= 50. Do not "fix" the discrepancy silently in
either direction.

---

## 7. Known limitations / model breakdown

1. **Fit ranges.** Beterov fits: 15 <= n <= 80 only. Below n ~ 15 use the
   explicit sum; above n = 80 both quasiclassical fits and our sum are
   unvalidated (and cell physics changes: interactions dominate anyway).
2. **Low-n' matrix elements** limit absolute tau_0 accuracy to a few percent;
   nP states are worst (documented theory/experiment tension in Beterov §
   comparison with de Oliveira et al.).
3. **BBR environment.** All BBR numbers assume an isotropic 300 K (or T_cell)
   Planck field. Real cells see the (usually warmer) oven plus optical
   windows; an effective BBR temperature different from T_cell by tens of K
   changes Gamma_BBR by ~10%. The API takes `T_bbr` separately from `T_cell`.
4. **BBR ionization** implements only the direct channel (factor ~1.3–2 low
   vs total); adequate for budget purposes, not for ion-yield prediction.
5. **Collisions.** Impact-regime, binary-collision formulas only. Self-broadening
   beta is invalid beyond the Weisskopf detuning (~4 GHz); Rydberg–ground
   broadening coefficient is order-of-magnitude (flagged) pending a dedicated
   literature pass; no molecular/ultralong-range resonance effects near
   specific n.
6. **vdW dephasing** treated as quasi-static nearest-neighbor shift; no
   many-body correlations, no blockade dynamics. Above N_r ~ 1e9 cm^-3
   (n ~ 50) the 3-level OBE itself is invalid.
7. **Superradiance/collective decay** among Rydberg levels in dense beams is
   ignored; can shorten effective lifetimes at high N_r in mm-scale samples.
8. **4-level extensions** (RF field on r–r' for electrometry) inherit this
   budget; the r' state needs its own Gamma_0 + Gamma_BBR from the same
   machinery.
9. **Hyperfine structure** of |g> and |e> is compressed into the 3-level model
   here; optical-pumping leak rates are configured in the OBE spec, not this
   document.
10. **Transit model** assumes ballistic crossing (no buffer gas). Any buffer
    gas invalidates both the transit formula and the collision table.

---

## References / sources used in this revision

- I. I. Beterov, I. I. Ryabtsev, D. B. Tretyakov, V. M. Entin, PRA 79, 052504
  (2009); Erratum PRA 80, 059902 (2009). Tables extracted from
  [arXiv:0810.0339v4](https://arxiv.org/abs/0810.0339).
- I. I. Beterov et al., "Ionization of Rydberg atoms by blackbody radiation",
  New J. Phys. 11, 013052 (2009). Extracted from
  [arXiv:0807.2535](https://arxiv.org/abs/0807.2535).
- D. A. Steck, "Rubidium 87 D Line Data", "Rubidium 85 D Line Data",
  "Cesium D Line Data", revision 2.3.4 (8 August 2025),
  [steck.us/alkalidata](https://steck.us/alkalidata/) — all three PDFs
  downloaded and machine-parsed in this revision.
- B. M. Patterson et al., PRA 91, 012506 (2015) — independent Cs 6P3/2
  lifetime 30.462(46) ns [LITERATURE-RECALL; cited for the tension note only].
- L. Weller, R. J. Bettles, P. Siddons, C. S. Adams, I. G. Hughes,
  J. Phys. B 44, 195006 (2011). Extracted from
  [arXiv:1107.3092](https://arxiv.org/abs/1107.3092) (Table I: self-broadening,
  incl. Kondo et al. PRA 73, 062504 (2006) Rb D2; Akulshin et al. JETP Lett.
  36, 303 (1982) Cs D2; Lewis, Phys. Rep. 58, 1 (1980) theory).
- A. K. Mohapatra, T. R. Jackson, C. S. Adams, PRL 98, 113003 (2007)
  ([arXiv:quant-ph/0612200](https://arxiv.org/abs/quant-ph/0612200)) — 2 MHz
  hot-cell Rydberg EIT linewidth.
- J. W. Farley, W. H. Wing, PRA 23, 2397 (1981) — BBR limit formula
  [LITERATURE-RECALL].
- M. Fleischhauer, A. Imamoglu, J. P. Marangos, Rev. Mod. Phys. 77, 633 (2005)
  — EIT window width [LITERATURE-RECALL].
- J. Gea-Banacloche, Y. Li, S. Jin, M. Xiao, PRA 51, 576 (1995) — Doppler
  mismatch in ladder EIT [LITERATURE-RECALL].
- K. Singer, J. Stanojevic, M. Weidemüller, R. Côté, J. Phys. B 38, S295
  (2005) — C6 fits [LITERATURE-RECALL].
- Li, Mourachko, Han, Gallagher, PRA 67, 052502 (2003); Mack et al., PRA 83,
  052515 (2011); Goy et al., PRA 26, 2733 (1982); Weber & Sansonetti, PRA 35,
  4650 (1987) — quantum defects [LITERATURE-RECALL].
