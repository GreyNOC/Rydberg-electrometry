"""Linear Zeeman physics of fine-structure Rydberg states.

Spec: docs/spec/00-conventions.md ruling R-17 (this module is the owner of
Zeeman shifts per the §4 ownership map; symbol-table row g_J, mu_B).
Serves spec 09 benchmarks C9 and the E7 fixtures (Comms. Phys. 2026
Zeeman-tuned continuous-coverage mechanism).

Physics (all NORMATIVE per R-17)
--------------------------------
* Lande factor, LS coupling with g_L = 1 and g_S = 2 exactly:
      g_J = 3/2 + [s(s+1) - l(l+1)] / (2 j (j+1)),   s = 1/2.
  (The ~0.12% QED correction to g_S is deliberately excluded by the ruling;
  it is far inside benchmark C9's 1% tolerance.)
* Single-state linear Zeeman shift  Delta_f = (mu_B B / h) g_J m_J  [Hz].
* Transition shift (unprimed -> primed, C9's rung-3 -> rung-4):
      Delta_f = (mu_B B / h) (g_J' m_J' - g_J m_J)   [Hz]
* Stretched-state tuning law: d f / d B = (mu_B / h)(g_J' j' - g_J j) etc.,
  with m_J = sign * j in both states (the Comms. Phys. 2026 RF-resonance
  tuning mechanism; check value mu_B/h = 1.399625 MHz/G).

Units
-----
SI internal (lock #1): B in tesla, shifts in Hz. Hz (not rad/s) is correct
here because R-17 defines the observable Delta_f at the API boundary; every
function carries the _hz / _hz_per_t suffix (lock #2). Gauss and MHz/G are
display units, available only through the explicit I/O helpers below.

Scope guard (NORMATIVE, R-17)
-----------------------------
Quadratic Zeeman (diamagnetic + second-order j-mixing), combined E x B
fields, and hyperfine Zeeman structure (g_F) are OUT OF SCOPE as *modelled
physics*: this module never returns them. They are IN SCOPE as *validity
conditions* — R-17's fence exists to stop the linear law being read as the
whole shift, and a fence is only worth having if it tests the term that
actually breaks first.

Two independent breakdown channels, both armable, both opt-in (the caller
supplies the datum this module refuses to guess):

1. **j-mixing** — the Zeeman energy approaching the fine-structure interval
   destroys j as a good quantum number. Fenced by
   :func:`require_linear_regime` against a caller-supplied interval from
   rydsim.atom (spec 01): |shift| > LINEAR_FENCE_FRACTION (5%) of it raises
   rydsim.provenance.IntegrityError.  This channel *cannot be armed at all
   for l = 0*, which has no same-l j-partner.

2. **The diamagnetic term** — ``(e^2 B^2 / 8 m_e) <rho^2>``, the leading
   NEGLECTED contribution to the shift this module returns. It is the binding
   condition for Rydberg states and for l = 0: it grows as ``n*^4 B^2`` while
   the linear term is n-independent and the fine-structure interval shrinks
   as ``n*^-3``. Worked, reproducible example over the spec 09 E7 field range
   (0-412 G; all four numbers regression-tested):

     - Cs-like nD5/2 stretched, n* = 42.5 — neglected/returned = 1.5 % at
       60 G, crosses the 5 % tolerance at 202 G, 10.2 % at 412 G;
     - Cs-like nS1/2, n* = 44.9 — 4.3 % at 60 G, 29.7 % at 412 G, and this
       is the l = 0 state for which channel 1 cannot be armed at all.

   Fenced by :func:`require_linear_dominates` / the ``n_star`` argument of
   :func:`state_shift_hz`: |diamagnetic| > 5 % of |linear| raises.

Passing neither datum leaves validity with the caller (documented, unchanged
behaviour). Passing only the fine-structure interval is NOT sufficient for a
Rydberg state — channel 2 is the one that breaks first.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import A0, E_CHARGE, H, M_E, MU_B
from .provenance import IntegrityError
from .wigner import clebsch_gordan

# mu_B / h [Hz/T], computed from scipy CODATA (lock #15 - never typed by hand).
# Check value (R-17, VERIFIED arithmetic): 1.399625 MHz/G = 1.3996e10 Hz/T.
MU_B_OVER_H_HZ_PER_T = MU_B / H

# Exact SI definition: 1 G = 1e-4 T. Display-unit conversion only (lock #1).
GAUSS_TO_TESLA = 1e-4

# Validity fence of the linear law (R-17): |shift| <= 5% of the FS interval.
LINEAR_FENCE_FRACTION = 0.05

# Same 5 % tolerance applied to the term that actually breaks the linear law
# for Rydberg states: the neglected diamagnetic shift vs the returned linear
# shift (module docstring, channel 2).
DIAMAGNETIC_FENCE_FRACTION = LINEAR_FENCE_FRACTION

# Diamagnetic prefactor e^2 a0^2 / (8 m_e h) [Hz / (T^2 a0^2)] — assembled
# from scipy CODATA (lock #15, never typed by hand). Multiplying by
# <rho^2>/a0^2 and B^2 [T^2] gives the shift in Hz. Cross-checked in
# tests/test_zeeman.py against the atomic-unit route (1/8) (B/B_au)^2 <rho^2>
# Hartree with B_au = scipy's "atomic unit of mag. flux density".
DIAMAGNETIC_HZ_PER_T2_A0SQ = E_CHARGE**2 * A0**2 / (8.0 * M_E * H)


def _half_int(x: float, name: str) -> int:
    """Return round(2x) after checking x is integer or half-integer.

    No rounding of physics inputs (audit refusal rule 12): a value that is
    not (half-)integer to 1e-9 raises ValueError.
    """
    tx = round(2 * x)
    if abs(2 * x - tx) > 1e-9:
        raise ValueError(f"{name}={x} is not integer or half-integer")
    return tx


def lande_g_j(l: int, j: float, s: float = 0.5) -> float:
    """Lande g-factor g_J = 3/2 + [s(s+1) - l(l+1)] / (2 j (j+1)).

    NORMATIVE form per ruling R-17 (00-conventions §5); the standard
    LS-coupling result with g_L = 1, g_S = 2 exactly (Sobelman, *Atomic
    Spectra and Radiative Transitions*, §8 — confidence VERIFIED: pure
    algebra, cross-derived in tests/test_zeeman.py from the projection
    theorem with exact rationals). Dimensionless.

    l : orbital angular momentum (non-negative integer).
    j : total electronic angular momentum; must satisfy the (l, s) triangle
        |l - s| <= j <= l + s in integer steps (for s = 1/2: j = l +- 1/2).
    s : spin, default 1/2 (alkali single valence electron; R-17 fixes this).
    """
    tl = _half_int(l, "l")
    if tl % 2 or tl < 0:
        raise ValueError(f"l={l} must be a non-negative integer")
    tj = _half_int(j, "j")
    ts = _half_int(s, "s")
    if not (abs(tl - ts) <= tj <= tl + ts) or (tl + ts - tj) % 2:
        raise ValueError(f"(l={l}, s={s}, j={j}) violates the triangle rule")
    if tj == 0:
        raise ValueError(f"g_J undefined for j=0 (l={l}, s={s}); shift is trivially 0")
    return 1.5 + (s * (s + 1.0) - l * (l + 1.0)) / (2.0 * j * (j + 1.0))


@dataclass(frozen=True)
class ZeemanState:
    """One fine-structure |l, j, m_J> state (n-independent for linear Zeeman).

    Validated on construction: l non-negative integer, j = l +- 1/2
    (s = 1/2 alkali), m_J half-integer with |m_J| <= j and the same
    half-integer character as j. Spec: ruling R-17.
    """

    l: int
    j: float
    m_j: float

    def __post_init__(self) -> None:
        lande_g_j(self.l, self.j)  # validates (l, j) against s = 1/2
        tj = _half_int(self.j, "j")
        tm = _half_int(self.m_j, "m_j")
        if abs(tm) > tj or (tj - tm) % 2:
            raise ValueError(
                f"m_j={self.m_j} invalid for j={self.j} (|m_j| <= j, same parity)"
            )

    @property
    def g_j(self) -> float:
        """Lande g-factor of this state (dimensionless, R-17 formula)."""
        return lande_g_j(self.l, self.j)


def mean_cos2_theta(l: int, m_l: int) -> float:
    """<l m_l| cos^2(theta) |l m_l> = [2l(l+1) - 2 m_l^2 - 1]/[(2l-1)(2l+3)].

    Standard spherical-harmonic result (equivalently the k = 2 Legendre
    projection of |Y_lm|^2). Dimensionless. VERIFIED by direct numerical
    integration of |Y_lm|^2 cos^2(theta) in tests/test_zeeman.py, plus the
    exact sum rule sum_m <cos^2> = (2l+1)/3. Correct at l = 0 (returns 1/3:
    numerator -1, denominator -3).
    """
    tl, tm = _half_int(l, "l"), _half_int(m_l, "m_l")
    if tl % 2 or tl < 0:
        raise ValueError(f"l={l} must be a non-negative integer")
    if tm % 2 or abs(tm) > tl:
        raise ValueError(f"m_l={m_l} must be an integer with |m_l| <= l={l}")
    return (2.0 * l * (l + 1.0) - 2.0 * m_l * m_l - 1.0) / (
        (2.0 * l - 1.0) * (2.0 * l + 3.0)
    )


def mean_sin2_theta(l: int, j: float, m_j: float) -> float:
    """<l j m_j| sin^2(theta) |l j m_j> in the fine-structure basis.

    sin^2(theta) is diagonal in m_l within a fixed l, so the fine-structure
    value is the Clebsch-Gordan-weighted average over the two |m_l, m_s>
    components of |l, j = l +- 1/2, m_j>:
        sum_{m_s} |<l m_j-m_s; 1/2 m_s | j m_j>|^2 (1 - <cos^2>_{l, m_j-m_s}).
    Exact (no approximation); dimensionless, in [2/(2l+3), 1]. l = 0 gives
    exactly 2/3. Uses rydsim.wigner (spec 00 lock #9: no re-implemented
    angular algebra).
    """
    lande_g_j(l, j)  # validates (l, j) for s = 1/2
    tj, tm = _half_int(j, "j"), _half_int(m_j, "m_j")
    if abs(tm) > tj or (tj - tm) % 2:
        raise ValueError(f"m_j={m_j} invalid for j={j}")
    total = 0.0
    weight = 0.0
    for tms in (1, -1):
        m_s = tms / 2.0
        m_l = m_j - m_s
        if abs(m_l) > l + 1e-9:
            continue
        c = clebsch_gordan(l, m_l, 0.5, m_s, j, m_j)
        weight += c * c
        total += c * c * (1.0 - mean_cos2_theta(l, int(round(m_l))))
    if abs(weight - 1.0) > 1e-9:  # pragma: no cover - guards a wiring error
        raise IntegrityError(
            f"|l={l}, j={j}, m_j={m_j}> decomposition is not normalised "
            f"(sum |CG|^2 = {weight!r}); refusing to average sin^2(theta) "
            "over an incomplete basis"
        )
    return total


def mean_rho2_a0sq(n_star: float, l: int, j: float, m_j: float) -> float:
    """<rho^2> = <r^2 sin^2(theta)> for |n* l j m_j>, in units of a0^2.

    Radial part is the hydrogenic expectation value with the effective
    principal quantum number (Bethe-Salpeter SS3 / Gallagher, *Rydberg Atoms*
    Table 2.2; the n -> n* substitution is the standard quantum-defect scaling
    for alkali Rydberg states):
        <r^2> = (n*^2 / 2) [5 n*^2 + 1 - 3 l(l+1)]  a0^2
    (asymptotically (5/2) n*^4 a0^2, the familiar Rydberg scaling).
    Confidence LITERATURE-RECALL for the n* substitution, VERIFIED for the
    hydrogenic formula (tests/test_zeeman.py integrates the exact hydrogen
    radial functions and reproduces it to 1e-9 at integer n).
    Angular part is :func:`mean_sin2_theta`, which is exact.

    Deliberately a closed form rather than a call into rydsim.radial: this is
    a *fence* estimate whose job is to be available for every (n*, l) without
    dragging the Numerov consensus machinery (and its n-floor refusals) into a
    validity check. It is accurate to the quantum-defect level, which is far
    inside the 5 % tolerance it feeds. A caller wanting the fence at radial's
    fidelity can compute <r^2 sin^2> from rydsim.radial and call
    :func:`diamagnetic_shift_hz`'s pieces directly.

    Refuses (IntegrityError) n* <= l: there is no bound orbital there and the
    formula would return a number for a state that does not exist.
    """
    ns = float(n_star)
    if not np.isfinite(ns) or ns <= l:
        raise IntegrityError(
            f"effective principal quantum number n*={n_star!r} is not a finite "
            f"value above l={l}; refusing to evaluate <r^2> for a state with "
            "no bound orbital (supply n* from rydsim.atom.n_star)"
        )
    r2 = 0.5 * ns**2 * (5.0 * ns * ns + 1.0 - 3.0 * l * (l + 1.0))
    return r2 * mean_sin2_theta(l, j, m_j)


def diamagnetic_shift_hz(
    n_star: float, l: int, j: float, m_j: float,
    b_field_t: float | np.ndarray,
) -> float | np.ndarray:
    """Leading NEGLECTED (diamagnetic) shift of |n* l j m_j> [Hz].

    ``E_dia/h = e^2 B^2 <rho^2> / (8 m_e h)`` — the term R-17 puts out of
    scope for *modelling* and this module therefore must fence against. Not a
    physics product of rydsim.zeeman: it exists so callers (and
    :func:`require_linear_dominates`) can see how much of the true shift the
    linear law is throwing away. Always positive (rho^2 >= 0) and even in B,
    unlike the linear term. Vectorized over b_field_t.
    """
    b = np.asarray(b_field_t, dtype=float)
    out = DIAMAGNETIC_HZ_PER_T2_A0SQ * mean_rho2_a0sq(n_star, l, j, m_j) * b * b
    return float(out) if np.ndim(b_field_t) == 0 else out


def require_linear_dominates(
    state: "ZeemanState",
    n_star: float,
    b_field_t: float | np.ndarray,
    max_fraction: float = DIAMAGNETIC_FENCE_FRACTION,
) -> None:
    """Fence the linear law against the term that actually breaks it first.

    Raises IntegrityError when the neglected diamagnetic shift
    (:func:`diamagnetic_shift_hz`) exceeds ``max_fraction`` (default 5%) of
    the linear shift this module returns, at any supplied B. This is the
    binding condition for Rydberg states — it scales as n*^4 B^2 against the
    linear term's n-independent B — and unlike the fine-structure fence it is
    armable for l = 0. At B = 0 both terms vanish and the check passes, which
    is correct: the linear law is exact there.
    """
    b = np.atleast_1d(np.asarray(b_field_t, dtype=float)).reshape(-1)
    lin = np.abs(MU_B_OVER_H_HZ_PER_T * state.g_j * state.m_j * b)
    dia = np.abs(
        np.atleast_1d(
            diamagnetic_shift_hz(n_star, state.l, state.j, state.m_j, b)
        ).reshape(-1)
    )
    bad = dia > max_fraction * lin
    if np.any(bad):
        k = int(np.argmax(np.where(bad, dia - max_fraction * lin, -np.inf)))
        b_bad, d_bad, l_bad = float(b[k]), float(dia[k]), float(lin[k])
        ratio = d_bad / l_bad if l_bad > 0.0 else float("inf")
        raise IntegrityError(
            f"linear Zeeman validity fence (R-17, diamagnetic channel): at "
            f"B = {b_bad:.6g} T the NEGLECTED diamagnetic shift "
            f"{d_bad:.6g} Hz is {ratio:.1%} of the linear shift "
            f"{l_bad:.6g} Hz returned for (n*={n_star:g}, l={state.l}, "
            f"j={state.j}, m_j={state.m_j}), above the {max_fraction:.0%} "
            "tolerance. Quadratic Zeeman is out of scope for rydsim.zeeman - "
            "refusing to present the linear term as the shift."
        )


def require_linear_regime(
    shift_hz: float | np.ndarray,
    fs_interval_hz: float,
    max_fraction: float = LINEAR_FENCE_FRACTION,
) -> None:
    """j-mixing fence of the linear Zeeman law (R-17 scope guard, channel 1).

    Raises IntegrityError when max|shift_hz| > max_fraction * fs_interval_hz
    (default 5%), i.e. when the Zeeman energy starts to mix fine-structure
    j-levels and j stops being a good quantum number. Also refuses
    (IntegrityError) a non-finite or non-positive fs_interval_hz — the caller
    must supply a real interval from rydsim.atom, never a guess. Units: Hz
    throughout.

    NOT sufficient on its own for a Rydberg state, and not the binding
    condition: the neglected diamagnetic term (channel 2,
    :func:`require_linear_dominates`) grows as n*^4 B^2 while this interval
    shrinks as n*^-3, so it overtakes the linear shift long before j-mixing
    does. Arm both. This channel is also unarmable for l = 0, which has no
    same-l j-partner.
    """
    fs = float(fs_interval_hz)
    if not np.isfinite(fs) or fs <= 0.0:
        raise IntegrityError(
            f"fine-structure interval {fs_interval_hz!r} Hz is not a finite "
            "positive number - refusing the linear-Zeeman validity check "
            "rather than guessing (R-17 fence, integrity-audit refusal policy)"
        )
    worst = float(np.max(np.abs(shift_hz)))
    if worst > max_fraction * fs:
        raise IntegrityError(
            f"linear Zeeman validity fence (R-17): |shift| = {worst:.6g} Hz "
            f"exceeds {max_fraction:.0%} of the fine-structure interval "
            f"{fs:.6g} Hz (ratio {worst / fs:.2%}). Quadratic Zeeman / "
            "j-mixing / E x B are out of scope for rydsim.zeeman - refusing "
            "to extrapolate the linear law."
        )


def state_shift_hz(
    state: ZeemanState,
    b_field_t: float | np.ndarray,
    fs_interval_hz: float | None = None,
    n_star: float | None = None,
) -> float | np.ndarray:
    """Linear Zeeman shift of one state: Delta_f = (mu_B B / h) g_J m_J [Hz].

    Spec: ruling R-17 (00-conventions §5). B in tesla (signed projection on
    the quantization axis; the shift is odd in both B and m_J). Vectorized
    over b_field_t.

    Validity fences, both optional and independent (module docstring):

    * ``fs_interval_hz`` [Hz], from rydsim.atom - the interval to the
      j-partner of the same l - arms the j-mixing fence
      (:func:`require_linear_regime`). Unarmable for l = 0.
    * ``n_star``, from rydsim.atom.n_star, arms the diamagnetic fence
      (:func:`require_linear_dominates`): the leading NEGLECTED term must stay
      under 5 % of the returned shift. This is the binding condition for
      Rydberg states and works for every l including 0.

    Both raise IntegrityError rather than return an extrapolated number. With
    neither supplied the caller owns validity, and for a Rydberg state
    ``fs_interval_hz`` alone is not enough - the diamagnetic channel breaks
    first.
    """
    b = np.asarray(b_field_t, dtype=float)
    shift = MU_B_OVER_H_HZ_PER_T * state.g_j * state.m_j * b
    if fs_interval_hz is not None:
        require_linear_regime(shift, fs_interval_hz)
    if n_star is not None:
        require_linear_dominates(state, n_star, b)
    return float(shift) if np.ndim(b_field_t) == 0 else shift


def _check_e1_selection(state_from: ZeemanState, state_to: ZeemanState) -> None:
    """Electric-dipole selection rules for the RF-tuned transition.

    |Delta l| = 1, |Delta j| <= 1, |Delta m_J| <= 1 (q = Delta m_J per the
    spec 03 polarization convention). Violations raise ValueError - a
    Zeeman tuning rate for a dipole-forbidden pair is not a meaningful
    RF-resonance quantity.
    """
    if abs(state_to.l - state_from.l) != 1:
        raise ValueError(
            f"E1-forbidden: Delta l = {state_to.l - state_from.l} (need +-1)"
        )
    if round(2 * abs(state_to.j - state_from.j)) > 2:
        raise ValueError(
            f"E1-forbidden: |Delta j| = {abs(state_to.j - state_from.j)} > 1"
        )
    if round(2 * abs(state_to.m_j - state_from.m_j)) > 2:
        raise ValueError(
            f"E1-forbidden: |Delta m_J| = {abs(state_to.m_j - state_from.m_j)} > 1"
        )


def transition_shift_hz(
    state_from: ZeemanState,
    state_to: ZeemanState,
    b_field_t: float | np.ndarray,
    fs_interval_from_hz: float | None = None,
    fs_interval_to_hz: float | None = None,
    n_star_from: float | None = None,
    n_star_to: float | None = None,
) -> float | np.ndarray:
    """Transition Zeeman shift Delta_f = (mu_B B/h)(g_J' m_J' - g_J m_J) [Hz].

    NORMATIVE R-17 form; spec 09 C9 writes it (g_J4 m_J4 - g_J3 m_J3) with
    rung 3 = state_from (the optically prepared Rydberg state) and rung 4 =
    state_to (the RF-coupled state). This is the shift of the *signed*
    frequency (E_to - E_from)/h; for an energetically downward transition
    the RF resonance |E_to - E_from|/h shifts by the negative. B in tesla,
    vectorized. E1 selection rules enforced (ValueError). Each supplied
    fine-structure interval [Hz] arms the j-mixing fence for its state, and
    each supplied n* arms the diamagnetic fence for its state (IntegrityError
    beyond 5% in either channel). Both fences are per-state deliberately: the
    two levels have different l and different n*, so their diamagnetic shifts
    do not cancel in the difference and a small *differential* shift cannot
    launder a large neglected term.
    """
    _check_e1_selection(state_from, state_to)
    up = state_shift_hz(state_to, b_field_t, fs_interval_to_hz, n_star_to)
    lo = state_shift_hz(state_from, b_field_t, fs_interval_from_hz, n_star_from)
    out = np.asarray(up) - np.asarray(lo)
    return float(out) if np.ndim(b_field_t) == 0 else out


def tuning_rate_hz_per_t(state_from: ZeemanState, state_to: ZeemanState) -> float:
    """Zeeman tuning rate d f / d B = (mu_B/h)(g_J' m_J' - g_J m_J) [Hz/T].

    The RF-resonance tuning law of the Comms. Phys. 2026 continuous-coverage
    mechanism (spec 09 §3.5 E7 / benchmark C9; ruling R-17). Exact slope of
    transition_shift_hz in B - linear-regime validity is NOT checked here
    (a rate has no fence by itself); the fence applies where the rate is
    integrated to a shift. Display conversion: hz_per_t_to_mhz_per_gauss.
    """
    _check_e1_selection(state_from, state_to)
    return MU_B_OVER_H_HZ_PER_T * (
        state_to.g_j * state_to.m_j - state_from.g_j * state_from.m_j
    )


def stretched_pair(
    l_from: int, j_from: float, l_to: int, j_to: float, sign: int = +1
) -> tuple[ZeemanState, ZeemanState]:
    """Stretched-m_J state pair (m_J = sign * j in both states).

    The m_J choice of the Zeeman-tuned RF-resonance scheme (spec 09 E7
    fixtures, e.g. Cs 45D5/2 m=5/2 -> 46P3/2 m=3/2): maximally polarized,
    optically pumped states whose transition stays E1-allowed because
    |Delta m_J| = |j' - j| <= 1. sign = +1 or -1 selects the +j or -j
    manifold (the tuning rate is odd under this flip).
    """
    if sign not in (+1, -1):
        raise ValueError(f"sign={sign} must be +1 or -1")
    pair = (
        ZeemanState(l_from, j_from, sign * j_from),
        ZeemanState(l_to, j_to, sign * j_to),
    )
    _check_e1_selection(*pair)
    return pair


def stretched_tuning_rate_hz_per_t(
    l_from: int, j_from: float, l_to: int, j_to: float, sign: int = +1
) -> float:
    """Stretched-state tuning law df/dB = sign*(mu_B/h)(g_J' j' - g_J j) [Hz/T].

    Convenience composition of stretched_pair + tuning_rate_hz_per_t
    (ruling R-17; spec 09 benchmark C9). Example: Cs 45D5/2 -> 46P3/2
    stretched gives (mu_B/h)(4/3 * 3/2 - 6/5 * 5/2) = -(mu_B/h),
    i.e. |df/dB| = 1.399625 MHz/G.
    """
    return tuning_rate_hz_per_t(*stretched_pair(l_from, j_from, l_to, j_to, sign))


def hz_per_t_to_mhz_per_gauss(rate_hz_per_t: float | np.ndarray) -> float | np.ndarray:
    """Display conversion, I/O only (lock #1): Hz/T -> MHz/G.

    1 G = 1e-4 T exactly; mu_B/h converts to the R-17 check value
    1.399625 MHz/G.
    """
    return rate_hz_per_t * GAUSS_TO_TESLA * 1e-6


def gauss_to_tesla(b_gauss: float | np.ndarray) -> float | np.ndarray:
    """Display conversion, I/O only (lock #1): gauss -> tesla (exact 1e-4)."""
    return b_gauss * GAUSS_TO_TESLA
