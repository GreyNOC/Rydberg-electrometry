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
fields, and hyperfine Zeeman structure (g_F) are OUT OF SCOPE. The linear
j-basis law is valid only while the Zeeman energy is small compared to the
fine-structure interval that protects j as a good quantum number: when a
fine-structure interval is supplied, any |shift| exceeding
LINEAR_FENCE_FRACTION (5%) of it raises rydsim.provenance.IntegrityError
rather than returning an extrapolated number (docs/spec/00-integrity-audit.md
refusal policy). Callers obtain the interval from rydsim.atom (spec 01);
this module never guesses one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import H, MU_B
from .provenance import IntegrityError

# mu_B / h [Hz/T], computed from scipy CODATA (lock #15 - never typed by hand).
# Check value (R-17, VERIFIED arithmetic): 1.399625 MHz/G = 1.3996e10 Hz/T.
MU_B_OVER_H_HZ_PER_T = MU_B / H

# Exact SI definition: 1 G = 1e-4 T. Display-unit conversion only (lock #1).
GAUSS_TO_TESLA = 1e-4

# Validity fence of the linear law (R-17): |shift| <= 5% of the FS interval.
LINEAR_FENCE_FRACTION = 0.05


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


def require_linear_regime(
    shift_hz: float | np.ndarray,
    fs_interval_hz: float,
    max_fraction: float = LINEAR_FENCE_FRACTION,
) -> None:
    """Validity fence of the linear Zeeman law (R-17 scope guard).

    Raises IntegrityError when max|shift_hz| > max_fraction * fs_interval_hz
    (default 5%), i.e. when the Zeeman energy starts to mix fine-structure
    j-levels and the quadratic/j-mixing terms this module explicitly does
    NOT model become non-negligible. Also refuses (IntegrityError) a
    non-finite or non-positive fs_interval_hz — the caller must supply a
    real interval from rydsim.atom, never a guess. Units: Hz throughout.
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
) -> float | np.ndarray:
    """Linear Zeeman shift of one state: Delta_f = (mu_B B / h) g_J m_J [Hz].

    Spec: ruling R-17 (00-conventions §5). B in tesla (signed projection on
    the quantization axis; the shift is odd in both B and m_J). Vectorized
    over b_field_t. If fs_interval_hz [Hz] is given (from rydsim.atom, the
    interval to the j-partner of the same l), the R-17 validity fence is
    enforced and IntegrityError raised beyond it; when it is None the caller
    owns the validity check (l = 0 states have no same-l partner - supply
    the relevant perturbing-level spacing instead).
    """
    b = np.asarray(b_field_t, dtype=float)
    shift = MU_B_OVER_H_HZ_PER_T * state.g_j * state.m_j * b
    if fs_interval_hz is not None:
        require_linear_regime(shift, fs_interval_hz)
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
) -> float | np.ndarray:
    """Transition Zeeman shift Delta_f = (mu_B B/h)(g_J' m_J' - g_J m_J) [Hz].

    NORMATIVE R-17 form; spec 09 C9 writes it (g_J4 m_J4 - g_J3 m_J3) with
    rung 3 = state_from (the optically prepared Rydberg state) and rung 4 =
    state_to (the RF-coupled state). This is the shift of the *signed*
    frequency (E_to - E_from)/h; for an energetically downward transition
    the RF resonance |E_to - E_from|/h shifts by the negative. B in tesla,
    vectorized. E1 selection rules enforced (ValueError). Each supplied
    fine-structure interval [Hz] arms the R-17 validity fence for its state
    (IntegrityError beyond 5%).
    """
    _check_e1_selection(state_from, state_to)
    up = state_shift_hz(state_to, b_field_t, fs_interval_to_hz)
    lo = state_shift_hz(state_from, b_field_t, fs_interval_from_hz)
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
