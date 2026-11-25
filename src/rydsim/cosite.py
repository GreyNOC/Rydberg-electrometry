"""Co-site interference: hidden weak signal under a strong interferer (T3).

Campaign T3 (docs/MISSION.md) asks how a superhet Rydberg receiver behaves
when a strong co-channel interferer shares the band with the signal of
interest. This module answers the narrow, well-posed version of that
question: how much does a strong interferer at offset Delta_I from the LO
reduce the receiver's small-signal gain for a weak signal at offset
delta_s, and what is the weakest signal still detectable beside it?

Physics (quasi-static, spec 08 §2.1 extended to two tones)
----------------------------------------------------------
All three fields drive the same Rydberg RF transition. In the LO frame the
atoms see the envelope of the phasor sum

    Z(t) = E_L + E_I e^{i theta} + E_s e^{i psi},  theta = Delta_I t, psi = delta_s t

and, for offsets well inside the atomic response bandwidth, the transmitted
probe power follows the static transfer curve: P(t) = T(|Z(t)|). This is
the assumption `superhet.compression_point` already makes; it is enforced
here by `require_quasi_static`.

To first order in E_s, with A = |E_L + E_I e^{i theta}| and
phi = arg(E_L + E_I e^{i theta}),

    |Z| = A + E_s cos(psi - phi) + O(E_s^2)
    P   = T(A) + T'(A) E_s cos(psi - phi) + O(E_s^2)

When delta_s is not an intermodulation product of Delta_I, theta and psi are
independent phases, so the demodulated amplitude at delta_s is E_s times

    G(E_I) = | <T'(A(theta)) e^{-i phi(theta)}>_theta |

G(0) = |T'(E_L)| recovers the single-tone transduction slope. The ratio
G(E_I)/G(0) is the desensitization (blocking) of the receiver.

The interferer also writes its own deterministic tones into the probe at
n * Delta_I (the Fourier harmonics of T(A(theta))). A hidden signal placed on
one of those spurs is masked; their field-equivalent level is reported.

Cross-check: `two_tone_fft_gain` re-derives G by brute force — a finite-E_s,
commensurate two-tone waveform pushed through the transfer curve and
demodulated by DFT — without using the phase-average identity above.

Validity limits (refused or reported, never silently exceeded):
  * offsets must sit inside the atomic response bandwidth (quasi-static);
    beyond it the atoms low-pass the interferer and a Floquet / time-domain
    OBE treatment is required (not implemented — refused);
  * the envelope range [|E_L - E_I|, E_L + E_I] must lie inside the sampled
    transfer curve (no extrapolation);
  * the theta-quadrature must converge (grid doubling), else IntegrityError.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np

from .provenance import IntegrityError
from .superhet import NoiseBudget, rin_psd, shot_noise_psd

if TYPE_CHECKING:
    from .experiment import LadderConfig

Transfer = Callable[[np.ndarray], np.ndarray]

#: Largest offset, as a fraction of the 3 dB instantaneous bandwidth, at which
#: the quasi-static transfer curve is accepted. For a single-pole response
#: |H(0.1 f_3dB)|/|H(0)| = 0.995: a 0.04 dB gain error.
QUASI_STATIC_MAX_FRACTION = 0.1


def require_quasi_static(offsets_hz: Sequence[float], ibw_hz: float,
                         max_fraction: float = QUASI_STATIC_MAX_FRACTION
                         ) -> None:
    """Refuse offsets the static transfer-curve model cannot represent."""
    if not np.isfinite(ibw_hz) or ibw_hz <= 0:
        raise IntegrityError(f"instantaneous bandwidth {ibw_hz!r} Hz is not a "
                             "usable validity bound")
    limit = max_fraction * ibw_hz
    bad = [f for f in offsets_hz if abs(f) > limit]
    if bad:
        raise IntegrityError(
            f"offset(s) {', '.join(f'{f:.4g}' for f in bad)} Hz exceed "
            f"{max_fraction:g} x IBW = {limit:.4g} Hz: the atoms no longer "
            "follow the interferer envelope quasi-statically, so the static "
            "transfer-curve model is invalid there. A time-domain / Floquet "
            "OBE solver is required for out-of-band interferers.")


def instantaneous_bandwidth_hz(cfg: "LadderConfig", e_lo: float,
                               e_to_omega: float) -> float:
    """3 dB atomic response bandwidth [Hz] at the LO operating point.

    OBE resolvent H(delta) of the 4-level ladder at Omega_RF = E_LO d/hbar,
    the same construction objective.evaluate() uses. Doppler-free: thermal
    averaging is not applied to the resolvent, so for hot vapor this is the
    single-velocity-class bandwidth (stated as a caveat on every finding).
    """
    from .lindblad import LadderSystem, ibw_3db

    sys_ = LadderSystem(
        omegas=[cfg.omega_probe, cfg.omega_coupling, e_lo * e_to_omega],
        deltas=[cfg.delta_probe, cfg.delta_coupling, cfg.delta_rf],
        decays=[0.0, cfg.gamma_e, cfg.gamma_r, cfg.gamma_rp],
        dephasings=[0.0, 0.0, cfg.deph_r, cfg.deph_r],
        transit=cfg.transit,
    )
    tau = 2 * np.pi
    return float(ibw_3db(sys_, 3, delta_max=tau * 100e6) / tau)


def require_no_spur_collision(signal_offset_hz: float,
                              interferer_offset_hz: float,
                              n_harmonics: int, rel_tol: float = 1e-3
                              ) -> None:
    """Refuse a signal offset that lands on an interferer harmonic.

    The gain formula assumes the signal and interferer phases are
    independent; at delta_s = n Delta_I they are not, and the hidden signal
    adds coherently to the interferer's own spur.
    """
    if signal_offset_hz == 0 or interferer_offset_hz == 0:
        raise ValueError("offsets must be non-zero (zero is the LO itself)")
    for n in range(1, n_harmonics + 1):
        spur = n * abs(interferer_offset_hz)
        if abs(abs(signal_offset_hz) - spur) <= rel_tol * spur:
            raise IntegrityError(
                f"signal offset {signal_offset_hz:.6g} Hz coincides with "
                f"interferer harmonic {n} ({spur:.6g} Hz): the phase-averaged "
                "gain does not apply; the hidden signal is masked by a "
                "deterministic spur (see interferer_spurs).")


def _envelope(e_lo: float, e_i: float, theta: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray]:
    z = e_lo + e_i * np.exp(1j * theta)
    return np.abs(z), np.exp(-1j * np.angle(z))


def _check_domain(e_lo: float, e_i: float, domain: tuple[float, float]) -> None:
    if e_lo <= 0 or e_i < 0:
        raise ValueError("require E_LO > 0 and E_I >= 0")
    lo, hi = abs(e_lo - e_i), e_lo + e_i
    if lo < domain[0] or hi > domain[1]:
        raise IntegrityError(
            f"envelope range [{lo:.4g}, {hi:.4g}] V/m leaves the sampled "
            f"transfer curve [{domain[0]:.4g}, {domain[1]:.4g}] V/m; "
            "extrapolating the atomic response is refused — widen the "
            "transfer-curve grid.")


@dataclass(frozen=True)
class TwoToneGain:
    """Small-signal gain for the hidden signal beside one interferer level."""

    gain: float          # G(E_I) = |d P_hat(delta_s) / d E_s|   [W/(V/m)]
    mean_power: float    # time-averaged probe power under the interferer [W]
    nodes: int           # theta-quadrature nodes at convergence


def small_signal_gain(transfer: Transfer, dtransfer: Transfer,
                      e_lo: float, e_i: float,
                      domain: tuple[float, float],
                      rtol: float = 1e-9, n_start: int = 64,
                      n_max: int = 1 << 16) -> TwoToneGain:
    """Phase-averaged hidden-signal gain G(E_I), with converged quadrature.

    The integrand is smooth and 2*pi-periodic, so the trapezoid rule
    converges geometrically; convergence is proven by node doubling and an
    unconverged result raises IntegrityError instead of being returned.
    """
    _check_domain(e_lo, e_i, domain)
    if e_i == 0.0:
        return TwoToneGain(abs(float(dtransfer(np.array([e_lo]))[0])),
                           float(transfer(np.array([e_lo]))[0]), 1)

    def evaluate(n: int) -> tuple[float, float]:
        theta = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
        a, rot = _envelope(e_lo, e_i, theta)
        g = abs(np.mean(dtransfer(a) * rot))
        return float(g), float(np.mean(transfer(a)))

    n = n_start
    g_prev, p_prev = evaluate(n)
    while n < n_max:
        n *= 2
        g, p = evaluate(n)
        scale = max(abs(g), abs(g_prev), 1e-300)
        if abs(g - g_prev) <= rtol * scale and abs(p - p_prev) <= rtol * abs(p):
            return TwoToneGain(g, p, n)
        g_prev, p_prev = g, p
    raise IntegrityError(
        f"theta quadrature did not converge to {rtol:g} by {n_max} nodes "
        f"(E_LO={e_lo:.4g}, E_I={e_i:.4g} V/m); the envelope likely passes "
        "too close to zero field for the sampled transfer curve.")


def interferer_spurs(transfer: Transfer, e_lo: float, e_i: float,
                     domain: tuple[float, float], n_harmonics: int = 5,
                     n_nodes: int = 4096) -> np.ndarray:
    """Probe-power tone amplitudes [W] the interferer writes at n*Delta_I.

    Element n-1 is the one-sided amplitude 2|<T(A) e^{-i n theta}>| of
    harmonic n = 1..n_harmonics. A hidden signal at one of these offsets
    competes with a deterministic tone of this size.
    """
    _check_domain(e_lo, e_i, domain)
    theta = np.linspace(0.0, 2 * np.pi, n_nodes, endpoint=False)
    p = transfer(_envelope(e_lo, e_i, theta)[0])
    k = np.arange(1, n_harmonics + 1)
    return 2.0 * np.abs(np.exp(-1j * np.outer(k, theta)) @ p) / n_nodes


def two_tone_fft_gain(transfer: Transfer, e_lo: float, e_i: float,
                      e_s: float, domain: tuple[float, float],
                      m_i: int = 7, m_s: int = 3,
                      n_samples: int = 8192) -> float:
    """Independent brute-force G: demodulate a commensurate two-tone waveform.

    Interferer and signal complete m_i and m_s cycles in one common period.
    m_i = 7, m_s = 3 keeps every intermodulation product that could land on
    m_s at order E_s^5 or higher, below the finite-E_s error budget.
    """
    if m_s % m_i == 0 or np.gcd(m_i, m_s) != 1:
        raise ValueError("m_i and m_s must be coprime and distinct")
    _check_domain(e_lo, e_i + e_s, domain)
    x = np.linspace(0.0, 2 * np.pi, n_samples, endpoint=False)
    z = e_lo + e_i * np.exp(1j * m_i * x) + e_s * np.exp(1j * m_s * x)
    p = transfer(np.abs(z))
    return float(2.0 * abs(np.mean(p * np.exp(-1j * m_s * x))) / e_s)


@dataclass(frozen=True)
class CositePoint:
    """Receiver state for the hidden signal at one interferer level."""

    e_i: float                 # interferer amplitude [V/m]
    gain: float                # G(E_I) [W/(V/m)]
    desense_db: float          # 20 log10(G(0)/G(E_I)); >0 = sensitivity lost
    mean_power: float          # [W]
    budget: NoiseBudget
    nef: float                 # hidden-signal NEF under the interferer [(V/m)/rtHz]
    e_min: float               # min detectable hidden field, given integration [V/m]
    hidden_below_interferer_db: float   # 20 log10(E_I / e_min); inf at E_I = 0


def cosite_scan(transfer: Transfer, dtransfer: Transfer, e_lo: float,
                e_i_grid: Sequence[float], domain: tuple[float, float],
                wavelength: float, rin_db_per_hz: float = -150.0,
                detector_nep_w_per_rthz: float = 1e-12,
                integration_time: float = 1.0, snr: float = 1.0
                ) -> list[CositePoint]:
    """Desensitization and hidden-signal floor across interferer levels.

    The noise budget is re-evaluated at the interferer-shifted mean probe
    power: shot and RIN noise move with it, so blocking is not gain alone.
    """
    if integration_time <= 0 or snr <= 0:
        raise ValueError("integration_time and snr must be positive")
    g0 = small_signal_gain(transfer, dtransfer, e_lo, 0.0, domain).gain
    if g0 <= 0:
        raise IntegrityError("zero transduction slope at the LO: no superhet "
                             "gain to desensitize")
    out: list[CositePoint] = []
    for e_i in e_i_grid:
        tt = small_signal_gain(transfer, dtransfer, e_lo, float(e_i), domain)
        budget = NoiseBudget(
            shot=shot_noise_psd(tt.mean_power, wavelength),
            rin=rin_psd(tt.mean_power, rin_db_per_hz),
            detector=detector_nep_w_per_rthz**2,
        )
        if tt.gain <= 0:
            nef = e_min = float("inf")
        else:
            nef = float(np.sqrt(budget.total) / tt.gain)
            e_min = snr * nef / np.sqrt(integration_time)
        below = (float("inf") if e_i == 0 else
                 float(20 * np.log10(float(e_i) / e_min)))
        desense = (float("inf") if tt.gain <= 0 else
                   float(20 * np.log10(g0 / tt.gain)))
        out.append(CositePoint(float(e_i), tt.gain, desense, tt.mean_power,
                               budget, nef, e_min, below))
    return out


def small_interferer_law(points: Sequence[CositePoint], e_lo: float,
                         u_max: float = 0.05, min_db: float = 1e-6
                         ) -> tuple[float, float] | None:
    """Fit desense_dB = k * (E_I/E_LO)^p over the weak-interferer points.

    Returns (p, k) from a log-log least-squares fit using points with
    E_I/E_LO <= u_max and desense above `min_db` (below that the gain ratio
    is numerically indistinguishable from 1); None with fewer than 3 points.
    A smooth transfer curve predicts p = 2: the first-order term in E_I
    averages to zero over the interferer phase.
    """
    u = np.array([p.e_i / e_lo for p in points])
    d = np.array([p.desense_db for p in points])
    keep = (u > 0) & (u <= u_max) & (d > min_db) & np.isfinite(d)
    if keep.sum() < 3:
        return None
    p, logk = np.polyfit(np.log(u[keep]), np.log(d[keep]), 1)
    return float(p), float(np.exp(logk))


def blocking_level(points: Sequence[CositePoint],
                   threshold_db: float = 1.0) -> float | None:
    """Interferer amplitude [V/m] at which desensitization first reaches
    threshold_db, log-interpolated between scan points; None if not reached.

    Requires a strictly increasing E_I scan; a crossing bracketed by a
    non-monotone segment is still interpolated on that segment only.
    """
    es = np.array([p.e_i for p in points])
    if np.any(np.diff(es) <= 0):
        raise ValueError("interferer grid must be strictly increasing")
    ds = np.array([p.desense_db for p in points])
    idx = np.where(ds >= threshold_db)[0]
    if len(idx) == 0:
        return None
    k = int(idx[0])
    if k == 0:
        return float(es[0])
    e0, e1, d0, d1 = es[k - 1], es[k], ds[k - 1], ds[k]
    if e0 <= 0:
        return float(e1)
    frac = (threshold_db - d0) / (d1 - d0)
    return float(np.exp(np.log(e0) + frac * (np.log(e1) - np.log(e0))))
