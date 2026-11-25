"""Two-tone Floquet (harmonic-balance) solver for out-of-band interferers.

`rydsim.cosite` treats a co-channel interferer quasi-statically and refuses
offsets beyond 0.1 x the atomic response bandwidth. This module removes
that limit for the single-velocity-class (Doppler-free) ladder: it solves
the OBE exactly under an LO plus an interferer at ANY offset, then the
first-order response to a weak hidden signal on top of that periodic state.

Model
-----
LO frame of the RF link k (levels k-1, k). The complex RF Rabi frequency is

    Omega(t) = Omega_L + Omega_I e^{i Delta t} + Omega_s e^{i delta t}

entering H/hbar as -(1/2)(Omega(t) |k-1><k| + c.c.) — the LadderSystem
convention with a time-dependent, complex amplitude. Delta > 0 puts the
interferer phasor on |k-1><k| with e^{+i Delta t}; for a resonant,
symmetric ladder the sign is immaterial, for a detuned RF link it selects
the side of the LO the interferer sits on.

Zeroth order (Omega_s = 0) the steady state is periodic,
rho(t) = sum_n rho_n e^{i n Delta t}, with

    (L0 - i n Delta) rho_n + L+ rho_{n-1} + L- rho_{n+1} = 0,  Tr rho_0 = 1

where L+ / L- are the commutator superoperators of the interferer's
e^{+i Delta t} / e^{-i Delta t} terms. First order in Omega_s, the signal
drives two ladders of sidebands, a_n at (n Delta + delta) and b_n at
(n Delta - delta):

    (L0 - i(n Delta + delta)) a_n + L+ a_{n-1} + L- a_{n+1} = -Ls+ rho_n
    (L0 - i(n Delta - delta)) b_n + L+ b_{n-1} + L- b_{n+1} = -Ls- rho_n

Observable: probe absorption ~ Im rho_10 (rydsim.experiment convention).
Its one-sided amplitude at +delta per unit Omega_s is

    g = | a_0[1,0] - conj(b_0[1,0]) |

which in the optically thin limit is proportional to the superhet gain, so
the ratio g(Omega_I)/g(0) is the desensitization. Both are solved on a
truncated sideband set |n| <= N that is doubled until the result moves by
less than `rtol`; an unconverged result raises IntegrityError.

Limits (stated on every finding that uses this module):
  * single velocity class — no Doppler averaging;
  * optically thin readout — the transmission nonlinearity of the
    quasi-static `cosite` transfer curve is linearized away, so absolute
    blocking levels differ from the thick-medium quasi-static ones;
  * noise budget not re-evaluated (mean probe power is constant to first
    order in the thin limit): this is a GAIN desensitization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .lindblad import LadderSystem
from .provenance import IntegrityError


def _comm(v: np.ndarray) -> np.ndarray:
    """Superoperator of -i[v, .] for row-major vec (lindblad.py convention)."""
    eye = np.eye(v.shape[0])
    return -1j * (np.kron(v, eye) - np.kron(eye, v.T))


def _link_ops(n: int, link: int) -> tuple[np.ndarray, np.ndarray]:
    """|k-1><k| and |k><k-1| for ladder link k (1-based, as linear_response)."""
    if not 1 <= link <= n - 1:
        raise ValueError(f"link must be in 1..{n - 1}")
    up = np.zeros((n, n), dtype=complex)
    up[link - 1, link] = 1.0
    return up, up.T.copy()


@dataclass(frozen=True)
class FloquetResponse:
    """Hidden-signal response under one interferer."""

    gain: float                 # |a_0[1,0] - conj(b_0[1,0])| per unit Omega_s [s/rad]
    mean_im_rho10: float        # Im rho_0[1,0]: mean probe-absorption proxy
    harmonics_im_rho10: np.ndarray  # one-sided |Im rho_10| tone at n*Delta, n=1..
    n_sidebands: int            # truncation N at convergence


class TwoToneFloquet:
    """Harmonic-balance solver for an LO + interferer on one ladder link.

    `base` carries the LO Rabi frequency on `rf_link` (its omegas entry) and
    every rate and detuning; the interferer and signal are added here.
    """

    def __init__(self, base: LadderSystem, rf_link: int | None = None):
        self.base = base
        n = base.n_levels
        self.n = n
        self.d2 = n * n
        self.rf_link = rf_link if rf_link is not None else n - 1
        self.up, self.down = _link_ops(n, self.rf_link)
        self.l0 = base.liouvillian()
        # per unit Rabi: H term -(1/2)(Omega up + Omega* down)
        self.c_up = _comm(-0.5 * self.up)
        self.c_down = _comm(-0.5 * self.down)
        self.scale = max(float(np.max(np.abs(self.l0))), 1.0)

    # -- assembly -----------------------------------------------------------

    def _system(self, omega_i: float, delta_i: float, shift: float,
                n_sb: int, trace_row: bool) -> sp.csc_matrix:
        """Block-tridiagonal operator for sideband frequencies n*Delta+shift.

        Block row n: (L0 - i(n Delta + shift)) x_n + L+ x_{n-1} + L- x_{n+1}.
        With trace_row, the first row of the n=0 block is replaced by the
        scaled trace condition (the zeroth-order normalization).
        """
        d2 = self.d2
        m = 2 * n_sb + 1
        eye = sp.identity(d2, dtype=complex, format="csr")
        l0 = sp.csr_matrix(self.l0)
        lp = sp.csr_matrix(omega_i * self.c_up)
        lm = sp.csr_matrix(omega_i * self.c_down)
        diag = [l0 - 1j * ((k - n_sb) * delta_i + shift) * eye for k in range(m)]
        blocks = [[None] * m for _ in range(m)]
        for k in range(m):
            blocks[k][k] = diag[k]
            if k > 0:
                blocks[k][k - 1] = lp
            if k < m - 1:
                blocks[k][k + 1] = lm
        a = sp.bmat(blocks, format="lil")
        if trace_row:
            r = n_sb * d2
            a[r, :] = 0.0
            for i in range(self.n):
                a[r, r + i * (self.n + 1)] = self.scale
        return a.tocsc()

    def periodic_state(self, omega_i: float, delta_i: float,
                       n_sb: int) -> np.ndarray:
        """rho_n for n = -N..N, shape (2N+1, n, n)."""
        if omega_i == 0.0:
            out = np.zeros((2 * n_sb + 1, self.n, self.n), dtype=complex)
            out[n_sb] = self.base.steady_state()
            return out
        a = self._system(omega_i, delta_i, 0.0, n_sb, trace_row=True)
        b = np.zeros(a.shape[0], dtype=complex)
        b[n_sb * self.d2] = self.scale
        x = spla.spsolve(a, b)
        return x.reshape(2 * n_sb + 1, self.n, self.n)

    def _signal_ladder(self, rho: np.ndarray, omega_i: float, delta_i: float,
                       shift: float, src: np.ndarray, n_sb: int) -> np.ndarray:
        a = self._system(omega_i, delta_i, shift, n_sb, trace_row=False)
        rhs = -np.concatenate([src @ r.reshape(self.d2) for r in rho])
        return spla.spsolve(a, rhs).reshape(2 * n_sb + 1, self.n, self.n)

    # -- public -------------------------------------------------------------

    def _evaluate(self, omega_i: float, delta_i: float, delta_s: float,
                  n_sb: int, n_harmonics: int) -> FloquetResponse:
        rho = self.periodic_state(omega_i, delta_i, n_sb)
        a = self._signal_ladder(rho, omega_i, delta_i, +delta_s, self.c_up, n_sb)
        b = self._signal_ladder(rho, omega_i, delta_i, -delta_s, self.c_down, n_sb)
        g = abs(a[n_sb, 1, 0] - np.conj(b[n_sb, 1, 0]))
        # Im rho_10(t) tone at +n*Delta: (rho_n - conj(rho_-n))/(2i), one-sided x2
        harm = np.array([
            abs(rho[n_sb + k, 1, 0] - np.conj(rho[n_sb - k, 1, 0]))
            for k in range(1, min(n_harmonics, n_sb) + 1)])
        return FloquetResponse(float(g), float(np.imag(rho[n_sb, 1, 0])),
                               harm, n_sb)

    def response(self, omega_i: float, delta_i: float, delta_s: float,
                 n_harmonics: int = 5, rtol: float = 1e-8,
                 n_start: int | None = None, n_max: int = 4096
                 ) -> FloquetResponse:
        """Hidden-signal gain under an interferer, converged in truncation.

        Angular frequencies throughout [rad/s]. delta_s must be non-zero and
        must not coincide with an interferer harmonic (then signal and spur
        are the same tone and 'gain' is undefined).
        """
        if omega_i < 0:
            raise ValueError("omega_i must be >= 0")
        if delta_s == 0:
            raise ValueError("delta_s must be non-zero (zero is the LO)")
        if omega_i > 0:
            if delta_i == 0:
                raise ValueError("delta_i must be non-zero with an interferer; "
                                 "a co-frequency interferer is just a larger LO")
            ratio = abs(delta_s / delta_i)
            # harmonic 0 is the LO itself (delta_s = 0, rejected above); a
            # signal far inside the first interferer harmonic is not a
            # collision however small delta_s / delta_i is.
            if round(ratio) >= 1 and abs(ratio - round(ratio)) < 1e-6:
                raise IntegrityError(
                    f"signal offset {delta_s:.6g} rad/s is harmonic "
                    f"{round(ratio)} of the interferer offset: signal and spur "
                    "coincide; the first-order gain is undefined there.")
        if omega_i == 0:
            n0 = 1
        else:
            n0 = n_start or int(max(8, 3 * omega_i / abs(delta_i) + 8))
        n_sb = n0
        if n_sb > n_max:
            raise IntegrityError(
                f"Floquet truncation needs N >= {n_sb} sidebands "
                f"(Omega_I/Delta = {omega_i / abs(delta_i):.3g}) but n_max = "
                f"{n_max}; for offsets this small use the quasi-static model "
                "(rydsim.cosite).")
        prev = self._evaluate(omega_i, delta_i, delta_s, n_sb, n_harmonics)
        if omega_i == 0:
            return prev
        while 2 * n_sb <= n_max:
            n_sb *= 2
            cur = self._evaluate(omega_i, delta_i, delta_s, n_sb, n_harmonics)
            if abs(cur.gain - prev.gain) <= rtol * max(cur.gain, 1e-300):
                return cur
            prev = cur
        raise IntegrityError(
            f"Floquet truncation did not converge to {rtol:g} by N = {n_sb} "
            f"sidebands (Omega_I/Delta = {omega_i / abs(delta_i):.3g}); for "
            "offsets this small use the quasi-static model (rydsim.cosite).")

    def desensitization_db(self, omega_i: float, delta_i: float,
                           delta_s: float, **kw) -> float:
        """20 log10(g(0)/g(Omega_I)); > 0 means sensitivity lost."""
        g0 = self.response(0.0, delta_i, delta_s).gain
        g = self.response(omega_i, delta_i, delta_s, **kw).gain
        if g <= 0:
            return float("inf")
        return float(20 * np.log10(g0 / g))


def blocking_rabi(solver: TwoToneFloquet, delta_i: float, delta_s: float,
                  omega_i_grid: np.ndarray, threshold_db: float = 1.0
                  ) -> tuple[float | None, list[float]]:
    """Interferer Rabi frequency [rad/s] at which desensitization first
    reaches threshold_db (log-interpolated), plus the scanned dB values.

    Returns (None, scan) if the threshold is not reached on the grid.
    """
    grid = np.asarray(omega_i_grid, float)
    if grid.ndim != 1 or len(grid) < 2 or np.any(np.diff(grid) <= 0) or grid[0] <= 0:
        raise ValueError("omega_i_grid must be positive and strictly increasing")
    g0 = solver.response(0.0, delta_i, delta_s).gain
    scan: list[float] = []
    for k, w in enumerate(grid):
        g = solver.response(float(w), delta_i, delta_s).gain
        d = float(20 * np.log10(g0 / g)) if g > 0 else float("inf")
        scan.append(d)
        if d >= threshold_db:
            if k == 0:
                return float(w), scan
            d0 = scan[k - 1]
            frac = (threshold_db - d0) / (d - d0)
            lw = np.log(grid[k - 1]) + frac * (np.log(w) - np.log(grid[k - 1]))
            return float(np.exp(lw)), scan
    return None, scan


def thin_transfer(base: LadderSystem, omega_grid: np.ndarray,
                  rf_link: int | None = None) -> np.ndarray:
    """Im rho_10 steady state vs LO Rabi frequency on `rf_link`: the
    optically thin, single-class transfer curve this module linearizes."""
    link = rf_link if rf_link is not None else base.n_levels - 1
    out = np.empty(len(omega_grid))
    for i, w in enumerate(np.asarray(omega_grid, float)):
        om = base.omegas.copy()
        om[link - 1] = w
        s = LadderSystem(omegas=om, deltas=base.deltas, decays=base.decays,
                         dephasings=base.dephasings, transit=base.transit,
                         decay_to=base.decay_to)
        out[i] = float(np.imag(s.steady_state()[1, 0]))
    return out


def optimal_thin_lo(base: LadderSystem, omega_max: float, n: int = 400,
                    rf_link: int | None = None) -> float:
    """LO Rabi frequency [rad/s] maximizing |d Im rho_10 / d Omega|.

    In the optically thin limit the probe power, and hence shot and RIN
    noise, is constant to first order, so the NEF optimum is the
    steepest-slope point. Refuses an optimum on the scan boundary.
    """
    from scipy.interpolate import CubicSpline

    grid = np.linspace(0.0, omega_max, n)
    spl = CubicSpline(grid, thin_transfer(base, grid, rf_link))
    fine = np.linspace(grid[2], grid[-3], 20 * n)
    k = int(np.argmax(np.abs(spl.derivative()(fine))))
    if k in (0, len(fine) - 1):
        raise IntegrityError(
            f"steepest-slope LO lies on the scan boundary "
            f"({fine[k] / (2 * np.pi):.4g} Hz); widen omega_max.")
    return float(fine[k])


@dataclass(frozen=True)
class SelectivityPoint:
    offset: float                  # interferer offset Delta [rad/s]
    blocking_rabi: float | None    # 1 dB (threshold) blocking Rabi [rad/s]
    peak_gain_db: float            # largest gain ENHANCEMENT seen before blocking (>=0)
    scan_db: tuple[float, ...]     # desensitization at each grid point scanned


def selectivity_curve(solver: TwoToneFloquet, offsets: np.ndarray,
                      delta_s: float, omega_i_grid: np.ndarray,
                      threshold_db: float = 1.0) -> list[SelectivityPoint]:
    """Blocking Rabi frequency vs interferer offset (the receiver's
    selectivity). Offsets on which delta_s is an interferer harmonic are
    refused by `TwoToneFloquet.response`."""
    out: list[SelectivityPoint] = []
    for d_i in np.asarray(offsets, float):
        b, scan = blocking_rabi(solver, float(d_i), delta_s, omega_i_grid,
                                threshold_db)
        enh = max([0.0] + [-x for x in scan])
        out.append(SelectivityPoint(float(d_i), b, float(enh), tuple(scan)))
    return out


def ac_stark_shift_crit(base: LadderSystem, threshold_db: float = 1.0,
                        rf_link: int | None = None,
                        s_max: float | None = None) -> float:
    """Light shift s [rad/s] that costs `threshold_db` of quasi-static gain.

    Far off resonance (Delta >> every other rate) an interferer of Rabi
    frequency Omega_I on link k shifts level k-1 by +s and level k by -s,
    s = Omega_I^2 / (4 Delta) (second-order effective Hamiltonian). In
    LadderSystem's cumulative-detuning convention that is a static detuning
    change (delta_{k-1}, delta_k) -> (delta_{k-1} - s, delta_k + 2 s): the
    coupling two-photon resonance moves too, not only the RF one. Solving
    the static problem for the s at which |d Im rho_10 / d Omega_L| has
    dropped by threshold_db gives the asymptotic blocking law

        Omega_I,block = sqrt(4 Delta s_crit)    (Delta -> infinity)

    valid for a hidden signal offset well below the slow relaxation rates
    (validated against the Floquet solver to ~0.2% at 50 MHz - 2 GHz).
    """
    from scipy.optimize import brentq

    link = rf_link if rf_link is not None else base.n_levels - 1
    if link < 2:
        raise ValueError("the AC-Stark mapping needs a link above the probe")
    w0 = float(base.omegas[link - 1])

    def slope(s: float) -> float:
        d = base.deltas.copy()
        d[link - 2] -= s
        d[link - 1] += 2 * s

        def im10(w: float) -> float:
            om = base.omegas.copy()
            om[link - 1] = w
            sys_ = LadderSystem(omegas=om, deltas=d, decays=base.decays,
                                dephasings=base.dephasings,
                                transit=base.transit, decay_to=base.decay_to)
            return float(np.imag(sys_.steady_state()[1, 0]))

        h = 1e-4 * w0
        return abs(im10(w0 + h) - im10(w0 - h)) / (2 * h)

    g0 = slope(0.0)
    hi = s_max if s_max is not None else 100.0 * max(w0, 1.0)

    def f(s: float) -> float:
        g = slope(s)
        return (20 * np.log10(g0 / g) if g > 0 else 1e9) - threshold_db

    if f(hi) < 0:
        raise IntegrityError(
            f"no light shift up to {hi:.4g} rad/s costs {threshold_db} dB of "
            "gain; the AC-Stark blocking law has no solution in range.")
    lo = hi * 1e-9
    if f(lo) > 0:
        raise IntegrityError("gain is already desensitized at negligible "
                             "light shift; the operating point is degenerate.")
    return float(brentq(f, lo, hi, rtol=1e-10))


def quasi_static_thin_blocking(base: LadderSystem, omega_i_grid: np.ndarray,
                               threshold_db: float = 1.0,
                               rf_link: int | None = None) -> float | None:
    """In-band reference: the `rydsim.cosite` phase-average blocking level on
    this module's own thin, single-class transfer curve (Rabi units).

    Independent of the harmonic-balance code path; the Floquet blocking at
    offsets << the response bandwidth must converge to it.
    """
    from scipy.interpolate import CubicSpline

    from .cosite import small_signal_gain

    link = rf_link if rf_link is not None else base.n_levels - 1
    w_l = float(base.omegas[link - 1])
    grid = np.asarray(omega_i_grid, float)
    # The quasi-static envelope |Omega_L + Omega_I e^{i theta}| passes through
    # zero at Omega_I = Omega_L, where the phase average stops converging;
    # only Omega_I < Omega_L is in this model's domain, so the transfer curve
    # is sampled densely on [0, 2 Omega_L] rather than over the whole grid.
    w = np.linspace(0.0, 2.05 * w_l, 1601)
    spl = CubicSpline(w, thin_transfer(base, w, rf_link))
    d = spl.derivative()
    dom = (0.0, float(w[-1]))
    g0 = small_signal_gain(spl, d, w_l, 0.0, dom).gain
    prev: tuple[float, float] | None = None
    for wi in grid[grid < w_l]:
        try:
            g = small_signal_gain(spl, d, w_l, float(wi), dom).gain
        except IntegrityError:
            return None
        db = 20 * np.log10(g0 / g) if g > 0 else float("inf")
        if db >= threshold_db:
            if prev is None:
                return float(wi)
            w0, d0 = prev
            frac = (threshold_db - d0) / (db - d0)
            return float(np.exp(np.log(w0) + frac * (np.log(wi) - np.log(w0))))
        prev = (float(wi), db)
    return None
