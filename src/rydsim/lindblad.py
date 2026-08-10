"""Lindblad master-equation engine for ladder EIT systems.

Builds the Liouvillian superoperator for an N-level atom driven by classical
fields in the rotating frame (RWA) and solves for the steady-state density
matrix as a linear system. This is the numerical core behind every EIT /
Autler-Townes spectrum in RydSim.

Conventions (docs/spec/00-conventions.md):
- Rotating-frame Hamiltonian: H/hbar = -sum_k Delta_k |k><k|
  - (1/2) sum_links Omega_ij (|i><j| + |j><i|), with Omega the *angular*
  Rabi frequency (rad/s) defined via Omega = d E / hbar, and Delta_k the
  cumulative detuning of level k (rad/s).
- Gamma_ij is the population decay rate from level i to level j (rad/s);
  gamma_extra[i] is additional *pure dephasing* of level i's coherence with
  the ground state (rad/s, half-width convention handled internally).

Validation: tests/test_lindblad.py checks trace preservation, positivity,
the two-level analytic steady state, and the 3-level weak-probe analytic
susceptibility to 1e-10.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LadderSystem:
    """An N-level ladder: level 0 = ground, level k driven by field k-1.

    Parameters
    ----------
    omegas : Rabi frequencies [rad/s] of the N-1 ladder links
        (probe = omegas[0], coupling = omegas[1], RF = omegas[2], ...).
    deltas : single-photon detunings [rad/s] of each field from its link.
    decays : decays[i] = total population decay rate of level i [rad/s],
        assumed (by default routing) to return to the level below;
        set `decay_to` for explicit routing.
    dephasings : extra pure-dephasing rate of each level's coherence [rad/s]
        (laser linewidths, collisions, transit — entered per level).
    transit : transit-time exchange rate [rad/s]: all levels relax toward
        the ground state at this rate (fresh-atom replacement in a beam).
    """

    omegas: np.ndarray
    deltas: np.ndarray
    decays: np.ndarray
    dephasings: np.ndarray | None = None
    transit: float = 0.0
    decay_to: dict[int, dict[int, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.omegas = np.atleast_1d(np.asarray(self.omegas, dtype=float))
        self.deltas = np.atleast_1d(np.asarray(self.deltas, dtype=float))
        self.decays = np.atleast_1d(np.asarray(self.decays, dtype=float))
        n = self.omegas.size + 1
        if self.deltas.size != n - 1:
            raise ValueError("need one detuning per field")
        if self.decays.size != n:
            raise ValueError("need one decay rate per level")
        if self.dephasings is None:
            self.dephasings = np.zeros(n)
        self.dephasings = np.atleast_1d(np.asarray(self.dephasings, dtype=float))
        if self.dephasings.size != n:
            raise ValueError("need one dephasing rate per level")

    @property
    def n_levels(self) -> int:
        return self.omegas.size + 1

    def hamiltonian(self) -> np.ndarray:
        """H/hbar in the rotating frame [rad/s], complex Hermitian (n,n)."""
        n = self.n_levels
        h = np.zeros((n, n), dtype=complex)
        # cumulative detunings on the diagonal: level k at -(Delta_1+...+Delta_k)
        cum = 0.0
        for k in range(1, n):
            cum += self.deltas[k - 1]
            h[k, k] = -cum
        for k in range(n - 1):
            h[k, k + 1] = -0.5 * self.omegas[k]
            h[k + 1, k] = -0.5 * self.omegas[k]
        return h

    def collapse_operators(self) -> list[tuple[float, np.ndarray]]:
        """List of (rate, C) Lindblad collapse operators."""
        n = self.n_levels
        ops: list[tuple[float, np.ndarray]] = []
        for i in range(1, n):
            routing = self.decay_to.get(i)
            if routing is None:
                routing = {i - 1: self.decays[i]}
            for j, rate in routing.items():
                if rate > 0:
                    c = np.zeros((n, n))
                    c[j, i] = 1.0
                    ops.append((rate, c))
        # pure dephasing: projector on each level
        for i in range(n):
            if self.dephasings[i] > 0:
                c = np.zeros((n, n))
                c[i, i] = 1.0
                ops.append((2.0 * self.dephasings[i], c))
                # factor 2: with L = sqrt(2 gamma) |i><i|, the coherence
                # rho_gi decays at gamma — the half-width convention used
                # for "laser linewidth gamma" entries. See spec 00.
        # transit: measure-and-replace channel (spec 06 §2.2): the FULL Kraus
        # set {|g><i| for ALL i, including i=g} gives uniform decay of every
        # population and coherence at gamma_t with repopulation of g.
        # Omitting the i=g operator makes ground coherences decay at
        # gamma_t/2 instead of gamma_t — a real (caught) bug class.
        if self.transit > 0:
            for i in range(0, n):
                c = np.zeros((n, n))
                c[0, i] = 1.0
                ops.append((self.transit, c))
        return ops

    def liouvillian(self) -> np.ndarray:
        """The (n^2, n^2) superoperator L with drho_vec/dt = L rho_vec.

        Vectorization: rho_vec = rho.reshape(n*n) row-major, i.e. index
        a = i*n + j holds rho_ij. Using the identities
        vec(A rho B) = (A kron B^T) vec(rho) for row-major vec.
        """
        n = self.n_levels
        h = self.hamiltonian()
        eye = np.eye(n)
        # -i [H, rho]  ->  -i (H kron I - I kron H^T)
        lv = -1j * (np.kron(h, eye) - np.kron(eye, h.T))
        for rate, c in self.collapse_operators():
            cd = c.conj().T
            cdc = cd @ c
            lv += rate * (
                np.kron(c, c.conj())
                - 0.5 * np.kron(cdc, eye)
                - 0.5 * np.kron(eye, cdc.T)
            )
        return lv

    def steady_state(self) -> np.ndarray:
        """Steady-state density matrix (n, n): solve L rho = 0, Tr rho = 1."""
        n = self.n_levels
        lv = self.liouvillian()
        # replace the first row with the trace condition, SCALED to the
        # magnitude of the other rows (spec 06 §2.3): an unscaled 1.0 row
        # next to ~1e8-magnitude rows costs ~8 digits of conditioning
        a = lv.copy()
        s = max(float(np.max(np.abs(lv))), 1.0)
        a[0, :] = 0.0
        a[0, ::n + 1] = s  # coefficients of rho_ii in row-major vec
        b = np.zeros(n * n, dtype=complex)
        b[0] = s
        rho_vec = np.linalg.solve(a, b)
        rho = rho_vec.reshape(n, n)
        # hermitize (numerical cleanup only; asymmetry is at solver precision)
        return 0.5 * (rho + rho.conj().T)


def probe_coherence(rho: np.ndarray) -> complex:
    """The ground-to-first-excited coherence rho_ge that sets probe response.

    With the convention above, Im(rho_ge... ) enters absorption via
    chi ∝ rho_[1,0] (see rydsim.eit for the full susceptibility chain).
    """
    return rho[1, 0]


def linear_response(sys: LadderSystem, drive_link: int,
                    deltas_beat: np.ndarray,
                    readout_ij: tuple[int, int] = (1, 0)) -> np.ndarray:
    """Complex transfer function H(delta) of the superhet beat (spec 08 §2.3).

    Modulate the Rabi frequency of `drive_link` as
    Omega(t) = Omega_L + dOmega cos(delta t); to first order the readout
    coherence responds as rho_ij(t) = rho0_ij + Re[rho1_ij e^{-i delta t}]
    with rho1 = -(L0 + i delta)^{-1} L1 rho0 per unit dOmega.

    Returns H(delta) = rho1[readout_ij] [s/rad units of coherence response
    per unit Rabi modulation]. |H| rolls off past the atomic response
    bandwidth: f_3dB from |H(delta_3dB)| = |H(0)|/sqrt(2) is the
    first-principles instantaneous bandwidth.

    Self-check contract (spec 08 B16, enforced in tests): H(0) equals the
    finite-difference derivative d rho_ij / d Omega of the steady state to
    < 0.1%.

    Note delta=0 is handled by lstsq (L0 is singular there; the RHS is
    traceless so the system is consistent).
    """
    n = sys.n_levels
    if not (1 <= drive_link <= n - 1):
        raise ValueError(f"drive_link must be in 1..{n-1}")
    lv0 = sys.liouvillian()
    rho0 = sys.steady_state()

    # L1 = -(i/hbar)[V, .] with V = -(hbar/2)(|k><k+1| + |k+1><k|) per unit
    # Rabi: dH/dOmega for the driven link (same convention as hamiltonian()).
    k = drive_link
    dv = np.zeros((n, n), dtype=complex)
    dv[k - 1, k] = -0.5
    dv[k, k - 1] = -0.5
    eye = np.eye(n)
    l1 = -1j * (np.kron(dv, eye) - np.kron(eye, dv.T))

    rhs = l1 @ rho0.reshape(n * n)
    out = np.empty(len(deltas_beat), dtype=complex)
    i, j = readout_ij
    scale = max(float(np.max(np.abs(lv0))), 1.0)
    for m, db in enumerate(np.asarray(deltas_beat, float)):
        a = lv0 + 1j * db * np.eye(n * n)
        b = -rhs
        if abs(db) < 1e-6 * scale:
            # L0 is singular at delta = 0 (trace zero-mode). The PHYSICAL
            # perturbation is traceless (Tr(rho0 + rho1) = 1), which is NOT
            # what lstsq's minimum-norm solution enforces — impose it by
            # replacing the redundant row with the scaled trace condition.
            a = a.copy()
            a[0, :] = 0.0
            a[0, ::n + 1] = scale
            b = b.copy()
            b[0] = 0.0
        rho1 = np.linalg.solve(a, b)
        out[m] = rho1.reshape(n, n)[i, j]
    return out


def ibw_3db(sys: LadderSystem, drive_link: int,
            delta_max: float, n_scan: int = 200) -> float:
    """First-principles instantaneous bandwidth [rad/s]: the beat frequency
    where |H| falls to |H(0)|/sqrt(2). Scans log-spaced then bisects."""
    h0 = abs(linear_response(sys, drive_link, np.array([0.0]))[0])
    grid = np.logspace(np.log10(delta_max) - 4, np.log10(delta_max), n_scan)
    hmag = np.abs(linear_response(sys, drive_link, grid))
    target = h0 / np.sqrt(2.0)
    below = np.where(hmag < target)[0]
    if len(below) == 0:
        raise ValueError("no 3 dB point within delta_max — widen the scan")
    hi = grid[below[0]]
    lo = grid[below[0] - 1] if below[0] > 0 else grid[0] / 10
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        if abs(linear_response(sys, drive_link, np.array([mid]))[0]) > target:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def weak_probe_chi3(delta_p: np.ndarray, omega_c: float, delta_c: float,
                    gamma_e: float, gamma_r: float) -> np.ndarray:
    """Analytic weak-probe 3-level ladder response (dimensionless shape).

    Returns the steady-state coherence rho_eg per unit (Omega_p/2), i.e.
        rho_eg = (i/2) Omega_p * S,   S = 1 / (G_e - i d_p + Omega_c^2/4 / (G_r - i (d_p + d_c)))
    with G_e = gamma_e/2 (coherence decay of e), G_r = gamma_r (ground-
    Rydberg coherence decay), matching the Lindblad solution in the weak-
    probe limit. All rates angular [rad/s]. Standard result; see spec 06.
    """
    delta_p = np.asarray(delta_p, dtype=complex)
    ge = gamma_e / 2.0
    gr = gamma_r
    denom = (ge - 1j * delta_p) + (omega_c**2 / 4.0) / (gr - 1j * (delta_p + delta_c))
    return 1.0 / denom
