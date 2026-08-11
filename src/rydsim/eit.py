"""Weak-probe ladder susceptibility with rigorous Doppler averaging.

Why this module exists
----------------------
The steady-state Lindblad engine (rydsim.lindblad) is exact but must be
solved per velocity class, and naive Gauss-Hermite averaging ALIASES the
EIT structure: transparency features are ~(gamma_EIT / k_2ph) ~ 1 m/s wide
in velocity space while 64-node quadrature samples every ~15 m/s. This was
caught by cross-validation during development (see docs/spec/00-integrity-
audit.md doctrine) and is exactly the kind of silent numerical error the
project exists to prevent.

For the weak-probe regime — the operating regime of Rydberg electrometry —
the ladder coherence has an exact closed form (continued fraction), valid
to first order in Omega_probe:

    rho_10 = (i Omega_p / 2) * S(v)
    S(v) = 1 / (G_1 - i D_1(v) + (Omega_2/2)^2 / (G_2 - i D_2(v)
                + (Omega_3/2)^2 / (G_3 - i D_3(v) + ...)))

with D_k(v) = sum_{j<=k} (delta_j - k_j v) the cumulative Doppler-shifted
detuning and G_k the coherence decay rate of level k against ground
(G_k = pop_decay_k / 2 + pure_dephasing_k + transit). This is vectorized
over a dense, resonance-refined velocity grid, integrated with trapezoid
weights against the 1D Maxwell-Boltzmann distribution, and every public
entry point can self-check convergence by density doubling.

Validation: tests/test_eit.py — agreement with the Lindblad engine class-
by-class (no averaging), quadrature convergence, and the emergent
wavelength-mismatch AT factor now measured on the *converged* average.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import EPS0, HBAR, KB
from .provenance import IntegrityError


@dataclass
class LadderChiParams:
    """Weak-probe ladder parameters (all angular rates, rad/s; k in rad/m).

    Arrays indexed by ladder link (0 = probe link g-e, 1 = coupling, ...).
    coherence_decays[k] = G_{k+1}: total coherence decay of level k+1
    against the ground state = (population decay of level k+1)/2 +
    accumulated pure dephasing + transit rate.
    """

    omegas: np.ndarray            # link Rabi frequencies; omegas[0] unused in S
    deltas: np.ndarray            # link detunings
    k_signed: np.ndarray          # signed wavevectors per link (0 for RF)
    coherence_decays: np.ndarray  # G_k for levels 1..N-1

    def __post_init__(self):
        self.omegas = np.atleast_1d(np.asarray(self.omegas, float))
        self.deltas = np.atleast_1d(np.asarray(self.deltas, float))
        self.k_signed = np.atleast_1d(np.asarray(self.k_signed, float))
        self.coherence_decays = np.atleast_1d(
            np.asarray(self.coherence_decays, float))
        n = self.omegas.size
        if not (self.deltas.size == self.k_signed.size
                == self.coherence_decays.size == n):
            raise ValueError("all parameter arrays must share length = #links")


def chi_ladder(params: LadderChiParams, probe_detunings: np.ndarray,
               v: np.ndarray | float = 0.0) -> np.ndarray:
    """Normalized weak-probe response S for each (probe detuning, velocity).

    Returns array of shape (len(probe_detunings),) if v is scalar, else
    (len(probe_detunings), len(v)). Response convention: absorption is
    Re(S) after the i*Omega_p/2 normalization used across rydsim
    (rho_10/(i Omega_p/2) = S with Re S >= 0 on resonance).
    """
    dp = np.atleast_1d(np.asarray(probe_detunings, float))
    v_arr = np.atleast_1d(np.asarray(v, float))
    scalar_v = np.isscalar(v) or np.asarray(v).ndim == 0

    n_links = params.omegas.size
    # cumulative detunings D_k(v): (n_links, Ndp, Nv)
    dp_grid = dp[:, None] + np.zeros((1, v_arr.size))
    cum = np.zeros((dp.size, v_arr.size), dtype=complex)
    # build from the innermost (top) link downward via continued fraction
    # D_k = sum_{j<=k} (delta_j - k_j v); link 0 detuning is the scanned dp
    deltas_eff = [dp_grid - params.k_signed[0] * v_arr[None, :]]
    for j in range(1, n_links):
        deltas_eff.append(
            deltas_eff[j - 1]
            + params.deltas[j] - params.k_signed[j] * v_arr[None, :])

    x = params.coherence_decays[-1] - 1j * deltas_eff[-1]
    for k in range(n_links - 2, -1, -1):
        x = (params.coherence_decays[k] - 1j * deltas_eff[k]
             + (params.omegas[k + 1] / 2.0) ** 2 / x)
    s = 1.0 / x
    return s[:, 0] if scalar_v else s


def resonance_refined_vgrid(params: LadderChiParams, sigma_v: float,
                            probe_span: tuple[float, float],
                            base_per_sigma: int = 96,
                            fine_rate: float | None = None,
                            oversample: int = 16,
                            v_max_sigmas: float = 6.5) -> np.ndarray:
    """Velocity grid dense enough to resolve every EIT/AT resonance.

    The narrowest velocity-space feature has width ~ G_min / |k_eff| where
    k_eff is the (mismatch) wavevector governing the relevant multiphoton
    resonance. We lay a coarse Gaussian-scale base grid, then add fine
    windows around every velocity where a multiphoton resonance condition
    can be met for any probe detuning in the scan span.
    """
    g_min = fine_rate or float(np.min(params.coherence_decays))
    v_lim = v_max_sigmas * sigma_v
    base = np.linspace(-v_lim, v_lim, int(2 * v_max_sigmas * base_per_sigma) + 1)

    ks = params.k_signed
    # effective wavevectors of the 1..n photon resonances
    k_cum = np.cumsum(ks)
    fine_segments = [base]
    dp_lo, dp_hi = probe_span
    for order, k_eff in enumerate(k_cum):
        if abs(k_eff) < 1e-6:
            continue
        # resonance: cumulative detuning zero (+- dressed offsets up to
        # sum of downstream Rabi/2 — widen the window accordingly)
        d_static = float(np.sum(params.deltas[1:order + 1])) if order else 0.0
        dress = float(np.sum(params.omegas[order + 1:]) / 2.0 + np.sum(
            params.omegas[1:order + 1]) / 2.0)
        width = g_min / abs(k_eff)
        v_lo = (dp_lo + d_static - dress) / k_eff
        v_hi = (dp_hi + d_static + dress) / k_eff
        v_lo, v_hi = min(v_lo, v_hi), max(v_lo, v_hi)
        v_lo = max(v_lo - 10 * width, -v_lim)
        v_hi = min(v_hi + 10 * width, v_lim)
        if v_hi <= v_lo:
            continue
        n_fine = int((v_hi - v_lo) / (width / oversample)) + 2
        n_fine = min(n_fine, 400_000)  # hard memory guard
        fine_segments.append(np.linspace(v_lo, v_hi, n_fine))
    grid = np.unique(np.concatenate(fine_segments))
    return grid


def doppler_average(params: LadderChiParams, probe_detunings: np.ndarray,
                    mass: float, temperature: float,
                    v_grid: np.ndarray | None = None,
                    check_convergence: bool = False,
                    convergence_tol: float = 1e-4,
                    chunk: int = 4096) -> np.ndarray:
    """Maxwell-Boltzmann-averaged weak-probe response.

    Trapezoid integration on a resonance-refined nonuniform velocity grid.
    With check_convergence=True the integral is recomputed on a step-halved
    AND domain-widened grid (spec 06 §4.4: both must be varied);
    disagreement beyond convergence_tol (spec: 1e-4 max-norm relative)
    raises IntegrityError rather than returning an unconverged number.
    """
    dp = np.asarray(probe_detunings, float)
    sigma = float(np.sqrt(KB * temperature / mass))
    if v_grid is None:
        v_grid = resonance_refined_vgrid(
            params, sigma, (float(dp.min()), float(dp.max())))

    # chunked trapezoid with overlapping boundaries (no dropped segments)
    def integrate_exact(vg: np.ndarray) -> np.ndarray:
        w = np.exp(-vg**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
        out = np.zeros(dp.size, dtype=complex)
        for i0 in range(0, vg.size - 1, chunk):
            sl = slice(i0, min(i0 + chunk + 1, vg.size))  # +1 overlap
            s = chi_ladder(params, dp, vg[sl])
            out += np.trapezoid(s * w[sl][None, :], vg[sl], axis=1)
        return out

    result = integrate_exact(v_grid)
    if check_convergence:
        mids = 0.5 * (v_grid[:-1] + v_grid[1:])
        # widen the domain by ~1.3 sigma beyond the base grid on each side
        # (spec: widen AND halve the step in the same re-check)
        v_lo, v_hi = float(v_grid.min()), float(v_grid.max())
        ext_lo = np.linspace(v_lo - 1.3 * sigma, v_lo, 32, endpoint=False)
        ext_hi = np.linspace(v_hi + 1.3 * sigma, v_hi, 32, endpoint=False)
        dense = np.sort(np.concatenate([v_grid, mids, ext_lo, ext_hi]))
        result2 = integrate_exact(dense)
        scale = np.max(np.abs(result2))
        err = float(np.max(np.abs(result - result2)) / scale) if scale else 0.0
        if err > convergence_tol:
            raise IntegrityError(
                f"Doppler average unconverged: doubling the velocity grid "
                f"changes the spectrum by {err:.2%} (> {convergence_tol:.2%}). "
                f"Refine the grid; do not trust this output.")
        return result2
    return result


def chi_si(s_norm: np.ndarray, number_density: float,
           dipole_ge: float) -> np.ndarray:
    """Dimensionless SI susceptibility from the normalized response S.

    chi(Delta_p) = i (N p_ge^2 / (eps0 hbar)) * S  (spec 06 §2.4, VERIFIED
    against Finkelstein et al., NJP 25, 035001 (2023) Eqs. 9-11).
    number_density [m^-3]; dipole_ge [C m] is the probe-transition dipole
    matrix element. Im(chi) >= 0 is absorption with our S convention.
    """
    return 1j * number_density * dipole_ge**2 / (EPS0 * HBAR) * np.asarray(s_norm)


def transmission(chi: np.ndarray, length: float,
                 lambda_probe: float) -> np.ndarray:
    """Thin-medium Beer-Lambert power transmission T = exp(-k_p L Im chi).

    Valid for optically thin-to-moderate media with undepleted coupling
    (spec 06 §7.2); the optically thick z-propagation chain is a cell-module
    concern.
    """
    k_p = 2 * np.pi / lambda_probe
    return np.exp(-k_p * length * np.imag(np.asarray(chi)))


def dipole_from_linewidth(gamma_e: float, omega_atom: float,
                          degeneracy_factor: float = 1.0) -> float:
    """Closed-transition dipole moment from the natural linewidth [C m].

    Gamma = omega^3 p^2 / (3 pi eps0 hbar c^3)  =>  p = sqrt(3 pi eps0
    hbar c^3 Gamma / omega^3) * sqrt(degeneracy_factor). For the aligned
    closed two-level cycle degeneracy_factor = 1; real degenerate D lines
    deviate (spec 06 §2.4 caveat) — the exact reduced elements land with
    the matrix-element module. Validated absolutely by the B-2 sum rule
    sigma_0 = 3 lambda^2 / (2 pi) in tests/test_spec06_absolute.py.
    """
    from scipy.constants import c as c_light
    return float(np.sqrt(3 * np.pi * EPS0 * HBAR * c_light**3 * gamma_e
                         / omega_atom**3 * degeneracy_factor))
