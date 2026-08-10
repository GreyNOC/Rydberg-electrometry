"""Doppler averaging of ladder-EIT spectra over a thermal vapor.

For an atom moving with axial velocity v, each field's detuning is shifted by
-k.v (beam direction signs handled via signed wavevectors). The observed
susceptibility is the Maxwell-Boltzmann average over velocity classes,
computed with Gauss-Hermite quadrature (exact for polynomials times the
Gaussian weight; converges fast for smooth spectra).

Emergent physics validated in tests/test_doppler.py:
- sub-Doppler EIT linewidth for counter-propagating mismatched wavelengths,
- the NIST wavelength-mismatch factor: probe-scanned AT splitting appears
  scaled by k_p/k_c = lambda_c/lambda_p (Rb: 480/780), which the field
  inversion must undo. We do NOT hard-code this factor; it must emerge from
  the velocity average, which is a strong end-to-end check of the engine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import KB
from .lindblad import LadderSystem


@dataclass
class DopplerLadder:
    """Ladder system in a thermal vapor with signed field wavevectors.

    Parameters
    ----------
    omegas, deltas, decays, dephasings, transit : as LadderSystem.
    k_signed : signed wavevectors [rad/m] of each optical field along the
        cell axis (positive = +z). RF fields on Rydberg-Rydberg links have
        negligible Doppler shift (k_RF tiny) — pass 0.0 for them.
    mass : atomic mass [kg].
    temperature : vapor temperature [K].
    """

    omegas: np.ndarray
    deltas: np.ndarray
    decays: np.ndarray
    k_signed: np.ndarray
    mass: float
    temperature: float
    dephasings: np.ndarray | None = None
    transit: float = 0.0

    def sigma_v(self) -> float:
        """1D rms thermal velocity sqrt(kB T / m) [m/s]."""
        return float(np.sqrt(KB * self.temperature / self.mass))

    def averaged_coherence(self, v_grid: np.ndarray,
                           probe_index: int = 0) -> complex:
        """Velocity-averaged rho[1,0]/(i Omega_p/2) at the current detunings.

        v_grid must RESOLVE the EIT structure: features are only
        ~gamma_coh/|k_mismatch| wide in velocity (of order 1 m/s in a hot
        mismatched ladder). Use rydsim.eit.resonance_refined_vgrid to build
        an adequate grid; naive coarse quadrature (e.g. 64-node Gauss-
        Hermite) ALIASES the physics and silently returns wrong spectra.
        Cost: one 2^(2N)-sized linear solve per velocity point — this path
        exists for cross-validating the analytic weak-probe engine
        (rydsim.eit), not for production scans.
        """
        v_grid = np.asarray(v_grid, dtype=float)
        s = self.sigma_v()
        w = np.exp(-v_grid**2 / (2 * s**2)) / (s * np.sqrt(2 * np.pi))
        omega_p = self.omegas[probe_index]
        vals = np.empty(v_grid.size, dtype=complex)
        for i, v in enumerate(v_grid):
            deltas_v = (np.asarray(self.deltas, dtype=float)
                        - np.asarray(self.k_signed) * v)
            sys = LadderSystem(
                omegas=self.omegas,
                deltas=deltas_v,
                decays=self.decays,
                dephasings=self.dephasings,
                transit=self.transit,
            )
            rho = sys.steady_state()
            vals[i] = rho[1, 0] / (1j * omega_p / 2.0)
        return complex(np.trapezoid(vals * w, v_grid))

    def spectrum(self, probe_detunings: np.ndarray, v_grid: np.ndarray,
                 probe_index: int = 0) -> np.ndarray:
        """Averaged normalized coherence vs probe detuning [rad/s].

        See averaged_coherence for the v_grid resolution requirement and
        the cost caveat (cross-validation tool, not a production scanner).
        """
        out = np.empty(len(probe_detunings), dtype=complex)
        base = np.asarray(self.deltas, dtype=float).copy()
        for i, dp in enumerate(probe_detunings):
            self.deltas = base.copy()
            self.deltas[probe_index] = dp
            out[i] = self.averaged_coherence(v_grid, probe_index=probe_index)
        self.deltas = base
        return out


def doppler_fwhm(wavelength: float, mass: float, temperature: float) -> float:
    """Doppler FWHM [Hz] of an optical line: sqrt(8 kB T ln2 / m) / lambda."""
    return float(np.sqrt(8.0 * KB * temperature * np.log(2.0) / mass) / wavelength)
