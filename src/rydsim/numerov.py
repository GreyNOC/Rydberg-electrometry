"""Generic Numerov integrator for y''(x) = g(x) y(x).

The Numerov method is O(h^6) locally / O(h^4) globally for equations with no
first-derivative term — the standard workhorse for radial Schrodinger
problems. rydsim.radial builds the Rydberg-specific machinery (sqrt-scaled
coordinate, model potentials, matching, normalization) on top of this; see
docs/spec/02-radial-wavefunctions-numerov.md.

Validation: tests/test_numerov.py integrates exactly solvable problems
(harmonic oscillator, hydrogen) and checks the global convergence order.
"""

from __future__ import annotations

import numpy as np


def numerov_inward(x: np.ndarray, g: np.ndarray,
                   y_outer: float, y_outer2: float,
                   rescale_threshold: float = 1e100) -> np.ndarray:
    """Integrate y'' = g y inward (from x[-1] toward x[0]).

    Parameters
    ----------
    x : strictly increasing uniform grid (spacing h).
    g : g(x) evaluated on the grid.
    y_outer, y_outer2 : seed values y(x[-1]) and y(x[-2]).
    rescale_threshold : if |y| exceeds this, the whole solution so far is
        rescaled to avoid overflow (the classically forbidden region grows
        exponentially inward before the outer turning point). Physical
        results must be normalization-independent, so rescaling is safe.

    Returns
    -------
    y : solution array on the same grid (unnormalized).
    """
    x = np.asarray(x, dtype=float)
    g = np.asarray(g, dtype=float)
    n = x.size
    if n < 3:
        raise ValueError("need at least 3 grid points")
    h = x[1] - x[0]
    if not np.allclose(np.diff(x), h, rtol=1e-10):
        raise ValueError("Numerov requires a uniform grid")

    h2_12 = h * h / 12.0
    f = 1.0 - h2_12 * g          # Numerov auxiliary: f_k = 1 - h^2/12 g_k

    y = np.empty(n)
    y[-1] = y_outer
    y[-2] = y_outer2

    for k in range(n - 3, -1, -1):
        # Numerov recurrence rearranged for inward stepping:
        # f_{k} y_{k} = (12 - 10 f_{k+1}) y_{k+1} - f_{k+2} y_{k+2}
        y[k] = ((12.0 - 10.0 * f[k + 1]) * y[k + 1] - f[k + 2] * y[k + 2]) / f[k]
        if abs(y[k]) > rescale_threshold:
            y[k:] /= abs(y[k])

    return y


def numerov_outward(x: np.ndarray, g: np.ndarray,
                    y_inner: float, y_inner2: float,
                    rescale_threshold: float = 1e100) -> np.ndarray:
    """Integrate y'' = g y outward (from x[0] toward x[-1]). Mirror of inward."""
    x = np.asarray(x, dtype=float)
    g = np.asarray(g, dtype=float)
    n = x.size
    if n < 3:
        raise ValueError("need at least 3 grid points")
    h = x[1] - x[0]
    if not np.allclose(np.diff(x), h, rtol=1e-10):
        raise ValueError("Numerov requires a uniform grid")

    h2_12 = h * h / 12.0
    f = 1.0 - h2_12 * g

    y = np.empty(n)
    y[0] = y_inner
    y[1] = y_inner2

    for k in range(2, n):
        y[k] = ((12.0 - 10.0 * f[k - 1]) * y[k - 1] - f[k - 2] * y[k - 2]) / f[k]
        if abs(y[k]) > rescale_threshold:
            y[: k + 1] /= abs(y[k])

    return y
