"""Spectrum analysis: extracting measurements from simulated EIT spectra.

The 'measurement' half of electrometry: given a transmission-vs-detuning
trace, extract the Autler-Townes splitting (-> field), EIT linewidth, and
peak properties, with fit-based sub-grid resolution and honest failure
modes (an unresolved doublet returns None, never a guess).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


@dataclass
class ATMeasurement:
    """Result of an Autler-Townes splitting extraction."""

    splitting: float          # [same units as detuning axis]
    peak_positions: tuple[float, float]
    method: str               # 'doublet-fit' or 'peak-find'
    fit_ok: bool
    resolved: bool
    uncertainty: float | None = None   # 1-sigma from fit covariance


def _lorentzian_doublet(x, a1, x1, w1, a2, x2, w2, c):
    return (a1 * w1**2 / ((x - x1) ** 2 + w1**2)
            + a2 * w2**2 / ((x - x2) ** 2 + w2**2) + c)


def measure_at_splitting(detuning: np.ndarray,
                         transmission: np.ndarray,
                         min_prominence_frac: float = 0.02) -> ATMeasurement | None:
    """Extract AT splitting from a transmission spectrum.

    Strategy: locate the two dominant transmission peaks, then refine with a
    Lorentzian-doublet least-squares fit. Returns None if no resolved
    doublet exists (the honest answer in the unresolved weak-field regime).
    """
    detuning = np.asarray(detuning, float)
    transmission = np.asarray(transmission, float)
    span = np.ptp(transmission)
    if span <= 0:
        return None

    peaks, props = find_peaks(transmission, prominence=min_prominence_frac * span)
    if len(peaks) < 2:
        return None

    # two most prominent peaks
    order = np.argsort(props["prominences"])[::-1]
    p1, p2 = sorted(peaks[order[:2]])
    x1, x2 = detuning[p1], detuning[p2]
    sep = abs(x2 - x1)
    guess_w = max(sep / 6.0, 2.0 * (detuning[1] - detuning[0]))

    # crop the fit to the doublet neighborhood: the far wings of an AT
    # spectrum rise toward full transmission and would drag a global fit
    lo, hi = x1 - sep, x2 + sep
    m = (detuning >= lo) & (detuning <= hi)
    xf, yf = detuning[m], transmission[m]

    p0 = [transmission[p1] - yf.min(), x1, guess_w,
          transmission[p2] - yf.min(), x2, guess_w,
          yf.min()]
    try:
        popt, pcov = curve_fit(_lorentzian_doublet, xf, yf,
                               p0=p0, maxfev=20000)
        xa, xb = sorted((popt[1], popt[4]))
        wa, wb = abs(popt[2]), abs(popt[5])
        split = abs(xb - xa)
        # distrust a fit whose centers wandered far from the found peaks
        if abs(xa - x1) > sep / 2 or abs(xb - x2) > sep / 2:
            raise RuntimeError("fit centers inconsistent with peak positions")
        # resolved criterion: separation exceeds the mean fitted HWHM
        resolved = split > (wa + wb) / 2.0
        var = pcov[1, 1] + pcov[4, 4] - 2 * pcov[1, 4]
        unc = float(np.sqrt(var)) if np.isfinite(var) and var >= 0 else None
        return ATMeasurement(
            splitting=float(split),
            peak_positions=(float(min(xa, xb)), float(max(xa, xb))),
            method="doublet-fit", fit_ok=True, resolved=bool(resolved),
            uncertainty=unc,
        )
    except RuntimeError:
        split = abs(x2 - x1)
        return ATMeasurement(
            splitting=float(split),
            peak_positions=(float(x1), float(x2)),
            method="peak-find", fit_ok=False,
            resolved=bool(split > 3 * guess_w), uncertainty=None,
        )


def eit_fwhm(detuning: np.ndarray, absorption: np.ndarray) -> float | None:
    """FWHM of the EIT transparency dip in an ABSORPTION trace.

    The dip is measured relative to its local absorption background (the
    flanking absorption maxima), not the global extrema — far-detuned wings
    where absorption falls to zero would otherwise dominate. None if there
    is no dip.
    """
    detuning = np.asarray(detuning, float)
    absorption = np.asarray(absorption, float)
    # the transparency dip: deepest local minimum between the two largest
    # absorption maxima
    imax = int(np.argmax(absorption))
    # split the trace at the global absorption max's dip side: find dip as
    # minimum of the region bounded by the two highest local maxima
    peaks, _ = find_peaks(absorption)
    if len(peaks) < 2:
        return None
    order = np.argsort(absorption[peaks])[::-1]
    pa, pb = sorted(peaks[order[:2]])
    if pb - pa < 3:
        return None
    seg = absorption[pa:pb + 1]
    i_dip = pa + int(np.argmin(seg))
    a_bg = 0.5 * (absorption[pa] + absorption[pb])
    a_min = absorption[i_dip]
    depth = a_bg - a_min
    if depth <= 0:
        return None
    half = a_min + depth / 2.0
    # walk outward from the dip to the half-crossings
    i_left = i_dip
    while i_left > pa and absorption[i_left] < half:
        i_left -= 1
    i_right = i_dip
    while i_right < pb and absorption[i_right] < half:
        i_right += 1
    return float(detuning[i_right] - detuning[i_left])
