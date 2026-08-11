"""Spec 01 benchmarks (AS-01..AS-15, docs/spec/01-atomic-structure.md §6)
plus the integrity guards mandated by docs/spec/00-integrity-audit.md §3
(items 1-5) and rulings R-10/R-11/R-14/R-15.

Every AS-x test recomputes from the rydsim.atom public API only (spec 01 §6
pytest contract — no literal reuse of the spec's 'calc' values).
"""

import math
import pathlib

import numpy as np
import pytest

from rydsim import atom
from rydsim.atom import (
    CS133,
    INV_CM_HZ,
    RB85,
    RB87,
    RB87_E_ION_FROM_5P32_F3,
    RB87_E_ION_FROM_5S12_F1,
    RitzSeries,
    Species,
    binding_energy_hz,
    coupling_laser_hz,
    coupling_laser_sourced_hz,
    coupling_wavelength_m,
    coupling_wavelength_sourced_m,
    element_symbol,
    energy_hz,
    energy_sourced_hz,
    hyperfine_levels,
    hyperfine_shift_hz,
    n_star,
    polarization_defect,
    probe_transition_hz,
    quantum_defect,
    rf_transition_hz,
    ritz_form_a,
    ritz_form_b,
    rydberg_constant_hz,
    rydberg_hfs_A_hz,
    transition_hz,
)
from rydsim.constants import C, RYD_HZ
from rydsim.provenance import Confidence, IntegrityError

THZ = 1e12
GHZ = 1e9
MHZ = 1e6


# ---------------------------------------------------------------------------
# §6 benchmark table
# ---------------------------------------------------------------------------

def test_as01_rydberg_constant_rb87():
    """AS-01: c*R_M(Rb-87) = 3 289 821 194.66 MHz +-0.5 MHz (Mack 2011)."""
    assert rydberg_constant_hz(RB87) == pytest.approx(
        3_289_821_194.66 * MHZ, abs=0.5 * MHZ)


def test_as01b_rydberg_constant_cs():
    """AS-01b: R_M(Cs) = 109 736.862 733 9 cm^-1 +-5e-6 cm^-1 (Deiglmayr 2016)."""
    rm_cm = rydberg_constant_hz(CS133) / INV_CM_HZ
    assert rm_cm == pytest.approx(109_736.862_733_9, abs=5e-6)


def test_as01_rb85_cross_check():
    """Spec 01 §2.1 cross-check: c*R_M(Rb-85) = 3 289 820 706.08 MHz."""
    assert rydberg_constant_hz(RB85) == pytest.approx(
        3_289_820_706.08 * MHZ, abs=0.5 * MHZ)


@pytest.mark.parametrize(
    "sp, n1, n2, expected_hz, tol_hz, row",
    [
        (RB85, 53, 54, 14.233 * GHZ, 5 * MHZ,
         "AS-02: Rb-85 53D5/2->54P3/2 (Sedlacek 2012, = C4a)"),
        (RB85, 50, 51, 17.04 * GHZ, 10 * MHZ,
         "AS-02b: Rb-85 50D5/2->51P3/2 (Holloway 2014, = C4c)"),
        (RB85, 28, 29, 104.77 * GHZ, 50 * MHZ,
         "AS-03: Rb-85 28D5/2->29P3/2 (Holloway 2014, = C4d)"),
        (RB87, 39, 40, 36.9 * GHZ, 100 * MHZ,
         "AS-03b: Rb-87 39D5/2->40P3/2 (Tu 2024, = C4e)"),
        (CS133, 47, 48, 6.94 * GHZ, 10 * MHZ,
         "AS-03c: Cs 47D5/2->48P3/2 (Jing 2020, = C4b; calc 6.9452 GHz)"),
    ],
)
def test_as02_as03_rf_intervals(sp, n1, n2, expected_hz, tol_hz, row):
    """AS-02/02b/03/03b/03c: nD5/2 -> (n+1)P3/2 microwave intervals.

    Interval magnitude via Eq. (1.9) binding-energy difference (E_I cancels
    exactly — Rb-85's +-40 MHz absolute systematic is immune, audit R8).
    """
    nu = abs(transition_hz(sp, n1, 2, 2.5, n2, 1, 1.5))
    assert nu == pytest.approx(expected_hz, abs=tol_hz), row


def test_as02_interval_is_downward():
    """Spec 01 §2.5 ordering: (n+1)P3/2 lies BELOW nD5/2 -> sign = -1."""
    nu_abs, sign = rf_transition_hz(RB85, 53, 2, 2.5, 54, 1, 1.5)
    assert sign == -1
    assert nu_abs > 0


@pytest.mark.parametrize(
    "n, expected_thz, row",
    [
        (19, 612.728_838_1, "AS-04: Mack 2011 Table III"),
        (20, 614.232_154_2, "AS-05: Mack 2011"),
        (21, 615.490_168_7, "AS-06: Mack 2011"),
    ],
)
def test_as04_06_rb87_absolute_uv(n, expected_thz, row):
    """AS-04..06: Rb-87 5P3/2(F=3)->nS1/2 absolute frequency +-1.5 MHz,
    via the Mack §3.3 anchor E_i(from 5P3/2(F=3)) minus the nS binding
    energy. Catches the 4.27 GHz hyperfine-reference pitfall (§4.3-1)."""
    nu = RB87_E_ION_FROM_5P32_F3.value - binding_energy_hz(RB87, n, 0, 0.5)
    assert nu == pytest.approx(expected_thz * THZ, abs=1.5 * MHZ), row


def test_as07_rb87_d2_f2_fp3():
    """AS-07: Rb-87 5S1/2(F=2)->5P3/2(F'=3) = 384.228 115 2 THz +-0.1 MHz
    via Eq. (1.10) — closes Steck centroid + (1.7) against Mack's comb value."""
    nu = probe_transition_hz(RB87, "D2", 2, 3)
    assert nu == pytest.approx(384.228_115_2 * THZ, abs=0.1 * MHZ)


def test_as07b_rb87_d2_centroid_data_integrity():
    """AS-07b: Rb-87 D2 centroid 384.230 484 468 5 THz +-10 kHz (Steck 2.3.4)."""
    assert RB87.d2.nu0_hz == pytest.approx(384.230_484_468_5 * THZ, abs=10e3)


@pytest.mark.parametrize(
    "n, l, j, expected_cm, tol_cm, row",
    [
        (27, 1, 0.5, 31_206.189_769_8, 3.3e-6, "AS-08: Deiglmayr Table I, 27P1/2"),
        (47, 1, 1.5, 31_348.316_589_8, 5.0e-6, "AS-08b: Deiglmayr Table I, 47P3/2"),
        (74, 1, 1.5, 31_384.351_900_1, 5.0e-6, "AS-08c: Deiglmayr Table I, 74P3/2"),
    ],
)
def test_as08_cs_uv_wavenumbers(n, l, j, expected_cm, tol_cm, row):
    """AS-08/08b/08c: Cs 6S1/2(centroid) -> nP wavenumbers, +-100-150 kHz.

    Tests the (E_I, Form-A P-defect) pair self-consistency; would expose the
    audit-R3 erratum value if it were ever encoded (it must not be)."""
    nu = energy_hz(CS133, n, l, j, ref="ground_centroid")
    assert nu / INV_CM_HZ == pytest.approx(expected_cm, abs=tol_cm), row


def test_as09_polarization_formula_vs_measured_g_defects():
    """AS-09: Eq. (1.6) at l=4, n->inf reproduces the measured G defects to
    <= 5 % (Rb: calc vs 0.003 999 -> 1.8 %; Cs: vs 0.007 039 -> 3.8 %)."""
    rb_calc = polarization_defect(RB85.alpha_core_au, math.inf, 4)
    rb_meas = RB85.series[(4, 3.5)].d0
    assert abs(rb_calc - rb_meas) / rb_meas < 0.05
    cs_calc = polarization_defect(CS133.alpha_core_au, math.inf, 4)
    cs_meas = CS133.series[(4, 3.5)].d0
    assert abs(cs_calc - cs_meas) / cs_meas < 0.05


def test_as10_rb_d2_isotope_shift():
    """AS-10: nu(Rb-87 D2) - nu(Rb-85 D2) = 78.095 MHz +-0.024 MHz
    (Steck internal consistency)."""
    shift = RB87.d2.nu0_hz - RB85.d2.nu0_hz
    assert shift == pytest.approx(78.095 * MHZ, abs=0.024 * MHZ)


def test_as11_cs_12s_form_b_minus_form_a():
    """AS-11: Cs 12S1/2 Form B - Form A energy difference = 49.2 MHz +-5 MHz
    (ruling R-14; asserts Form B is actually in use for Cs nS1/2)."""
    series = CS133.series[(0, 0.5)]
    assert series.form == "iterated"
    n = np.array([12.0])
    d_a = ritz_form_a(series, n)[0]
    d_b = ritz_form_b(series, n)[0]
    c_rm = rydberg_constant_hz(CS133)
    e_a = -c_rm / (12.0 - d_a) ** 2
    e_b = -c_rm / (12.0 - d_b) ** 2
    assert abs(e_b - e_a) == pytest.approx(49.2 * MHZ, abs=5 * MHZ)
    # dispatch check: the public API must return the Form-B fixed point
    with pytest.warns(UserWarning):  # n=12 < n_min_mhz=25
        assert quantum_defect(CS133, 12, 0, 0.5) == pytest.approx(d_b, abs=1e-12)


def test_as12_ground_hyperfine_splittings():
    """AS-12: ground splittings via Eq. (1.7): 2A (Rb-87), 3A (Rb-85),
    4A (Cs, exact — defines the SI second); +-1 kHz / exact."""
    rb87 = (hyperfine_shift_hz(RB87.ground_hfs, 1.5, 0.5, 2)
            - hyperfine_shift_hz(RB87.ground_hfs, 1.5, 0.5, 1))
    assert rb87 == pytest.approx(6_834_682_610.904_290, abs=1e3)
    rb85 = (hyperfine_shift_hz(RB85.ground_hfs, 2.5, 0.5, 3)
            - hyperfine_shift_hz(RB85.ground_hfs, 2.5, 0.5, 2))
    assert rb85 == pytest.approx(3_035_732_439.0, abs=1e3)
    cs = (hyperfine_shift_hz(CS133.ground_hfs, 3.5, 0.5, 4)
          - hyperfine_shift_hz(CS133.ground_hfs, 3.5, 0.5, 3))
    assert cs == pytest.approx(9_192_631_770.0, abs=1.0)


def test_as13_cs_f4_shift_above_centroid():
    """AS-13: Cs 6S1/2(F=4) sits +4.021 776 4 GHz above centroid +-1 kHz
    (equals Deiglmayr's quoted correction)."""
    shift = hyperfine_shift_hz(CS133.ground_hfs, 3.5, 0.5, 4)
    assert shift == pytest.approx(4.021_776_4 * GHZ, abs=1e3)


def test_as14_hydrogen_limit():
    """AS-14: delta = 0, M -> inf  =>  E_b(n) = h c R_inf / n^2 exactly
    (1e-12 relative, analytic)."""
    hyd = Species(
        name="H-limit", mass_u=math.inf, nuclear_spin=0.5, abundance=1.0,
        e_ion_hz=0.0, e_ion_unc_hz=0.0, ground_n=1, alpha_core_au=0.0,
        series={(0, 0.5): RitzSeries(0, 0.5, 0.0, n_min_mhz=1, n_min_hard=1,
                                     source="analytic hydrogen limit",
                                     confidence="VERIFIED (analytic)")})
    for n in (1, 2, 5, 50):
        assert binding_energy_hz(hyd, n, 0, 0.5) == pytest.approx(
            RYD_HZ / n**2, rel=1e-12)


def test_as15_rb87_rydberg_hfs_a_30s():
    """AS-15: A(30S1/2, Rb-87) = 16.75 GHz * n*^-3 = 0.864 MHz +-30 %
    (Mack/Li scaling, n* = 26.87)."""
    a = rydberg_hfs_A_hz(RB87, 30, 0, 0.5)
    assert a == pytest.approx(0.864 * MHZ, rel=0.30)
    assert n_star(RB87, 30, 0, 0.5) == pytest.approx(26.87, abs=0.01)


# ---------------------------------------------------------------------------
# Integrity guards (audit §3 items 1-5) and rulings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sp, n", [(RB85, 7), (RB87, 7), (CS133, 11)])
def test_hard_floor_raises(sp, n):
    """Audit §3 item 1: below n_min_hard (Rb 8 / Cs 12) -> IntegrityError."""
    with pytest.raises(IntegrityError):
        quantum_defect(sp, n, 0, 0.5)


def test_low_lying_intermediate_levels_unreachable():
    """Audit §3 item 2: Rb 5D / Cs 7P are MISSING data — the Ritz machinery
    refuses to synthesize them (both are below the hard floor)."""
    with pytest.raises(IntegrityError):
        energy_hz(RB85, 5, 2, 2.5)
    with pytest.raises(IntegrityError):
        energy_hz(CS133, 7, 1, 1.5)


def test_mhz_grade_floor_warns():
    """Audit §3 item 1: below n_min_mhz (19/25) a warning is attached."""
    with pytest.warns(UserWarning, match="MHz-grade"):
        quantum_defect(RB85, 15, 0, 0.5)
    with pytest.warns(UserWarning, match="MHz-grade"):
        quantum_defect(CS133, 20, 0, 0.5)


def test_cs_erratum_value_not_encoded():
    """Audit R3 / §3 item 4: the unverifiable Cs erratum E_I digits
    (941 542 216.33 / 216.431 MHz) must not exist in code; only the 2016
    Deiglmayr value ships, with the +0.47 MHz offset as a declared
    systematic."""
    src = pathlib.Path(atom.__file__).read_text(encoding="utf-8")
    assert "216.33" not in src
    assert "216.431" not in src
    assert CS133.e_ion_hz == pytest.approx(941_542_215.86 * MHZ, abs=1.0)
    assert CS133.e_ion_systematic_hz == pytest.approx(0.47 * MHZ)
    assert "erratum" in CS133.e_ion_note


def test_rb85_absolute_systematic_rides_on_the_result():
    """Audit §3 item 5 / R8: every Rb-85 ABSOLUTE optical frequency carries
    the +-40 MHz Lee-1978/Sanguinetti E_I systematic in its UNCERTAINTY
    FIELD — 'not a footnote'.

    Asserting `RB85.e_ion_systematic_hz == 40 MHz` (what this test used to do)
    only proves a number was typed into the dataclass; it passes unchanged
    while every public output drops it on the floor. These assertions run on
    actual results of the three absolute-scale entry points, and check the
    decomposition audit §4 item 10 requires: statistical (the published E_I
    fit uncertainty) kept SEPARATE from the declared systematic.
    """
    e = energy_sourced_hz(RB85, 50, 0, 0.5, ref="ground_centroid")
    assert e.value == pytest.approx(energy_hz(RB85, 50, 0, 0.5,
                                              ref="ground_centroid"))
    assert e.systematic == pytest.approx(40 * MHZ)          # Lee/Sanguinetti
    assert e.statistical == pytest.approx(7 * MHZ)          # Sanguinetti (7)
    assert e.total_uncertainty == pytest.approx(math.hypot(7 * MHZ, 40 * MHZ))
    assert e.as_sourced().uncertainty == pytest.approx(e.total_uncertainty)

    c = coupling_laser_sourced_hz(RB85, "D2", 50, 2, 2.5)
    assert c.systematic == pytest.approx(40 * MHZ)
    # nu_c = E(state) - nu0(D2): BOTH published uncertainties enter.
    assert c.statistical == pytest.approx(math.hypot(7 * MHZ, 14e3))

    lam = coupling_wavelength_sourced_m(RB85, "D2", 50, 2, 2.5)
    assert lam.value == pytest.approx(coupling_wavelength_m(RB85, "D2", 50,
                                                            2, 2.5))
    # dlambda/lambda = dnu/nu, propagated (not re-derived from the value)
    assert lam.total_uncertainty / lam.value == pytest.approx(
        c.total_uncertainty / c.value, rel=1e-12)

    # Cs carries the +0.47 MHz 2025-erratum offset as a systematic (audit R3);
    # Rb-87 (Mack, no tension) carries none.
    assert energy_sourced_hz(CS133, 47, 2, 2.5).systematic == pytest.approx(
        0.47 * MHZ)
    assert energy_sourced_hz(RB87, 50, 0, 0.5).systematic == 0.0


def test_binding_energy_is_immune_to_the_absolute_systematic():
    """Audit R8's other half: E_I cancels in the binding energy, so
    ref='ionization' must carry NEITHER the E_I statistical uncertainty nor
    the systematic. If it did, every Rydberg-Rydberg interval would inherit a
    40 MHz band it demonstrably does not have (test_interval_via_binding_
    energies_not_absolute proves the cancellation is exact)."""
    b = energy_sourced_hz(RB85, 50, 0, 0.5, ref="ionization")
    assert b.statistical == 0.0
    assert b.systematic == 0.0
    assert b.total_uncertainty == 0.0
    assert b.value == pytest.approx(-binding_energy_hz(RB85, 50, 0, 0.5))


def test_validity_flags_ride_on_the_result_not_a_warning():
    """Audit §3 item 1 requires the sub-MHz-grade floor to be 'attached to
    the result'; a warnings.warn is shown once and can be filtered away.
    Below n_min_mhz the flag is in the result's validity_flags (audit §4
    item 8); above it the list is empty."""
    with pytest.warns(UserWarning):
        flagged = energy_sourced_hz(RB85, 15, 0, 0.5)
    assert any("n_min_mhz" in f for f in flagged.validity_flags)
    assert energy_sourced_hz(RB85, 50, 0, 0.5).validity_flags == ()
    # l >= 5 leaves the fitted series entirely -> flagged as well
    assert any("Eq. (1.6)" in f
               for f in energy_sourced_hz(RB85, 50, 5, 5.5).validity_flags)


def test_sourced_confidence_never_over_rates_an_arc_row():
    """Audit §4 item 4: an ARC-transcribed defect row must not be reported as
    VERIFIED. Rb-87 nS1/2 is Mack primary (VERIFIED); Rb-87 nF5/2 is
    'VERIFIED-ARC (digits)', a class the provenance enum does not carry, so
    it must come back one class DOWN, never up — with the verbatim tag kept
    in the source string."""
    assert energy_sourced_hz(RB87, 50, 0, 0.5).confidence is Confidence.VERIFIED
    f = energy_sourced_hz(RB87, 50, 3, 2.5)
    assert f.confidence is Confidence.LITERATURE_RECALL
    assert "VERIFIED-ARC" in f.source


def test_r11_centroid_sign():
    """Ruling R-11: E_I(centroid) = E_I(from F) + dE_hfs(F). For Rb-87,
    dE_hfs(F=1) = -(5/4)A = -4271.68 MHz, so the centroid E_I must equal
    Mack's from-F=1 value MINUS 4271.68 MHz. The printed spec-01 Eq. (1.8)
    sign is the documented error this ruling fixes."""
    shift_f1 = hyperfine_shift_hz(RB87.ground_hfs, 1.5, 0.5, 1)
    assert shift_f1 == pytest.approx(-4_271.676_632 * MHZ, abs=1e3)
    derived = RB87_E_ION_FROM_5S12_F1.value + shift_f1
    assert derived == pytest.approx(RB87.e_ion_hz, abs=0.05 * MHZ)


def test_r15_coupling_wavelengths_computed_at_runtime():
    """Ruling R-15: lambda_c computed from spec-01 energies; the classic
    fixtures 480.0 nm (Rb 5P3/2->50D) and ~509.4 nm (Cs 6P3/2->47D) must
    emerge from the energy machinery, inside the spec 04/06 ranges."""
    lam_rb = coupling_wavelength_m(RB87, "D2", 50, 2, 2.5)
    assert 479e-9 < lam_rb < 484e-9
    assert lam_rb == pytest.approx(480.0e-9, abs=1.0e-9)
    lam_cs = coupling_wavelength_m(CS133, "D2", 47, 2, 2.5)
    assert 508e-9 < lam_cs < 512e-9
    assert lam_cs == pytest.approx(509.4e-9, abs=1.0e-9)


@pytest.mark.parametrize(
    "n, mack_thz, row",
    [
        (19, 612.728_838_1, "AS-04: Mack 2011 Table III"),
        (20, 614.232_154_2, "AS-05: Mack 2011"),
        (21, 615.490_168_7, "AS-06: Mack 2011"),
    ],
)
def test_r15_coupling_laser_against_published_absolute_frequency(n, mack_thz, row):
    """INDEPENDENT anchor for the R-15 machinery: Mack 2011's measured
    5P3/2(F=3) -> nS1/2 frequencies, +-1.5 MHz (the AS-04..06 tolerance).

    coupling_laser_hz references the intermediate level at its hyperfine
    CENTROID, Mack's lines start from F'=3, so the published value must be
    corrected by dE_hfs(5P3/2, F=3) = +193.74 MHz, computed here from Steck's
    A and B through Eq. (1.7). Nothing in the expectation comes from
    rydsim.atom's own energy output: it is a published frequency plus a
    published hyperfine constant. Measured deviations 0.73 / 0.57 / 0.06 MHz.

    This is the check the shipped test suite lacked — the existing R-15 test
    compares against the recalled round numbers 480.0/509.4 nm, which cannot
    distinguish a correct lambda_c from one that is 0.5 % off.
    """
    dhfs_f3 = hyperfine_shift_hz(RB87.d2.upper_hfs, 1.5, 1.5, 3)
    expected = mack_thz * THZ + dhfs_f3
    assert coupling_laser_hz(RB87, "D2", n, 0, 0.5) == pytest.approx(
        expected, abs=1.5 * MHZ), row
    assert coupling_wavelength_m(RB87, "D2", n, 0, 0.5) == pytest.approx(
        C / expected, rel=3e-9), row


def test_r15_lambda_c_series_limit_identity():
    """Structural identity binding lambda_c to the Rydberg series, checked
    against three independently computed public quantities.

    1/lambda_c(n) = (E_I - nu0)/c - (c*R_M)/(c * n*^2), so
        (1/lambda_inf - 1/lambda_c(n)) * n*^2 == rydberg_constant_hz(sp)/c
    exactly, for every n, l, j and both species. A wrong reference level
    (ionization vs ground centroid), a dropped hyperfine centroid, or a sign
    slip in the D-line subtraction all break it; the recalled 480 nm anchor
    does not.
    """
    for sp, line, l, j in ((RB87, "D2", 2, 2.5), (RB85, "D2", 0, 0.5),
                           (CS133, "D2", 2, 2.5), (CS133, "D1", 1, 1.5)):
        dline = sp.d2 if line == "D2" else sp.d1
        inv_lam_inf = (sp.e_ion_hz - dline.nu0_hz) / C
        for n in (25, 40, 60, 100):
            lam = coupling_wavelength_m(sp, line, n, l, j)
            ns = n_star(sp, n, l, j)
            assert (inv_lam_inf - 1.0 / lam) * ns**2 == pytest.approx(
                rydberg_constant_hz(sp) / C, rel=1e-12)


def test_r15_lambda_c_is_state_and_species_dependent():
    """Why lock #10 forbids a hard-coded lambda_c: the AT/Doppler factor
    lambda_c/lambda_p is the single multiplicative link between the measured
    probe-axis splitting and the SI-traceable field, and lambda_c moves with
    BOTH n and species. Pinning the size of the error a fixed 480 nm makes."""
    lam = {n: float(coupling_wavelength_m(RB87, "D2", n, 2, 2.5))
           for n in (20, 30, 50, 100)}
    assert lam[20] > lam[30] > lam[50] > lam[100]        # monotone in n
    assert (lam[20] - 480e-9) / 480e-9 > 0.013           # +1.3 % at n = 20
    assert (lam[100] - 480e-9) / 480e-9 < -0.001         # -0.14 % at n = 100
    # Cs is a different laser entirely. The quantity that multiplies the
    # field is the lock-#10 mismatch lambda_c/lambda_p; leaving a Cs cell at
    # the Rb default 480 nm biases it by exactly the audit's 6.19 %.
    lam_cs = float(coupling_wavelength_m(CS133, "D2", 47, 2, 2.5))
    mismatch_true = lam_cs / CS133.d2.lambda_vac_m
    mismatch_at_480 = 480.0e-9 / CS133.d2.lambda_vac_m
    assert mismatch_true == pytest.approx(0.598013, abs=1e-5)
    assert mismatch_at_480 == pytest.approx(0.563151, abs=1e-5)
    assert mismatch_true / mismatch_at_480 - 1.0 == pytest.approx(0.0619,
                                                                  abs=5e-4)


def test_coupling_laser_refuses_a_target_below_the_intermediate_level():
    """Refuse-to-guess rather than return a negative wavelength. Constructed
    with a species whose E_I is lowered so the Rydberg state falls under the
    D2 upper level; every real state above the hard floor clears the D lines
    by >= 480 THz, so this gate cannot fire on legitimate physics."""
    import dataclasses
    for n in (19, 30, 80):
        assert coupling_laser_hz(RB87, "D2", n, 0, 0.5) > 0.0
    sunk = dataclasses.replace(RB87, e_ion_hz=RB87.d2.nu0_hz - 1e9)
    with pytest.raises(IntegrityError, match="not positive"):
        coupling_laser_hz(sunk, "D2", 50, 0, 0.5)


def test_element_symbol_is_the_single_source_of_the_species_mapping():
    """Audit R10: the species -> element map lives in exactly one place.
    Callers that need cell-side element tables must not slice the isotope
    name; a species with no declared element refuses."""
    assert element_symbol(RB85) == element_symbol(RB87) == "Rb"
    assert element_symbol(CS133) == "Cs"
    nameless = Species(
        name="X1", mass_u=1.0, nuclear_spin=0.5, abundance=1.0,
        e_ion_hz=1.0, e_ion_unc_hz=0.0, ground_n=1, alpha_core_au=0.0,
        series={})
    with pytest.raises(IntegrityError):
        element_symbol(nameless)


def test_rm_anchor_check_catches_regression():
    """Spec 01 §4.1 step 1: a corrupted mass fails the c*R_M anchor check."""
    bad = Species(
        name="Rb87", mass_u=86.0, nuclear_spin=1.5, abundance=0.2783,
        e_ion_hz=RB87.e_ion_hz, e_ion_unc_hz=RB87.e_ion_unc_hz,
        ground_n=5, alpha_core_au=9.076, series=RB87.series)
    with pytest.raises(IntegrityError, match="regression"):
        rydberg_constant_hz(bad)


# ---------------------------------------------------------------------------
# API behavior: validation, vectorization, hyperfine machinery
# ---------------------------------------------------------------------------

def test_invalid_quantum_numbers_raise():
    with pytest.raises(ValueError):
        quantum_defect(RB85, 30, 2, 0.5)        # j != l +- 1/2
    with pytest.raises(ValueError):
        quantum_defect(RB85, 30.5, 0, 0.5)      # non-integer n
    with pytest.raises(ValueError):
        quantum_defect(RB85, 30, -1, 0.5)
    with pytest.raises(ValueError):
        hyperfine_shift_hz(RB87.ground_hfs, 1.5, 0.5, 5)   # F out of range
    with pytest.raises(ValueError):
        probe_transition_hz(RB87, "D2", 2, 5)   # F' outside I +- 3/2


_HFS_LEVELS = [
    ("Rb85 5S1/2", RB85.ground_hfs, 2.5, 0.5),
    ("Rb87 5S1/2", RB87.ground_hfs, 1.5, 0.5),
    ("Cs 6S1/2", CS133.ground_hfs, 3.5, 0.5),
    ("Rb85 5P1/2", RB85.d1.upper_hfs, 2.5, 0.5),
    ("Rb87 5P1/2", RB87.d1.upper_hfs, 1.5, 0.5),
    ("Cs 6P1/2", CS133.d1.upper_hfs, 3.5, 0.5),
    ("Rb85 5P3/2", RB85.d2.upper_hfs, 2.5, 1.5),
    ("Rb87 5P3/2", RB87.d2.upper_hfs, 1.5, 1.5),
    ("Cs 6P3/2", CS133.d2.upper_hfs, 3.5, 1.5),
]


@pytest.mark.parametrize("label, hfs, i_nuc, j_elec", _HFS_LEVELS)
def test_hyperfine_centroid_sum_rule(label, hfs, i_nuc, j_elec):
    """INDEPENDENT analytic identity: the hyperfine shifts are measured from
    the fine-structure CENTROID, so sum_F (2F+1) dE(F) = 0 exactly — for the
    magnetic-dipole and the electric-quadrupole term separately.

    This pins three things no other test covers together: the Casimir
    formula's K and B-ratio algebra, the degeneracy weights, and the F ladder
    itself (drop or add one rung and the weighted sum stops vanishing).
    Checked on all nine hyperfine levels the module ships.
    """
    levels = hyperfine_levels(i_nuc, j_elec)
    shifts = [hyperfine_shift_hz(hfs, i_nuc, j_elec, f) for f in levels]
    weighted = sum((2 * f + 1) * s for f, s in zip(levels, shifts))
    scale = sum((2 * f + 1) * abs(s) for f, s in zip(levels, shifts))
    assert scale > 0.0
    assert abs(weighted) / scale < 1e-14, label


@pytest.mark.parametrize("label, hfs, i_nuc, j_elec", _HFS_LEVELS)
def test_hyperfine_rejects_F_off_the_integer_ladder(label, hfs, i_nuc, j_elec):
    """Refuse-to-guess: F must run in INTEGER steps from |I-J| to I+J
    (angular-momentum addition). The triangle bounds alone admit every
    half-integer inside the interval, and Eq. (1.7) evaluates on them without
    complaint — hyperfine_shift_hz(RB87.ground_hfs, 1.5, 0.5, 1.5) used to
    return -1281.50 MHz for a level Rb-87 does not have.

    Every in-interval value NOT on the ladder must raise; every value ON the
    ladder must be accepted (the fix must not over-refuse).
    """
    levels = hyperfine_levels(i_nuc, j_elec)
    assert levels == tuple(float(abs(i_nuc - j_elec) + k)
                           for k in range(int(2 * min(i_nuc, j_elec)) + 1))
    for f in levels:
        hyperfine_shift_hz(hfs, i_nuc, j_elec, f)          # must not raise
    bogus = [f + 0.5 for f in levels[:-1]]
    assert bogus, label
    for f in bogus:
        with pytest.raises(ValueError, match="integer ladder"):
            hyperfine_shift_hz(hfs, i_nuc, j_elec, f)


def test_probe_transition_rejects_nonexistent_hyperfine_levels():
    """The consequence the refusal prevents: probe_transition_hz(RB87, 'D2',
    2, 2.5) used to return 384.227 966 11 THz — an entirely plausible D2
    number sitting 149.09 MHz (about 25 natural linewidths, well inside the
    Doppler profile) below the real F=2 -> F'=3 line."""
    for sp, line, f, fp in ((RB87, "D2", 2, 2.5), (RB87, "D2", 1.5, 3),
                            (CS133, "D1", 4, 3.5), (RB85, "D2", 2.5, 4)):
        with pytest.raises(ValueError, match="integer ladder"):
            probe_transition_hz(sp, line, f, fp)
    # the real neighbours still work and bracket where the bogus value sat
    assert probe_transition_hz(RB87, "D2", 2, 3) == pytest.approx(
        384.228_115_2 * THZ, abs=0.1 * MHZ)
    assert probe_transition_hz(RB87, "D2", 2, 2) < probe_transition_hz(
        RB87, "D2", 2, 3)


def test_vectorized_matches_scalar():
    ns = np.array([30, 40, 50])
    vec = quantum_defect(RB85, ns, 2, 2.5)
    assert vec.shape == (3,)
    for i, n in enumerate(ns):
        assert vec[i] == pytest.approx(quantum_defect(RB85, int(n), 2, 2.5),
                                       rel=1e-14)
    e_vec = energy_hz(CS133, ns, 0, 0.5)
    assert np.all(e_vec < 0) and np.all(np.diff(e_vec) > 0)


def test_form_b_vectorized_matches_scalar():
    ns = np.array([30, 47, 60])
    vec = quantum_defect(CS133, ns, 2, 2.5)
    for i, n in enumerate(ns):
        assert vec[i] == pytest.approx(quantum_defect(CS133, int(n), 2, 2.5),
                                       rel=1e-14)


def test_high_l_polarization_defect():
    """l >= 5 uses Eq. (1.6); defect is positive, tiny, and decreasing in l."""
    d5 = quantum_defect(RB85, 50, 5, 5.5)
    d6 = quantum_defect(RB85, 50, 6, 6.5)
    g = RB85.series[(4, 3.5)].d0
    assert 0 < d6 < d5 < g


def test_unsourced_rydberg_hfs_is_exactly_zero():
    """Audit §3 item 3: nP3/2/nD/nF Rydberg HFS -> exactly 0 (bounded,
    declared), never a scaled guess."""
    assert rydberg_hfs_A_hz(RB87, 50, 2, 2.5) == 0.0
    assert rydberg_hfs_A_hz(CS133, 50, 1, 1.5) == 0.0
    a = rydberg_hfs_A_hz(RB87, np.array([40, 50]), 3, 2.5)
    assert np.all(a == 0.0)


def test_cs_p12_rydberg_hfs():
    """Spec 01 §2.4: Cs 27p1/2 F=3-4 interval = 4A = 1.2(1) MHz (Deiglmayr)
    reproduced by the scaling coefficient."""
    a = rydberg_hfs_A_hz(CS133, 27, 1, 0.5)
    assert 4 * a == pytest.approx(1.2 * MHZ, rel=0.10)


def test_cs_hfs_coefficient_tags():
    """Audit R7: Cs nS coefficient is UNVERIFIED (+-40 %); Rb-87 nS VERIFIED."""
    from rydsim.provenance import Confidence
    cs = atom.rydberg_hfs_coefficient(CS133, 0, 0.5)
    assert cs.confidence is Confidence.UNVERIFIED
    assert cs.uncertainty / cs.value == pytest.approx(0.40, abs=0.02)
    rb = atom.rydberg_hfs_coefficient(RB87, 0, 0.5)
    assert rb.confidence is Confidence.VERIFIED


def test_dline_gamma_tau_consistency():
    """Steck internal consistency: Gamma = 1/tau to the printed digits."""
    for sp in (RB85, RB87, CS133):
        for line in (sp.d1, sp.d2):
            assert line.gamma_rad_s * line.tau_s == pytest.approx(1.0, rel=5e-4)


def test_probe_transition_cs_f4_fp5():
    """Spec 01 §3.7 derived check: Cs 6S1/2(F=4)->6P3/2(F'=5)
    = 351.721 960 6 THz (verified during authoring)."""
    nu = probe_transition_hz(CS133, "D2", 4, 5)
    assert nu == pytest.approx(351.721_960_6 * THZ, abs=0.2 * MHZ)


def test_interval_via_binding_energies_not_absolute():
    """Audit R8: intervals computed via Eq. (1.9) so E_I genuinely cancels —
    shifting E_I must not move any interval."""
    import dataclasses
    shifted = dataclasses.replace(RB85, e_ion_hz=RB85.e_ion_hz + 100 * MHZ)
    a = transition_hz(RB85, 53, 2, 2.5, 54, 1, 1.5)
    b = transition_hz(shifted, 53, 2, 2.5, 54, 1, 1.5)
    assert a == b


def test_provenance_tags_present():
    """No-fabrication rule: every series row carries source + confidence."""
    for sp in (RB85, RB87, CS133):
        for row in sp.series.values():
            assert row.source and row.confidence
        assert sp.mass_source
        assert sp.d1.source and sp.d2.source
