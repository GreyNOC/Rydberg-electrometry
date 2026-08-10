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
    coupling_wavelength_m,
    energy_hz,
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
from rydsim.constants import RYD_HZ
from rydsim.provenance import IntegrityError

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


def test_rb85_absolute_systematic_declared():
    """Audit §3 item 5: Rb-85 absolute optical scale carries the +-40 MHz
    Lee-1978/Sanguinetti E_I systematic as data, not a footnote."""
    assert RB85.e_ion_systematic_hz == pytest.approx(40 * MHZ)
    assert RB87.e_ion_systematic_hz == 0.0


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
