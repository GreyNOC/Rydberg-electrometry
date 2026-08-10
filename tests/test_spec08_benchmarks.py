"""Spec 08 benchmark suite (docs/spec/08-superheterodyne-noise-sensitivity.md §6).

B1-B14: deterministic arithmetic/convention locks (published-value anchored).
B15/B16: resolvent linear response vs time-domain OBE and vs the
finite-difference slope — the first-principles IBW machinery.
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from rydsim.lindblad import LadderSystem, ibw_3db, linear_response
from rydsim.superhet import (
    envelope_fourier,
    noise_temperature,
    rin_psd,
    shot_noise_psd,
    shot_noise_psd_photocurrent_form,
    sql_nef,
)

EA0 = 8.478353619788951e-30  # e*a0 [C m], CODATA via scipy


def test_b1_envelope_fundamental_deviation():
    u = 0.1
    a1 = envelope_fourier(u, 1)
    assert a1 / u - 1 == pytest.approx(-u**2 / 8, abs=1e-4)


def test_b2_envelope_second_harmonic():
    u = 0.1
    a2 = envelope_fourier(u, 2)
    assert a2 == pytest.approx(-(u**2) / 4, rel=0.05)


def test_b3_shot_noise_jing_probe():
    s = shot_noise_psd(120e-6, 852.347e-9, eta=1.0)
    assert np.sqrt(s) == pytest.approx(7.479e-12, rel=0.005)


def test_b4_shot_noise_identity():
    a = shot_noise_psd(120e-6, 852.347e-9, eta=0.85)
    b = shot_noise_psd_photocurrent_form(120e-6, 852.347e-9, eta=0.85)
    assert a == pytest.approx(b, rel=1e-12)


def test_b5_envelope_only_compression():
    """1-dB compression of the exact envelope fundamental at u = 0.876."""
    us = np.linspace(0.5, 1.0, 501)
    a1 = np.array([envelope_fourier(u, 1) for u in us])
    ratio_db = 20 * np.log10(a1 / us)
    i = np.argmin(np.abs(ratio_db + 1.0))
    assert us[i] == pytest.approx(0.876, abs=0.01)


def test_b6_sql_h_convention_sciadv():
    """3.70 nV/cm/rtHz: mu = 1218 ea0, N = 5.2e5, T2 = 57.8 ns, h-convention."""
    nef = sql_nef(1218 * EA0, 5.2e5, 57.8e-9, convention="h")
    nef_nv_cm = nef / 100 * 1e9
    assert nef_nv_cm == pytest.approx(3.70, rel=0.03)


def test_b7_sql_hbar_pulsed_sciadv():
    """Same 3.70 via hbar-pulsed: tau = 3.83 us, R = 100 Hz."""
    nef = sql_nef(1218 * EA0, 5.2e5, 3.83e-6, convention="hbar",
                  mode="pulsed", rep_rate=100.0)
    assert nef / 100 * 1e9 == pytest.approx(3.70, rel=0.03)


def test_b8_duty_cycle_relation():
    """Continuous hbar SQL with tau = 3.83 us = pulsed/51.1 = 0.0724."""
    nef = sql_nef(1218 * EA0, 5.2e5, 3.83e-6)
    assert nef / 100 * 1e9 == pytest.approx(0.0724, rel=0.03)


def test_b9_achieved_over_sql_ratio():
    assert 10.0 / 3.70 == pytest.approx(2.70, rel=0.01)  # published rounds to 2.6


def test_b10_noise_temperature_sciadv():
    """10.0 nV/cm/rtHz at 36.9 GHz, G = 1.64 -> 828 K (published 830 K)."""
    nef_si = 10.0e-9 * 100  # nV/cm -> V/m
    t_eq, _ = noise_temperature(nef_si, 36.9e9)
    assert t_eq == pytest.approx(828.0, rel=0.02)


def test_b11_noise_temperature_jing():
    nef_si = 55e-9 * 100
    t_eq, nf = noise_temperature(nef_si, 6.94e9)
    assert t_eq == pytest.approx(7.08e5, rel=0.05)
    assert nf == pytest.approx(33.9, abs=0.3)


def test_b12_jing_time_consistency():
    t = (55e-9 / 780e-12) ** 2
    assert 4e3 < t < 6e3


def test_b13_sciadv_time_consistency():
    t = (10.0 / 0.54) ** 2
    assert 250 < t < 450


def test_b14_sql_scaling_exponents():
    d = 1218 * EA0
    base = sql_nef(d, 1e6, 1e-6)
    assert np.log(sql_nef(d, 2e6, 1e-6) / base) / np.log(2) == pytest.approx(-0.5, abs=1e-6)
    # E_min(t) = NEF/sqrt(t) scaling is definitional; check tau exponent too
    assert np.log(sql_nef(d, 1e6, 2e-6) / base) / np.log(2) == pytest.approx(-0.5, abs=1e-6)


def test_rin_corner_model():
    """RIN 1/f knee: at f = f_c the PSD doubles vs the white floor."""
    p = 100e-6
    white = rin_psd(p, -140.0)
    at_corner = rin_psd(p, -140.0, f_hz=1e5, f_corner_hz=1e5)
    assert at_corner == pytest.approx(2 * white, rel=1e-12)
    far_above = rin_psd(p, -140.0, f_hz=1e8, f_corner_hz=1e5)
    assert far_above == pytest.approx(white, rel=0.01)


# ---- B15/B16: the OBE linear-response engine ----

def _fixture_system(omega_rf):
    """4-level cold fixture (Cs-like rates, spec 08 B15)."""
    return LadderSystem(
        omegas=[2 * np.pi * 0.1e6, 2 * np.pi * 4e6, omega_rf],
        deltas=[0.0, 0.0, 0.0],
        decays=[0.0, 2 * np.pi * 5.234e6, 2 * np.pi * 2e3, 2 * np.pi * 2e3],
        dephasings=[0.0, 0.0, 2 * np.pi * 50e3, 2 * np.pi * 50e3],
        transit=2 * np.pi * 30e3,
    )


def test_b16_h0_equals_finite_difference_slope():
    """|H(0)| == d rho_10 / d Omega_RF by central difference, < 0.1%."""
    om = 2 * np.pi * 3e6
    sys = _fixture_system(om)
    h0 = linear_response(sys, 3, np.array([0.0]))[0]

    d_om = om * 1e-4
    sys_p = _fixture_system(om + d_om)
    sys_m = _fixture_system(om - d_om)
    fd = (sys_p.steady_state()[1, 0] - sys_m.steady_state()[1, 0]) / (2 * d_om)
    assert abs(h0 - fd) / abs(fd) < 1e-3


def test_b15_resolvent_vs_time_domain():
    """|H(delta)| and phase from the resolvent match brute-force time-domain
    OBE integration with a modulated RF Rabi frequency (<1%, <1 deg)."""
    om = 2 * np.pi * 3e6
    delta = 2 * np.pi * 0.4e6           # well inside the response bandwidth
    sys = _fixture_system(om)
    h = linear_response(sys, 3, np.array([delta]))[0]

    # time-domain: rho' = L(t) rho with Omega_RF(t) = om + d_om cos(delta t)
    n = 4
    d_om = om * 1e-3
    lv0 = sys.liouvillian()
    dv = np.zeros((n, n), dtype=complex)
    dv[2, 3] = dv[3, 2] = -0.5
    eye = np.eye(n)
    l1 = -1j * (np.kron(dv, eye) - np.kron(eye, dv.T))

    rho0 = sys.steady_state().reshape(n * n)

    def rhs(t, y):
        yc = y[: n * n] + 1j * y[n * n:]
        dy = (lv0 + d_om * np.cos(delta * t) * l1) @ yc
        return np.concatenate([dy.real, dy.imag])

    t_end = 40 * 2 * np.pi / delta
    y0 = np.concatenate([rho0.real, rho0.imag])
    sol = solve_ivp(rhs, (0, t_end), y0, method="BDF",
                    rtol=1e-9, atol=1e-12, dense_output=True)
    # sample the last 20 beat periods on an ENDPOINT-EXCLUSIVE grid: with
    # endpoint=True the discrete mean of e^{i delta t} is nonzero at the
    # 1/n_samples level, leaking the (imaginary-dominant) DC coherence into
    # the demod at the same order as the response itself — the exact bug
    # this test originally exposed.
    ts = np.linspace(t_end / 2, t_end, 4096, endpoint=False)
    ys = sol.sol(ts)
    # row-major vec: index of rho[1,0] is 1*n + 0
    idx = 1 * n + 0
    tr10 = ys[idx] + 1j * ys[n * n + idx]
    tr10 = tr10 - np.mean(tr10)  # remove DC before demodulating
    # demodulate at delta: coefficient of e^{-i delta t}
    demod = 2 * np.mean(tr10 * np.exp(1j * delta * ts))
    # linear response: rho1 per unit dOmega; time-domain gives rho1 * d_om
    h_td = demod / d_om
    assert abs(abs(h_td) - abs(h)) / abs(h) < 0.01
    phase_diff = np.angle(h_td / h)
    assert abs(np.degrees(phase_diff)) < 1.0


def test_ibw_first_principles_magnitude():
    """IBW of the fixture sits in the physically expected MHz-class band
    (Meyer 2020: bandwidth ceiling ~ intermediate scattering rate)."""
    sys = _fixture_system(2 * np.pi * 3e6)
    bw = ibw_3db(sys, 3, delta_max=2 * np.pi * 100e6)
    assert 2 * np.pi * 0.05e6 < bw < 2 * np.pi * 20e6
