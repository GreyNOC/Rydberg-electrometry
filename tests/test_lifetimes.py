"""Benchmarks for rydsim.lifetimes (spec 04 SS6, rows B1-B15 + self-checks).

Every test docstring cites its spec 04 SS6 row (or SS4.3 self-check). The
Beterov anchor values are transcribed below with per-row sources (audit R9:
the anchors live in the TEST file, next to their citation).

Run:  pytest tests/test_lifetimes.py -q
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

import rydsim.lifetimes as lt
from rydsim.atom import CS133, RB85, RB87, coupling_wavelength_m
from rydsim.cell import mean_transverse_speed, number_density_m3, transit_rate
from rydsim.constants import A0, AMU, C, E_CHARGE, EPS0, HBAR, KB
from rydsim.eit import LadderChiParams, doppler_average
from rydsim.lindblad import LadderSystem
from rydsim.lifetimes import (
    CellParams, LaserParams, build_budget, decay_table, default_cell,
    laser_dephasing_rates,
)
from rydsim.provenance import IntegrityError

# ---------------------------------------------------------------------------
# Anchor values (transcribed with sources; audit R9)
# ---------------------------------------------------------------------------

# Beterov, Ryabtsev, Tretyakov, Entin, PRA 79, 052504 (2009) + Erratum
# PRA 80, 059902 (2009); Tables VII (Rb) / VIII (Cs) 0-K and 300-K/77-K
# effective lifetimes, digits as extracted from arXiv:0810.0339v4 in
# spec 04 SS6 [VERIFIED there; re-used here verbatim with this citation].
BETEROV_VII_RB = {
    # (n, l, j): (tau_0 [us], tau_eff(300 K) [us])
    (50, 0, 0.5): (141.31, 65.176),
    (60, 0, 0.5): (252.44, 103.53),
    (50, 1, 1.5): (239.23, 84.74),
    (50, 2, 2.5): (118.21, 65.35),
}
BETEROV_VII_RB_50S_77K = 109.87          # us, same table
BETEROV_VIII_CS = {
    (50, 0, 0.5): (125.64, 60.41),
    (50, 2, 2.5): (70.12, 46.61),
}
# Steck datasheets rev 2.3.4 (8 Aug 2025) [VERIFIED, spec 01/04]:
STECK_TAU_RB_5P32_NS = 26.2348
STECK_TAU_CS_6P32_NS = 30.405
# Spec 07 (O'Sullivan & Stoicheff 1985 / Yerokhin 2016) [VERIFIED]:
# alpha_0(Rb 50S1/2)/h = 50.5 MHz/(V/cm)^2 = 5050 Hz/(V/m)^2 (ruling R-5).
ALPHA0_RB50S_HZ_V2M2 = 50.5e6 / 1.0e4

RB87_MASS_KG = RB87.mass_u * AMU
CS_MASS_KG = CS133.mass_u * AMU


# ---------------------------------------------------------------------------
# NIST low-n table self-checks (no-fabrication rule)
# ---------------------------------------------------------------------------

class TestNistTable:
    def test_first_p_rows_match_steck_dlines(self):
        """NIST-vs-Steck cross-check of the sub-floor level table: the first
        P term energies must reproduce the Steck D-line frequencies (both
        datasets VERIFIED; agreement bounds transcription errors). Rb NIST
        digits are older -> 0.5 GHz tolerance; Cs is digit-modern (kHz)."""
        for sp, tol_hz in ((RB87, 5e8), (CS133, 1e5)):
            ng = sp.ground_n
            for line, jj in (("D1", 0.5), ("D2", 1.5)):
                dline = sp.d1 if line == "D1" else sp.d2
                nu_nist = (lt.binding_hz_any(sp, ng, 0, 0.5)
                           - lt.binding_hz_any(sp, ng, 1, jj))
                assert abs(nu_nist - dline.nu0_hz) < tol_hz

    def test_nist_limits_match_spec01_e_ion(self):
        """NIST ionization limits vs the spec-01 E_I values (Cs must be
        digit-identical to Deiglmayr 2016 — audit item 9)."""
        inv_cm = C * 100.0
        assert abs(lt.NIST_ION_LIMIT_CM["Cs"] * inv_cm
                   - CS133.e_ion_hz) < 1e6
        assert abs(lt.NIST_ION_LIMIT_CM["Rb"] * inv_cm
                   - RB87.e_ion_hz) < 0.02 * inv_cm

    def test_missing_row_refuses(self):
        """Audit SS3 item 2 / conventions SS6 gap 3: a sub-floor state with
        no NIST data row raises IntegrityError, never a Ritz guess."""
        with pytest.raises(IntegrityError, match="NIST"):
            lt.binding_hz_any(CS133, 5, 4, 4.5)

    def test_ground_state_binding_is_e_ion(self):
        assert lt.binding_hz_any(RB87, 5, 0, 0.5) == RB87.e_ion_hz


# ---------------------------------------------------------------------------
# B14 / S1 — algebraic identity of the Einstein-rate helper
# ---------------------------------------------------------------------------

class TestEinsteinRate:
    def test_b14_sixj_reduction_identity(self):
        """Spec 04 SS6 B14 / self-check S1: sum_{J'} A(nLJ -> n'L'J') at
        equal omega and R reduces to the L-basis rate
        (omega^3 e^2 a0^2 R^2 / 3 pi eps0 hbar c^3) L_max/(2L+1), to 1e-12
        relative — catches 6-j and degeneracy-factor bugs."""
        omega = 2.0 * math.pi * 3.8e14
        for (l, j, lp) in [(0, 0.5, 1), (1, 1.5, 0), (1, 0.5, 2),
                           (2, 2.5, 1), (2, 1.5, 3), (3, 3.5, 2)]:
            total = sum(
                lt.einstein_rate(omega, 1.0, l, j, lp, jp)
                for jp in (lp - 0.5, lp + 0.5)
                if jp > 0 and abs(j - jp) <= 1.0)
            l_basis = (omega**3 * (E_CHARGE * A0)**2
                       / (3.0 * math.pi * EPS0 * HBAR * C**3)
                       * max(l, lp) / (2.0 * l + 1.0))
            assert abs(total / l_basis - 1.0) < 1e-12

    def test_rate_direction_symmetry_carries_degeneracy(self):
        """Spec 04 SS2.2: the upward coefficient equals the downward A
        times (2j_up + 1)/(2j_down + 1) (the g'/g asymmetry)."""
        omega = 2.0 * math.pi * 1.0e12
        down = lt.einstein_rate(omega, 10.0, 2, 2.5, 1, 1.5)   # D5/2 -> P3/2
        up = lt.einstein_rate(omega, 10.0, 1, 1.5, 2, 2.5)     # P3/2 -> D5/2
        assert abs(up / down - (2 * 2.5 + 1) / (2 * 1.5 + 1)) < 1e-12


# ---------------------------------------------------------------------------
# B1 / B1b — D-line lifetimes tie the A machinery to Steck
# ---------------------------------------------------------------------------

class TestDLineLifetimes:
    def test_b1_rb_5p32(self):
        """Spec 04 SS6 B1: Rb 5P3/2 lifetime from the A-sum machinery =
        26.2348 ns (Steck Rb87 rev 2.3.4), tol 3%. The D-line radial
        element is the experiment-derived Steck value (spec 02 SS7.1
        mandate) — this row certifies formula, angular chain and energies."""
        tau = lt.radiative_lifetime_sum(RB87, 5, 1, 1.5)
        assert abs(tau * 1e9 / STECK_TAU_RB_5P32_NS - 1.0) < 0.03

    def test_b1b_cs_6p32(self):
        """Spec 04 SS6 B1b: Cs 6P3/2 = 30.405 ns (Steck Cs rev 2.3.4 —
        ruling R-20/R-25 supersedes the older 30.473 ns), tol 3% (covers
        the Patterson 2015 30.462(46) ns tension)."""
        tau = lt.radiative_lifetime_sum(CS133, 6, 1, 1.5)
        assert abs(tau * 1e9 / STECK_TAU_CS_6P32_NS - 1.0) < 0.03


# ---------------------------------------------------------------------------
# B2-B8b — Beterov lifetime anchors (fit + sum paths)
# ---------------------------------------------------------------------------

class TestBeterovAnchors:
    def test_b2_rb50s_tau0_fit(self):
        """Spec 04 SS6 B2 (fit): Beterov Table II fit vs Table VII value
        141.31 us, tol 0.5% (the fit itself prints 141.26 us)."""
        tau = float(lt.radiative_lifetime_fit(RB87, 50, 0, 0.5))
        assert abs(tau * 1e6 / 141.31 - 1.0) < 0.005

    def test_b3_rb50s_taueff300_fit(self):
        """Spec 04 SS6 B3 (analytic): Eq. 16 gives 64.4 us vs the tabulated
        65.176 us — tol 2%."""
        tau = float(lt.effective_lifetime_fit(RB87, 50, 0, 0.5, 300.0))
        assert abs(tau * 1e6 / 65.176 - 1.0) < 0.02

    @pytest.mark.parametrize("state,anchors", sorted(BETEROV_VII_RB.items()))
    def test_b2_b5_b6_b7_rb_sum(self, state, anchors):
        """Spec 04 SS6 B2/B5/B6/B7 (sum paths, tol 10%): Rb 50S/60S/50P3/2/
        50D5/2 tau_0 and tau_eff(300 K) vs Beterov Table VII. The 10% band
        is dominated by the documented low-n' matrix-element bottleneck
        (spec 04 SS4.1 item 3); the A-vs-B spread ships as uncertainty."""
        n, l, j = state
        tau0_ref, taueff_ref = anchors
        table = decay_table(RB87, n, l, j)
        assert abs(1e6 / table.gamma0_rad_s / tau0_ref - 1.0) < 0.10
        assert abs(table.effective_lifetime(300.0) * 1e6
                   / taueff_ref - 1.0) < 0.10

    def test_b4_rb50s_taueff77_sum(self):
        """Spec 04 SS6 B4: Rb 50S tau_eff(77 K) = 109.87 us, tol 10% sum."""
        tau = lt.effective_lifetime_sum(RB87, 50, 0, 0.5, 77.0)
        assert abs(tau * 1e6 / BETEROV_VII_RB_50S_77K - 1.0) < 0.10

    @pytest.mark.parametrize("state,anchors", sorted(BETEROV_VIII_CS.items()))
    def test_b8_b8b_cs_sum(self, state, anchors):
        """Spec 04 SS6 B8/B8b (sum, tol 10%): Cs 50S and 50D5/2 vs Beterov
        Table VIII. n_eff uses the spec-01 Form-B machinery for Cs S/D5/2
        (ruling R-14)."""
        n, l, j = state
        tau0_ref, taueff_ref = anchors
        table = decay_table(CS133, n, l, j)
        assert abs(1e6 / table.gamma0_rad_s / tau0_ref - 1.0) < 0.10
        assert abs(table.effective_lifetime(300.0) * 1e6
                   / taueff_ref - 1.0) < 0.10

    def test_b8_cs50s_analytic_paths(self):
        """Spec 04 SS6 B8 (analytic, tol 1%): Cs 50S fit paths give
        125.68 / 59.8 us."""
        assert abs(float(lt.radiative_lifetime_fit(CS133, 50, 0, 0.5))
                   * 1e6 / 125.68 - 1.0) < 0.01
        assert abs(float(lt.effective_lifetime_fit(CS133, 50, 0, 0.5, 300.0))
                   * 1e6 / 59.8 - 1.0) < 0.01

    def test_s2_s3_crosschecks_rb50s(self):
        """Self-checks S2/S3 (spec 04 SS4.3): A-sum vs Beterov fit within
        10% on tau_0 and 20% on Gamma_BBR — the first-class validation of
        the whole dipole chain. ``require()`` raises on breach."""
        lt.lifetime_crosscheck(RB87, 50, 0, 0.5).require()
        lt.bbr_crosscheck(RB87, 50, 0, 0.5, 300.0).require()

    def test_grid_step_insensitivity(self):
        """Numerical guard: tau_0(Rb 50S) at the default step and at h
        doubled agree to < 0.5% (O(h^4) Numerov — a breach means a grid
        regression, spec 02 SS4.1)."""
        t1 = lt.radiative_lifetime_sum(RB87, 50, 0, 0.5)
        t2 = lt.radiative_lifetime_sum(RB87, 50, 0, 0.5, h=2e-3)
        assert abs(t2 / t1 - 1.0) < 0.005


# ---------------------------------------------------------------------------
# B9 / B9b — BBR depopulation
# ---------------------------------------------------------------------------

class TestBBR:
    def test_b9_rb50s_bbr_rate(self):
        """Spec 04 SS6 B9: Rb 50S Gamma_BBR(300 K) = 8.27e3 1/s
        (= 1/65.176us - 1/141.31us, Beterov Table VII), 15% sum; the
        Eq. 14 fit prints 8.45e3 (within ~2.2% of the table value)."""
        ref = 1.0 / 65.176e-6 - 1.0 / 141.31e-6
        assert abs(lt.bbr_rate_sum(RB87, 50, 0, 0.5, 300.0) / ref - 1) < 0.15
        assert abs(float(lt.bbr_rate_fit(RB87, 50, 0, 0.5, 300.0))
                   / ref - 1.0) < 0.03

    def test_b9b_farley_wing_band(self):
        """Spec 04 SS6 B9b / self-check S4: Gamma_BBR/Gamma_FW in
        [0.7, 1.2] for Rb nS at 300 K. Full sum at n = 50 (cached table);
        the Beterov fit (validated at 2% against the sum in B9) carries the
        band check to n = 40, 55, 70 — the band is 30x wider than the fit
        error. Source: Farley & Wing, PRA 23, 2397 (1981)."""
        r50 = (lt.bbr_rate_sum(RB87, 50, 0, 0.5, 300.0)
               / lt.bbr_rate_farley_wing(RB87, 50, 0, 0.5, 300.0))
        assert 0.7 < r50 < 1.2
        for n in (40, 55, 70):
            r = (float(lt.bbr_rate_fit(RB87, n, 0, 0.5, 300.0))
                 / lt.bbr_rate_farley_wing(RB87, n, 0, 0.5, 300.0))
            assert 0.7 < r < 1.2, f"n={n}: ratio {r:.3f}"

    def test_farley_wing_check_value(self):
        """Spec 04 SS2.2 check value: Gamma_FW = 2.034e7 (T/300)/n_eff^2
        1/s — the runtime-computed constant must reproduce the printed
        rounding to 0.1% (lock #15: constants never typed)."""
        from rydsim.atom import n_star
        ne = float(n_star(RB87, 50, 0, 0.5))
        got = lt.bbr_rate_farley_wing(RB87, 50, 0, 0.5, 300.0) * ne**2
        assert abs(got / 2.034e7 - 1.0) < 1e-3

    def test_bbr_tail_guard_raises_when_window_too_small(self):
        """SS4.2 truncation rule: an undersized upward window must raise,
        never return a silently truncated rate. dn_up = 3 leaves nothing
        past the n' = n +- 1,2 transfer peak, so the tail is not even
        estimable — the refusal must say so rather than quote a number."""
        with pytest.raises(IntegrityError, match="unconverged"):
            decay_table(RB87, 45, 0, 0.5, dn_up=3).bbr_rate(300.0)
        tail = decay_table(RB87, 45, 0, 0.5, dn_up=3).bbr_tail(300.0)
        assert not tail.estimable and not tail.converged
        assert math.isinf(tail.tail_bound_rel)


# ---------------------------------------------------------------------------
# BBR truncation: the tail guard must estimate the OMITTED tail
# (audit: "BBR tail guard is not a truncation estimate")
# ---------------------------------------------------------------------------

def _truncate(table, n_cut):
    """A view of ``table`` with the upward window cut at n' <= n_cut."""
    return replace(table,
                   channels=tuple(c for c in table.channels if c.n <= n_cut),
                   n_top=n_cut)


class TestBBRTruncation:
    def test_tail_estimate_predicts_the_measured_truncation(self):
        """INDEPENDENT check of the extrapolation: build one wide table
        (Rb 30S, n' <= 70) and compare the tail estimate made from a NARROW
        view (n' <= 45) against the rate that the extra channels 46..70
        actually add. The estimate must track the measured increment to
        within a factor of 2, the p = 3 bound must exceed it, and the
        completed rate Gamma(cut) * (1 + tail) must be stable across cuts —
        that stability is what makes 'the omitted tail' a measured quantity
        rather than a heuristic.

        The replaced guard could not do this: it summed the weight of the
        three highest-n' INCLUDED channels, which for Rb 50S reads 8.1e-4
        at dn_up = 25 (true truncation 0.6 %) and 1.2e-3 at dn_up = 20
        (true truncation 0.8 %) — i.e. it grew while the error shrank.
        """
        wide = decay_table(RB87, 30, 0, 0.5, dn_up=40)
        narrow = _truncate(wide, 45)
        t_n = narrow.bbr_tail(300.0)
        t_w = wide.bbr_tail(300.0)
        measured = (t_w.rate_s - t_n.rate_s) / t_n.rate_s
        assert measured > 0.0
        assert 0.5 * measured <= t_n.tail_estimate_rel <= 2.0 * measured
        assert t_n.tail_bound_rel > measured
        completed_n = t_n.rate_s * (1.0 + t_n.tail_estimate_rel)
        completed_w = t_w.rate_s * (1.0 + t_w.tail_estimate_rel)
        assert abs(completed_n / completed_w - 1.0) < 0.01
        # monotone in the window size, as a truncation error must be
        assert t_w.tail_estimate_rel < t_n.tail_estimate_rel
        assert t_w.tail_bound_rel < t_n.tail_bound_rel

    def test_bound_is_above_the_estimate_and_uses_the_asymptotic_slope(self):
        """The p = 3 continuation is the slowest decay the discrete tail can
        have (omega -> E_bind, nbar -> const, |R|^2 ~ n'^-3), so it must
        bound the measured-slope estimate; the measured slopes themselves
        must sit at or above 3."""
        tail = decay_table(RB87, 50, 0, 0.5).bbr_tail(300.0)
        assert tail.tail_bound_rel >= tail.tail_estimate_rel
        assert tail.slopes and all(p >= 3.0 for _, p in tail.slopes)

    @pytest.mark.parametrize("state", [(50, 1, 0.5), (30, 1, 1.5)])
    def test_cs_np_series_converges_at_the_shipped_window(self, state):
        """Coverage hole closed (audit): NO Cs nP sum was exercised, and the
        old guard refused the whole series at the shipped dn_up = 25 —
        including Cs 50P1/2, whose Gamma_BBR it declined to return agrees
        with the INDEPENDENT Beterov Eq. 14 analytic fit to well inside the
        S3 20 % band. Also covers the tau_eff round trip."""
        n, l, j = state
        table = decay_table(CS133, n, l, j)
        tail = table.bbr_tail(300.0)
        assert tail.converged, (tail.tail_estimate_rel, tail.tail_bound_rel)
        fit = float(lt.bbr_rate_fit(CS133, n, l, j, 300.0))
        assert abs(tail.rate_s / fit - 1.0) < 0.20
        assert table.effective_lifetime(300.0) > 0.0

    def test_guard_still_fires_when_the_truncation_is_real(self):
        """The guard must remain a guard. On a view of the Rb 30S table cut
        at n' = n + 10 the truncation measured against the n' = n + 40 table
        is 1.0 %, so a 0.5 % tolerance MUST refuse (and the refusal is
        deserved: measured > tol), while the shipped 5 % tolerance passes.
        A window with fewer than five channels past the transfer peak is
        refused outright — the tail is not estimable there at all."""
        wide = decay_table(RB87, 30, 0, 0.5, dn_up=40)
        short = _truncate(wide, 40)
        measured = (wide.bbr_tail(300.0).rate_s
                    / short.bbr_tail(300.0).rate_s - 1.0)
        assert measured > 0.005          # the refusal below is deserved
        with pytest.raises(IntegrityError, match="unconverged"):
            short.bbr_rate(300.0, tail_tol=0.005)
        assert short.bbr_rate(300.0) > 0.0
        assert not _truncate(wide, 38).bbr_tail(300.0).estimable
        with pytest.raises(IntegrityError, match="unconverged"):
            _truncate(wide, 38).bbr_rate(300.0)


# ---------------------------------------------------------------------------
# Weighted A-vs-B consensus gate
# (audit: "per-channel consensus veto makes routine states uncomputable")
# ---------------------------------------------------------------------------

def _chan(n, l, j, omega_hz, rate_A, spread, *, above=True, ceiling=1e-4):
    """A DecayChannel with rate_B implied by the requested radial spread
    (rate ~ R^2, so rate_B = rate_A (1 - spread)^2)."""
    return lt.DecayChannel(
        n=n, l=l, j=j, omega_hz=omega_hz, rate_A=rate_A,
        rate_B=rate_A * (1.0 - spread) ** 2, radial_a0=1.0,
        spread_rel=spread, above_floor=above, ab_ceiling=ceiling)


class TestConsensusGate:
    """The spec-02 ceiling must be enforced where the rate lives."""

    def test_states_the_unweighted_veto_made_uncomputable(self):
        """Reproduces the audit's failing set. Every one of these states was
        refused by the per-channel veto over a channel carrying 1e-11..1e-8
        of the total rate; each must now compute, and the INDEPENDENT
        Beterov Table-II fit must land inside the shipped tau_0 interval.
        Rb 30D5/2 and Rb 58D3/2 are mainstream electrometry states."""
        for sp, n, l, j in ((RB87, 30, 2, 2.5), (RB87, 24, 1, 1.5),
                            (CS133, 20, 2, 1.5)):
            table = decay_table(sp, n, l, j)
            fit = float(lt.radiative_lifetime_fit(sp, n, l, j))
            lo, hi = table.tau0_interval()
            assert lo <= fit <= hi, (sp.name, n, l, j, lo, fit, hi)
            assert table.consensus_share_floor > 0.0

    def test_exempted_channels_are_recorded_and_negligible(self):
        """The exemption is provenance, not silence (audit SS4 item 5): the
        skipped channels are listed with their share, and Rb 30D5/2's
        vetoing channel — the one that used to block the whole lifetime —
        is there carrying < 1e-9 of the rate."""
        table = decay_table(RB87, 30, 2, 2.5)
        assert table.consensus_exempt
        by_state = {(n, l, j): (share, spread)
                    for n, l, j, share, spread in table.consensus_exempt}
        share, spread = by_state[(31, 3, 2.5)]
        assert spread > 1e-3            # would have vetoed the whole table
        assert share < 1e-9             # while carrying nothing
        assert all(s < table.consensus_share_floor
                   for _, _, _, s, _ in table.consensus_exempt)

    def test_gate_fires_on_a_channel_that_carries_rate(self):
        """A ceiling breach on a DOMINANT channel is still a hard refusal —
        the weighting must not have disarmed the tripwire."""
        good = [_chan(6, 1, 1.5, 4.0e14, 1.0e7, 1e-6),
                _chan(7, 1, 1.5, 4.5e14, 1.0e5, 1e-6)]
        lt._consensus_audit(RB87, 50, 0, 0.5, good)      # no raise
        bad = [_chan(6, 1, 1.5, 4.0e14, 1.0e7, 1e-2),
               _chan(7, 1, 1.5, 4.5e14, 1.0e5, 1e-6)]
        with pytest.raises(IntegrityError, match="consensus failed"):
            lt._consensus_audit(RB87, 50, 0, 0.5, bad)

    def test_exemption_premise_is_itself_gated(self):
        """Many individually-tiny breaches must not add up to a material
        amount of rate: once the exempted set passes 1 % of the total the
        engine refuses instead of shipping it."""
        chans = [_chan(6, 1, 1.5, 4.0e14, 1.0e7, 1e-6)]
        # 300 breaching channels at ~5e-5 share each -> ~1.5 % in total
        chans += [_chan(10 + k, 1, 1.5, 4.0e14, 5.0e2, 0.5)
                  for k in range(300)]
        with pytest.raises(IntegrityError, match="exemption invalid"):
            lt._consensus_audit(RB87, 50, 0, 0.5, chans)

    def test_ceiling_is_the_shared_gate_not_a_transcription(self):
        """R10 reconciliation: this module used to TRANSCRIBE the spec 02 §6
        ceilings, and had already diverged from radial's copy (which applied
        the 1e-4 row to every large low-l pair, where this one scoped it to
        S<->P). Two gates disagreeing about the same physics is the
        constant-forking hazard in code form.

        The better-sourced rule — 1e-4 is S<->P-measured (B8/B11), so
        applying it to P<->D / D<->F is an unsourced extrapolation — was
        adopted into the shared gate. This test asserts the DELEGATION
        (agreement with radial across every regime) rather than the constant
        values, so the two can no longer drift apart. Asserting the numbers
        here would only re-pin this module's copy of them.
        """
        from rydsim.radial import _ab_ceiling as shared

        for nu_min, me_abs, l1, l2 in (
                (50.0, 1000.0, 0, 1),    # S<->P, large  -> the measured row
                (50.0, 1000.0, 2, 3),    # D<->F         -> unsourced for 1e-4
                (50.0, 1000.0, 1, 2),    # P<->D         -> ditto
                (50.0, 1.0, 0, 1),       # cancellation-suppressed
                (9.0, 1000.0, 0, 1),     # below the Ritz floor
                (30.0, 60.0, 0, 1), (20.0, 5.0, 3, 4), (16.0, 900.0, 1, 2)):
            assert lt._ab_ceiling(nu_min, me_abs, l1, l2) == \
                shared(nu_min, me_abs, l1, l2), (nu_min, me_abs, l1, l2)

        # and the scoping itself is real: same nu and |ME|, different class
        assert shared(50.0, 1000.0, 0, 1) < shared(50.0, 1000.0, 2, 3)


# ---------------------------------------------------------------------------
# Shipped tau_0 uncertainty vs the low-n' systematic
# (audit: "A-vs-B spread does not cover the reproducible bias")
# ---------------------------------------------------------------------------

class TestTau0Uncertainty:
    @pytest.mark.parametrize("state,anchors", sorted(BETEROV_VII_RB.items()))
    def test_rb_interval_covers_beterov(self, state, anchors):
        """The shipped tau_0 interval must contain the accepted value. The
        A-vs-B spread alone does NOT: for Rb 50S it is 5.5 % against a
        reproducible -8.7 % offset from Beterov Table VII, so the stated
        band excluded 141.31 us. The low-n' model-potential systematic
        (spec 02 SS7.1/B15, 5..10 % on radial elements to sub-floor
        partners, propagated as (1+b)^2 - 1 through the sub-floor share of
        Gamma_0) is what closes it."""
        n, l, j = state
        tau0_ref = anchors[0] * 1e-6
        table = decay_table(RB87, n, l, j)
        lo, hi = table.tau0_interval()
        assert lo <= tau0_ref <= hi, (lo, tau0_ref, hi)

    @pytest.mark.parametrize("state,anchors", sorted(BETEROV_VIII_CS.items()))
    def test_cs_interval_covers_beterov(self, state, anchors):
        """Same for Cs against Beterov Table VIII."""
        n, l, j = state
        tau0_ref = anchors[0] * 1e-6
        lo, hi = decay_table(CS133, n, l, j).tau0_interval()
        assert lo <= tau0_ref <= hi, (lo, tau0_ref, hi)

    def test_spread_alone_does_not_cover_rb50s(self):
        """The defect this interval fixes, stated as a test: the numerical
        spread by itself excludes the Beterov value, so it must never be
        shipped as THE uncertainty of tau_0."""
        table = decay_table(RB87, 50, 0, 0.5)
        tau0 = 1.0 / table.gamma0_rad_s
        s = table.gamma0_spread_rel
        assert not (tau0 * (1 - s) <= 141.31e-6 <= tau0 * (1 + s))
        assert table.gamma0_bias_band_rel[1] > s

    def test_systematic_scales_with_the_subfloor_share(self):
        """The declared band is derived, not typed: it is the sub-floor
        share of Gamma_0 times the spec-02 radial bias band propagated
        through rate ~ R^2. Recomputed here from the channel table."""
        table = decay_table(RB87, 50, 0, 0.5)
        f = sum(c.rate_A for c in table.channels
                if c.downward and c.sub_floor_partner
                and not c.experimental_radial) / table.gamma0_rad_s
        assert table.subfloor_gamma0_fraction == pytest.approx(f, rel=1e-12)
        lo, hi = table.gamma0_bias_band_rel
        assert lo == pytest.approx(f * (1.05**2 - 1.0), rel=1e-12)
        assert hi == pytest.approx(f * (1.10**2 - 1.0), rel=1e-12)
        assert 0.0 < f < 1.0

    @pytest.mark.parametrize("state", [(20, 1, 1.5), (20, 2, 1.5)])
    def test_s2_window_low_end_is_covered_not_untested(self, state):
        """Spec 04 SS4.3 S2 declares a 10 % band for 20 <= n <= 70, and the
        audit measured Rb nP1/2, nP3/2, nD3/2, nD5/2 BREACHING it at the low
        end (n = 20: +11.8 %, +12.4 %, +13.0 %, +10.1 %) with no test
        covering those n. These rows make the low end visible: the sum-vs-
        fit deviation is asserted against the (independent) Beterov fit at
        the size the audit measured, and the shipped tau_0 interval must
        still contain the fit."""
        n, l, j = state
        table = decay_table(RB87, n, l, j)
        tau_sum = 1.0 / table.gamma0_rad_s
        fit = float(lt.radiative_lifetime_fit(RB87, n, l, j))
        dev = tau_sum / fit - 1.0
        assert 0.08 < dev < 0.16, f"S2 low-end deviation moved: {dev:+.3%}"
        lo, hi = table.tau0_interval()
        assert lo <= fit <= hi

    def test_cs_np1_2_tension_is_declared_not_absorbed(self):
        """The one series the declared systematic does NOT explain, recorded
        rather than papered over. Over the whole in-scope grid (Rb/Cs x
        S,P1/2,P3/2,D3/2,D5/2 x n = 20,30,50,70) the Beterov fit lies inside
        tau0_interval() for 36 of 40 states; the exception is Cs nP1/2,
        where the sum sits +23..28 % above the fit with NO n dependence
        (+27.8 % at n=20, +23.2 % at n=70) — far more than the 5..10 %
        spec-02 radial bias can produce, and in the opposite direction from
        the Rb nS case. Cs nP3/2 agrees to ~1 %, and the sums put
        tau(P1/2) > tau(P3/2) while Beterov's Table-II coefficients
        (2.9921 vs 3.2849) put them the other way round — the same nP fit
        tension spec 04 SS4.1 item 3 already flags. Escalated as a spec-09
        register item, NOT absorbed into the code's uncertainty."""
        for n in (20, 70):
            table = decay_table(CS133, n, 1, 0.5)
            fit = float(lt.radiative_lifetime_fit(CS133, n, 1, 0.5))
            dev = 1.0 / table.gamma0_rad_s / fit - 1.0
            assert 0.20 < dev < 0.32, f"Cs {n}P1/2 tension moved: {dev:+.3%}"
            assert table.tau0_interval()[0] > fit      # still not covered


# ---------------------------------------------------------------------------
# B13 — direct BBR photoionization (Beterov NJP Eq. 27)
# ---------------------------------------------------------------------------

class TestBBRIonization:
    def test_b13_rb50s(self):
        """Spec 04 SS6 B13: W_BBR_ion(Rb 50S, 300 K) ~ 1.5e2 1/s, tol 50%
        (assignment worked value 148.7 1/s). ~1% of total depopulation."""
        w = lt.bbr_ionization_rate(RB87, 50, 0, 0.5, 300.0)
        assert abs(w / 148.7 - 1.0) < 0.50
        total = 1.0 / lt.effective_lifetime_fit(RB87, 50, 0, 0.5, 300.0)
        assert w / total < 0.02

    def test_l_coverage_and_refusals(self):
        """NJP Table 2 covers S/P/D only; F states refuse (no A_L, no
        mu_F - mu_G difference) — refuse-to-guess."""
        assert lt.bbr_ionization_rate(RB87, 50, 1, 1.5, 300.0) > 0.0
        assert lt.bbr_ionization_rate(CS133, 50, 2, 2.5, 300.0) > 0.0
        with pytest.raises(IntegrityError):
            lt.bbr_ionization_rate(RB87, 50, 3, 2.5, 300.0)


# ---------------------------------------------------------------------------
# Fit-path validity gates (audit SS3 item 14)
# ---------------------------------------------------------------------------

class TestFitValidity:
    @pytest.mark.parametrize("n", [10, 14, 81, 90])
    def test_beterov_window_refused(self, n):
        with pytest.raises(IntegrityError, match="15 <= n <= 80"):
            lt.radiative_lifetime_fit(RB87, n, 0, 0.5)
        with pytest.raises(IntegrityError, match="15 <= n <= 80"):
            lt.bbr_rate_fit(RB87, n, 0, 0.5, 300.0)

    def test_unfitted_series_refused(self):
        """No Beterov fit exists for F states — IntegrityError, not an
        extrapolation (spec 04 SS3.1)."""
        with pytest.raises(IntegrityError, match="nS/nP/nD"):
            lt.radiative_lifetime_fit(RB87, 50, 3, 3.5)


# ---------------------------------------------------------------------------
# B11 / B11b — resonant self-broadening (R-6, R-21)
# ---------------------------------------------------------------------------

class TestSelfBroadening:
    def test_b11_closed_form(self):
        """Spec 04 SS6 B11 / self-check S5: the closed form (Weller Eqs.
        2-3, computed at runtime from Steck Gamma and lambda) reproduces
        beta/2pi = 1.03e-7 Hz cm^3 (Rb D2) and 1.16e-7 (Cs D2) to 3%.
        Internal storage is SI Hz m^3 (ruling R-21): 1e-6 x Hz cm^3."""
        assert abs(lt.self_broadening_beta_hz_m3(RB87, "D2").value
                   / 1.03e-13 - 1.0) < 0.03
        assert abs(lt.self_broadening_beta_hz_m3(CS133, "D2").value
                   / 1.16e-13 - 1.0) < 0.03
        assert abs(lt.self_broadening_beta_hz_m3(RB87, "D1").value
                   / 0.73e-13 - 1.0) < 0.03

    def test_b11b_theory_within_measured_1sigma(self):
        """Spec 04 SS6 B11b: theory within 1 sigma of Kondo 2006
        1.10(17)e-7 Hz cm^3 (ruling R-6)."""
        meas = lt.SELF_BROADENING_MEASURED[("Rb", "D2")]
        theo = lt.self_broadening_beta_hz_m3(RB87, "D2").value
        assert abs(theo - meas.value) < meas.uncertainty

    def test_fwhm_and_magnitudes(self):
        """Spec 04 SS2.3.5(i) magnitudes: 1e10 cm^-3 -> ~1.1 kHz
        (negligible); 2e13 cm^-3 -> ~2.2 MHz (dominant)."""
        lo = lt.self_broadening_fwhm_hz(RB87, "D2", 1e16)
        hi = lt.self_broadening_fwhm_hz(RB87, "D2", 2e19)
        assert 0.8e3 < lo < 1.4e3
        assert 1.6e6 < hi < 2.8e6

    def test_measured_missing_refuses(self):
        with pytest.raises(IntegrityError, match="measured"):
            lt.self_broadening_beta_hz_m3(CS133, "D1", which="measured")


# ---------------------------------------------------------------------------
# B12 — transit-time broadening (ruling R-3, single implementation)
# ---------------------------------------------------------------------------

class TestTransit:
    def test_b12_rb_300k(self):
        """Spec 04 SS6 B12: transit FWHM 106 kHz for Rb-87, 300 K,
        w = 0.75 mm (tol 20%, convention accuracy)."""
        f = lt.transit_fwhm_hz(RB87_MASS_KG, 300.0, 0.75e-3)
        assert abs(f / 106e3 - 1.0) < 0.20

    def test_transit_prefactor_lock(self):
        """Conventions SS8 new-test (i) / ruling R-3: gamma_t w0 / <v_perp>
        = sqrt(2 ln 2) exactly, from the ONE implementation
        (rydsim.cell.transit_rate; audit R17). Also the numeric anchors
        106 kHz @ 0.75 mm and 79.6 kHz @ 1 mm."""
        v = mean_transverse_speed(RB87_MASS_KG, 300.0)
        g = transit_rate(RB87_MASS_KG, 300.0, 1.0e-3)
        assert abs(g * 1.0e-3 / v - math.sqrt(2 * math.log(2))) < 1e-12
        assert abs(g / math.pi / 79.6e3 - 1.0) < 0.01
        assert abs(lt.transit_fwhm_hz(CS_MASS_KG, 300.0, 0.75e-3)
                   / 86e3 - 1.0) < 0.02      # Cs number, spec 04 SS2.3.3


# ---------------------------------------------------------------------------
# vdW, Fermi — collision/interaction terms (R-26, R-11)
# ---------------------------------------------------------------------------

class TestInteractionTerms:
    def test_c6_au_conversion_check_value(self):
        """Spec 04 SS2.3.5(iii) check value: 1 a.u. C6 = 1.4448e-19
        GHz um^6 (runtime-derived AU_C6_HZ_M6, lock #15)."""
        ghz_um6 = lt.AU_C6_HZ_M6 / 1e9 / 1e-36
        assert abs(ghz_um6 / 1.4448e-19 - 1.0) < 1e-3

    def test_c6_rb60s_selfcheck(self):
        """Self-check (spec 04 SS3.5): |C6|(Rb 60S) ~ 140 GHz um^6 from
        the Singer fit."""
        c6 = lt.c6_over_h_rb_ns(60).value / 1e9 / 1e-36
        assert abs(c6 / 140.0 - 1.0) < 0.05

    def test_vdw_dephasing_r26_bookkeeping(self):
        """Ruling R-26: Delta_nu_vdW = (C6/h)/r_nn^6 with NO second /h.
        Rb 50S at N_r = 1e8 cm^-3 (r_nn = 13.4 um) -> ~3 kHz; scaling
        N_r^2 exact."""
        c6 = lt.c6_over_h_rb_ns(50).value
        dv = lt.vdw_dephasing_hz(c6, 1e14)
        assert 1.5e3 < dv < 6e3
        assert abs(lt.vdw_dephasing_hz(c6, 1e15) / dv - 100.0) < 1e-6

    def test_fermi_shift_coefficient(self):
        """Spec 04 SS2.3.5(ii) check value: shift coefficient -9.9e-8
        Hz cm^3 for Rb (a_s = -16.1 a0, runtime-derived), i.e. -9.9e-14
        Hz m^3; audit item 15 re-derivation."""
        coeff = lt.fermi_shift_hz(RB87, 1.0)          # per unit density
        assert abs(coeff / (-9.9e-14) - 1.0) < 0.02

    def test_fermi_cs_refuses(self):
        """No sourced e^-/Cs scattering length -> IntegrityError."""
        with pytest.raises(IntegrityError, match="scattering length"):
            lt.fermi_shift_hz(CS133, 1e16)


# ---------------------------------------------------------------------------
# Laser dephasing (lock #8, ruling R-16) and Stark (ruling R-5)
# ---------------------------------------------------------------------------

class TestLaserAndStark:
    def test_r16_formula(self):
        """Ruling R-16 / spec 04 SS2.3.4: gamma_gr = pi (Dnu_p + Dnu_c +
        2 c sqrt(Dnu_p Dnu_c)); rates ADD in a ladder (common-mode noise
        does NOT cancel: c = +1 doubles the independent-laser rate for
        equal linewidths; c = -1 nulls it)."""
        ge, er, gr0 = laser_dephasing_rates(2e5, 2e5, 0.0)
        assert abs(ge - math.pi * 2e5) < 1e-9
        assert abs(er - math.pi * 2e5) < 1e-9
        assert abs(gr0 - math.pi * 4e5) < 1e-9
        _, _, gr_p = laser_dephasing_rates(2e5, 2e5, +1.0)
        _, _, gr_m = laser_dephasing_rates(2e5, 2e5, -1.0)
        assert abs(gr_p / gr0 - 2.0) < 1e-12
        assert abs(gr_m) < 1e-9
        with pytest.raises(ValueError):
            laser_dephasing_rates(2e5, 2e5, 1.5)

    def test_r5_stark_magnitude(self):
        """Ruling R-5: with the VERIFIED spec-07 alpha_0(Rb 50S) =
        50.5 MHz/(V/cm)^2, 10 mV/cm gives ~2.5 kHz — NOT the ~30 kHz the
        banned spec-04 placeholder implied."""
        w = lt.stark_inhomogeneous_fwhm_hz(ALPHA0_RB50S_HZ_V2M2, 1.0)
        assert abs(w / 2525.0 - 1.0) < 1e-6
        assert w < 5e3          # the x12 placeholder would give ~30 kHz

    def test_r5_budget_refuses_without_alpha(self):
        """Audit R1: build_budget with E_stray > 0 and no caller-supplied
        alpha raises IntegrityError citing R-5 — the 6e2 literal must
        never appear in code."""
        d = default_cell("typical")
        with pytest.raises(IntegrityError, match="R-5"):
            build_budget(RB87, (50, 0, 0.5), d.cell, d.lasers,
                         d.Omega_p, d.Omega_c)


# ---------------------------------------------------------------------------
# The dephasing budget (spec 04 SS5, SS5.1)
# ---------------------------------------------------------------------------

def _typical_budget():
    d = default_cell("typical")
    cell = replace(d.cell, alpha0_over_h_hz_v2m2=ALPHA0_RB50S_HZ_V2M2)
    return build_budget(RB87, (50, 0, 0.5), cell, d.lasers,
                        d.Omega_p, d.Omega_c)


class TestBudget:
    def test_typical_cell_rows(self):
        """Spec 04 SS5.1 'typical cell' budget rows (R-5-corrected):
        transit 106 kHz; laser gamma_gr/pi 400 kHz; Stark ~2.5 kHz (NOT
        30 kHz); Gamma_r/2pi ~ 2.4 kHz; power term ~1.5 MHz."""
        b = _typical_budget()
        assert abs(b.gamma_transit / math.pi / 106e3 - 1.0) < 0.01
        assert abs(b.gamma_laser_gr / math.pi / 400e3 - 1.0) < 1e-9
        stark = b.inhomo["stark"]["delta_nu_hz"]
        assert abs(stark / 2525.0 - 1.0) < 1e-6
        assert b.Gamma_r / (2 * math.pi) == pytest.approx(2.44e3, rel=0.05)
        power = b.Omega_c**2 / b.Gamma_e / (2 * math.pi)
        assert power == pytest.approx(1.5e6, rel=0.05)

    def test_good_cell_rows(self):
        """Spec 04 SS5.1 'good cell': transit 79 kHz, laser 20 kHz."""
        d = default_cell("good")
        cell = replace(d.cell, alpha0_over_h_hz_v2m2=ALPHA0_RB50S_HZ_V2M2)
        b = build_budget(RB87, (50, 0, 0.5), cell, d.lasers,
                        d.Omega_p, d.Omega_c)
        assert abs(b.gamma_transit / math.pi / 79.6e3 - 1.0) < 0.01
        assert abs(b.gamma_laser_gr / math.pi / 20e3 - 1.0) < 1e-9

    def test_gamma_ij_assembly(self):
        """Spec 04 SS2.4: gamma_ij = (Gamma_i + Gamma_j)/2 + pure
        dephasings + transit — assembled exactly (S7 wiring, static
        half)."""
        b = _typical_budget()
        g = b.gamma_ij()
        assert g[("g", "e")] == pytest.approx(
            b.Gamma_e / 2 + b.gamma_laser_ge + b.gamma_self_ge
            + b.gamma_transit, rel=1e-12)
        assert g[("g", "r")] == pytest.approx(
            (b.Gamma_r_rad + b.Gamma_r_bbr) / 2 + b.gamma_laser_gr
            + b.gamma_rydgnd_gr + b.gamma_transit, rel=1e-12)

    def test_inhomogeneous_terms_are_not_rates(self):
        """Audit SS3 item 15: Doppler/Stark/vdW/beam live in ``inhomo`` as
        descriptors; none of them appears in gamma_ij. In particular
        Dk sigma_v (~2pi 136 MHz-scale) must NOT be inside gamma_gr."""
        b = _typical_budget()
        assert "doppler" in b.inhomo and "stark" in b.inhomo
        sigma_v = b.inhomo["doppler"]["sigma_v_m_s"]
        assert sigma_v == pytest.approx(169.4, rel=0.01)  # spec 04 SS2.3.8
        dk = 2 * math.pi * (1 / 480e-9 - 1 / 780e-9)
        assert b.gamma_ij()[("g", "r")] < 0.01 * dk * sigma_v

    def test_cs_budget_and_hot_cell_refusal(self):
        """Cs budget builds cold (Fermi omitted with note); a hot Cs cell
        (N_g > 1e19 m^-3) refuses — no sourced scattering length."""
        d = default_cell("typical")
        cell = replace(d.cell, alpha0_over_h_hz_v2m2=1e3)
        b = build_budget(CS133, (50, 0, 0.5), cell, d.lasers,
                         d.Omega_p, d.Omega_c)
        assert b.shift_fermi_hz == 0.0
        assert any("Fermi" in n for n in b.notes)
        hot = replace(cell, N_ground_m3=5e19)
        with pytest.raises(IntegrityError, match="hot cell"):
            build_budget(CS133, (50, 0, 0.5), hot, d.lasers,
                         d.Omega_p, d.Omega_c)

    def test_vdw_gate_and_refusal(self):
        """vdW warning above 0.1 MHz (spec 04 SS2.3.5(iii)); non-Rb-nS
        Rydberg density with no sourced C6 refuses (audit R11)."""
        d = default_cell("typical")
        cell = replace(d.cell, alpha0_over_h_hz_v2m2=ALPHA0_RB50S_HZ_V2M2,
                       N_rydberg_m3=1e15)              # 1e9 cm^-3
        b = build_budget(RB87, (50, 0, 0.5), cell, d.lasers,
                         d.Omega_p, d.Omega_c)
        assert b.inhomo["vdw"]["delta_nu_hz"] > 1e5
        assert any("vdW" in w for w in b.warnings())
        with pytest.raises(IntegrityError, match="C6"):
            build_budget(RB87, (50, 2, 2.5), cell, d.lasers,
                         d.Omega_p, d.Omega_c)

    def test_perturber_density_is_elemental_not_isotope_partial(self):
        """04<->05 interface (audit): the self-broadening beta and the Fermi
        pseudopotential are both defined against the TOTAL elemental vapour
        density — Weller 2011 quotes beta*N for a natural-abundance Rb cell,
        and every ground-state atom inside the Rydberg orbit is a perturber
        regardless of isotope. Independent checks: the budget density equals
        rydsim.cell's elemental Alcock density computed here; Rb-85 and
        Rb-87 (abundances 0.72 / 0.28) must see the SAME perturber density
        while their isotope densities differ by the abundance ratio; and the
        collisional terms equal beta*N / the Fermi coefficient times that
        density, recomputed independently."""
        d = default_cell("typical")
        cell = replace(d.cell, alpha0_over_h_hz_v2m2=ALPHA0_RB50S_HZ_V2M2)
        b87 = build_budget(RB87, (50, 0, 0.5), cell, d.lasers,
                           d.Omega_p, d.Omega_c)
        b85 = build_budget(RB85, (50, 0, 0.5), cell, d.lasers,
                           d.Omega_p, d.Omega_c)
        n_elem = number_density_m3("Rb", cell.T_cell_K)
        assert b87.N_ground_m3 == pytest.approx(n_elem, rel=1e-12)
        assert b85.N_ground_m3 == pytest.approx(n_elem, rel=1e-12)
        assert b87.N_isotope_m3 / b85.N_isotope_m3 == pytest.approx(
            RB87.abundance / RB85.abundance, rel=1e-12)
        beta = lt.self_broadening_beta_hz_m3(RB87, "D2").value
        assert b87.gamma_self_ge == pytest.approx(
            math.pi * beta * n_elem, rel=1e-12)
        assert b87.shift_fermi_hz == pytest.approx(
            lt.fermi_shift_hz(RB87, n_elem), rel=1e-12)

    def test_hot_cell_collisional_terms_match_spec04_magnitudes(self):
        """Spec 04 SS2.3.5 magnitude anchors are ELEMENTAL: ~2e13 cm^-3 at
        130 C giving ~2.2 MHz self-broadening and ~-2 MHz Fermi shift. With
        the isotope-partial density the same cell reported 1.02 MHz and
        -0.98 MHz — a 2.25x error in gamma_gr, the term spec 04 calls
        dominant in hot cells. Reference values recomputed here from the
        elemental Alcock density and the published coefficients."""
        cell = CellParams(T_cell_K=403.15, E_stray_v_m=0.0)
        b = build_budget(RB87, (50, 0, 0.5), cell, LaserParams(),
                         1.0e6, 1.0e6)
        n_elem = number_density_m3("Rb", 403.15)
        assert 3.0e19 < n_elem < 4.0e19          # ~3.6e13 cm^-3 at 130 C
        assert b.gamma_self_ge / math.pi == pytest.approx(
            lt.self_broadening_fwhm_hz(RB87, "D2", n_elem), rel=1e-12)
        assert 3.0e6 < b.gamma_self_ge / math.pi < 4.5e6
        assert -4.0e6 < b.shift_fermi_hz < -3.0e6
        assert b.gamma_ij()[("g", "r")] > 5.0e6

    def test_below_n15_sum_path_is_usable(self):
        """Audit: build_budget's 'sum' path — the route spec 04 SS7.1
        designates FOR n < 15 — was blocked by the Beterov-window gate
        inside the ~1 % W_BBR_ion term, with a message telling the caller to
        use the path they had already selected. It must degrade to 'not
        evaluated, reason recorded' instead. Gamma_r is checked against the
        sums computed independently here; the direct public call still
        raises."""
        cell = CellParams(E_stray_v_m=0.0)
        b = build_budget(RB87, (12, 0, 0.5), cell, LaserParams(),
                         1.0e6, 1.0e6, gamma_r_method="sum")
        table = decay_table(RB87, 12, 0, 0.5)
        assert b.Gamma_r_rad == pytest.approx(table.gamma0_rad_s, rel=1e-12)
        assert b.Gamma_r_bbr == pytest.approx(table.bbr_rate(300.0),
                                              rel=1e-12)
        assert b.W_bbr_ion == 0.0
        assert any("photoionization not evaluated" in n for n in b.notes)
        with pytest.raises(IntegrityError, match="15 <= n <= 80"):
            lt.bbr_ionization_rate(RB87, 12, 0, 0.5, 300.0)

    def test_eit_sanity_width_r24(self):
        """Ruling R-24: the sanity form is spec 06's 2 gamma_gr +
        Omega_c^2/(2 gamma_ge) — non-quantitative in Doppler media, used
        for orders of magnitude only."""
        b = _typical_budget()
        g = b.gamma_ij()
        expect = 2 * g[("g", "r")] + b.Omega_c**2 / (2 * g[("g", "e")])
        assert b.eit_fwhm_sanity_rad_s() == pytest.approx(expect, rel=1e-12)


# ---------------------------------------------------------------------------
# B15 / S7 — Lindblad dephasing wiring (the factor-2 conventions)
# ---------------------------------------------------------------------------

def _probe_absorption(gamma_e, omega_c, gamma_p, gamma_c, dp_grid,
                      omega_p=None, extra_r_dephasing=0.0):
    """Steady-state Im(rho_ge)-proportional absorption vs probe detuning
    from the rydsim.lindblad engine (spec 04 SS2.4 row-4 operator mapping:
    per-level projector dephasings [0, pi Dnu_p, pi(Dnu_p + Dnu_c) + x])."""
    if omega_p is None:
        omega_p = 1e-3 * gamma_e
    out = np.empty(dp_grid.size)
    for i, dp in enumerate(dp_grid):
        if omega_c == 0.0:
            # 2-level reference: with the coupling off, level r is
            # disconnected and the 3-level steady state is non-unique
            # (singular Liouvillian) — drop it.
            sys = LadderSystem(
                omegas=np.array([omega_p]), deltas=np.array([dp]),
                decays=np.array([0.0, gamma_e]),
                dephasings=np.array([0.0, gamma_p]))
        else:
            sys = LadderSystem(
                omegas=np.array([omega_p, omega_c]),
                deltas=np.array([dp, 0.0]),
                decays=np.array([0.0, gamma_e, 0.0]),
                dephasings=np.array([0.0, gamma_p,
                                     gamma_p + gamma_c + extra_r_dephasing]))
        rho = sys.steady_state()
        out[i] = abs(rho[1, 0].imag)
    return out


def _fwhm(x, y):
    """FWHM of a single-peaked profile by linear interpolation."""
    half = y.max() / 2.0
    above = y >= half
    i0, i1 = np.argmax(above), len(above) - np.argmax(above[::-1]) - 1
    def cross(ia, ib):
        return x[ia] + (x[ib] - x[ia]) * (half - y[ia]) / (y[ib] - y[ia])
    return cross(i1, i1 + 1) - cross(i0, i0 - 1)


class TestLindbladWiring:
    GAMMA_E = RB87.d2.gamma_rad_s        # 2pi x 6.0666 MHz

    def test_b15_probe_line_fwhm(self):
        """Spec 04 SS6 B15 / self-check S7 (probe half): with only probe
        laser noise on (Dnu_p = 2 MHz), the weak-probe absorption line
        FWHM = Gamma_e/2pi + Dnu_p (no Doppler), tol 2%. Verifies the
        sqrt(2 gamma) projector convention of SS2.4 row 4."""
        dnu_p = 2.0e6
        gamma_p = math.pi * dnu_p
        expect = self.GAMMA_E + 2.0 * gamma_p          # angular FWHM
        dp = np.linspace(-3 * expect, 3 * expect, 401)
        a = _probe_absorption(self.GAMMA_E, 0.0, gamma_p, 0.0, dp)
        assert abs(_fwhm(dp, a) / expect - 1.0) < 0.02

    @pytest.mark.parametrize("corr,factor", [(0.0, 1.0), (1.0, 2.0)])
    def test_b15_eit_dip_two_photon_width(self, corr, factor):
        """Spec 04 SS6 B15 / S7 (EIT half) + ruling R-16 extension: as
        Omega_c -> 0 the EIT dip FWHM -> 2 gamma_gr (angular), with
        gamma_gr = pi(Dnu_p + Dnu_c + 2c sqrt(Dnu_p Dnu_c)): the c = +1
        common-mode case must WIDEN the dip by exactly 2x (ladder — noise
        adds, never cancels). Tol 3% against the first-order form
        2 gamma_gr + Omega_c^2/(2 gamma_ge)."""
        dnu = 2.0e3                                    # 2 kHz per laser
        gamma_p = gamma_c = math.pi * dnu
        cross = 2.0 * corr * math.sqrt(gamma_p * gamma_c)
        gamma_gr = gamma_p + gamma_c + cross
        gamma_ge = self.GAMMA_E / 2.0 + gamma_p
        omega_c = math.sqrt(0.2 * gamma_gr * 2.0 * gamma_ge)
        expect = 2.0 * gamma_gr + omega_c**2 / (2.0 * gamma_ge)
        dp = np.linspace(-6 * expect, 6 * expect, 801)
        a_on = _probe_absorption(self.GAMMA_E, omega_c, gamma_p, gamma_c,
                                 dp, extra_r_dephasing=cross)
        a_off = _probe_absorption(self.GAMMA_E, 0.0, gamma_p, gamma_c,
                                  dp, extra_r_dephasing=cross)
        dip = a_off - a_on
        width = _fwhm(dp, dip)
        assert abs(width / expect - 1.0) < 0.03
        # R-16 relation on the rate itself: for equal linewidths,
        # gamma_gr = pi Dnu (2 + 2c) -> the c = +1 dip is exactly 2x the
        # c = 0 dip (factor parameterizes this).
        assert gamma_gr == pytest.approx(
            math.pi * dnu * 2.0 * factor, rel=1e-12)


# ---------------------------------------------------------------------------
# B10 — hot-cell Rydberg EIT linewidth (velocity-integrated, qualitative)
# ---------------------------------------------------------------------------

class TestHotCellEIT:
    def test_b10_typical_cell_band(self):
        """Spec 04 SS6 B10: velocity-integrated hot-cell Rydberg EIT
        linewidth with 'typical cell' parameters lands in the measured
        2-10 MHz band (canonical ~2 MHz, Mohapatra/Jackson/Adams PRL 98,
        113003 (2007)). Uses the budget's gamma_ij in the weak-probe
        continued-fraction chi with the resonance-refined velocity grid
        (rydsim.eit — GH quadrature is forbidden for EIT, ruling R-2);
        lambda_c computed at runtime from spec 01 energies (ruling R-15).
        Doppler enters ONLY through the velocity integral (SS2.3.8)."""
        b = _typical_budget()
        g = b.gamma_ij()
        lam_p = RB87.d2.lambda_vac_m
        lam_c = float(coupling_wavelength_m(RB87, "D2", 50, 0, 0.5))
        assert 4.7e-7 < lam_c < 4.9e-7                 # sanity (R-15)
        k_p = 2 * math.pi / lam_p
        k_c = 2 * math.pi / lam_c
        dp = np.linspace(-2 * math.pi * 15e6, 2 * math.pi * 15e6, 501)
        par_on = LadderChiParams(
            omegas=np.array([b.Omega_p, b.Omega_c]),
            deltas=np.array([0.0, 0.0]),
            k_signed=np.array([k_p, -k_c]),            # counter-propagating
            coherence_decays=np.array([g[("g", "e")], g[("g", "r")]]))
        par_off = LadderChiParams(
            omegas=np.array([b.Omega_p, 0.0]),
            deltas=np.array([0.0, 0.0]),
            k_signed=np.array([k_p, -k_c]),
            coherence_decays=np.array([g[("g", "e")], g[("g", "r")]]))
        s_on = doppler_average(par_on, dp, RB87_MASS_KG, 300.0)
        s_off = doppler_average(par_off, dp, RB87_MASS_KG, 300.0)
        dip = np.real(s_off) - np.real(s_on)
        assert dip.max() > 0.02 * np.real(s_off).max()  # visible window
        fwhm_hz = _fwhm(dp, dip) / (2 * math.pi)
        assert 2e6 <= fwhm_hz <= 10e6, f"EIT FWHM {fwhm_hz/1e6:.2f} MHz"


# ---------------------------------------------------------------------------
# Consistency of einstein_rate with the spec 03 layer (D2 closure)
# ---------------------------------------------------------------------------

class TestSteckClosure:
    def test_einstein_rate_matches_angular_einstein_A(self):
        """The lifetime engine's rate helper and rydsim.angular.einstein_A
        agree on the D2 line through the Steck->Racah conversion (spec 03
        B18/B22 identity, here as an inter-module wiring check)."""
        from rydsim.angular import einstein_A, steck_to_racah
        from rydsim.constants import AU_DIPOLE
        omega = 2 * math.pi * RB87.d2.nu0_hz
        d_racah = steck_to_racah(RB87.d2.d_reduced_ea0, 0.5) * AU_DIPOLE
        a_direct = einstein_A(omega, d_racah, 1.5)
        assert abs(a_direct * RB87.d2.tau_s - 1.0) < 2e-4
        tau_engine = lt.radiative_lifetime_sum(RB87, 5, 1, 1.5)
        assert abs(tau_engine / RB87.d2.tau_s - 1.0) < 2e-4
