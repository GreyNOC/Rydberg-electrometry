"""Spec 02 benchmarks — radial wavefunctions and matrix elements.

The Wave-2 implementation agent died before writing this file; these tests
were authored against docs/spec/02-radial-wavefunctions-numerov.md §6 and
the module's actual API. Every exact-value row is an analytic identity
(Gordon's formula / hydrogen closed forms), an independent published
quantity (spec 01's measured quantum defects), or a cross-method consensus
— no value here is asserted from recall, and no test inspects the module's
own source text.

Layout:
  1. the exact hydrogen ORACLE (Gordon) — validates the oracle only;
  2. spec 02 §6 B1-B7 through the ENGINE (coulomb_wavefunction ->
     radial_matrix_element), i.e. the path that produces shipped numbers;
  3. Method C (Kaulakys) vs the oracle, B13;
  4. Method B's Numerov-independent arm: the hyperu/Whittaker fence
     (audit R4, binding form) and B12;
  5. the consensus machinery and its refusals — exercised by BEHAVIOUR;
  6. the MSD94 model-potential tripwire (audit R20): the only quantity in
     the module that is actually sensitive to the parameter tables.
"""

import dataclasses
import math

import numpy as np
import pytest

from rydsim import radial as R
from rydsim.atom import CS133, RB87, Species, n_star
from rydsim.provenance import IntegrityError

# ---------------------------------------------------------------------------
# 1. Method D (exact): Gordon's formula on hydrogen — the ORACLE, not the
#    engine. Renamed from test_gordon_exact_hydrogen so nobody reads it as
#    coverage of the Numerov chain (that is §2 below).
# ---------------------------------------------------------------------------

HYDROGEN_EXACT = [
    ("B1 1s->2p", 1, 0, 2, 1, 1.2902662020),      # = 128 sqrt(6)/243
    ("B2 2s->3p", 2, 0, 3, 1, 3.0648154066),
    ("B3 2p->3d", 2, 1, 3, 2, 4.7479916115),
    ("B4 10s->11p", 10, 0, 11, 1, 40.4352023233),
    ("B5 50s->50p", 50, 0, 50, 1, 3749.2499249850),  # = (3/2) n sqrt(n^2-l^2)
    ("B6 50s->51p", 50, 0, 51, 1, 851.4038694455),
]


@pytest.mark.parametrize("label,n1,l1,n2,l2,expect", HYDROGEN_EXACT)
def test_gordon_oracle_exact_hydrogen(label, n1, l1, n2, l2, expect):
    """The analytic ORACLE reproduces the spec 02 §6 hydrogen values.

    This exercises ``radial_me_gordon`` (exact-rational 2F1), NOT the
    Numerov engine — see ``test_numerov_engine_hydrogen_b1_b6``.
    """
    got = abs(R.radial_me_gordon(n1, l1, n2, l2))
    assert got == pytest.approx(expect, rel=1e-7), label


def test_gordon_closed_form_identity():
    """|R(n s -> n p)| = (3/2) n sqrt(n^2 - l^2) for the same-n hydrogenic
    pair — an independent analytic route to the 50s->50p row."""
    for n in (20, 50, 80):
        closed = 1.5 * n * np.sqrt(n**2 - 1.0)
        assert abs(R.radial_me_gordon(n, 0, n, 1)) == pytest.approx(closed, rel=1e-9)


def test_gordon_rejects_forbidden_and_invalid():
    with pytest.raises((ValueError, IntegrityError)):
        R.radial_me_gordon(2, 0, 3, 2)      # |dl| = 2, dipole-forbidden
    with pytest.raises((ValueError, IntegrityError)):
        R.radial_me_gordon(2, 2, 3, 1)      # l >= n


# ---------------------------------------------------------------------------
# 2. Spec 02 §6 B1-B7 THROUGH THE ENGINE.
#
# These are the absolute benchmarks the spec labels "H **Numerov** R(...)":
# they run coulomb_wavefunction -> radial_matrix_element, which is the same
# scaled equation (2.5), divergence guard, norm (2.6) and ME weight (2.7)
# that every shipped alkali number goes through. The alkali A-vs-B spread
# cannot cover this: Methods A and B share the identical ODE solver, grid,
# guard and quadrature, so any systematic error in the scaled equation
# cancels exactly in A - B.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,n1,l1,n2,l2,expect", HYDROGEN_EXACT)
def test_numerov_engine_hydrogen_b1_b6(label, n1, l1, n2, l2, expect):
    """Spec 02 §6 B1-B6 at the module defaults: rel <= 1e-7 (B1-B4),
    <= 1e-6 (B5, B6) against exact hydrogen.

    Truth is Gordon's exact-rational value (an analytic identity, not a
    recalled constant); the HYDROGEN_EXACT literals above are a second,
    independent transcription of the same numbers and are checked against
    the oracle in §1.

    NOTE on r_inner: at the module's OLD default of 1e-2 a0 this row
    measured 6.57e-7 for B1 — 6.6x outside the spec's own 1e-7 tolerance.
    That was an inner-truncation artifact of the benchmark knob, not of the
    engine; the default is now 1e-4 (see ``coulomb_wavefunction``'s
    docstring for the measured r_inner table and for why 1e-5 is worse).
    The §6 tolerances are unchanged.
    """
    n_max = max(n1, n2)
    r_outer = 2.0 * n_max * (n_max + 15.0)
    a = R.coulomb_wavefunction(n1, l1, r_outer=r_outer)
    b = R.coulomb_wavefunction(n2, l2, r_outer=r_outer)
    got = abs(R.radial_matrix_element(a, b, 1))
    exact = abs(R.radial_me_gordon(n1, l1, n2, l2))
    assert exact == pytest.approx(expect, rel=1e-9), f"{label}: oracle drift"
    tol = 1e-6 if n_max >= 50 else 1e-7
    assert got == pytest.approx(exact, rel=tol), label


def test_numerov_engine_global_order_b7():
    """Spec 02 §6 B7: the scaled engine converges at h^4.

    err(h=0.01)/err(h=0.005) on B1 must land in [8, 32] (spec measured
    16.2; this implementation measures 16.3). This is the only test that
    pins Eq. (2.5)'s g-function — the factor 8 and the
    (2l+1/2)(2l+3/2)/x^2 term — as a discretization, independently of any
    tolerance on the value itself.
    """
    exact = abs(R.radial_me_gordon(1, 0, 2, 1))

    def err(h: float) -> float:
        a = R.coulomb_wavefunction(1, 0, h=h, r_outer=2 * 2 * (2 + 15))
        b = R.coulomb_wavefunction(2, 1, h=h, r_outer=2 * 2 * (2 + 15))
        return abs(abs(R.radial_matrix_element(a, b, 1)) - exact) / exact

    ratio = err(0.01) / err(0.005)
    assert 8.0 <= ratio <= 32.0, f"global order ratio {ratio:.2f} not h^4-like"


def test_numerov_engine_k2_against_hydrogen_closed_form():
    """<n s| r^2 |n s> = (n^2/2)(5 n^2 + 1 - 3 l(l+1)) for hydrogen
    (Bethe & Salpeter §3) — the only independent check of the k >= 2 weight
    x^(2k+2) in Eq. (2.7). The consensus entry point has NO third method for
    k >= 2 (Kaulakys is k = 1 only), so without this the exponent is
    untested."""
    for n in (10, 30, 50):
        closed = 0.5 * n * n * (5.0 * n * n + 1.0)
        sol = R.coulomb_wavefunction(n, 0, r_outer=2.0 * n * (n + 15.0))
        got = R.radial_matrix_element(sol, sol, 2)
        assert got == pytest.approx(closed, rel=1e-7), f"n={n}"


def test_outer_cutoff_adequacy_b14():
    """Spec 02 §6 B14: the 2n(n+15) outer cutoff is converged — extending it
    changes the ME by <= 5e-8 relative."""
    s1, s2 = (50, 0, 0.5), (51, 1, 1.5)
    a = R.radial_wavefunction(RB87, *s1, r_outer=2 * 51 * (51 + 15))
    b = R.radial_wavefunction(RB87, *s2, r_outer=2 * 51 * (51 + 15))
    me_std = R.radial_matrix_element(a, b)
    a2 = R.radial_wavefunction(RB87, *s1, r_outer=2 * 51 * (51 + 25))
    b2 = R.radial_wavefunction(RB87, *s2, r_outer=2 * 51 * (51 + 25))
    me_wide = R.radial_matrix_element(a2, b2)
    assert abs(me_wide - me_std) / abs(me_std) < 5e-8


# ---------------------------------------------------------------------------
# 3. Method C: Kaulakys semiclassical vs the exact hydrogenic limit (B13)
# ---------------------------------------------------------------------------

def test_kaulakys_matches_gordon_hydrogen():
    """Spec 02 §6 B13: Kaulakys vs Gordon at nu = 50, dnu = 1 and 2,
    rel <= 2e-4."""
    for dn in (1, 2):
        exact = abs(R.radial_me_gordon(50, 0, 50 + dn, 1))
        semi = abs(R.radial_me_kaulakys(50.0, 0, 50.0 + dn, 1))
        assert semi == pytest.approx(exact, rel=2e-4), f"dn={dn}"


# ---------------------------------------------------------------------------
# 4. Method B's Numerov-independent arm: hyperu/Whittaker (audit R4, B12)
# ---------------------------------------------------------------------------

def test_hyperu_fence_r4_binding_form():
    """Audit R4 verbatim: "assert error(nu=20) < 1e-6 AND assert the
    collapse (error(nu=35) > 1e-2) actually occurs where claimed".

    The whole point of R4 is that hyperu accuracy is scipy-version-
    dependent, so the fence location must be re-measured on the INSTALLED
    scipy rather than asserted monotonically. A monotonicity test
    (err(25) > err(12)) — the previous implementation — passes both when a
    future scipy improves hyperu (making the WHITTAKER_NU_MAX refusal over-
    conservative) and when it regresses so the collapse moves below 25;
    neither is detectable from monotonicity alone.

    Measured on scipy 1.17.1: 7.4e-12 @ 12, 3.5e-8 @ 20, 5.5e-6 @ 25,
    1.9e-4 @ 28, 1.8e-3 @ 30, 0.49 @ 35.
    """
    # accuracy where the module claims accuracy
    assert R.hyperu_hydrogen_error(20, 0) < 1e-6
    # the fence's upper edge: usable, but already ~1e-5 (worse than B12's
    # own 1e-6 tolerance — documented in whittaker_u, not silently ignored)
    err25 = R.hyperu_hydrogen_error(25, 0)
    assert err25 < 1e-5
    assert err25 > 1e-7, (
        "hyperu at nu=25 is now better than 1e-7: the fence "
        f"(WHITTAKER_NU_MAX = {R.WHITTAKER_NU_MAX:g}) may have become "
        "over-conservative on this scipy — revisit audit R4")
    # the collapse occurs where claimed
    assert R.hyperu_hydrogen_error(30, 0) > 1e-4
    assert R.hyperu_hydrogen_error(35, 0) > 1e-2


def test_coulomb_route_refuses_above_the_fence():
    """Audit R4 + integrator ruling 2026-08-10: the Whittaker route refuses
    above WHITTAKER_NU_MAX regardless of apparent accuracy.

    The fence moved 25 -> 20 (it must sit where the method meets B12's own
    1e-6 contract: 3.5e-8 at nu = 20, up to 5.7e-6 just above it). The test
    is written against the CONSTANT, not a literal, so it cannot silently
    keep asserting a superseded boundary — the old name said 25.
    """
    r = np.array([100.0, 200.0])
    R.whittaker_u(R.WHITTAKER_NU_MAX, 0, r)          # at the fence: allowed
    with pytest.raises(IntegrityError):
        R.whittaker_u(R.WHITTAKER_NU_MAX + 0.5, 0, r)
    with pytest.raises(IntegrityError):
        R.whittaker_u(30.0, 0, r)


def test_whittaker_refuses_nonfinite_instead_of_returning_nan():
    """A silent NaN array is not a refusal (audit: "never a silently
    degraded number").

    scipy 1.17.1's hyperu returns NaN for non-integer nu beyond
    r ~ 0.2-0.9 nu^2 — inside the classically allowed region where the
    orbital is LARGEST. Handing that back would let a np.allclose-style B12
    comparison succeed on an all-NaN slice, i.e. a scipy bump could
    silently disable the module's only Numerov-independent cross-check.
    """
    nu, l = 18.5, 1
    r = np.linspace(0.2 * nu * nu, 1.8 * nu * nu, 400)
    raw = R._whittaker_u_unguarded(nu, l, r)
    assert not np.all(np.isfinite(raw)), (
        "this scipy no longer produces the non-finite regime this guard "
        "exists for — re-measure the fence (audit R4)")
    with pytest.raises(IntegrityError, match="non-finite"):
        R.whittaker_u(nu, l, r)
    # the clean non-integer case still computes
    nu_ok = 12.5
    r_ok = np.linspace(0.2 * nu_ok**2, 1.8 * nu_ok**2, 400)
    u_ok = R.whittaker_u(nu_ok, 0, r_ok)
    assert np.all(np.isfinite(u_ok)) and np.max(np.abs(u_ok)) > 0.0


@pytest.mark.parametrize("nu,l", [(10, 0), (10, 2), (15, 0), (15, 1),
                                  (20, 0), (20, 1), (20, 2)])
def test_b12_whittaker_vs_numerov_method_b(nu, l):
    """Spec 02 §6 B12: the hyperu-Whittaker closed form and the pure-Coulomb
    Numerov Method B agree pointwise over the classical region to rel 1e-6.

    This is the ONE check that makes Method B Numerov-code-independent
    (spec 02 §2.4): the closed form shares no code with the ODE solver, the
    grid, the divergence guard or the quadrature. It was absent entirely.
    Measured here: 2.6e-11 @ nu=10, 2.4e-10 @ 15, 5.0e-8 @ 20.
    """
    sol = R.coulomb_wavefunction(float(nu), l,
                                 r_outer=2.0 * nu * (nu + 15.0))
    r, u = sol.r, sol.u
    mask = (r >= 0.2 * nu * nu) & (r <= 1.8 * nu * nu)
    u_w = R.whittaker_u(float(nu), l, r[mask])
    u_n = u[mask] * math.copysign(1.0, float(np.dot(u[mask], u_w)))
    dev = np.max(np.abs(u_n - u_w)) / np.max(np.abs(u_w))
    assert dev <= 1e-6, f"nu={nu}, l={l}: pointwise deviation {dev:.3e}"


@pytest.mark.parametrize("nu,l,norm_offset", [(10.5, 0, 7.3e-6),
                                              (12.5, 0, 4.4e-6)])
def test_b12_noninteger_nu_shape_and_norm_offset(nu, l, norm_offset):
    """B12 at NON-INTEGER nu — the QDT case Method B actually ships.

    Spec 02 §2.4 asserts the Seaton analytic normalization and the numerical
    unit norm agree, but its supporting measurement ("verified for integer
    nu against analytic hydrogen") only covers integer nu. At non-integer nu
    the Whittaker function is IRREGULAR at the origin (W ~ r^-l), so the
    numerically unit-normalized Numerov solution and the analytically
    normalized closed form differ by a cutoff-dependent SCALE. Measured
    scale offsets at nu = 10.5: 7.3e-6 (l=0), 6.6e-4 (l=1), 3.0e-3 (l=2);
    moving r_inner 1e-4 -> 1 moves the l=0 offset to 3.7e-5, confirming the
    cutoff dependence.

    The SHAPE — which is what B12 exists to cross-check — agrees to 3e-11
    regardless. So this row asserts the shape at 1e-6 and pins the norm
    offset separately instead of folding one into the other.
    """
    sol = R.coulomb_wavefunction(nu, l, r_outer=2.0 * math.ceil(nu)
                                 * (math.ceil(nu) + 15.0))
    r, u = sol.r, sol.u
    mask = (r >= 0.2 * nu * nu) & (r <= 1.8 * nu * nu)
    u_w = R.whittaker_u(nu, l, r[mask])
    u_n = u[mask]
    scale = float(np.dot(u_n, u_w) / np.dot(u_n, u_n))
    shape_dev = np.max(np.abs(u_n * scale - u_w)) / np.max(np.abs(u_w))
    assert shape_dev <= 1e-6, f"nu={nu}, l={l}: shape {shape_dev:.3e}"
    assert abs(abs(scale) - 1.0) == pytest.approx(norm_offset, rel=0.25), (
        f"nu={nu}, l={l}: Seaton-vs-unit-norm offset moved")


# ---------------------------------------------------------------------------
# 5. Consensus machinery — the module's reason to exist
# ---------------------------------------------------------------------------

def test_rb_50s_50p_consensus_and_spread():
    """Spec 02 §6 B8: Rb 50S1/2->50P3/2 cross-method spread <= 1e-4 relative.

    The spec's own quoted value for this pair is 2510.9 a0.
    """
    res = R.radial_matrix_element_consensus(RB87, (50, 0, 0.5), (50, 1, 1.5))
    assert abs(res.value) == pytest.approx(2510.9, rel=2e-3)
    assert res.spread_rel <= 1e-4
    assert len(res.per_method) >= 2          # never a single-method result


def test_consensus_reports_three_methods_for_rydberg_dipole():
    res = R.radial_matrix_element_consensus(RB87, (50, 0, 0.5), (51, 1, 1.5))
    assert set(res.per_method) >= {"model_potential", "coulomb"}
    assert res.spread_rel <= 1e-4


def test_fine_structure_ratio():
    """Spec 02 §6 B16: |R(50S->50P1/2)| / |R(50S->50P3/2)| = 1.0158 +- 0.002
    (the j-dependence enters only through the quantum defects)."""
    a = R.radial_matrix_element_consensus(RB87, (50, 0, 0.5), (50, 1, 0.5))
    b = R.radial_matrix_element_consensus(RB87, (50, 0, 0.5), (50, 1, 1.5))
    assert abs(a.value) / abs(b.value) == pytest.approx(1.0158, abs=0.002)


@pytest.mark.parametrize("n", [40, 50, 60, 70])
def test_scaling_invariant_rb(n):
    """Spec 02 §6 B10: |R|/(nu nu') in [1.10, 1.16] for nS1/2->nP3/2."""
    res = R.radial_matrix_element_consensus(RB87, (n, 0, 0.5), (n, 1, 1.5))
    nu1 = float(n_star(RB87, n, 0, 0.5))
    nu2 = float(n_star(RB87, n, 1, 1.5))
    coeff = abs(res.value) / (nu1 * nu2)
    assert 1.10 <= coeff <= 1.16, f"n={n}: coeff={coeff:.4f}"
    assert res.orbit_scale_a0k == pytest.approx(nu1 * nu2, rel=1e-12)


def test_scaling_invariant_cs():
    """Spec 02 §6 B11: same invariant holds for Cs (1.1304 at n = 50)."""
    res = R.radial_matrix_element_consensus(CS133, (50, 0, 0.5), (50, 1, 1.5))
    nu1 = float(n_star(CS133, 50, 0, 0.5))
    nu2 = float(n_star(CS133, 50, 1, 1.5))
    assert abs(res.value) / (nu1 * nu2) == pytest.approx(1.1304, abs=0.03)
    assert res.spread_rel <= 1e-4


def test_step_convergence():
    """Halving the Numerov step must not move the consensus value."""
    fine = R.radial_matrix_element_consensus(RB87, (50, 0, 0.5), (50, 1, 1.5),
                                             h=5e-4)
    std = R.radial_matrix_element_consensus(RB87, (50, 0, 0.5), (50, 1, 1.5),
                                            h=1e-3)
    assert abs(fine.value - std.value) / abs(std.value) < 1e-5


# --- the cancellation-suppressed regime (the 28-pair false-refusal class) ---

SUPPRESSED_PAIRS = (
    [(RB87, (n, 2, 2.5), (n + 1, 3, 3.5)) for n in range(16, 36)]
    + [(RB87, (n, 2, 2.5), (n, 3, 3.5)) for n in range(16, 20)]
    + [(RB87, (16, 1, 1.5), (17, 2, 2.5))]
    + [(CS133, (n, 2, 2.5), (n + 1, 3, 3.5)) for n in range(16, 19)]
)


@pytest.mark.parametrize("sp,s1,s2", SUPPRESSED_PAIRS,
                         ids=lambda v: str(v) if isinstance(v, tuple) else
                         getattr(v, "name", str(v)))
def test_suppressed_pairs_compute_not_refused(sp: Species, s1, s2):
    """The 28 legitimate Rb/Cs nD->n'F (and one nP->n'D) dipole pairs that
    the public entry point used to refuse.

    nD -> (n+1)F passes through a node in Delta-n around n ~ 30 for Rb, so
    |ME| -> 0 while every method's ABSOLUTE error floor stays put. Gating a
    RELATIVE spread there refuses numerically perfect results: Rb87
    (30,2,2.5)->(31,3,3.5) was rejected on |A|-|B| = 2.95e-5 a0 against an
    887.8 a0 orbit scale — a deviation of 3.3e-8 of scale — because the raw
    relative spread was 1.6e-3. The consequence was worse than the refusal:
    stark.py and lifetimes.py each re-implemented the consensus privately to
    route around this module (spec 02 §4.4 pitfall 8 forbids exactly that).

    The assertion is quantitative and independent of the gate: the two
    Numerov methods must agree to 2e-5 of the orbit scale — 5x tighter than
    the shipped 1e-4 ceiling — so this cannot be satisfied by a gate that
    was merely loosened. Measured worst case over these 28 pairs: 4.5e-6 of
    scale (Rb 16P3/2->17D5/2, the lowest-nu row, where the Coulomb
    approximation genuinely degrades); the Rydberg-n D->F rows sit at
    ~3e-8.
    """
    res = R.radial_matrix_element_consensus(sp, s1, s2)
    nu1 = float(n_star(sp, *s1))
    nu2 = float(n_star(sp, *s2))
    orbit = nu1 * nu2
    diff = abs(abs(res.per_method["model_potential"])
               - abs(res.per_method["coulomb"]))
    assert diff <= 2e-5 * orbit, (
        f"{sp.name} {s1}->{s2}: |A|-|B| = {diff:.3e} a0 = "
        f"{diff / orbit:.2e} of the orbit scale")
    assert res.orbit_scale_a0k == pytest.approx(orbit, rel=1e-12)
    assert res.spread_over_orbit <= 1e-3


def test_suppressed_pair_reports_orbit_relative_uncertainty():
    """On a suppressed element the RELATIVE spread is not an uncertainty.

    Rb87 30D5/2->31F7/2: |ME| = 0.01836 a0 on an 887.8 a0 orbit. The
    Kaulakys arm's absolute error floor (~0.03 a0) exceeds |ME| itself, so
    spread_rel reads 165 % on a result whose two Numerov methods agree to
    3.3e-8 of scale. The result object must carry the orbit-relative figure
    so a consumer can quote something meaningful.
    """
    res = R.radial_matrix_element_consensus(RB87, (30, 2, 2.5), (31, 3, 3.5))
    assert abs(res.value) < 1e-3 * res.orbit_scale_a0k     # is suppressed
    assert res.spread_rel > 1.0                            # meaningless here
    assert res.spread_over_orbit < 1e-4                    # meaningful here
    assert res.spread_abs == pytest.approx(
        res.spread_over_orbit * res.orbit_scale_a0k, rel=1e-12)


def test_high_l_pair_computes_and_is_gated_on_orbit_scale():
    """min(l,l') >= 4: Method A IS Method B there (spec 02 §2.1 sends l >= 4
    to the pure Coulomb potential), so the A-B difference measures solver
    noise, not the core model. It must still compute, and the reported
    difference must sit far under the 1e-3-of-orbit gate."""
    res = R.radial_matrix_element_consensus(RB87, (40, 4, 4.5), (41, 5, 5.5))
    diff = abs(abs(res.per_method["model_potential"])
               - abs(res.per_method["coulomb"]))
    assert diff <= 1e-4 * res.orbit_scale_a0k


# --- refusals: exercised by BEHAVIOUR, never by reading the source ---------

def test_low_n_refused_not_guessed():
    """Spec 01 §4.2 hard floor propagates into the radial layer: low-lying
    alkali levels are data-only, never Ritz-extrapolated.

    NOTE for the integrator: spec 02 §6 carries a Rb 5S->5P3/2 = 5.57 a0
    "model-potential bias documentation" row which this floor makes
    unreachable through the public API. That is a genuine spec-01/spec-02
    interface tension, recorded here rather than papered over; the bias row
    needs a declared low-n diagnostic path if it is ever to run.
    """
    with pytest.raises(IntegrityError):
        R.radial_matrix_element_consensus(RB87, (5, 0, 0.5), (5, 1, 1.5))


def test_negative_k_refused():
    with pytest.raises(IntegrityError):
        R.radial_matrix_element_consensus(RB87, (50, 0, 0.5), (50, 1, 1.5), k=-1)


def test_ab_consensus_failure_raises_rather_than_averaging(monkeypatch):
    """A real A-vs-B disagreement must raise — the module never averages a
    failed consensus (spec 02 §1, audit §3 item 8).

    The previous version of this test did ``inspect.getsource`` and asserted
    the substrings "check" and "IntegrityError" appeared, which stayed green
    with the refusal disabled entirely. This one injects a wrong Method-B
    result and asserts the BEHAVIOUR, then asserts check=False still returns
    both raw per-method values rather than an average.
    """
    true_me = R.radial_matrix_element

    def bent(a, b, k=1):
        v = true_me(a, b, k)
        return v * 1.01 if a.method == "coulomb" else v

    monkeypatch.setattr(R, "radial_matrix_element", bent)
    with pytest.raises(IntegrityError) as exc:
        R.radial_matrix_element_consensus(RB87, (50, 0, 0.5), (50, 1, 1.5))
    msg = str(exc.value)
    assert "A-vs-B" in msg and "1e-04" in msg.replace("1e-4", "1e-04")

    res = R.radial_matrix_element_consensus(RB87, (50, 0, 0.5), (50, 1, 1.5),
                                            check=False)
    a = res.per_method["model_potential"]
    b = res.per_method["coulomb"]
    # raw, not averaged: B is exactly the injected 1.01x, up to the pair's
    # own 2.0e-6 A-vs-B spread
    assert abs(abs(b) / abs(a) - 1.01) < 1e-5
    assert res.value == a                              # value IS Method A
    assert res.spread_rel == pytest.approx(0.01, abs=1e-4)


def test_ak_consensus_failure_raises(monkeypatch):
    """Method C is a real veto too: a Kaulakys result outside
    max(B9 relative ceiling, the formula's absolute floor) must raise."""
    true_k = R.radial_me_kaulakys
    monkeypatch.setattr(R, "radial_me_kaulakys",
                        lambda nu1, l1, nu2, l2: true_k(nu1, l1, nu2, l2) * 1.05)
    with pytest.raises(IntegrityError, match="A-vs-Kaulakys"):
        R.radial_matrix_element_consensus(RB87, (50, 0, 0.5), (51, 1, 1.5))


def test_suppressed_gate_still_has_teeth(monkeypatch):
    """The suppressed-regime gate is loosened in the right units, not
    disabled: a Method-B result off by more than 1e-4 of the orbit scale
    still raises on the very pair the old rule false-refused."""
    true_me = R.radial_matrix_element

    def bent(a, b, k=1):
        v = true_me(a, b, k)
        # +0.5 a0 on an 887.8 a0 orbit = 5.6e-4 of scale (gate is 1e-4)
        return v + 0.5 if a.method == "coulomb" else v

    monkeypatch.setattr(R, "radial_matrix_element", bent)
    with pytest.raises(IntegrityError, match="suppressed"):
        R.radial_matrix_element_consensus(RB87, (30, 2, 2.5), (31, 3, 3.5))


def test_check_ab_consensus_is_the_single_public_gate():
    """``check_ab_consensus`` is exported so stark.py and lifetimes.py can
    stop carrying private copies of these rules (audit: "one gate").

    Also pins the PAIR-CLASS SCOPING adopted in the 2026-08-10 integrator
    reconciliation: spec 02 §6's 1e-4 ceiling was measured on S<->P pairs
    only (B8 Rb 50S->50P3/2, B11 Cs 50S->50P3/2), so applying it to
    near-degenerate P<->D / D<->F pairs — which measure 1.0e-4..1.2e-4 —
    was an unsourced extrapolation. Those keep the sourced 1e-3 ceiling.
    Both regimes are asserted here so neither can drift.
    """
    orbit = 28.6 * 31.0
    df = dict(nu1=28.6, nu2=31.0, l1=2, l2=3, k=1, label="unit-DF")
    sp = dict(nu1=28.6, nu2=31.0, l1=0, l2=1, k=1, label="unit-SP")

    # suppressed element: absolute deviation gate in units of the orbit
    # scale, independent of pair class
    R.check_ab_consensus(0.018, 0.018 + 0.9e-4 * orbit, **df)
    with pytest.raises(IntegrityError, match="suppressed"):
        R.check_ab_consensus(0.018, 0.018 + 1.1e-4 * orbit, **df)

    # ordinary S<->P element: the measured 1e-4 row binds
    R.check_ab_consensus(1000.0, 1000.0 * (1 + 0.9e-4), **sp)
    with pytest.raises(IntegrityError):
        R.check_ab_consensus(1000.0, 1000.0 * (1 + 1.1e-4), **sp)

    # ordinary D<->F element: 1e-4 does NOT bind (that would be the
    # unsourced extrapolation), 1e-3 does
    R.check_ab_consensus(1000.0, 1000.0 * (1 + 1.1e-4), **df)
    with pytest.raises(IntegrityError):
        R.check_ab_consensus(1000.0, 1000.0 * (1 + 1.1e-3), **df)


# ---------------------------------------------------------------------------
# 6. THE MSD94 TRIPWIRE (audit R20 / spec 02 §6 B8)
#
# R20's declared mitigation is "B8 (A-vs-B spread <= 1e-4) catches any
# material potential error because Method B is potential-independent". That
# is inoperative for most of the table: everywhere else in this module the
# ENERGY is an input from spec 01's measured defects, so the model potential
# only shapes the wavefunction inside the divergence-guard truncation radius.
# Measured: doubling Rb a3(l=0) moves the 50S->50P consensus ME by 4e-7
# relative (A-B spread 2.02e-6 -> 2.40e-6, 40x inside the ceiling) and
# flipping the sign of the UNVERIFIED Rb a4(l=1) moves it by 6e-8 — the whole
# suite stays green under both.
#
# The quantity MSD94 actually fitted is the EIGENVALUE, so that is what has
# to be asserted. ``model_potential_defect`` solves the model potential as a
# genuine bound-state problem; truth is spec 01's measured quantum defects,
# which are independent published data, not this module's output.
# ---------------------------------------------------------------------------

#: Guard band on |delta_model - delta_measured| (l-centroids, n = 12).
#: Worst baseline residual over the eight rows below is 5.4e-3 (Cs P);
#: numerical wobble of delta_model is <= 1.1e-4 (h, r_min, spin-orbit
#: variations). 1.2e-2 is 2.2x the worst residual and 100x the wobble.
DEFECT_BAND = 1.2e-2
DEFECT_N = 12


def _model_centroid(sp: Species, n: int, l: int, params=None) -> float:
    """(2j+1)-weighted l-centroid of the MSD94-predicted defect.

    MSD94's spin-orbit term is the hydrogenic form, so the model splits a
    fine-structure doublet by ~4e-4 where experiment splits it by ~1.3e-2:
    only the centroid is a meaningful comparison (measured model Rb nP
    splitting 3.9e-4).
    """
    if l == 0:
        return R.model_potential_defect(sp, n, 0, 0.5, params=params)
    jm, jp = l - 0.5, l + 0.5
    dm = R.model_potential_defect(sp, n, l, jm, params=params)
    dp = R.model_potential_defect(sp, n, l, jp, params=params)
    return ((2 * jm + 1) * dm + (2 * jp + 1) * dp) / (2 * (2 * l + 1))


def _measured_centroid(sp: Species, n: int, l: int) -> float:
    """Same centroid from spec 01's measured quantum defects — the truth."""
    if l == 0:
        return n - float(n_star(sp, n, 0, 0.5))
    jm, jp = l - 0.5, l + 0.5
    dm = n - float(n_star(sp, n, l, jm))
    dp = n - float(n_star(sp, n, l, jp))
    return ((2 * jm + 1) * dm + (2 * jp + 1) * dp) / (2 * (2 * l + 1))


@pytest.mark.parametrize("sp,l", [(RB87, 0), (RB87, 1), (RB87, 2), (RB87, 3),
                                  (CS133, 0), (CS133, 1), (CS133, 2), (CS133, 3)])
def test_msd94_reproduces_measured_quantum_defects(sp: Species, l: int):
    """THE tripwire. The MSD94 potential must reproduce spec 01's measured
    quantum defects — the quantity it was fitted to — to within
    DEFECT_BAND.

    Measured residuals (model - measured, n = 12 centroids): Rb S +3.25e-4,
    P +2.18e-3, D +9.98e-4, F +8.47e-4; Cs S +5.64e-4, P +5.38e-3,
    D +1.32e-3, F -1.27e-4.
    """
    got = _model_centroid(sp, DEFECT_N, l)
    ref = _measured_centroid(sp, DEFECT_N, l)
    assert got == pytest.approx(ref, abs=DEFECT_BAND), (
        f"{sp.name} l={l}: model {got:.6f} vs measured {ref:.6f}")


def _perturbed(params, field: str, index: int, *, factor=None, value=None):
    row = list(getattr(params, field))
    row[index] = row[index] * factor if factor is not None else value
    return dataclasses.replace(params, **{field: tuple(row)})


# (species, l-row, field, perturbation, expected |deviation| once corrupted)
CORRUPTIONS = [
    (RB87, 0, "a3", dict(factor=2.0), 4.1e-1),      # the audit's own case
    (RB87, 1, "a4", dict(value=+0.81633314), 4.2e-2),   # sign flip
    (RB87, 0, "a1", dict(factor=1.05), 5.3e-2),
    (RB87, 0, "a2", dict(factor=1.05), 3.7e-2),
    (RB87, 0, "r_c", dict(factor=1.10), 1.8e-2),
    (RB87, 2, "a2", dict(factor=1.05), 1.2e-1),
    (RB87, 2, "a3", dict(factor=1.05), 7.8e-2),
    (CS133, 0, "a1", dict(factor=1.05), 6.8e-2),
    (CS133, 1, "a3", dict(factor=1.05), 4.4e-2),
    (CS133, 2, "r_c", dict(factor=1.05), 9.6e-2),
]


@pytest.mark.parametrize("sp,l,field,kw,expect_dev", CORRUPTIONS,
                         ids=[f"{s.name}-l{l}-{f}" for s, l, f, _, _ in CORRUPTIONS])
def test_msd94_parameter_corruption_is_detected(sp, l, field, kw, expect_dev):
    """...and the tripwire has teeth: each corruption drives the reproduced
    defect OUT of DEFECT_BAND, so the guard above would fail.

    This is the mutation test the audit asked for. Under the A-vs-B spread
    (R20's declared mitigation) every one of these passes silently.

    Honest scope statement — what this tripwire does NOT catch: the a4 rows
    are only weakly determined by the energies. Measured defect shift for a
    +1 % change: Rb a4(l=0) 1.0e-4, a4(l=1) 2.2e-4, a4(l=3) 3.2e-5; Cs
    a4(l=0) 1.7e-5, a4(l=3) 2.7e-8. Rb a4(l=1) has to move ~50 % before it
    breaches the band, which is why the O(1) sign flip is the case used
    here. The a1/a2/a3/r_c rows are caught at the few-percent level for
    l <= 2. See test_a4_l1_impact_bound_is_reproducible for the disputed
    digit itself.
    """
    base = (R.RB_MODEL_POTENTIAL if sp.name.startswith("Rb")
            else R.CS_MODEL_POTENTIAL)
    bad = _perturbed(base, field, l, **kw)
    got = _model_centroid(sp, DEFECT_N, l, params=bad)
    ref = _measured_centroid(sp, DEFECT_N, l)
    dev = abs(got - ref)
    assert dev > DEFECT_BAND, (
        f"{sp.name} {field}[{l}] {kw}: corrupted defect deviates only "
        f"{dev:.3e} — inside the {DEFECT_BAND:.1e} guard band, so the "
        "tripwire would not fire")
    assert dev == pytest.approx(expect_dev, rel=0.25), (
        f"{sp.name} {field}[{l}]: deviation {dev:.3e}, recorded {expect_dev:.1e}")


def test_msd94_defect_solver_refuses_when_the_state_is_not_bound_there():
    """A parameter table corrupted badly enough that the state leaves the
    search window must refuse, not return a neighbouring root."""
    bad = _perturbed(R.RB_MODEL_POTENTIAL, "a1", 0, factor=0.05)
    with pytest.raises(IntegrityError, match="MSD94 eigenvalue search"):
        R.model_potential_defect(RB87, DEFECT_N, 0, 0.5, params=bad)


def test_a4_l1_impact_bound_is_reproducible():
    """A4_L1_NOTE is exported, embedded in _MSD94_SOURCE and imported by
    rydsim.dipoles: it is a shipped, machine-readable integrity claim about
    the module's ONE unverified digit, so every number in it must be
    reproducible (house rule).

    The bound previously printed here and in spec 02 §3.3 / audit R20 —
    "delta-a4 = 8e-8 perturbs Z_1(r) by < 4e-8" — was wrong by 21.8x and
    6.3x. Both figures are regenerated from the two transcriptions.
    """
    adopted = -0.81633314          # ryd-numerov (in RB_MODEL_POTENTIAL)
    arc = -0.8163314               # ARC master
    assert R.RB_MODEL_POTENTIAL.a4[1] == adopted
    d_a4 = abs(adopted - arc)
    assert d_a4 == pytest.approx(1.74e-6, rel=1e-3)
    assert "1.74e-6" in R.A4_L1_NOTE

    a2 = R.RB_MODEL_POTENTIAL.a2[1]
    r = np.linspace(1e-3, 20.0, 200_001)
    d_z1 = d_a4 * r**2 * np.exp(-a2 * r)
    peak = float(np.max(d_z1))
    r_peak = float(r[int(np.argmax(d_z1))])
    assert peak == pytest.approx(2.53e-7, rel=1e-2)
    assert r_peak == pytest.approx(2.0 / a2, rel=1e-3)     # analytic maximum
    assert "2.53e-7" in R.A4_L1_NOTE and "1.04 a0" in R.A4_L1_NOTE

    # ...and the physical conclusion, on the quantity that IS sensitive to
    # the table: adopting the ARC reading moves the reproduced Rb P defect
    # by 4.7e-8, i.e. 2.6e5 times inside the tripwire's guard band.
    arc_params = _perturbed(R.RB_MODEL_POTENTIAL, "a4", 1, value=arc)
    base_d = R.model_potential_defect(RB87, DEFECT_N, 1, 1.5)
    arc_d = R.model_potential_defect(RB87, DEFECT_N, 1, 1.5, params=arc_params)
    assert abs(arc_d - base_d) < 1e-6
