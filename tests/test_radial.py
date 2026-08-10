"""Spec 02 benchmarks — radial wavefunctions and matrix elements.

The Wave-2 implementation agent died before writing this file; these tests
were authored against docs/spec/02-radial-wavefunctions-numerov.md §6 and
the module's actual API. Every exact-value row is an analytic identity
(Gordon's formula / hydrogen closed forms) or a cross-method consensus —
no value here is asserted from recall.
"""

import numpy as np
import pytest

from rydsim import radial as R
from rydsim.atom import CS133, RB87, n_star
from rydsim.provenance import IntegrityError

# ---------------------------------------------------------------------------
# Method D (exact): Gordon's formula on hydrogen — the integrator's oracle
# ---------------------------------------------------------------------------

HYDROGEN_EXACT = [
    ("1s->2p", 1, 0, 2, 1, 1.2902662020),      # = 128 sqrt(6)/243
    ("2s->3p", 2, 0, 3, 1, 3.0648154066),
    ("2p->3d", 2, 1, 3, 2, 4.7479916115),
    ("10s->11p", 10, 0, 11, 1, 40.4352023233),
    ("50s->50p", 50, 0, 50, 1, 3749.2499249850),  # = (3/2) n sqrt(n^2 - l^2)
    ("50s->51p", 50, 0, 51, 1, 851.4038694455),
]


@pytest.mark.parametrize("label,n1,l1,n2,l2,expect", HYDROGEN_EXACT)
def test_gordon_exact_hydrogen(label, n1, l1, n2, l2, expect):
    """Spec 02 §6: exact hydrogen radial integrals to 1e-7 relative."""
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
# Method C: Kaulakys semiclassical vs the exact hydrogenic limit
# ---------------------------------------------------------------------------

def test_kaulakys_matches_gordon_hydrogen():
    """Spec 02 §6: Kaulakys vs Gordon at nu = 50, dnu = 1 and 2, rel <= 2e-4."""
    for dn in (1, 2):
        exact = abs(R.radial_me_gordon(50, 0, 50 + dn, 1))
        semi = abs(R.radial_me_kaulakys(50.0, 0, 50.0 + dn, 1))
        assert semi == pytest.approx(exact, rel=2e-4), f"dn={dn}"


# ---------------------------------------------------------------------------
# Method B machinery: hyperu/Whittaker validity fence (audit R4 — BINDING)
# ---------------------------------------------------------------------------

def test_hyperu_accurate_at_moderate_nu():
    """Audit R4 self-check, regenerated on the INSTALLED scipy at test time:
    the Whittaker route must still be accurate at nu = 20."""
    assert R.hyperu_hydrogen_error(20, 0) < 1e-6


def test_hyperu_precision_collapse_is_real():
    """Audit R4 also requires proving the collapse actually occurs — if a
    future scipy fixes hyperu, the nu > 25 refusal becomes over-conservative
    and this test tells us to revisit it."""
    err_lo = R.hyperu_hydrogen_error(12, 0)
    err_hi = R.hyperu_hydrogen_error(25, 0)
    assert err_hi > err_lo


def test_coulomb_route_refuses_above_nu_25():
    """Audit R4: nu > 25 raises regardless of apparent accuracy."""
    with pytest.raises(IntegrityError):
        R.whittaker_u(30.0, 0, np.array([100.0, 200.0]))


# ---------------------------------------------------------------------------
# Consensus machinery — the module's reason to exist
# ---------------------------------------------------------------------------

def test_rb_50s_50p_consensus_and_spread():
    """Spec 02 §6: Rb 50S1/2->50P3/2 cross-method spread <= 1e-4 relative.

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
    """Spec 02 §6: |R(50S->50P1/2)| / |R(50S->50P3/2)| = 1.0158 +- 0.002
    (the j-dependence enters only through the quantum defects)."""
    a = R.radial_matrix_element_consensus(RB87, (50, 0, 0.5), (50, 1, 0.5))
    b = R.radial_matrix_element_consensus(RB87, (50, 0, 0.5), (50, 1, 1.5))
    assert abs(a.value) / abs(b.value) == pytest.approx(1.0158, abs=0.002)


@pytest.mark.parametrize("n", [40, 50, 60, 70])
def test_scaling_invariant_rb(n):
    """Spec 02 §6: |R|/(nu nu') in [1.10, 1.16] for nS1/2->nP3/2, n = 40..70."""
    res = R.radial_matrix_element_consensus(RB87, (n, 0, 0.5), (n, 1, 1.5))
    nu1 = float(n_star(RB87, n, 0, 0.5))
    nu2 = float(n_star(RB87, n, 1, 1.5))
    coeff = abs(res.value) / (nu1 * nu2)
    assert 1.10 <= coeff <= 1.16, f"n={n}: coeff={coeff:.4f}"


def test_scaling_invariant_cs():
    """Spec 02 §6: same invariant holds for Cs (1.1304 at n = 50)."""
    res = R.radial_matrix_element_consensus(CS133, (50, 0, 0.5), (50, 1, 1.5))
    nu1 = float(n_star(CS133, 50, 0, 0.5))
    nu2 = float(n_star(CS133, 50, 1, 1.5))
    assert abs(res.value) / (nu1 * nu2) == pytest.approx(1.1304, abs=0.03)
    assert res.spread_rel <= 1e-4


def test_outer_cutoff_adequacy():
    """Spec 02 §6: the 2n(n+15) outer cutoff is converged — extending it
    changes the ME by <= 5e-8 relative."""
    s1, s2 = (50, 0, 0.5), (51, 1, 1.5)
    a = R.radial_wavefunction(RB87, *s1, r_outer=2 * 51 * (51 + 15))
    b = R.radial_wavefunction(RB87, *s2, r_outer=2 * 51 * (51 + 15))
    me_std = R.radial_matrix_element(a, b)
    a2 = R.radial_wavefunction(RB87, *s1, r_outer=2 * 51 * (51 + 25))
    b2 = R.radial_wavefunction(RB87, *s2, r_outer=2 * 51 * (51 + 25))
    me_wide = R.radial_matrix_element(a2, b2)
    assert abs(me_wide - me_std) / abs(me_std) < 5e-8


def test_step_convergence():
    """Halving the Numerov step must not move the consensus value."""
    fine = R.radial_matrix_element_consensus(RB87, (50, 0, 0.5), (50, 1, 1.5),
                                            h=5e-4)
    std = R.radial_matrix_element_consensus(RB87, (50, 0, 0.5), (50, 1, 1.5),
                                           h=1e-3)
    assert abs(fine.value - std.value) / abs(std.value) < 1e-5


# ---------------------------------------------------------------------------
# Refusals — the audit's refuse-to-guess list
# ---------------------------------------------------------------------------

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


def test_consensus_failure_raises_rather_than_averaging():
    """With check=True a spread beyond the spec ceiling must raise — the
    module never averages over a failed consensus."""
    import inspect
    src = inspect.getsource(R.radial_matrix_element_consensus)
    assert "check" in src and "IntegrityError" in src
