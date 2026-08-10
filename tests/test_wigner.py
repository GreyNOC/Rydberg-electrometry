"""Validation: rydsim.wigner vs exact symbolic values (sympy) + sum rules."""

import itertools
import math

import pytest
from sympy import Rational, S
from sympy.physics.wigner import wigner_3j as sy3j, wigner_6j as sy6j
from sympy.physics.quantum.cg import CG

from rydsim.wigner import clebsch_gordan, wigner_3j, wigner_6j


def _r(x: float) -> Rational:
    return Rational(round(2 * x), 2)


def test_3j_grid_vs_sympy():
    """Exhaustive check over all valid (j,m) with j up to 4 (integer + half)."""
    js = [x / 2 for x in range(0, 9)]  # 0 .. 4 in half steps
    n_checked = 0
    for j1, j2 in itertools.product(js, repeat=2):
        j3min, j3max = abs(j1 - j2), j1 + j2
        j3 = j3min
        while j3 <= j3max + 1e-9:
            for m1 in [x / 2 for x in range(-round(2 * j1), round(2 * j1) + 1, 2)]:
                for m2 in [x / 2 for x in range(-round(2 * j2), round(2 * j2) + 1, 2)]:
                    m3 = -(m1 + m2)
                    if abs(m3) > j3 + 1e-9:
                        continue
                    ours = wigner_3j(j1, j2, j3, m1, m2, m3)
                    exact = float(sy3j(_r(j1), _r(j2), _r(j3), _r(m1), _r(m2), _r(m3)))
                    assert ours == pytest.approx(exact, abs=1e-12), (
                        f"3j({j1},{j2},{j3};{m1},{m2},{m3}) = {ours} != {exact}"
                    )
                    n_checked += 1
            j3 += 1.0
    assert n_checked > 500  # make sure the sweep actually ran


def test_3j_known_value():
    # (1 1 1; 1 -1 0) — a standard tabulated value: 1/sqrt(6)
    assert wigner_3j(1, 1, 1, 1, -1, 0) == pytest.approx(1 / math.sqrt(6), rel=1e-14)


def test_3j_large_j_stability():
    """Log-space evaluation must survive j ~ 100 (Rydberg-Rydberg regime)."""
    ours = wigner_3j(100, 100, 0, 50, -50, 0)
    # (j j 0; m -m 0) = (-1)^(j-m)/sqrt(2j+1) exactly
    exact = (-1) ** (100 - 50) / math.sqrt(201)
    assert ours == pytest.approx(exact, rel=1e-10)


def test_6j_grid_vs_sympy():
    """Check all valid 6j with entries up to 3 in half steps."""
    js = [x / 2 for x in range(0, 7)]  # 0 .. 3
    n_checked = 0
    for j1, j2, j3, j4, j5, j6 in itertools.product(js, repeat=6):
        ours = wigner_6j(j1, j2, j3, j4, j5, j6)
        try:
            exact = float(sy6j(_r(j1), _r(j2), _r(j3), _r(j4), _r(j5), _r(j6)))
        except ValueError:
            # sympy raises on violated triangle rules; ours returns 0
            assert ours == 0.0
            continue
        assert ours == pytest.approx(exact, abs=1e-12), (
            f"6j({j1},{j2},{j3};{j4},{j5},{j6}) = {ours} != {exact}"
        )
        n_checked += 1
    assert n_checked > 100


def test_6j_physical_case():
    """The 6j entering alkali fine-structure dipole reductions (l s j basis)."""
    # {1/2 1 3/2 / 2 5/2 1} appears in D2 nP3/2 -> nD5/2 reductions
    ours = wigner_6j(0.5, 1, 1.5, 2, 2.5, 1)
    exact = float(sy6j(S(1) / 2, 1, S(3) / 2, 2, S(5) / 2, 1))
    assert ours == pytest.approx(exact, abs=1e-14)


def test_cg_vs_sympy():
    for (j1, m1, j2, m2, j3, m3) in [
        (0.5, 0.5, 0.5, -0.5, 1, 0),
        (0.5, 0.5, 0.5, -0.5, 0, 0),
        (1, 0, 1, 0, 2, 0),
        (1.5, 0.5, 1, -1, 0.5, -0.5),
        (2, 1, 1.5, 0.5, 2.5, 1.5),
    ]:
        ours = clebsch_gordan(j1, m1, j2, m2, j3, m3)
        exact = float(CG(_r(j1), _r(m1), _r(j2), _r(m2), _r(j3), _r(m3)).doit())
        assert ours == pytest.approx(exact, abs=1e-13), (j1, m1, j2, m2, j3, m3)


def test_3j_orthogonality_sum_rule():
    """sum_{m1 m2} (2j3+1) 3j(...m3)^2 = 1 for each valid (j3, m3)."""
    j1, j2 = 2, 1.5
    j3 = 2.5
    for m3 in [-2.5, -0.5, 1.5]:
        total = 0.0
        for m1 in [-2, -1, 0, 1, 2]:
            for m2 in [-1.5, -0.5, 0.5, 1.5]:
                if abs(m1 + m2 + m3) < 1e-9:
                    total += (2 * j3 + 1) * wigner_3j(j1, j2, j3, m1, m2, m3) ** 2
        assert total == pytest.approx(1.0, rel=1e-12)


def test_selection_rules_zero():
    assert wigner_3j(1, 1, 3, 0, 0, 0) == 0.0          # triangle violated
    assert wigner_3j(1, 1, 1, 1, 1, 1) == 0.0          # m-sum nonzero
    assert wigner_3j(1, 1, 1, 0, 0, 0) == 0.0          # odd-perimeter zero (j1+j2+j3 odd rule)
    assert wigner_6j(1, 1, 3, 1, 1, 1) == 0.0          # triangle violated


def test_half_integer_validation():
    with pytest.raises(ValueError):
        wigner_3j(0.3, 1, 1, 0, 0, 0)
