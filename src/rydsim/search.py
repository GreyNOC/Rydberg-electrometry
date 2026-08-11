"""DESIGNER search layer (milestones D1-D3): sampling and frontier mapping.

Pure orchestration over `rydsim.objective` — every number reported here is
computed by the validated engine (DESIGNER §5 firewall). This module holds
the sampler, the oracle campaign runner, and the honest bookkeeping that a
published frontier requires: how many proposals were physically invalid and
why, and which points are oracle-measured (all of them, at this stage) vs
surrogate-interpolated (none yet — the surrogate lands at D2).
"""

from __future__ import annotations

import dataclasses
import itertools
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .objective import Evaluation, SensorDesign, evaluate, pareto_front


@dataclass
class CampaignResult:
    """Outcome of an oracle campaign over a design set."""

    evaluations: list[Evaluation]
    infeasible_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def feasible(self) -> list[Evaluation]:
        return [e for e in self.evaluations if e.feasible]

    @property
    def n_total(self) -> int:
        return len(self.evaluations)

    @property
    def n_feasible(self) -> int:
        return len(self.feasible)

    def frontier(self, keys: tuple[str, ...] = ("nef", "neg_ibw")) -> list[Evaluation]:
        return pareto_front(self.evaluations, keys=keys)

    def summary(self) -> str:
        lines = [
            f"campaign: {self.n_feasible}/{self.n_total} designs feasible "
            f"({100*self.n_feasible/max(self.n_total,1):.0f}%)",
        ]
        for reason, count in sorted(self.infeasible_reasons.items(),
                                    key=lambda kv: -kv[1]):
            lines.append(f"  rejected x{count}: {reason[:90]}")
        if self.feasible:
            best_nef = min(self.feasible, key=lambda e: e.nef)
            best_ibw = max(self.feasible, key=lambda e: e.ibw_hz)
            lines.append(
                f"best NEF: {best_nef.nef_nv_per_cm_rthz:.3g} nV/cm/rtHz "
                f"(IBW {best_ibw.ibw_hz/1e6:.3g} MHz at the IBW-optimal point)")
            lines.append(f"Pareto frontier: {len(self.frontier())} points")
        return "\n".join(lines)


def grid_designs(base: SensorDesign, **axes) -> list[SensorDesign]:
    """Cartesian product of the named SensorDesign fields over given values.

    Example: grid_designs(base, coupling_rabi_hz=[2e6, 5e6],
                                temperature_k=[300, 330])
    """
    valid = {f.name for f in dataclasses.fields(SensorDesign)}
    unknown = set(axes) - valid
    if unknown:
        raise ValueError(f"unknown SensorDesign fields: {sorted(unknown)}")
    names = list(axes)
    out = []
    for combo in itertools.product(*(axes[n] for n in names)):
        out.append(dataclasses.replace(base, **dict(zip(names, combo))))
    return out


def random_designs(base: SensorDesign, n_samples: int,
                   ranges: dict[str, tuple[float, float]],
                   seed: int = 0, log_scale: set[str] | None = None
                   ) -> list[SensorDesign]:
    """Latin-hypercube-ish random sampling over continuous design axes.

    Deterministic given `seed` (reproducibility rule). `log_scale` names
    axes sampled uniformly in log space (Rabi frequencies, powers).
    """
    valid = {f.name for f in dataclasses.fields(SensorDesign)}
    unknown = set(ranges) - valid
    if unknown:
        raise ValueError(f"unknown SensorDesign fields: {sorted(unknown)}")
    rng = np.random.default_rng(seed)
    log_scale = log_scale or set()
    names = list(ranges)
    # stratified (Latin hypercube) draws per axis
    strata = {
        nm: rng.permutation((np.arange(n_samples) + rng.random(n_samples))
                            / n_samples)
        for nm in names
    }
    out = []
    for i in range(n_samples):
        kw = {}
        for nm in names:
            lo, hi = ranges[nm]
            u = strata[nm][i]
            if nm in log_scale:
                kw[nm] = float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))))
            else:
                kw[nm] = float(lo + u * (hi - lo))
        out.append(dataclasses.replace(base, **kw))
    return out


def run_campaign(designs: list[SensorDesign], **eval_kw) -> CampaignResult:
    """Evaluate a design set through the oracle, tracking rejection reasons.

    Infeasible points are counted and reported, never silently dropped —
    a frontier is only meaningful alongside the fraction of the space that
    was physically invalid.
    """
    evals = [evaluate(d, **eval_kw) for d in designs]
    reasons = Counter(e.reason for e in evals if not e.feasible)
    return CampaignResult(evaluations=evals, infeasible_reasons=dict(reasons))


def frontier_table(front: list[Evaluation]) -> str:
    """Human-readable Pareto table, sorted by NEF."""
    rows = sorted(front, key=lambda e: e.nef)
    head = (f"{'NEF [nV/cm/rtHz]':>18} {'IBW [MHz]':>11} {'E_LO [V/m]':>11} "
            f"{'DR [dB]':>8}  design")
    out = [head, "-" * len(head)]
    for e in rows:
        d = e.design
        label = (f"{d.species} n={d.n} {d.state or ''} "
                 f"Oc={d.coupling_rabi_hz/1e6:.2f}MHz T={d.temperature_k:.0f}K")
        out.append(
            f"{e.nef_nv_per_cm_rthz:>18.4g} {e.ibw_hz/1e6:>11.4g} "
            f"{e.e_lo:>11.4g} {(e.dynamic_range_db or float('nan')):>8.1f}  {label}")
    return "\n".join(out)
