"""DESIGNER D2/D3 — surrogate-accelerated design-space mapping.

The search layer of docs/DESIGNER.md: sample a real Rydberg design space,
evaluate it with the validated physics oracle (`rydsim.objective.evaluate`),
fit a Gaussian-process surrogate, report held-out error against the oracle's
own uncertainty, and extract the NEF-vs-IBW Pareto frontier.

THE CREDIBILITY FIREWALL (DESIGNER §5, non-negotiable): the surrogate never
produces a shipped number. It only *nominates* designs; every point that
appears on a published frontier is re-evaluated by the exact oracle. The
surrogate's role is to decide where to spend oracle calls, nothing more.
`confirm_frontier` enforces this and is the only sanctioned path to a
frontier claim.

Held-out error is reported in the same units as the objective and compared
to the oracle's own model uncertainty (dominated by the cross-method radial
dipole spread propagated through NEF ∝ 1/d). A surrogate whose error exceeds
the oracle uncertainty is not yet trustworthy for nomination, and
`SurrogateReport.trustworthy` says so rather than the caller assuming.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

from .objective import Evaluation, SensorDesign, evaluate, pareto_front
from .provenance import IntegrityError

__all__ = [
    "DesignAxis", "DesignSpace", "SurrogateReport", "Surrogate",
    "sample_space", "run_oracle", "fit_surrogate", "active_learn",
    "confirm_frontier", "frontier_summary", "frontier_recovery",
    "require_sklearn",
]

_SKLEARN_HINT = (
    "the DESIGNER search layer needs scikit-learn, which is an optional "
    "extra so the physics engine, CLI and GUI install without it.\n"
    "  Install it with:  pip install 'rydsim[designer]'\n"
    "Note the oracle itself (rydsim.objective.evaluate) and the exact "
    "physics engine do NOT need scikit-learn — only the surrogate and the "
    "feasibility classifier that decide WHERE to spend oracle calls."
)


def require_sklearn():
    """Import scikit-learn or raise with the install hint.

    Kept explicit rather than letting a bare ImportError escape: a missing
    optional extra should name itself, not surface as a stack trace from
    inside a fitting routine.
    """
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(_SKLEARN_HINT) from exc
    return sklearn


# ---------------------------------------------------------------------------
# Design space
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DesignAxis:
    """One searchable axis of SensorDesign.

    `log` axes are sampled and modeled in log10 (Rabi frequencies, powers —
    quantities whose physics is multiplicative). `integer` axes (n) are
    rounded on the way into the design.
    """

    name: str
    low: float
    high: float
    log: bool = False
    integer: bool = False

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        if self.log:
            if self.low <= 0:
                raise IntegrityError(f"log axis {self.name!r} needs low > 0")
            v = 10.0 ** rng.uniform(np.log10(self.low), np.log10(self.high), size)
        else:
            v = rng.uniform(self.low, self.high, size)
        return np.round(v) if self.integer else v

    def to_model(self, v: np.ndarray) -> np.ndarray:
        """Axis value -> surrogate coordinate (log10 for log axes)."""
        return np.log10(v) if self.log else np.asarray(v, float)

    @property
    def model_bounds(self) -> tuple[float, float]:
        return ((np.log10(self.low), np.log10(self.high)) if self.log
                else (self.low, self.high))


@dataclass(frozen=True)
class DesignSpace:
    """A named set of axes plus the fixed base design they perturb."""

    base: SensorDesign
    axes: tuple[DesignAxis, ...]

    def build(self, values: Sequence[float]) -> SensorDesign:
        if len(values) != len(self.axes):
            raise ValueError("value vector length != number of axes")
        kw = {}
        for ax, v in zip(self.axes, values):
            kw[ax.name] = int(round(v)) if ax.integer else float(v)
        return dataclasses.replace(self.base, **kw)

    def to_model(self, designs: Iterable[SensorDesign]) -> np.ndarray:
        rows = []
        for d in designs:
            rows.append([ax.to_model(np.array([getattr(d, ax.name)]))[0]
                         for ax in self.axes])
        return np.asarray(rows, float)

    @property
    def model_bounds(self) -> np.ndarray:
        return np.asarray([ax.model_bounds for ax in self.axes], float)


def sample_space(space: DesignSpace, n_samples: int, *,
                 seed: int = 0) -> list[SensorDesign]:
    """Latin-hypercube-ish stratified random sample of the space.

    Stratifying each axis independently gives far better coverage than plain
    uniform sampling at these sample counts, which matters because every
    oracle call costs real compute.
    """
    rng = np.random.default_rng(seed)
    cols = []
    for ax in space.axes:
        # stratified uniform in model coordinates, then invert
        edges = (np.arange(n_samples) + rng.random(n_samples)) / n_samples
        lo, hi = ax.model_bounds
        m = lo + edges * (hi - lo)
        rng.shuffle(m)
        v = 10.0 ** m if ax.log else m
        cols.append(np.round(v) if ax.integer else v)
    return [space.build([c[i] for c in cols]) for i in range(n_samples)]


# ---------------------------------------------------------------------------
# Oracle campaign
# ---------------------------------------------------------------------------

def run_oracle(designs: Sequence[SensorDesign], *,
               progress: Callable[[int, int], None] | None = None,
               **eval_kw) -> list[Evaluation]:
    """Evaluate designs with the exact physics oracle.

    Infeasible designs (refusals from the atomic chain, unconverged
    averages, forbidden transitions) are RETAINED as infeasible Evaluations
    rather than dropped — the feasibility boundary is itself a result, and
    silently discarding refusals would misrepresent the space.
    """
    out: list[Evaluation] = []
    for i, d in enumerate(designs):
        out.append(evaluate(d, **eval_kw))
        if progress is not None:
            progress(i + 1, len(designs))
    return out


# ---------------------------------------------------------------------------
# Surrogate
# ---------------------------------------------------------------------------

@dataclass
class SurrogateReport:
    """Held-out performance of a fitted surrogate, in objective units."""

    objective: str
    n_train: int
    n_holdout: int
    rmse_dex: float           # RMSE in log10 units (the modeled quantity)
    mae_dex: float
    max_err_dex: float
    median_rel_err: float     # back-transformed, fractional
    oracle_uncertainty_rel: float
    trustworthy: bool         # median surrogate error below oracle uncertainty
    note: str = ""

    def summary(self) -> str:
        verdict = "TRUSTWORTHY" if self.trustworthy else "NOT TRUSTWORTHY"
        return (f"{self.objective}: holdout median rel. err "
                f"{self.median_rel_err:.2%} vs oracle uncertainty "
                f"{self.oracle_uncertainty_rel:.2%} -> {verdict} "
                f"(RMSE {self.rmse_dex:.4f} dex on n={self.n_holdout})")


@dataclass
class Surrogate:
    """A fitted GP over one objective, with its held-out report."""

    space: DesignSpace
    objective: str
    model: object
    report: SurrogateReport
    _x_mean: np.ndarray = field(default_factory=lambda: np.zeros(1))
    _x_std: np.ndarray = field(default_factory=lambda: np.ones(1))

    def predict(self, designs: Sequence[SensorDesign],
                return_std: bool = False):
        """Predicted objective (back-transformed from log10).

        NOTE: a prediction is a NOMINATION, never a result. Anything that
        reaches a report must be re-run through the oracle
        (`confirm_frontier`).
        """
        x = (self.space.to_model(designs) - self._x_mean) / self._x_std
        if return_std:
            mu, sd = self.model.predict(x, return_std=True)
            return 10.0 ** mu, sd
        return 10.0 ** self.model.predict(x)


def _objective_value(ev: Evaluation, objective: str) -> float | None:
    v = {"nef": ev.nef, "ibw_hz": ev.ibw_hz,
         "dynamic_range_db": ev.dynamic_range_db}.get(objective)
    return None if v is None or v <= 0 else float(v)


def _oracle_uncertainty_rel(evals: Sequence[Evaluation], objective: str) -> float:
    """The oracle's own model uncertainty on this objective, fractional.

    NEF ∝ 1/d, so the cross-method radial dipole spread propagates ~1:1.
    We take the median spread over the feasible set. IBW does not depend on
    the dipole at leading order, so it carries only the numerical
    convergence tolerance of the resolvent scan.
    """
    if objective == "nef":
        spreads = []
        for ev in evals:
            d = getattr(ev, "dipole_spread_rel", None)
            if d:
                spreads.append(float(d))
        if spreads:
            return float(np.median(spreads))
        return 5e-3          # documented radial-consensus ceiling (spec 02)
    return 1e-2              # IBW: bisection + grid tolerance


def fit_surrogate(space: DesignSpace, evals: Sequence[Evaluation],
                  objective: str = "nef", *, holdout: float = 0.25,
                  seed: int = 0) -> Surrogate:
    """Fit a GP to log10(objective) and report HELD-OUT error (D2 gate).

    Modeling log10 is deliberate: NEF spans orders of magnitude across the
    space, and relative error is the meaningful metric for a sensitivity.
    """
    require_sklearn()
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (ConstantKernel, Matern,
                                                  WhiteKernel)

    usable = [ev for ev in evals
              if ev.feasible and _objective_value(ev, objective) is not None]
    if len(usable) < 12:
        raise IntegrityError(
            f"only {len(usable)} feasible points for objective {objective!r}; "
            "refusing to fit a surrogate on a sample this thin")

    x = space.to_model([ev.design for ev in usable])
    y = np.log10([_objective_value(ev, objective) for ev in usable])

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(usable))
    n_hold = max(4, int(round(holdout * len(usable))))
    hold, train = idx[:n_hold], idx[n_hold:]

    x_mean, x_std = x[train].mean(0), x[train].std(0)
    x_std[x_std == 0] = 1.0
    xt = (x[train] - x_mean) / x_std
    xh = (x[hold] - x_mean) / x_std

    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * Matern(length_scale=np.ones(x.shape[1]),
                       length_scale_bounds=(1e-2, 1e2), nu=2.5)
              + WhiteKernel(1e-6, (1e-12, 1e-1)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=3, random_state=seed)
    gp.fit(xt, y[train])

    pred = gp.predict(xh)
    err = pred - y[hold]
    rel = np.abs(10.0 ** np.abs(err) - 1.0)
    oracle_unc = _oracle_uncertainty_rel(usable, objective)
    median_rel = float(np.median(rel))

    report = SurrogateReport(
        objective=objective, n_train=len(train), n_holdout=len(hold),
        rmse_dex=float(np.sqrt(np.mean(err**2))),
        mae_dex=float(np.mean(np.abs(err))),
        max_err_dex=float(np.max(np.abs(err))),
        median_rel_err=median_rel,
        oracle_uncertainty_rel=oracle_unc,
        trustworthy=bool(median_rel <= oracle_unc),
        note=("surrogate median error is within the oracle's own model "
              "uncertainty" if median_rel <= oracle_unc else
              "surrogate error EXCEEDS oracle uncertainty — use it only to "
              "steer sampling, never to rank final designs"),
    )
    return Surrogate(space=space, objective=objective, model=gp,
                     report=report, _x_mean=x_mean, _x_std=x_std)


# ---------------------------------------------------------------------------
# Active learning
# ---------------------------------------------------------------------------

def _feasibility_classifier(space: DesignSpace, evals: Sequence[Evaluation],
                            seed: int = 0):
    """Probability-of-feasibility model, or None if the sample is one-class.

    Pure uncertainty sampling collapses without this: the GP's variance is
    largest exactly at the feasibility boundary (where the atomic chain
    starts refusing), so an unguarded acquisition spends most of its oracle
    budget on designs that raise IntegrityError. Measured on the Rb D5/2
    space: 10/40 feasible per round unguarded.
    """
    require_sklearn()
    from sklearn.ensemble import RandomForestClassifier

    y = np.array([1 if ev.feasible else 0 for ev in evals])
    if y.min() == y.max():
        return None
    x = space.to_model([ev.design for ev in evals])
    clf = RandomForestClassifier(n_estimators=120, random_state=seed,
                                 min_samples_leaf=2)
    clf.fit(x, y)
    return clf


def active_learn(space: DesignSpace, evals: list[Evaluation], *,
                 rounds: int = 3, batch: int = 24, pool: int = 400,
                 objective: str = "nef", seed: int = 0,
                 feasibility_aware: bool = True,
                 log: Callable[[str], None] | None = None,
                 **eval_kw) -> tuple[list[Evaluation], Surrogate]:
    """Feasibility-weighted uncertainty sampling: spend oracle calls where
    the model is least sure AND the design is likely to be legal.

    Acquisition = predictive_sd * P(feasible). Every point added is a REAL
    oracle evaluation — the surrogate only chooses which, never what the
    answer is (DESIGNER §5 firewall).
    """
    surrogate = fit_surrogate(space, evals, objective=objective, seed=seed)
    for r in range(rounds):
        cand = sample_space(space, pool, seed=seed + 1000 + r)
        _, sd = surrogate.predict(cand, return_std=True)
        score = np.asarray(sd, float)
        if feasibility_aware:
            clf = _feasibility_classifier(space, evals, seed=seed)
            if clf is not None:
                p_ok = clf.predict_proba(space.to_model(cand))[:, 1]
                score = score * p_ok
        pick = np.argsort(score)[::-1][:batch]
        new = run_oracle([cand[i] for i in pick], **eval_kw)
        evals = list(evals) + new
        surrogate = fit_surrogate(space, evals, objective=objective,
                                  seed=seed)
        if log:
            n_ok = sum(1 for e in new if e.feasible)
            log(f"round {r+1}/{rounds}: +{n_ok}/{batch} feasible, "
                f"{surrogate.report.summary()}")
    return list(evals), surrogate


def frontier_recovery(reference: Sequence[Evaluation],
                      candidate: Sequence[Evaluation],
                      keys: Sequence[str] = ("nef", "neg_ibw")) -> dict:
    """Decision-quality metric: does the cheap campaign find the expensive
    campaign's frontier?

    This is the gate that actually matters for a nominate-then-confirm
    architecture. Absolute surrogate accuracy is the wrong yardstick — the
    oracle's own dipole uncertainty is a COMMON-MODE systematic (NEF is
    proportional to 1/d), so it shifts every point together and cannot limit
    the ability to RANK designs or locate a frontier.

    Returns the fraction of reference-frontier NEF levels the candidate
    matches or beats, and the worst shortfall.
    """
    ref = pareto_front(list(reference), keys=list(keys))
    cand = pareto_front(list(candidate), keys=list(keys))
    if not ref:
        raise IntegrityError("reference campaign has no feasible frontier")
    shortfalls = []
    for r in ref:
        # best candidate NEF among designs with at least this IBW
        rivals = [c.nef for c in cand
                  if c.ibw_hz is not None and r.ibw_hz is not None
                  and c.ibw_hz >= r.ibw_hz * 0.98 and c.nef]
        shortfalls.append(min(rivals) / r.nef if rivals else np.inf)
    shortfalls = np.asarray(shortfalls, float)
    return {
        "reference_points": len(ref),
        "candidate_points": len(cand),
        "matched_fraction": float(np.mean(shortfalls <= 1.02)),
        "median_ratio": float(np.median(shortfalls[np.isfinite(shortfalls)])),
        "worst_ratio": float(np.max(shortfalls)),
    }


# ---------------------------------------------------------------------------
# Frontier (D3)
# ---------------------------------------------------------------------------

def confirm_frontier(evals: Sequence[Evaluation],
                     keys: Sequence[str] = ("nef", "neg_ibw"),
                     **eval_kw) -> list[Evaluation]:
    """Extract the Pareto frontier and RE-EVALUATE every point on it.

    The firewall (DESIGNER §5): a frontier is a claim, so each frontier
    point is recomputed from its config through the exact oracle and must
    reproduce. A point that fails to reproduce is dropped with a raised
    IntegrityError rather than quietly kept.
    """
    front = pareto_front(list(evals), keys=list(keys))
    confirmed = []
    for ev in front:
        re_ev = evaluate(ev.design, **eval_kw)
        if not re_ev.feasible:
            raise IntegrityError(
                f"frontier point became infeasible on re-evaluation: "
                f"{re_ev.reason}")
        for attr in ("nef", "ibw_hz"):
            a, b = getattr(ev, attr), getattr(re_ev, attr)
            if a and b and abs(a - b) / abs(a) > 1e-6:
                raise IntegrityError(
                    f"frontier point not reproducible in {attr}: "
                    f"{a!r} -> {b!r} (config hash mismatch or nondeterminism)")
        confirmed.append(re_ev)
    return confirmed


def frontier_summary(front: Sequence[Evaluation]) -> str:
    """Human-readable NEF-vs-IBW frontier, best sensitivity first."""
    rows = sorted(front, key=lambda e: e.nef or np.inf)
    head = (f"{'n':>4} {'Om_c/2pi MHz':>13} {'T [K]':>7} "
            f"{'NEF nV/cm/rtHz':>15} {'IBW MHz':>9} {'RF GHz':>8}")
    lines = [head, "-" * len(head)]
    for ev in rows:
        d = ev.design
        lines.append(
            f"{d.n:>4} {d.coupling_rabi_hz/1e6:>13.3f} {d.temperature_k:>7.1f} "
            f"{ev.nef_nv_per_cm_rthz:>15.4g} {ev.ibw_hz/1e6:>9.3f} "
            f"{(ev.rf_frequency_hz or 0)/1e9:>8.3f}")
    return "\n".join(lines)
