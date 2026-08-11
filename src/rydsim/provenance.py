"""Provenance and scientific-integrity primitives.

Implements the GreyNOC house rule for RydSim: every constant carries a source
and confidence tag; every finding carries the provenance of everything on its
critical path. See docs/spec/00-integrity-audit.md and docs/COLLABORATION.md.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import platform
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Confidence(str, Enum):
    """Confidence tag for a sourced value (spec integrity policy)."""

    VERIFIED = "VERIFIED"                  # checked against primary source
    VERIFIED_ARC = "VERIFIED-ARC"          # matches ARC/secondary code, primary unfetched
    LITERATURE_RECALL = "LITERATURE-RECALL"  # standard literature value, not re-verified
    UNVERIFIED = "UNVERIFIED"              # from memory / provisional — flagged in outputs
    COMPUTED = "COMPUTED"                  # derived inside RydSim from tagged inputs
    MEASURED_DISAGREEMENT = "MEASURED-DISAGREEMENT"  # value with cross-method uncertainty


@dataclass(frozen=True)
class SourcedValue:
    """A numeric value that knows where it came from.

    Arithmetic intentionally NOT overloaded: call ``.value`` explicitly so
    that every use-site is a visible provenance boundary.
    """

    value: float
    unit: str
    source: str
    confidence: Confidence = Confidence.LITERATURE_RECALL
    uncertainty: float | None = None   # 1-sigma, same unit; None = not stated
    note: str = ""

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("SourcedValue requires a non-empty source (no-fabrication rule)")

    def __str__(self) -> str:
        u = f" ± {self.uncertainty:g}" if self.uncertainty is not None else ""
        return f"{self.value:g}{u} {self.unit} [{self.confidence.value}: {self.source}]"


class IntegrityError(RuntimeError):
    """Raised when a mandatory self-check fails.

    RydSim refuses to produce a number rather than guess: cross-method
    disagreement beyond tolerance, out-of-validity-domain evaluation, or an
    UNVERIFIED constant on a critical path in strict mode all raise this.
    """


@dataclass
class CrossCheck:
    """Result of computing one quantity by independent methods."""

    quantity: str
    methods: dict[str, float]           # method name -> value (common unit)
    unit: str
    tolerance_rel: float                # allowed relative spread before failing

    @property
    def spread_rel(self) -> float:
        vals = [v for v in self.methods.values() if v == v]  # drop NaN
        if len(vals) < 2:
            return 0.0
        lo, hi = min(vals), max(vals)
        scale = max(abs(lo), abs(hi))
        return 0.0 if scale == 0 else (hi - lo) / scale

    @property
    def ok(self) -> bool:
        return self.spread_rel <= self.tolerance_rel

    def require(self) -> None:
        if not self.ok:
            detail = ", ".join(f"{k}={v:.6g}" for k, v in self.methods.items())
            raise IntegrityError(
                f"cross-check FAILED for {self.quantity}: methods disagree by "
                f"{self.spread_rel:.2%} (> {self.tolerance_rel:.2%}): {detail} [{self.unit}]"
            )


@dataclass
class Finding:
    """A quantitative claim produced by a calibration study.

    Serialized to findings/ as JSON + Markdown with everything needed to
    reproduce it: config, code version, constant provenance, uncertainty
    budget, and standing model caveats.
    """

    title: str
    result: dict[str, Any]                       # the headline numbers, with units
    config: dict[str, Any]                       # full declarative input config
    uncertainty_budget: dict[str, str] = field(default_factory=dict)
    constants_used: list[str] = field(default_factory=list)   # str(SourcedValue)
    cross_checks: list[dict[str, Any]] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    validation_state: str = "unknown"            # e.g. "83/83 benchmarks passing"
    created_utc: str = ""
    rydsim_version: str = ""
    config_hash: str = ""
    environment: str = ""

    def finalize(self, version: str) -> "Finding":
        self.created_utc = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
        self.rydsim_version = version
        blob = json.dumps(self.config, sort_keys=True, default=str).encode()
        self.config_hash = hashlib.sha256(blob).hexdigest()[:16]
        self.environment = (
            f"python {platform.python_version()} / {platform.system()} {platform.release()}"
        )
        return self

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2, default=str)

    def to_markdown(self) -> str:
        lines = [
            f"# Finding: {self.title}",
            "",
            f"*GreyNOC RydSim {self.rydsim_version} · {self.created_utc} · "
            f"config `{self.config_hash}` · {self.environment}*",
            "",
            "## Result",
            "",
        ]
        for k, v in self.result.items():
            lines.append(f"- **{k}**: {v}")
        lines += ["", "## Uncertainty budget", ""]
        if self.uncertainty_budget:
            lines += [f"- **{k}**: {v}" for k, v in self.uncertainty_budget.items()]
        else:
            lines.append("- (not evaluated — this finding is qualitative)")
        if self.cross_checks:
            lines += ["", "## Cross-method checks", ""]
            for c in self.cross_checks:
                lines.append(f"- {c}")
        lines += ["", "## Validation state", "", f"- {self.validation_state}"]
        lines += ["", "## Constants on the critical path", ""]
        lines += [f"- {c}" for c in self.constants_used] or ["- (none recorded)"]
        lines += ["", "## Caveats (standing model limitations)", ""]
        lines += [f"- {c}" for c in self.caveats] or ["- (none recorded)"]
        lines += ["", "## Reproduce", "",
                  "```bash",
                  f"rydsim run --config <saved config for {self.config_hash}>",
                  "```", "",
                  "*Reproducible or it didn't happen.*", ""]
        return "\n".join(lines)
