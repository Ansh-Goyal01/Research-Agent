"""Finding types shared by every verification check.

Checks report; gates decide. A check never raises, never prints, and never
halts the pipeline -- it returns Findings describing what is true. The gate
(verification.gate) is the single place that maps findings to a halt decision.

That separation is deliberate: it lets a human override be added at the gate,
with an audit record, without touching check logic. It also makes every check
trivially unit-testable, which is what the original LLM-based "audits" were not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Severity(str, Enum):
    """Severity of a verification finding.

    CRITICAL halts the run. It is reserved for findings that mean the paper
    would assert something the experiment did not establish -- fabrication,
    not sloppiness.
    """

    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Finding:
    """A single verification result.

    Frozen because findings are evidence: once a check has observed something,
    no downstream stage should be able to edit it. The original pipeline's
    reviewer could rewrite its own audit verdict; this type makes that
    impossible by construction.
    """

    check: str
    severity: Severity
    summary: str
    expected: str = ""
    observed: str = ""
    remediation: str = ""

    def __str__(self) -> str:
        line = f"[{str(self.severity).upper()}] {self.check}: {self.summary}"
        if self.expected or self.observed:
            line += f"\n    expected: {self.expected}\n    observed: {self.observed}"
        if self.remediation:
            line += f"\n    fix: {self.remediation}"
        return line


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one check: the findings it produced, plus what it examined.

    `examined` exists so a passing check can prove it actually looked at
    something. A check that silently examines nothing and returns no findings
    is indistinguishable from a passing check -- that ambiguity is how the
    original reviewer "passed" a paper whose evidence it had never seen.
    """

    check: str
    findings: tuple[Finding, ...] = field(default_factory=tuple)
    examined: tuple[str, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        return not self.findings

    @property
    def critical(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.CRITICAL)

    @property
    def vacuous(self) -> bool:
        """True when the check examined nothing, so its pass means nothing."""
        return not self.examined
