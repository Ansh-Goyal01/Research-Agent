"""Deterministic verification layer.

Every check here is a pure function: no LLM calls, no network, no state_store
reads. Checks take artifacts as arguments and return Findings.

This is deliberate. The pipeline's original failure was that every audit was an
LLM asked to audit its own upstream output, so nothing could ever be settled.
Any question that can be answered by comparing two artifacts is answered here,
in code, and cannot be reasoned around.

Checks report, the gate decides: `verification.gate.decide` is the only place
a finding becomes a halt.
"""

from verification.conformance import (
    audit_conformance,
    audit_feasibility,
    planned_datasets,
)
from verification.findings import CheckResult, Finding, Severity
from verification.gate import GateDecision, decide, render
from verification.reporting import audit_reported_numbers
from verification.statistics import (
    Assessment,
    Verdict,
    audit_significance,
    classify,
    evaluate,
)

__all__ = [
    "Assessment",
    "CheckResult",
    "Finding",
    "GateDecision",
    "Severity",
    "Verdict",
    "audit_conformance",
    "audit_feasibility",
    "audit_reported_numbers",
    "audit_significance",
    "classify",
    "decide",
    "evaluate",
    "planned_datasets",
    "render",
]
