"""Checks report; the gate decides.

`verification.conformance` answers questions of fact and never halts anything.
This module is the single place where findings become a halt decision, so
there is exactly one line of code to read to know what can stop a run.

The failure being designed against is specific. In the original pipeline the
reviewer's verdict reached `orchestrator.py` only as an argument to `print()`:

    print("Verdict: " + review["overall_verdict"])   # "reject"
    ok = get_human_approval(final + "Approve final paper for export?")

The verdict was rendered, not obeyed. A gate that prints findings and then asks
an unconditional "continue anyway?" is that same bug with better checks behind
it, so the decision below must be the thing the orchestrator branches on.
"""

from __future__ import annotations

from dataclasses import dataclass

from verification.findings import CheckResult, Finding


@dataclass(frozen=True)
class GateDecision:
    """What the orchestrator must do, and the evidence for it.

    Frozen for the same reason `Finding` is: a decision is a record. No
    downstream stage should be able to edit the verdict it was handed.
    """

    halt: bool
    reason: str
    findings: tuple[Finding, ...] = ()
    overridden: bool = False
    override_note: str = ""


def render(result: CheckResult) -> str:
    """Human-readable report for one check. Presentation only, no decision."""
    if result.vacuous:
        head = f"{result.check}: EXAMINED NOTHING"
    elif result.passed:
        head = f"{result.check}: pass ({len(result.examined)} item(s) examined)"
    else:
        head = f"{result.check}: {len(result.findings)} finding(s)"

    lines = [head]
    lines.extend(str(f) for f in result.findings)
    return "\n".join(lines)


def decide(result: CheckResult, override_note: str = "") -> GateDecision:
    """Map a CheckResult to a halt decision.

    Three cases have to be resolved, and they are genuine policy choices rather
    than mechanical translation:

    1. `result.critical` is non-empty  -- fabrication-grade findings.
    2. `result.vacuous` is True        -- the check passed having examined
                                          nothing. A pass that proves nothing
                                          is what let the original reviewer
                                          approve a paper it never verified.
    3. `override_note` is non-empty    -- a human has typed a justification.
                                          Whether that is allowed to convert a
                                          halt into a pass is the whole
                                          question this module exists to answer.

    The policy is deliberately the strict one on all three counts.

    Critical findings halt unconditionally, and `override_note` is recorded but
    never permitted to clear them. An override that a human can grant by typing
    a sentence is not a weaker version of this gate -- it is the original bug,
    since the run proceeds either way and the verdict decides nothing. The
    escape hatch for a check that is genuinely wrong is to edit
    `harness/registry.py`, which is a reviewable diff rather than a prompt
    answered at 2am by the person who wants the run to finish.

    A vacuous pass halts for the same reason with a distinct message: a check
    that examined nothing has not cleared the run, and reporting it as a pass
    is precisely how the original reviewer certified a paper it never read.

    Args:
        result: the check outcome to judge.
        override_note: a human's written justification. Recorded on the
            decision for the audit trail; it does not change the outcome.

    Returns:
        GateDecision carrying `halt`, a `reason` fit to print, and the findings
        that drove it.
    """
    if result.critical:
        return GateDecision(
            halt=True,
            reason=(
                f"{result.check}: {len(result.critical)} critical finding(s). "
                "Run halted before it can produce a claim the experiment does "
                "not support."
            ),
            findings=result.critical,
            override_note=override_note,
        )

    if result.vacuous:
        return GateDecision(
            halt=True,
            reason=(
                f"{result.check}: examined nothing. A check that inspected no "
                "artifact has not established anything, so its pass cannot "
                "clear the run."
            ),
            override_note=override_note,
        )

    if result.findings:
        return GateDecision(
            halt=False,
            reason=(
                f"{result.check}: {len(result.findings)} non-critical "
                "finding(s); continuing."
            ),
            findings=result.findings,
            override_note=override_note,
        )

    return GateDecision(
        halt=False,
        reason=f"{result.check}: passed, {len(result.examined)} item(s) examined.",
        override_note=override_note,
    )
