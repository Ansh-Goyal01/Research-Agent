"""Tests for the halt decision.

`test_verification.py` proves the checks see the right things. These tests
prove that seeing them actually stops the run, which is the step the original
pipeline was missing: its reviewer detected problems perfectly well and then
printed them next to an unconditional approval prompt.

The load-bearing test here is `test_override_note_cannot_clear_a_critical_halt`.
If that ever passes as a non-halt, the gate has become advisory and the project
is back where it started.
"""

from __future__ import annotations

from verification.findings import CheckResult, Finding, Severity
from verification.gate import decide, render


def _critical() -> CheckResult:
    return CheckResult(
        check="feasibility",
        findings=(
            Finding(
                check="feasibility",
                severity=Severity.CRITICAL,
                summary="Planned dataset 'CIFAR-10-LT' is not in the registry.",
                expected="iris, wine, ...",
                observed="CIFAR-10-LT",
            ),
        ),
        examined=("CIFAR-10-LT",),
    )


# ---------------------------------------------------------------------------
# Halting
# ---------------------------------------------------------------------------


def test_critical_finding_halts() -> None:
    verdict = decide(_critical())
    assert verdict.halt
    assert verdict.findings, "the halt must carry its evidence"


def test_override_note_cannot_clear_a_critical_halt() -> None:
    """A human override must not be able to talk the gate out of a halt.

    The escape hatch for a wrong check is editing harness/registry.py -- a
    reviewable diff -- not a sentence typed at the prompt by whoever wants the
    run to finish. If this test is ever relaxed, read the project's baseline
    story again before doing it.
    """
    verdict = decide(_critical(), override_note="I checked it manually, it's fine")

    assert verdict.halt, "critical findings must halt regardless of override"
    assert verdict.override_note, "the attempted override is still recorded"
    assert not verdict.overridden, "no override path may report success"


def test_vacuous_pass_halts() -> None:
    """A check that examined nothing has not cleared anything."""
    verdict = decide(CheckResult(check="conformance", findings=(), examined=()))

    assert verdict.halt
    assert "examined nothing" in verdict.reason


# ---------------------------------------------------------------------------
# Passing
# ---------------------------------------------------------------------------


def test_genuine_pass_does_not_halt() -> None:
    verdict = decide(CheckResult(check="feasibility", examined=("wine",)))
    assert not verdict.halt
    assert not verdict.findings


def test_non_critical_findings_do_not_halt_but_are_reported() -> None:
    """MINOR/MAJOR findings are surfaced without stopping the run.

    Reserving the halt for CRITICAL keeps it meaningful. A gate that stops on
    everything gets disabled wholesale the first time it is inconvenient.
    """
    result = CheckResult(
        check="feasibility",
        findings=(
            Finding(
                check="feasibility",
                severity=Severity.MINOR,
                summary="Runtime estimate is optimistic.",
            ),
        ),
        examined=("wine",),
    )

    verdict = decide(result)

    assert not verdict.halt
    assert verdict.findings, "non-blocking findings must still reach the operator"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_render_distinguishes_vacuous_from_genuine_pass() -> None:
    """The two must never read alike; conflating them is the original bug."""
    vacuous = render(CheckResult(check="conformance", examined=()))
    genuine = render(CheckResult(check="conformance", examined=("iris",)))

    assert "EXAMINED NOTHING" in vacuous
    assert "pass" in genuine
    assert vacuous != genuine
