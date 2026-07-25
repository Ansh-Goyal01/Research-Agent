"""Adversarial tests for the significance gate.

The question these answer: if a run reported an honest set of numbers but then
claimed a conclusion those numbers do not support -- the fabrication moved from
the data to the verdict -- would anything stop it?

The decorative-reviewer failure this project exists to kill had two forms. The
data form (train on iris, claim it was CIFAR) is caught by conformance. The
verdict form (measure a null result, write "supported") is caught here. A rule
with no tests is that same decorative reviewer rebuilt one level up, so these
tests pin the exact boundaries of the pre-registered rule.
"""

from __future__ import annotations

import numpy as np

from verification.findings import Severity
from verification.gate import decide
from verification.statistics import (
    ALPHA,
    MARGIN,
    MIN_PAIRED,
    Verdict,
    audit_significance,
    classify,
    evaluate,
)


def _results(method, baseline, claim):
    """A minimal results.json-shaped dict with paired scores and a claim."""
    return {
        "primary_metric": "macro_f1",
        "conditions": {
            "method": {"scores": list(method)},
            "baseline": {"scores": list(baseline)},
        },
        "hypothesis_verdict": claim,
    }


# ---------------------------------------------------------------------------
# The decision rule: classify()
# ---------------------------------------------------------------------------


def test_clear_consistent_improvement_is_supported() -> None:
    """A large, reliable gain past the margin must read as SUPPORTED."""
    method = [0.90, 0.91, 0.89, 0.92, 0.90]
    baseline = [0.80, 0.81, 0.79, 0.82, 0.80]
    assert classify(method, baseline).verdict is Verdict.SUPPORTED


def test_no_difference_is_inconclusive_not_refuted() -> None:
    """The case the council cared about: a null result is INCONCLUSIVE.

    A p-value-only two-way rule would call this "refuted" and read as broken. A
    pipeline that never says "I don't know" is still lying.
    """
    scores = [0.80, 0.82, 0.78, 0.81, 0.79]
    assert classify(scores, list(scores)).verdict is Verdict.INCONCLUSIVE


def test_significant_but_worse_is_refuted() -> None:
    """REFUTED is reserved for a significant regression, not any null result."""
    method = [0.70, 0.71, 0.69, 0.72, 0.70]
    baseline = [0.80, 0.81, 0.79, 0.82, 0.80]
    assert classify(method, baseline).verdict is Verdict.REFUTED


def test_significant_but_trivial_gain_is_inconclusive() -> None:
    """The effect-size half of the rule: below the margin is not SUPPORTED.

    These differ by a hair (~0.002) with almost no variance, so a t-test alone
    would call it significant. The SESOI margin is what keeps a statistically
    detectable but practically meaningless gain out of "supported".
    """
    method = [0.8020, 0.8021, 0.8019, 0.8020, 0.8021]
    baseline = [0.8000, 0.8001, 0.7999, 0.8000, 0.8001]
    a = classify(method, baseline)
    assert a.evidence["mean_diff"] < MARGIN
    assert a.verdict is Verdict.INCONCLUSIVE


def test_too_few_runs_is_inconclusive() -> None:
    """Below MIN_PAIRED there is no honest variance estimate, so no verdict."""
    method = [0.9] * (MIN_PAIRED - 1)
    baseline = [0.7] * (MIN_PAIRED - 1)
    a = classify(method, baseline)
    assert a.verdict is Verdict.INCONCLUSIVE
    assert a.evidence["reason"] == "insufficient_runs"


def test_unpaired_lengths_are_inconclusive() -> None:
    """Mismatched fold counts cannot be paired, so the test cannot run."""
    a = classify([0.9, 0.9, 0.9], [0.7, 0.7])
    assert a.verdict is Verdict.INCONCLUSIVE
    assert a.evidence["reason"] == "unpaired"


def test_identical_nonzero_difference_does_not_crash() -> None:
    """Zero-variance differences would make scipy return NaN; we handle it.

    A perfectly consistent +0.1 on every fold is strong evidence, not missing
    evidence, and must not silently become NaN -> falsy -> inconclusive.
    """
    method = [0.80, 0.80, 0.80, 0.80]
    baseline = [0.70, 0.70, 0.70, 0.70]
    a = classify(method, baseline)
    assert a.verdict is Verdict.SUPPORTED
    assert a.evidence["p_value"] == 0.0


def test_evidence_carries_the_statistics() -> None:
    """The verdict must show its work: p-value, effect size, CI, margin."""
    a = classify([0.9, 0.91, 0.89, 0.92], [0.8, 0.81, 0.79, 0.82])
    for key in ("p_value", "effect_size_dz", "ci95", "mean_diff", "alpha", "margin"):
        assert key in a.evidence
    assert a.evidence["alpha"] == ALPHA
    assert a.evidence["margin"] == MARGIN


# ---------------------------------------------------------------------------
# evaluate(): reading the reported results
# ---------------------------------------------------------------------------


def test_evaluate_reads_conditions() -> None:
    r = _results([0.9, 0.91, 0.89, 0.92], [0.8, 0.81, 0.79, 0.82], "supported")
    assert evaluate(r).verdict is Verdict.SUPPORTED


def test_evaluate_without_comparison_is_inconclusive() -> None:
    """No paired conditions means nothing to test -- never a default to a claim."""
    a = evaluate({"hypothesis_verdict": "supported", "metrics": []})
    assert a.verdict is Verdict.INCONCLUSIVE
    assert a.evidence["reason"] == "no_comparison_reported"


# ---------------------------------------------------------------------------
# audit_significance(): the gate boundary
# ---------------------------------------------------------------------------


def test_overclaimed_support_is_critical() -> None:
    """The headline case: honest numbers, a fabricated 'supported' verdict.

    This is the iris fabrication relocated to the conclusion. The numbers are a
    dead heat; the run claims victory. It must be CRITICAL, exactly as a dataset
    swap is.
    """
    scores = [0.80, 0.82, 0.78, 0.81, 0.79]
    result = audit_significance(_results(scores, list(scores), "supported"))
    assert result.critical
    assert result.critical[0].severity is Severity.CRITICAL
    assert not result.vacuous


def test_claimed_supported_and_computed_supported_passes() -> None:
    """An honest, evidenced 'supported' must sail through -- the gate is not a
    blanket blocker on positive results, only on unsupported ones."""
    method = [0.90, 0.91, 0.89, 0.92, 0.90]
    baseline = [0.80, 0.81, 0.79, 0.82, 0.80]
    result = audit_significance(_results(method, baseline, "supported"))
    assert result.passed
    assert not result.vacuous


def test_claimed_inconclusive_never_halts() -> None:
    """Honesty is always allowed. Claiming 'inconclusive' cannot overclaim."""
    scores = [0.80, 0.82, 0.78, 0.81, 0.79]
    result = audit_significance(_results(scores, list(scores), "inconclusive"))
    assert result.passed


def test_overclaimed_refuted_is_critical() -> None:
    """Overclaiming a negative is a fabrication too: the data is a null result
    but the run asserts the hypothesis was refuted."""
    scores = [0.80, 0.82, 0.78, 0.81, 0.79]
    result = audit_significance(_results(scores, list(scores), "refuted"))
    assert result.critical


def test_underclaim_is_reported_but_not_critical() -> None:
    """Said 'inconclusive' while the test found real support: honest and
    conservative, so a MINOR nudge, never a halt."""
    method = [0.90, 0.91, 0.89, 0.92, 0.90]
    baseline = [0.80, 0.81, 0.79, 0.82, 0.80]
    result = audit_significance(_results(method, baseline, "inconclusive"))
    assert not result.critical
    assert not decide(result).halt, "an under-claim must never halt the run"
    assert result.findings
    assert result.findings[0].severity is Severity.MINOR


def test_empty_results_is_vacuous() -> None:
    """A run that reported nothing has established nothing; the gate must treat
    that as a vacuous pass (which decide() then halts), not a clean pass."""
    assert audit_significance(None).vacuous
    assert audit_significance({}).vacuous


def test_claim_matching_computed_examines_something() -> None:
    """Every non-empty audit must prove it looked at the numbers."""
    method = [0.90, 0.91, 0.89, 0.92]
    baseline = [0.80, 0.81, 0.79, 0.82]
    result = audit_significance(_results(method, baseline, "supported"))
    assert result.examined


# ---------------------------------------------------------------------------
# The mutation guard: prove the gate is load-bearing
# ---------------------------------------------------------------------------


def test_gate_is_load_bearing_on_the_overclaim() -> None:
    """If the overclaim rule were neutered, this fixture must start passing.

    The fixture is the canonical fabrication: a genuine null result claimed as
    'supported'. Recording it here as CRITICAL is what makes the earlier test a
    real regression guard rather than a coincidence.
    """
    scores = np.array([0.80, 0.82, 0.78, 0.81, 0.79])
    result = audit_significance(_results(scores, scores, "supported"))
    critical_summaries = " ".join(f.summary for f in result.critical)
    assert "not supported by the data" in critical_summaries
