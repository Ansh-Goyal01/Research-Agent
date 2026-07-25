"""Adversarial tests for the paper number-conformance gate.

The question: if the paper's prose stated a performance number the experiment
never produced -- the fabrication moved from the data and the verdict into the
Results section -- would anything stop it? The canonical case is a paper that
types 0.95 when the run measured 0.73.
"""

from __future__ import annotations

from verification.findings import Severity
from verification.gate import decide
from verification.reporting import audit_reported_numbers

# A results summary shaped like results_analyst emits: structured metric plus
# the verdict evidence the significance gate stored.
SUMMARY = {
    "metrics": [{"metric_name": "macro_f1", "mean": 0.73, "std": 0.05, "n_runs": 5}],
    "verdict_evidence": {"p_value": 0.112, "mean_diff": -0.033, "ci95": [-0.077, 0.012]},
    "key_findings": ["balanced weighting did not help"],
}


def test_faithful_paper_passes() -> None:
    """A Results section quoting only produced numbers must pass cleanly."""
    paper = "We observe a macro-F1 of 0.73 (std 0.05), with p = 0.112."
    result = audit_reported_numbers(paper, SUMMARY)
    assert result.passed
    assert not result.vacuous


def test_fabricated_metric_is_critical() -> None:
    """The headline case: 0.95 claimed when the run produced 0.73."""
    paper = "Our method reaches a macro-F1 of 0.95, a strong result."
    result = audit_reported_numbers(paper, SUMMARY)
    assert result.critical
    assert result.critical[0].severity is Severity.CRITICAL
    assert "0.95" in result.critical[0].observed
    assert decide(result).halt


def test_percentage_form_of_a_real_number_passes() -> None:
    """0.73 reported as 73% is the same number, not a fabrication."""
    paper = "Accuracy was 73% across folds."
    assert audit_reported_numbers(paper, SUMMARY).passed


def test_fabricated_percentage_is_critical() -> None:
    """A percentage with no produced number behind it is still a fabrication."""
    paper = "The classifier achieved 97.33% accuracy."
    assert audit_reported_numbers(paper, SUMMARY).critical


def test_mean_plus_minus_std_range_passes() -> None:
    """A range phrased as mean +/- std must not false-flag its endpoints."""
    paper = "Scores ranged from 0.68 to 0.78 around the 0.73 mean."
    assert audit_reported_numbers(paper, SUMMARY).passed


def test_honest_rounding_passes() -> None:
    """Prose rounding of a produced number stays within tolerance."""
    summary = {"metrics": [{"metric_name": "macro_f1", "mean": 0.677, "std": 0.0}]}
    paper = "The mean macro-F1 was 0.68."
    assert audit_reported_numbers(paper, summary).passed


def test_structural_integers_are_not_flagged() -> None:
    """Sample counts, seeds, and fold counts are not metric-shaped and must be
    left alone, or the gate would halt every honest paper."""
    paper = ("We evaluate on 2000 samples with seed 42 using 5-fold "
             "cross-validation over 20 features. Macro-F1 was 0.73.")
    assert audit_reported_numbers(paper, SUMMARY).passed


def test_paper_with_no_numbers_is_vacuous() -> None:
    """A Results section stating no numbers has not reported the experiment."""
    paper = "Our method performed well and the hypothesis looked promising."
    result = audit_reported_numbers(paper, SUMMARY)
    assert result.vacuous
    assert decide(result).halt


def test_no_summary_is_vacuous() -> None:
    assert audit_reported_numbers("macro-F1 of 0.73", None).vacuous


def test_gate_is_load_bearing_on_fabrication() -> None:
    """Neutering the match would make this fabricated figure start passing."""
    paper = "We report a macro-F1 of 0.9733."
    result = audit_reported_numbers(paper, SUMMARY)
    assert any("0.9733" in f.observed for f in result.critical)
