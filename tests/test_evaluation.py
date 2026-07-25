"""Tests for harness-owned evaluation.

The claim under test: the per-fold scores and results.json are produced by the
harness from harness-provisioned data, so a generated script cannot fabricate
numbers or leak test data. These pin that the harness computes an honest, paired,
reproducible comparison and emits the schema the downstream gates read.
"""

from __future__ import annotations

import json

import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from harness import evaluation, registry
from verification import audit_conformance, audit_significance, decide, evaluate


def _paths(tmp_path):
    return {
        "manifest_path": tmp_path / "manifest.json",
        "results_path": tmp_path / "results.json",
    }


def test_produces_paired_scores_and_writes_results(tmp_path) -> None:
    p = _paths(tmp_path)
    res = evaluation.evaluate_comparison(
        "breast_cancer",
        LogisticRegression(max_iter=1000, random_state=42),
        DummyClassifier(strategy="most_frequent"),
        seed=42, primary_metric="accuracy", n_splits=5, **p,
    )
    m = res["conditions"]["method"]["scores"]
    b = res["conditions"]["baseline"]["scores"]
    assert len(m) == len(b) == 5
    # A real classifier must beat the majority-class dummy on breast_cancer.
    assert sum(m) / 5 > sum(b) / 5
    # Both artifacts were written by the harness.
    assert p["results_path"].exists() and p["manifest_path"].exists()
    assert json.loads(p["results_path"].read_text())["primary_metric"] == "accuracy"


def test_scores_are_reproducible(tmp_path) -> None:
    """Same seed and estimators must give identical folds and scores."""
    args = ("wine", LogisticRegression(max_iter=1000, random_state=0),
            DummyClassifier(strategy="stratified", random_state=0))
    a = evaluation.evaluate_comparison(*args, seed=7, **_paths(tmp_path / "a"))
    b = evaluation.evaluate_comparison(*args, seed=7, **_paths(tmp_path / "b"))
    assert a["conditions"]["method"]["scores"] == b["conditions"]["method"]["scores"]


def test_pipeline_with_scaler_is_evaluated_without_leakage(tmp_path) -> None:
    """A scaler-containing Pipeline is fit inside each fold by the harness, so
    the scaler never sees the test fold. It must still run and score."""
    res = evaluation.evaluate_comparison(
        "breast_cancer",
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
        DummyClassifier(strategy="most_frequent"),
        seed=42, primary_metric="macro_f1", n_splits=5, **_paths(tmp_path),
    )
    assert len(res["conditions"]["method"]["scores"]) == 5


def test_harness_results_pass_conformance_and_feed_significance(tmp_path) -> None:
    """The end-to-end contract: harness output conforms to the plan and is a
    valid input to the verdict rule."""
    p = _paths(tmp_path)
    evaluation.evaluate_comparison(
        "breast_cancer",
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
        LogisticRegression(max_iter=1000, random_state=42),
        seed=42, primary_metric="macro_f1", n_splits=5, **p,
    )
    manifest = json.loads(p["manifest_path"].read_text())
    results = json.loads(p["results_path"].read_text())

    conf = audit_conformance({"datasets": [{"name": "breast_cancer"}]}, manifest)
    assert conf.passed

    sig = audit_significance(results)
    assert not decide(sig).halt  # no LLM claim -> can never overclaim
    assert evaluate(results).verdict.value in {"supported", "refuted", "inconclusive"}


def test_unknown_metric_raises(tmp_path) -> None:
    with pytest.raises(evaluation.UnknownMetricError):
        evaluation.evaluate_comparison(
            "iris", DummyClassifier(), DummyClassifier(),
            primary_metric="auroc", **_paths(tmp_path),
        )


def test_too_few_splits_raises(tmp_path) -> None:
    with pytest.raises(ValueError):
        evaluation.evaluate_comparison(
            "iris", DummyClassifier(), DummyClassifier(),
            n_splits=2, **_paths(tmp_path),
        )


def test_unknown_dataset_raises(tmp_path) -> None:
    """Data still comes only through the registry -- no evaluation off-registry."""
    with pytest.raises(registry.UnknownDatasetError):
        evaluation.evaluate_comparison(
            "CIFAR-10-LT", DummyClassifier(), DummyClassifier(), **_paths(tmp_path),
        )
