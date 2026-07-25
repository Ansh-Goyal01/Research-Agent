"""Harness-owned evaluation: the scores are computed here, not by the LLM script.

This is the structural fix for the last remaining trust boundary. Conformance
made the *data* impossible to substitute; significance made the *verdict*
impossible to narrate. But the per-fold scores those checks reason about were
still produced by the generated script, so a script could in principle report
numbers it never measured, or -- more insidiously -- leak test data into
training (fit a scaler on the whole set before cross-validation) and inflate
every fold honestly-looking.

    THE FIX, SAME AS EVERY OTHER GATE: take the ability away.

The generated script no longer runs cross-validation, computes metrics, or
writes results.json. It makes exactly one research decision -- which estimator
is the method and which is the baseline -- and hands both to this function. The
harness provisions the data, builds the folds, and does fit-on-train /
score-on-test itself. Because the fit happens inside each fold under harness
control, even a Pipeline with a scaler cannot see the test fold: leakage is
prevented by construction, not detected after the fact. And because the harness
writes results.json, the scores in it are the ones it measured.

What the script still controls is the legitimate part -- the modelling choice.
What it can no longer touch is the measurement. That is the whole point.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from harness.registry import DEFAULT_MANIFEST_PATH, provision

DEFAULT_RESULTS_PATH = "outputs/code/results.json"

# The metrics an experiment may evaluate on. Kept small and explicit: a scorer
# the harness does not recognise is a plan the harness cannot honestly measure,
# so it fails loudly rather than guessing.
_SCORERS = {
    "macro_f1": lambda y_true, y_pred: float(f1_score(y_true, y_pred, average="macro")),
    "accuracy": lambda y_true, y_pred: float(accuracy_score(y_true, y_pred)),
}

# Fewest folds a paired significance test needs downstream (verification.
# statistics.MIN_PAIRED). Enforced here so a run cannot be set up to under-report.
_MIN_SPLITS = 3


class UnknownMetricError(KeyError):
    """Raised when a plan asks for a metric the harness cannot compute."""


def available_metrics() -> list[str]:
    """Metric names the harness can score, for prompts and validation."""
    return sorted(_SCORERS)


def _fold_scores(X, y, estimator, splits, scorer) -> list[float]:
    """Per-fold test scores for one estimator on pre-built folds.

    The estimator is cloned per fold so no state leaks between folds, and it is
    fit only on the training indices -- the test fold is never seen during fit,
    which is what makes the score honest even for a scaler-containing Pipeline.
    """
    scores = []
    for train_idx, test_idx in splits:
        est = clone(estimator)
        est.fit(X[train_idx], y[train_idx])
        pred = est.predict(X[test_idx])
        scores.append(scorer(y[test_idx], pred))
    return scores


def evaluate_comparison(
    dataset: str,
    method,
    baseline,
    *,
    seed: int = 0,
    primary_metric: str = "macro_f1",
    n_splits: int = 5,
    method_label: str = "method",
    baseline_label: str = "baseline",
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
) -> dict:
    """Cross-validate a method against a baseline on the SAME folds and record it.

    This is the single call a generated experiment script makes. It provisions
    the data (which writes the manifest the conformance gate trusts), evaluates
    both estimators on identical stratified folds, and writes results.json in the
    schema the significance gate reads. The returned dict is that same content.

    Args:
        dataset: a registered dataset name.
        method, baseline: unfitted sklearn-style estimators. Cloned per fold.
        seed: threaded into provisioning, fold shuffling, and (via the caller's
            estimators) their random_state, so the run is reproducible.
        primary_metric: one of `available_metrics()`.
        n_splits: CV folds; must be >= 3 so a paired test has something to work on.

    Raises:
        UnknownMetricError: the metric is not one the harness can score.
        ValueError: n_splits is too small.
    """
    if primary_metric not in _SCORERS:
        raise UnknownMetricError(
            f"{primary_metric!r} is not scorable. Available: {available_metrics()}"
        )
    if n_splits < _MIN_SPLITS:
        raise ValueError(f"n_splits must be >= {_MIN_SPLITS}, got {n_splits}")

    X, y = provision(dataset, seed=seed, manifest_path=manifest_path)
    scorer = _SCORERS[primary_metric]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(skf.split(X, y))

    method_scores = _fold_scores(X, y, method, splits, scorer)
    baseline_scores = _fold_scores(X, y, baseline, splits, scorer)

    def _stats(scores: list[float]) -> dict:
        arr = np.asarray(scores, dtype=float)
        return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)), "n_runs": len(scores)}

    results = {
        "primary_metric": primary_metric,
        "conditions": {
            "method": {"label": method_label, "scores": method_scores},
            "baseline": {"label": baseline_label, "scores": baseline_scores},
        },
        "metrics": [
            {"metric_name": f"{primary_metric}_method", **_stats(method_scores)},
            {"metric_name": f"{primary_metric}_baseline", **_stats(baseline_scores)},
        ],
        # No hypothesis_verdict: the verdict is not the script's to author. The
        # significance gate computes it from these scores by a pre-registered
        # rule, and there is now no LLM claim that could contradict it.
        "key_findings": [],
    }

    path = Path(results_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))
    return results
