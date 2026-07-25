"""Adversarial tests for the verification spine.

These tests exist to answer one question: if the pipeline tried to fabricate a
result again, exactly as it did before, would anything stop it?

The central case is `test_golden_baseline_plan_is_rejected`. It loads the real
committed plan from the run that fabricated results -- a few-shot VAE study on
CIFAR-10-LT -- and asserts the feasibility gate halts it. That run produced a
paper claiming 0.9733 accuracy supported its hypothesis, while the experiment
underneath had quietly trained a LogisticRegression on iris. Four LLM review
stages approved it. This test is the regression guard for that failure, and it
runs against the archived artifacts rather than a mock, so it cannot drift away
from the bug it describes.

A verification layer with no tests is the decorative reviewer recreated one
level up. These are the tests that make the gates load-bearing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness import registry
from verification.conformance import audit_conformance, audit_feasibility
from verification.findings import CheckResult, Finding, Severity

BASE_DIR = Path(__file__).resolve().parent.parent
GOLDEN_STATE = BASE_DIR / "baseline-golden" / "workflow_state.json"


def _plan(*dataset_names: str) -> dict:
    """Minimal experiment plan naming the given datasets."""
    return {"datasets": [{"name": n} for n in dataset_names]}


# ---------------------------------------------------------------------------
# The regression case: the exact plan that caused the fabrication
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not GOLDEN_STATE.exists(),
    reason="archived baseline run not present",
)
def test_golden_baseline_plan_is_rejected() -> None:
    """The real fabricating run's plan must not survive the feasibility gate.

    This is the test the whole spine exists for. If it ever passes silently,
    the pipeline has regained the ability to plan experiments it cannot run.
    """
    state = json.loads(GOLDEN_STATE.read_text(encoding="utf-8"))
    plan = state["experiment_plan"]

    result = audit_feasibility(plan)

    assert not result.passed, "the CIFAR-10-LT plan must be rejected"
    assert result.critical, "rejection must be CRITICAL, not advisory"
    assert not result.vacuous, "the check must prove it examined something"
    assert any("CIFAR-10-LT" in f.observed for f in result.critical)


@pytest.mark.skipif(
    not GOLDEN_STATE.exists(),
    reason="archived baseline run not present",
)
def test_golden_baseline_substitution_is_caught_after_the_fact() -> None:
    """Even if planning were bypassed, the iris substitution is still caught.

    Defence in depth: feasibility stops the plan, and conformance independently
    stops the run. Neither depends on the other being correct.
    """
    state = json.loads(GOLDEN_STATE.read_text(encoding="utf-8"))
    plan = state["experiment_plan"]

    # What the fabricating run actually consumed.
    iris_manifest = registry.expected_fingerprint("iris")

    result = audit_conformance(plan, iris_manifest)

    assert not result.passed
    assert result.critical
    observed = " ".join(f.observed for f in result.critical)
    assert "iris" in observed


# ---------------------------------------------------------------------------
# Feasibility gate
# ---------------------------------------------------------------------------


def test_registered_dataset_is_feasible() -> None:
    result = audit_feasibility(_plan("wine"))
    assert result.passed
    assert not result.vacuous


def test_unregistered_dataset_is_critical() -> None:
    result = audit_feasibility(_plan("CIFAR-10-LT"))
    assert result.critical
    assert result.critical[0].severity is Severity.CRITICAL


def test_plan_with_no_dataset_is_critical() -> None:
    """A plan naming nothing must fail, not vacuously pass."""
    result = audit_feasibility({"datasets": []})
    assert result.critical


def test_mixed_plan_rejects_only_the_infeasible_entry() -> None:
    result = audit_feasibility(_plan("wine", "ImageNet"))
    assert len(result.critical) == 1
    assert result.critical[0].observed == "ImageNet"


def test_bare_string_datasets_are_understood() -> None:
    """Plans sometimes carry bare strings instead of dicts; both must work."""
    result = audit_feasibility({"datasets": ["wine"]})
    assert result.passed


# ---------------------------------------------------------------------------
# Conformance gate
# ---------------------------------------------------------------------------


def test_matching_run_passes_conformance() -> None:
    manifest = registry.expected_fingerprint("wine")
    result = audit_conformance(_plan("wine"), manifest)
    assert result.passed
    assert not result.vacuous


def test_wrong_dataset_is_critical() -> None:
    """The iris-for-wine swap: the canonical fabrication shape."""
    manifest = registry.expected_fingerprint("iris")
    result = audit_conformance(_plan("wine"), manifest)
    assert result.critical
    assert "wine" in result.critical[0].expected


def test_missing_manifest_is_critical() -> None:
    """Absence of evidence must never be recorded as a pass."""
    result = audit_conformance(_plan("wine"), None)
    assert result.critical


def test_forged_manifest_is_caught_by_fingerprint() -> None:
    """A script that lies about its dataset name is still caught by the data.

    This is the adversarial case that source-scanning cannot handle: the
    manifest claims the planned dataset, but the numbers underneath are from
    somewhere else. Shape and class histogram give it away.
    """
    forged = registry.expected_fingerprint("iris")
    forged["dataset"] = "wine"  # claim the planned name

    result = audit_conformance(_plan("wine"), forged)

    assert result.critical, "fingerprint mismatch must be caught"
    mismatched = {f.summary.split("'")[1] for f in result.critical}
    assert "n_features" in mismatched or "n_samples" in mismatched


def test_tampered_hash_alone_is_caught() -> None:
    """Same shape, different bytes: only the hash can detect this."""
    manifest = registry.expected_fingerprint("wine")
    manifest["data_hash"] = "0" * 16

    result = audit_conformance(_plan("wine"), manifest)

    assert result.critical
    assert any("data_hash" in f.summary for f in result.critical)


# ---------------------------------------------------------------------------
# Properties of the harness itself
# ---------------------------------------------------------------------------


def test_every_registered_dataset_loads_and_fingerprints() -> None:
    """The registry must not contain entries that cannot actually run."""
    for name in registry.list_feasible():
        X, y = registry.load_dataset(name)
        fp = registry.fingerprint(X, y, name=name)
        assert fp["n_samples"] > 0
        assert fp["n_classes"] >= 2
        assert sum(fp["class_histogram"]) == fp["n_samples"]


def test_fingerprints_are_reproducible() -> None:
    """Same dataset and seed must fingerprint identically across calls."""
    for name in registry.list_feasible():
        assert registry.expected_fingerprint(name, 0) == registry.expected_fingerprint(
            name, 0
        )


def test_synthetic_seeds_produce_different_data() -> None:
    """Seeding must actually vary synthetic data, or multi-run stats are fake."""
    a = registry.expected_fingerprint("synthetic_imbalanced_binary", 0)
    b = registry.expected_fingerprint("synthetic_imbalanced_binary", 1)
    assert a["data_hash"] != b["data_hash"]


def test_unknown_dataset_raises_rather_than_defaulting() -> None:
    """Loading an unregistered dataset must fail loudly, never fall back.

    A silent fallback here would reintroduce substitution at the harness level.
    """
    with pytest.raises(registry.UnknownDatasetError):
        registry.load_dataset("CIFAR-10-LT")


# ---------------------------------------------------------------------------
# provision: the harness writes the manifest the conformance gate trusts
# ---------------------------------------------------------------------------


def test_provision_writes_a_manifest_matching_the_truth(tmp_path) -> None:
    """The harness, not the script, records what was consumed.

    provision must write exactly the fingerprint the conformance check will
    re-derive, so a script that used the harness always conforms by
    construction. If these ever diverge, honest runs would start failing.
    """
    manifest_file = tmp_path / "manifest.json"
    X, y = registry.provision("wine", seed=42, manifest_path=manifest_file)

    assert X.shape[0] == y.shape[0] > 0
    written = json.loads(manifest_file.read_text())
    assert written == registry.expected_fingerprint("wine", seed=42)


def test_provisioned_run_passes_conformance_end_to_end(tmp_path) -> None:
    """A plan plus the manifest its provisioned run emitted must conform.

    This is the happy path the whole spine has to permit: constrain the plan to
    the registry, hand the data to the script via provision, and the run it
    produces verifies. A gate that blocked this too would be uselessly strict.
    """
    manifest_file = tmp_path / "manifest.json"
    registry.provision("breast_cancer", seed=42, manifest_path=manifest_file)
    manifest = json.loads(manifest_file.read_text())

    result = audit_conformance(_plan("breast_cancer"), manifest)

    assert result.passed
    assert not result.vacuous


def test_provision_manifest_defeats_a_dataset_swap(tmp_path) -> None:
    """If the run provisions the wrong dataset, conformance still catches it.

    provision is honest about what it loaded, so swapping the dataset changes
    the manifest it writes -- and that no longer matches the plan.
    """
    manifest_file = tmp_path / "manifest.json"
    registry.provision("iris", seed=42, manifest_path=manifest_file)
    manifest = json.loads(manifest_file.read_text())

    result = audit_conformance(_plan("wine"), manifest)

    assert result.critical


# ---------------------------------------------------------------------------
# Finding semantics
# ---------------------------------------------------------------------------


def test_findings_are_immutable() -> None:
    """Evidence must not be editable by anything downstream of the check."""
    finding = Finding(check="c", severity=Severity.CRITICAL, summary="s")
    with pytest.raises(Exception):
        finding.summary = "rewritten"  # type: ignore[misc]


def test_check_examining_nothing_is_vacuous() -> None:
    """A pass that examined nothing must be distinguishable from a real pass."""
    empty = CheckResult(check="c")
    assert empty.passed
    assert empty.vacuous, "an empty check must not read as a meaningful pass"
