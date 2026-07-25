"""Experiment harness: the pipeline's feasible universe and its data source.

Generated experiment scripts import `load_dataset` from here. They do not
choose their own data, which is what makes dataset substitution structurally
impossible rather than merely detectable after the fact.
"""

from harness.registry import (
    FAST_BUDGET_SECONDS,
    REGISTRY,
    DatasetSpec,
    UnknownDatasetError,
    describe_for_planner,
    expected_fingerprint,
    fingerprint,
    is_feasible,
    list_feasible,
    load_dataset,
    provision,
)

__all__ = [
    "FAST_BUDGET_SECONDS",
    "REGISTRY",
    "DatasetSpec",
    "UnknownDatasetError",
    "describe_for_planner",
    "expected_fingerprint",
    "fingerprint",
    "is_feasible",
    "list_feasible",
    "load_dataset",
    "provision",
]
