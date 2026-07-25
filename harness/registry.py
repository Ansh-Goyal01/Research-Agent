"""The feasible universe of experiments, and the data injection harness.

This module is the structural fix for the pipeline's original failure.

WHAT WENT WRONG
    The experiment designer planned a few-shot VAE study on CIFAR-10-LT while
    the implementation agent was told "sklearn only, CPU only, under 60
    seconds". Those constraints cannot both be satisfied, so the model quietly
    loaded sklearn's iris dataset, fit a LogisticRegression, and reported
    0.9733 accuracy as confirmation of the VAE hypothesis. Every downstream
    stage believed it.

WHY A CHECK IS NOT ENOUGH
    Scanning the generated source for `load_iris` only catches that exact
    substitution. `fetch_openml`, a local CSV, `make_classification`, or a
    dataset name held in a variable all evade it -- and a passing scan
    manufactures confidence in dimensions nobody checked.

THE FIX
    Take the choice away. The registry defines every dataset the pipeline is
    allowed to study -- all of which genuinely run on CPU in seconds -- and the
    generated script receives its data from `load_dataset()` rather than
    loading its own. A script cannot substitute a dataset it never selects.
    Conformance holds by construction; `fingerprint()` exists only to catch a
    script that bypasses the harness entirely.

    The registry doubles as the feasibility constraint. Because every entry is
    cheap by construction, "is this plan feasible?" reduces to "is this dataset
    in the registry?" -- which is a dict lookup, not a judgment call.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
    make_classification,
)

# Every dataset here must load and train in well under the execution timeout on
# a laptop CPU. That is the whole point: the research domain is scoped to what
# the available compute can actually answer, so no stage is ever handed an
# impossible brief.
FAST_BUDGET_SECONDS = 60


@dataclass(frozen=True)
class DatasetSpec:
    """A dataset the pipeline is permitted to study.

    `research_uses` is not decoration -- the experiment designer is shown these
    so it proposes hypotheses the data can actually test. Constraining the
    question is cheaper than detecting a bad answer.
    """

    name: str
    task: str
    n_samples: int
    n_features: int
    n_classes: int
    imbalance_ratio: float
    description: str
    research_uses: tuple[str, ...]
    loader: Callable[[int], tuple[np.ndarray, np.ndarray]]

    @property
    def is_imbalanced(self) -> bool:
        return self.imbalance_ratio >= 2.0


def _load_sklearn(loader: Callable) -> Callable[[int], tuple[np.ndarray, np.ndarray]]:
    """Adapt a stock sklearn loader to the (seed) -> (X, y) harness signature.

    The seed is ignored: these datasets are fixed on disk, so they are already
    perfectly reproducible. Keeping the signature uniform means callers never
    branch on dataset kind.
    """

    def _inner(seed: int) -> tuple[np.ndarray, np.ndarray]:
        bunch = loader()
        return np.asarray(bunch.data, dtype=float), np.asarray(bunch.target)

    return _inner


def _make_imbalanced(
    n_samples: int,
    n_features: int,
    weights: tuple[float, ...],
    class_sep: float,
) -> Callable[[int], tuple[np.ndarray, np.ndarray]]:
    """Build a synthetic imbalanced classification set.

    Real imbalanced benchmarks (credit fraud, rare disease) need a network
    fetch, which would make runs non-reproducible and slow. Generating from a
    seed keeps the study honest -- the imbalance is exactly what we specify,
    and any run can be reproduced from the manifest alone.
    """

    def _inner(seed: int) -> tuple[np.ndarray, np.ndarray]:
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=max(2, n_features // 2),
            n_redundant=max(1, n_features // 5),
            n_classes=len(weights),
            weights=list(weights),
            class_sep=class_sep,
            flip_y=0.01,
            random_state=seed,
        )
        return np.asarray(X, dtype=float), np.asarray(y)

    return _inner


REGISTRY: dict[str, DatasetSpec] = {
    "iris": DatasetSpec(
        name="iris",
        task="classification",
        n_samples=150,
        n_features=4,
        n_classes=3,
        imbalance_ratio=1.0,
        description="Classic balanced 3-class flower measurements.",
        research_uses=("baseline sanity check", "multiclass calibration"),
        loader=_load_sklearn(load_iris),
    ),
    "wine": DatasetSpec(
        name="wine",
        task="classification",
        n_samples=178,
        n_features=13,
        n_classes=3,
        imbalance_ratio=1.5,
        description="Chemical analysis of wines from three cultivars.",
        research_uses=("feature selection", "mild class imbalance"),
        loader=_load_sklearn(load_wine),
    ),
    "breast_cancer": DatasetSpec(
        name="breast_cancer",
        task="classification",
        n_samples=569,
        n_features=30,
        n_classes=2,
        imbalance_ratio=1.7,
        description="Diagnostic measurements for binary tumour classification.",
        research_uses=(
            "probability calibration",
            "feature selection",
            "recall-critical evaluation",
        ),
        loader=_load_sklearn(load_breast_cancer),
    ),
    "digits": DatasetSpec(
        name="digits",
        task="classification",
        n_samples=1797,
        n_features=64,
        n_classes=10,
        imbalance_ratio=1.0,
        description="8x8 handwritten digit images as flat pixel vectors.",
        research_uses=("dimensionality reduction", "multiclass comparison"),
        loader=_load_sklearn(load_digits),
    ),
    "synthetic_imbalanced_binary": DatasetSpec(
        name="synthetic_imbalanced_binary",
        task="classification",
        n_samples=2000,
        n_features=20,
        n_classes=2,
        imbalance_ratio=9.0,
        description="Synthetic binary task with a 90/10 class split.",
        research_uses=(
            "class imbalance",
            "resampling methods",
            "threshold tuning",
            "PR-curve evaluation",
        ),
        loader=_make_imbalanced(2000, 20, (0.9, 0.1), class_sep=1.0),
    ),
    "synthetic_imbalanced_severe": DatasetSpec(
        name="synthetic_imbalanced_severe",
        task="classification",
        n_samples=3000,
        n_features=20,
        n_classes=2,
        imbalance_ratio=49.0,
        description="Synthetic binary task with a 98/2 split -- rare-event regime.",
        research_uses=(
            "severe imbalance",
            "cost-sensitive learning",
            "anomaly-style evaluation",
        ),
        loader=_make_imbalanced(3000, 20, (0.98, 0.02), class_sep=1.2),
    ),
    "synthetic_multiclass_imbalanced": DatasetSpec(
        name="synthetic_multiclass_imbalanced",
        task="classification",
        n_samples=2500,
        n_features=25,
        n_classes=4,
        imbalance_ratio=12.0,
        description="Synthetic 4-class task with a long-tailed class distribution.",
        research_uses=(
            "long-tailed classification",
            "macro-F1 evaluation",
            "per-class calibration",
        ),
        loader=_make_imbalanced(2500, 25, (0.6, 0.25, 0.1, 0.05), class_sep=1.0),
    ),
}


class UnknownDatasetError(KeyError):
    """Raised when a plan names a dataset outside the feasible registry.

    This is the error that would have stopped the original failure at design
    time: "CIFAR-10-LT" is not in the registry, so the plan is rejected before
    any code is generated and there is nothing to silently substitute.
    """


def load_dataset(name: str, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for a registered dataset.

    Generated experiment scripts call this instead of importing their own
    loader. That single indirection is what makes dataset substitution
    structurally impossible rather than merely detectable.
    """
    spec = REGISTRY.get(name)
    if spec is None:
        raise UnknownDatasetError(
            f"{name!r} is not a feasible dataset. Available: {sorted(REGISTRY)}"
        )
    return spec.loader(seed)


# Where the harness records what an experiment actually consumed. The path is
# relative on purpose: generated scripts run with cwd set to the project root
# (see tools/code_executor.run_script), so this resolves to the same
# outputs/code/ directory results.json already uses.
DEFAULT_MANIFEST_PATH = "outputs/code/manifest.json"


def provision(
    name: str,
    seed: int = 0,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a registered dataset AND write its true fingerprint to a manifest.

    This is the single call a generated experiment script makes to obtain data.
    It matters that the *harness* writes the manifest, not the script: the
    manifest is the evidence the conformance check trusts, so if the LLM-authored
    script could author it too, it could forge a record that matches the plan
    while training on something else. Here the script never touches the manifest
    -- it receives (X, y) and the harness has already recorded, from the same
    bytes, exactly what it handed over. A script that ignores this call and loads
    its own data simply never produces a manifest, which the conformance gate
    already treats as critical.
    """
    X, y = load_dataset(name, seed)
    manifest = fingerprint(X, y, name=name, seed=seed)
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    return X, y


def fingerprint(X: np.ndarray, y: np.ndarray, *, name: str = "", seed: int = 0) -> dict:
    """Summarise the data an experiment actually consumed.

    Backstop, not primary defence. If a script bypasses `load_dataset` and
    loads its own data, the shape and class histogram will not match the
    manifest and the conformance check fails. Values are rounded before hashing
    so the digest is stable across platforms and BLAS versions.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    digest = hashlib.sha256(np.round(X, 6).tobytes()).hexdigest()[:16]
    return {
        "dataset": name,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]) if X.ndim > 1 else 1,
        "n_classes": int(len(np.unique(y))),
        "class_histogram": sorted(
            (int(c) for c in Counter(y.tolist()).values()), reverse=True
        ),
        "data_hash": digest,
        "seed": int(seed),
    }


def expected_fingerprint(name: str, seed: int = 0) -> dict:
    """Fingerprint a registered dataset by loading it fresh.

    The conformance check compares this against what the experiment reported,
    which is what catches a harness bypass.
    """
    X, y = load_dataset(name, seed)
    return fingerprint(X, y, name=name, seed=seed)


def is_feasible(name: str) -> bool:
    """True when a dataset is in the registry and therefore runnable in budget."""
    return name in REGISTRY


def list_feasible() -> list[str]:
    """Registered dataset names, sorted."""
    return sorted(REGISTRY)


def describe_for_planner() -> str:
    """Render the registry as a prompt fragment for the experiment designer.

    Showing the planner exactly what exists -- with the research questions each
    dataset can support -- is what stops it proposing CIFAR-10-LT in the first
    place. Cheaper to constrain the question than to detect a bad answer.
    """
    lines = []
    for spec in (REGISTRY[k] for k in sorted(REGISTRY)):
        lines.append(
            f"- {spec.name}: {spec.description} "
            f"({spec.n_samples} samples, {spec.n_features} features, "
            f"{spec.n_classes} classes, imbalance ratio {spec.imbalance_ratio}:1). "
            f"Suitable for: {', '.join(spec.research_uses)}."
        )
    return "\n".join(lines)
