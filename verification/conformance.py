"""Does the experiment that ran match the experiment that was planned?

Two checks, at two moments:

    audit_feasibility  -- design time. Every planned dataset must exist in the
                          harness registry. This is the check that would have
                          stopped the original failure before a line of code
                          was generated: "CIFAR-10-LT" is not registered, so
                          the plan is rejected and there is nothing to
                          substitute.

    audit_conformance  -- after execution. The data the experiment actually
                          consumed, as recorded in its manifest, must match a
                          fresh fingerprint of the planned dataset. This is the
                          backstop for a script that ignores the harness and
                          loads its own data.

Both are pure functions over plain dicts. No LLM, no network, no global state.
An interviewer can read either one in under a minute and satisfy themselves it
cannot be talked out of its answer -- which is precisely what the LLM "audits"
they replace could not offer.
"""

from __future__ import annotations

from harness import registry
from verification.findings import CheckResult, Finding, Severity

# Fields that must match exactly between the manifest and a fresh load of the
# planned dataset. Shape and class balance are what actually distinguish one
# dataset from another; the hash catches same-shape substitutions.
_FINGERPRINT_FIELDS = (
    "n_samples",
    "n_features",
    "n_classes",
    "class_histogram",
    "data_hash",
)


def planned_datasets(experiment_plan: dict) -> list[str]:
    """Dataset names from a plan, tolerating both shapes the designer emits.

    The designer has historically returned datasets either as dicts with a
    "name" key or as bare strings. Accepting both here keeps the check robust
    to prompt drift -- a check that crashes on an unexpected shape is a check
    that silently stops protecting you.
    """
    raw = experiment_plan.get("datasets") or []
    names: list[str] = []
    for entry in raw:
        if isinstance(entry, dict):
            name = entry.get("name")
        else:
            name = entry
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return names


def audit_feasibility(experiment_plan: dict) -> CheckResult:
    """Reject any plan naming a dataset outside the feasible registry.

    This is the load-bearing check. The pipeline fabricated results because it
    was handed a brief it could not satisfy -- CIFAR-10-LT and a VAE, under a
    sklearn/CPU/60s budget -- and resolved the contradiction by quietly
    swapping in iris. Removing impossible briefs at the source is a stronger
    fix than detecting their symptoms downstream.
    """
    planned = planned_datasets(experiment_plan)

    if not planned:
        return CheckResult(
            check="feasibility",
            findings=(
                Finding(
                    check="feasibility",
                    severity=Severity.CRITICAL,
                    summary="Plan names no dataset, so no experiment can be verified against it.",
                    expected="at least one dataset from the registry",
                    observed="none",
                    remediation=f"Choose one of: {', '.join(registry.list_feasible())}",
                ),
            ),
            examined=("experiment_plan.datasets",),
        )

    findings = []
    for name in planned:
        if registry.is_feasible(name):
            continue
        findings.append(
            Finding(
                check="feasibility",
                severity=Severity.CRITICAL,
                summary=(
                    f"Planned dataset {name!r} is not in the feasible registry. "
                    "Generating code for it would force the model to substitute "
                    "something it can actually run."
                ),
                expected=f"one of: {', '.join(registry.list_feasible())}",
                observed=name,
                remediation=(
                    "Re-plan against a registered dataset, or add this dataset to "
                    "harness/registry.py once it is proven to run within budget."
                ),
            )
        )

    return CheckResult(
        check="feasibility",
        findings=tuple(findings),
        examined=tuple(planned),
    )


def audit_conformance(experiment_plan: dict, manifest: dict | None) -> CheckResult:
    """Verify the executed experiment consumed the planned data.

    Compares the manifest the run emitted against a fresh fingerprint of the
    planned dataset. A missing manifest is itself critical: without it there is
    no evidence of what ran, and "no evidence" must never read as "passed".
    That conflation is exactly how the original reviewer approved a paper whose
    experiment it had never seen.
    """
    planned = planned_datasets(experiment_plan)

    if not planned:
        return CheckResult(
            check="conformance",
            findings=(
                Finding(
                    check="conformance",
                    severity=Severity.CRITICAL,
                    summary="No planned dataset to compare the run against.",
                    expected="a planned dataset name",
                    observed="none",
                    remediation="Fix the experiment plan before running conformance.",
                ),
            ),
            examined=("experiment_plan.datasets",),
        )

    if not manifest:
        return CheckResult(
            check="conformance",
            findings=(
                Finding(
                    check="conformance",
                    severity=Severity.CRITICAL,
                    summary=(
                        "Experiment emitted no data manifest, so what it actually "
                        "ran on cannot be established."
                    ),
                    expected="outputs/code/manifest.json written by the harness",
                    observed="missing",
                    remediation=(
                        "Ensure the generated script loads data via harness.load_dataset(), "
                        "which writes the manifest as a side effect."
                    ),
                ),
            ),
            examined=("outputs/code/manifest.json",),
        )

    expected_name = planned[0]
    observed_name = manifest.get("dataset", "")

    if observed_name != expected_name:
        return CheckResult(
            check="conformance",
            findings=(
                Finding(
                    check="conformance",
                    severity=Severity.CRITICAL,
                    summary=(
                        f"Experiment ran on {observed_name or '(unnamed)'!r} "
                        f"but the plan specified {expected_name!r}. Any conclusion "
                        "drawn from this run is about the wrong data."
                    ),
                    expected=expected_name,
                    observed=observed_name or "(unnamed)",
                    remediation="Regenerate the experiment using harness.load_dataset().",
                ),
            ),
            examined=(f"manifest.dataset={observed_name}", f"plan={expected_name}"),
        )

    # Name matches; now confirm the bytes do. This is what catches a script
    # that writes a truthful-looking manifest while loading different data.
    if not registry.is_feasible(expected_name):
        return CheckResult(
            check="conformance",
            findings=(
                Finding(
                    check="conformance",
                    severity=Severity.CRITICAL,
                    summary=f"Cannot verify {expected_name!r}: not in the registry.",
                    expected=f"one of: {', '.join(registry.list_feasible())}",
                    observed=expected_name,
                    remediation="Run the feasibility check before conformance.",
                ),
            ),
            examined=(expected_name,),
        )

    truth = registry.expected_fingerprint(
        expected_name, seed=int(manifest.get("seed", 0))
    )
    findings = []
    for field in _FINGERPRINT_FIELDS:
        want, got = truth.get(field), manifest.get(field)
        if want == got:
            continue
        findings.append(
            Finding(
                check="conformance",
                severity=Severity.CRITICAL,
                summary=(
                    f"Data fingerprint mismatch on {field!r}: the experiment did not "
                    f"consume the registered {expected_name!r} data."
                ),
                expected=str(want),
                observed=str(got),
                remediation=(
                    "The script bypassed harness.load_dataset(). Regenerate it "
                    "against the harness API."
                ),
            )
        )

    return CheckResult(
        check="conformance",
        findings=tuple(findings),
        examined=tuple(f"{f}={manifest.get(f)}" for f in _FINGERPRINT_FIELDS),
    )
