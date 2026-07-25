"""Is the verdict the paper claims the verdict the numbers support?

This is the third gate, and it closes the last place the pipeline could still
lie. Conformance proved the experiment ran on the planned data; this proves the
*conclusion* drawn from that run is one the data actually licenses.

WHY THIS EXISTS
    In the original failure the verdict was a string an LLM chose --
    "supported" -- and every downstream stage repeated it. Even with honest
    data, a model asked "did your hypothesis hold?" will tend to say yes. The
    verdict has to be computed from the measurements by a rule fixed *before*
    the measurements exist, not narrated after the fact.

THE RULE (pre-registered)
    The experiment reports, for its primary metric, paired per-fold scores for
    two conditions: the method under test and a baseline. From the per-fold
    differences we compute a paired t-test, an effect size, and a confidence
    interval, and map them to one of three verdicts by the fixed rule in
    `classify`. The thresholds below are the whole policy; they are named
    constants precisely so the rule can be read and argued with in one place.

    Three-way is the point. A two-way supported/refuted rule driven by p-value
    alone reports "refuted" on every small-n null result and reads as broken.
    Here REFUTED means *significantly worse*, SUPPORTED means *significantly
    better by a margin worth caring about*, and INCONCLUSIVE -- the common,
    honest outcome -- means the run does not settle the question. A pipeline
    that can never say "I don't know" is still lying.

WHAT HALTS THE RUN
    Not a negative result -- an honest "inconclusive" or "refuted" is exactly
    what this project wants the pipeline to be able to produce. What halts is an
    *overclaim*: the script/paper asserting "supported" (or "refuted") when the
    deterministic test does not agree. That mismatch is the iris fabrication in
    verdict form -- a claim the experiment did not establish -- so it is
    CRITICAL. Claiming "inconclusive" is never an overclaim and never halts.

    Like `conformance.fingerprint`, this is not a total guarantee: the per-fold
    scores are still reported by the generated script. It removes the model's
    licence to *narrate the verdict*, and grounds the verdict in arithmetic
    anyone can re-run, which is the load-bearing move. Making the scores
    themselves harness-computed is a later, larger step (see the project notes).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy import stats

from verification.findings import CheckResult, Finding, Severity

# ---------------------------------------------------------------------------
# Pre-registered decision rule. These constants ARE the policy: committing them
# here, before any particular run's numbers exist, is what stops the rule being
# quietly bent to make a favoured result come out "supported".
# ---------------------------------------------------------------------------

#: Significance level for the two-sided paired t-test.
ALPHA = 0.05

#: Smallest effect size of interest, on the metric's own scale (metrics here are
#: in [0, 1]). An improvement smaller than this is not worth claiming even if it
#: is statistically detectable, so a significant-but-trivial result is
#: INCONCLUSIVE rather than SUPPORTED. This is the effect-size half of the rule.
MARGIN = 0.01

#: Fewest paired folds a real test needs. Below this there is no honest variance
#: estimate, so the verdict is INCONCLUSIVE regardless of the point difference.
MIN_PAIRED = 3

#: Spread below this is floating-point noise, not signal. Per-fold metric scores
#: are in [0, 1] and reported to a few decimals, so a difference std this small
#: means the folds are effectively constant. Below it we skip scipy's t-test --
#: which divides by a near-zero standard error and (rightly) warns of unreliable
#: precision -- and read the constant difference directly, which is what it is.
_MIN_STD = 1e-9

_CHECK = "significance"


class Verdict(str, Enum):
    """The three outcomes a run is allowed to reach.

    A str-Enum so it serialises straight into results.json and state without a
    custom encoder, and compares equal to its own value in tests.
    """

    SUPPORTED = "supported"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"

    def __str__(self) -> str:
        return self.value


# Claim strings the generated script may write. Anything a run asserts that maps
# to a *definite* verdict is checkable; "inconclusive" and unrecognised claims
# cannot be an overclaim of a positive result.
_DEFINITE_CLAIMS = {"supported", "refuted"}


@dataclass(frozen=True)
class Assessment:
    """The computed verdict plus the evidence that produced it.

    Frozen for the same reason `Finding` is: the verdict is a record derived by
    a fixed rule, and nothing downstream should be able to edit it into a more
    flattering shape. `evidence` carries the raw statistics so a reader (or the
    paper) can show its work.
    """

    verdict: Verdict
    evidence: dict


def _paired_stats(method: np.ndarray, baseline: np.ndarray) -> dict:
    """Paired t-test, effect size and CI for method-minus-baseline differences.

    Kept separate from the decision so the numbers can be inspected on their
    own. Handles the zero-variance case scipy would return NaN for: identical
    differences on every fold are perfectly consistent evidence, not missing
    evidence.
    """
    diff = method - baseline
    n = int(diff.shape[0])
    mean_diff = float(np.mean(diff))
    # Sample std with ddof=1; the paired t-test and dz both use it.
    std_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    df = n - 1

    if std_diff <= _MIN_STD:
        # Effectively no spread in the differences. If they are all essentially
        # zero there is genuinely no effect (p=1); otherwise the effect is
        # perfectly reliable (p->0). Either way scipy would divide by a near-zero
        # standard error and return NaN with a precision-loss warning, so we read
        # the constant difference directly instead.
        p_value = 1.0 if mean_diff == 0.0 else 0.0
        effect_size = 0.0 if mean_diff == 0.0 else math.inf * (1 if mean_diff > 0 else -1)
        ci_low = ci_high = mean_diff
    else:
        _t_stat, p_value = stats.ttest_rel(method, baseline)
        p_value = float(p_value)
        se = std_diff / math.sqrt(n)
        t_crit = float(stats.t.ppf(1 - ALPHA / 2, df))
        ci_low = mean_diff - t_crit * se
        ci_high = mean_diff + t_crit * se
        effect_size = mean_diff / std_diff  # Cohen's dz for paired samples

    return {
        "n_paired": n,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "p_value": p_value,
        "effect_size_dz": effect_size,
        "ci95": [ci_low, ci_high],
        "alpha": ALPHA,
        "margin": MARGIN,
    }


def classify(method_scores, baseline_scores) -> Assessment:
    """Map paired per-fold scores to a three-way verdict by the fixed rule.

    Pure and deterministic: same scores in, same verdict out, no LLM, no state.
    This is the function an interviewer can read in a minute and confirm cannot
    be talked into "supported".
    """
    method = np.asarray(method_scores, dtype=float).ravel()
    baseline = np.asarray(baseline_scores, dtype=float).ravel()

    if method.shape[0] != baseline.shape[0]:
        return Assessment(
            Verdict.INCONCLUSIVE,
            {
                "reason": "unpaired",
                "n_method": int(method.shape[0]),
                "n_baseline": int(baseline.shape[0]),
            },
        )

    if method.shape[0] < MIN_PAIRED:
        return Assessment(
            Verdict.INCONCLUSIVE,
            {"reason": "insufficient_runs", "n_paired": int(method.shape[0]),
             "min_required": MIN_PAIRED},
        )

    stats_ = _paired_stats(method, baseline)
    significant = stats_["p_value"] < ALPHA
    mean_diff = stats_["mean_diff"]

    if significant and mean_diff >= MARGIN:
        verdict = Verdict.SUPPORTED
    elif significant and mean_diff <= -MARGIN:
        verdict = Verdict.REFUTED
    else:
        verdict = Verdict.INCONCLUSIVE

    return Assessment(verdict, stats_)


# ---------------------------------------------------------------------------
# Reading the reported results into paired arrays
# ---------------------------------------------------------------------------


def _extract_conditions(results: dict) -> tuple[list, list] | None:
    """Pull paired method/baseline per-fold scores out of a results dict.

    Returns None when the structure the rule needs is absent, so the caller can
    distinguish "no comparison to test" from "a comparison that came out null".
    Tolerant of the shape drift an LLM-authored file tends to have.
    """
    conditions = results.get("conditions")
    if not isinstance(conditions, dict):
        return None
    method = conditions.get("method") or {}
    baseline = conditions.get("baseline") or {}
    m_scores = method.get("scores") if isinstance(method, dict) else None
    b_scores = baseline.get("scores") if isinstance(baseline, dict) else None
    if not isinstance(m_scores, list) or not isinstance(b_scores, list):
        return None
    if not m_scores or not b_scores:
        return None
    return m_scores, b_scores


def evaluate(results: dict) -> Assessment:
    """The trusted verdict for a run, computed from its reported scores.

    This is what downstream stages should report instead of the script's
    self-authored `hypothesis_verdict`. When the run reports no paired
    comparison, the verdict is INCONCLUSIVE with a reason -- never a default to
    the claim.
    """
    conditions = _extract_conditions(results)
    if conditions is None:
        return Assessment(
            Verdict.INCONCLUSIVE,
            {"reason": "no_comparison_reported"},
        )
    return classify(*conditions)


def audit_significance(results: dict | None) -> CheckResult:
    """Halt only when the claimed verdict overclaims what the numbers support.

    The verdict itself is never a reason to halt -- refuted and inconclusive are
    honest, wanted outcomes. What is CRITICAL is a run asserting a *definite*
    verdict (supported/refuted) that the deterministic rule does not reach: that
    is a conclusion the experiment did not establish, which is the exact shape
    of the original fabrication.
    """
    if not results:
        # Nothing reported at all. This is genuinely vacuous -- there is no
        # evidence to examine -- and the gate treats a vacuous pass as a halt,
        # which is right: a run that reported nothing has established nothing.
        return CheckResult(check=_CHECK)

    assessment = evaluate(results)
    computed = assessment.verdict
    claimed_raw = str(results.get("hypothesis_verdict", "")).strip().lower()

    examined = (
        f"claimed={claimed_raw or '(none)'}",
        f"computed={computed}",
        f"evidence={assessment.evidence}",
    )

    # An overclaim: the run asserts a definite verdict the rule does not agree
    # with. Claiming "inconclusive" (or nothing recognisable) can never overclaim
    # a positive result, so it is always allowed through.
    if claimed_raw in _DEFINITE_CLAIMS and claimed_raw != computed.value:
        return CheckResult(
            check=_CHECK,
            findings=(
                Finding(
                    check=_CHECK,
                    severity=Severity.CRITICAL,
                    summary=(
                        f"Reported verdict {claimed_raw!r} is not supported by the "
                        f"data: the pre-registered test computes {computed.value!r}. "
                        "The paper would assert a conclusion the experiment did not "
                        "establish."
                    ),
                    expected=f"verdict consistent with the test: {computed.value}",
                    observed=claimed_raw,
                    remediation=(
                        "Report the computed verdict. If the claim is right and the "
                        "rule is wrong, change the rule in verification/statistics.py "
                        "-- a reviewable diff -- not this one run's numbers."
                    ),
                ),
            ),
            examined=examined,
        )

    # A run that under-claims (said inconclusive while the test found support) is
    # honest and conservative, not a fabrication. Surface it, do not halt.
    findings: tuple[Finding, ...] = ()
    if claimed_raw not in _DEFINITE_CLAIMS and computed is not Verdict.INCONCLUSIVE:
        findings = (
            Finding(
                check=_CHECK,
                severity=Severity.MINOR,
                summary=(
                    f"Run reported {claimed_raw or '(none)'!r} but the test computes "
                    f"{computed.value!r}; reporting the stronger, evidenced verdict "
                    "would be fair to the result."
                ),
                expected=computed.value,
                observed=claimed_raw or "(none)",
            ),
        )

    return CheckResult(check=_CHECK, findings=findings, examined=examined)
