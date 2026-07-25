"""Does every number in the paper trace back to a number the experiment produced?

The fourth and last gate on the fabrication spine. Conformance proved the run
used the planned data; significance proved the verdict matches the numbers; this
proves the *paper's prose* does not invent numbers the results never contained.

WHY THIS EXISTS
    The paper is written section-by-section by an LLM from a results summary.
    Nothing downstream checks that the figures it types into the Results section
    are the ones the experiment measured. The original fabrication was exactly a
    number with no run behind it -- 0.9733 -- so a paper that can state any
    figure it likes has reopened the hole one stage after conformance closed it.

WHAT IT CHECKS
    Build the set of numbers the experiment actually produced (metric means and
    stds, the verdict's p-value / effect size / CI). Then scan the paper for
    *metric-shaped* numbers -- figures with two or more decimals, or explicit
    percentages, which is what performance claims look like -- and require each
    to match a produced number within a rounding tolerance. A metric-shaped
    figure matching nothing is a claim with no experiment behind it: CRITICAL.

WHAT IT DELIBERATELY DOES NOT CHECK
    Structural integers -- sample counts, seeds, fold counts, years -- are not
    examined, because they are not where fabrication hides and flagging them
    would make the gate cry wolf. A gate that is often wrong is ignored, which
    is worse than no gate. The matching is generous on purpose (rounding, the
    x100 percentage form, and mean +/- std) so an honestly-reported result is
    never halted for phrasing a real number a little differently.
"""

from __future__ import annotations

import re
from numbers import Number

from verification.findings import CheckResult, Finding, Severity

_CHECK = "reporting"

#: A paper figure counts as matching a produced number if it is within this of
#: one (after rounding). Loose enough for honest prose rounding (0.677 -> 0.68),
#: tight enough that a fabricated 0.95 for a real 0.73 cannot slip through.
_TOL = 0.005

#: Metric-shaped tokens: a decimal with >=2 fractional digits (0.73, 97.33), or
#: any explicit percentage (95%, 4.2%). Integers and one-decimal numbers (5,
#: 0.5 hours, 8.0 GB) are intentionally excluded -- see the module docstring.
_DECIMAL = re.compile(r"\d+\.\d{2,}")
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def _iter_numbers(obj):
    """Yield every numeric leaf in a nested dict/list, skipping bools.

    bool is a subclass of int; a stray True in the evidence must not be read as
    the number 1 and silently widen the allowed set.
    """
    if isinstance(obj, bool):
        return
    if isinstance(obj, Number):
        yield float(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_numbers(v)


def _produced_values(result_summary: dict) -> set[float]:
    """Every number the experiment actually produced, as a set for matching.

    Drawn from the structured summary only -- metric means/stds and the verdict
    evidence -- never from prose fields like key_findings, which are themselves
    LLM-authored and are the thing being checked, not a source of truth.
    """
    values: set[float] = set()
    for v in _iter_numbers(result_summary.get("metrics", [])):
        values.add(v)
    for v in _iter_numbers(result_summary.get("verdict_evidence", {})):
        values.add(v)

    # mean +/- std is a normal way to phrase a result in prose; treat both ends
    # as produced so "0.73 +/- 0.05" or "from 0.68 to 0.78" does not false-flag.
    for metric in result_summary.get("metrics", []):
        if isinstance(metric, dict):
            mean, std = metric.get("mean"), metric.get("std")
            if isinstance(mean, Number) and isinstance(std, Number):
                values.add(float(mean) + float(std))
                values.add(float(mean) - float(std))
    return values


def _candidates(value: float, is_percent: bool) -> tuple[float, ...]:
    """The metric values a paper figure could plausibly denote.

    The x100 form is applied only where it is legitimate: to an explicit
    percentage, and to a bare decimal above 1 (a percentage written without the
    sign, e.g. "97.33"). It must NOT be applied to a plain fraction like 0.95 --
    reading that as 0.0095 would let a fabricated 0.95 match a tiny CI bound and
    slip through, which is exactly the failure this guards against.
    """
    if is_percent:
        return (value / 100.0, value)
    if value > 1.0:
        return (value, value / 100.0)
    return (value,)


def _matches(value: float, is_percent: bool, produced: set[float]) -> bool:
    """True if a paper figure matches any produced number within tolerance."""
    for candidate in _candidates(value, is_percent):
        for truth in produced:
            if abs(round(candidate - truth, 6)) <= _TOL:
                return True
    return False


def _paper_figures(paper_text: str) -> list[tuple[str, float, bool]]:
    """Extract metric-shaped figures as (token, value, is_percent) triples.

    Percentages are captured whole (including the sign) so the message can quote
    what the paper actually said; the decimal value is what gets matched.
    """
    figures: list[tuple[str, float, bool]] = []
    for m in _PERCENT.finditer(paper_text):
        figures.append((m.group(0).strip(), float(m.group(1)), True))
    for m in _DECIMAL.finditer(paper_text):
        figures.append((m.group(0), float(m.group(0)), False))
    return figures


def audit_reported_numbers(paper_text: str, result_summary: dict | None) -> CheckResult:
    """Flag any metric-shaped number in the paper with no produced number behind it.

    A single unmatched performance figure is CRITICAL: it is a claim the
    experiment did not establish, which is the exact shape of the original
    fabrication. A paper that reports only produced numbers passes; a paper with
    no metric-shaped numbers at all is vacuous (the gate then halts it -- a
    Results section that states no numbers has not reported the experiment).
    """
    if not result_summary:
        return CheckResult(check=_CHECK)

    produced = _produced_values(result_summary)
    figures = _paper_figures(paper_text or "")

    if not figures:
        # Nothing metric-shaped to verify. Do not silently pass -- an empty
        # examined set marks this vacuous, and the gate treats that as a halt.
        return CheckResult(check=_CHECK)

    findings = []
    for token, value, is_percent in figures:
        if _matches(value, is_percent, produced):
            continue
        findings.append(
            Finding(
                check=_CHECK,
                severity=Severity.CRITICAL,
                summary=(
                    f"Paper states {token!r}, which matches no number the "
                    "experiment produced. The paper would report a result the "
                    "run did not measure."
                ),
                expected="a figure present in the verified results",
                observed=token,
                remediation=(
                    "Report only numbers from results.json / result_summary. If "
                    "this figure is real, it belongs in the results the harness "
                    "recorded, not only in the prose."
                ),
            )
        )

    return CheckResult(
        check=_CHECK,
        findings=tuple(findings),
        examined=tuple(tok for tok, _, _ in figures),
    )
