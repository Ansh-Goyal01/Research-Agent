"""
consistency_checker.py  — Research Agent
Pre-review deterministic checks before the LLM reviewer runs.

Added checks:
  - Duplicate abstract detection
  - BNN hallucination detection
  - Markdown heading artifact detection (####)
  - Dataset-task mismatch
  - Key Findings debug block detection
"""
import re


def check_numbers(sections, result_summary):
    issues = []
    actual_metrics = result_summary.get("metrics", [])
    actual_numbers = set()
    for m in actual_metrics:
        actual_numbers.add(str(round(m["mean"], 2)))
        actual_numbers.add(str(round(m["mean"] * 100, 1)))
        actual_numbers.add(str(round(m["mean"] * 100, 2)))

    number_pattern = re.compile(r'\b\d+\.\d+\b')
    for section_name in ["abstract", "results", "discussion", "conclusion"]:
        content = sections.get(section_name, "")
        numbers_found = number_pattern.findall(content)
        for num in numbers_found:
            val = float(num)
            if 50 < val < 100:
                close = any(
                    abs(val - float(n)) < 2.0
                    for n in actual_numbers if n
                )
                if not close and len(actual_numbers) > 0:
                    issues.append({
                        "type": "number_mismatch",
                        "section": section_name,
                        "value": num,
                        "message": num + "% in " + section_name + " does not match actual metrics"
                    })
    return issues


def check_contributions(sections):
    issues = []
    intro       = sections.get("introduction", "")
    methodology = sections.get("methodology", "")

    contrib_count = len(re.findall(r'\(\d+\)|\d+\)', intro))
    if contrib_count == 0:
        issues.append({
            "type": "missing_contributions",
            "section": "introduction",
            "message": "No numbered contributions list found in introduction"
        })

    if len(methodology) < 500:
        issues.append({
            "type": "thin_methodology",
            "section": "methodology",
            "message": "Methodology is " + str(len(methodology)) + " chars — needs 2000+"
        })
    return issues


def check_citations(sections, paper_list):
    issues = []
    paper_titles = [p["title"].lower() for p in paper_list.get("papers", [])]
    full_text = " ".join([
        sections.get("introduction", ""),
        sections.get("related_work", ""),
        sections.get("methodology", ""),
        sections.get("discussion", "")
    ]).lower()

    citation_pattern = re.compile(r'\[(\d+)\]')
    cited_numbers    = set(citation_pattern.findall(full_text))
    max_ref          = len(paper_titles)

    for num in cited_numbers:
        if int(num) > max_ref:
            issues.append({
                "type": "invalid_citation",
                "section": "unknown",
                "message": "Citation [" + num + "] exceeds available papers (" + str(max_ref) + ")"
            })
    return issues


def check_section_lengths(sections):
    issues = []
    minimums = {
        "abstract":     600,
        "introduction": 1500,
        "related_work": 1500,
        "methodology":  2000,
        "experiments":  1500,
        "results":      1500,
        "discussion":   1500,
        "conclusion":    800,
        "limitations":   600,
    }
    for section, min_len in minimums.items():
        content = sections.get(section, "")
        if len(content) < min_len:
            issues.append({
                "type": "short_section",
                "section": section,
                "message": (section + " is " + str(len(content)) +
                            " chars, needs " + str(min_len) + "+")
            })
    return issues


def check_duplicate_abstract(sections):
    issues = []
    abstract = sections.get("abstract", "")
    if not abstract or len(abstract) < 50:
        return issues
    first_50  = abstract[:50].lower().strip()
    body_text = " ".join([
        sections.get("introduction", ""),
        sections.get("related_work", ""),
        sections.get("methodology", ""),
    ]).lower()
    if len(first_50) > 20 and first_50 in body_text:
        issues.append({
            "type": "duplicate_abstract",
            "section": "abstract",
            "message": "Abstract text found repeated in body sections — remove freestanding ABSTRACT heading"
        })
    return issues


def check_bnn_hallucination(sections, experiment_plan):
    issues = []
    baselines = [b["name"].lower() for b in experiment_plan.get("baselines", [])]
    bnn_in_baselines = any("bayesian" in b or "monte carlo" in b for b in baselines)
    if bnn_in_baselines:
        return issues

    full_text = " ".join([
        sections.get("abstract", ""),
        sections.get("methodology", ""),
        sections.get("introduction", ""),
    ]).lower()

    for phrase in ["bayesian neural network", "monte carlo dropout", " bnn ", "mc dropout"]:
        if phrase in full_text:
            issues.append({
                "type": "bnn_hallucination",
                "section": "methodology",
                "message": (
                    "Paper mentions '" + phrase.strip() + "' but actual baselines are: " +
                    str([b["name"] for b in experiment_plan.get("baselines", [])]) +
                    ". Remove BNN references."
                )
            })
            break
    return issues


def check_markdown_artifacts(sections):
    issues = []
    for sec_name, content in sections.items():
        if isinstance(content, str) and "####" in content:
            issues.append({
                "type": "markdown_artifacts",
                "section": sec_name,
                "message": "Raw #### markdown headings in " + sec_name + " — replace with 'A. Theme' labels"
            })
    return issues


def check_debug_blocks(sections):
    issues = []
    results = sections.get("results", "")
    if "Key Findings" in results or "key_findings" in results.lower():
        issues.append({
            "type": "debug_block",
            "section": "results",
            "message": "Key Findings debug block found in results — remove it, findings belong in prose"
        })
    return issues


def run_all_checks(sections, result_summary, paper_list, experiment_plan=None):
    print("[ConsistencyChecker] Running checks...")
    all_issues = []
    all_issues.extend(check_numbers(sections, result_summary))
    all_issues.extend(check_contributions(sections))
    all_issues.extend(check_citations(sections, paper_list))
    all_issues.extend(check_section_lengths(sections))
    all_issues.extend(check_duplicate_abstract(sections))
    all_issues.extend(check_markdown_artifacts(sections))
    all_issues.extend(check_debug_blocks(sections))
    if experiment_plan:
        all_issues.extend(check_bnn_hallucination(sections, experiment_plan))

    if all_issues:
        print("[ConsistencyChecker] Found " + str(len(all_issues)) + " issues:")
        for issue in all_issues:
            print("  [" + issue["type"].upper() + "] " +
                  issue["section"] + ": " + issue["message"][:100])
    else:
        print("[ConsistencyChecker] All checks passed.")

    return all_issues
