"""
reviewer.py  — Research Agent
Strict IEEE peer reviewer scoring paper quality 0-10.

Fixes:
  1. Checks for duplicate abstract (critical issue)
  2. Checks for BNN/MC-dropout hallucination vs actual baselines
  3. Checks for markdown heading artifacts (####)
  4. Checks dataset-task alignment
  5. Checks intro first sentence completeness
  6. Quality score rubric updated: 9+ requires all critical issues resolved
"""
import json
from memory import state_store, audit_logger
from memory.token_tracker import log_usage, print_status
from memory.groq_client import create_client, call_with_retry
import config


def _pre_checks(paper_draft: dict, experiment_plan: dict) -> list:
    """
    Deterministic checks that don't need an LLM.
    Returns list of issue dicts with severity/section/description.
    """
    issues = []
    abstract = paper_draft.get("abstract", "")
    intro    = paper_draft.get("introduction", "")
    related  = paper_draft.get("related_work", "")
    method   = paper_draft.get("methodology", "")
    results  = paper_draft.get("results", "")
    full_text = " ".join([abstract, intro, related, method, results])

    # 1. Duplicate abstract
    if abstract and len(abstract) > 50:
        # Check if abstract text also appears in body sections
        first_50 = abstract[:50].lower().strip()
        body = " ".join([intro, related, method, results]).lower()
        if first_50 in body and len(first_50) > 20:
            issues.append({
                "severity": "critical",
                "section": "abstract",
                "description": (
                    "Abstract is duplicated as a freestanding body section. "
                    "IEEE format has ONE abstract in the header block only. "
                    "Remove the repeated ABSTRACT section from the body."
                )
            })

    # 2. BNN/MC-dropout hallucination
    bnn_phrases = ["bayesian neural network", "monte carlo dropout", "bnn",
                   "mc dropout", "variational inference"]
    baselines = [b["name"].lower() for b in experiment_plan.get("baselines", [])]
    bnn_in_baselines = any("bayesian" in b or "monte carlo" in b for b in baselines)
    if not bnn_in_baselines:
        for phrase in bnn_phrases:
            if phrase in full_text.lower():
                issues.append({
                    "severity": "critical",
                    "section": "methodology",
                    "description": (
                        "Paper claims '" + phrase + "' but experiment baselines are: " +
                        str([b["name"] for b in experiment_plan.get("baselines", [])]) +
                        ". Remove all BNN/MC-dropout references — they were not implemented."
                    )
                })
                break

    # 3. Markdown heading artifacts
    for sec_name, sec_content in paper_draft.items():
        if isinstance(sec_content, str) and "####" in sec_content:
            issues.append({
                "severity": "major",
                "section": sec_name,
                "description": (
                    "Raw markdown heading markers (####) found in " + sec_name +
                    ". These render as visible text in HTML. "
                    "Replace with plain subsection labels: 'A. Theme', 'B. Theme'."
                )
            })
            break

    # 4. Intro first sentence fragment
    if intro:
        first_char = intro.lstrip()[0] if intro.strip() else ""
        if first_char in [",", ".", ";", ":"]:
            issues.append({
                "severity": "critical",
                "section": "introduction",
                "description": (
                    "Introduction begins with '" + first_char + "' — truncated first sentence. "
                    "The introduction must start with a complete sentence."
                )
            })

    # 5. Dataset-task mismatch check
    dataset_names = [d["name"].lower() for d in experiment_plan.get("datasets", [])]
    topic = ""  # Will be checked by LLM below
    for ds in dataset_names:
        if "20 newsgroups" in ds:
            # Check if topic is hallucination-related (mismatch)
            if any(w in full_text.lower() for w in ["hallucination", "llm reliability",
                                                       "truthfulness"]):
                issues.append({
                    "severity": "critical",
                    "section": "experiments",
                    "description": (
                        "Dataset '20 Newsgroups' is a general text classification dataset "
                        "but the paper claims to detect LLM hallucinations — these are "
                        "different tasks. Use HaluEval or TruthfulQA for hallucination detection, "
                        "or reframe the task as 'text classification with uncertainty quantification'."
                    )
                })

    # 6. Key Findings debug block
    if "<h3>Key Findings</h3>" in full_text or "Key Findings" in results:
        issues.append({
            "severity": "major",
            "section": "results",
            "description": (
                "A 'Key Findings' bullet block appears in the results section. "
                "This is debug output, not IEEE paper content. "
                "Remove it — findings belong in results prose."
            )
        })

    return issues


def run():
    client         = create_client()
    paper_draft    = state_store.get_state("paper_draft")
    paper_list     = state_store.get_state("paper_list")
    result_summary = state_store.get_state("result_summary")
    input_topic    = state_store.get_state("input_topic")
    experiment_plan = state_store.get_state("experiment_plan") or {}

    if not paper_draft:
        raise ValueError("[Reviewer] No paper_draft in state.")

    paper_titles    = [p["title"] for p in paper_list["papers"]]
    actual_metrics  = result_summary.get("metrics", [])
    domain          = input_topic.get("domain", "")
    topic           = input_topic.get("topic", "")
    modality        = experiment_plan.get("modality", "tabular")

    print("[Reviewer] Running deterministic pre-checks...")
    pre_issues = _pre_checks(paper_draft, experiment_plan)
    print("[Reviewer] Pre-check issues: " + str(len(pre_issues)))

    print("[Reviewer] Running LLM review...")
    prompt = (
        "Review this paper as a strict IEEE Transactions journal reviewer.\n\n"
        "Domain: " + domain + "\n"
        "Topic: " + topic + "\n"
        "Data modality: " + modality + "\n"
        "Actual baselines used: " + str([b["name"] for b in experiment_plan.get("baselines",[])]) + "\n\n"
        "ABSTRACT:\n" + paper_draft.get("abstract","")[:500] + "\n\n"
        "INTRODUCTION:\n" + paper_draft.get("introduction","")[:500] + "\n\n"
        "RELATED WORK:\n" + paper_draft.get("related_work","")[:400] + "\n\n"
        "METHODOLOGY:\n" + paper_draft.get("methodology","")[:500] + "\n\n"
        "RESULTS:\n" + paper_draft.get("results","")[:600] + "\n\n"
        "LIMITATIONS:\n" + paper_draft.get("limitations","")[:400] + "\n\n"
        "Allowed citations (ONLY these): " + json.dumps(paper_titles[:12]) + "\n"
        "Actual metrics: " + json.dumps(actual_metrics[:12]) + "\n\n"
        "Check ALL of these:\n"
        "1. Citations not in allowed list — CRITICAL\n"
        "2. Numbers not matching actual metrics — CRITICAL\n"
        "3. Method claims models not in baselines list (e.g. BNN) — CRITICAL\n"
        "4. Abstract appears twice in paper — CRITICAL\n"
        "5. Introduction starts with incomplete sentence — CRITICAL\n"
        "6. Dataset task mismatch with paper topic — CRITICAL\n"
        "7. Markdown heading artifacts (####) in text — MAJOR\n"
        "8. Citations irrelevant to " + domain + " — MAJOR\n"
        "9. Unsupported claims without evidence — MAJOR\n"
        "10. Repetitive content across sections — MINOR\n"
        "11. Weak or vague future work — MINOR\n"
        "12. Table caption format wrong (should be TABLE I above table) — MINOR\n\n"
        "QUALITY SCORE RUBRIC:\n"
        "9-10: Zero critical issues, zero major issues, paper ready for submission\n"
        "7-8:  Zero critical issues, 1-2 major issues\n"
        "5-6:  1-2 critical issues\n"
        "3-4:  3+ critical issues\n"
        "1-2:  Fundamental problems\n\n"
        "Return ONLY this JSON:\n"
        "{\n"
        '  "issues": [{"severity": "minor", "section": "results", "description": "issue"}],\n'
        '  "overall_verdict": "accept",\n'
        '  "required_changes": ["change1"],\n'
        '  "citation_audit_passed": true,\n'
        '  "number_audit_passed": true,\n'
        '  "domain_relevance_passed": true,\n'
        '  "hallucination_flags": [],\n'
        '  "quality_score": 8.5,\n'
        '  "strengths": ["strength1"],\n'
        '  "recommended_additional_citations": ["topic"]\n'
        "}"
    )

    raw = call_with_retry(
        client,
        messages=[
            {"role": "system", "content": (
                "Strict IEEE reviewer for " + domain + ". "
                "Apply the quality score rubric exactly. Return ONLY valid JSON."
            )},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1800,
        temperature=0.2,
        agent_name="reviewer"
    )
    used = log_usage("reviewer", prompt, raw)
    print_status("reviewer", used)

    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.split("```")[0]

    result = json.loads(raw)

    # Merge deterministic pre-check issues into LLM issues
    existing_descs = {i["description"][:60] for i in result.get("issues", [])}
    for pi in pre_issues:
        if pi["description"][:60] not in existing_descs:
            result.setdefault("issues", []).insert(0, pi)

    # Re-compute quality score based on merged issue counts
    n_critical = sum(1 for i in result["issues"] if i["severity"] == "critical")
    n_major    = sum(1 for i in result["issues"] if i["severity"] == "major")
    if n_critical == 0 and n_major == 0:
        result["quality_score"] = max(result.get("quality_score", 8.0), 8.5)
    elif n_critical == 0 and n_major <= 2:
        result["quality_score"] = min(result.get("quality_score", 7.5), 8.0)
    elif n_critical <= 2:
        result["quality_score"] = min(result.get("quality_score", 6.0), 6.5)
    else:
        result["quality_score"] = min(result.get("quality_score", 4.0), 5.0)

    state_store.update_state("reviewer_feedback", result)
    audit_logger.log("reviewer", {"sections": list(paper_draft.keys())}, result)

    print("[Reviewer] Verdict: " + result["overall_verdict"])
    print("[Reviewer] Quality score: " + str(result.get("quality_score","N/A")) + "/10")
    for issue in result.get("issues", []):
        print("  [" + issue["severity"].upper() + "] " +
              issue["section"] + ": " + issue["description"][:100])
    return result
