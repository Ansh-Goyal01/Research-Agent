import json
from memory import state_store, audit_logger
from memory.token_tracker import log_usage, print_status
from memory.groq_client import create_client, call_with_retry
import config


def run():
    client = create_client()
    paper_draft = state_store.get_state("paper_draft")
    paper_list = state_store.get_state("paper_list")
    result_summary = state_store.get_state("result_summary")
    input_topic = state_store.get_state("input_topic")

    if not paper_draft:
        raise ValueError("[Reviewer] No paper_draft in state.")

    paper_titles = [p["title"] for p in paper_list["papers"]]
    actual_metrics = result_summary.get("metrics", [])
    domain = input_topic.get("domain", "")
    topic = input_topic.get("topic", "")

    print("[Reviewer] Reviewing paper...")

    prompt = "Review this paper as a strict IEEE journal reviewer.\n\n"
    prompt += "Domain: " + domain + "\n"
    prompt += "Topic: " + topic + "\n\n"
    prompt += "ABSTRACT:\n" + paper_draft.get("abstract", "")[:400] + "\n\n"
    prompt += "INTRODUCTION:\n" + paper_draft.get("introduction", "")[:400] + "\n\n"
    prompt += "METHODOLOGY:\n" + paper_draft.get("methodology", "")[:400] + "\n\n"
    prompt += "RESULTS:\n" + paper_draft.get("results", "")[:600] + "\n\n"
    prompt += "LIMITATIONS:\n" + paper_draft.get("limitations", "")[:400] + "\n\n"
    prompt += "Allowed citations: " + json.dumps(paper_titles[:12]) + "\n"
    prompt += "Actual metrics: " + json.dumps(actual_metrics) + "\n\n"
    prompt += "Check:\n"
    prompt += "1. Citations not in allowed list - critical\n"
    prompt += "2. Numbers not matching actual metrics - critical\n"
    prompt += "3. Citations irrelevant to " + domain + " - major\n"
    prompt += "4. Topic drift from " + topic + " - major\n"
    prompt += "5. Unsupported claims - major\n"
    prompt += "6. Repetitive content - minor\n"
    prompt += "7. Weak future work - minor\n"
    prompt += "8. Incomplete results table - minor\n\n"
    prompt += "Return ONLY this JSON:\n"
    prompt += "{\n"
    prompt += '  "issues": [{"severity": "minor", "section": "results", "description": "issue"}],\n'
    prompt += '  "overall_verdict": "accept",\n'
    prompt += '  "required_changes": ["change1"],\n'
    prompt += '  "citation_audit_passed": true,\n'
    prompt += '  "number_audit_passed": true,\n'
    prompt += '  "domain_relevance_passed": true,\n'
    prompt += '  "hallucination_flags": [],\n'
    prompt += '  "quality_score": 8.5,\n'
    prompt += '  "strengths": ["strength1", "strength2"],\n'
    prompt += '  "recommended_additional_citations": ["topic"]\n'
    prompt += "}"

    raw = call_with_retry(
        client,
        messages=[
            {"role": "system", "content": "Strict IEEE reviewer for " + domain + ". Return ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1500,
        temperature=0.2,
        agent_name="reviewer"
    )

    used = log_usage("reviewer", prompt, raw)
    print_status("reviewer", used)

    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    result = json.loads(raw)
    state_store.update_state("reviewer_feedback", result)
    audit_logger.log("reviewer", {"sections": list(paper_draft.keys())}, result)

    print("[Reviewer] Verdict: " + result["overall_verdict"])
    print("[Reviewer] Quality score: " + str(result.get("quality_score", "N/A")))
    for issue in result.get("issues", []):
        print("  [" + issue["severity"].upper() + "] " + issue["section"] + ": " + issue["description"])
    return result