import json
from groq import Groq
from memory import state_store, audit_logger
import config


def run():
    client = Groq(api_key=config.GROQ_API_KEY)
    paper_draft = state_store.get_state("paper_draft")
    paper_list = state_store.get_state("paper_list")
    result_summary = state_store.get_state("result_summary")

    if not paper_draft:
        raise ValueError("[Reviewer] No paper_draft in state.")

    paper_titles = [p["title"] for p in paper_list["papers"]]
    actual_metrics = result_summary.get("metrics", [])

    print("[Reviewer] Reviewing paper...")

    prompt = "Review this research paper as a strict IEEE journal reviewer.\n\n"
    prompt += "ABSTRACT:\n" + paper_draft.get("abstract", "")[:500] + "\n\n"
    prompt += "INTRODUCTION (first 500 chars):\n" + paper_draft.get("introduction", "")[:500] + "\n\n"
    prompt += "METHODOLOGY (first 500 chars):\n" + paper_draft.get("methodology", "")[:500] + "\n\n"
    prompt += "RESULTS:\n" + paper_draft.get("results", "")[:800] + "\n\n"
    prompt += "LIMITATIONS:\n" + paper_draft.get("limitations", "")[:500] + "\n\n"
    prompt += "VERIFICATION DATA:\n"
    prompt += "Allowed citations: " + json.dumps(paper_titles[:15]) + "\n"
    prompt += "Actual metrics: " + json.dumps(actual_metrics) + "\n\n"
    prompt += "Check for:\n"
    prompt += "1. Citations not in allowed list\n"
    prompt += "2. Numbers not matching actual metrics\n"
    prompt += "3. Unsupported claims\n"
    prompt += "4. Missing sections or weak content\n"
    prompt += "5. Hallucinated facts\n\n"
    prompt += "Return this exact JSON:\n"
    prompt += "{\n"
    prompt += '  "issues": [\n'
    prompt += '    {"severity": "minor", "section": "results", "description": "issue description"}\n'
    prompt += '  ],\n'
    prompt += '  "overall_verdict": "accept",\n'
    prompt += '  "required_changes": ["specific change 1", "specific change 2"],\n'
    prompt += '  "citation_audit_passed": true,\n'
    prompt += '  "number_audit_passed": true,\n'
    prompt += '  "hallucination_flags": [],\n'
    prompt += '  "quality_score": 8.5\n'
    prompt += "}\n"
    prompt += "Return ONLY the JSON object."

    response = client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": "You are a strict IEEE journal reviewer. Return ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=config.MAX_TOKENS,
        temperature=0.2
    )

    raw = response.choices[0].message.content.strip()
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