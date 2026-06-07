import json
from groq import Groq
from memory import state_store, audit_logger
import config


def run():
    client = Groq(api_key=config.GROQ_API_KEY)
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

    prompt = "Review this research paper as a strict IEEE journal reviewer.\n\n"
    prompt += "Research domain: " + domain + "\n"
    prompt += "Research topic: " + topic + "\n\n"
    prompt += "ABSTRACT:\n" + paper_draft.get("abstract", "")[:500] + "\n\n"
    prompt += "INTRODUCTION (first 500 chars):\n" + paper_draft.get("introduction", "")[:500] + "\n\n"
    prompt += "METHODOLOGY (first 500 chars):\n" + paper_draft.get("methodology", "")[:500] + "\n\n"
    prompt += "RESULTS:\n" + paper_draft.get("results", "")[:800] + "\n\n"
    prompt += "LIMITATIONS:\n" + paper_draft.get("limitations", "")[:500] + "\n\n"
    prompt += "VERIFICATION DATA:\n"
    prompt += "Allowed citations: " + json.dumps(paper_titles[:15]) + "\n"
    prompt += "Actual metrics: " + json.dumps(actual_metrics) + "\n\n"
    prompt += "Check ALL of these:\n"
    prompt += "1. Citations not in allowed list - flag as critical\n"
    prompt += "2. Numbers not matching actual metrics - flag as critical\n"
    prompt += "3. Citations irrelevant to domain '" + domain + "' - flag as major\n"
    prompt += "4. Topic drift away from '" + topic + "' - flag as major\n"
    prompt += "5. Unsupported claims - flag as major\n"
    prompt += "6. Repetitive content across sections - flag as minor\n"
    prompt += "7. Missing or weak future work - flag as minor\n"
    prompt += "8. Results table completeness - flag missing values as minor\n\n"
    prompt += "Return this exact JSON:\n"
    prompt += "{\n"
    prompt += '  "issues": [\n'
    prompt += '    {"severity": "minor", "section": "results", "description": "specific issue"}\n'
    prompt += '  ],\n'
    prompt += '  "overall_verdict": "accept",\n'
    prompt += '  "required_changes": ["specific actionable change"],\n'
    prompt += '  "citation_audit_passed": true,\n'
    prompt += '  "number_audit_passed": true,\n'
    prompt += '  "domain_relevance_passed": true,\n'
    prompt += '  "hallucination_flags": [],\n'
    prompt += '  "quality_score": 8.5,\n'
    prompt += '  "strengths": ["strength1", "strength2"],\n'
    prompt += '  "recommended_additional_citations": ["topic to search for more refs"]\n'
    prompt += "}\n"
    prompt += "Return ONLY the JSON object."

    response = client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": (
                "You are a strict IEEE journal reviewer specializing in " + domain + ". "
                "Flag any content that drifts from the research topic. "
                "Return ONLY valid JSON."
            )},
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
    print("[Reviewer] Domain relevance: " + str(result.get("domain_relevance_passed", "N/A")))

    if result.get("strengths"):
        print("[Reviewer] Strengths:")
        for s in result["strengths"]:
            print("  + " + s)

    for issue in result.get("issues", []):
        print("  [" + issue["severity"].upper() + "] " + issue["section"] + ": " + issue["description"])

    return result