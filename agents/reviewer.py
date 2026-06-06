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
    actual_metrics = result_summary["metrics"]
    print("[Reviewer] Reviewing paper...")
    prompt = (
        "Review this paper draft:\n\n"
        "ABSTRACT:\n" + paper_draft.get("abstract","") + "\n\n"
        "RESULTS:\n" + paper_draft.get("results","") + "\n\n"
        "LIMITATIONS:\n" + paper_draft.get("limitations","") + "\n\n"
        "Allowed citations: " + json.dumps(paper_titles) + "\n"
        "Actual metrics: " + json.dumps(actual_metrics) + "\n\n"
        "Return a JSON object:\n"
        "{\n"
        "  \"issues\": [{\"severity\": \"minor\", \"section\": \"results\", \"description\": \"issue\"}],\n"
        "  \"overall_verdict\": \"accept\",\n"
        "  \"required_changes\": [\"change1\"],\n"
        "  \"citation_audit_passed\": true,\n"
        "  \"number_audit_passed\": true,\n"
        "  \"hallucination_flags\": []\n"
        "}\n"
        "Return ONLY the JSON object."
    )
    response = client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": "You are a peer reviewer. Return ONLY valid JSON."},
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
    return result
