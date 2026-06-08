import json
from memory import state_store, audit_logger
from memory.token_tracker import log_usage, print_status
from memory.groq_client import create_client, call_with_retry
import config


def run():
    client = create_client()
    paper_list = state_store.get_state("paper_list")
    input_topic = state_store.get_state("input_topic")
    if not paper_list:
        raise ValueError("[GapAnalyst] No paper_list in state.")

    papers = paper_list["papers"]
    topic = input_topic.get("topic", "")
    domain = input_topic.get("domain", "")

    trimmed = [
        {
            "title": p.get("title", ""),
            "year": p.get("year", 0),
            "abstract_summary": p.get("abstract_summary", p.get("abstract", ""))[:150],
            "stated_limitations": p.get("stated_limitations", [])[:2]
        }
        for p in papers[:10]
    ]

    print("[GapAnalyst] Analyzing " + str(len(trimmed)) + " papers for gaps in: " + topic)

    prompt = "Find research gaps STRICTLY related to: " + topic + "\nDomain: " + domain + "\n\n"
    prompt += "Papers:\n" + json.dumps(trimmed, indent=1) + "\n\n"
    prompt += "Return JSON:\n"
    prompt += '{"gaps": [{"gap_id": "gap_1", "description": "...", "supporting_paper_titles": ["title"], "impact_score": 8.5, "feasibility_score": 7.0}], "top_gap_id": "gap_1", "rationale": "..."}\n'
    prompt += "3-5 gaps only. ONLY gaps related to " + topic + ". Return ONLY JSON."

    raw = call_with_retry(
        client,
        messages=[
            {"role": "system", "content": "Research analyst for " + domain + ". Stay on topic. Return ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1500,
        temperature=0.3,
        agent_name="gap_analyst"
    )

    used = log_usage("gap_analyst", prompt, raw)
    print_status("gap_analyst", used)

    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    result = json.loads(raw)
    state_store.update_state("gap_analysis", result)
    audit_logger.log("gap_analyst", {"papers": len(papers)}, result)
    print("[GapAnalyst] Found " + str(len(result["gaps"])) + " gaps.")
    return result