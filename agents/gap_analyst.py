import json
from groq import Groq
from memory import state_store, audit_logger
import config


def run():
    client = Groq(api_key=config.GROQ_API_KEY)
    paper_list = state_store.get_state("paper_list")
    input_topic = state_store.get_state("input_topic")
    if not paper_list:
        raise ValueError("[GapAnalyst] No paper_list in state.")

    papers = paper_list["papers"]
    topic = input_topic.get("topic", "")
    domain = input_topic.get("domain", "")

    print("[GapAnalyst] Analyzing " + str(len(papers)) + " papers for gaps in: " + topic)

    prompt = "Analyze these papers and find research gaps STRICTLY related to:\n"
    prompt += "Topic: " + topic + "\n"
    prompt += "Domain: " + domain + "\n\n"
    prompt += "RULES:\n"
    prompt += "- Only identify gaps directly related to the topic above\n"
    prompt += "- Do NOT identify gaps in unrelated fields\n"
    prompt += "- Every gap must be traceable to at least one paper below\n"
    prompt += "- Score impact and feasibility honestly\n\n"
    prompt += "Papers:\n" + json.dumps(papers, indent=2) + "\n\n"
    prompt += "Return a JSON object:\n"
    prompt += "{\n"
    prompt += '  "gaps": [\n'
    prompt += '    {\n'
    prompt += '      "gap_id": "gap_1",\n'
    prompt += '      "description": "specific gap directly related to ' + topic + '",\n'
    prompt += '      "supporting_paper_titles": ["title1"],\n'
    prompt += '      "impact_score": 8.5,\n'
    prompt += '      "feasibility_score": 7.0\n'
    prompt += '    }\n'
    prompt += '  ],\n'
    prompt += '  "top_gap_id": "gap_1",\n'
    prompt += '  "rationale": "why this gap matters for ' + topic + '"\n'
    prompt += "}\n"
    prompt += "Identify 3-5 gaps. Return ONLY the JSON object."

    response = client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": (
                "You are a research analyst specializing in " + domain + ". "
                "Only identify gaps directly related to the given topic. "
                "Return ONLY valid JSON."
            )},
            {"role": "user", "content": prompt}
        ],
        max_tokens=config.MAX_TOKENS,
        temperature=0.3
    )

    raw = response.choices[0].message.content.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    result = json.loads(raw)
    state_store.update_state("gap_analysis", result)
    audit_logger.log("gap_analyst", {"papers": len(papers)}, result)
    print("[GapAnalyst] Found " + str(len(result["gaps"])) + " gaps.")
    return result