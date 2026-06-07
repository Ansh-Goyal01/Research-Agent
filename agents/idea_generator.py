import json
from groq import Groq
from memory import state_store, audit_logger
import config


def run():
    client = Groq(api_key=config.GROQ_API_KEY)
    gap_analysis = state_store.get_state("gap_analysis")
    paper_list = state_store.get_state("paper_list")
    input_topic = state_store.get_state("input_topic")
    if not gap_analysis:
        raise ValueError("[IdeaGenerator] No gap_analysis in state.")

    domain = input_topic.get("domain", "")
    topic = input_topic.get("topic", "")
    venue = input_topic.get("target_venue", "IEEE")

    print("[IdeaGenerator] Generating ideas...")

    prompt = "Generate 3 research ideas STRICTLY within this domain:\n"
    prompt += "Topic: " + topic + "\n"
    prompt += "Domain: " + domain + "\n"
    prompt += "Target venue: " + venue + "\n\n"
    prompt += "STRICT RULES:\n"
    prompt += "- Every idea must be directly about: " + topic + "\n"
    prompt += "- Do NOT drift to unrelated topics\n"
    prompt += "- Hypotheses must be specific and falsifiable\n"
    prompt += "- Compute cost must be low or medium (CPU only)\n"
    prompt += "- Each idea must be testable with scikit-learn on a laptop\n\n"
    prompt += "Gap analysis to base ideas on:\n"
    prompt += json.dumps(gap_analysis, indent=2) + "\n\n"
    prompt += "Return a JSON object:\n"
    prompt += "{\n"
    prompt += '  "ideas": [\n'
    prompt += '    {\n'
    prompt += '      "idea_id": "idea_1",\n'
    prompt += '      "hypothesis": "one sentence falsifiable hypothesis directly about ' + topic + '",\n'
    prompt += '      "novelty_explanation": "what is new compared to existing work",\n'
    prompt += '      "minimum_viable_experiment": "simplest sklearn experiment to test this",\n'
    prompt += '      "compute_cost": "low",\n'
    prompt += '      "time_estimate": "1-2 days",\n'
    prompt += '      "novelty_score": 8.0,\n'
    prompt += '      "feasibility_score": 9.0,\n'
    prompt += '      "impact_score": 8.5\n'
    prompt += '    }\n'
    prompt += '  ],\n'
    prompt += '  "recommended_idea_id": "idea_1"\n'
    prompt += "}\n"
    prompt += "Return ONLY the JSON object."

    response = client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": (
                "You are a research idea generator. "
                "You must stay strictly within the given topic and domain. "
                "Never suggest ideas outside the specified research area. "
                "Return ONLY valid JSON."
            )},
            {"role": "user", "content": prompt}
        ],
        max_tokens=config.MAX_TOKENS,
        temperature=0.5
    )

    raw = response.choices[0].message.content.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    result = json.loads(raw)
    state_store.update_state("idea_candidates", result)
    audit_logger.log("idea_generator", {"top_gap": gap_analysis["top_gap_id"]}, result)
    print("[IdeaGenerator] Generated " + str(len(result["ideas"])) + " ideas.")
    return result