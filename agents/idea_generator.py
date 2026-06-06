import json
from groq import Groq
from memory import state_store, audit_logger
import config

def run():
    client = Groq(api_key=config.GROQ_API_KEY)
    gap_analysis = state_store.get_state("gap_analysis")
    paper_list = state_store.get_state("paper_list")
    if not gap_analysis:
        raise ValueError("[IdeaGenerator] No gap_analysis in state.")
    print("[IdeaGenerator] Generating ideas...")
    prompt = (
        "Based on this gap analysis:\n" +
        json.dumps(gap_analysis, indent=2) +
        "\n\nGenerate 3 research ideas. Return a JSON object:\n"
        "{\n"
        "  \"ideas\": [\n"
        "    {\n"
        "      \"idea_id\": \"idea_1\",\n"
        "      \"hypothesis\": \"one sentence hypothesis\",\n"
        "      \"novelty_explanation\": \"what is new\",\n"
        "      \"minimum_viable_experiment\": \"simplest test\",\n"
        "      \"compute_cost\": \"low\",\n"
        "      \"time_estimate\": \"1-2 weeks\",\n"
        "      \"novelty_score\": 8.0,\n"
        "      \"feasibility_score\": 9.0,\n"
        "      \"impact_score\": 8.5\n"
        "    }\n"
        "  ],\n"
        "  \"recommended_idea_id\": \"idea_1\"\n"
        "}\n"
        "Return ONLY the JSON object."
    )
    response = client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": "You are a research idea generator. Return ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=config.MAX_TOKENS,
        temperature=0.7
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
