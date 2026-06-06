import json
from groq import Groq
from memory import state_store, audit_logger
import config

def run():
    client = Groq(api_key=config.GROQ_API_KEY)
    idea = state_store.get_state("chosen_idea")
    if not idea:
        raise ValueError("[ExperimentDesigner] No chosen_idea in state.")
    print("[ExperimentDesigner] Designing experiment...")
    prompt = (
        "Design an experiment for:\n"
        "Hypothesis: " + idea["hypothesis"] + "\n"
        "Compute cost: " + idea["compute_cost"] + "\n\n"
        "Return a JSON object:\n"
        "{\n"
        "  \"hypothesis_null\": \"null hypothesis\",\n"
        "  \"hypothesis_alternative\": \"alternative hypothesis\",\n"
        "  \"independent_variables\": [\"var1\"],\n"
        "  \"dependent_variables\": [\"metric1\"],\n"
        "  \"baselines\": [{\"name\": \"method\", \"citation\": \"paper\", \"reason_for_inclusion\": \"why\"}],\n"
        "  \"datasets\": [{\"name\": \"dataset\", \"url\": \"url\", \"size\": \"size\", \"license\": \"license\"}],\n"
        "  \"metrics\": [{\"name\": \"accuracy\", \"formula\": \"correct/total\", \"higher_is_better\": true}],\n"
        "  \"statistical_tests\": [\"t-test\"],\n"
        "  \"ablation_design\": \"description\",\n"
        "  \"compute_requirements\": \"CPU only, 8GB RAM\",\n"
        "  \"estimated_runtime_hours\": 1.0\n"
        "}\n"
        "Use only real public datasets. Return ONLY the JSON object."
    )
    response = client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": "You are an experiment designer. Return ONLY valid JSON."},
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
    state_store.update_state("experiment_plan", result)
    audit_logger.log("experiment_designer", {"idea": idea["idea_id"]}, result)
    print("[ExperimentDesigner] Done. Runtime: " + str(result["estimated_runtime_hours"]) + "h")
    return result
