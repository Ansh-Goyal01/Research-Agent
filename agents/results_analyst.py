import json
from groq import Groq
from memory import state_store, audit_logger
import config

def run():
    client = Groq(api_key=config.GROQ_API_KEY)
    code_result = state_store.get_state("code_result")
    experiment_plan = state_store.get_state("experiment_plan")
    chosen_idea = state_store.get_state("chosen_idea")
    if not code_result:
        raise ValueError("[ResultsAnalyst] No code_result in state.")
    results_file = config.OUTPUTS_CODE / "results.json"
    actual_results = {}
    if results_file.exists():
        with open(results_file, "r") as f:
            actual_results = json.load(f)
    print("[ResultsAnalyst] Analyzing results...")
    prompt = (
        "Analyze these experiment results:\n"
        "Hypothesis: " + chosen_idea["hypothesis"] + "\n\n"
        "Actual results:\n" + json.dumps(actual_results, indent=2) + "\n\n"
        "Stdout:\n" + code_result.get("stdout", "")[:1000] + "\n\n"
        "Return a JSON object:\n"
        "{\n"
        "  \"metrics\": [{\"metric_name\":\"\", \"mean\":0.0, \"std\":0.0, \"n_runs\":1}],\n"
        "  \"hypothesis_verdict\": \"supported\",\n"
        "  \"key_findings\": [\"finding1\"],\n"
        "  \"anomalies\": [],\n"
        "  \"suggested_ablation\": \"suggestion\"\n"
        "}\n"
        "Only include metrics from actual results. Return ONLY the JSON object."
    )
    response = client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": "You are a results analyst. Return ONLY valid JSON."},
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
    state_store.update_state("result_summary", result)
    audit_logger.log("results_analyst", {"success": code_result["success"]}, result)
    print("[ResultsAnalyst] Verdict: " + result["hypothesis_verdict"])
    return result
