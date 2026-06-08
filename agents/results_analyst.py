import json
from memory import state_store, audit_logger
from memory.token_tracker import log_usage, print_status
from memory.groq_client import create_client, call_with_retry
import config


def run():
    client = create_client()
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

    prompt = "Analyze these experiment results:\n"
    prompt += "Hypothesis: " + chosen_idea["hypothesis"] + "\n\n"
    prompt += "Actual results:\n" + json.dumps(actual_results, indent=2) + "\n\n"
    prompt += "Stdout:\n" + code_result.get("stdout", "")[:1000] + "\n\n"
    prompt += "Return a JSON object:\n"
    prompt += "{\n"
    prompt += '  "metrics": [{"metric_name": "name", "mean": 0.0, "std": 0.0, "n_runs": 5}],\n'
    prompt += '  "hypothesis_verdict": "supported",\n'
    prompt += '  "best_model": "model name",\n'
    prompt += '  "key_findings": ["finding1", "finding2"],\n'
    prompt += '  "anomalies": [],\n'
    prompt += '  "suggested_ablation": "suggestion"\n'
    prompt += "}\n"
    prompt += "Only include metrics from actual results. Return ONLY the JSON object."

    raw = call_with_retry(
        client,
        messages=[
            {"role": "system", "content": "Results analyst. Only report real numbers. Return ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=config.MAX_TOKENS,
        temperature=0.2,
        agent_name="results_analyst"
    )

    used = log_usage("results_analyst", prompt, raw)
    print_status("results_analyst", used)

    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    result = json.loads(raw)
    if not result.get("best_model") and actual_results.get("best_model"):
        result["best_model"] = actual_results["best_model"]

    state_store.update_state("result_summary", result)
    audit_logger.log("results_analyst", {"success": code_result["success"]}, result)
    print("[ResultsAnalyst] Verdict: " + result["hypothesis_verdict"])
    return result