import json
import os
from groq import Groq
from memory import state_store, audit_logger
import config

SYSTEM_PROMPT = """You are a rigorous results analyst for machine learning research.
Rules:
- Only report numbers that exist in the actual output logs.
- Do NOT fabricate or estimate any metric values.
- If a metric is missing, explicitly say so.
- Be honest about whether the hypothesis was supported.
- Return ONLY valid JSON. No markdown, no explanation."""

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

    prompt = f"""Analyze these experiment results:

Hypothesis being tested: {chosen_idea['hypothesis']}

Actual results from experiment output:
{json.dumps(actual_results, indent=2)}

Stdout from experiment:
{code_result.get('stdout', '')[:2000]}

Experiment plan metrics:
{json.dumps(experiment_plan.get('metrics', []), indent=2)}

Return a JSON object with this exact structure:
{{
  "metrics": [
    {{"metric_name": "name", "mean": 0.85, "std": 0.02, "n_runs": 5}}
  ],
  "hypothesis_verdict": "supported",
  "key_findings": ["finding1", "finding2"],
  "anomalies": ["any unusual observations"],
  "suggested_ablation": "one follow-up experiment suggestion"
}}

Only include metrics that appear in the actual results. Return ONLY the JSON object."""

    response = client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        max_tokens=config.MAX_TOKENS,
        temperature=0.2
    )

    raw_output = response.choices[0].message.content.strip()
    if "```" in raw_output:
        raw_output = raw_output.split("```")[1]
        if raw_output.startswith("json"):
            raw_output = raw_output[4:]

    result = json.loads(raw_output)

    # The verdict is not the analyst's to author. The significance gate already
    # computed it from the reported numbers by a pre-registered rule; overwrite
    # whatever the LLM narrated with that trusted value so the paper reports the
    # verdict the data licenses, not the one the model preferred. The LLM still
    # supplies the prose (key findings, anomalies) -- just not the conclusion.
    verification_verdict = state_store.get_state("verification_verdict")
    if verification_verdict:
        result["hypothesis_verdict"] = verification_verdict["verdict"]
        result["verdict_evidence"] = verification_verdict["evidence"]

    state_store.update_state("result_summary", result)
    audit_logger.log("results_analyst", {"code_success": code_result["success"]}, result)
    print(f"[ResultsAnalyst] Verdict: {result['hypothesis_verdict']}")
    print(f"[ResultsAnalyst] Key findings: {len(result['key_findings'])}")
    return result