import json
from groq import Groq
from memory import state_store, audit_logger
import config

SYSTEM_PROMPT = """You are a methodical experiment designer for machine learning research.
Rules:
- Every dataset must be real, public, and verifiable.
- Every baseline must be a real published method.
- Keep compute requirements suitable for a laptop with no GPU.
- Return ONLY valid JSON. No markdown, no explanation."""

def run():
    client = Groq(api_key=config.GROQ_API_KEY)
    idea = state_store.get_state("chosen_idea")
    paper_list = state_store.get_state("paper_list")
    if not idea:
        raise ValueError("[ExperimentDesigner] No chosen_idea in state.")

    print("[ExperimentDesigner] Designing experiment plan...")

    prompt = f"""Design a complete experiment plan for this research idea:

Hypothesis: {idea['hypothesis']}
Novelty: {idea['novelty_explanation']}
Experiment description: {idea['minimum_viable_experiment']}
Compute cost target: {idea['compute_cost']}

Available papers for baseline references:
{json.dumps([p['title'] for p in paper_list['papers'][:10]], indent=2)}

Return a JSON object with this exact structure:
{{
  "hypothesis_null": "null hypothesis statement",
  "hypothesis_alternative": "alternative hypothesis statement",
  "independent_variables": ["variable1", "variable2"],
  "dependent_variables": ["metric1", "metric2"],
  "baselines": [
    {{"name": "method name", "citation": "paper title", "reason_for_inclusion": "why"}}
  ],
  "datasets": [
    {{"name": "dataset name", "url": "real url", "size": "approximate size", "license": "license type"}}
  ],
  "metrics": [
    {{"name": "metric name", "formula": "formula or description", "higher_is_better": true}}
  ],
  "statistical_tests": ["test1", "test2"],
  "ablation_design": "description of ablation study",
  "compute_requirements": "CPU only, 8GB RAM, estimated X hours",
  "estimated_runtime_hours": 2.0
}}

Use only real publicly available datasets. Return ONLY the JSON object."""

    response = client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        max_tokens=config.MAX_TOKENS,
        temperature=0.3
    )

    raw_output = response.choices[0].message.content.strip()
    if "```" in raw_output:
        raw_output = raw_output.split("```")[1]
        if raw_output.startswith("json"):
            raw_output = raw_output[4:]

    result = json.loads(raw_output)
    state_store.update_state("experiment_plan", result)
    audit_logger.log("experiment_designer", {"idea_id": idea.get("idea_id")}, result)
    print(f"[ExperimentDesigner] Plan ready. Estimated runtime: {result['estimated_runtime_hours']} hours")
    return result