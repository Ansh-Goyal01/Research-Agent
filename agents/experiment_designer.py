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

    prompt = "Design a lightweight reproducible ML experiment.\n"
    prompt += "Research hypothesis: " + idea["hypothesis"] + "\n\n"
    prompt += "HARD CONSTRAINTS (never violate these):\n"
    prompt += "- CPU only, no GPU\n"
    prompt += "- Max 8GB RAM\n"
    prompt += "- Runtime under 1 hour\n"
    prompt += "- Libraries: scikit-learn, numpy, pandas, matplotlib ONLY\n"
    prompt += "- Dataset must be downloadable via a direct URL\n"
    prompt += "- Use UCI Metro Interstate Traffic Volume dataset\n"
    prompt += "- Dataset URL: https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz\n"
    prompt += "- Baselines: RandomForestClassifier and GradientBoostingClassifier from sklearn\n"
    prompt += "- Task: classify traffic congestion level (high/low based on traffic_volume median)\n"
    prompt += "- Metrics: accuracy, f1_score, precision, recall\n\n"
    prompt += "Return this exact JSON:\n"
    prompt += "{\n"
    prompt += '  "hypothesis_null": "XAI features do not improve traffic congestion classification",\n'
    prompt += '  "hypothesis_alternative": "XAI feature selection improves classification accuracy",\n'
    prompt += '  "independent_variables": ["traffic_volume", "temp", "hour", "day_of_week"],\n'
    prompt += '  "dependent_variables": ["congestion_level"],\n'
    prompt += '  "baselines": [\n'
    prompt += '    {"name": "Random Forest", "citation": "Breiman 2001", "reason_for_inclusion": "strong ensemble baseline"},\n'
    prompt += '    {"name": "Gradient Boosting", "citation": "Friedman 2001", "reason_for_inclusion": "boosting baseline"}\n'
    prompt += '  ],\n'
    prompt += '  "datasets": [\n'
    prompt += '    {"name": "UCI Metro Interstate Traffic Volume", "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz", "size": "48MB", "license": "CC BY 4.0"}\n'
    prompt += '  ],\n'
    prompt += '  "metrics": [\n'
    prompt += '    {"name": "accuracy", "formula": "correct/total", "higher_is_better": true},\n'
    prompt += '    {"name": "f1_score", "formula": "2*precision*recall/(precision+recall)", "higher_is_better": true}\n'
    prompt += '  ],\n'
    prompt += '  "statistical_tests": ["paired t-test", "Wilcoxon signed-rank test"],\n'
    prompt += '  "ablation_design": "Compare model with all features vs model with only top-5 XAI-selected features",\n'
    prompt += '  "compute_requirements": "CPU only, 16GB RAM, scikit-learn only",\n'
    prompt += '  "estimated_runtime_hours": 0.5\n'
    prompt += "}\n"
    prompt += "Return ONLY this JSON object, nothing else."

    response = client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": "You are an experiment designer. Return ONLY valid JSON. Never suggest GPU or deep learning."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=config.MAX_TOKENS,
        temperature=0.1
    )

    raw = response.choices[0].message.content.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    result = json.loads(raw)
    result["estimated_runtime_hours"] = min(result.get("estimated_runtime_hours", 0.5), 1.0)
    result["compute_requirements"] = "CPU only, 16GB RAM, scikit-learn only"

    state_store.update_state("experiment_plan", result)
    audit_logger.log("experiment_designer", {"idea": idea["idea_id"]}, result)
    print("[ExperimentDesigner] Done. Runtime: " + str(result["estimated_runtime_hours"]) + "h")
    return result