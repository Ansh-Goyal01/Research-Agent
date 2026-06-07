import json
from groq import Groq
from tools.dataset_finder import find_best_dataset
from memory import state_store, audit_logger
import config


def run():
    client = Groq(api_key=config.GROQ_API_KEY)
    idea = state_store.get_state("chosen_idea")
    input_topic = state_store.get_state("input_topic")
    if not idea:
        raise ValueError("[ExperimentDesigner] No chosen_idea in state.")

    topic = input_topic.get("topic", "")
    domain = input_topic.get("domain", "")

    dataset = find_best_dataset(topic, domain)

    print("[ExperimentDesigner] Designing experiment for: " + topic)
    print("[ExperimentDesigner] Using dataset: " + dataset["name"] + " from " + dataset["source"])

    prompt = "Design a lightweight reproducible ML experiment.\n\n"
    prompt += "Research topic: " + topic + "\n"
    prompt += "Domain: " + domain + "\n"
    prompt += "Hypothesis: " + idea["hypothesis"] + "\n\n"
    prompt += "HARD CONSTRAINTS:\n"
    prompt += "- CPU only, no GPU\n"
    prompt += "- Max 8GB RAM\n"
    prompt += "- Runtime under 1 hour\n"
    prompt += "- Libraries: scikit-learn, numpy, pandas, matplotlib ONLY\n"
    prompt += "- Dataset to use: " + dataset["name"] + "\n"
    prompt += "- Dataset source: " + dataset["source"] + "\n"
    prompt += "- Dataset URL: " + dataset["url"] + "\n\n"
    prompt += "Design baselines and metrics appropriate for this domain: " + domain + "\n\n"
    prompt += "Return this exact JSON:\n"
    prompt += "{\n"
    prompt += '  "hypothesis_null": "null hypothesis",\n'
    prompt += '  "hypothesis_alternative": "' + idea["hypothesis"] + '",\n'
    prompt += '  "independent_variables": ["var1", "var2"],\n'
    prompt += '  "dependent_variables": ["metric1"],\n'
    prompt += '  "baselines": [\n'
    prompt += '    {"name": "Random Forest", "citation": "Breiman 2001", "reason_for_inclusion": "strong ensemble baseline"},\n'
    prompt += '    {"name": "Gradient Boosting", "citation": "Friedman 2001", "reason_for_inclusion": "boosting baseline"}\n'
    prompt += '  ],\n'
    prompt += '  "datasets": [\n'
    prompt += '    {"name": "' + dataset["name"] + '", "url": "' + dataset["url"] + '", "size": "' + dataset["size"] + '", "license": "' + dataset["license"] + '"}\n'
    prompt += '  ],\n'
    prompt += '  "metrics": [\n'
    prompt += '    {"name": "accuracy", "formula": "correct/total", "higher_is_better": true},\n'
    prompt += '    {"name": "f1_score", "formula": "2*P*R/(P+R)", "higher_is_better": true}\n'
    prompt += '  ],\n'
    prompt += '  "statistical_tests": ["paired t-test", "Wilcoxon signed-rank test"],\n'
    prompt += '  "ablation_design": "compare model with all features vs top XAI-selected features",\n'
    prompt += '  "compute_requirements": "CPU only, 16GB RAM, scikit-learn only",\n'
    prompt += '  "estimated_runtime_hours": 0.30\n'
    prompt += "}\n"
    prompt += "Return ONLY the JSON object."

    response = client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": (
                "You are an experiment designer for " + domain + ". "
                "Never suggest GPU or deep learning. "
                "Return ONLY valid JSON."
            )},
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
    result["estimated_runtime_hours"] = min(result.get("estimated_runtime_hours", 0.30), 1.0)
    result["compute_requirements"] = "CPU only, 16GB RAM, scikit-learn only"

    state_store.update_state("experiment_plan", result)
    audit_logger.log("experiment_designer", {"idea": idea["idea_id"]}, result)
    print("[ExperimentDesigner] Done. Dataset: " + dataset["name"] + " | Runtime: " + str(result["estimated_runtime_hours"]) + "h")
    return result