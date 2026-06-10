"""
experiment_designer.py  — Research Agent
Modality-aware experiment design.
Hardware: Intel i5-13200H (12-core), 16GB DDR5, no GPU. Budget: 60 min.

Fix: was hardcoding RF/GB/XGBoost/LightGBM baselines for EVERY topic.
Now dispatches model choices based on detected modality from dataset_finder.
"""
import json
from tools.dataset_finder import find_best_dataset, detect_modality
from memory import state_store, audit_logger
from memory.groq_client import create_client, call_with_retry
import config

# ── Modality → model choices ──────────────────────────────────────────────────

MODALITY_MODELS = {
    "text": [
        {"name": "TF-IDF + Logistic Regression", "citation": "Manning 2008",
         "reason_for_inclusion": "strong linear baseline for text classification"},
        {"name": "TF-IDF + Linear SVC",          "citation": "Cortes 1995",
         "reason_for_inclusion": "SVM is a standard text classification baseline"},
        {"name": "TF-IDF + XGBoost",             "citation": "Chen 2016",
         "reason_for_inclusion": "gradient boosting on TF-IDF features"},
        {"name": "TF-IDF + LightGBM",            "citation": "Ke 2017",
         "reason_for_inclusion": "efficient boosting on sparse text features"},
    ],
    "timeseries": [
        {"name": "Isolation Forest",             "citation": "Liu 2008",
         "reason_for_inclusion": "standard unsupervised anomaly detection baseline"},
        {"name": "XGBoost + handcrafted features", "citation": "Chen 2016",
         "reason_for_inclusion": "strong supervised baseline with RMS/kurtosis features"},
        {"name": "LightGBM + rolling features",  "citation": "Ke 2017",
         "reason_for_inclusion": "efficient boosting with temporal features"},
        {"name": "Random Forest + features",     "citation": "Breiman 2001",
         "reason_for_inclusion": "ensemble baseline for structured sensor data"},
    ],
    "image": [
        {"name": "HOG + SVM",                    "citation": "Dalal 2005",
         "reason_for_inclusion": "classical hand-crafted feature baseline"},
        {"name": "HOG + Random Forest",          "citation": "Breiman 2001",
         "reason_for_inclusion": "ensemble on handcrafted features"},
        {"name": "CNN (3-layer, from scratch)",  "citation": "LeCun 1998",
         "reason_for_inclusion": "standard small CNN for image classification"},
        {"name": "ResNet-18 transfer learning",  "citation": "He 2016",
         "reason_for_inclusion": "pretrained deep model fine-tuned CPU-feasible"},
    ],
    "audio": [
        {"name": "MFCC + SVM",                   "citation": "Cortes 1995",
         "reason_for_inclusion": "classical audio feature + classifier baseline"},
        {"name": "MFCC + Random Forest",         "citation": "Breiman 2001",
         "reason_for_inclusion": "ensemble on MFCC features"},
        {"name": "MFCC + XGBoost",               "citation": "Chen 2016",
         "reason_for_inclusion": "boosting on spectral features"},
        {"name": "CNN on Mel-spectrogram",       "citation": "Hershey 2017",
         "reason_for_inclusion": "deep learning audio classification"},
    ],
    "tabular": [
        {"name": "Random Forest",                "citation": "Breiman 2001",
         "reason_for_inclusion": "strong ensemble baseline for tabular data"},
        {"name": "Gradient Boosting",            "citation": "Friedman 2001",
         "reason_for_inclusion": "boosting baseline for structured data"},
        {"name": "XGBoost",                      "citation": "Chen 2016",
         "reason_for_inclusion": "gradient boosted trees, state-of-the-art on tabular"},
        {"name": "LightGBM",                     "citation": "Ke 2017",
         "reason_for_inclusion": "efficient boosting, fast on large tabular datasets"},
    ],
}

MODALITY_METRICS = {
    "text":       ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"],
    "timeseries": ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted",
                   "roc_auc"],
    "image":      ["accuracy", "f1_weighted", "top1_error"],
    "audio":      ["accuracy", "f1_weighted", "precision_weighted"],
    "tabular":    ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"],
}

HARDWARE_NOTE = (
    "Intel i5-13200H (12 cores, 16 threads), 16GB DDR5 RAM, "
    "no GPU, Python 3.11, scikit-learn + xgboost + lightgbm. "
    "Max wall-clock: 60 minutes total."
)


def run():
    client      = create_client()
    idea        = state_store.get_state("chosen_idea")
    input_topic = state_store.get_state("input_topic")
    if not idea:
        raise ValueError("[ExperimentDesigner] No chosen_idea in state.")

    topic  = input_topic.get("topic", "")
    domain = input_topic.get("domain", "")

    # 1. Detect modality BEFORE asking dataset_finder
    modality = detect_modality(topic, domain)
    print("[ExperimentDesigner] Modality: " + modality)

    # 2. Get best dataset for this modality
    dataset  = find_best_dataset(topic, domain)
    baselines = MODALITY_MODELS.get(modality, MODALITY_MODELS["tabular"])
    metrics   = MODALITY_METRICS.get(modality, MODALITY_METRICS["tabular"])

    print("[ExperimentDesigner] Designing experiment for: " + topic)
    print("[ExperimentDesigner] Dataset: " + dataset["name"])
    print("[ExperimentDesigner] Baselines: " + str([b["name"] for b in baselines]))

    prompt = (
        "Design a lightweight reproducible ML experiment for an IEEE paper.\n\n"
        "Research topic: " + topic + "\n"
        "Domain: " + domain + "\n"
        "Hypothesis: " + idea["hypothesis"] + "\n"
        "Detected data modality: " + modality + "\n\n"
        "HARD CONSTRAINTS:\n"
        "- " + HARDWARE_NOTE + "\n"
        "- Runtime under 60 minutes total\n"
        "- Use ONLY the baselines listed below — do NOT add or substitute others\n"
        "- Dataset: " + dataset["name"] + "\n"
        "- Dataset URL: " + dataset["url"] + "\n\n"
        "BASELINES (use exactly these — chosen for modality '" + modality + "'):\n"
        + json.dumps(baselines, indent=2) + "\n\n"
        "IMPORTANT: Do NOT use Bayesian Neural Networks or Monte Carlo dropout.\n"
        "For uncertainty estimation use: conformal prediction or prediction variance.\n\n"
        "Return this EXACT JSON:\n"
        "{\n"
        '  "hypothesis_null": "null hypothesis",\n'
        '  "hypothesis_alternative": "' + idea["hypothesis"] + '",\n'
        '  "independent_variables": ["var1"],\n'
        '  "dependent_variables": ["metric1"],\n'
        '  "baselines": ' + json.dumps(baselines) + ",\n"
        '  "datasets": [{"name": "' + dataset["name"] + '", '
        '"url": "' + dataset["url"] + '", '
        '"size": "' + dataset["size"] + '", '
        '"license": "' + dataset["license"] + '"}],\n'
        '  "metrics": ' + json.dumps([{"name": m, "higher_is_better": True} for m in metrics]) + ",\n"
        '  "statistical_tests": ["paired t-test", "Wilcoxon signed-rank test"],\n'
        '  "ablation_design": "compare full features vs top-5 XAI selected features",\n'
        '  "compute_requirements": "' + HARDWARE_NOTE + '",\n'
        '  "estimated_runtime_hours": 0.5,\n'
        '  "modality": "' + modality + '"\n'
        "}\n"
        "Return ONLY the JSON object."
    )

    raw = call_with_retry(
        client,
        messages=[
            {"role": "system", "content": (
                "Experiment designer for " + domain + ". "
                "Never suggest GPU. Never add baselines not in the provided list. "
                "Return ONLY valid JSON."
            )},
            {"role": "user", "content": prompt}
        ],
        max_tokens=config.MAX_TOKENS,
        temperature=0.1,
        agent_name="experiment_designer"
    )

    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.split("```")[0]

    result = json.loads(raw)
    result["estimated_runtime_hours"] = min(result.get("estimated_runtime_hours", 0.5), 1.0)
    result["compute_requirements"]    = HARDWARE_NOTE
    result["modality"]                = modality
    result["_dataset_meta"]           = dataset

    state_store.update_state("experiment_plan", result)
    audit_logger.log("experiment_designer", {"idea": idea["idea_id"]}, result)
    print("[ExperimentDesigner] Done. Modality=" + modality +
          "  Runtime: " + str(result["estimated_runtime_hours"]) + "h")
    return result
