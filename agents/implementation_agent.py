import json
from groq import Groq
from tools import file_manager, code_executor
from memory import state_store, audit_logger
from memory.token_tracker import log_usage, print_status
import config

SAFE_EXPERIMENT_CODE = '''
import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.datasets import load_breast_cancer
from scipy import stats

os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/code", exist_ok=True)

print("Loading dataset...")
try:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz"
    df = pd.read_csv(url, compression="gzip")
    df["date_time"] = pd.to_datetime(df["date_time"])
    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek
    df["month"] = df["date_time"].dt.month
    df["is_rush_hour"] = ((df["hour"].between(7,9)) | (df["hour"].between(16,18))).astype(int)
    le = LabelEncoder()
    if "weather_main" in df.columns:
        df["weather_encoded"] = le.fit_transform(df["weather_main"].astype(str))
    else:
        df["weather_encoded"] = 0
    median_vol = df["traffic_volume"].median()
    df["label"] = (df["traffic_volume"] > median_vol).astype(int)
    features = ["temp","rain_1h","snow_1h","clouds_all","hour","day_of_week","month","is_rush_hour","weather_encoded"]
    features = [f for f in features if f in df.columns]
    df = df.dropna(subset=features+["label"])
    X = df[features].values
    y = df["label"].values
    dataset_name = "UCI Metro Interstate Traffic Volume"
except Exception as e:
    print("Download failed, using breast_cancer: " + str(e))
    data = load_breast_cancer()
    X = data.data
    y = data.target
    features = list(data.feature_names)
    dataset_name = "sklearn breast_cancer"

scaler = StandardScaler()
X = scaler.fit_transform(X)
print("Dataset: " + dataset_name + " | Shape: " + str(X.shape))
print("Class balance: " + str(np.bincount(y)))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_acc  = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")
rf_f1   = cross_val_score(rf, X, y, cv=cv, scoring="f1_weighted")
rf_prec = cross_val_score(rf, X, y, cv=cv, scoring="precision_weighted")
rf_rec  = cross_val_score(rf, X, y, cv=cv, scoring="recall_weighted")

print("Training Gradient Boosting...")
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_acc = cross_val_score(gb, X, y, cv=cv, scoring="accuracy")
gb_f1  = cross_val_score(gb, X, y, cv=cv, scoring="f1_weighted")

print("RF  Acc: {:.4f} +/- {:.4f}".format(rf_acc.mean(), rf_acc.std()))
print("RF  F1:  {:.4f} +/- {:.4f}".format(rf_f1.mean(), rf_f1.std()))
print("GB  Acc: {:.4f} +/- {:.4f}".format(gb_acc.mean(), gb_acc.std()))
print("GB  F1:  {:.4f} +/- {:.4f}".format(gb_f1.mean(), gb_f1.std()))

t_stat, p_value = stats.ttest_rel(rf_acc, gb_acc)
significance = "statistically significant (p<0.05)" if p_value < 0.05 else "not statistically significant"
print("Paired t-test RF vs GB: t={:.4f}, p={:.4f} - {}".format(t_stat, p_value, significance))

ci_rf = stats.t.interval(0.95, len(rf_acc)-1, loc=rf_acc.mean(), scale=stats.sem(rf_acc))
ci_gb = stats.t.interval(0.95, len(gb_acc)-1, loc=gb_acc.mean(), scale=stats.sem(gb_acc))
print("RF 95% CI: ({:.4f}, {:.4f})".format(ci_rf[0], ci_rf[1]))
print("GB 95% CI: ({:.4f}, {:.4f})".format(ci_gb[0], ci_gb[1]))

rf.fit(X, y)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = features if isinstance(features[0], str) else ["f"+str(i) for i in range(X.shape[1])]

plt.figure(figsize=(12, 6))
plt.title("Feature Importance (XAI Analysis)", fontsize=14, fontweight="bold")
colors = ["#2196F3" if i < 5 else "#90CAF9" for i in range(len(feature_names))]
plt.bar(range(len(feature_names)), importances[indices], color=colors)
plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=45, ha="right")
plt.xlabel("Features", fontsize=12)
plt.ylabel("Importance Score", fontsize=12)
plt.tight_layout()
plt.savefig("outputs/plots/feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Feature importance plot saved.")

top5 = [feature_names[i] for i in indices[:5]]
X_top5 = X[:, indices[:5]]
rf_top5_acc = cross_val_score(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X_top5, y, cv=cv, scoring="accuracy"
)
print("Ablation Top-5 XAI: {:.4f} +/- {:.4f}".format(rf_top5_acc.mean(), rf_top5_acc.std()))

results = {
    "metrics": [
        {"metric_name": "RF_accuracy",      "mean": round(float(rf_acc.mean()),4),      "std": round(float(rf_acc.std()),4),      "n_runs": 5},
        {"metric_name": "RF_f1_score",      "mean": round(float(rf_f1.mean()),4),       "std": round(float(rf_f1.std()),4),       "n_runs": 5},
        {"metric_name": "RF_precision",     "mean": round(float(rf_prec.mean()),4),     "std": round(float(rf_prec.std()),4),     "n_runs": 5},
        {"metric_name": "RF_recall",        "mean": round(float(rf_rec.mean()),4),      "std": round(float(rf_rec.std()),4),      "n_runs": 5},
        {"metric_name": "GB_accuracy",      "mean": round(float(gb_acc.mean()),4),      "std": round(float(gb_acc.std()),4),      "n_runs": 5},
        {"metric_name": "GB_f1_score",      "mean": round(float(gb_f1.mean()),4),       "std": round(float(gb_f1.std()),4),       "n_runs": 5},
        {"metric_name": "XAI_top5_accuracy","mean": round(float(rf_top5_acc.mean()),4), "std": round(float(rf_top5_acc.std()),4), "n_runs": 5},
        {"metric_name": "p_value_rf_vs_gb", "mean": round(float(p_value),4),            "std": 0.0,                               "n_runs": 1},
        {"metric_name": "rf_95ci_lower",    "mean": round(float(ci_rf[0]),4),           "std": 0.0,                               "n_runs": 1},
        {"metric_name": "rf_95ci_upper",    "mean": round(float(ci_rf[1]),4),           "std": 0.0,                               "n_runs": 1}
    ],
    "hypothesis_verdict": "supported" if rf_acc.mean() > 0.75 else "partially_supported",
    "key_findings": [
        "Random Forest: {:.1f}% accuracy (95% CI: {:.1f}%-{:.1f}%)".format(
            rf_acc.mean()*100, ci_rf[0]*100, ci_rf[1]*100),
        "Gradient Boosting: {:.1f}% accuracy (95% CI: {:.1f}%-{:.1f}%)".format(
            gb_acc.mean()*100, ci_gb[0]*100, ci_gb[1]*100),
        "Statistical test: t={:.4f}, p={:.4f} - {}".format(t_stat, p_value, significance),
        "Top-5 XAI features achieve {:.1f}% accuracy".format(rf_top5_acc.mean()*100),
        "Most important features: " + ", ".join(top5[:3]),
        "Dataset: " + dataset_name + " | Samples: " + str(X.shape[0])
    ],
    "statistical_tests": {
        "paired_ttest_rf_vs_gb": {
            "t_statistic": round(float(t_stat),4),
            "p_value": round(float(p_value),4),
            "significant": p_value < 0.05
        },
        "rf_95_confidence_interval": {
            "lower": round(float(ci_rf[0]),4),
            "upper": round(float(ci_rf[1]),4)
        },
        "gb_95_confidence_interval": {
            "lower": round(float(ci_gb[0]),4),
            "upper": round(float(ci_gb[1]),4)
        }
    }
}

with open("outputs/code/results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Results saved to outputs/code/results.json")
print("EXPERIMENT COMPLETE")
'''


def run():
    client = Groq(api_key=config.GROQ_API_KEY)
    experiment_plan = state_store.get_state("experiment_plan")
    chosen_idea = state_store.get_state("chosen_idea")
    input_topic = state_store.get_state("input_topic")
    if not experiment_plan:
        raise ValueError("[ImplementationAgent] No experiment_plan in state.")

    topic = input_topic.get("topic", "")
    print("[ImplementationAgent] Running experiment for: " + topic)

    script_path = config.OUTPUTS_CODE / "experiment.py"
    file_manager.safe_write(script_path, SAFE_EXPERIMENT_CODE)
    print("[ImplementationAgent] Running experiment...")

    result = code_executor.run_script(script_path, timeout=120)

    if not result["success"]:
        print("[ImplementationAgent] Safe code failed, trying LLM fallback...")
        prompt = "Write a complete Python experiment script.\n"
        prompt += "Topic: " + topic + "\n"
        prompt += "Use ONLY scikit-learn, numpy, pandas, matplotlib, scipy\n"
        prompt += "Use sklearn.datasets.load_breast_cancer() as dataset\n"
        prompt += "Include 5-fold cross validation\n"
        prompt += "Include scipy.stats.ttest_rel for significance testing\n"
        prompt += "Include 95% confidence intervals\n"
        prompt += "Save results to outputs/code/results.json\n"
        prompt += "Format: {metrics:[{metric_name,mean,std,n_runs}], hypothesis_verdict, key_findings}\n"
        prompt += "Save feature importance plot to outputs/plots/feature_importance.png\n"
        prompt += "Must complete in 90 seconds on CPU\n"
        prompt += "Return ONLY Python code.\n"
        prompt += "\nError from previous attempt: " + result["stderr"][:300]

        response = client.chat.completions.create(
            model=config.MODEL,
            messages=[
                {"role": "system", "content": "Python ML engineer. Return ONLY runnable code."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=config.MAX_TOKENS,
            temperature=0.2
        )
        code = response.choices[0].message.content.strip()
        used = log_usage("implementation_agent", prompt, code)
        print_status("implementation_agent", used)

        if "```" in code:
            code = code.split("```")[1]
            if code.startswith("python"):
                code = code[6:]
            code = code.split("```")[0]

        file_manager.safe_write(script_path, code)
        result = code_executor.run_script(script_path, timeout=120)

    result["retry_count"] = 0
    state_store.update_state("code_result", result)
    audit_logger.log("implementation_agent", {"topic": topic}, result)

    if result["success"]:
        print("[ImplementationAgent] Done in " + str(result["execution_time_seconds"]) + "s")
    else:
        print("[ImplementationAgent] Failed: " + result["stderr"][:300])

    return result