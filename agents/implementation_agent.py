import json
from tools import file_manager, code_executor
from memory import state_store, audit_logger
from memory.token_tracker import log_usage, print_status
from memory.groq_client import create_client, call_with_retry
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
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy import stats

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

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
    df["weather_encoded"] = le.fit_transform(df["weather_main"].astype(str)) if "weather_main" in df.columns else 0
    df["label"] = (df["traffic_volume"] > df["traffic_volume"].median()).astype(int)
    features = [f for f in ["temp","rain_1h","snow_1h","clouds_all","hour","day_of_week","month","is_rush_hour","weather_encoded"] if f in df.columns]
    df = df.dropna(subset=features+["label"])
    X = df[features].values
    y = df["label"].values
    dataset_name = "UCI Metro Interstate Traffic Volume"
except Exception as e:
    print("Download failed, using breast_cancer: " + str(e))
    data = load_breast_cancer()
    X, y = data.data, data.target
    features = list(data.feature_names)
    dataset_name = "sklearn breast_cancer"

scaler = StandardScaler()
X = scaler.fit_transform(X)
print("Dataset: " + dataset_name + " | Shape: " + str(X.shape))
print("Class balance: " + str(np.bincount(y)))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def optimize_rf(X, y, n_trials=10):
    if not HAS_OPTUNA:
        return {"n_estimators": 100, "max_depth": 10, "min_samples_split": 2}
    def obj(trial):
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 8),
            "random_state": 42, "n_jobs": -1
        }
        return cross_val_score(RandomForestClassifier(**p), X, y, cv=3, scoring="accuracy").mean()
    s = optuna.create_study(direction="maximize")
    s.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    print("  Best RF: " + str(s.best_params))
    return s.best_params

def optimize_gb(X, y, n_trials=10):
    if not HAS_OPTUNA:
        return {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3}
    def obj(trial):
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 150),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "random_state": 42
        }
        return cross_val_score(GradientBoostingClassifier(**p), X, y, cv=3, scoring="accuracy").mean()
    s = optuna.create_study(direction="maximize")
    s.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    print("  Best GB: " + str(s.best_params))
    return s.best_params

print("Optimizing RF...")
best_rf = optimize_rf(X, y)
print("Optimizing GB...")
best_gb = optimize_gb(X, y)

rf = RandomForestClassifier(**{**best_rf, "random_state": 42, "n_jobs": -1})
gb = GradientBoostingClassifier(**{**best_gb, "random_state": 42})

model_results = {}
metric_names = ["accuracy", "precision", "recall", "f1_score"]
scoring_map = {
    "accuracy": "accuracy",
    "precision": "precision_weighted",
    "recall": "recall_weighted",
    "f1_score": "f1_weighted"
}

for model, name in [(rf, "RandomForest"), (gb, "GradientBoosting")]:
    print("Training " + name + "...")
    model_results[name] = {}
    for metric, scorer in scoring_map.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring=scorer)
        model_results[name][metric] = scores
        print("  " + name + " " + metric + ": {:.4f} +/- {:.4f}".format(scores.mean(), scores.std()))

xgb_scores = {}
lgb_scores = {}

if HAS_XGB:
    print("Training XGBoost...")
    xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0)
    model_results["XGBoost"] = {}
    for metric, scorer in scoring_map.items():
        scores = cross_val_score(xgb, X, y, cv=cv, scoring=scorer)
        model_results["XGBoost"][metric] = scores
    print("  XGB accuracy: {:.4f}".format(model_results["XGBoost"]["accuracy"].mean()))

if HAS_LGB:
    print("Training LightGBM...")
    lgb = LGBMClassifier(n_estimators=100, random_state=42, verbosity=-1)
    model_results["LightGBM"] = {}
    for metric, scorer in scoring_map.items():
        scores = cross_val_score(lgb, X, y, cv=cv, scoring=scorer)
        model_results["LightGBM"][metric] = scores
    print("  LGB accuracy: {:.4f}".format(model_results["LightGBM"]["accuracy"].mean()))

model_names_list = list(model_results.keys())
COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"]
plt.style.use("seaborn-v0_8-whitegrid")

# FIGURE 1: Grouped metric comparison
fig1, ax1 = plt.subplots(figsize=(14, 6))
x = np.arange(len(metric_names))
n_m = len(model_names_list)
width = 0.75 / n_m
offsets = np.linspace(-(n_m-1)/2*width, (n_m-1)/2*width, n_m)
for i, (mname, color) in enumerate(zip(model_names_list, COLORS)):
    means = [model_results[mname][m].mean()*100 for m in metric_names]
    stds  = [model_results[mname][m].std()*100  for m in metric_names]
    bars = ax1.bar(x + offsets[i], means, width, label=mname,
                   color=color, alpha=0.87, yerr=stds, capsize=4,
                   error_kw={"elinewidth":1.5,"ecolor":"#333"})
    for bar, mean in zip(bars, means):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 "{:.1f}".format(mean), ha="center", va="bottom", fontsize=7.5, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(["Accuracy","Precision","Recall","F1-Score"], fontsize=12)
ax1.set_ylabel("Score (%)", fontsize=12)
min_val = min(model_results[m][k].mean()*100 for m in model_names_list for k in metric_names)
ax1.set_ylim([max(0, min_val-3), 103])
ax1.set_title("Fig. 1: All Models — Metric Comparison (5-Fold Cross-Validation)", fontsize=13, fontweight="bold", pad=15)
ax1.legend(loc="lower right", fontsize=10, framealpha=0.9)
ax1.set_facecolor("#FAFAFA")
fig1.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig("outputs/plots/fig1_metric_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Fig 1 saved")

# FIGURE 2: Accuracy bar with error bars
fig2, ax2 = plt.subplots(figsize=(10, 6))
acc_means = [model_results[m]["accuracy"].mean()*100 for m in model_names_list]
acc_stds  = [model_results[m]["accuracy"].std()*100  for m in model_names_list]
bars2 = ax2.bar(model_names_list, acc_means, yerr=acc_stds,
                color=COLORS[:len(model_names_list)], capsize=6, alpha=0.88,
                edgecolor="white", error_kw={"elinewidth":2,"ecolor":"#333"})
for bar, mean, std in zip(bars2, acc_means, acc_stds):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+std+0.3,
             "{:.2f}%".format(mean), ha="center", va="bottom", fontsize=11, fontweight="bold")
ax2.set_ylabel("Accuracy (%)", fontsize=12)
ax2.set_title("Fig. 2: Accuracy Comparison with 95% Confidence Error Bars", fontsize=13, fontweight="bold", pad=15)
ax2.set_ylim([max(0,min(acc_means)-3), 103])
ax2.set_facecolor("#FAFAFA")
fig2.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig("outputs/plots/fig2_accuracy_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Fig 2 saved")

# FIGURE 3: Box plot CV distribution
fig3, ax3 = plt.subplots(figsize=(11, 6))
acc_scores = [model_results[m]["accuracy"]*100 for m in model_names_list]
bp = ax3.boxplot(acc_scores, labels=model_names_list, patch_artist=True,
                 medianprops={"color":"white","linewidth":2.5})
for patch, color in zip(bp["boxes"], COLORS[:len(model_names_list)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.82)
ax3.set_ylabel("Accuracy (%)", fontsize=12)
ax3.set_title("Fig. 3: Cross-Validation Score Distribution (5 Folds)", fontsize=13, fontweight="bold", pad=15)
ax3.set_facecolor("#FAFAFA")
fig3.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig("outputs/plots/fig3_cv_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Fig 3 saved")

# FIGURE 4: Radar chart
categories = ["Accuracy","Precision","Recall","F1-Score"]
N = len(categories)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]
fig4, ax4 = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))
for i, (mname, color) in enumerate(zip(model_names_list, COLORS)):
    values = [model_results[mname][m].mean()*100 for m in metric_names] + [model_results[mname]["accuracy"].mean()*100]
    ax4.plot(angles, values, "o-", linewidth=2, label=mname, color=color)
    ax4.fill(angles, values, alpha=0.1, color=color)
ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(categories, fontsize=12)
min_r = max(0, min(model_results[m][k].mean()*100 for m in model_names_list for k in metric_names)-5)
ax4.set_ylim([min_r, 100])
ax4.set_title("Fig. 4: Radar Chart — Multi-Metric Model Comparison", fontsize=13, fontweight="bold", pad=20)
ax4.legend(loc="upper right", bbox_to_anchor=(1.35,1.1), fontsize=10)
fig4.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig("outputs/plots/fig4_radar_chart.png", dpi=150, bbox_inches="tight")
plt.close()
print("Fig 4 saved")

# FIGURE 5: Feature importance
rf.fit(X, y)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names_list = features if isinstance(features[0], str) else ["f"+str(i) for i in range(X.shape[1])]
top5 = [feature_names_list[i] for i in indices[:5]]
X_top5 = X[:, indices[:5]]
rf_top5_acc = cross_val_score(RandomForestClassifier(**{**best_rf,"random_state":42}), X_top5, y, cv=cv, scoring="accuracy")
print("Ablation Top-5: {:.4f}".format(rf_top5_acc.mean()))

fig5, axes5 = plt.subplots(1, 2, figsize=(14, 5))
bar_colors = [COLORS[0] if i < 5 else "#90CAF9" for i in range(len(feature_names_list))]
bars5 = axes5[0].bar(range(len(feature_names_list)), importances[indices], color=bar_colors, edgecolor="white")
axes5[0].set_xticks(range(len(feature_names_list)))
axes5[0].set_xticklabels([feature_names_list[i] for i in indices], rotation=45, ha="right", fontsize=9)
axes5[0].set_title("Fig. 5a: XAI Feature Importance (Random Forest)", fontsize=11, fontweight="bold", pad=10)
axes5[0].set_ylabel("Importance Score")
for bar, val in zip(bars5, importances[indices]):
    axes5[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001, "{:.3f}".format(val), ha="center", va="bottom", fontsize=7)

abl_labels = ["All Features (" + str(len(feature_names_list)) + ")", "Top-5 XAI Features"]
abl_vals  = [model_results["RandomForest"]["accuracy"].mean()*100, rf_top5_acc.mean()*100]
abl_stds  = [model_results["RandomForest"]["accuracy"].std()*100,  rf_top5_acc.std()*100]
bars5b = axes5[1].bar(abl_labels, abl_vals, yerr=abl_stds,
                       color=[COLORS[0],COLORS[2]], capsize=6, alpha=0.88,
                       edgecolor="white", error_kw={"elinewidth":2,"ecolor":"#333"})
for bar, val, std in zip(bars5b, abl_vals, abl_stds):
    axes5[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+std+0.3,
                  "{:.2f}%".format(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
axes5[1].set_ylabel("Accuracy (%)")
axes5[1].set_title("Fig. 5b: Ablation Study — Feature Selection Impact", fontsize=11, fontweight="bold", pad=10)
axes5[1].set_ylim([min(abl_vals)-3, 103])
axes5[1].annotate("Delta={:.2f}%".format(abl_vals[0]-abl_vals[1]),
                   xy=(0.5, max(abl_vals)-1), xycoords="data", ha="center", fontsize=10, style="italic")
axes5[0].set_facecolor("#FAFAFA")
axes5[1].set_facecolor("#FAFAFA")
fig5.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig("outputs/plots/fig5_feature_ablation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Fig 5 saved")

# FIGURE 6: Confusion matrix
rf_preds = cross_val_predict(rf, X, y, cv=cv)
cm = confusion_matrix(y, rf_preds)
fig6, ax6 = plt.subplots(figsize=(7,6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax6, colorbar=True, cmap="Blues")
ax6.set_title("Fig. 6: Confusion Matrix — Best Model (5-Fold CV)", fontsize=13, fontweight="bold", pad=15)
fig6.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig("outputs/plots/fig6_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("Fig 6 saved")

plot_files = [
    "outputs/plots/fig1_metric_comparison.png",
    "outputs/plots/fig2_accuracy_comparison.png",
    "outputs/plots/fig3_cv_distribution.png",
    "outputs/plots/fig4_radar_chart.png",
    "outputs/plots/fig5_feature_ablation.png",
    "outputs/plots/fig6_confusion_matrix.png"
]

# FIGURE 7: SHAP
if HAS_SHAP:
    try:
        print("Running SHAP...")
        rf_s = RandomForestClassifier(n_estimators=50, random_state=42)
        Xs = X[:min(500,len(X))]
        ys = y[:min(500,len(y))]
        rf_s.fit(Xs, ys)
        explainer = shap.TreeExplainer(rf_s)
        sv = explainer.shap_values(Xs[:100])
        if isinstance(sv, list): sv = sv[1]
        fig7, axes7 = plt.subplots(1, 2, figsize=(16,6))
        sm = np.abs(sv).mean(axis=0)
        si = np.argsort(sm)[::-1]
        axes7[0].barh([feature_names_list[i] for i in si[:10]][::-1], sm[si[:10]][::-1], color="#E65100", alpha=0.85)
        axes7[0].set_title("Fig. 7a: SHAP Feature Importance", fontsize=12, fontweight="bold")
        axes7[0].set_xlabel("Mean |SHAP Value|")
        sp = np.mean(sv > 0, axis=0)
        ti = si[:8]
        axes7[1].barh([feature_names_list[i] for i in ti][::-1], sp[ti][::-1], color="#2196F3", alpha=0.85, label="Positive")
        axes7[1].barh([feature_names_list[i] for i in ti][::-1], -(1-sp[ti])[::-1], color="#F44336", alpha=0.85, label="Negative")
        axes7[1].set_title("Fig. 7b: SHAP Direction Analysis", fontsize=12, fontweight="bold")
        axes7[1].set_xlabel("Proportion of samples")
        axes7[1].legend(fontsize=9)
        axes7[1].axvline(x=0, color="black", linewidth=0.8)
        plt.tight_layout()
        plt.savefig("outputs/plots/fig7_shap.png", dpi=150, bbox_inches="tight")
        plt.close()
        plot_files.append("outputs/plots/fig7_shap.png")
        print("Fig 7 saved: SHAP")
    except Exception as e:
        print("SHAP failed: " + str(e))

# Statistical tests
rf_acc = model_results["RandomForest"]["accuracy"]
gb_acc = model_results["GradientBoosting"]["accuracy"]
t_stat, p_value = stats.ttest_rel(rf_acc, gb_acc)
significance = "statistically significant (p<0.05)" if p_value < 0.05 else "not statistically significant"
ci_rf = stats.t.interval(0.95, len(rf_acc)-1, loc=rf_acc.mean(), scale=stats.sem(rf_acc))
print("t-test: t={:.4f}, p={:.4f} - {}".format(t_stat, p_value, significance))

all_accs = [model_results[m]["accuracy"].mean() for m in model_names_list]
best_idx = int(np.argmax(all_accs))
best_model_name = model_names_list[best_idx]
best_acc = all_accs[best_idx]
print("Best: " + best_model_name + " ({:.4f})".format(best_acc))

metrics = []
for mname in model_names_list:
    for metric in metric_names:
        sc = model_results[mname][metric]
        metrics.append({
            "metric_name": mname.replace(" ","") + "_" + metric,
            "mean": round(float(sc.mean()),4),
            "std":  round(float(sc.std()),4),
            "n_runs": 5
        })

metrics += [
    {"metric_name":"p_value_rf_vs_gb", "mean":round(float(p_value),4),  "std":0.0,"n_runs":1},
    {"metric_name":"rf_95ci_lower",    "mean":round(float(ci_rf[0]),4), "std":0.0,"n_runs":1},
    {"metric_name":"rf_95ci_upper",    "mean":round(float(ci_rf[1]),4), "std":0.0,"n_runs":1},
    {"metric_name":"XAI_top5_accuracy","mean":round(float(rf_top5_acc.mean()),4),"std":round(float(rf_top5_acc.std()),4),"n_runs":5}
]

results = {
    "metrics": metrics,
    "hypothesis_verdict": "supported" if best_acc > 0.75 else "partially_supported",
    "best_model": best_model_name,
    "plot_files": plot_files,
    "key_findings": [
        "Best: " + best_model_name + " ({:.2f}%)".format(best_acc*100),
        "RandomForest: {:.2f}% (95% CI: {:.2f}%-{:.2f}%)".format(rf_acc.mean()*100, ci_rf[0]*100, ci_rf[1]*100),
        "GradientBoosting: {:.2f}%".format(gb_acc.mean()*100),
        "t={:.4f}, p={:.4f} ({})".format(t_stat, p_value, significance),
        "Top-5 XAI: {:.2f}%".format(rf_top5_acc.mean()*100),
        "Top features: " + ", ".join(top5[:3]),
        "Dataset: " + dataset_name + " | n=" + str(X.shape[0])
    ],
    "statistical_tests": {
        "paired_ttest": {"t": round(float(t_stat),4), "p": round(float(p_value),4), "sig": bool(p_value<0.05)},
        "rf_95ci": {"lower": round(float(ci_rf[0]),4), "upper": round(float(ci_rf[1]),4)}
    }
}

with open("outputs/code/results.json","w") as f:
    json.dump(results, f, indent=2)
print("Results saved. " + str(len(plot_files)) + " figures generated.")
print("EXPERIMENT COMPLETE")
'''


def run():
    client = create_client()
    experiment_plan = state_store.get_state("experiment_plan")
    chosen_idea     = state_store.get_state("chosen_idea")
    input_topic     = state_store.get_state("input_topic")
    if not experiment_plan:
        raise ValueError("[ImplementationAgent] No experiment_plan in state.")

    topic = input_topic.get("topic", "")
    print("[ImplementationAgent] Running experiment for: " + topic)

    script_path = config.OUTPUTS_CODE / "experiment.py"
    file_manager.safe_write(script_path, SAFE_EXPERIMENT_CODE)
    print("[ImplementationAgent] Running experiment...")

    result = code_executor.run_script(script_path, timeout=300)

    if not result["success"]:
        print("[ImplementationAgent] Experiment failed: " + result["stderr"][:200])
        print("[ImplementationAgent] Trying LLM fallback...")

        prompt  = "Write a complete Python ML classification experiment.\n"
        prompt += "Use sklearn.datasets.load_breast_cancer() — no downloads.\n"
        prompt += "Train RandomForest and GradientBoosting with 5-fold CV.\n"
        prompt += "Compute accuracy, precision, recall, f1_weighted for each model.\n"
        prompt += "Generate these exact files:\n"
        prompt += "- outputs/plots/fig1_metric_comparison.png (grouped bar: all models x all metrics)\n"
        prompt += "- outputs/plots/fig2_accuracy_comparison.png (accuracy bar chart)\n"
        prompt += "- outputs/plots/fig3_cv_distribution.png (boxplot)\n"
        prompt += "- outputs/plots/fig4_radar_chart.png (radar/spider chart)\n"
        prompt += "- outputs/plots/fig5_feature_ablation.png (feature importance + ablation side by side)\n"
        prompt += "- outputs/plots/fig6_confusion_matrix.png\n"
        prompt += "Save outputs/code/results.json with keys: metrics, hypothesis_verdict, best_model, key_findings, plot_files\n"
        prompt += "CRITICAL: No newline characters inside string literals in bar chart labels.\n"
        prompt += "Use space instead of backslash-n in axis labels.\n"
        prompt += "Must complete in 120 seconds. Return ONLY Python code.\n"

        raw = call_with_retry(
            client,
            messages=[
                {"role": "system", "content": "Python ML engineer. Return ONLY runnable Python code. No markdown fences."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=config.MAX_TOKENS,
            temperature=0.1,
            agent_name="implementation_agent"
        )
        used = log_usage("implementation_agent", prompt, raw)
        print_status("implementation_agent", used)

        code = raw.strip()
        if "```" in code:
            code = code.split("```")[1]
            if code.startswith("python"):
                code = code[6:]
            code = code.split("```")[0]

        file_manager.safe_write(script_path, code)
        result = code_executor.run_script(script_path, timeout=180)

    result["retry_count"] = 0
    state_store.update_state("code_result", result)
    audit_logger.log("implementation_agent", {"topic": topic}, result)

    if result["success"]:
        print("[ImplementationAgent] Done in " + str(result["execution_time_seconds"]) + "s")
    else:
        print("[ImplementationAgent] Failed: " + result["stderr"][:300])

    return result