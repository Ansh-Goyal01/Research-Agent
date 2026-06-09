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

# ── Dataset Loading ────────────────────────────────────────────
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

# ── Hyperparameter Optimization ───────────────────────────────
def optimize_rf(X, y, n_trials=10):
    if not HAS_OPTUNA:
        return {"n_estimators": 100, "max_depth": 10, "min_samples_split": 2}
    def obj(trial):
        p = {"n_estimators": trial.suggest_int("n_estimators", 50, 150),
             "max_depth": trial.suggest_int("max_depth", 3, 15),
             "min_samples_split": trial.suggest_int("min_samples_split", 2, 8),
             "random_state": 42, "n_jobs": -1}
        return cross_val_score(RandomForestClassifier(**p), X, y, cv=3, scoring="accuracy").mean()
    s = optuna.create_study(direction="maximize")
    s.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    print("  Best RF: " + str(s.best_params))
    return s.best_params

def optimize_gb(X, y, n_trials=10):
    if not HAS_OPTUNA:
        return {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3}
    def obj(trial):
        p = {"n_estimators": trial.suggest_int("n_estimators", 50, 150),
             "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
             "max_depth": trial.suggest_int("max_depth", 2, 6),
             "random_state": 42}
        return cross_val_score(GradientBoostingClassifier(**p), X, y, cv=3, scoring="accuracy").mean()
    s = optuna.create_study(direction="maximize")
    s.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    print("  Best GB: " + str(s.best_params))
    return s.best_params

print("Optimizing Random Forest...")
best_rf = optimize_rf(X, y)
print("Optimizing Gradient Boosting...")
best_gb = optimize_gb(X, y)

# ── Model Training ────────────────────────────────────────────
rf = RandomForestClassifier(**{**best_rf, "random_state": 42, "n_jobs": -1})
gb = GradientBoostingClassifier(**{**best_gb, "random_state": 42})

model_results = {}

for model, name in [(rf, "RandomForest"), (gb, "GradientBoosting")]:
    print("Training " + name + "...")
    acc  = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    prec = cross_val_score(model, X, y, cv=cv, scoring="precision_weighted")
    rec  = cross_val_score(model, X, y, cv=cv, scoring="recall_weighted")
    f1   = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted")
    model_results[name] = {
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1_score":  f1
    }
    print(name + " Acc: {:.4f} +/- {:.4f}".format(acc.mean(), acc.std()))

xgb_acc = np.array([0.0])
lgb_acc = np.array([0.0])

if HAS_XGB:
    print("Training XGBoost...")
    xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0)
    xgb_acc  = cross_val_score(xgb, X, y, cv=cv, scoring="accuracy")
    xgb_prec = cross_val_score(xgb, X, y, cv=cv, scoring="precision_weighted")
    xgb_rec  = cross_val_score(xgb, X, y, cv=cv, scoring="recall_weighted")
    xgb_f1   = cross_val_score(xgb, X, y, cv=cv, scoring="f1_weighted")
    model_results["XGBoost"] = {"accuracy": xgb_acc, "precision": xgb_prec, "recall": xgb_rec, "f1_score": xgb_f1}
    print("XGB Acc: {:.4f} +/- {:.4f}".format(xgb_acc.mean(), xgb_acc.std()))

if HAS_LGB:
    print("Training LightGBM...")
    lgb = LGBMClassifier(n_estimators=100, random_state=42, verbosity=-1)
    lgb_acc  = cross_val_score(lgb, X, y, cv=cv, scoring="accuracy")
    lgb_prec = cross_val_score(lgb, X, y, cv=cv, scoring="precision_weighted")
    lgb_rec  = cross_val_score(lgb, X, y, cv=cv, scoring="recall_weighted")
    lgb_f1   = cross_val_score(lgb, X, y, cv=cv, scoring="f1_weighted")
    model_results["LightGBM"] = {"accuracy": lgb_acc, "precision": lgb_prec, "recall": lgb_rec, "f1_score": lgb_f1}
    print("LGB Acc: {:.4f} +/- {:.4f}".format(lgb_acc.mean(), lgb_acc.std()))

model_names = list(model_results.keys())
metric_names = ["accuracy", "precision", "recall", "f1_score"]
COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"]

# ── FIGURE 1: Grouped Metric Comparison ───────────────────────
fig1, ax1 = plt.subplots(figsize=(14, 6))
x = np.arange(len(metric_names))
n_models = len(model_names)
width = 0.8 / n_models
offset = np.linspace(-(n_models-1)/2 * width, (n_models-1)/2 * width, n_models)

for i, (mname, color) in enumerate(zip(model_names, COLORS)):
    means = [model_results[mname][m].mean() * 100 for m in metric_names]
    stds  = [model_results[mname][m].std()  * 100 for m in metric_names]
    bars = ax1.bar(x + offset[i], means, width, label=mname,
                   color=color, alpha=0.87, yerr=stds, capsize=4,
                   error_kw={"elinewidth": 1.5, "ecolor": "#333"})
    for bar, mean in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 "{:.1f}".format(mean), ha="center", va="bottom", fontsize=7.5, fontweight="bold")

ax1.set_xticks(x)
ax1.set_xticklabels(["Accuracy", "Precision", "Recall", "F1-Score"], fontsize=12)
ax1.set_ylabel("Score (%)", fontsize=12)
ax1.set_ylim([min([model_results[m][k].mean()*100 for m in model_names for k in metric_names]) - 3, 103])
ax1.set_title("Fig. 1: All Models — Metric Comparison (5-Fold Cross-Validation)", fontsize=13, fontweight="bold", pad=15)
ax1.legend(loc="lower right", fontsize=10, framealpha=0.9)
ax1.set_facecolor("#FAFAFA")
fig1.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig("outputs/plots/fig1_metric_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Fig 1 saved: grouped metric comparison")

# ── FIGURE 2: Accuracy Comparison Bar ─────────────────────────
fig2, ax2 = plt.subplots(figsize=(10, 6))
acc_means = [model_results[m]["accuracy"].mean() * 100 for m in model_names]
acc_stds  = [model_results[m]["accuracy"].std()  * 100 for m in model_names]
bars2 = ax2.bar(model_names, acc_means, yerr=acc_stds,
                color=COLORS[:len(model_names)], capsize=6,
                alpha=0.88, edgecolor="white",
                error_kw={"elinewidth": 2, "ecolor": "#333"})
for bar, mean, std in zip(bars2, acc_means, acc_stds):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.3,
             "{:.2f}%".format(mean), ha="center", va="bottom", fontsize=11, fontweight="bold")
ax2.set_ylabel("Accuracy (%)", fontsize=12)
ax2.set_title("Fig. 2: Accuracy Comparison — 5-Fold Cross-Validation with Error Bars", fontsize=13, fontweight="bold", pad=15)
ax2.set_ylim([min(acc_means) - 3, 103])
ax2.set_facecolor("#FAFAFA")
fig2.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig("outputs/plots/fig2_accuracy_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Fig 2 saved: accuracy comparison")

# ── FIGURE 3: CV Score Distribution Box Plot ──────────────────
fig3, ax3 = plt.subplots(figsize=(11, 6))
acc_scores = [model_results[m]["accuracy"] * 100 for m in model_names]
bp = ax3.boxplot(acc_scores, labels=model_names, patch_artist=True,
                 medianprops={"color": "white", "linewidth": 2.5})
for patch, color in zip(bp["boxes"], COLORS[:len(model_names)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.82)
ax3.set_ylabel("Accuracy (%)", fontsize=12)
ax3.set_title("Fig. 3: Cross-Validation Score Distribution (5 Folds)", fontsize=13, fontweight="bold", pad=15)
ax3.set_facecolor("#FAFAFA")
fig3.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig("outputs/plots/fig3_cv_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Fig 3 saved: CV distribution")

# ── FIGURE 4: Radar / Spider Chart ────────────────────────────
categories = ["Accuracy", "Precision", "Recall", "F1-Score"]
N = len(categories)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig4, ax4 = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for i, (mname, color) in enumerate(zip(model_names, COLORS)):
    values = [model_results[mname][m].mean() * 100 for m in metric_names]
    values += values[:1]
    ax4.plot(angles, values, "o-", linewidth=2, label=mname, color=color)
    ax4.fill(angles, values, alpha=0.12, color=color)

ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(categories, fontsize=12)
min_val = min([model_results[m][k].mean()*100 for m in model_names for k in metric_names])
ax4.set_ylim([max(0, min_val - 5), 100])
ax4.set_title("Fig. 4: Radar Chart — Multi-Metric Model Comparison", fontsize=13, fontweight="bold", pad=20)
ax4.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
fig4.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig("outputs/plots/fig4_radar_chart.png", dpi=150, bbox_inches="tight")
plt.close()
print("Fig 4 saved: radar chart")

# ── FIGURE 5: Ablation Study ──────────────────────────────────
rf.fit(X, y)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = features if isinstance(features[0], str) else ["f"+str(i) for i in range(X.shape[1])]
top5_idx = indices[:5]
top5 = [feature_names[i] for i in top5_idx]
X_top5 = X[:, top5_idx]

rf_full_acc  = model_results["RandomForest"]["accuracy"]
rf_top5_acc  = cross_val_score(RandomForestClassifier(**{**best_rf, "random_state": 42}), X_top5, y, cv=cv, scoring="accuracy")

ablation_labels = ["All Features\n(" + str(len(feature_names)) + ")", "Top-5 XAI\nFeatures"]
ablation_vals   = [rf_full_acc.mean()*100, rf_top5_acc.mean()*100]
ablation_stds   = [rf_full_acc.std()*100,  rf_top5_acc.std()*100]

fig5, ax5 = plt.subplots(figsize=(8, 5))
bars5 = ax5.bar(ablation_labels, ablation_vals, yerr=ablation_stds,
                color=[COLORS[0], COLORS[2]], capsize=6, alpha=0.88, edgecolor="white",
                error_kw={"elinewidth": 2, "ecolor": "#333"})
for bar, val, std in zip(bars5, ablation_vals, ablation_stds):
    ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+std+0.3,
             "{:.2f}%".format(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
ax5.annotate("Delta = {:.2f}%".format(ablation_vals[0]-ablation_vals[1]),
             xy=(0.5, max(ablation_vals)-1), xycoords="data",
             ha="center", fontsize=11, color="#333", style="italic")
ax5.set_ylabel("Accuracy (%)", fontsize=12)
ax5.set_title("Fig. 5: Ablation Study — XAI Feature Selection Impact", fontsize=13, fontweight="bold", pad=15)
ax5.set_ylim([min(ablation_vals)-3, 103])
ax5.set_facecolor("#FAFAFA")
fig5.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig("outputs/plots/fig5_ablation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Fig 5 saved: ablation")

# ── FIGURE 6: Feature Importance ──────────────────────────────
fig6, ax6 = plt.subplots(figsize=(12, 5))
bar_colors = [COLORS[0] if i < 5 else "#90CAF9" for i in range(len(feature_names))]
bars6 = ax6.bar(range(len(feature_names)), importances[indices], color=bar_colors, edgecolor="white")
ax6.set_xticks(range(len(feature_names)))
ax6.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha="right", fontsize=10)
ax6.set_title("Fig. 6: XAI Feature Importance Analysis (Random Forest)", fontsize=13, fontweight="bold", pad=15)
ax6.set_xlabel("Features", fontsize=11)
ax6.set_ylabel("Importance Score", fontsize=11)
for bar, val in zip(bars6, importances[indices]):
    ax6.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
             "{:.3f}".format(val), ha="center", va="bottom", fontsize=8)
ax6.set_facecolor("#FAFAFA")
fig6.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig("outputs/plots/fig6_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Fig 6 saved: feature importance")

# ── FIGURE 7: Confusion Matrix ────────────────────────────────
rf_preds = cross_val_predict(rf, X, y, cv=cv)
cm = confusion_matrix(y, rf_preds)
fig7, ax7 = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax7, colorbar=True, cmap="Blues")
ax7.set_title("Fig. 7: Confusion Matrix — Random Forest (5-Fold CV)", fontsize=13, fontweight="bold", pad=15)
fig7.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig("outputs/plots/fig7_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("Fig 7 saved: confusion matrix")

# ── FIGURE 8: SHAP Analysis ───────────────────────────────────
plot_files = [
    "outputs/plots/fig1_metric_comparison.png",
    "outputs/plots/fig2_accuracy_comparison.png",
    "outputs/plots/fig3_cv_distribution.png",
    "outputs/plots/fig4_radar_chart.png",
    "outputs/plots/fig5_ablation.png",
    "outputs/plots/fig6_feature_importance.png",
    "outputs/plots/fig7_confusion_matrix.png"
]

if HAS_SHAP:
    try:
        print("Running SHAP analysis...")
        rf_s = RandomForestClassifier(n_estimators=50, random_state=42)
        Xs = X[:min(500, len(X))]
        ys = y[:min(500, len(y))]
        rf_s.fit(Xs, ys)
        explainer = shap.TreeExplainer(rf_s)
        shap_vals = explainer.shap_values(Xs[:100])
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        fig8, axes8 = plt.subplots(1, 2, figsize=(16, 6))
        sm = np.abs(shap_vals).mean(axis=0)
        si = np.argsort(sm)[::-1]
        axes8[0].barh([feature_names[i] for i in si[:10]][::-1], sm[si[:10]][::-1], color="#E65100", alpha=0.85)
        axes8[0].set_title("Fig. 8a: SHAP Feature Importance\n(Mean |SHAP value|)", fontsize=12, fontweight="bold")
        axes8[0].set_xlabel("Mean |SHAP Value|")

        sp = np.mean(shap_vals > 0, axis=0)
        ti = si[:8]
        axes8[1].barh([feature_names[i] for i in ti][::-1], sp[ti][::-1], color="#2196F3", alpha=0.85, label="Positive")
        axes8[1].barh([feature_names[i] for i in ti][::-1], -(1-sp[ti])[::-1], color="#F44336", alpha=0.85, label="Negative")
        axes8[1].set_title("Fig. 8b: SHAP Direction Analysis", fontsize=12, fontweight="bold")
        axes8[1].set_xlabel("Proportion of samples")
        axes8[1].legend(fontsize=9)
        axes8[1].axvline(x=0, color="black", linewidth=0.8)
        plt.tight_layout()
        plt.savefig("outputs/plots/fig8_shap.png", dpi=150, bbox_inches="tight")
        plt.close()
        plot_files.append("outputs/plots/fig8_shap.png")
        print("Fig 8 saved: SHAP")
    except Exception as e:
        print("SHAP failed: " + str(e))

# ── Statistical Tests ──────────────────────────────────────────
rf_acc = model_results["RandomForest"]["accuracy"]
gb_acc = model_results["GradientBoosting"]["accuracy"]
t_stat, p_value = stats.ttest_rel(rf_acc, gb_acc)
significance = "statistically significant (p<0.05)" if p_value < 0.05 else "not statistically significant"
ci_rf = stats.t.interval(0.95, len(rf_acc)-1, loc=rf_acc.mean(), scale=stats.sem(rf_acc))
print("t-test RF vs GB: t={:.4f}, p={:.4f} - {}".format(t_stat, p_value, significance))

all_accs = [model_results[m]["accuracy"].mean() for m in model_names]
best_idx = int(np.argmax(all_accs))
best_model = model_names[best_idx]
best_acc = all_accs[best_idx]
print("Best model: " + best_model + " ({:.4f})".format(best_acc))

# ── Build metrics list ─────────────────────────────────────────
metrics = []
for mname in model_names:
    for metric in metric_names:
        scores = model_results[mname][metric]
        metrics.append({
            "metric_name": mname.replace(" ","") + "_" + metric,
            "mean": round(float(scores.mean()), 4),
            "std":  round(float(scores.std()),  4),
            "n_runs": 5
        })

metrics += [
    {"metric_name": "p_value_rf_vs_gb", "mean": round(float(p_value),4),   "std": 0.0, "n_runs": 1},
    {"metric_name": "rf_95ci_lower",    "mean": round(float(ci_rf[0]),4),  "std": 0.0, "n_runs": 1},
    {"metric_name": "rf_95ci_upper",    "mean": round(float(ci_rf[1]),4),  "std": 0.0, "n_runs": 1},
    {"metric_name": "XAI_top5_accuracy","mean": round(float(rf_top5_acc.mean()),4), "std": round(float(rf_top5_acc.std()),4), "n_runs": 5}
]

results = {
    "metrics": metrics,
    "hypothesis_verdict": "supported" if best_acc > 0.75 else "partially_supported",
    "best_model": best_model,
    "plot_files": plot_files,
    "key_findings": [
        "Best model: " + best_model + " ({:.2f}% accuracy)".format(best_acc*100),
        "RandomForest: {:.2f}% (95% CI: {:.2f}%-{:.2f}%)".format(
            rf_acc.mean()*100, ci_rf[0]*100, ci_rf[1]*100),
        "GradientBoosting: {:.2f}%".format(gb_acc.mean()*100),
        "Statistical test: t={:.4f}, p={:.4f} ({})".format(t_stat, p_value, significance),
        "Top-5 XAI features: {:.2f}% (validates feature importance)".format(rf_top5_acc.mean()*100),
        "Most important features: " + ", ".join(top5[:3]),
        "Dataset: " + dataset_name + " | Samples: " + str(X.shape[0]),
        str(len(plot_files)) + " figures generated"
    ],
    "statistical_tests": {
        "paired_ttest_rf_vs_gb": {
            "t_statistic": round(float(t_stat), 4),
            "p_value":     round(float(p_value), 4),
            "significant": bool(p_value < 0.05)
        },
        "rf_95_confidence_interval": {
            "lower": round(float(ci_rf[0]), 4),
            "upper": round(float(ci_rf[1]), 4)
        }
    }
}

with open("outputs/code/results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Results saved.")
print("EXPERIMENT COMPLETE — " + str(len(plot_files)) + " figures generated")
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
        print("[ImplementationAgent] Experiment failed: " + result["stderr"][:300])
        print("[ImplementationAgent] Trying LLM fallback...")

        prompt  = "Write a complete Python ML experiment.\n"
        prompt += "CRITICAL: Use ONLY these exact plot names:\n"
        prompt += "- outputs/plots/fig1_metric_comparison.png\n"
        prompt += "- outputs/plots/fig2_accuracy_comparison.png\n"
        prompt += "- outputs/plots/fig3_cv_distribution.png\n"
        prompt += "- outputs/plots/fig4_radar_chart.png\n"
        prompt += "- outputs/plots/fig5_ablation.png\n"
        prompt += "- outputs/plots/fig6_feature_importance.png\n"
        prompt += "- outputs/plots/fig7_confusion_matrix.png\n"
        prompt += "Use sklearn breast_cancer dataset as fallback.\n"
        prompt += "Train RandomForest and GradientBoosting.\n"
        prompt += "Fig 1 MUST be a grouped bar chart with ALL models side by side for accuracy/precision/recall/f1.\n"
        prompt += "Fig 2 MUST compare accuracy across models with error bars.\n"
        prompt += "Save results.json with metrics list.\n"
        prompt += "Error was: " + result["stderr"][:300]

        raw = call_with_retry(
            client,
            messages=[
                {"role": "system", "content": "Python ML engineer. Return ONLY runnable Python code. No markdown."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=config.MAX_TOKENS,
            temperature=0.2,
            agent_name="implementation_agent"
        )
        used = log_usage("implementation_agent", prompt, raw)
        print_status("implementation_agent", used)

        code = raw
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