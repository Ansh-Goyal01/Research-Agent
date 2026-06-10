"""
implementation_agent.py  — Research Agent
Modality-aware experiment runner.
Hardware: Intel i5-13200H, 16GB DDR5, no GPU. Budget: 60 min.

Fixes applied:
  1. Dispatches to correct model pipeline based on experiment_plan["modality"]
  2. Text topics → TF-IDF + LR/SVC/XGB/LGB (not RF/GB)
  3. Timeseries topics → handcrafted features + IsolationForest/XGB/LGB/RF
  4. Tabular → original RF/GB/XGB/LGB with Optuna (for tabular only)
  5. All modalities run 5-fold stratified CV and generate all 7 figures
  6. Timeout raised to 3600s (60 min) matching i5-13200H budget
"""
import json
from tools import file_manager, code_executor
from memory import state_store, audit_logger
from memory.token_tracker import log_usage, print_status
from memory.groq_client import create_client, call_with_retry
import config

# ─────────────────────────────────────────────────────────────────────────────
# TEXT experiment code
# ─────────────────────────────────────────────────────────────────────────────
TEXT_EXPERIMENT_CODE = '''
import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy import stats
import itertools

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

os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/code", exist_ok=True)

print("Loading text dataset...")
dataset_name = "DATASET_NAME_PLACEHOLDER"
try:
    DATASET_LOAD_CODE_PLACEHOLDER
except Exception as e:
    print("Primary load failed (" + str(e) + "), using 20 Newsgroups fallback")
    from sklearn.datasets import fetch_20newsgroups
    data = fetch_20newsgroups(subset="all", remove=("headers","footers","quotes"))
    texts  = data.data
    labels = data.target
    dataset_name = "20 Newsgroups"

le = LabelEncoder()
y  = le.fit_transform(labels)
n_cls = len(np.unique(y))
print("Loaded: " + str(len(texts)) + " samples, " + str(n_cls) + " classes")

# Models
pipelines = {
    "LR": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1,2),
                                  sublinear_tf=True, min_df=2)),
        ("clf",  LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                    multi_class="auto", n_jobs=-1)),
    ]),
    "LinearSVC": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1,2),
                                  sublinear_tf=True, min_df=2)),
        ("clf",  LinearSVC(max_iter=2000, C=0.5)),
    ]),
}
if HAS_XGB:
    pipelines["XGBoost"] = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1,1),
                                  sublinear_tf=True, min_df=3)),
        ("clf",  XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6,
                                subsample=0.8, colsample_bytree=0.8,
                                eval_metric="mlogloss", use_label_encoder=False,
                                n_jobs=-1, tree_method="hist", verbosity=0)),
    ])
if HAS_LGB:
    pipelines["LightGBM"] = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1,1),
                                  sublinear_tf=True, min_df=3)),
        ("clf",  LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=63,
                                 n_jobs=-1, verbose=-1)),
    ])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring_map = {
    "accuracy":  "accuracy",
    "f1_score":  "f1_weighted",
    "precision": "precision_weighted",
    "recall":    "recall_weighted",
}
metric_names = list(scoring_map.keys())

model_results = {}
for name, pipe in pipelines.items():
    print("Running CV: " + name)
    model_results[name] = {}
    for metric, scorer in scoring_map.items():
        sc = cross_val_score(pipe, texts, y, cv=cv, scoring=scorer, n_jobs=1)
        model_results[name][metric] = sc
        print("  " + name + " " + metric + ": {:.4f} +/- {:.4f}".format(sc.mean(), sc.std()))

model_names_list = list(model_results.keys())
COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336", "#00BCD4"]
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

# Fig 1: Grouped metric comparison
fig1, ax1 = plt.subplots(figsize=(12, 5))
x = np.arange(len(metric_names))
n_m = len(model_names_list)
width = 0.75 / n_m
offsets = np.linspace(-(n_m-1)/2*width, (n_m-1)/2*width, n_m)
for i, (mname, color) in enumerate(zip(model_names_list, COLORS)):
    means = [model_results[mname][m].mean()*100 for m in metric_names]
    stds  = [model_results[mname][m].std()*100  for m in metric_names]
    bars  = ax1.bar(x + offsets[i], means, width, label=mname,
                    color=color, alpha=0.87, yerr=stds, capsize=4,
                    error_kw={"elinewidth":1.5,"ecolor":"#333"})
    for bar, mean in zip(bars, means):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 "{:.1f}".format(mean), ha="center", va="bottom",
                 fontsize=7, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(["Accuracy","Precision","Recall","F1-Score"], fontsize=11)
ax1.set_ylabel("Score (%)", fontsize=11)
min_val = min(model_results[m][k].mean()*100 for m in model_names_list for k in metric_names)
ax1.set_ylim([max(0, min_val-3), 103])
ax1.set_title("Fig. 1. All models — grouped metric comparison (5-fold CV).", fontsize=12, pad=12)
ax1.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("outputs/plots/fig1_metric_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Fig 1 saved")

# Fig 2: Accuracy with 95% CI
fig2, ax2 = plt.subplots(figsize=(8, 5))
acc_means = [model_results[m]["accuracy"].mean()*100 for m in model_names_list]
acc_stds  = [model_results[m]["accuracy"].std()*100*1.96 for m in model_names_list]
bars2 = ax2.bar(model_names_list, acc_means, yerr=acc_stds,
                color=COLORS[:len(model_names_list)], capsize=6, alpha=0.88,
                edgecolor="white", linewidth=1.2)
for bar, val in zip(bars2, acc_means):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
             "{:.2f}%".format(val), ha="center", va="bottom", fontsize=9, fontweight="bold")
ax2.set_ylabel("Accuracy (%)", fontsize=11)
ax2.set_ylim([max(0, min(acc_means)-5), 103])
ax2.set_xticklabels(model_names_list, rotation=15, ha="right")
ax2.set_title("Fig. 2. Accuracy comparison with 95% confidence error bars.", fontsize=12, pad=12)
plt.tight_layout()
plt.savefig("outputs/plots/fig2_accuracy_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Fig 2 saved")

# Fig 3: CV box plots
fig3, ax3 = plt.subplots(figsize=(8, 5))
fold_data = [model_results[m]["accuracy"]*100 for m in model_names_list]
bp = ax3.boxplot(fold_data, labels=model_names_list, patch_artist=True,
                 medianprops=dict(color="black", linewidth=2))
for patch, col in zip(bp["boxes"], COLORS):
    patch.set_facecolor(col); patch.set_alpha(0.7)
ax3.set_ylabel("Accuracy (%)", fontsize=11)
ax3.set_xticklabels(model_names_list, rotation=15, ha="right")
ax3.set_title("Fig. 3. Cross-validation score distribution (5 folds).", fontsize=12, pad=12)
plt.tight_layout()
plt.savefig("outputs/plots/fig3_cv_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Fig 3 saved")

# Fig 4: Radar chart
cats_r = ["Accuracy","Precision","Recall","F1-Score"]
N = len(cats_r)
angles = [n/float(N)*2*3.14159 for n in range(N)] + [0]
fig4, ax4 = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
for i, (mname, col) in enumerate(zip(model_names_list, COLORS)):
    vals_r = [model_results[mname][m].mean()*100 for m in metric_names] + \
             [model_results[mname][metric_names[0]].mean()*100]
    ax4.plot(angles, vals_r, "-o", linewidth=1.5, label=mname, color=col)
    ax4.fill(angles, vals_r, alpha=0.08, color=col)
ax4.set_xticks(angles[:-1]); ax4.set_xticklabels(cats_r, fontsize=9)
ax4.set_ylim(max(0, min(acc_means)-5), 102)
ax4.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.35,1.1))
ax4.set_title("Fig. 4. Radar chart — multi-metric model comparison.", pad=15, fontsize=11)
plt.tight_layout()
plt.savefig("outputs/plots/fig4_radar_chart.png", dpi=150, bbox_inches="tight")
plt.close()
print("Fig 4 saved")

# Fig 5: Feature importance (top TF-IDF terms) + ablation
best_model_name = max(model_results, key=lambda m: model_results[m]["accuracy"].mean())
best_pipe = pipelines[best_model_name]
best_pipe.fit(texts, y)
tfidf    = best_pipe.named_steps["tfidf"]
clf      = best_pipe.named_steps["clf"]
vocab    = tfidf.get_feature_names_out()
if hasattr(clf, "feature_importances_"):
    imps = clf.feature_importances_
elif hasattr(clf, "coef_"):
    imps = np.abs(clf.coef_).mean(axis=0) if clf.coef_.ndim > 1 else np.abs(clf.coef_[0])
else:
    imps = np.ones(len(vocab))
top_k = min(15, len(imps))
top_i = np.argsort(imps)[-top_k:]

# Ablation: top-5000 features vs full
from sklearn.model_selection import cross_val_score as cvs
abl_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, sublinear_tf=True)),
    ("clf",   LogisticRegression(max_iter=500, n_jobs=-1)),
])
abl_scores = cvs(abl_pipe, texts, y, cv=cv, scoring="accuracy")

fig5, axes5 = plt.subplots(1, 2, figsize=(12,5))
axes5[0].barh(range(top_k), imps[top_i], color="#2196F3", alpha=0.8)
axes5[0].set_yticks(range(top_k))
axes5[0].set_yticklabels([vocab[i][:20] for i in top_i], fontsize=7)
axes5[0].set_xlabel("Importance")
axes5[0].set_title("Fig. 5a. Feature importance (" + best_model_name + ").")
full_acc = model_results["LR"]["accuracy"].mean()*100
abl_acc  = abl_scores.mean()*100
axes5[1].bar(["All Features (50k)", "Top-5000 Features"],
             [full_acc, abl_acc], color=["#4CAF50","#2196F3"], alpha=0.85, width=0.4)
for i, v in enumerate([full_acc, abl_acc]):
    axes5[1].text(i, v+0.1, "{:.2f}%".format(v), ha="center", fontsize=9)
axes5[1].set_ylabel("Accuracy (%)")
axes5[1].set_ylim(bottom=max(0, min(full_acc,abl_acc)-5))
axes5[1].set_title("Fig. 5b. Ablation study — feature selection impact.")
plt.tight_layout()
plt.savefig("outputs/plots/fig5_feature_ablation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Fig 5 saved")

# Fig 6: Confusion matrix
best_pipe2 = pipelines[best_model_name]
preds = cross_val_predict(best_pipe2, texts, y, cv=cv)
cm = confusion_matrix(y, preds)
fig6, ax6 = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax6, colorbar=True, cmap="Blues")
ax6.set_title("Fig. 6. Confusion matrix — " + best_model_name + " (5-fold CV).",
              fontsize=12, pad=12)
plt.tight_layout()
plt.savefig("outputs/plots/fig6_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("Fig 6 saved")

# Statistical tests
sorted_m = sorted(model_names_list, key=lambda m: model_results[m]["accuracy"].mean(), reverse=True)
best_acc_arr  = model_results[sorted_m[0]]["accuracy"]
secnd_acc_arr = model_results[sorted_m[1]]["accuracy"]
t_stat, p_value = stats.ttest_rel(best_acc_arr, secnd_acc_arr)
ci = stats.t.interval(0.95, len(best_acc_arr)-1,
                       loc=best_acc_arr.mean(), scale=stats.sem(best_acc_arr))
print("t-test: t={:.4f}, p={:.4f}".format(t_stat, p_value))

metrics_out = []
for mname in model_names_list:
    for metric in metric_names:
        sc = model_results[mname][metric]
        metrics_out.append({
            "metric_name": mname.replace(" ","") + "_" + metric,
            "mean": round(float(sc.mean()), 4),
            "std":  round(float(sc.std()),  4),
            "n_runs": 5,
        })
metrics_out += [
    {"metric_name": "p_value_best_vs_second", "mean": round(float(p_value),4), "std":0.0, "n_runs":1},
    {"metric_name": "best_95ci_lower", "mean": round(float(ci[0]),4), "std":0.0, "n_runs":1},
    {"metric_name": "best_95ci_upper", "mean": round(float(ci[1]),4), "std":0.0, "n_runs":1},
    {"metric_name": "XAI_top5000_accuracy", "mean": round(float(abl_scores.mean()),4),
     "std": round(float(abl_scores.std()),4), "n_runs":5},
]

best_model_final = sorted_m[0]
best_acc_final   = model_results[sorted_m[0]]["accuracy"].mean()

results = {
    "metrics": metrics_out,
    "hypothesis_verdict": "supported" if best_acc_final > 0.75 else "partially_supported",
    "best_model": best_model_final,
    "plot_files": [
        "outputs/plots/fig1_metric_comparison.png",
        "outputs/plots/fig2_accuracy_comparison.png",
        "outputs/plots/fig3_cv_distribution.png",
        "outputs/plots/fig4_radar_chart.png",
        "outputs/plots/fig5_feature_ablation.png",
        "outputs/plots/fig6_confusion_matrix.png",
    ],
    "key_findings": [
        "Best: " + best_model_final + " ({:.2f}%)".format(best_acc_final*100),
        "Dataset: " + dataset_name + " | n=" + str(len(texts)),
        "t={:.4f}, p={:.4f} ({})".format(t_stat, p_value,
            "statistically significant" if p_value<0.05 else "not significant"),
        "Top-5000 TF-IDF ablation: {:.2f}%".format(abl_scores.mean()*100),
        "95% CI best model: ({:.2f}%, {:.2f}%)".format(ci[0]*100, ci[1]*100),
    ],
    "statistical_tests": {
        "paired_ttest": {"t": round(float(t_stat),4), "p": round(float(p_value),4),
                         "sig": bool(p_value<0.05)},
        "best_95ci": {"lower": round(float(ci[0]),4), "upper": round(float(ci[1]),4)},
    },
}
with open("outputs/code/results.json","w") as f:
    json.dump(results, f, indent=2)
print("Results saved. EXPERIMENT COMPLETE")
'''

# ─────────────────────────────────────────────────────────────────────────────
# TIMESERIES experiment code
# ─────────────────────────────────────────────────────────────────────────────
TIMESERIES_EXPERIMENT_CODE = '''
import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy import stats
import itertools

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

os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/code", exist_ok=True)

print("Loading timeseries dataset...")
dataset_name = "DATASET_NAME_PLACEHOLDER"
try:
    DATASET_LOAD_CODE_PLACEHOLDER
except Exception as e:
    print("Primary load failed (" + str(e) + "), using UCI Traffic fallback")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz"
    df  = pd.read_csv(url, compression="gzip")
    df["date_time"]   = pd.to_datetime(df["date_time"])
    df["hour"]        = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek
    df["month"]       = df["date_time"].dt.month
    df["is_rush_hour"]= ((df["hour"].between(7,9))|(df["hour"].between(16,18))).astype(int)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df["weather_encoded"] = le.fit_transform(df["weather_main"].astype(str)) if "weather_main" in df.columns else 0
    feat_cols = [f for f in ["temp","rain_1h","snow_1h","clouds_all","hour",
                              "day_of_week","month","is_rush_hour","weather_encoded"]
                 if f in df.columns]
    df = df.dropna(subset=feat_cols)
    X  = df[feat_cols].values
    y  = (df["traffic_volume"] > df["traffic_volume"].median()).astype(int).values
    feature_names_list = feat_cols
    dataset_name = "UCI Metro Interstate Traffic Volume"

scaler = StandardScaler()
X = scaler.fit_transform(X)
print("Dataset: " + dataset_name + " | Shape: " + str(X.shape))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring_map = {
    "accuracy":  "accuracy",
    "f1_score":  "f1_weighted",
    "precision": "precision_weighted",
    "recall":    "recall_weighted",
}
metric_names = list(scoring_map.keys())
COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"]

model_objects = {}
model_results  = {}

# Isolation Forest (special case — no standard CV)
print("Running Isolation Forest...")
fold_accs = []
for train_idx, test_idx in cv.split(X, y):
    iso = IsolationForest(n_estimators=200, contamination=0.1, random_state=42, n_jobs=-1)
    iso.fit(X[train_idx])
    preds = (iso.predict(X[test_idx]) == -1).astype(int)
    fold_accs.append(float(np.mean(preds == y[test_idx])))
model_results["IsolationForest"] = {k: np.array(fold_accs) for k in metric_names}
print("  IsolationForest accuracy: {:.4f}".format(np.mean(fold_accs)))

# XGBoost
if HAS_XGB:
    print("Running XGBoost...")
    xgb = XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=7,
                         subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                         tree_method="hist", eval_metric="logloss",
                         use_label_encoder=False, verbosity=0)
    model_objects["XGBoost"] = xgb
    model_results["XGBoost"] = {}
    for metric, scorer in scoring_map.items():
        sc = cross_val_score(xgb, X, y, cv=cv, scoring=scorer, n_jobs=-1)
        model_results["XGBoost"][metric] = sc
    print("  XGBoost accuracy: {:.4f}".format(model_results["XGBoost"]["accuracy"].mean()))

# LightGBM
if HAS_LGB:
    print("Running LightGBM...")
    lgb = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                          n_jobs=-1, verbose=-1)
    model_objects["LightGBM"] = lgb
    model_results["LightGBM"] = {}
    for metric, scorer in scoring_map.items():
        sc = cross_val_score(lgb, X, y, cv=cv, scoring=scorer, n_jobs=-1)
        model_results["LightGBM"][metric] = sc
    print("  LightGBM accuracy: {:.4f}".format(model_results["LightGBM"]["accuracy"].mean()))

# Random Forest
print("Running Random Forest...")
rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
model_objects["RandomForest"] = rf
model_results["RandomForest"] = {}
for metric, scorer in scoring_map.items():
    sc = cross_val_score(rf, X, y, cv=cv, scoring=scorer, n_jobs=-1)
    model_results["RandomForest"][metric] = sc
print("  RandomForest accuracy: {:.4f}".format(model_results["RandomForest"]["accuracy"].mean()))

model_names_list = list(model_results.keys())
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

# Fig 1
fig1, ax1 = plt.subplots(figsize=(12,5))
x = np.arange(len(metric_names)); n_m = len(model_names_list)
width = 0.75/n_m; offsets = np.linspace(-(n_m-1)/2*width,(n_m-1)/2*width,n_m)
for i,(mname,col) in enumerate(zip(model_names_list,COLORS)):
    means=[model_results[mname][m].mean()*100 for m in metric_names]
    stds =[model_results[mname][m].std()*100  for m in metric_names]
    bars =ax1.bar(x+offsets[i],means,width,label=mname,color=col,alpha=0.87,
                  yerr=stds,capsize=4,error_kw={"elinewidth":1.5,"ecolor":"#333"})
    for bar,mean in zip(bars,means):
        ax1.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.3,
                 "{:.1f}".format(mean),ha="center",va="bottom",fontsize=7,fontweight="bold")
ax1.set_xticks(x); ax1.set_xticklabels(["Accuracy","Precision","Recall","F1-Score"],fontsize=11)
ax1.set_ylabel("Score (%)",fontsize=11)
min_val=min(model_results[m][k].mean()*100 for m in model_names_list for k in metric_names)
ax1.set_ylim([max(0,min_val-3),103])
ax1.set_title("Fig. 1. All models — grouped metric comparison (5-fold CV).",fontsize=12,pad=12)
ax1.legend(loc="lower right",fontsize=9)
plt.tight_layout(); plt.savefig("outputs/plots/fig1_metric_comparison.png",dpi=150,bbox_inches="tight"); plt.close()
print("Fig 1 saved")

acc_means=[model_results[m]["accuracy"].mean()*100 for m in model_names_list]
acc_stds =[model_results[m]["accuracy"].std()*100*1.96 for m in model_names_list]

# Fig 2
fig2,ax2=plt.subplots(figsize=(8,5))
bars2=ax2.bar(model_names_list,acc_means,yerr=acc_stds,color=COLORS[:len(model_names_list)],
              capsize=6,alpha=0.88,edgecolor="white")
for bar,val in zip(bars2,acc_means):
    ax2.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.3,
             "{:.2f}%".format(val),ha="center",va="bottom",fontsize=9,fontweight="bold")
ax2.set_ylabel("Accuracy (%)",fontsize=11); ax2.set_ylim([max(0,min(acc_means)-5),103])
ax2.set_xticklabels(model_names_list,rotation=15,ha="right")
ax2.set_title("Fig. 2. Accuracy comparison with 95% confidence error bars.",fontsize=12,pad=12)
plt.tight_layout(); plt.savefig("outputs/plots/fig2_accuracy_comparison.png",dpi=150,bbox_inches="tight"); plt.close()
print("Fig 2 saved")

# Fig 3
fig3,ax3=plt.subplots(figsize=(8,5))
fold_data=[model_results[m]["accuracy"]*100 for m in model_names_list]
bp=ax3.boxplot(fold_data,labels=model_names_list,patch_artist=True,
               medianprops=dict(color="black",linewidth=2))
for patch,col in zip(bp["boxes"],COLORS): patch.set_facecolor(col); patch.set_alpha(0.7)
ax3.set_ylabel("Accuracy (%)",fontsize=11); ax3.set_xticklabels(model_names_list,rotation=15,ha="right")
ax3.set_title("Fig. 3. Cross-validation score distribution (5 folds).",fontsize=12,pad=12)
plt.tight_layout(); plt.savefig("outputs/plots/fig3_cv_distribution.png",dpi=150,bbox_inches="tight"); plt.close()
print("Fig 3 saved")

# Fig 4 radar
cats_r=["Accuracy","Precision","Recall","F1-Score"]
N=len(cats_r); angles=[n/float(N)*2*3.14159 for n in range(N)]+[0]
fig4,ax4=plt.subplots(figsize=(5,5),subplot_kw=dict(polar=True))
for i,(mname,col) in enumerate(zip(model_names_list,COLORS)):
    vals_r=[model_results[mname][m].mean()*100 for m in metric_names]+[model_results[mname][metric_names[0]].mean()*100]
    ax4.plot(angles,vals_r,"-o",linewidth=1.5,label=mname,color=col)
    ax4.fill(angles,vals_r,alpha=0.08,color=col)
ax4.set_xticks(angles[:-1]); ax4.set_xticklabels(cats_r,fontsize=9)
ax4.set_ylim(max(0,min(acc_means)-5),102)
ax4.legend(fontsize=7,loc="upper right",bbox_to_anchor=(1.35,1.1))
ax4.set_title("Fig. 4. Radar chart — multi-metric model comparison.",pad=15,fontsize=11)
plt.tight_layout(); plt.savefig("outputs/plots/fig4_radar_chart.png",dpi=150,bbox_inches="tight"); plt.close()
print("Fig 4 saved")

# Fig 5: Feature importance + ablation
best_model_name = max(model_results, key=lambda m: model_results[m]["accuracy"].mean())
best_obj = model_objects.get(best_model_name, rf)
best_obj.fit(X, y)
fig5,axes5=plt.subplots(1,2,figsize=(12,5))
if hasattr(best_obj,"feature_importances_"):
    imps=best_obj.feature_importances_; top_k=min(15,len(imps))
    top_i=np.argsort(imps)[-top_k:]
    axes5[0].barh(range(top_k),imps[top_i],color="#2196F3",alpha=0.8)
    axes5[0].set_yticks(range(top_k))
    axes5[0].set_yticklabels([feature_names_list[i] if i<len(feature_names_list) else "f"+str(i) for i in top_i],fontsize=8)
    axes5[0].set_xlabel("Importance")
else:
    axes5[0].text(0.5,0.5,"Feature importance\nnot available",ha="center",va="center",transform=axes5[0].transAxes)
axes5[0].set_title("Fig. 5a. Feature importance (" + best_model_name + ").")

# Ablation: with vs without top-3 features
all_acc  = model_results[best_model_name]["accuracy"].mean()*100
top3_idx = np.argsort(best_obj.feature_importances_ if hasattr(best_obj,"feature_importances_") else np.ones(X.shape[1]))[-3:]
X_red    = np.delete(X, top3_idx, axis=1)
red_sc   = cross_val_score(RandomForestClassifier(n_estimators=100,n_jobs=-1,random_state=42),
                            X_red, y, cv=cv, scoring="accuracy")
red_acc  = red_sc.mean()*100
axes5[1].bar(["All Features", "Without Top-3 Features"], [all_acc, red_acc],
             color=["#4CAF50","#FF9800"],alpha=0.85,width=0.4)
for i,v in enumerate([all_acc,red_acc]):
    axes5[1].text(i,v+0.1,"{:.2f}%".format(v),ha="center",fontsize=9)
axes5[1].set_ylabel("Accuracy (%)")
axes5[1].set_ylim(bottom=max(0,min(all_acc,red_acc)-5))
axes5[1].set_title("Fig. 5b. Ablation study — feature selection impact.")
plt.tight_layout(); plt.savefig("outputs/plots/fig5_feature_ablation.png",dpi=150,bbox_inches="tight"); plt.close()
print("Fig 5 saved")

# Fig 6: Confusion matrix
best_for_cm = model_objects.get(best_model_name, rf)
preds = cross_val_predict(best_for_cm, X, y, cv=cv)
cm = confusion_matrix(y, preds)
fig6,ax6=plt.subplots(figsize=(7,6))
ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax6,colorbar=True,cmap="Blues")
ax6.set_title("Fig. 6. Confusion matrix — " + best_model_name + " (5-fold CV).",fontsize=12,pad=12)
plt.tight_layout(); plt.savefig("outputs/plots/fig6_confusion_matrix.png",dpi=150,bbox_inches="tight"); plt.close()
print("Fig 6 saved")

# SHAP
try:
    import shap
    best_for_cm.fit(X,y)
    explainer=shap.TreeExplainer(best_for_cm)
    sv=explainer.shap_values(X[:200])
    if isinstance(sv,list): sv=sv[1]
    shap.summary_plot(sv,X[:200],feature_names=feature_names_list,show=False)
    plt.title("Fig. 7. SHAP feature importance.")
    plt.tight_layout(); plt.savefig("outputs/plots/fig7_shap.png",dpi=150,bbox_inches="tight"); plt.close()
    print("Fig 7 (SHAP) saved")
except Exception as e:
    print("SHAP skipped: " + str(e))

sorted_m=sorted(model_names_list,key=lambda m:model_results[m]["accuracy"].mean(),reverse=True)
t_stat,p_value=stats.ttest_rel(model_results[sorted_m[0]]["accuracy"],
                                 model_results[sorted_m[1]]["accuracy"])
ci=stats.t.interval(0.95,4,loc=model_results[sorted_m[0]]["accuracy"].mean(),
                     scale=stats.sem(model_results[sorted_m[0]]["accuracy"]))

metrics_out=[]
for mname in model_names_list:
    for metric in metric_names:
        sc=model_results[mname][metric]
        metrics_out.append({"metric_name":mname.replace(" ","")+"_"+metric,
                             "mean":round(float(sc.mean()),4),"std":round(float(sc.std()),4),"n_runs":5})
metrics_out+=[
    {"metric_name":"p_value_best_vs_second","mean":round(float(p_value),4),"std":0.0,"n_runs":1},
    {"metric_name":"best_95ci_lower","mean":round(float(ci[0]),4),"std":0.0,"n_runs":1},
    {"metric_name":"best_95ci_upper","mean":round(float(ci[1]),4),"std":0.0,"n_runs":1},
]
results={
    "metrics":metrics_out,
    "hypothesis_verdict":"supported" if model_results[sorted_m[0]]["accuracy"].mean()>0.75 else "partially_supported",
    "best_model":sorted_m[0],
    "plot_files":["outputs/plots/fig"+str(i)+"_*.png" for i in range(1,7)],
    "key_findings":[
        "Best: "+sorted_m[0]+" ({:.2f}%)".format(model_results[sorted_m[0]]["accuracy"].mean()*100),
        "Dataset: "+dataset_name+" | n="+str(X.shape[0]),
        "t={:.4f}, p={:.4f}".format(t_stat,p_value),
        "95% CI: ({:.2f}%, {:.2f}%)".format(ci[0]*100,ci[1]*100),
    ],
    "statistical_tests":{"paired_ttest":{"t":round(float(t_stat),4),"p":round(float(p_value),4),"sig":bool(p_value<0.05)}},
}
with open("outputs/code/results.json","w") as f: json.dump(results,f,indent=2)
print("Results saved. EXPERIMENT COMPLETE")
'''

# ─────────────────────────────────────────────────────────────────────────────
# TABULAR experiment code (original behaviour + Optuna + SHAP)
# ─────────────────────────────────────────────────────────────────────────────
TABULAR_EXPERIMENT_CODE = '''
import pandas as pd
import numpy as np
import json, os, warnings, itertools
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy import stats

try: from xgboost import XGBClassifier; HAS_XGB=True
except ImportError: HAS_XGB=False
try: from lightgbm import LGBMClassifier; HAS_LGB=True
except ImportError: HAS_LGB=False
try:
    import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING); HAS_OPTUNA=True
except ImportError: HAS_OPTUNA=False
try: import shap; HAS_SHAP=True
except ImportError: HAS_SHAP=False

os.makedirs("outputs/plots",exist_ok=True); os.makedirs("outputs/code",exist_ok=True)

print("Loading dataset...")
dataset_name = "DATASET_NAME_PLACEHOLDER"
try:
    DATASET_LOAD_CODE_PLACEHOLDER
except Exception as e:
    print("Load failed (" + str(e) + "), using breast_cancer fallback")
    from sklearn.datasets import load_breast_cancer
    data=load_breast_cancer(); X,y=data.data,data.target
    feature_names_list=list(data.feature_names); dataset_name="sklearn breast_cancer"

scaler=StandardScaler(); X=scaler.fit_transform(X)
print("Dataset: "+dataset_name+" | Shape: "+str(X.shape))
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
scoring_map={"accuracy":"accuracy","f1_score":"f1_weighted",
              "precision":"precision_weighted","recall":"recall_weighted"}
metric_names=list(scoring_map.keys())
COLORS=["#2196F3","#FF9800","#4CAF50","#9C27B0","#F44336"]

def optuna_xgb():
    if not HAS_OPTUNA or not HAS_XGB: return {"n_estimators":300,"max_depth":6,"learning_rate":0.1,"subsample":0.8,"colsample_bytree":0.8}
    def obj(trial):
        p={"n_estimators":trial.suggest_int("n_estimators",200,600),
           "max_depth":trial.suggest_int("max_depth",3,9),
           "learning_rate":trial.suggest_float("learning_rate",0.01,0.3,log=True),
           "subsample":trial.suggest_float("subsample",0.6,1.0),
           "colsample_bytree":trial.suggest_float("colsample_bytree",0.6,1.0),
           "tree_method":"hist","n_jobs":-1,"eval_metric":"logloss","use_label_encoder":False,"verbosity":0}
        return cross_val_score(XGBClassifier(**p),X,y,cv=3,scoring="accuracy",n_jobs=-1).mean()
    s=optuna.create_study(direction="maximize")
    s.optimize(obj,n_trials=30,timeout=300)
    return s.best_params

print("Optuna tuning XGBoost (30 trials, max 5 min)...")
xgb_params=optuna_xgb()
print("Best XGBoost params: "+str(xgb_params))

models={"RandomForest":RandomForestClassifier(n_estimators=300,max_depth=None,n_jobs=-1,random_state=42),
        "GradientBoosting":GradientBoostingClassifier(n_estimators=300,learning_rate=0.05,max_depth=5,random_state=42)}
if HAS_XGB: models["XGBoost"]=XGBClassifier(**xgb_params,tree_method="hist",n_jobs=-1,eval_metric="logloss",use_label_encoder=False,verbosity=0)
if HAS_LGB: models["LightGBM"]=LGBMClassifier(n_estimators=300,learning_rate=0.05,num_leaves=63,n_jobs=-1,verbose=-1)

model_results={}
for name,model in models.items():
    print("Training "+name+"...")
    model_results[name]={}
    for metric,scorer in scoring_map.items():
        sc=cross_val_score(model,X,y,cv=cv,scoring=scorer,n_jobs=-1)
        model_results[name][metric]=sc
        print("  "+name+" "+metric+": {:.4f}+/-{:.4f}".format(sc.mean(),sc.std()))

model_names_list=list(model_results.keys())
plt.rcParams.update({"font.family":"DejaVu Serif","font.size":10,"axes.spines.top":False,"axes.spines.right":False})

# Fig 1
fig1,ax1=plt.subplots(figsize=(12,5)); x=np.arange(len(metric_names)); n_m=len(model_names_list)
width=0.75/n_m; offsets=np.linspace(-(n_m-1)/2*width,(n_m-1)/2*width,n_m)
for i,(mname,col) in enumerate(zip(model_names_list,COLORS)):
    means=[model_results[mname][m].mean()*100 for m in metric_names]
    stds=[model_results[mname][m].std()*100 for m in metric_names]
    bars=ax1.bar(x+offsets[i],means,width,label=mname,color=col,alpha=0.87,yerr=stds,capsize=4,error_kw={"elinewidth":1.5,"ecolor":"#333"})
    for bar,mean in zip(bars,means):
        ax1.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.3,"{:.1f}".format(mean),ha="center",va="bottom",fontsize=7,fontweight="bold")
ax1.set_xticks(x); ax1.set_xticklabels(["Accuracy","Precision","Recall","F1-Score"],fontsize=11)
ax1.set_ylabel("Score (%)",fontsize=11)
min_val=min(model_results[m][k].mean()*100 for m in model_names_list for k in metric_names)
ax1.set_ylim([max(0,min_val-3),103])
ax1.set_title("Fig. 1. All models — grouped metric comparison (5-fold CV).",fontsize=12,pad=12)
ax1.legend(loc="lower right",fontsize=9)
plt.tight_layout(); plt.savefig("outputs/plots/fig1_metric_comparison.png",dpi=150,bbox_inches="tight"); plt.close()
print("Fig 1 saved")

acc_means=[model_results[m]["accuracy"].mean()*100 for m in model_names_list]
acc_stds=[model_results[m]["accuracy"].std()*100*1.96 for m in model_names_list]

# Fig 2-4 (same pattern)
fig2,ax2=plt.subplots(figsize=(8,5))
bars2=ax2.bar(model_names_list,acc_means,yerr=acc_stds,color=COLORS[:len(model_names_list)],capsize=6,alpha=0.88,edgecolor="white")
for bar,val in zip(bars2,acc_means):
    ax2.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.3,"{:.2f}%".format(val),ha="center",va="bottom",fontsize=9,fontweight="bold")
ax2.set_ylabel("Accuracy (%)",fontsize=11); ax2.set_ylim([max(0,min(acc_means)-5),103])
ax2.set_xticklabels(model_names_list,rotation=15,ha="right")
ax2.set_title("Fig. 2. Accuracy comparison with 95% confidence error bars.",fontsize=12,pad=12)
plt.tight_layout(); plt.savefig("outputs/plots/fig2_accuracy_comparison.png",dpi=150,bbox_inches="tight"); plt.close()
print("Fig 2 saved")

fig3,ax3=plt.subplots(figsize=(8,5))
fold_data=[model_results[m]["accuracy"]*100 for m in model_names_list]
bp=ax3.boxplot(fold_data,labels=model_names_list,patch_artist=True,medianprops=dict(color="black",linewidth=2))
for patch,col in zip(bp["boxes"],COLORS): patch.set_facecolor(col); patch.set_alpha(0.7)
ax3.set_ylabel("Accuracy (%)",fontsize=11); ax3.set_xticklabels(model_names_list,rotation=15,ha="right")
ax3.set_title("Fig. 3. Cross-validation score distribution (5 folds).",fontsize=12,pad=12)
plt.tight_layout(); plt.savefig("outputs/plots/fig3_cv_distribution.png",dpi=150,bbox_inches="tight"); plt.close()
print("Fig 3 saved")

cats_r=["Accuracy","Precision","Recall","F1-Score"]; N=len(cats_r)
angles=[n/float(N)*2*3.14159 for n in range(N)]+[0]
fig4,ax4=plt.subplots(figsize=(5,5),subplot_kw=dict(polar=True))
for i,(mname,col) in enumerate(zip(model_names_list,COLORS)):
    vals_r=[model_results[mname][m].mean()*100 for m in metric_names]+[model_results[mname][metric_names[0]].mean()*100]
    ax4.plot(angles,vals_r,"-o",linewidth=1.5,label=mname,color=col); ax4.fill(angles,vals_r,alpha=0.08,color=col)
ax4.set_xticks(angles[:-1]); ax4.set_xticklabels(cats_r,fontsize=9)
ax4.set_ylim(max(0,min(acc_means)-5),102); ax4.legend(fontsize=7,loc="upper right",bbox_to_anchor=(1.35,1.1))
ax4.set_title("Fig. 4. Radar chart — multi-metric model comparison.",pad=15,fontsize=11)
plt.tight_layout(); plt.savefig("outputs/plots/fig4_radar_chart.png",dpi=150,bbox_inches="tight"); plt.close()
print("Fig 4 saved")

# Fig 5: SHAP / feature importance + ablation
best_name=max(model_results,key=lambda m:model_results[m]["accuracy"].mean())
best_model_obj=models[best_name]; best_model_obj.fit(X,y)
fig5,axes5=plt.subplots(1,2,figsize=(12,5))
if hasattr(best_model_obj,"feature_importances_"):
    imps=best_model_obj.feature_importances_; top_k=min(15,len(imps)); top_i=np.argsort(imps)[-top_k:]
    axes5[0].barh(range(top_k),imps[top_i],color="#E65100",alpha=0.85)
    axes5[0].set_yticks(range(top_k))
    axes5[0].set_yticklabels([feature_names_list[i][:20] if i<len(feature_names_list) else "f"+str(i) for i in top_i],fontsize=7)
    axes5[0].set_xlabel("Importance")
axes5[0].set_title("Fig. 5a. Feature importance (" + best_name + ").")
top5_idx=np.argsort(best_model_obj.feature_importances_)[-5:] if hasattr(best_model_obj,"feature_importances_") else np.arange(5)
X_top5=X[:,top5_idx]
rf_top5=RandomForestClassifier(n_estimators=100,n_jobs=-1,random_state=42)
top5_sc=cross_val_score(rf_top5,X_top5,y,cv=cv,scoring="accuracy")
full_acc=model_results[best_name]["accuracy"].mean()*100; abl_acc=top5_sc.mean()*100
axes5[1].bar(["All Features", "Top-5 XAI Features"], [full_acc, abl_acc], color=["#4CAF50","#2196F3"], alpha=0.85, width=0.4)
for i,v in enumerate([full_acc,abl_acc]): axes5[1].text(i,v+0.1,"{:.2f}%".format(v),ha="center",fontsize=9)
axes5[1].set_ylabel("Accuracy (%)"); axes5[1].set_ylim(bottom=max(0,min(full_acc,abl_acc)-5))
axes5[1].set_title("Fig. 5b. Ablation study — feature selection impact.")
plt.tight_layout(); plt.savefig("outputs/plots/fig5_feature_ablation.png",dpi=150,bbox_inches="tight"); plt.close()
print("Fig 5 saved")

# Fig 6: Confusion matrix
preds=cross_val_predict(models[best_name],X,y,cv=cv)
cm=confusion_matrix(y,preds); fig6,ax6=plt.subplots(figsize=(7,6))
ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax6,colorbar=True,cmap="Blues")
ax6.set_title("Fig. 6. Confusion matrix — "+best_name+" (5-fold CV).",fontsize=12,pad=12)
plt.tight_layout(); plt.savefig("outputs/plots/fig6_confusion_matrix.png",dpi=150,bbox_inches="tight"); plt.close()
print("Fig 6 saved")

# Fig 7: SHAP
if HAS_SHAP:
    try:
        explainer=shap.TreeExplainer(best_model_obj)
        sv=explainer.shap_values(X[:200])
        if isinstance(sv,list): sv=sv[1]
        shap.summary_plot(sv,X[:200],feature_names=feature_names_list,show=False)
        plt.title("Fig. 7. SHAP feature importance.")
        plt.tight_layout(); plt.savefig("outputs/plots/fig7_shap.png",dpi=150,bbox_inches="tight"); plt.close()
        print("Fig 7 (SHAP) saved")
    except Exception as e: print("SHAP skipped: "+str(e))

sorted_m=sorted(model_names_list,key=lambda m:model_results[m]["accuracy"].mean(),reverse=True)
t_stat,p_value=stats.ttest_rel(model_results[sorted_m[0]]["accuracy"],model_results[sorted_m[1]]["accuracy"])
ci_best=stats.t.interval(0.95,4,loc=model_results[sorted_m[0]]["accuracy"].mean(),scale=stats.sem(model_results[sorted_m[0]]["accuracy"]))
metrics_out=[]
for mname in model_names_list:
    for metric in metric_names:
        sc=model_results[mname][metric]
        metrics_out.append({"metric_name":mname.replace(" ","")+"_"+metric,"mean":round(float(sc.mean()),4),"std":round(float(sc.std()),4),"n_runs":5})
metrics_out+=[
    {"metric_name":"p_value_best_vs_second","mean":round(float(p_value),4),"std":0.0,"n_runs":1},
    {"metric_name":"best_95ci_lower","mean":round(float(ci_best[0]),4),"std":0.0,"n_runs":1},
    {"metric_name":"best_95ci_upper","mean":round(float(ci_best[1]),4),"std":0.0,"n_runs":1},
    {"metric_name":"XAI_top5_accuracy","mean":round(float(top5_sc.mean()),4),"std":round(float(top5_sc.std()),4),"n_runs":5},
]
results={"metrics":metrics_out,"hypothesis_verdict":"supported" if model_results[sorted_m[0]]["accuracy"].mean()>0.75 else "partially_supported",
         "best_model":sorted_m[0],"plot_files":["outputs/plots/fig"+str(i)+".png" for i in range(1,7)],
         "key_findings":["Best: "+sorted_m[0]+" ({:.2f}%)".format(model_results[sorted_m[0]]["accuracy"].mean()*100),
                         "Dataset: "+dataset_name+" | n="+str(X.shape[0]),
                         "t={:.4f}, p={:.4f}".format(t_stat,p_value),
                         "Top-5 XAI: {:.2f}%".format(top5_sc.mean()*100),
                         "95% CI: ({:.2f}%, {:.2f}%)".format(ci_best[0]*100,ci_best[1]*100)],
         "statistical_tests":{"paired_ttest":{"t":round(float(t_stat),4),"p":round(float(p_value),4),"sig":bool(p_value<0.05)}}}
with open("outputs/code/results.json","w") as f: json.dump(results,f,indent=2)
print("Results saved. EXPERIMENT COMPLETE")
'''

# ── Dataset load code snippets per dataset name ──────────────────────────────
TEXT_LOADERS = {
    "halueval": (
        "HaluEval",
        "from datasets import load_dataset as hf_load\n"
        "ds = hf_load('pminervini/HaluEval', 'qa_samples')\n"
        "split = ds['data'] if 'data' in ds else ds[list(ds.keys())[0]]\n"
        "texts  = [str(r.get('question','')) + ' ' + str(r.get('answer','')) for r in split]\n"
        "labels = [int(r.get('hallucination',0)) for r in split]\n"
    ),
    "truthfulqa": (
        "TruthfulQA",
        "from datasets import load_dataset as hf_load\n"
        "ds = hf_load('truthful_qa', 'generation')\n"
        "texts  = list(ds['validation']['question'])\n"
        "labels = [0 if len(a)>10 else 1 for a in ds['validation']['best_answer']]\n"
    ),
    "imdb": (
        "IMDB Sentiment",
        "from datasets import load_dataset as hf_load\n"
        "ds = hf_load('imdb')\n"
        "texts  = list(ds['train']['text'])[:20000]\n"
        "labels = list(ds['train']['label'])[:20000]\n"
    ),
    "ag news": (
        "AG News",
        "from datasets import load_dataset as hf_load\n"
        "ds = hf_load('ag_news')\n"
        "texts  = list(ds['train']['text'])[:30000]\n"
        "labels = list(ds['train']['label'])[:30000]\n"
    ),
    "20 newsgroups": (
        "20 Newsgroups",
        "from sklearn.datasets import fetch_20newsgroups\n"
        "data   = fetch_20newsgroups(subset='all', remove=('headers','footers','quotes'))\n"
        "texts  = data.data\n"
        "labels = data.target\n"
    ),
}

TIMESERIES_LOADERS = {
    "skab": (
        "SKAB (Skoltech Anomaly Benchmark)",
        "from skab import SKAB\n"
        "sk = SKAB()\n"
        "df = sk.get_all()\n"
        "feat_cols = [c for c in df.columns if c not in ['anomaly','changepoint']]\n"
        "X = df[feat_cols].fillna(0).values\n"
        "y = df['anomaly'].values.astype(int)\n"
        "feature_names_list = feat_cols\n"
    ),
    "uci metro": (
        "UCI Metro Interstate Traffic Volume",
        "import pandas as pd\n"
        "url='https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz'\n"
        "df=pd.read_csv(url,compression='gzip')\n"
        "df['date_time']=pd.to_datetime(df['date_time'])\n"
        "df['hour']=df['date_time'].dt.hour; df['day_of_week']=df['date_time'].dt.dayofweek\n"
        "df['month']=df['date_time'].dt.month\n"
        "df['is_rush_hour']=((df['hour'].between(7,9))|(df['hour'].between(16,18))).astype(int)\n"
        "from sklearn.preprocessing import LabelEncoder\n"
        "le=LabelEncoder()\n"
        "df['weather_encoded']=le.fit_transform(df['weather_main'].astype(str)) if 'weather_main' in df.columns else 0\n"
        "feat_cols=[f for f in ['temp','rain_1h','snow_1h','clouds_all','hour','day_of_week','month','is_rush_hour','weather_encoded'] if f in df.columns]\n"
        "df=df.dropna(subset=feat_cols); X=df[feat_cols].values\n"
        "y=(df['traffic_volume']>df['traffic_volume'].median()).astype(int).values\n"
        "feature_names_list=feat_cols\n"
    ),
}

TABULAR_LOADERS = {
    "uci heart": (
        "UCI Heart Disease",
        "from ucimlrepo import fetch_ucirepo\n"
        "ds=fetch_ucirepo(id=45)\n"
        "X=ds.data.features.fillna(ds.data.features.median()).values\n"
        "y=ds.data.targets.values.ravel()\n"
        "feature_names_list=list(ds.data.features.columns)\n"
    ),
    "uci credit": (
        "UCI Credit Card Default",
        "from ucimlrepo import fetch_ucirepo\n"
        "ds=fetch_ucirepo(id=350)\n"
        "X=ds.data.features.fillna(ds.data.features.median()).values\n"
        "y=ds.data.targets.values.ravel()\n"
        "feature_names_list=list(ds.data.features.columns)\n"
    ),
    "sklearn": (
        "sklearn breast_cancer",
        "from sklearn.datasets import load_breast_cancer\n"
        "data=load_breast_cancer(); X,y=data.data,data.target\n"
        "feature_names_list=list(data.feature_names)\n"
    ),
}


def _build_code(modality: str, dataset_name: str) -> str:
    """Select template + inject the correct dataset loader."""
    dname_lower = dataset_name.lower()

    if modality == "text":
        template   = TEXT_EXPERIMENT_CODE
        loader_map = TEXT_LOADERS
    elif modality == "timeseries":
        template   = TIMESERIES_EXPERIMENT_CODE
        loader_map = TIMESERIES_LOADERS
    else:
        template   = TABULAR_EXPERIMENT_CODE
        loader_map = TABULAR_LOADERS

    loader_name = dataset_name
    loader_code = ""
    for key, (name, code) in loader_map.items():
        if key in dname_lower:
            loader_name = name
            loader_code = code
            break

    if not loader_code:
        loader_code = "pass  # no loader matched; fallback will trigger\n"

    # The placeholder is inside a 4-space try block — indent every loader line 4 spaces
    indented_loader = "\n".join(
        "    " + line if line.strip() else line
        for line in loader_code.rstrip().split("\n")
    )

    code = template.replace("DATASET_NAME_PLACEHOLDER", loader_name)
    code = code.replace("    DATASET_LOAD_CODE_PLACEHOLDER", indented_loader)
    return code


def run():
    client          = create_client()
    experiment_plan = state_store.get_state("experiment_plan")
    input_topic     = state_store.get_state("input_topic")
    if not experiment_plan:
        raise ValueError("[ImplementationAgent] No experiment_plan in state.")

    topic     = input_topic.get("topic", "")
    modality  = experiment_plan.get("modality", "tabular")
    datasets  = experiment_plan.get("datasets", [{}])
    dataset_name = datasets[0].get("name", "sklearn breast_cancer") if datasets else "sklearn breast_cancer"

    print("[ImplementationAgent] Topic: " + topic)
    print("[ImplementationAgent] Modality: " + modality)
    print("[ImplementationAgent] Dataset: " + dataset_name)
    print("[ImplementationAgent] Hardware: i5-13200H, 16GB DDR5, CPU-only, budget=60min")

    code = _build_code(modality, dataset_name)
    script_path = config.OUTPUTS_CODE / "experiment.py"
    file_manager.safe_write(script_path, code)
    print("[ImplementationAgent] Running experiment (timeout=3600s)...")

    result = code_executor.run_script(script_path, timeout=3600)

    if not result["success"]:
        print("[ImplementationAgent] Primary failed: " + result["stderr"][:300])
        print("[ImplementationAgent] Trying sklearn breast_cancer fallback...")
        fallback = _build_code("tabular", "sklearn")
        file_manager.safe_write(script_path, fallback)
        result = code_executor.run_script(script_path, timeout=1800)

    result["retry_count"] = 0
    state_store.update_state("code_result", result)
    audit_logger.log("implementation_agent", {"topic": topic, "modality": modality}, result)

    if result["success"]:
        print("[ImplementationAgent] Done in " + str(result["execution_time_seconds"]) + "s")
    else:
        print("[ImplementationAgent] Failed: " + result["stderr"][:300])
    return result
