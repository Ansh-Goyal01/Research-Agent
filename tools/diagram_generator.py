import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path
import config


COLORS = {
    "input":      "#1565C0",
    "process":    "#2E7D32",
    "model":      "#6A1B9A",
    "output":     "#C62828",
    "xai":        "#E65100",
    "arrow":      "#37474F",
    "background": "#FAFAFA",
    "box_bg":     "white",
    "text":       "white",
    "subtitle":   "#555555"
}


def _draw_box(ax, x, y, w, h, label, sublabel="", color="#1565C0", fontsize=9):
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.02",
        facecolor=color,
        edgecolor="white",
        linewidth=1.5,
        zorder=3
    )
    ax.add_patch(box)
    ax.text(x, y + (0.015 if sublabel else 0), label,
            ha="center", va="center",
            fontsize=fontsize, fontweight="bold",
            color="white", zorder=4)
    if sublabel:
        ax.text(x, y - 0.025, sublabel,
                ha="center", va="center",
                fontsize=fontsize - 1.5,
                color="rgba(255,255,255,0.85)" if False else "#ffffffcc",
                zorder=4, style="italic")


def _arrow(ax, x1, y1, x2, y2, label=""):
    ax.annotate("",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=COLORS["arrow"],
            lw=1.8,
            mutation_scale=14
        ),
        zorder=2
    )
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx+0.01, my, label,
                fontsize=7, color=COLORS["arrow"],
                ha="left", va="center",
                style="italic", zorder=5)


def generate_architecture(topic, domain, method_acronym, methodology_text, output_path=None):
    if output_path is None:
        output_path = config.OUTPUTS_PLOTS / "fig0_architecture.png"

    topic_lower = topic.lower() + " " + domain.lower()

    if any(w in topic_lower for w in ["traffic", "congestion", "transport", "vehicle"]):
        _draw_traffic_architecture(topic, method_acronym, output_path)
    elif any(w in topic_lower for w in ["medical", "clinical", "health", "cancer", "image"]):
        _draw_medical_architecture(topic, method_acronym, output_path)
    elif any(w in topic_lower for w in ["industrial", "iot", "sensor", "maintenance"]):
        _draw_industrial_architecture(topic, method_acronym, output_path)
    elif any(w in topic_lower for w in ["nlp", "text", "language", "sentiment"]):
        _draw_nlp_architecture(topic, method_acronym, output_path)
    elif any(w in topic_lower for w in ["conformal", "uncertainty", "probabilistic"]):
        _draw_uncertainty_architecture(topic, method_acronym, output_path)
    else:
        _draw_generic_architecture(topic, method_acronym, output_path)

    print("[DiagramGenerator] Architecture saved: " + str(output_path))
    return output_path


def _draw_traffic_architecture(topic, acronym, output_path):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(COLORS["background"])
    fig.patch.set_facecolor(COLORS["background"])

    title = (acronym + ": " if acronym else "") + "System Architecture"
    ax.text(0.5, 0.94, title, ha="center", va="center",
            fontsize=14, fontweight="bold", color="#1a1a2e")
    ax.text(0.5, 0.89, topic, ha="center", va="center",
            fontsize=9, color=COLORS["subtitle"], style="italic")

    ax.add_patch(FancyBboxPatch((0.02, 0.08), 0.96, 0.77,
        boxstyle="round,pad=0.01", facecolor="white",
        edgecolor="#e0e0e0", linewidth=1.5, zorder=0))

    _draw_box(ax, 0.12, 0.65, 0.16, 0.10, "Data Sources",
              "Traffic sensors\nGPS / IoT", COLORS["input"])
    _draw_box(ax, 0.12, 0.45, 0.16, 0.10, "Historical\nData",
              "Time-series\nrecords", COLORS["input"])
    _draw_box(ax, 0.12, 0.25, 0.16, 0.10, "Real-time\nFeed",
              "Live traffic\nAPI", COLORS["input"])

    _draw_box(ax, 0.35, 0.55, 0.16, 0.14, "Preprocessing",
              "Normalization\nFeature eng.\nMissing values", COLORS["process"])

    _draw_box(ax, 0.58, 0.72, 0.14, 0.09, "Random\nForest", "", COLORS["model"])
    _draw_box(ax, 0.58, 0.55, 0.14, 0.09, "Gradient\nBoosting", "", COLORS["model"])
    _draw_box(ax, 0.58, 0.38, 0.14, 0.09, "XGBoost", "", COLORS["model"])
    _draw_box(ax, 0.58, 0.21, 0.14, 0.09, "LightGBM", "", COLORS["model"])

    _draw_box(ax, 0.78, 0.65, 0.14, 0.10, "XAI Module",
              "SHAP values\nFeature importance", COLORS["xai"])
    _draw_box(ax, 0.78, 0.40, 0.14, 0.10, "Model\nSelector",
              "Best model\nby CV score", COLORS["process"])

    _draw_box(ax, 0.93, 0.55, 0.10, 0.20, "Output",
              "Congestion\nlevel +\nExplanation", COLORS["output"])

    _arrow(ax, 0.20, 0.65, 0.27, 0.60)
    _arrow(ax, 0.20, 0.45, 0.27, 0.52)
    _arrow(ax, 0.20, 0.25, 0.27, 0.48)
    _arrow(ax, 0.43, 0.62, 0.51, 0.72)
    _arrow(ax, 0.43, 0.58, 0.51, 0.55)
    _arrow(ax, 0.43, 0.52, 0.51, 0.38)
    _arrow(ax, 0.43, 0.48, 0.51, 0.21)
    _arrow(ax, 0.65, 0.72, 0.71, 0.68)
    _arrow(ax, 0.65, 0.55, 0.71, 0.62)
    _arrow(ax, 0.65, 0.38, 0.71, 0.42)
    _arrow(ax, 0.65, 0.21, 0.71, 0.38)
    _arrow(ax, 0.85, 0.55, 0.88, 0.58)
    _arrow(ax, 0.85, 0.40, 0.88, 0.50)

    legend_items = [
        mpatches.Patch(color=COLORS["input"],   label="Data Input"),
        mpatches.Patch(color=COLORS["process"], label="Processing"),
        mpatches.Patch(color=COLORS["model"],   label="ML Models"),
        mpatches.Patch(color=COLORS["xai"],     label="XAI Module"),
        mpatches.Patch(color=COLORS["output"],  label="Output"),
    ]
    ax.legend(handles=legend_items, loc="lower right",
              framealpha=0.9, fontsize=8, ncol=5,
              bbox_to_anchor=(0.99, 0.09))

    ax.text(0.5, 0.04,
            "Fig. 1. Proposed system architecture for " + topic[:60],
            ha="center", va="center", fontsize=8,
            color=COLORS["subtitle"], style="italic")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight",
                facecolor=COLORS["background"])
    plt.close()


def _draw_medical_architecture(topic, acronym, output_path):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(COLORS["background"])
    fig.patch.set_facecolor(COLORS["background"])

    title = (acronym + ": " if acronym else "") + "System Architecture"
    ax.text(0.5, 0.94, title, ha="center", va="center",
            fontsize=14, fontweight="bold", color="#1a1a2e")
    ax.text(0.5, 0.89, topic, ha="center", va="center",
            fontsize=9, color=COLORS["subtitle"], style="italic")

    ax.add_patch(FancyBboxPatch((0.02, 0.08), 0.96, 0.77,
        boxstyle="round,pad=0.01", facecolor="white",
        edgecolor="#e0e0e0", linewidth=1.5, zorder=0))

    _draw_box(ax, 0.12, 0.60, 0.16, 0.12, "Medical\nDataset",
              "Clinical records\nImages / signals", COLORS["input"])

    _draw_box(ax, 0.32, 0.72, 0.14, 0.09, "Data\nCleaning",
              "Missing values\nNormalization", COLORS["process"])
    _draw_box(ax, 0.32, 0.48, 0.14, 0.09, "Feature\nExtraction",
              "Domain features\nStatistics", COLORS["process"])

    _draw_box(ax, 0.52, 0.72, 0.14, 0.09, "Classifier\nTraining",
              "RF / GB / XGB\n5-fold CV", COLORS["model"])
    _draw_box(ax, 0.52, 0.48, 0.14, 0.09, "Conformal\nPrediction",
              "Coverage\nguarantee", COLORS["model"])

    _draw_box(ax, 0.72, 0.65, 0.14, 0.12, "XAI\nExplainability",
              "Feature importance\nSHAP analysis", COLORS["xai"])

    _draw_box(ax, 0.90, 0.60, 0.12, 0.18, "Clinical\nOutput",
              "Prediction +\nConfidence +\nExplanation", COLORS["output"])

    _arrow(ax, 0.20, 0.65, 0.25, 0.72)
    _arrow(ax, 0.20, 0.55, 0.25, 0.48)
    _arrow(ax, 0.39, 0.72, 0.45, 0.72)
    _arrow(ax, 0.39, 0.48, 0.45, 0.52)
    _arrow(ax, 0.59, 0.72, 0.65, 0.70)
    _arrow(ax, 0.59, 0.48, 0.65, 0.60)
    _arrow(ax, 0.79, 0.65, 0.84, 0.65)

    legend_items = [
        mpatches.Patch(color=COLORS["input"],   label="Data Input"),
        mpatches.Patch(color=COLORS["process"], label="Preprocessing"),
        mpatches.Patch(color=COLORS["model"],   label="ML Models"),
        mpatches.Patch(color=COLORS["xai"],     label="XAI"),
        mpatches.Patch(color=COLORS["output"],  label="Output"),
    ]
    ax.legend(handles=legend_items, loc="lower right",
              framealpha=0.9, fontsize=8, ncol=5,
              bbox_to_anchor=(0.99, 0.09))

    ax.text(0.5, 0.04,
            "Fig. 1. Proposed system architecture for " + topic[:60],
            ha="center", va="center", fontsize=8,
            color=COLORS["subtitle"], style="italic")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight",
                facecolor=COLORS["background"])
    plt.close()


def _draw_industrial_architecture(topic, acronym, output_path):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(COLORS["background"])
    fig.patch.set_facecolor(COLORS["background"])

    title = (acronym + ": " if acronym else "") + "System Architecture"
    ax.text(0.5, 0.94, title, ha="center", fontsize=14,
            fontweight="bold", color="#1a1a2e")
    ax.text(0.5, 0.89, topic, ha="center", fontsize=9,
            color=COLORS["subtitle"], style="italic")

    ax.add_patch(FancyBboxPatch((0.02, 0.08), 0.96, 0.77,
        boxstyle="round,pad=0.01", facecolor="white",
        edgecolor="#e0e0e0", linewidth=1.5, zorder=0))

    _draw_box(ax, 0.11, 0.70, 0.14, 0.09, "IoT\nSensors",
              "Vibration\nTemp / Pressure", COLORS["input"])
    _draw_box(ax, 0.11, 0.50, 0.14, 0.09, "SCADA\nSystem",
              "Equipment logs\nAlarms", COLORS["input"])
    _draw_box(ax, 0.11, 0.30, 0.14, 0.09, "Historical\nRecords",
              "Maintenance\nhistory", COLORS["input"])

    _draw_box(ax, 0.30, 0.55, 0.14, 0.14, "Signal\nProcessing",
              "Denoising\nFeature eng.\nWindowing", COLORS["process"])

    _draw_box(ax, 0.50, 0.72, 0.14, 0.09, "Anomaly\nDetection",
              "Isolation Forest\nLOF", COLORS["model"])
    _draw_box(ax, 0.50, 0.55, 0.14, 0.09, "Failure\nClassifier",
              "XGBoost\nRandom Forest", COLORS["model"])
    _draw_box(ax, 0.50, 0.38, 0.14, 0.09, "RUL\nPredictor",
              "Remaining useful\nlife estimation", COLORS["model"])

    _draw_box(ax, 0.70, 0.65, 0.14, 0.12, "XAI\nModule",
              "SHAP values\nFeature importance\nConformal bounds", COLORS["xai"])

    _draw_box(ax, 0.88, 0.55, 0.12, 0.22, "Maintenance\nDecision",
              "Alert level\nRoot cause\nAction plan", COLORS["output"])

    _arrow(ax, 0.18, 0.70, 0.23, 0.62)
    _arrow(ax, 0.18, 0.50, 0.23, 0.55)
    _arrow(ax, 0.18, 0.30, 0.23, 0.48)
    _arrow(ax, 0.37, 0.62, 0.43, 0.72)
    _arrow(ax, 0.37, 0.55, 0.43, 0.55)
    _arrow(ax, 0.37, 0.48, 0.43, 0.38)
    _arrow(ax, 0.57, 0.72, 0.63, 0.70)
    _arrow(ax, 0.57, 0.55, 0.63, 0.65)
    _arrow(ax, 0.57, 0.38, 0.63, 0.60)
    _arrow(ax, 0.77, 0.65, 0.82, 0.62)

    legend_items = [
        mpatches.Patch(color=COLORS["input"],   label="Data Sources"),
        mpatches.Patch(color=COLORS["process"], label="Processing"),
        mpatches.Patch(color=COLORS["model"],   label="ML Models"),
        mpatches.Patch(color=COLORS["xai"],     label="XAI"),
        mpatches.Patch(color=COLORS["output"],  label="Decision"),
    ]
    ax.legend(handles=legend_items, loc="lower right",
              framealpha=0.9, fontsize=8, ncol=5,
              bbox_to_anchor=(0.99, 0.09))

    ax.text(0.5, 0.04,
            "Fig. 1. Proposed system architecture for " + topic[:60],
            ha="center", fontsize=8,
            color=COLORS["subtitle"], style="italic")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight",
                facecolor=COLORS["background"])
    plt.close()


def _draw_uncertainty_architecture(topic, acronym, output_path):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(COLORS["background"])
    fig.patch.set_facecolor(COLORS["background"])

    title = (acronym + ": " if acronym else "") + "System Architecture"
    ax.text(0.5, 0.94, title, ha="center", fontsize=14,
            fontweight="bold", color="#1a1a2e")
    ax.text(0.5, 0.89, topic, ha="center", fontsize=9,
            color=COLORS["subtitle"], style="italic")

    ax.add_patch(FancyBboxPatch((0.02, 0.08), 0.96, 0.77,
        boxstyle="round,pad=0.01", facecolor="white",
        edgecolor="#e0e0e0", linewidth=1.5, zorder=0))

    _draw_box(ax, 0.12, 0.60, 0.16, 0.14, "Input Data",
              "Training set\nCalibration set\nTest set", COLORS["input"])

    _draw_box(ax, 0.33, 0.72, 0.14, 0.09, "Base\nClassifier",
              "RF / XGB\ntraining", COLORS["model"])
    _draw_box(ax, 0.33, 0.48, 0.14, 0.09, "Nonconformity\nScores",
              "Score function\ncalibration", COLORS["process"])

    _draw_box(ax, 0.54, 0.72, 0.14, 0.09, "Conformal\nPredictor",
              "Coverage\nguarantee 1-α", COLORS["model"])
    _draw_box(ax, 0.54, 0.48, 0.14, 0.09, "Prediction\nSets",
              "Valid intervals\np-values", COLORS["process"])

    _draw_box(ax, 0.74, 0.65, 0.14, 0.12, "XAI\nIntegration",
              "Feature importance\nUncertainty maps\nSHAP", COLORS["xai"])

    _draw_box(ax, 0.92, 0.60, 0.11, 0.20, "Output",
              "Prediction +\nConfidence\nset +\nExplanation", COLORS["output"])

    _arrow(ax, 0.20, 0.65, 0.26, 0.72)
    _arrow(ax, 0.20, 0.55, 0.26, 0.48)
    _arrow(ax, 0.40, 0.72, 0.47, 0.72)
    _arrow(ax, 0.40, 0.48, 0.47, 0.52)
    _arrow(ax, 0.61, 0.72, 0.67, 0.70)
    _arrow(ax, 0.61, 0.48, 0.67, 0.60)
    _arrow(ax, 0.81, 0.65, 0.87, 0.65)

    legend_items = [
        mpatches.Patch(color=COLORS["input"],   label="Data"),
        mpatches.Patch(color=COLORS["model"],   label="ML Models"),
        mpatches.Patch(color=COLORS["process"], label="Conformal"),
        mpatches.Patch(color=COLORS["xai"],     label="XAI"),
        mpatches.Patch(color=COLORS["output"],  label="Output"),
    ]
    ax.legend(handles=legend_items, loc="lower right",
              framealpha=0.9, fontsize=8, ncol=5,
              bbox_to_anchor=(0.99, 0.09))

    ax.text(0.5, 0.04,
            "Fig. 1. Proposed system architecture for " + topic[:60],
            ha="center", fontsize=8,
            color=COLORS["subtitle"], style="italic")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight",
                facecolor=COLORS["background"])
    plt.close()


def _draw_nlp_architecture(topic, acronym, output_path):
    _draw_generic_architecture(topic, acronym, output_path)


def _draw_generic_architecture(topic, acronym, output_path):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(COLORS["background"])
    fig.patch.set_facecolor(COLORS["background"])

    title = (acronym + ": " if acronym else "") + "System Architecture"
    ax.text(0.5, 0.94, title, ha="center", fontsize=14,
            fontweight="bold", color="#1a1a2e")
    ax.text(0.5, 0.89, topic, ha="center", fontsize=9,
            color=COLORS["subtitle"], style="italic")

    ax.add_patch(FancyBboxPatch((0.02, 0.08), 0.96, 0.77,
        boxstyle="round,pad=0.01", facecolor="white",
        edgecolor="#e0e0e0", linewidth=1.5, zorder=0))

    _draw_box(ax, 0.12, 0.60, 0.16, 0.16, "Raw Data",
              "Dataset\ncollection\nvalidation", COLORS["input"])

    _draw_box(ax, 0.32, 0.72, 0.14, 0.10, "Preprocessing",
              "Cleaning\nNormalization", COLORS["process"])
    _draw_box(ax, 0.32, 0.48, 0.14, 0.10, "Feature\nEngineering",
              "Extraction\nSelection", COLORS["process"])

    _draw_box(ax, 0.52, 0.72, 0.14, 0.10, "Model\nTraining",
              "RF / XGB / GB\n5-fold CV", COLORS["model"])
    _draw_box(ax, 0.52, 0.48, 0.14, 0.10, "Hyperparameter\nTuning",
              "Grid search\nOptimization", COLORS["model"])

    _draw_box(ax, 0.72, 0.65, 0.14, 0.14, "XAI\nModule",
              "Feature importance\nSHAP values\nInterpretability", COLORS["xai"])

    _draw_box(ax, 0.91, 0.60, 0.12, 0.20, "Output",
              "Prediction\n+ Explanation\n+ Confidence", COLORS["output"])

    _arrow(ax, 0.20, 0.65, 0.25, 0.72)
    _arrow(ax, 0.20, 0.55, 0.25, 0.48)
    _arrow(ax, 0.39, 0.72, 0.45, 0.72)
    _arrow(ax, 0.39, 0.48, 0.45, 0.52)
    _arrow(ax, 0.59, 0.72, 0.65, 0.70)
    _arrow(ax, 0.59, 0.52, 0.65, 0.60)
    _arrow(ax, 0.79, 0.65, 0.85, 0.65)

    legend_items = [
        mpatches.Patch(color=COLORS["input"],   label="Data Input"),
        mpatches.Patch(color=COLORS["process"], label="Processing"),
        mpatches.Patch(color=COLORS["model"],   label="ML Models"),
        mpatches.Patch(color=COLORS["xai"],     label="XAI Module"),
        mpatches.Patch(color=COLORS["output"],  label="Output"),
    ]
    ax.legend(handles=legend_items, loc="lower right",
              framealpha=0.9, fontsize=8, ncol=5,
              bbox_to_anchor=(0.99, 0.09))

    ax.text(0.5, 0.04,
            "Fig. 1. Proposed system architecture for " + topic[:60],
            ha="center", fontsize=8,
            color=COLORS["subtitle"], style="italic")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight",
                facecolor=COLORS["background"])
    plt.close()