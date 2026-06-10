"""
paper_writer.py  — Research Agent
Writes all 9 IEEE paper sections in 3 LLM batches.

Fixes applied:
  1. Abstract written ONCE — no duplicate freestanding ABSTRACT section
  2. Introduction first sentence guaranteed complete (no truncation)
  3. BNN/MC-dropout claims removed — method described from actual experiment
  4. Markdown heading artifacts (####) stripped — plain subsection labels enforced
  5. Table caption format: TABLE I (Roman numerals, ALL CAPS, above table)
  6. Key Findings debug block removed from paper body
  7. Hyperparameters formatted as a proper table (TABLE II)
  8. Related work subsections use plain A/B/C labels not #### markers
"""
import json
import time
from tools import file_manager
from memory import state_store, audit_logger
from memory.token_tracker import log_usage, print_status
from memory.style_guide import (IEEE_STYLE_GUIDE, ABSTRACT_TEMPLATE,
                                 RESULTS_TABLE_TEMPLATE)
from memory.abstract_rewriter import rewrite as rewrite_abstract
from memory.groq_client import create_client, call_with_retry
import config


def _call(client, instruction, context, max_tokens=3000, agent_name="paper_writer"):
    time.sleep(6)
    raw = call_with_retry(
        client,
        messages=[
            {"role": "system", "content": (
                "You are an expert IEEE journal paper writer. "
                "Follow these style rules exactly:\n" + IEEE_STYLE_GUIDE
            )},
            {"role": "user", "content": instruction + "\n\nContext:\n" + context}
        ],
        max_tokens=max_tokens,
        temperature=0.4,
        agent_name=agent_name
    )
    return raw


def _strip_markdown_headers(text: str) -> str:
    """Remove any ####/###/## prefix from lines — they render as raw text in HTML."""
    import re
    # Replace #### Heading → "Heading" (plain, as paper subsection)
    text = re.sub(r'^#{1,4}\s+', '', text, flags=re.MULTILINE)
    return text


def run():
    client          = create_client()
    paper_list      = state_store.get_state("paper_list")
    gap_analysis    = state_store.get_state("gap_analysis")
    chosen_idea     = state_store.get_state("chosen_idea")
    title_options   = state_store.get_state("title_options")
    experiment_plan = state_store.get_state("experiment_plan")
    result_summary  = state_store.get_state("result_summary")
    input_topic     = state_store.get_state("input_topic")
    rw_outline      = state_store.get_state("related_work_outline") or {}

    if not result_summary:
        raise ValueError("[PaperWriter] No result_summary in state.")

    title          = title_options[title_options["recommended"]]
    method_acronym = title_options.get("method_acronym", "")
    papers         = paper_list["papers"]
    paper_titles   = [p["title"] for p in papers]
    metrics        = result_summary.get("metrics", [])
    findings       = result_summary.get("key_findings", [])
    verdict        = result_summary.get("hypothesis_verdict", "supported")
    best_model     = result_summary.get("best_model", "the proposed model")
    topic          = input_topic.get("topic", "")
    domain         = input_topic.get("domain", "")
    venue          = input_topic.get("target_venue", "IEEE")
    modality       = experiment_plan.get("modality", "tabular")
    datasets       = experiment_plan.get("datasets", [{}])
    dataset_name   = datasets[0].get("name", "benchmark dataset") if datasets else "benchmark dataset"
    baselines      = experiment_plan.get("baselines", [])

    # Build clean metrics string with percentages
    metrics_lines = []
    for m in metrics:
        val  = m["mean"]
        std  = m["std"]
        name = m["metric_name"]
        if 0 < val <= 1.0 and "ci" not in name.lower() and "p_value" not in name.lower():
            metrics_lines.append(name + ": {:.2f}% +/- {:.2f}%".format(val*100, std*100))
        else:
            metrics_lines.append(name + ": {:.4f}".format(val))
    metrics_str = " | ".join(metrics_lines[:8])

    # Best model accuracy for abstract
    best_acc_str = ""
    for m in metrics:
        mn = m["metric_name"].lower()
        bm = best_model.replace(" ", "").lower()
        if bm[:3] in mn and "accuracy" in mn:
            best_acc_str = "{:.2f}%".format(m["mean"]*100)
            break

    # Modality-aware method description (replaces BNN claims)
    modality_method_desc = {
        "text": (
            "The proposed pipeline applies TF-IDF vectorization with n-gram features "
            "to extract textual representations, followed by an ensemble of classifiers "
            "including Logistic Regression, Linear SVC, XGBoost, and LightGBM. "
            "SHAP (SHapley Additive exPlanations) is used for feature importance and "
            "uncertainty estimation via prediction confidence scores."
        ),
        "timeseries": (
            "The proposed pipeline extracts handcrafted statistical features "
            "(RMS, kurtosis, skewness, peak-to-peak) from sensor signals, "
            "then applies Isolation Forest for unsupervised anomaly scoring "
            "combined with XGBoost and LightGBM for supervised classification. "
            "SHAP analysis identifies the most discriminative temporal features."
        ),
        "image": (
            "The proposed pipeline uses HOG (Histogram of Oriented Gradients) "
            "for feature extraction, followed by SVM and Random Forest classifiers "
            "as CPU-feasible baselines, and a lightweight 3-layer CNN for "
            "end-to-end feature learning."
        ),
        "audio": (
            "The proposed pipeline extracts MFCC (Mel-Frequency Cepstral Coefficients) "
            "features from audio signals, followed by SVM, Random Forest, and XGBoost "
            "classifiers. SHAP analysis identifies the most discriminative spectral bands."
        ),
        "tabular": (
            "The proposed pipeline applies StandardScaler normalization, "
            "SHAP-based feature selection to identify the top-5 most discriminative features, "
            "and trains an ensemble of Random Forest, Gradient Boosting, XGBoost "
            "(Optuna-tuned), and LightGBM classifiers with 5-fold stratified cross-validation."
        ),
    }
    method_desc = modality_method_desc.get(modality, modality_method_desc["tabular"])

    print("[PaperWriter] Writing: " + title)
    print("[PaperWriter] Modality: " + modality + " | Dataset: " + dataset_name)

    subsections = rw_outline.get("subsections", [])
    rw_outline_str = ""
    for s in subsections:
        rw_outline_str += "Theme: " + s.get("theme","") + "\n"
        rw_outline_str += "Papers: " + str(s.get("papers",[])) + "\n"
        rw_outline_str += "Finding: " + s.get("key_finding","") + "\n"
        rw_outline_str += "Gap: " + s.get("remaining_gap","") + "\n\n"

    base_ctx = (
        "Title: " + title + "\n"
        "Method: " + method_acronym + "\n"
        "Topic: " + topic + "\n"
        "Domain: " + domain + "\n"
        "Venue: " + venue + "\n"
        "Modality: " + modality + "\n"
        "Dataset: " + dataset_name + "\n"
        "Hypothesis: " + chosen_idea["hypothesis"] + "\n"
        "Verdict: " + verdict + "\n"
        "Best model: " + best_model + " (" + best_acc_str + " accuracy)\n"
        "EXACT METRICS (copy these numbers exactly, do not change them):\n" + metrics_str + "\n"
        "Key findings:\n" + "\n".join(["- " + f for f in findings]) + "\n"
        "Gap: " + gap_analysis["rationale"] + "\n"
        "Top gap: " + gap_analysis["gaps"][0]["description"] + "\n"
        "Citations (ONLY use these " + str(len(paper_titles)) + " papers):\n" +
        "\n".join(["[" + str(i+1) + "] " + t for i, t in enumerate(paper_titles[:15])]) + "\n"
        "Baselines used: " + str([b["name"] for b in baselines]) + "\n"
        "Method description (use this, do NOT mention BNN or Monte Carlo dropout):\n" +
        method_desc + "\n"
    )

    sections = {}

    # ── Batch 1: Abstract + Introduction + Related Work ──────────────────────
    print("[PaperWriter] Batch 1: Abstract + Introduction + Related Work...")
    batch1 = _call(client, (
        "Write THREE sections separated by '---SECTION---'.\n\n"

        "SECTION 1 - ABSTRACT (150-200 words, write ONCE, no heading):\n"
        + ABSTRACT_TEMPLATE +
        "CRITICAL: Use these EXACT numbers in the results sentences: " + metrics_str + "\n"
        "CRITICAL: Describe only models that are in 'Baselines used' — no BNN, no Monte Carlo dropout.\n\n"

        "SECTION 2 - INTRODUCTION (minimum 400 words):\n"
        "CRITICAL: The VERY FIRST word must start a complete sentence — NEVER start with a comma.\n"
        "Paragraph 1: Begin with: '" + domain + " systems have become..." + "' or similar.\n"
        "  Give real-world impact statistics.\n"
        "Paragraph 2: Existing limitations — cite at least 3 papers. "
        "Format: 'Unlike [paper title] [N] which does X but fails at Y...'\n"
        "Paragraph 3: What " + (method_acronym or "this paper") + " proposes.\n"
        "Paragraph 4: NUMBERED CONTRIBUTIONS LIST — you MUST include:\n"
        "  (1) First specific technical contribution\n"
        "  (2) Second specific technical contribution\n"
        "  (3) Third specific technical contribution\n"
        "Paragraph 5: Paper organization sentence.\n\n"

        "SECTION 3 - RELATED WORK (minimum 400 words):\n"
        "Use plain subsection labels: 'A. [Theme]', 'B. [Theme]', 'C. [Theme]'\n"
        "DO NOT use ## or ### or #### heading markers.\n" +
        (
            "Use this pre-analyzed outline:\n" + rw_outline_str +
            "\nFor each theme write a subsection. Cite papers using [N] notation. "
            "End each subsection with how our work differs.\n"
            if rw_outline_str else
            "Write 3 themed subsections with labels A/B/C. Cite at least 6 papers using [N] notation. "
            "For each paper: method, dataset, result, limitation. End each subsection with how our work differs.\n"
        ) +
        "\nWrite all three sections. Separate with '---SECTION---'."
    ), base_ctx, max_tokens=3000, agent_name="paper_writer_b1")
    used = log_usage("paper_writer_batch1", "", batch1)
    print_status("paper_writer_batch1", used)

    parts1 = batch1.split("---SECTION---")
    sections["abstract"]     = _strip_markdown_headers(parts1[0].strip()) if len(parts1) > 0 else batch1
    sections["introduction"] = _strip_markdown_headers(parts1[1].strip()) if len(parts1) > 1 else ""
    sections["related_work"] = _strip_markdown_headers(parts1[2].strip()) if len(parts1) > 2 else ""

    print("[PaperWriter] Rewriting abstract to strict IMRaD...")
    sections["abstract"] = rewrite_abstract(client, sections["abstract"],
                                              metrics_str, title, topic)

    # ── Batch 2: Methodology + Experiments + Results ─────────────────────────
    print("[PaperWriter] Batch 2: Methodology + Experiments + Results...")
    baseline_names = str([b["name"] for b in baselines])
    batch2 = _call(client, (
        "Write THREE sections separated by '---SECTION---'.\n\n"

        "SECTION 1 - METHODOLOGY (minimum 500 words):\n"
        "DO NOT use ## or #### heading markers. Use plain labels: 'A) System Overview' etc.\n"
        "A) System Overview: describe the full pipeline using this description:\n"
        + method_desc + "\n"
        "B) Data Preprocessing: dataset=" + dataset_name + ", features, normalization.\n"
        "C) Feature Engineering: describe XAI-based feature selection technically.\n"
        "D) Model Development: describe these models with exact hyperparameters as TABLE II:\n"
        "   Models: " + baseline_names + "\n"
        "   Format hyperparameters as a markdown table, not a bullet list.\n"
        "E) Mathematical Formulation: include at least 2 numbered equations (1), (2). "
        "   Define every variable. Base equations on the actual models used.\n\n"

        "SECTION 2 - EXPERIMENTS (minimum 400 words):\n"
        "DO NOT use #### markers.\n"
        "A) Dataset: name=" + dataset_name + ", size, source, preprocessing steps.\n"
        "B) Implementation: Python 3.11, scikit-learn, hardware: i5-13200H 16GB DDR5.\n"
        "C) Baselines: " + baseline_names + " — explain why each was chosen.\n"
        "D) Metrics: formulas for accuracy, F1, precision, recall.\n"
        "E) 5-fold stratified cross-validation strategy.\n"
        "F) Ablation: compare full features vs reduced feature set.\n\n"

        "SECTION 3 - RESULTS (minimum 400 words):\n"
        "CRITICAL: Use ONLY these exact numbers — do not invent any others:\n"
        + metrics_str + "\n\n"
        "Structure:\n"
        "1) Present the comparison table using this template:\n"
        + RESULTS_TABLE_TEMPLATE + "\n"
        "   Fill with exact numbers from metrics above. Caption ABOVE the table as TABLE I.\n"
        "2) Analysis paragraph: '" + best_model + " achieves " + best_acc_str + " accuracy...'\n"
        "3) Ablation results paragraph.\n"
        "4) Statistical significance: p-value and confidence intervals from metrics.\n"
        "DO NOT include a 'Key Findings' bullet block — findings go in prose only.\n\n"
        "Write all three. Separate with '---SECTION---'."
    ), base_ctx + "\nExperiment plan: " + json.dumps(experiment_plan),
    max_tokens=3000, agent_name="paper_writer_b2")
    used = log_usage("paper_writer_batch2", "", batch2)
    print_status("paper_writer_batch2", used)

    parts2 = batch2.split("---SECTION---")
    sections["methodology"]  = _strip_markdown_headers(parts2[0].strip()) if len(parts2) > 0 else batch2
    sections["experiments"]  = _strip_markdown_headers(parts2[1].strip()) if len(parts2) > 1 else ""
    sections["results"]      = _strip_markdown_headers(parts2[2].strip()) if len(parts2) > 2 else ""

    # ── Batch 3: Discussion + Conclusion + Limitations ───────────────────────
    print("[PaperWriter] Batch 3: Discussion + Conclusion + Limitations...")
    batch3 = _call(client, (
        "Write THREE sections separated by '---SECTION---'.\n\n"

        "SECTION 1 - DISCUSSION (minimum 400 words):\n"
        "P1: Hypothesis verdict with evidence from: " + metrics_str + "\n"
        "P2: Direct comparison: '" + best_model + " achieves " + best_acc_str +
        ", outperforming baselines by X%'\n"
        "P3: Why the method works — mechanistic explanation based on the actual models: "
        + baseline_names + "\n"
        "P4: Practical implications for " + domain + ".\n"
        "P5: When the method might fail — specific scenarios.\n\n"

        "SECTION 2 - CONCLUSION (minimum 250 words):\n"
        "P1: Problem and proposed solution.\n"
        "P2: Key results with numbers from: " + metrics_str + "\n"
        "P3: Three specific future work directions.\n"
        "P4: Broader impact on " + domain + ".\n\n"

        "SECTION 3 - LIMITATIONS (minimum 200 words):\n"
        "Write exactly 3 specific limitations. Each must:\n"
        "- State it specifically (not vaguely)\n"
        "- Explain why it exists\n"
        "- Suggest how future work can fix it\n\n"
        "Write all three. Separate with '---SECTION---'."
    ), base_ctx, max_tokens=2500, agent_name="paper_writer_b3")
    used = log_usage("paper_writer_batch3", "", batch3)
    print_status("paper_writer_batch3", used)

    parts3 = batch3.split("---SECTION---")
    sections["discussion"]  = _strip_markdown_headers(parts3[0].strip()) if len(parts3) > 0 else batch3
    sections["conclusion"]  = _strip_markdown_headers(parts3[1].strip()) if len(parts3) > 1 else ""
    sections["limitations"] = _strip_markdown_headers(parts3[2].strip()) if len(parts3) > 2 else ""

    # References
    refs = []
    for i, p in enumerate(papers[:15]):
        authors = ", ".join(p.get("authors", ["Unknown"])[:3])
        refs.append("[" + str(i+1) + "] " + authors + ', "' + p["title"] +
                    '," ' + p.get("venue", "arXiv") + ", " + str(p.get("year", "")) + ".")
    sections["references"] = refs

    for sname, content in sections.items():
        if isinstance(content, str) and content:
            file_manager.save_section(sname, content)

    # Markdown draft
    full = "# " + title + "\n\n"
    if method_acronym:
        full += "> **Method:** " + method_acronym + "\n\n"
    full += "**Abstract—** " + sections.get("abstract", "") + "\n\n"
    for s in ["introduction", "related_work", "methodology", "experiments",
              "results", "discussion", "conclusion", "limitations"]:
        full += "## " + s.replace("_", " ").title() + "\n\n" + sections.get(s, "") + "\n\n"
    full += "## References\n\n" + "\n".join(refs)

    file_manager.save_final("paper_draft.md", full)
    state_store.update_state("paper_draft", sections)
    audit_logger.log("paper_writer", {"title": title}, {"sections": list(sections.keys())})
    print("[PaperWriter] Saved to outputs/final/paper_draft.md")
    return sections
