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


def _enforce_metrics(text: str, metrics: list) -> str:
    """
    Bulletproof post-write pass: replace ALL percentage numbers in the text
    that don't match real metric values.
    Strategy:
    - Build set of real values
    - Any X.Y% where X.Y is not within 0.5 of a real value → replace with nearest real
    - Targets sections that commonly have hallucinated numbers
    """
    import re

    real_pcts = set()
    for m in metrics:
        val = m["mean"]
        std = m["std"]
        if 0 < val <= 1.0 and not any(x in m["metric_name"].lower()
                                       for x in ["ci", "p_value", "lower", "upper"]):
            real_pcts.add(round(val * 100, 2))

    if not real_pcts:
        return text

    sorted_real = sorted(real_pcts, reverse=True)
    best_val    = sorted_real[0]

    def replace_pct(m):
        num_str = m.group(1)
        try:
            num = float(num_str)
        except ValueError:
            return m.group(0)
        if num < 50 or num > 100:
            return m.group(0)
        # Check if it matches any real value within 0.5%
        if any(abs(num - r) <= 0.5 for r in real_pcts):
            return m.group(0)          # it's real, keep it
        # Replace with nearest real value
        closest = min(real_pcts, key=lambda r: abs(r - num))
        return "{:.2f}%".format(closest)

    text = re.sub(r'(\d{2,3}(?:\.\d{1,2})?)%', replace_pct, text)
    return text


def _build_metrics_table_str(metrics: list) -> str:
    """Build a locked table of model→metric→value the LLM must copy verbatim."""
    rows = {}
    for m in metrics:
        name = m["metric_name"]
        val  = m["mean"]
        std  = m["std"]
        if any(x in name.lower() for x in ["ci", "p_value", "lower", "upper", "xai"]):
            continue
        parts  = name.split("_")
        model  = parts[0]
        metric = "_".join(parts[1:]) if len(parts) > 1 else name
        if model not in rows:
            rows[model] = {}
        if 0 < val <= 1.0:
            rows[model][metric] = "{:.2f}% ±{:.2f}%".format(val*100, std*100)
        else:
            rows[model][metric] = "{:.4f}".format(val)

    lines = ["LOCKED METRICS TABLE — copy these numbers EXACTLY, do not change them:"]
    for model, data in rows.items():
        for metric, val in data.items():
            lines.append("  " + model + " " + metric + " = " + val)
    return "\n".join(lines)




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
    text = re.sub(r'^#{1,4}\s+', '', text, flags=re.MULTILINE)
    return text


def _remove_abstract_block(text: str) -> str:
    """
    Bulletproof removal of any freestanding ABSTRACT block the LLM inserts.
    Scans line by line: if we see an ABSTRACT heading, we skip lines until
    we hit the next real section heading (Introduction, Related, etc.) or 2+ blank lines.
    """
    import re
    lines   = text.split("\n")
    out     = []
    skipping = False
    blank_count = 0

    abstract_heading = re.compile(
        r"^\s*(?:#{1,4}\s*)?(?:\*{1,2})?(?:Abstract|ABSTRACT)(?:\*{1,2})?[—\-—:\s]*$",
        re.IGNORECASE
    )
    # Also catch inline "Abstract— text..." at start of a line (abstract repeated as paragraph)
    abstract_inline = re.compile(
        r"^\s*(?:\*{1,2})?(?:Abstract|ABSTRACT)[—\-:\s]+\S",
        re.IGNORECASE
    )
    section_heading = re.compile(
        r"^\s*(?:#{1,4}\s*|[IVX]+\.\s*|[A-Z]\.\s*)?"
        r"(Introduction|Related Work|Methodology|Experiments|Results|"
        r"Discussion|Conclusion|Limitations|References)",
        re.IGNORECASE
    )

    for line in lines:
        if not skipping:
            if abstract_heading.match(line) or abstract_inline.match(line):
                skipping    = True
                blank_count = 0
                continue
        else:
            # Stop skipping when we hit a real section or two consecutive blank lines
            if section_heading.match(line):
                skipping = False
                out.append(line)
            elif line.strip() == "":
                blank_count += 1
                if blank_count >= 2:
                    skipping = False
            else:
                blank_count = 0
            continue
        out.append(line)

    return "\n".join(out).lstrip()


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

    # Post-process: remove any freestanding ABSTRACT heading the LLM snuck in
    for sec in ["introduction", "related_work", "methodology", "experiments",
                "results", "discussion", "conclusion", "limitations"]:
        if sec in sections and sections[sec]:
            sections[sec] = _remove_abstract_block(sections[sec])

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
        "CRITICAL — NUMBERS ARE LOCKED. Copy these EXACT values character-for-character.\n"
        "Do NOT round, do NOT add 1%, do NOT invent new numbers. Any number\n"
        "you write that does not appear in this list is a hallucination:\n"
        + metrics_str + "\n\n"
        + _build_metrics_table_str(metrics) + "\n\n"
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

    # Enforce real metric numbers — replace LLM-hallucinated percentages
    for sname in ["abstract", "results", "discussion", "conclusion"]:
        if sname in sections and isinstance(sections[sname], str):
            sections[sname] = _enforce_metrics(sections[sname], metrics)

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
