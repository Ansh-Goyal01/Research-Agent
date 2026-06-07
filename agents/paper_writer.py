import json
import time
from groq import Groq
from tools import file_manager
from memory import state_store, audit_logger
from memory.token_tracker import log_usage, print_status
from memory.style_guide import IEEE_STYLE_GUIDE, ABSTRACT_TEMPLATE, RESULTS_TABLE_TEMPLATE
from memory.abstract_rewriter import rewrite as rewrite_abstract
import config


def _call(client, instruction, context, max_tokens=3000):
    time.sleep(6)
    response = client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": (
                "You are an expert IEEE journal paper writer. "
                "Follow these style rules exactly:\n" +
                IEEE_STYLE_GUIDE
            )},
            {"role": "user", "content": instruction + "\n\nContext:\n" + context}
        ],
        max_tokens=max_tokens,
        temperature=0.4
    )
    return response.choices[0].message.content.strip()


def run():
    client = Groq(api_key=config.GROQ_API_KEY)
    paper_list = state_store.get_state("paper_list")
    gap_analysis = state_store.get_state("gap_analysis")
    chosen_idea = state_store.get_state("chosen_idea")
    title_options = state_store.get_state("title_options")
    experiment_plan = state_store.get_state("experiment_plan")
    result_summary = state_store.get_state("result_summary")
    input_topic = state_store.get_state("input_topic")
    related_work_outline = state_store.get_state("related_work_outline") or {}

    if not result_summary:
        raise ValueError("[PaperWriter] No result_summary in state.")

    title = title_options[title_options["recommended"]]
    method_acronym = title_options.get("method_acronym", "")
    papers = paper_list["papers"]
    paper_titles = [p["title"] for p in papers]
    metrics = result_summary.get("metrics", [])
    findings = result_summary.get("key_findings", [])
    verdict = result_summary.get("hypothesis_verdict", "supported")
    best_model = result_summary.get("best_model", "Random Forest")
    topic = input_topic.get("topic", "")
    domain = input_topic.get("domain", "")
    venue = input_topic.get("target_venue", "IEEE")

    metrics_str = " | ".join([
        m["metric_name"] + ": " + str(m["mean"]) + "±" + str(m["std"])
        for m in metrics
    ])

    print("[PaperWriter] Writing: " + title)
    if method_acronym:
        print("[PaperWriter] Method acronym: " + method_acronym)

    subsections = related_work_outline.get("subsections", [])
    rw_outline = ""
    if subsections:
        for s in subsections:
            rw_outline += "Theme: " + s["theme"] + "\n"
            rw_outline += "Papers: " + str(s["papers"]) + "\n"
            rw_outline += "Finding: " + s["key_finding"] + "\n"
            rw_outline += "Gap: " + s["remaining_gap"] + "\n\n"

    base_ctx = (
        "Title: " + title + "\n"
        "Method acronym: " + method_acronym + "\n"
        "Topic: " + topic + "\n"
        "Domain: " + domain + "\n"
        "Venue: " + venue + "\n"
        "Hypothesis: " + chosen_idea["hypothesis"] + "\n"
        "Verdict: " + verdict + "\n"
        "Best model: " + best_model + "\n"
        "Actual metrics (ONLY use these): " + metrics_str + "\n"
        "Key findings: " + json.dumps(findings) + "\n"
        "Gap addressed: " + gap_analysis["rationale"] + "\n"
        "Top gap: " + gap_analysis["gaps"][0]["description"] + "\n"
        "Supporting papers for gap: " + json.dumps(gap_analysis["gaps"][0].get("supporting_paper_titles", [])) + "\n"
        "Available citations (ONLY use these): " + json.dumps(paper_titles[:15]) + "\n"
        "Baselines: " + str([b["name"] for b in experiment_plan.get("baselines", [])]) + "\n"
        "Dataset: " + str([d["name"] for d in experiment_plan.get("datasets", [])]) + "\n"
    )

    sections = {}

    print("[PaperWriter] Batch 1: Abstract + Introduction + Related Work...")
    rw_instruction = (
        "Write the Related Work section using this pre-analyzed outline:\n" +
        rw_outline +
        "\nFor each theme write a subsection. For each paper: what they did, dataset, result, limitation. "
        "End each subsection with how our work differs. "
        "Use only papers from citations list. Minimum 400 words."
        if rw_outline else
        "Write a related work section with 3 themed subsections. "
        "Cite at least 6 papers. For each: method, dataset, result, limitation. "
        "End each subsection with how our work differs. Minimum 400 words."
    )

    batch1_instruction = (
        "Write THREE sections separated by '---SECTION---'.\n\n"
        "SECTION 1 - ABSTRACT:\n" + ABSTRACT_TEMPLATE + "\n\n"
        "SECTION 2 - INTRODUCTION (400+ words):\n"
        "Paragraph 1: Broader problem with real-world impact statistics\n"
        "Paragraph 2: Limitations of existing work — cite 3+ papers. "
        "For each: what it does AND what it fails to do. "
        "Directly reference gap: '" + gap_analysis["rationale"] + "'\n"
        "Paragraph 3: What " + (method_acronym or "this paper") + " proposes\n"
        "Paragraph 4: Numbered contributions (3-4 specific technical items)\n"
        "Paragraph 5: Paper organization\n\n"
        "SECTION 3 - RELATED WORK:\n" + rw_instruction + "\n\n"
        "Write all three. Separate with '---SECTION---'."
    )
    batch1 = _call(client, batch1_instruction, base_ctx, max_tokens=3000)
    used = log_usage("paper_writer_batch1", batch1_instruction, batch1)
    print_status("paper_writer_batch1", used)

    parts1 = batch1.split("---SECTION---")
    sections["abstract"] = parts1[0].strip() if len(parts1) > 0 else batch1
    sections["introduction"] = parts1[1].strip() if len(parts1) > 1 else ""
    sections["related_work"] = parts1[2].strip() if len(parts1) > 2 else ""

    print("[PaperWriter] Rewriting abstract...")
    sections["abstract"] = rewrite_abstract(
        client,
        sections["abstract"],
        metrics_str,
        title,
        topic
    )

    print("[PaperWriter] Batch 2: Methodology + Experiments + Results...")
    batch2_ctx = (
        base_ctx +
        "\nFull experiment plan: " + json.dumps(experiment_plan) +
        "\nResults table template:\n" + RESULTS_TABLE_TEMPLATE
    )
    batch2_instruction = (
        "Write THREE sections separated by '---SECTION---'.\n\n"
        "SECTION 1 - METHODOLOGY (500+ words):\n"
        "A) System Overview: full pipeline description\n"
        "B) Data Preprocessing: dataset, features, normalization\n"
        "C) Feature Engineering: XAI-based selection with technical detail\n"
        "D) Model Development: " + str([b["name"] for b in experiment_plan.get("baselines", [])]) + " with hyperparameters\n"
        "E) Mathematical Formulation: 2+ numbered equations, all variables defined\n\n"
        "SECTION 2 - EXPERIMENTS (400+ words):\n"
        "A) Dataset statistics\n"
        "B) Implementation details (Python, sklearn, hardware)\n"
        "C) Baselines with justification\n"
        "D) Evaluation metrics with formulas\n"
        "E) 5-fold cross-validation strategy\n"
        "F) Ablation study design\n\n"
        "SECTION 3 - RESULTS (400+ words):\n"
        "1) Main comparison table (use RESULTS_TABLE_TEMPLATE format, bold best per column)\n"
        "2) Analysis: '" + best_model + " achieves X%, outperforming baselines by Y%'\n"
        "3) Ablation table\n"
        "4) Feature importance top-5\n"
        "5) Statistical significance with p-value and confidence intervals\n"
        "USE ONLY: " + metrics_str + "\n\n"
        "Write all three. Separate with '---SECTION---'."
    )
    batch2 = _call(client, batch2_instruction, batch2_ctx, max_tokens=3000)
    used = log_usage("paper_writer_batch2", batch2_instruction, batch2)
    print_status("paper_writer_batch2", used)

    parts2 = batch2.split("---SECTION---")
    sections["methodology"] = parts2[0].strip() if len(parts2) > 0 else batch2
    sections["experiments"] = parts2[1].strip() if len(parts2) > 1 else ""
    sections["results"] = parts2[2].strip() if len(parts2) > 2 else ""

    print("[PaperWriter] Batch 3: Discussion + Conclusion + Limitations...")
    batch3_instruction = (
        "Write THREE sections separated by '---SECTION---'.\n\n"
        "SECTION 1 - DISCUSSION (400+ words):\n"
        "Paragraph 1: Hypothesis verdict with evidence from " + metrics_str + "\n"
        "Paragraph 2: Direct comparison — '" + best_model + " achieves X%, outperforming [paper] by Y%'\n"
        "Paragraph 3: Why the method works — mechanistic explanation\n"
        "Paragraph 4: Practical implications for " + domain + " deployment\n"
        "Paragraph 5: When the method might fail\n\n"
        "SECTION 2 - CONCLUSION (250+ words):\n"
        "Paragraph 1: Problem summary and what was proposed\n"
        "Paragraph 2: Key results with exact numbers from: " + metrics_str + "\n"
        "Paragraph 3: Three specific future work directions with technical detail\n"
        "Paragraph 4: Broader impact on " + domain + "\n\n"
        "SECTION 3 - LIMITATIONS (200+ words):\n"
        "Write exactly 3 limitations. Each must:\n"
        "- State it specifically (not vaguely)\n"
        "- Explain why it exists\n"
        "- Suggest how future work addresses it\n\n"
        "Write all three. Separate with '---SECTION---'."
    )
    batch3 = _call(client, batch3_instruction, base_ctx, max_tokens=2500)
    used = log_usage("paper_writer_batch3", batch3_instruction, batch3)
    print_status("paper_writer_batch3", used)

    parts3 = batch3.split("---SECTION---")
    sections["discussion"] = parts3[0].strip() if len(parts3) > 0 else batch3
    sections["conclusion"] = parts3[1].strip() if len(parts3) > 1 else ""
    sections["limitations"] = parts3[2].strip() if len(parts3) > 2 else ""

    refs = []
    for i, p in enumerate(papers[:20]):
        authors = ", ".join(p.get("authors", ["Unknown"])[:3])
        ref = (
            "[" + str(i+1) + "] " +
            authors + ', "' +
            p["title"] + '," ' +
            p.get("venue", "arXiv") + ", " +
            str(p.get("year", "")) + "."
        )
        refs.append(ref)
    sections["references"] = refs

    for sname, content in sections.items():
        if isinstance(content, str) and content:
            file_manager.save_section(sname, content)

    full = "# " + title + "\n\n"
    if method_acronym:
        full += "> Method: **" + method_acronym + "**\n\n"
    for s in ["abstract", "introduction", "related_work", "methodology",
              "experiments", "results", "discussion", "conclusion", "limitations"]:
        full += "## " + s.replace("_", " ").title() + "\n\n"
        full += sections.get(s, "") + "\n\n"
    full += "## References\n\n" + "\n".join(refs)

    file_manager.save_final("paper_draft.md", full)
    state_store.update_state("paper_draft", sections)
    audit_logger.log("paper_writer", {"title": title}, {"sections": list(sections.keys())})
    print("[PaperWriter] Saved to outputs/final/paper_draft.md")
    return sections