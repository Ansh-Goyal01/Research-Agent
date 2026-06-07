import json
import time
from groq import Groq
from tools import file_manager
from memory import state_store, audit_logger
from memory.token_tracker import log_usage, print_status
from memory.style_guide import IEEE_STYLE_GUIDE, ABSTRACT_TEMPLATE, RELATED_WORK_TEMPLATE, RESULTS_TABLE_TEMPLATE
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

    if not result_summary:
        raise ValueError("[PaperWriter] No result_summary in state.")

    title = title_options[title_options["recommended"]]
    papers = paper_list["papers"]
    paper_titles = [p["title"] for p in papers]
    metrics = result_summary.get("metrics", [])
    findings = result_summary.get("key_findings", [])
    verdict = result_summary.get("hypothesis_verdict", "supported")
    topic = input_topic.get("topic", "")
    domain = input_topic.get("domain", "")
    venue = input_topic.get("target_venue", "IEEE")

    metrics_str = " | ".join([
        m["metric_name"] + ": " + str(m["mean"]) + "±" + str(m["std"])
        for m in metrics
    ])

    print("[PaperWriter] Writing: " + title)

    base_ctx = (
        "Title: " + title + "\n"
        "Topic: " + topic + "\n"
        "Domain: " + domain + "\n"
        "Venue: " + venue + "\n"
        "Hypothesis: " + chosen_idea["hypothesis"] + "\n"
        "Verdict: " + verdict + "\n"
        "Actual metrics (use ONLY these numbers): " + metrics_str + "\n"
        "Key findings: " + json.dumps(findings) + "\n"
        "Research gap addressed: " + gap_analysis["rationale"] + "\n"
        "Top gap description: " + gap_analysis["gaps"][0]["description"] + "\n"
        "Supporting papers for gap: " + json.dumps(gap_analysis["gaps"][0].get("supporting_paper_titles", [])) + "\n"
        "Available citations (ONLY use these): " + json.dumps(paper_titles[:12]) + "\n"
        "Baselines used: " + str([b["name"] for b in experiment_plan.get("baselines", [])]) + "\n"
        "Dataset: " + str([d["name"] for d in experiment_plan.get("datasets", [])]) + "\n"
    )

    sections = {}

    print("[PaperWriter] Batch 1: Abstract + Introduction + Related Work...")
    batch1_instruction = (
        "Write THREE sections separated by '---SECTION---'.\n\n"

        "SECTION 1 - ABSTRACT:\n" +
        ABSTRACT_TEMPLATE + "\n\n"

        "SECTION 2 - INTRODUCTION:\n"
        "Follow this exact structure:\n"
        "Paragraph 1: Broader problem with real-world impact\n"
        "Paragraph 2: Limitations of existing work — cite at least 3 papers from citations list. "
        "For each cited paper say what it does AND what it fails to do. "
        "Connect directly to gap: '" + gap_analysis["rationale"] + "'\n"
        "Paragraph 3: What this paper proposes — one clear paragraph\n"
        "Paragraph 4: Numbered contributions list (3-4 items, specific and technical)\n"
        "Paragraph 5: Paper organization sentence\n"
        "Minimum 400 words.\n\n"

        "SECTION 3 - RELATED WORK:\n" +
        RELATED_WORK_TEMPLATE + "\n"
        "Use only papers from the citations list. Minimum 400 words.\n\n"

        "Write all three sections now. Separate with '---SECTION---'."
    )
    batch1 = _call(client, batch1_instruction, base_ctx, max_tokens=3000)
    used = log_usage("paper_writer_batch1", batch1_instruction, batch1)
    print_status("paper_writer_batch1", used)

    parts1 = batch1.split("---SECTION---")
    sections["abstract"] = parts1[0].strip() if len(parts1) > 0 else batch1
    sections["introduction"] = parts1[1].strip() if len(parts1) > 1 else ""
    sections["related_work"] = parts1[2].strip() if len(parts1) > 2 else ""

    print("[PaperWriter] Batch 2: Methodology + Experiments + Results...")
    batch2_ctx = (
        base_ctx +
        "\nFull experiment plan: " + json.dumps(experiment_plan) +
        "\nResults table template to follow:\n" + RESULTS_TABLE_TEMPLATE
    )
    batch2_instruction = (
        "Write THREE sections separated by '---SECTION---'.\n\n"

        "SECTION 1 - METHODOLOGY (500+ words):\n"
        "Subsections:\n"
        "A) System Overview: describe full pipeline in one paragraph\n"
        "B) Data Acquisition and Preprocessing: describe dataset, features, cleaning steps\n"
        "C) Feature Engineering: explain XAI-based feature selection with technical detail\n"
        "D) Model Development: describe each model with hyperparameters\n"
        "E) Mathematical Formulation: include at least 2 numbered equations with all variables defined\n\n"

        "SECTION 2 - EXPERIMENTS (400+ words):\n"
        "Subsections:\n"
        "A) Dataset: name, size, source, train/test split\n"
        "B) Implementation: language, library versions, hardware specs\n"
        "C) Baselines: for each baseline explain why it was chosen\n"
        "D) Evaluation Metrics: formulas for each metric\n"
        "E) Cross-validation: describe 5-fold strategy\n"
        "F) Ablation Study: describe what components are removed and why\n\n"

        "SECTION 3 - RESULTS (400+ words):\n"
        "Follow this exact structure:\n"
        "1) Present the comparison table using RESULTS_TABLE_TEMPLATE format\n"
        "2) Analyze table: 'The proposed method achieves X% accuracy, outperforming Baseline1 by Y%'\n"
        "3) Ablation study results table\n"
        "4) Feature importance top 5 findings\n"
        "5) Statistical significance: mention p-values and what tests were used\n"
        "USE ONLY THESE EXACT NUMBERS: " + metrics_str + "\n\n"

        "Write all three sections. Separate with '---SECTION---'."
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
        "Paragraph 1: Restate hypothesis and verdict with evidence from metrics\n"
        "Paragraph 2: Direct comparison — 'Our method achieves X%, outperforming [paper] by Y%'\n"
        "Paragraph 3: Why the method works — mechanistic explanation\n"
        "Paragraph 4: Practical implications for real deployment in " + domain + "\n"
        "Paragraph 5: When the method might fail — honest assessment\n\n"

        "SECTION 2 - CONCLUSION (250+ words):\n"
        "Paragraph 1: Summary of the problem and what was proposed\n"
        "Paragraph 2: Restate contributions with exact numbers from: " + metrics_str + "\n"
        "Paragraph 3: Three specific future work directions with technical detail\n"
        "Paragraph 4: Broader impact on " + domain + "\n"
        "Do NOT introduce new information.\n\n"

        "SECTION 3 - LIMITATIONS (200+ words):\n"
        "Write exactly 3 limitations. For each:\n"
        "- State the limitation specifically (not vaguely)\n"
        "- Explain why it exists in this work\n"
        "- Suggest how future work could address it\n"
        "Example of GOOD limitation: 'The dataset contains only 569 samples from a single "
        "institution, which may limit generalizability to other clinical settings.'\n"
        "Example of BAD limitation: 'The dataset is limited.'\n\n"

        "Write all three sections. Separate with '---SECTION---'."
    )
    batch3 = _call(client, batch3_instruction, base_ctx, max_tokens=2500)
    used = log_usage("paper_writer_batch3", batch3_instruction, batch3)
    print_status("paper_writer_batch3", used)

    parts3 = batch3.split("---SECTION---")
    sections["discussion"] = parts3[0].strip() if len(parts3) > 0 else batch3
    sections["conclusion"] = parts3[1].strip() if len(parts3) > 1 else ""
    sections["limitations"] = parts3[2].strip() if len(parts3) > 2 else ""

    refs = []
    for i, p in enumerate(papers[:15]):
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