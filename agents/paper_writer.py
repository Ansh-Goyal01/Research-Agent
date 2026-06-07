import json
import time
from groq import Groq
from tools import file_manager
from memory import state_store, audit_logger
from memory.token_tracker import log_usage, print_status
import config


def _call(client, instruction, context, max_tokens=3000):
    time.sleep(6)
    response = client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": (
                "You are an IEEE journal paper writer. "
                "Write detailed technical academic prose. "
                "Minimum 250 words per section. "
                "Use citations from the provided list only."
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

    metrics_str = " | ".join([
        m["metric_name"] + ": " + str(m["mean"]) + "±" + str(m["std"])
        for m in metrics
    ])

    print("[PaperWriter] Writing: " + title)

    base_ctx = (
        "Title: " + title + "\n"
        "Topic: " + input_topic.get("topic", "") + "\n"
        "Domain: " + input_topic.get("domain", "") + "\n"
        "Venue: " + input_topic.get("target_venue", "IEEE") + "\n"
        "Hypothesis: " + chosen_idea["hypothesis"] + "\n"
        "Verdict: " + verdict + "\n"
        "Metrics: " + metrics_str + "\n"
        "Findings: " + json.dumps(findings) + "\n"
        "Gap: " + gap_analysis["rationale"] + "\n"
        "Citations (use ONLY these): " + json.dumps(paper_titles[:12]) + "\n"
        "Experiment: baselines=" + str([b["name"] for b in experiment_plan.get("baselines", [])]) +
        " datasets=" + str([d["name"] for d in experiment_plan.get("datasets", [])]) + "\n"
    )

    sections = {}

    print("[PaperWriter] Batch 1: Abstract + Introduction + Related Work...")
    batch1_instruction = (
        "Write THREE sections for a research paper. "
        "Separate each section with exactly '---SECTION---'.\n\n"
        "SECTION 1 - ABSTRACT (150-200 words): "
        "IEEE-style abstract covering problem, approach, key results with exact numbers, conclusion.\n\n"
        "SECTION 2 - INTRODUCTION (400+ words): "
        "4 paragraphs: problem motivation, limitations of existing work (cite 3+ papers), "
        "our contributions as numbered list, paper organization.\n\n"
        "SECTION 3 - RELATED WORK (400+ words): "
        "3 subsections matching the domain. Cite at least 6 papers. "
        "For each cited work: what they did and what gap remains.\n\n"
        "Write all three sections now."
    )
    batch1 = _call(client, batch1_instruction, base_ctx, max_tokens=3000)
    used = log_usage("paper_writer_batch1", batch1_instruction, batch1)
    print_status("paper_writer_batch1", used)

    parts1 = batch1.split("---SECTION---")
    sections["abstract"] = parts1[0].strip() if len(parts1) > 0 else batch1
    sections["introduction"] = parts1[1].strip() if len(parts1) > 1 else ""
    sections["related_work"] = parts1[2].strip() if len(parts1) > 2 else ""

    print("[PaperWriter] Batch 2: Methodology + Experiments + Results...")
    batch2_ctx = base_ctx + "\nFull experiment plan: " + json.dumps(experiment_plan)
    batch2_instruction = (
        "Write THREE sections for a research paper. "
        "Separate each with '---SECTION---'.\n\n"
        "SECTION 1 - METHODOLOGY (500+ words): "
        "Subsections: System Overview, Data Preprocessing, Feature Engineering (XAI-based), "
        "Model Development with hyperparameters, Mathematical formulation (2+ equations).\n\n"
        "SECTION 2 - EXPERIMENTS (400+ words): "
        "Dataset description with statistics, experimental setup, baselines, "
        "evaluation metrics with formulas, 5-fold cross-validation, ablation study design.\n\n"
        "SECTION 3 - RESULTS (400+ words): "
        "Results table comparing all models, ablation results, feature importance findings, "
        "statistical significance. Use ONLY these exact numbers: " + metrics_str + "\n\n"
        "Write all three sections now."
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
        "Write THREE sections for a research paper. "
        "Separate each with '---SECTION---'.\n\n"
        "SECTION 1 - DISCUSSION (400+ words): "
        "Interpret results vs hypothesis, compare with related work, "
        "practical implications, why XAI improves trust, unexpected findings.\n\n"
        "SECTION 2 - CONCLUSION (250+ words): "
        "Summary of contributions, exact metric numbers, practical impact, "
        "3 specific future work directions.\n\n"
        "SECTION 3 - LIMITATIONS (200+ words): "
        "3 specific honest limitations referencing methodology and results.\n\n"
        "Write all three sections now."
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
        ref = "[" + str(i+1) + "] " + authors + ', "' + p["title"] + '," ' + p.get("venue", "arXiv") + ", " + str(p.get("year", "")) + "."
        refs.append(ref)
    sections["references"] = refs

    for sname, content in sections.items():
        if isinstance(content, str) and content:
            file_manager.save_section(sname, content)

    full = "# " + title + "\n\n"
    for s in ["abstract", "introduction", "related_work", "methodology",
              "experiments", "results", "discussion", "conclusion", "limitations"]:
        full += "## " + s.replace("_", " ").title() + "\n\n" + sections.get(s, "") + "\n\n"
    full += "## References\n\n" + "\n".join(refs)

    file_manager.save_final("paper_draft.md", full)
    state_store.update_state("paper_draft", sections)
    audit_logger.log("paper_writer", {"title": title}, {"sections": list(sections.keys())})
    print("[PaperWriter] Saved to outputs/final/paper_draft.md")
    return sections