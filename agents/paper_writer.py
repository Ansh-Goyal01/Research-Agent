import json
import time
from groq import Groq
from tools import file_manager
from memory import state_store, audit_logger
import config


def _section(client, section_name, instruction, context, max_tokens=2048):
    time.sleep(3)
    response = client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": (
                "You are an expert academic paper writer targeting IEEE journals. "
                "Write detailed, technical, well-structured prose. "
                "Every claim must be supported by citations from the provided list. "
                "Use formal academic English. Minimum 300 words per section."
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

    print("[PaperWriter] Writing: " + title)

    metrics_str = ""
    for m in metrics:
        metrics_str += m["metric_name"] + ": " + str(m["mean"]) + " +/- " + str(m["std"]) + " (n=" + str(m["n_runs"]) + "), "

    base_ctx = (
        "Paper title: " + title + "\n"
        "Domain: " + str(input_topic.get("domain", "")) + "\n"
        "Target venue: " + str(input_topic.get("target_venue", "IEEE")) + "\n"
        "Research hypothesis: " + chosen_idea["hypothesis"] + "\n"
        "Hypothesis verdict: " + verdict + "\n"
        "Actual experimental metrics: " + metrics_str + "\n"
        "Key findings: " + json.dumps(findings) + "\n"
        "Gap addressed: " + gap_analysis["rationale"] + "\n"
        "Available citations (ONLY use these): " + json.dumps(paper_titles) + "\n"
        "Experiment details: " + json.dumps(experiment_plan) + "\n"
    )

    sections = {}

    print("[PaperWriter] Abstract...")
    sections["abstract"] = _section(client, "abstract",
        "Write a precise IEEE-style abstract of exactly 150-200 words covering: "
        "problem motivation, proposed approach, experimental setup, key results with exact numbers, and conclusion. "
        "Include the actual metric values from the experimental metrics provided.",
        base_ctx, max_tokens=512)

    print("[PaperWriter] Introduction...")
    sections["introduction"] = _section(client, "introduction",
        "Write a detailed introduction (400+ words) with 4 paragraphs: "
        "1) Problem motivation with statistics, "
        "2) Limitations of existing approaches citing at least 3 papers from the citations list, "
        "3) Proposed solution and contributions as a numbered list, "
        "4) Paper organization. "
        "Be specific about the research gap.",
        base_ctx + "\nPaper abstracts: " + json.dumps([p.get("abstract", "") for p in papers[:8]]))

    print("[PaperWriter] Related work...")
    sections["related_work"] = _section(client, "related_work",
        "Write a comprehensive related work section (400+ words) organized into 3 subsections: "
        "1) Traffic congestion prediction methods, "
        "2) Explainable AI techniques, "
        "3) Emergency vehicle prioritization systems. "
        "Cite at least 8 papers from the citations list. "
        "For each cited work explain what they did and what gap remains.",
        base_ctx + "\nFull paper list: " + json.dumps([{"title": p["title"], "abstract": p.get("abstract","")[:200]} for p in papers[:15]]))

    print("[PaperWriter] Methodology...")
    sections["methodology"] = _section(client, "methodology",
        "Write a detailed methodology section (500+ words) with subsections: "
        "1) System Overview with architecture description, "
        "2) Data Acquisition and Preprocessing, "
        "3) Feature Engineering (explain XAI-based feature selection), "
        "4) Model Development (Random Forest and Gradient Boosting with hyperparameters), "
        "5) XAI Integration (feature importance analysis), "
        "6) Mathematical formulation with at least 2 equations. "
        "Be technically precise.",
        base_ctx)

    print("[PaperWriter] Experiments...")
    sections["experiments"] = _section(client, "experiments",
        "Write a detailed experiments section (400+ words) covering: "
        "1) Dataset description with statistics, "
        "2) Experimental setup and implementation details, "
        "3) Baseline methods, "
        "4) Evaluation metrics with formulas, "
        "5) Cross-validation strategy (5-fold), "
        "6) Ablation study design.",
        base_ctx)

    print("[PaperWriter] Results...")
    sections["results"] = _section(client, "results",
        "Write a detailed results section (400+ words) using ONLY the exact metric numbers provided. "
        "Include: "
        "1) Main results table comparing all models, "
        "2) Ablation study results, "
        "3) Feature importance analysis findings, "
        "4) Statistical significance discussion. "
        "Every number must come from the experimental metrics. "
        "Format key results as: Model achieved X% accuracy (mean=X, std=Y, n=5 folds).",
        base_ctx)

    print("[PaperWriter] Discussion...")
    sections["discussion"] = _section(client, "discussion",
        "Write a detailed discussion section (400+ words) covering: "
        "1) Interpretation of results in context of the hypothesis, "
        "2) Comparison with related work findings, "
        "3) Practical implications for smart city deployment, "
        "4) Why XAI improves trust in traffic management, "
        "5) Unexpected findings and their explanations. "
        "Reference actual metric values throughout.",
        base_ctx + "\nAnomalies: " + json.dumps(result_summary.get("anomalies", [])))

    print("[PaperWriter] Conclusion...")
    sections["conclusion"] = _section(client, "conclusion",
        "Write a conclusion section (250+ words) with: "
        "1) Summary of contributions, "
        "2) Key experimental findings with exact numbers, "
        "3) Practical impact, "
        "4) Future work directions (at least 3 specific directions).",
        base_ctx)

    print("[PaperWriter] Limitations...")
    sections["limitations"] = _section(client, "limitations",
        "Write a limitations section (200+ words) with at least 3 specific, honest limitations of this work. "
        "Each limitation should reference a specific aspect of the methodology or results.",
        base_ctx)

    refs = []
    for i, p in enumerate(papers[:20]):
        authors = ", ".join(p.get("authors", ["Unknown"])[:3])
        ref = "[" + str(i+1) + "] " + authors + ", \"" + p["title"] + ",\" " + p.get("venue", "arXiv") + ", " + str(p.get("year", "")) + "."
        refs.append(ref)
    sections["references"] = refs

    for sname, content in sections.items():
        if isinstance(content, str):
            file_manager.save_section(sname, content)

    full = "# " + title + "\n\n"
    for s in ["abstract", "introduction", "related_work", "methodology", "experiments", "results", "discussion", "conclusion", "limitations"]:
        full += "## " + s.replace("_", " ").title() + "\n\n" + sections[s] + "\n\n"
    full += "## References\n\n" + "\n".join(refs)

    file_manager.save_final("paper_draft.md", full)
    state_store.update_state("paper_draft", sections)
    audit_logger.log("paper_writer", {"title": title}, {"sections": list(sections.keys())})
    print("[PaperWriter] Saved to outputs/final/paper_draft.md")
    return sections