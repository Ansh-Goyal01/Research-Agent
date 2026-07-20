import json
from groq import Groq
from tools import file_manager
from memory import state_store, audit_logger
import config

SYSTEM_PROMPT = """You are an academic paper writer.
Rules:
- Every citation must come from the provided paper list only.
- Every number in Results must match the result_summary exactly.
- Abstract must be under 250 words.
- Limitations must include at least 2 genuine limitations.
- Write in clear academic English.
- Return ONLY valid JSON. No extra text."""

def _write_section(client, section_name, instruction, context):
    response = client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{instruction}\n\nContext:\n{context}"}
        ],
        max_tokens=config.MAX_TOKENS,
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

    if not result_summary:
        raise ValueError("[PaperWriter] No result_summary in state.")

    title = title_options[title_options["recommended"]]
    papers = paper_list["papers"]
    paper_titles = [p["title"] for p in papers]

    print(f"[PaperWriter] Writing paper: {title}")

    base_context = f"""
Paper title: {title}
Hypothesis: {chosen_idea['hypothesis']}
Key findings: {json.dumps(result_summary['key_findings'])}
Metrics: {json.dumps(result_summary['metrics'])}
Available citations (use ONLY these): {json.dumps(paper_titles[:15])}
Gap addressed: {gap_analysis['rationale']}
"""

    sections = {}

    print("[PaperWriter] Writing abstract...")
    sections["abstract"] = _write_section(client, "abstract",
        "Write a concise academic abstract under 250 words covering: motivation, method, results, conclusion.",
        base_context)

    print("[PaperWriter] Writing introduction...")
    sections["introduction"] = _write_section(client, "introduction",
        "Write the introduction section motivating the problem using evidence from the available citations.",
        base_context + f"\nPaper abstracts: {json.dumps([p['abstract_summary'] for p in papers[:5]])}")

    print("[PaperWriter] Writing related work...")
    sections["related_work"] = _write_section(client, "related_work",
        "Write a related work section citing only papers from the available citations list.",
        base_context + f"\nPaper details: {json.dumps([{'title': p['title'], 'contributions': p['key_contributions']} for p in papers[:10]])}")

    print("[PaperWriter] Writing methodology...")
    sections["methodology"] = _write_section(client, "methodology",
        "Write the methodology section describing the proposed approach clearly.",
        base_context + f"\nExperiment plan: {json.dumps(experiment_plan)}")

    print("[PaperWriter] Writing experiments...")
    sections["experiments"] = _write_section(client, "experiments",
        "Write the experiments section describing datasets, baselines, and evaluation setup.",
        base_context + f"\nExperiment plan: {json.dumps(experiment_plan)}")

    print("[PaperWriter] Writing results...")
    sections["results"] = _write_section(client, "results",
        "Write the results section. Use ONLY the exact numbers from the metrics provided.",
        base_context)

    print("[PaperWriter] Writing discussion...")
    sections["discussion"] = _write_section(client, "discussion",
        "Write the discussion section interpreting findings and their implications.",
        base_context + f"\nAnomalies: {json.dumps(result_summary.get('anomalies', []))}")

    print("[PaperWriter] Writing conclusion...")
    sections["conclusion"] = _write_section(client, "conclusion",
        "Write the conclusion summarizing contributions and future work.",
        base_context)

    print("[PaperWriter] Writing limitations...")
    sections["limitations"] = _write_section(client, "limitations",
        "Write a honest limitations section with at least 2 genuine limitations of this work.",
        base_context)

    references = [f"[{i+1}] {p['title']} - {', '.join(p['authors'][:2])} ({p['year']})"
                  for i, p in enumerate(papers[:15])]
    sections["references"] = references

    for section_name, content in sections.items():
        if isinstance(content, str):
            file_manager.save_section(section_name, content)

    full_paper = f"# {title}\n\n"
    for section in ["abstract", "introduction", "related_work", "methodology",
                    "experiments", "results", "discussion", "conclusion", "limitations"]:
        full_paper += f"## {section.replace('_', ' ').title()}\n\n{sections[section]}\n\n"
    full_paper += "## References\n\n" + "\n".join(references)

    file_manager.save_final("paper_draft.md", full_paper)
    state_store.update_state("paper_draft", sections)
    audit_logger.log("paper_writer", {"title": title}, {"sections": list(sections.keys())})
    print("[PaperWriter] Paper draft saved to outputs/final/paper_draft.md")
    return sections