import json
from groq import Groq
from tools import file_manager
from memory import state_store, audit_logger
import config

def _section(client, instruction, context):
    response = client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": "You are an academic paper writer. Write clear academic prose."},
            {"role": "user", "content": instruction + "\n\nContext:\n" + context}
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
    print("[PaperWriter] Writing: " + title)
    ctx = (
        "Title: " + title + "\n"
        "Hypothesis: " + chosen_idea["hypothesis"] + "\n"
        "Key findings: " + json.dumps(result_summary["key_findings"]) + "\n"
        "Metrics: " + json.dumps(result_summary["metrics"]) + "\n"
        "Citations allowed: " + json.dumps(paper_titles[:15])
    )
    sections = {}
    print("[PaperWriter] Abstract...")
    sections["abstract"] = _section(client, "Write a concise academic abstract under 250 words.", ctx)
    print("[PaperWriter] Introduction...")
    sections["introduction"] = _section(client, "Write the introduction motivating the problem.", ctx)
    print("[PaperWriter] Related work...")
    sections["related_work"] = _section(client, "Write related work citing only the allowed citations.", ctx)
    print("[PaperWriter] Methodology...")
    sections["methodology"] = _section(client, "Write the methodology section.", ctx + "\nPlan: " + json.dumps(experiment_plan))
    print("[PaperWriter] Experiments...")
    sections["experiments"] = _section(client, "Write the experiments section.", ctx + "\nPlan: " + json.dumps(experiment_plan))
    print("[PaperWriter] Results...")
    sections["results"] = _section(client, "Write results using ONLY the exact metric numbers provided.", ctx)
    print("[PaperWriter] Discussion...")
    sections["discussion"] = _section(client, "Write the discussion section.", ctx)
    print("[PaperWriter] Conclusion...")
    sections["conclusion"] = _section(client, "Write the conclusion.", ctx)
    print("[PaperWriter] Limitations...")
    sections["limitations"] = _section(client, "Write limitations with at least 2 genuine limitations.", ctx)
    refs = [
        "[" + str(i+1) + "] " + p["title"] + " - " + ", ".join(p["authors"][:2]) + " (" + str(p["year"]) + ")"
        for i, p in enumerate(papers[:15])
    ]
    sections["references"] = refs
    full = "# " + title + "\n\n"
    for s in ["abstract","introduction","related_work","methodology","experiments","results","discussion","conclusion","limitations"]:
        full += "## " + s.replace("_"," ").title() + "\n\n" + sections[s] + "\n\n"
    full += "## References\n\n" + "\n".join(refs)
    file_manager.save_final("paper_draft.md", full)
    state_store.update_state("paper_draft", sections)
    audit_logger.log("paper_writer", {"title": title}, {"sections": list(sections.keys())})
    print("[PaperWriter] Saved to outputs/final/paper_draft.md")
    return sections
