import os

files = {}

files["agents/scout.py"] = """import json
from groq import Groq
from tools.paper_search import search_papers
from memory import state_store, audit_logger
import config

def run(topic_data):
    client = Groq(api_key=config.GROQ_API_KEY)
    topic = topic_data["topic"]
    start_year = topic_data.get("date_range", [2021, 2025])[0]
    max_papers = topic_data.get("max_papers", 25)
    print("[Scout] Searching: " + topic)
    raw_papers = search_papers(topic, max_results=max_papers, start_year=start_year)
    print("[Scout] Found " + str(len(raw_papers)) + " papers.")
    prompt = (
        "Papers fetched from arXiv on: " + topic + "\\n\\n" +
        json.dumps(raw_papers, indent=2) +
        "\\n\\nFor each paper return a JSON array with fields: "
        "title, authors, year, venue, url, abstract_summary, "
        "key_contributions, stated_limitations, semantic_scholar_id. "
        "Return ONLY the JSON array, no markdown."
    )
    response = client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": "You are a literature scout. Return ONLY valid JSON arrays."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=config.MAX_TOKENS,
        temperature=0.3
    )
    raw_output = response.choices[0].message.content.strip()
    if "```" in raw_output:
        raw_output = raw_output.split("```")[1]
        if raw_output.startswith("json"):
            raw_output = raw_output[4:]
    try:
        papers = json.loads(raw_output)
    except Exception as e:
        print("[Scout] Parse error: " + str(e))
        papers = raw_papers
    result = {
        "papers": papers,
        "search_queries_used": [topic],
        "timestamp": str(__import__("datetime").datetime.now())
    }
    state_store.update_state("paper_list", result)
    audit_logger.log("scout", topic_data, result)
    print("[Scout] Done. " + str(len(papers)) + " papers saved.")
    return result
"""

files["agents/gap_analyst.py"] = """import json
from groq import Groq
from memory import state_store, audit_logger
import config

def run():
    client = Groq(api_key=config.GROQ_API_KEY)
    paper_list = state_store.get_state("paper_list")
    if not paper_list:
        raise ValueError("[GapAnalyst] No paper_list in state.")
    papers = paper_list["papers"]
    print("[GapAnalyst] Analyzing " + str(len(papers)) + " papers...")
    prompt = (
        "Analyze these papers and find research gaps:\\n\\n" +
        json.dumps(papers, indent=2) +
        "\\n\\nReturn a JSON object with this structure:\\n"
        "{\\n"
        "  \\"gaps\\": [\\n"
        "    {\\n"
        "      \\"gap_id\\": \\"gap_1\\",\\n"
        "      \\"description\\": \\"description\\",\\n"
        "      \\"supporting_paper_titles\\": [\\"title1\\"],\\n"
        "      \\"impact_score\\": 8.5,\\n"
        "      \\"feasibility_score\\": 7.0\\n"
        "    }\\n"
        "  ],\\n"
        "  \\"top_gap_id\\": \\"gap_1\\",\\n"
        "  \\"rationale\\": \\"why this gap matters\\"\\n"
        "}\\n"
        "Return ONLY the JSON object."
    )
    response = client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": "You are a research analyst. Return ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=config.MAX_TOKENS,
        temperature=0.3
    )
    raw = response.choices[0].message.content.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    result = json.loads(raw)
    state_store.update_state("gap_analysis", result)
    audit_logger.log("gap_analyst", {"papers": len(papers)}, result)
    print("[GapAnalyst] Found " + str(len(result["gaps"])) + " gaps.")
    return result
"""

files["agents/idea_generator.py"] = """import json
from groq import Groq
from memory import state_store, audit_logger
import config

def run():
    client = Groq(api_key=config.GROQ_API_KEY)
    gap_analysis = state_store.get_state("gap_analysis")
    paper_list = state_store.get_state("paper_list")
    if not gap_analysis:
        raise ValueError("[IdeaGenerator] No gap_analysis in state.")
    print("[IdeaGenerator] Generating ideas...")
    prompt = (
        "Based on this gap analysis:\\n" +
        json.dumps(gap_analysis, indent=2) +
        "\\n\\nGenerate 3 research ideas. Return a JSON object:\\n"
        "{\\n"
        "  \\"ideas\\": [\\n"
        "    {\\n"
        "      \\"idea_id\\": \\"idea_1\\",\\n"
        "      \\"hypothesis\\": \\"one sentence hypothesis\\",\\n"
        "      \\"novelty_explanation\\": \\"what is new\\",\\n"
        "      \\"minimum_viable_experiment\\": \\"simplest test\\",\\n"
        "      \\"compute_cost\\": \\"low\\",\\n"
        "      \\"time_estimate\\": \\"1-2 weeks\\",\\n"
        "      \\"novelty_score\\": 8.0,\\n"
        "      \\"feasibility_score\\": 9.0,\\n"
        "      \\"impact_score\\": 8.5\\n"
        "    }\\n"
        "  ],\\n"
        "  \\"recommended_idea_id\\": \\"idea_1\\"\\n"
        "}\\n"
        "Return ONLY the JSON object."
    )
    response = client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": "You are a research idea generator. Return ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=config.MAX_TOKENS,
        temperature=0.7
    )
    raw = response.choices[0].message.content.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    result = json.loads(raw)
    state_store.update_state("idea_candidates", result)
    audit_logger.log("idea_generator", {"top_gap": gap_analysis["top_gap_id"]}, result)
    print("[IdeaGenerator] Generated " + str(len(result["ideas"])) + " ideas.")
    return result
"""

files["agents/title_agent.py"] = """import json
from groq import Groq
from memory import state_store, audit_logger
import config

def run(chosen_idea_id=None):
    client = Groq(api_key=config.GROQ_API_KEY)
    ideas = state_store.get_state("idea_candidates")
    if not ideas:
        raise ValueError("[TitleAgent] No idea_candidates in state.")
    if chosen_idea_id:
        idea = next((i for i in ideas["ideas"] if i["idea_id"] == chosen_idea_id), ideas["ideas"][0])
    else:
        idea = ideas["ideas"][0]
    state_store.update_state("chosen_idea", idea)
    print("[TitleAgent] Generating titles...")
    prompt = (
        "Generate 3 titles for this research:\\n"
        "Hypothesis: " + idea["hypothesis"] + "\\n"
        "Novelty: " + idea["novelty_explanation"] + "\\n\\n"
        "Return a JSON object:\\n"
        "{\\n"
        "  \\"descriptive\\": \\"title stating what was done\\",\\n"
        "  \\"punchy\\": \\"hook: subtitle\\",\\n"
        "  \\"question_form\\": \\"title as question?\\",\\n"
        "  \\"recommended\\": \\"descriptive\\",\\n"
        "  \\"rationale\\": \\"why recommended\\"\\n"
        "}\\n"
        "Return ONLY the JSON object."
    )
    response = client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": "You are an academic title writer. Return ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1024,
        temperature=0.7
    )
    raw = response.choices[0].message.content.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    result = json.loads(raw)
    state_store.update_state("title_options", result)
    audit_logger.log("title_agent", {"idea_id": idea["idea_id"]}, result)
    print("[TitleAgent] Recommended: " + result[result["recommended"]])
    return result
"""

files["agents/experiment_designer.py"] = """import json
from groq import Groq
from memory import state_store, audit_logger
import config

def run():
    client = Groq(api_key=config.GROQ_API_KEY)
    idea = state_store.get_state("chosen_idea")
    if not idea:
        raise ValueError("[ExperimentDesigner] No chosen_idea in state.")
    print("[ExperimentDesigner] Designing experiment...")
    prompt = (
        "Design an experiment for:\\n"
        "Hypothesis: " + idea["hypothesis"] + "\\n"
        "Compute cost: " + idea["compute_cost"] + "\\n\\n"
        "Return a JSON object:\\n"
        "{\\n"
        "  \\"hypothesis_null\\": \\"null hypothesis\\",\\n"
        "  \\"hypothesis_alternative\\": \\"alternative hypothesis\\",\\n"
        "  \\"independent_variables\\": [\\"var1\\"],\\n"
        "  \\"dependent_variables\\": [\\"metric1\\"],\\n"
        "  \\"baselines\\": [{\\\"name\\\": \\"method\\", \\"citation\\": \\"paper\\", \\"reason_for_inclusion\\": \\"why\\"}],\\n"
        "  \\"datasets\\": [{\\\"name\\\": \\"dataset\\", \\"url\\": \\"url\\", \\"size\\": \\"size\\", \\"license\\": \\"license\\"}],\\n"
        "  \\"metrics\\": [{\\\"name\\\": \\"accuracy\\", \\"formula\\": \\"correct/total\\", \\"higher_is_better\\": true}],\\n"
        "  \\"statistical_tests\\": [\\"t-test\\"],\\n"
        "  \\"ablation_design\\": \\"description\\",\\n"
        "  \\"compute_requirements\\": \\"CPU only, 8GB RAM\\",\\n"
        "  \\"estimated_runtime_hours\\": 1.0\\n"
        "}\\n"
        "Use only real public datasets. Return ONLY the JSON object."
    )
    response = client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": "You are an experiment designer. Return ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=config.MAX_TOKENS,
        temperature=0.3
    )
    raw = response.choices[0].message.content.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    result = json.loads(raw)
    state_store.update_state("experiment_plan", result)
    audit_logger.log("experiment_designer", {"idea": idea["idea_id"]}, result)
    print("[ExperimentDesigner] Done. Runtime: " + str(result["estimated_runtime_hours"]) + "h")
    return result
"""

files["agents/implementation_agent.py"] = """import json
from groq import Groq
from tools import file_manager, code_executor
from memory import state_store, audit_logger
import config

def run():
    client = Groq(api_key=config.GROQ_API_KEY)
    experiment_plan = state_store.get_state("experiment_plan")
    chosen_idea = state_store.get_state("chosen_idea")
    if not experiment_plan:
        raise ValueError("[ImplementationAgent] No experiment_plan in state.")
    print("[ImplementationAgent] Writing experiment code...")
    prompt = (
        "Write a complete Python experiment script for:\\n"
        "Hypothesis: " + chosen_idea["hypothesis"] + "\\n\\n"
        "Plan:\\n" + json.dumps(experiment_plan, indent=2) + "\\n\\n"
        "Requirements:\\n"
        "- Use only scikit-learn, numpy, pandas, matplotlib\\n"
        "- CPU only\\n"
        "- Save metrics to outputs/code/results.json in format:\\n"
        "  {metrics: [{metric_name, mean, std, n_runs}], hypothesis_verdict, key_findings}\\n"
        "- Save plots to outputs/plots/\\n"
        "- Must complete in under 60 seconds\\n"
        "Return ONLY the Python code."
    )
    response = client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": "You are a Python ML engineer. Return ONLY Python code."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=config.MAX_TOKENS,
        temperature=0.2
    )
    code = response.choices[0].message.content.strip()
    if "```" in code:
        code = code.split("```")[1]
        if code.startswith("python"):
            code = code[6:]
        code = code.split("```")[0]
    script_path = config.OUTPUTS_CODE / "experiment.py"
    file_manager.safe_write(script_path, code)
    print("[ImplementationAgent] Running experiment...")
    result = code_executor.run_script(script_path)
    retry_count = 0
    if not result["success"] and retry_count < 2:
        print("[ImplementationAgent] Failed, retrying...")
        retry_count += 1
        fix_prompt = (
            "Fix this Python code that failed with:\\n" +
            result["stderr"] + "\\n\\nCode:\\n" + code +
            "\\n\\nReturn ONLY the fixed Python code."
        )
        fix_response = client.chat.completions.create(
            model=config.MODEL,
            messages=[
                {"role": "system", "content": "You are a Python debugger. Return ONLY Python code."},
                {"role": "user", "content": fix_prompt}
            ],
            max_tokens=config.MAX_TOKENS,
            temperature=0.2
        )
        code = fix_response.choices[0].message.content.strip()
        if "```" in code:
            code = code.split("```")[1]
            if code.startswith("python"):
                code = code[6:]
            code = code.split("```")[0]
        file_manager.safe_write(script_path, code)
        result = code_executor.run_script(script_path)
    result["retry_count"] = retry_count
    state_store.update_state("code_result", result)
    audit_logger.log("implementation_agent", {"idea": chosen_idea.get("idea_id")}, result)
    if result["success"]:
        print("[ImplementationAgent] Done in " + str(result["execution_time_seconds"]) + "s")
    else:
        print("[ImplementationAgent] Failed: " + result["stderr"][:200])
    return result
"""

files["agents/results_analyst.py"] = """import json
from groq import Groq
from memory import state_store, audit_logger
import config

def run():
    client = Groq(api_key=config.GROQ_API_KEY)
    code_result = state_store.get_state("code_result")
    experiment_plan = state_store.get_state("experiment_plan")
    chosen_idea = state_store.get_state("chosen_idea")
    if not code_result:
        raise ValueError("[ResultsAnalyst] No code_result in state.")
    results_file = config.OUTPUTS_CODE / "results.json"
    actual_results = {}
    if results_file.exists():
        with open(results_file, "r") as f:
            actual_results = json.load(f)
    print("[ResultsAnalyst] Analyzing results...")
    prompt = (
        "Analyze these experiment results:\\n"
        "Hypothesis: " + chosen_idea["hypothesis"] + "\\n\\n"
        "Actual results:\\n" + json.dumps(actual_results, indent=2) + "\\n\\n"
        "Stdout:\\n" + code_result.get("stdout", "")[:1000] + "\\n\\n"
        "Return a JSON object:\\n"
        "{\\n"
        "  \\"metrics\\": [{\\\"metric_name\\\":\\\"\\\", \\"mean\\\":0.0, \\"std\\\":0.0, \\"n_runs\\\":1}],\\n"
        "  \\"hypothesis_verdict\\": \\"supported\\",\\n"
        "  \\"key_findings\\": [\\"finding1\\"],\\n"
        "  \\"anomalies\\": [],\\n"
        "  \\"suggested_ablation\\": \\"suggestion\\"\\n"
        "}\\n"
        "Only include metrics from actual results. Return ONLY the JSON object."
    )
    response = client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": "You are a results analyst. Return ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=config.MAX_TOKENS,
        temperature=0.2
    )
    raw = response.choices[0].message.content.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    result = json.loads(raw)
    state_store.update_state("result_summary", result)
    audit_logger.log("results_analyst", {"success": code_result["success"]}, result)
    print("[ResultsAnalyst] Verdict: " + result["hypothesis_verdict"])
    return result
"""

files["agents/paper_writer.py"] = """import json
from groq import Groq
from tools import file_manager
from memory import state_store, audit_logger
import config

def _section(client, instruction, context):
    response = client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": "You are an academic paper writer. Write clear academic prose."},
            {"role": "user", "content": instruction + "\\n\\nContext:\\n" + context}
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
        "Title: " + title + "\\n"
        "Hypothesis: " + chosen_idea["hypothesis"] + "\\n"
        "Key findings: " + json.dumps(result_summary["key_findings"]) + "\\n"
        "Metrics: " + json.dumps(result_summary["metrics"]) + "\\n"
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
    sections["methodology"] = _section(client, "Write the methodology section.", ctx + "\\nPlan: " + json.dumps(experiment_plan))
    print("[PaperWriter] Experiments...")
    sections["experiments"] = _section(client, "Write the experiments section.", ctx + "\\nPlan: " + json.dumps(experiment_plan))
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
    full = "# " + title + "\\n\\n"
    for s in ["abstract","introduction","related_work","methodology","experiments","results","discussion","conclusion","limitations"]:
        full += "## " + s.replace("_"," ").title() + "\\n\\n" + sections[s] + "\\n\\n"
    full += "## References\\n\\n" + "\\n".join(refs)
    file_manager.save_final("paper_draft.md", full)
    state_store.update_state("paper_draft", sections)
    audit_logger.log("paper_writer", {"title": title}, {"sections": list(sections.keys())})
    print("[PaperWriter] Saved to outputs/final/paper_draft.md")
    return sections
"""

files["agents/reviewer.py"] = """import json
from groq import Groq
from memory import state_store, audit_logger
import config

def run():
    client = Groq(api_key=config.GROQ_API_KEY)
    paper_draft = state_store.get_state("paper_draft")
    paper_list = state_store.get_state("paper_list")
    result_summary = state_store.get_state("result_summary")
    if not paper_draft:
        raise ValueError("[Reviewer] No paper_draft in state.")
    paper_titles = [p["title"] for p in paper_list["papers"]]
    actual_metrics = result_summary["metrics"]
    print("[Reviewer] Reviewing paper...")
    prompt = (
        "Review this paper draft:\\n\\n"
        "ABSTRACT:\\n" + paper_draft.get("abstract","") + "\\n\\n"
        "RESULTS:\\n" + paper_draft.get("results","") + "\\n\\n"
        "LIMITATIONS:\\n" + paper_draft.get("limitations","") + "\\n\\n"
        "Allowed citations: " + json.dumps(paper_titles) + "\\n"
        "Actual metrics: " + json.dumps(actual_metrics) + "\\n\\n"
        "Return a JSON object:\\n"
        "{\\n"
        "  \\"issues\\": [{\\\"severity\\\": \\"minor\\", \\"section\\": \\"results\\", \\"description\\": \\"issue\\"}],\\n"
        "  \\"overall_verdict\\": \\"accept\\",\\n"
        "  \\"required_changes\\": [\\"change1\\"],\\n"
        "  \\"citation_audit_passed\\": true,\\n"
        "  \\"number_audit_passed\\": true,\\n"
        "  \\"hallucination_flags\\": []\\n"
        "}\\n"
        "Return ONLY the JSON object."
    )
    response = client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": "You are a peer reviewer. Return ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=config.MAX_TOKENS,
        temperature=0.2
    )
    raw = response.choices[0].message.content.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    result = json.loads(raw)
    state_store.update_state("reviewer_feedback", result)
    audit_logger.log("reviewer", {"sections": list(paper_draft.keys())}, result)
    print("[Reviewer] Verdict: " + result["overall_verdict"])
    return result
"""

for path, content in files.items():
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Written: " + path)

print("\\nAll agent files written successfully!")