from memory import state_store
from memory.consistency_checker import run_all_checks
from agents import scout, gap_analyst, idea_generator
from agents import title_agent, experiment_designer
from agents import implementation_agent, results_analyst
from agents import paper_writer, reviewer
from agents import related_work_agent
import datetime


def get_human_approval(prompt, options=None):
    print("\n" + "="*60)
    print("HUMAN APPROVAL REQUIRED")
    print("="*60)
    print(prompt)
    if options:
        for i, opt in enumerate(options, 1):
            print(f"  [{i}] {opt}")
        while True:
            choice = input("Your choice (number): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(options):
                return int(choice) - 1
            print("Invalid choice. Try again.")
    else:
        response = input("Approve? (y/n): ").strip().lower()
        return response == "y"


def run(topic_data):
    print("\n" + "="*60)
    print("RESEARCH AGENT PIPELINE STARTING")
    print("Topic: " + topic_data["topic"])
    print("="*60)

    state_store.update_state("input_topic", topic_data)

    print("\n[Stage 1/10] Scout Agent...")
    state_store.snapshot_state("before_scout")
    scout_result = scout.run(topic_data)
    print("Papers found: " + str(len(scout_result["papers"])))

    print("\n[Stage 2/10] Gap Analyst...")
    state_store.snapshot_state("before_gap")
    gap_analyst.run()

    print("\n[Stage 3/10] Related Work Synthesis...")
    state_store.snapshot_state("before_related_work")
    related_work_agent.run()

    print("\n[Stage 4/10] Idea Generator...")
    state_store.snapshot_state("before_ideas")
    idea_result = idea_generator.run()
    ideas = idea_result["ideas"]
    opts = []
    for idea in ideas:
        opts.append(
            idea["idea_id"] + ": " + idea["hypothesis"][:80] +
            " (N:" + str(idea["novelty_score"]) +
            " F:" + str(idea["feasibility_score"]) +
            " I:" + str(idea["impact_score"]) + ")"
        )
    idx = get_human_approval("Choose a research idea to pursue:", options=opts)
    chosen_id = ideas[idx]["idea_id"]
    print("Chosen: " + chosen_id)

    print("\n[Stage 5/10] Title Agent...")
    state_store.snapshot_state("before_title")
    title_result = title_agent.run(chosen_idea_id=chosen_id)
    print("Title: " + title_result[title_result["recommended"]])

    print("\n[Stage 6/10] Experiment Designer...")
    state_store.snapshot_state("before_experiment")
    exp = experiment_designer.run()
    summary = (
        "\nHypothesis: " + exp["hypothesis_alternative"] +
        "\nDatasets: " + str([d["name"] for d in exp["datasets"]]) +
        "\nBaselines: " + str([b["name"] for b in exp["baselines"]]) +
        "\nRuntime: " + str(exp["estimated_runtime_hours"]) + " hours" +
        "\nCompute: " + exp["compute_requirements"]
    )
    ok = get_human_approval(summary)
    if not ok:
        print("Experiment plan rejected. Exiting.")
        return None
    print("\n[Architecture] Generating system architecture diagram...")
    from tools.diagram_generator import generate_architecture
    title_state = state_store.get_state("title_options") or {}
    acronym = title_state.get("method_acronym", "")
    methodology_text = ""
    generate_architecture(
        topic_data["topic"],
        topic_data.get("domain", ""),
        acronym,
        methodology_text
    )
    print("[Architecture] Done.")
    print("\n[Stage 7/10] Implementation Agent...")
    state_store.snapshot_state("before_implementation")
    code_result = implementation_agent.run()
    if not code_result["success"]:
        print("Experiment failed: " + code_result["stderr"][:200])
        ok = get_human_approval("Experiment failed. Continue anyway?")
        if not ok:
            return None

    print("\n[Stage 8/10] Results Analyst...")
    state_store.snapshot_state("before_results")
    result_summary = results_analyst.run()

    print("\n[Stage 9/10] Paper Writer...")
    state_store.snapshot_state("before_paper")
    draft = paper_writer.run()

    print("\n[Consistency Check] Running pre-review checks...")
    paper_list_state = state_store.get_state("paper_list")
    consistency_issues = run_all_checks(draft, result_summary, paper_list_state)
    state_store.update_state("consistency_issues", consistency_issues)
    short_sections = [i for i in consistency_issues if i["type"] == "short_section"]
    if len(short_sections) > 3:
        print("[ConsistencyChecker] WARNING: " + str(len(short_sections)) + " sections too short")

    print("\n[Stage 10/10] Reviewer...")
    state_store.snapshot_state("before_review")
    review = reviewer.run()
    print("Verdict: " + review["overall_verdict"])
    print("Quality score: " + str(review.get("quality_score", "N/A")) + "/10")

    if review.get("strengths"):
        print("Strengths:")
        for s in review["strengths"]:
            print("  + " + s)

    for issue in review.get("issues", []):
        print("  [" + issue["severity"].upper() + "] " + issue["section"] + ": " + issue["description"])

    final = (
        "\nVerdict: " + review["overall_verdict"] +
        "\nQuality Score: " + str(review.get("quality_score", "N/A")) + "/10" +
        "\nCritical: " + str(sum(1 for i in review["issues"] if i["severity"] == "critical")) +
        "\nMajor: " + str(sum(1 for i in review["issues"] if i["severity"] == "major")) +
        "\nMinor: " + str(sum(1 for i in review["issues"] if i["severity"] == "minor")) +
        "\nConsistency issues: " + str(len(consistency_issues)) +
        "\nCitation audit: " + str(review["citation_audit_passed"]) +
        "\nNumber audit: " + str(review["number_audit_passed"]) +
        "\n\nStrengths:\n" + "\n".join(["  + " + s for s in review.get("strengths", [])]) +
        "\n\nPaper: outputs/final/paper_draft.md"
    )
    ok = get_human_approval(final + "\n\nApprove final paper for export?")
    if not ok:
        print("Not approved. Draft saved.")
        return None

    state_store.update_state("final_decision", {
        "human_approved": True,
        "approver_notes": "Approved via CLI",
        "export_formats": ["markdown"],
        "timestamp": datetime.datetime.now().isoformat()
    })

   from tools.export import export_html
    from memory import state_store as ss
    title_opts = ss.get_state("title_options")
    final_title = title_opts[title_opts["recommended"]] if title_opts else "Research Paper"
    result_sum = ss.get_state("result_summary") or {}
    html_path = export_html(draft, result_sum, final_title)

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("Markdown : outputs/final/paper_draft.md")
    print("HTML     : outputs/final/paper.html")
    print("Figures  : outputs/plots/ (5 figures)")
    print("="*60)
    return draft