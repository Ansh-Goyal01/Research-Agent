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


def _already_done(key):
    return state_store.get_state(key) is not None


def run(topic_data, resume=False):
    print("\n" + "="*60)
    print("RESEARCH AGENT PIPELINE STARTING")
    print("Topic: " + topic_data["topic"])
    if resume:
        print("MODE: Resuming from last checkpoint")
    print("="*60)

    if not resume:
        state_store.write_state({})

    state_store.update_state("input_topic", topic_data)

    # Stage 1
    if not resume or not _already_done("paper_list"):
        print("\n[Stage 1/11] Scout Agent...")
        state_store.snapshot_state("before_scout")
        scout_result = scout.run(topic_data)
        print("Papers found: " + str(len(scout_result["papers"])))
    else:
        print("\n[Stage 1/11] Scout Agent... SKIPPED (already done)")

    # Stage 2
    if not resume or not _already_done("gap_analysis"):
        print("\n[Stage 2/11] Gap Analyst...")
        state_store.snapshot_state("before_gap")
        gap_analyst.run()
    else:
        print("\n[Stage 2/11] Gap Analyst... SKIPPED")

    # Stage 3
    if not resume or not _already_done("related_work_outline"):
        print("\n[Stage 3/11] Related Work Agent...")
        state_store.snapshot_state("before_related_work")
        related_work_agent.run()
    else:
        print("\n[Stage 3/11] Related Work Agent... SKIPPED")

    # Stage 4 — always show ideas even on resume
    if not resume or not _already_done("idea_candidates"):
        print("\n[Stage 4/11] Idea Generator...")
        state_store.snapshot_state("before_ideas")
        idea_generator.run()

    if not resume or not _already_done("chosen_idea"):
        idea_result = state_store.get_state("idea_candidates")
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
    else:
        chosen_id = state_store.get_state("chosen_idea", {}).get("idea_id")
        print("\n[Stage 4/11] Idea Generator... SKIPPED (using: " + str(chosen_id) + ")")

    # Stage 5
    if not resume or not _already_done("title_options"):
        print("\n[Stage 5/11] Title Agent...")
        state_store.snapshot_state("before_title")
        title_result = title_agent.run(chosen_idea_id=chosen_id)
        print("Title: " + title_result[title_result["recommended"]])
    else:
        print("\n[Stage 5/11] Title Agent... SKIPPED")

    # Stage 6
    if not resume or not _already_done("experiment_plan"):
        print("\n[Stage 6/11] Experiment Designer...")
        state_store.snapshot_state("before_experiment")
        exp = experiment_designer.run()
        summary = (
            "\nHypothesis: " + exp["hypothesis_alternative"] +
            "\nDatasets: " + str([d["name"] for d in exp["datasets"]]) +
            "\nBaselines: " + str([b["name"] for b in exp["baselines"]]) +
            "\nRuntime: " + str(exp["estimated_runtime_hours"]) + " hours"
        )
        ok = get_human_approval(summary)
        if not ok:
            print("Experiment plan rejected. Exiting.")
            return None
    else:
        print("\n[Stage 6/11] Experiment Designer... SKIPPED")

    # Architecture diagram
    if not resume or not (config.OUTPUTS_PLOTS / "fig0_architecture.png").exists():
        print("\n[Architecture] Generating system architecture diagram...")
        from tools.diagram_generator import generate_architecture
        import config
        title_state = state_store.get_state("title_options") or {}
        acronym = title_state.get("method_acronym", "")
        generate_architecture(topic_data["topic"], topic_data.get("domain", ""), acronym, "")
        print("[Architecture] Done.")
    else:
        print("\n[Architecture] Diagram... SKIPPED")

    # Stage 7
    if not resume or not _already_done("code_result"):
        print("\n[Stage 7/11] Implementation Agent...")
        state_store.snapshot_state("before_implementation")
        code_result = implementation_agent.run()
        if not code_result["success"]:
            print("Experiment failed: " + code_result["stderr"][:200])
            ok = get_human_approval("Experiment failed. Continue anyway?")
            if not ok:
                return None
    else:
        print("\n[Stage 7/11] Implementation Agent... SKIPPED")

    # Stage 8
    if not resume or not _already_done("result_summary"):
        print("\n[Stage 8/11] Results Analyst...")
        state_store.snapshot_state("before_results")
        result_summary = results_analyst.run()
    else:
        result_summary = state_store.get_state("result_summary")
        print("\n[Stage 8/11] Results Analyst... SKIPPED")

    # Stage 9
    if not resume or not _already_done("paper_draft"):
        print("\n[Stage 9/11] Paper Writer...")
        state_store.snapshot_state("before_paper")
        draft = paper_writer.run()
    else:
        draft = state_store.get_state("paper_draft")
        print("\n[Stage 9/11] Paper Writer... SKIPPED")

    # Consistency check
    print("\n[Consistency Check] Running pre-review checks...")
    paper_list_state = state_store.get_state("paper_list")
    consistency_issues = run_all_checks(draft, result_summary, paper_list_state)
    state_store.update_state("consistency_issues", consistency_issues)

    # Stage 10
    print("\n[Stage 10/11] Reviewer...")
    state_store.snapshot_state("before_review")
    review = reviewer.run()
    print("Verdict: " + review["overall_verdict"])
    print("Quality score: " + str(review.get("quality_score", "N/A")) + "/10")
    if review.get("strengths"):
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
        "\nPaper: outputs/final/paper_draft.md"
    )
    ok = get_human_approval(final + "\n\nApprove final paper for export?")
    if not ok:
        print("Not approved. Draft saved.")
        return None

    # Export
    print("\n[Stage 11/11] Exporting...")
    from tools.export import export_html
    title_opts = state_store.get_state("title_options")
    final_title = title_opts[title_opts["recommended"]] if title_opts else "Research Paper"
    export_html(draft, result_summary, final_title)

    state_store.update_state("final_decision", {
        "human_approved": True,
        "timestamp": datetime.datetime.now().isoformat()
    })

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("Markdown : outputs/final/paper_draft.md")
    print("HTML     : outputs/final/paper.html")
    print("Figures  : outputs/plots/ (6 figures)")
    print("="*60)
    return draft


import config