from memory import state_store
from agents import scout, gap_analyst, idea_generator
from agents import title_agent, experiment_designer
from agents import implementation_agent, results_analyst
from agents import paper_writer, reviewer
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

    print("\n[Stage 1/9] Scout Agent...")
    state_store.snapshot_state("before_scout")
    scout_result = scout.run(topic_data)
    print("Papers found: " + str(len(scout_result["papers"])))

    print("\n[Stage 2/9] Gap Analyst...")
    state_store.snapshot_state("before_gap")
    gap_analyst.run()

    print("\n[Stage 3/9] Idea Generator...")
    state_store.snapshot_state("before_ideas")
    idea_result = idea_generator.run()
    ideas = idea_result["ideas"]
    opts = []
    for idea in ideas:
        opts.append(
            idea["idea_id"] + ": " + idea["hypothesis"][:80] +
            " (Novelty:" + str(idea["novelty_score"]) +
            " Feasibility:" + str(idea["feasibility_score"]) +
            " Impact:" + str(idea["impact_score"]) + ")"
        )
    idx = get_human_approval("Choose a research idea to pursue:", options=opts)
    chosen_id = ideas[idx]["idea_id"]
    print("Chosen: " + chosen_id)

    print("\n[Stage 4/9] Title Agent...")
    state_store.snapshot_state("before_title")
    title_result = title_agent.run(chosen_idea_id=chosen_id)
    print("Title: " + title_result[title_result["recommended"]])

    print("\n[Stage 5/9] Experiment Designer...")
    state_store.snapshot_state("before_experiment")
    exp = experiment_designer.run()
    summary = (
        "\nHypothesis: " + exp["hypothesis_alternative"] +
        "\nDatasets: " + str([d["name"] for d in exp["datasets"]]) +
        "\nBaselines: " + str([b["name"] for b in exp["baselines"]]) +
        "\nRuntime estimate: " + str(exp["estimated_runtime_hours"]) + " hours" +
        "\nCompute: " + exp["compute_requirements"]
    )
    ok = get_human_approval(summary)
    if not ok:
        print("Experiment plan rejected. Exiting.")
        return None

    print("\n[Stage 6/9] Implementation Agent...")
    state_store.snapshot_state("before_implementation")
    code_result = implementation_agent.run()
    if not code_result["success"]:
        print("Experiment failed: " + code_result["stderr"][:200])
        ok = get_human_approval("Experiment failed. Continue anyway?")
        if not ok:
            return None

    print("\n[Stage 7/9] Results Analyst...")
    state_store.snapshot_state("before_results")
    results_analyst.run()

    print("\n[Stage 8/9] Paper Writer...")
    state_store.snapshot_state("before_paper")
    draft = paper_writer.run()

    print("\n[Stage 9/9] Reviewer...")
    state_store.snapshot_state("before_review")
    review = reviewer.run()
    print("Verdict: " + review["overall_verdict"])
    for issue in review["issues"]:
        print("  [" + issue["severity"].upper() + "] " + issue["section"] + ": " + issue["description"])

    final = (
        "\nVerdict: " + review["overall_verdict"] +
        "\nCritical issues: " + str(sum(1 for i in review["issues"] if i["severity"] == "critical")) +
        "\nMajor issues: " + str(sum(1 for i in review["issues"] if i["severity"] == "major")) +
        "\nMinor issues: " + str(sum(1 for i in review["issues"] if i["severity"] == "minor")) +
        "\nCitation audit passed: " + str(review["citation_audit_passed"]) +
        "\nNumber audit passed: " + str(review["number_audit_passed"]) +
        "\n\nPaper saved at: outputs/final/paper_draft.md"
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

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("Paper saved to: outputs/final/paper_draft.md")
    print("="*60)
    return draft