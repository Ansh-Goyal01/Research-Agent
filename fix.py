orchestrator_code = '''from memory import state_store, audit_logger
from agents import (scout, gap_analyst, idea_generator,
                    title_agent, experiment_designer,
                    implementation_agent, results_analyst,
                    paper_writer, reviewer)
import datetime


def get_human_approval(prompt, options=None):
    print("\\n" + "="*60)
    print("HUMAN APPROVAL REQUIRED")
    print("="*60)
    print(prompt)
    if options:
        for i, opt in enumerate(options, 1):
            print(f"  [{i}] {opt}")
        while True:
            choice = input("\\nYour choice (number): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(options):
                return int(choice) - 1
            print("Invalid choice. Try again.")
    else:
        response = input("\\nApprove? (y/n): ").strip().lower()
        return response == "y"


def run(topic_data):
    print("\\n" + "="*60)
    print("RESEARCH AGENT PIPELINE STARTING")
    print(f"Topic: {topic_data[\'topic\']}")
    print("="*60)

    state_store.update_state("input_topic", topic_data)

    print("\\n[Stage 1/9] Running Scout Agent...")
    state_store.snapshot_state("before_scout")
    scout_result = scout.run(topic_data)
    print(f"Papers found: {len(scout_result[\'papers\'])}")

    print("\\n[Stage 2/9] Running Gap Analyst...")
    state_store.snapshot_state("before_gap_analyst")
    gap_result = gap_analyst.run()

    print("\\n[Stage 3/9] Running Idea Generator...")
    state_store.snapshot_state("before_idea_generator")
    idea_result = idea_generator.run()

    ideas = idea_result["ideas"]
    idea_options = []
    for idea in ideas:
        idea_options.append(
            f"{idea[\'idea_id\']}: {idea[\'hypothesis\'][:80]}... "
            f"(Novelty: {idea[\'novelty_score\']}, "
            f"Feasibility: {idea[\'feasibility_score\']}, "
            f"Impact: {idea[\'impact_score\']})"
        )

    chosen_idx = get_human_approval(
        "Choose a research idea to pursue:",
        options=idea_options
    )
    chosen_idea_id = ideas[chosen_idx]["idea_id"]
    print(f"\\nChosen idea: {chosen_idea_id}")

    print("\\n[Stage 4/9] Running Title Agent...")
    state_store.snapshot_state("before_title")
    title_result = title_agent.run(chosen_idea_id=chosen_idea_id)
    print(f"Descriptive: {title_result[\'descriptive\']}")
    print(f"Punchy:      {title_result[\'punchy\']}")
    print(f"Question:    {title_result[\'question_form\']}")

    print("\\n[Stage 5/9] Running Experiment Designer...")
    state_store.snapshot_state("before_experiment")
    exp_result = experiment_designer.run()

    exp_summary = (
        f"\\nExperiment Plan Summary:"
        f"\\n  Hypothesis: {exp_result[\'hypothesis_alternative\']}"
        f"\\n  Datasets: {[d[\'name\'] for d in exp_result[\'datasets\']]}"
        f"\\n  Baselines: {[b[\'name\'] for b in exp_result[\'baselines\']]}"
        f"\\n  Estimated runtime: {exp_result[\'estimated_runtime_hours\']} hours"
        f"\\n  Compute: {exp_result[\'compute_requirements\']}"
    )
    approved = get_human_approval(exp_summary)
    if not approved:
        print("Experiment plan rejected. Exiting.")
        return None

    print("\\n[Stage 6/9] Running Implementation Agent...")
    state_store.snapshot_state("before_implementation")
    code_result = implementation_agent.run()

    if not code_result["success"]:
        print(f"\\nExperiment failed: {code_result[\'stderr\'][:300]}")
        approved = get_human_approval("Experiment failed. Continue anyway?")
        if not approved:
            return None

    print("\\n[Stage 7/9] Running Results Analyst...")
    state_store.snapshot_state("before_results")
    results = results_analyst.run()

    print("\\n[Stage 8/9] Running Paper Writer...")
    state_store.snapshot_state("before_paper_writer")
    draft = paper_writer.run()

    print("\\n[Stage 9/9] Running Reviewer...")
    state_store.snapshot_state("before_reviewer")
    review = reviewer.run()

    print(f"\\nReview verdict: {review[\'overall_verdict\']}")
    if review["issues"]:
        print("Issues found:")
        for issue in review["issues"]:
            print(f"  [{issue[\'severity\'].upper()}] {issue[\'section\']}: {issue[\'description\']}")

    final_summary = (
        f"\\nFinal Review Summary:"
        f"\\n  Verdict: {review[\'overall_verdict\']}"
        f"\\n  Critical issues: {sum(1 for i in review[\'issues\'] if i[\'severity\'] == \'critical\')}"
        f"\\n  Major issues: {sum(1 for i in review[\'issues\'] if i[\'severity\'] == \'major\')}"
        f"\\n  Minor issues: {sum(1 for i in review[\'issues\'] if i[\'severity\'] == \'minor\')}"
        f"\\n  Citation audit passed: {review[\'citation_audit_passed\']}"
        f"\\n  Number audit passed: {review[\'number_audit_passed\']}"
        f"\\n\\nPaper draft saved to: outputs/final/paper_draft.md"
    )

    approved = get_human_approval(final_summary + "\\n\\nApprove final paper for export?")
    if not approved:
        print("Final paper not approved. Draft saved.")
        return None

    state_store.update_state("final_decision", {
        "human_approved": True,
        "approver_notes": "Approved via CLI",
        "export_formats": ["markdown"],
        "timestamp": datetime.datetime.now().isoformat()
    })

    print("\\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("Paper saved to: outputs/final/paper_draft.md")
    print("="*60)
    return draft
'''

main_code = '''from orchestrator import run

if __name__ == "__main__":
    topic_data = {
        "topic": "few-shot learning for image classification",
        "domain": "computer vision",
        "target_venue": "CVPR 2026",
        "date_range": [2021, 2025],
        "max_papers": 15
    }
    run(topic_data)
'''

with open("orchestrator.py", "w", encoding="utf-8") as f:
    f.write(orchestrator_code)
print("orchestrator.py written")

with open("main.py", "w", encoding="utf-8") as f:
    f.write(main_code)
print("main.py written")
