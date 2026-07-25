from memory import state_store
from agents import scout, gap_analyst, idea_generator
from agents import title_agent, experiment_designer
from agents import implementation_agent, results_analyst
from agents import paper_writer, reviewer
from verification import (
    audit_conformance,
    audit_feasibility,
    audit_reported_numbers,
    audit_significance,
    decide,
    evaluate,
    planned_datasets,
    render,
)
import config
import datetime
import json


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
            " (N:" + str(idea["novelty_score"]) +
            " F:" + str(idea["feasibility_score"]) +
            " I:" + str(idea["impact_score"]) + ")"
        )
    idx = get_human_approval("Choose a research idea to pursue:", options=opts)
    chosen_id = ideas[idx]["idea_id"]
    print("Chosen: " + chosen_id)

    print("\n[Stage 4/9] Title Agent...")
    state_store.snapshot_state("before_title")
    title_result = title_agent.run(chosen_idea_id=chosen_id)
    print("Title: " + title_agent.recommended_title(title_result))

    print("\n[Stage 5/9] Experiment Designer...")
    state_store.snapshot_state("before_experiment")
    exp = experiment_designer.run()

    # Feasibility gate, before the human is asked anything. Whether a dataset
    # can actually run under this budget is a question of fact, not a matter
    # of taste, so it is settled in code first. Presenting an infeasible plan
    # for approval is how the original run got a CIFAR-10-LT brief waved
    # through and then silently answered with iris.
    feasibility = audit_feasibility(exp)
    print("\n--- verification: feasibility ---")
    print(render(feasibility))
    verdict = decide(feasibility)
    if verdict.halt:
        print("\nHALTED: " + verdict.reason)
        state_store.update_state("verification_halt", {
            "stage": "experiment_design",
            "reason": verdict.reason,
            "findings": [str(f) for f in verdict.findings],
            "timestamp": datetime.datetime.now().isoformat()
        })
        return None

    summary = (
        "\nHypothesis: " + exp["hypothesis_alternative"] +
        "\nDatasets: " + str(planned_datasets(exp)) +
        "\nBaselines: " + str([b["name"] for b in exp["baselines"]]) +
        "\nRuntime estimate: " + str(exp["estimated_runtime_hours"]) + " hours"
    )
    ok = get_human_approval(summary)
    if not ok:
        print("Rejected. Exiting.")
        return None

    print("\n[Stage 6/9] Implementation Agent...")
    state_store.snapshot_state("before_implementation")
    code_result = implementation_agent.run()
    if not code_result["success"]:
        print("Experiment failed: " + code_result["stderr"][:200])
        ok = get_human_approval("Experiment failed. Continue anyway?")
        if not ok:
            return None

    # Conformance gate. The experiment has now run; before any downstream stage
    # is allowed to describe its results, prove the data it consumed is the data
    # the plan specified. The manifest is written by the harness, not the script,
    # so a script that quietly swapped datasets left no manifest (missing =>
    # critical) or one whose fingerprint will not match. This is the check that
    # would have caught the original iris substitution -- and unlike the old
    # reviewer, it runs before, not after, the paper is written.
    manifest_path = config.OUTPUTS_CODE / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    conformance = audit_conformance(exp, manifest)
    print("\n--- verification: conformance ---")
    print(render(conformance))
    verdict = decide(conformance)
    if verdict.halt:
        print("\nHALTED: " + verdict.reason)
        state_store.update_state("verification_halt", {
            "stage": "implementation",
            "reason": verdict.reason,
            "findings": [str(f) for f in verdict.findings],
            "timestamp": datetime.datetime.now().isoformat()
        })
        return None

    # Significance gate. The experiment ran on the right data; now decide, from
    # the numbers it reported, whether its claimed verdict is one the data
    # actually licenses. The verdict was previously a string an LLM chose and
    # every stage repeated -- the decorative reviewer moved into the conclusion.
    # Here it is computed by a rule fixed before the run existed. A run that
    # claims "supported" while the pre-registered test finds otherwise is
    # asserting a result it did not establish: CRITICAL, exactly like a dataset
    # swap. An honest "inconclusive" or "refuted" passes freely.
    results_path = config.OUTPUTS_CODE / "results.json"
    results_json = json.loads(results_path.read_text()) if results_path.exists() else None
    significance = audit_significance(results_json)
    print("\n--- verification: significance ---")
    print(render(significance))
    verdict = decide(significance)
    if verdict.halt:
        print("\nHALTED: " + verdict.reason)
        state_store.update_state("verification_halt", {
            "stage": "significance",
            "reason": verdict.reason,
            "findings": [str(f) for f in verdict.findings],
            "timestamp": datetime.datetime.now().isoformat()
        })
        return None

    # The verdict that survives is the computed one, not the script's claim.
    # Downstream stages report this, so the paper cannot state a conclusion the
    # test did not reach.
    assessment = evaluate(results_json or {})
    state_store.update_state("verification_verdict", {
        "verdict": assessment.verdict.value,
        "evidence": assessment.evidence,
    })

    print("\n[Stage 7/9] Results Analyst...")
    state_store.snapshot_state("before_results")
    results_analyst.run()

    print("\n[Stage 8/9] Paper Writer...")
    state_store.snapshot_state("before_paper")
    draft = paper_writer.run()

    # Reporting gate. The paper is written by an LLM from a summary; before it is
    # reviewed or exported, prove every performance figure in its prose traces to
    # a number the experiment actually produced. This closes the last place a
    # fabricated 0.95 could enter -- the Results section itself -- and it is the
    # check with real teeth, unlike the reviewer's self-reported audit booleans.
    paper_path = config.OUTPUTS_FINAL / "paper_draft.md"
    paper_text = paper_path.read_text(encoding="utf-8") if paper_path.exists() else ""
    result_summary = state_store.get_state("result_summary")
    reporting = audit_reported_numbers(paper_text, result_summary)
    print("\n--- verification: reporting ---")
    print(render(reporting))
    verdict = decide(reporting)
    if verdict.halt:
        print("\nHALTED: " + verdict.reason)
        state_store.update_state("verification_halt", {
            "stage": "reporting",
            "reason": verdict.reason,
            "findings": [str(f) for f in verdict.findings],
            "timestamp": datetime.datetime.now().isoformat()
        })
        return None

    print("\n[Stage 9/9] Reviewer...")
    state_store.snapshot_state("before_review")
    review = reviewer.run()
    print("Verdict: " + review["overall_verdict"])
    for issue in review["issues"]:
        print("  [" + issue["severity"].upper() + "] " + issue["section"] + ": " + issue["description"])

    # The number audit is now a *verified* fact: the reporting gate above would
    # have halted a fabricated figure, so reaching here means every metric-shaped
    # number in the paper traces to the results. The reviewer's own booleans are
    # LLM self-reports the orchestrator only prints, so they are labelled
    # advisory rather than presented as guarantees they never were.
    final = (
        "\nNumber conformance (VERIFIED): pass -- every figure in the paper "
        "traces to the recorded results." +
        "\nReviewer verdict (LLM, advisory): " + review["overall_verdict"] +
        "\nReviewer critical issues (advisory): " + str(sum(1 for i in review["issues"] if i["severity"] == "critical")) +
        "\nReviewer major issues (advisory): " + str(sum(1 for i in review["issues"] if i["severity"] == "major")) +
        "\nCitation audit (LLM, advisory -- not independently verified): " + str(review.get("citation_audit_passed")) +
        "\n\nPaper at: outputs/final/paper_draft.md"
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