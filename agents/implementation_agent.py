import json
from groq import Groq
from tools import file_manager, code_executor
from memory import state_store, audit_logger
from verification import planned_datasets
import config

SYSTEM_PROMPT = """You are an expert Python engineer writing machine learning experiment code.
Rules:
- Code must run on CPU only, no GPU required.
- Use ONLY these libraries: scikit-learn, numpy, pandas, matplotlib. No others
  are installed. In particular do NOT use imbalanced-learn / imblearn. For class
  imbalance prefer class_weight="balanced" on the estimator.
- The HARNESS runs the evaluation, not you. Obtain the data AND all scores ONLY
  by calling evaluate_comparison from the harness:
      from harness import evaluation
      results = evaluation.evaluate_comparison(
          dataset_name, method_estimator, baseline_estimator,
          seed=SEED, primary_metric="macro_f1",  # or "accuracy"
          n_splits=5, method_label="...", baseline_label="...")
  You define `method_estimator` and `baseline_estimator` as scikit-learn
  estimators (optionally sklearn Pipelines). The harness provisions the data,
  runs stratified K-fold cross-validation on identical folds, scores each fold,
  and writes both the data manifest and outputs/code/results.json.
- You MUST NOT load, download, or generate any dataset yourself (no load_iris,
  fetch_openml, make_classification, read_csv, etc.). The harness is the only
  data source.
- You MUST NOT run your own cross-validation, compute your own metrics, or write
  results.json or manifest.json. The harness does all measurement. Reporting a
  number you computed yourself is not permitted and will fail verification.
- Set random_state=SEED on any estimator that accepts it, for reproducibility.
- Save any plots to outputs/plots/.
- Code must be completely self-contained and runnable.
- Return ONLY the Python code. No explanation, no markdown, no code blocks."""

def run():
    client = Groq(api_key=config.GROQ_API_KEY)
    experiment_plan = state_store.get_state("experiment_plan")
    chosen_idea = state_store.get_state("chosen_idea")
    if not experiment_plan:
        raise ValueError("[ImplementationAgent] No experiment_plan in state.")

    # The dataset is decided by the plan, not the model. Injecting the exact
    # name (rather than letting the script re-derive it from the plan text)
    # is what pins conformance: the harness will fingerprint precisely this
    # dataset, and the script is handed exactly it.
    names = planned_datasets(experiment_plan)
    if not names:
        raise ValueError("[ImplementationAgent] Plan names no dataset; feasibility gate should have caught this.")
    dataset_name = names[0]
    seed = config.GLOBAL_SEED

    # Clear stale artifacts from any previous run. This is a safety property, not
    # tidiness: if a newly generated script bypasses the harness and never calls
    # provision(), we need NO manifest to remain, so the conformance gate sees
    # "missing" and halts rather than validating against last run's manifest.
    for stale in (config.OUTPUTS_CODE / "manifest.json", config.OUTPUTS_CODE / "results.json"):
        stale.unlink(missing_ok=True)

    print("[ImplementationAgent] Writing experiment code...")

    prompt = f"""Write a complete Python experiment script for this research:

Hypothesis: {chosen_idea['hypothesis']}

Experiment Plan:
{json.dumps(experiment_plan, indent=2)}

Requirements:
- Run the whole experiment with EXACTLY one harness call for data and scoring:
      from harness import evaluation
      results = evaluation.evaluate_comparison(
          "{dataset_name}",
          method_estimator,
          baseline_estimator,
          seed={seed},
          primary_metric="macro_f1",   # or "accuracy" for a balanced dataset
          n_splits=5,
          method_label="<short name of the method>",
          baseline_label="<short name of the baseline>",
      )
- Define `method_estimator` and `baseline_estimator` as scikit-learn estimators
  (or sklearn Pipelines) that operationalise the hypothesis: the method is the
  technique under test, the baseline is the control it is compared against. Set
  random_state={seed} on any estimator that accepts it. For class imbalance use
  class_weight="balanced" on the method rather than any resampling library.
- Choose primary_metric to fit the hypothesis: "macro_f1" for imbalanced data,
  "accuracy" for balanced data. These are the only supported metrics.
- Do NOT load data, run cross-validation, compute metrics, or write any JSON
  file. The harness provisions the data, evaluates both estimators on identical
  stratified folds, and writes outputs/code/results.json and the manifest.
- Use scikit-learn, numpy, pandas, matplotlib only. CPU only, under 60 seconds.
- The returned `results` dict has EXACTLY these keys:
    results["primary_metric"]                                 -> str
    results["conditions"]["method"|"baseline"]["scores"]      -> list[float] (per fold)
    results["conditions"]["method"|"baseline"]["label"]       -> str
    results["metrics"]  -> list of {{"metric_name", "mean", "std", "n_runs"}}
  If you print or plot, use ONLY these keys. Do NOT assume any other key exists
  (there is no results["method_mean"] etc.). Compute a mean as
  results["metrics"][i]["mean"] or from the scores list.
- You MAY print a short summary and save plots to outputs/plots/. Any plotting
  is optional; wrap it so a plotting error cannot crash the run.
- Save any plots to outputs/plots/
- Print progress to stdout
- Must complete in under 60 seconds on a laptop

Return ONLY the Python script code."""

    response = client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
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
    print(f"[ImplementationAgent] Code written to {script_path}")
    print("[ImplementationAgent] Running experiment...")

    result = code_executor.run_script(script_path)
    retry_count = 0

    if not result["success"] and retry_count < 2:
        print(f"[ImplementationAgent] Run failed, retrying with error feedback...")
        retry_count += 1
        fix_prompt = f"""This Python code failed with this error:
{result['stderr']}

Original code:
{code}

Fix the error and return the complete corrected Python script. Constraints still
apply: use ONLY scikit-learn, numpy, pandas, matplotlib (no imblearn or other
extra packages); do the data + evaluation ONLY via
`evaluation.evaluate_comparison("{dataset_name}", method_estimator, baseline_estimator, seed={seed}, primary_metric=..., n_splits=5)`;
define the two estimators yourself; and do NOT load data, run your own CV, or
write any JSON. The returned `results` dict has ONLY: results["conditions"]
["method"|"baseline"]["scores"] (per-fold lists) and results["metrics"] (list of
{{"metric_name","mean","std","n_runs"}}) -- do not access any other key (there is
no results["method_mean"]). Wrap any plotting so it cannot crash the run.
Return ONLY the code."""

        fix_response = client.chat.completions.create(
            model=config.MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
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
        print(f"[ImplementationAgent] Experiment completed in {result['execution_time_seconds']}s")
    else:
        print(f"[ImplementationAgent] Experiment failed: {result['stderr'][:200]}")

    return result