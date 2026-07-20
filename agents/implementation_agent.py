import json
import os
from groq import Groq
from tools import file_manager, code_executor
from memory import state_store, audit_logger
import config

SYSTEM_PROMPT = """You are an expert Python engineer writing machine learning experiment code.
Rules:
- Code must run on CPU only, no GPU required.
- Use only these libraries: scikit-learn, numpy, pandas, matplotlib.
- Every run must save results to a JSON file in outputs/code/results.json.
- Use argparse for hyperparameters.
- Save all plots to outputs/plots/.
- Code must be completely self-contained and runnable.
- Return ONLY the Python code. No explanation, no markdown, no code blocks."""

def run():
    client = Groq(api_key=config.GROQ_API_KEY)
    experiment_plan = state_store.get_state("experiment_plan")
    chosen_idea = state_store.get_state("chosen_idea")
    if not experiment_plan:
        raise ValueError("[ImplementationAgent] No experiment_plan in state.")

    print("[ImplementationAgent] Writing experiment code...")

    prompt = f"""Write a complete Python experiment script for this research:

Hypothesis: {chosen_idea['hypothesis']}

Experiment Plan:
{json.dumps(experiment_plan, indent=2)}

Requirements:
- Use scikit-learn, numpy, pandas, matplotlib only
- CPU only, no deep learning frameworks
- Save final metrics to: outputs/code/results.json
- Format of results.json must be:
  {{
    "metrics": [
      {{"metric_name": "accuracy", "mean": 0.85, "std": 0.02, "n_runs": 5}}
    ],
    "hypothesis_verdict": "supported",
    "key_findings": ["finding1", "finding2"]
  }}
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

Fix the error and return the complete corrected Python script. Return ONLY the code."""

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