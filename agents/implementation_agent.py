import json
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
        "Write a complete Python experiment script for:\n"
        "Hypothesis: " + chosen_idea["hypothesis"] + "\n\n"
        "Plan:\n" + json.dumps(experiment_plan, indent=2) + "\n\n"
        "Requirements:\n"
        "- Use only scikit-learn, numpy, pandas, matplotlib\n"
        "- CPU only\n"
        "- Save metrics to outputs/code/results.json in format:\n"
        "  {metrics: [{metric_name, mean, std, n_runs}], hypothesis_verdict, key_findings}\n"
        "- Save plots to outputs/plots/\n"
        "- Must complete in under 60 seconds\n"
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
            "Fix this Python code that failed with:\n" +
            result["stderr"] + "\n\nCode:\n" + code +
            "\n\nReturn ONLY the fixed Python code."
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
