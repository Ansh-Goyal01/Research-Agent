"""
code_executor.py  — Research Agent
Runs experiment scripts as subprocesses.

Fix: default timeout raised to 3600s (60 min) to match i5-13200H budget.
stdout/stderr capture limits raised so full experiment output is preserved.
"""
import subprocess
import time
from pathlib import Path
import config


def run_script(script_path, timeout=None):
    # Default 60-minute budget for i5-13200H 16GB DDR5
    timeout = timeout or getattr(config, "CODE_EXECUTION_TIMEOUT", 3600)
    script_path = Path(script_path)

    if not script_path.exists():
        return {
            "success": False,
            "stdout": "",
            "stderr": "Script not found: " + str(script_path),
            "execution_time_seconds": 0.0,
            "output_files": []
        }

    start = time.time()
    print("[CodeExecutor] Running: " + str(script_path) +
          " (timeout=" + str(timeout) + "s)")
    try:
        result = subprocess.run(
            ["python", str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(config.BASE_DIR)
        )
        elapsed = round(time.time() - start, 2)
        success = result.returncode == 0

        # Print live output summary
        if result.stdout:
            for line in result.stdout.strip().split("\n")[-10:]:
                print("  " + line)

        return {
            "success": success,
            "stdout": result.stdout[:10000],   # raised from 3000
            "stderr": result.stderr[:5000],    # raised from 1000
            "execution_time_seconds": elapsed,
            "output_files": []
        }
    except subprocess.TimeoutExpired:
        elapsed = round(time.time() - start, 2)
        print("[CodeExecutor] TIMEOUT after " + str(elapsed) + "s")
        return {
            "success": False,
            "stdout": "",
            "stderr": "Timed out after " + str(timeout) + "s (budget exhausted)",
            "execution_time_seconds": float(timeout),
            "output_files": []
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "execution_time_seconds": round(time.time() - start, 2),
            "output_files": []
        }
