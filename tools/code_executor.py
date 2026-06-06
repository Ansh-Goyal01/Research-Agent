import subprocess
import time
from pathlib import Path
import config

def run_script(script_path, timeout=None):
    timeout = timeout or config.CODE_EXECUTION_TIMEOUT
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
    try:
        result = subprocess.run(
            ["python", str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(config.BASE_DIR)
        )
        elapsed = round(time.time() - start, 2)
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout[:3000],
            "stderr": result.stderr[:1000],
            "execution_time_seconds": elapsed,
            "output_files": []
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": "Timed out after " + str(timeout) + " seconds",
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
