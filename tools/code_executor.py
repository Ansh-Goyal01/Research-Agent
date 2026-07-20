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
            "stderr": f"Script not found: {script_path}",
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
        output_files = _find_new_outputs(start)
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout[:3000],
            "stderr": result.stderr[:1000],
            "execution_time_seconds": elapsed,
            "output_files": output_files
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Timed out after {timeout} seconds",
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

def _find_new_outputs(since_timestamp):
    found = []
    for folder in [config.OUTPUTS_CODE, config.OUTPUTS_PLOTS]:
        for f in Path(folder).glob("*"):
            if f.is_file() and f.stat().st_mtime >= since_timestamp:
                found.append(str(f))
    return found