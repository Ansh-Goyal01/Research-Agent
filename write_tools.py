files = {}

files["tools/paper_search.py"] = """import requests
import urllib.parse
import xml.etree.ElementTree as ET
import config

def search_arxiv(query, max_results=10, start_year=2021):
    params = {
        "search_query": "all:" + query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }
    url = config.ARXIV_BASE_URL + "?" + urllib.parse.urlencode(params)
    response = requests.get(url, timeout=15)
    root = ET.fromstring(response.content)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    papers = []
    for entry in root.findall("atom:entry", ns):
        try:
            year = int(entry.find("atom:published", ns).text[:4])
            if year < start_year:
                continue
            authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]
            papers.append({
                "title": entry.find("atom:title", ns).text.strip().replace("\\n", " "),
                "authors": authors[:5],
                "year": year,
                "venue": "arXiv",
                "url": entry.find("atom:id", ns).text.strip(),
                "abstract": entry.find("atom:summary", ns).text.strip().replace("\\n", " ")[:500],
                "semantic_scholar_id": None
            })
        except Exception:
            continue
    return papers

def search_semantic_scholar(query, max_results=10):
    url = config.SEMANTIC_SCHOLAR_BASE_URL + "/paper/search"
    params = {
        "query": query,
        "limit": max_results,
        "fields": "title,authors,year,venue,externalIds,abstract"
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        data = response.json()
        papers = []
        for p in data.get("data", []):
            if not p.get("year") or p["year"] < 2021:
                continue
            papers.append({
                "title": p.get("title", ""),
                "authors": [a["name"] for a in p.get("authors", [])[:5]],
                "year": p.get("year", 0),
                "venue": p.get("venue", "Unknown"),
                "url": "https://www.semanticscholar.org/paper/" + p.get("paperId", ""),
                "abstract": p.get("abstract", "")[:500] if p.get("abstract") else "",
                "semantic_scholar_id": p.get("paperId")
            })
        return papers
    except Exception:
        return []

def search_papers(query, max_results=25, start_year=2021):
    arxiv_results = search_arxiv(query, max_results=max_results//2, start_year=start_year)
    ss_results = search_semantic_scholar(query, max_results=max_results//2)
    seen = set()
    combined = []
    for p in arxiv_results + ss_results:
        key = p["title"].lower()[:50]
        if key not in seen:
            seen.add(key)
            combined.append(p)
    return combined[:max_results]
"""

files["tools/file_manager.py"] = """import json
import shutil
from pathlib import Path
from datetime import datetime
import config

ALLOWED_DIRS = [
    config.OUTPUTS_CODE,
    config.OUTPUTS_PLOTS,
    config.OUTPUTS_SECTIONS,
    config.OUTPUTS_FINAL,
    config.OUTPUTS_ARCHIVE,
    config.BASE_DIR / "state",
    config.BASE_DIR / "logs"
]

def _is_allowed(path):
    path = Path(path).resolve()
    return any(str(path).startswith(str(d.resolve())) for d in ALLOWED_DIRS)

def safe_write(path, content, mode="w"):
    path = Path(path)
    if not _is_allowed(path):
        raise PermissionError("Write to " + str(path) + " not allowed.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        f.write(content)
    return path

def safe_read(path):
    with open(Path(path), "r", encoding="utf-8") as f:
        return f.read()

def save_section(section_name, content):
    path = config.OUTPUTS_SECTIONS / (section_name + ".md")
    return safe_write(path, content)

def save_final(filename, content):
    final_path = config.OUTPUTS_FINAL / filename
    if final_path.exists():
        archive_name = filename + "." + datetime.now().strftime("%Y%m%d_%H%M%S") + ".bak"
        shutil.copy(final_path, config.OUTPUTS_ARCHIVE / archive_name)
    return safe_write(final_path, content)

def save_json(path, data):
    return safe_write(path, json.dumps(data, indent=2, default=str))

def load_json(path):
    return json.loads(safe_read(path))
"""

files["tools/code_executor.py"] = """import subprocess
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
"""

files["memory/state_store.py"] = """import json
from datetime import datetime
import config

def read_state():
    if not config.STATE_FILE.exists():
        return {}
    with open(config.STATE_FILE, "r") as f:
        return json.load(f)

def write_state(data):
    with open(config.STATE_FILE, "w") as f:
        json.dump(data, f, indent=2)

def update_state(key, value):
    state = read_state()
    state[key] = value
    state["last_updated"] = datetime.now().isoformat()
    write_state(state)

def get_state(key, default=None):
    return read_state().get(key, default)

def snapshot_state(stage_name):
    state = read_state()
    path = config.BASE_DIR / "state" / ("snapshot_" + stage_name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")
    with open(path, "w") as f:
        json.dump(state, f, indent=2)
    return path
"""

files["memory/audit_logger.py"] = """import jsonlines
import json
import hashlib
from datetime import datetime
import config

def _hash(data):
    return hashlib.md5(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()[:8]

def log(agent_name, input_data, output_data, status="success"):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": agent_name,
        "input_hash": _hash(input_data),
        "output_hash": _hash(output_data),
        "status": status
    }
    with jsonlines.open(config.AUDIT_LOG, mode="a") as writer:
        writer.write(entry)
    log_path = config.AGENT_LOGS_DIR / (agent_name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")
    with open(log_path, "w") as f:
        json.dump({"input": input_data, "output": output_data, "status": status}, f, indent=2)
"""

for path, content in files.items():
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Written: " + path)

print("All files written!")