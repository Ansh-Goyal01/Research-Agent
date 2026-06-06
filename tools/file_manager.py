import json
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
