import json
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
