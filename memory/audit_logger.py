import jsonlines
from datetime import datetime
from pathlib import Path
import hashlib
import json
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

    agent_log_path = config.AGENT_LOGS_DIR / f"{agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(agent_log_path, "w") as f:
        json.dump({"input": input_data, "output": output_data, "status": status}, f, indent=2)
