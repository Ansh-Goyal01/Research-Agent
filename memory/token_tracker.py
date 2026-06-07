import json
from datetime import datetime, date
import config


TRACKER_FILE = config.BASE_DIR / "state" / "token_usage.json"


def _load():
    try:
        if TRACKER_FILE.exists():
            with open(TRACKER_FILE, "r") as f:
                data = json.load(f)
            if data.get("date") != str(date.today()):
                return {"date": str(date.today()), "total": 0, "by_agent": {}}
            return data
    except Exception:
        pass
    return {"date": str(date.today()), "total": 0, "by_agent": {}}


def _save(data):
    with open(TRACKER_FILE, "w") as f:
        json.dump(data, f, indent=2)


def estimate_tokens(text):
    return len(str(text)) // 4


def log_usage(agent_name, prompt, response):
    data = _load()
    used = estimate_tokens(prompt) + estimate_tokens(response)
    data["total"] += used
    data["by_agent"][agent_name] = data["by_agent"].get(agent_name, 0) + used
    _save(data)
    return used


def get_total():
    return _load().get("total", 0)


def get_remaining():
    return max(0, 100000 - get_total())


def print_status(agent_name, used):
    total = get_total()
    remaining = get_remaining()
    bar_filled = int((total / 100000) * 20)
    bar = "█" * bar_filled + "░" * (20 - bar_filled)
    print(f"[Tokens] {agent_name}: +{used:,} | Today: {total:,}/100,000 [{bar}] {remaining:,} left")