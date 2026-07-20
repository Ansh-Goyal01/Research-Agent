import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent

load_dotenv(BASE_DIR / ".env")

# Never hardcode the key here. Put it in .env (gitignored):
#     GROQ_API_KEY=gsk_your_key_here
# Get a free key from console.groq.com
#
# Empty is a valid state: demo mode replays a recorded run and spends zero
# tokens, so the server must boot without a key. Agents that actually call
# Groq validate this at the call site instead of failing at import.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 4096
MAX_RETRIES = 3

STATE_FILE       = BASE_DIR / "state" / "workflow_state.json"
AUDIT_LOG        = BASE_DIR / "logs" / "audit.jsonl"
AGENT_LOGS_DIR   = BASE_DIR / "logs" / "agent_calls"
OUTPUTS_CODE     = BASE_DIR / "outputs" / "code"
OUTPUTS_PLOTS    = BASE_DIR / "outputs" / "plots"
OUTPUTS_SECTIONS = BASE_DIR / "outputs" / "sections"
OUTPUTS_FINAL    = BASE_DIR / "outputs" / "final"
OUTPUTS_ARCHIVE  = BASE_DIR / "outputs" / "final" / "archive"

ARXIV_BASE_URL            = "https://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org/graph/v1"
MAX_PAPERS = 25

CODE_EXECUTION_TIMEOUT = 60
