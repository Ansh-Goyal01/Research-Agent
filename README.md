# Research Agent 🔬

An automated multi-agent system that discovers research gaps, generates ideas, runs experiments, and drafts full research papers.

## Architecture
10 specialized AI agents powered by Groq (Llama 3.3 70B):
- Scout → Gap Analyst → Idea Generator → Title Agent
- Experiment Designer → Implementation → Results Analyst
- Paper Writer → Reviewer → Orchestrator

## Setup
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Create `config.py` with your Groq API key:
```python
from pathlib import Path
BASE_DIR = Path(__file__).parent
GROQ_API_KEY = "your_groq_key_here"
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
```
4. Run: `python main.py`

## Usage
Edit the topic in `main.py`:
```python
topic_data = {
    "topic": "your research topic here",
    "domain": "your domain",
    "target_venue": "Conference 2026",
    "date_range": [2021, 2025],
    "max_papers": 15
}
```

## Output
- Full paper draft: `outputs/final/paper_draft.md`
- Experiment code: `outputs/code/experiment.py`
- Results: `outputs/code/results.json`
- Plots: `outputs/plots/`

## Stack
- LLM: Groq API (Llama 3.3 70B) — free tier
- Paper search: arXiv API + Semantic Scholar API
- Experiment execution: Python subprocess sandbox
- State management: TinyDB / JSON