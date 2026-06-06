Research Agent

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-F55036?style=flat-square)](https://console.groq.com)
[![arXiv](https://img.shields.io/badge/Search-arXiv_+_Semantic_Scholar-B31B1B?style=flat-square)](https://arxiv.org)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active_Development-22C55E?style=flat-square)]()
[![Free](https://img.shields.io/badge/Cost-100%25_Free_to_Run-A855F7?style=flat-square)]()

> An end-to-end multi-agent system that takes a research topic and autonomously searches literature, finds gaps, proposes hypotheses, runs real experiments, and writes a complete IEEE-style paper draft — with human approval at every critical step.

---

## The idea

Most AI writing tools generate plausible-sounding text. This is different.

Every claim in the output is grounded in papers actually fetched from arXiv and Semantic Scholar. Every number in the results section comes from a Python experiment that actually ran on your machine. Every citation is verified before the paper is written.

Change one line. Get a research paper draft.

```python
TOPIC = "explainable AI for urban traffic congestion prediction"
```

---

## How it works

Ten agents run in sequence, each with a single responsibility:
Scout → Gap Analyst → Idea Generator → ✋ approve
→ Title → Experiment Designer  → ✋ approve
→ Implementation → Results Analyst
→ Paper Writer → Reviewer       → ✋ approve
→ outputs/final/paper_draft.md

Three checkpoints where you decide whether to continue. Everything else is fully automated.

---

## What each agent does

| Agent | Input | Output |
|---|---|---|
| Scout | research topic | 20 verified papers from arXiv + S2 |
| Gap Analyst | paper list | ranked research gaps with scores |
| Idea Generator | gaps | 3 falsifiable hypotheses |
| Title Agent | chosen idea | 3 title variants + recommendation |
| Experiment Designer | idea | full experiment plan, CPU-only |
| Implementation | plan | working Python script + results.json |
| Results Analyst | experiment output | metrics, verdict, findings |
| Paper Writer | all of the above | 9-section IEEE-style draft |
| Reviewer | draft + source data | citation audit, number audit, score |
| Orchestrator | everything | manages flow, retries, approvals |

---

## Setup

**1. Clone**
```bash
git clone https://github.com/Ansh-Goyal01/Research-Agent.git
cd Research-Agent
```

**2. Install**
```bash
pip install -r requirements.txt
```

**3. Get a free Groq key**

Go to [console.groq.com](https://console.groq.com) → sign up → API Keys → Create key. Free tier gives 100k tokens/day, no credit card.

**4. Create `config.py`** (never committed — in `.gitignore`)
```python
from pathlib import Path

BASE_DIR = Path(__file__).parent
GROQ_API_KEY = "your_key_here"
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
MAX_PAPERS = 20
CODE_EXECUTION_TIMEOUT = 120
```

**5. Run**
```bash
python main.py
```

---

## Changing the topic

Open `main.py` and change these four lines only:

```python
TOPIC  = "your research topic"
DOMAIN = "your domain"
VENUE  = "IEEE Transactions on / CVPR / NeurIPS 2026"
YEAR_FROM = 2021
```

Everything else — the search, the experiment, the paper — adapts automatically.

---

## Outputs
outputs/
├── final/
│   └── paper_draft.md          ← complete paper
├── code/
│   ├── experiment.py           ← generated experiment script
│   └── results.json            ← actual metrics
└── plots/
└── feature_importance.png  ← XAI visualization

The paper includes abstract, introduction, related work, methodology, experiments, results, discussion, conclusion, limitations, and references — all grounded in real fetched papers and real experiment numbers.

---

## Guardrails

This system is built to not hallucinate.

- Citations are verified against the fetched paper list before writing
- Numbers in the results section come from `results.json` only
- Experiment code runs in a sandboxed subprocess with a 120s timeout
- Every agent call is logged to `logs/audit.jsonl` with input/output hashes
- `config.py`, `state/`, `outputs/`, and `logs/` are all gitignored

---

## Stack

- **LLM** — Groq API, LLaMA 3.3 70B (free)
- **Search** — arXiv API, Semantic Scholar API (both free, no auth)
- **Experiments** — scikit-learn, numpy, pandas, matplotlib
- **State** — JSON files with snapshot versioning
- **Validation** — Pydantic v2
- **Logging** — jsonlines audit trail

No GPU required. Runs on any laptop.

---

## Roadmap

- [ ] Streamlit UI for approvals instead of CLI
- [ ] LaTeX export and Overleaf integration
- [ ] Batch mode for multiple topics
- [ ] Support for Gemini and Claude backends
- [ ] Automatic venue-specific formatting

---

## Author

Built by **Ansh Goyal** — ECE undergrad at JIIT Noida, working on XAI, computer vision, and applied ML research.

[GitHub](https://github.com/Ansh-Goyal01) · [Email](mailto:anshgoyal5500@gmail.com)
Save, close. Push:
powershellgit add README.md
git commit -m "redesign README - clean and professional"
git push origin main
This version is clean, specific, honest, and reads like a real project — not AI filler. The badges will render in color on GitHub automatically. Check it after pushing boss.You said: i'll paste this in github only like edit the readme in github only so give for thati'll paste this in github only like edit the readme in github only so give for that19:16Claude responded: Go to your repo → click README.Go to your repo → click README.md → click the pencil icon (Edit) → select all → delete → paste this exactly:
markdown# Research Agent

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-F55036?style=flat-square)](https://console.groq.com)
[![arXiv](https://img.shields.io/badge/Search-arXiv_+_Semantic_Scholar-B31B1B?style=flat-square)](https://arxiv.org)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active_Development-22C55E?style=flat-square)]()
[![Free](https://img.shields.io/badge/Cost-100%25_Free_to_Run-A855F7?style=flat-square)]()

> An end-to-end multi-agent system that takes a research topic and autonomously searches literature, finds gaps, proposes hypotheses, runs real experiments, and writes a complete IEEE-style paper draft — with human approval at every critical step.

---

## The idea

Most AI writing tools generate plausible-sounding text. This is different.

Every claim in the output is grounded in papers actually fetched from arXiv and Semantic Scholar. Every number in the results section comes from a Python experiment that actually ran on your machine. Every citation is verified before the paper is written.

Change one line. Get a research paper draft.

```python
TOPIC = "explainable AI for urban traffic congestion prediction"
```

---

## How it works

Ten agents run in sequence, each with a single responsibility:
Scout → Gap Analyst → Idea Generator → ✋ approve
→ Title → Experiment Designer  → ✋ approve
→ Implementation → Results Analyst
→ Paper Writer → Reviewer       → ✋ approve
→ outputs/final/paper_draft.md

Three checkpoints where you decide whether to continue. Everything else is fully automated.

---

## What each agent does

| Agent | Input | Output |
|---|---|---|
| Scout | research topic | 20 verified papers from arXiv + S2 |
| Gap Analyst | paper list | ranked research gaps with scores |
| Idea Generator | gaps | 3 falsifiable hypotheses |
| Title Agent | chosen idea | 3 title variants + recommendation |
| Experiment Designer | idea | full experiment plan, CPU-only |
| Implementation | plan | working Python script + results.json |
| Results Analyst | experiment output | metrics, verdict, findings |
| Paper Writer | all of the above | 9-section IEEE-style draft |
| Reviewer | draft + source data | citation audit, number audit, score |
| Orchestrator | everything | manages flow, retries, approvals |

---

## Setup

**1. Clone**
```bash
git clone https://github.com/Ansh-Goyal01/Research-Agent.git
cd Research-Agent
```

**2. Install**
```bash
pip install -r requirements.txt
```

**3. Get a free Groq key**

Go to [console.groq.com](https://console.groq.com) → sign up → API Keys → Create key. Free tier gives 100k tokens/day, no credit card needed.

**4. Create `config.py`** in the project root — this file is gitignored and never committed:

```python
from pathlib import Path

BASE_DIR = Path(__file__).parent
GROQ_API_KEY = "your_key_here"
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
MAX_PAPERS = 20
CODE_EXECUTION_TIMEOUT = 120
```

**5. Run**
```bash
python main.py
```

---

## Changing the topic

Open `main.py` and change these lines only:

```python
TOPIC  = "your research topic"
DOMAIN = "your domain"
VENUE  = "IEEE Transactions on / CVPR / NeurIPS 2026"
```

Everything else adapts automatically.

---

## Outputs
outputs/
├── final/
│   └── paper_draft.md          ← complete paper
├── code/
│   ├── experiment.py           ← generated experiment script
│   └── results.json            ← actual metrics
└── plots/
└── feature_importance.png  ← XAI visualization

The paper includes abstract, introduction, related work, methodology, experiments, results, discussion, conclusion, limitations, and references — all grounded in real fetched papers and real experiment numbers.

---

## Guardrails

- Citations verified against fetched paper list before writing
- Numbers in results section come from `results.json` only
- Experiment runs in sandboxed subprocess with 120s timeout
- Every agent call logged to `logs/audit.jsonl` with input/output hashes
- `config.py`, `state/`, `outputs/`, `logs/` all gitignored

---

## Stack

| Layer | Tool |
|---|---|
| LLM | Groq API — LLaMA 3.3 70B (free) |
| Paper Search | arXiv API + Semantic Scholar API |
| ML Experiments | scikit-learn, numpy, pandas, matplotlib |
| State Management | JSON with snapshot versioning |
| Schema Validation | Pydantic v2 |
| Audit Logging | jsonlines |

No GPU required. Runs on any laptop.

---

## Roadmap

- [ ] Streamlit UI for approvals
- [ ] LaTeX export and Overleaf integration
- [ ] Batch mode for multiple topics
- [ ] Gemini and Claude backend support
- [ ] Venue-specific formatting (NeurIPS, IEEE, ACL)

---

## Author

Built by **Ansh Goyal** — ECE undergrad at JIIT Noida, working on XAI, computer vision, and applied ML.

[GitHub](https://github.com/Ansh-Goyal01) · [Email](mailto:anshgoyal5500@gmail.com)
