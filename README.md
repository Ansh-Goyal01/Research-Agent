\# 🔬 Research Agent — Autonomous AI Research Pipeline



<div align="center">



!\[Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge\&logo=python)

!\[Groq](https://img.shields.io/badge/Groq-LLaMA\_3.3\_70B-orange?style=for-the-badge)

!\[License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

!\[Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)



\*\*A fully automated multi-agent system that discovers research gaps, generates ideas, runs real experiments, and drafts complete IEEE-quality research papers — from a single topic input.\*\*



\[Features](#-features) • \[Architecture](#-architecture) • \[Setup](#-setup) • \[Usage](#-usage) • \[Output](#-output) • \[Stack](#-tech-stack)



</div>



\---



\## 🎯 What It Does



Change one line in `main.py` → get a full research paper draft.



```python

TOPIC = "explainable AI for urban traffic congestion prediction"

```



The system does everything else automatically:

\- Searches 20+ real papers from arXiv and Semantic Scholar

\- Identifies research gaps with impact and feasibility scores

\- Proposes and ranks novel research ideas

\- Designs a reproducible experiment (CPU-friendly)

\- Runs the experiment and collects real metrics

\- Writes a complete 9-section IEEE-style paper

\- Reviews the paper for citation accuracy and hallucinations

\- Saves the final draft to `outputs/final/paper\_draft.md`



\---



\## ✨ Features



\- \*\*10 specialized AI agents\*\* each with a distinct role and strict output schema

\- \*\*Real paper search\*\* via arXiv API and Semantic Scholar API

\- \*\*Bulletproof experiment execution\*\* with synthetic data fallback

\- \*\*XAI integration\*\* — feature importance analysis built into every experiment

\- \*\*3 human approval checkpoints\*\* — you stay in control of key decisions

\- \*\*Anti-hallucination guardrails\*\* — citations verified against fetched papers

\- \*\*Full audit trail\*\* — every agent call logged with input/output hashes

\- \*\*State snapshots\*\* — resume from any stage if something fails

\- \*\*Zero GPU required\*\* — runs entirely on CPU, optimized for laptops



\---



\## 🏗 Architecture

┌─────────────────────────────────────────────────────────┐

│                    ORCHESTRATOR                          │

│         (manages flow, retries, approvals)              │

└──────────────────┬──────────────────────────────────────┘

│

┌──────────────▼──────────────┐

│                             │

▼                             ▼

┌─────────┐   ┌────────────┐   ┌──────────────────┐

│  Scout  │──▶│ Gap Analyst│──▶│ Idea Generator   │

│(search) │   │  (gaps)    │   │ (hypotheses)     │

└─────────┘   └────────────┘   └────────┬─────────┘

│

✋ HUMAN APPROVAL #1

│

┌───────────────────▼──────────────┐

│           Title Agent             │

└───────────────────┬──────────────┘

│

┌───────────────────▼──────────────┐

│      Experiment Designer          │

└───────────────────┬──────────────┘

│

✋ HUMAN APPROVAL #2

│

┌───────────────────▼──────────────┐

│      Implementation Agent         │

│    (writes + runs experiment)     │

└───────────────────┬──────────────┘

│

┌───────────────────▼──────────────┐

│       Results Analyst             │

└───────────────────┬──────────────┘

│

┌───────────────────▼──────────────┐

│         Paper Writer              │

│   (9 sections, IEEE-style)        │

└───────────────────┬──────────────┘

│

┌───────────────────▼──────────────┐

│           Reviewer                │

│  (citation + number audit)        │

└───────────────────┬──────────────┘

│

✋ HUMAN APPROVAL #3

│

┌───────────────────▼──────────────┐

│         Final Export              │

│    outputs/final/paper\_draft.md   │

└──────────────────────────────────┘



\---



\## 📁 Project Structure

research\_agent/

├── main.py                    # ← Only file you need to edit

├── orchestrator.py            # Pipeline manager

├── config.py                  # API keys and paths (not committed)

├── schemas.py                 # Pydantic data models

│

├── agents/

│   ├── scout.py               # Searches arXiv + Semantic Scholar

│   ├── gap\_analyst.py         # Identifies research gaps

│   ├── idea\_generator.py      # Proposes research ideas

│   ├── title\_agent.py         # Generates paper titles

│   ├── experiment\_designer.py # Designs reproducible experiments

│   ├── implementation\_agent.py# Writes and runs experiment code

│   ├── results\_analyst.py     # Interprets experimental outputs

│   ├── paper\_writer.py        # Drafts all 9 paper sections

│   └── reviewer.py            # Audits citations and claims

│

├── tools/

│   ├── paper\_search.py        # arXiv + Semantic Scholar wrappers

│   ├── code\_executor.py       # Safe subprocess runner

│   └── file\_manager.py        # Path-whitelisted file operations

│

├── memory/

│   ├── state\_store.py         # JSON state management

│   └── audit\_logger.py        # Append-only audit log

│

├── outputs/

│   ├── code/                  # Generated experiment scripts

│   ├── plots/                 # Feature importance charts

│   ├── sections/              # Individual section drafts

│   └── final/                 # Complete paper draft

│

├── state/                     # Pipeline snapshots

└── logs/                      # Audit trail



\---



\## ⚙️ Setup



\### 1. Clone the repository

```bash

git clone https://github.com/Ansh-Goyal01/Research-Agent.git

cd Research-Agent

```



\### 2. Install dependencies

```bash

pip install -r requirements.txt

```



\### 3. Get a free Groq API key

1\. Go to \[console.groq.com](https://console.groq.com)

2\. Sign up free → API Keys → Create key

3\. Copy your `gsk\_...` key



\### 4. Create your config file

Create `config.py` in the project root:

```python

from pathlib import Path



BASE\_DIR = Path(\_\_file\_\_).parent

GROQ\_API\_KEY = "your\_groq\_key\_here"

MODEL = "llama-3.3-70b-versatile"

MAX\_TOKENS = 4096

MAX\_RETRIES = 3

STATE\_FILE       = BASE\_DIR / "state" / "workflow\_state.json"

AUDIT\_LOG        = BASE\_DIR / "logs" / "audit.jsonl"

AGENT\_LOGS\_DIR   = BASE\_DIR / "logs" / "agent\_calls"

OUTPUTS\_CODE     = BASE\_DIR / "outputs" / "code"

OUTPUTS\_PLOTS    = BASE\_DIR / "outputs" / "plots"

OUTPUTS\_SECTIONS = BASE\_DIR / "outputs" / "sections"

OUTPUTS\_FINAL    = BASE\_DIR / "outputs" / "final"

OUTPUTS\_ARCHIVE  = BASE\_DIR / "outputs" / "final" / "archive"

ARXIV\_BASE\_URL            = "https://export.arxiv.org/api/query"

SEMANTIC\_SCHOLAR\_BASE\_URL = "https://api.semanticscholar.org/graph/v1"

MAX\_PAPERS = 20

CODE\_EXECUTION\_TIMEOUT = 120

```



> ⚠️ `config.py` is in `.gitignore` and will never be committed.



\---



\## 🚀 Usage



\### Run the pipeline

```bash

python main.py

```



\### Change the research topic

Edit only these lines in `main.py`:

```python

TOPIC  = "your research topic here"

DOMAIN = "your research domain"

VENUE  = "Target Conference or Journal 2026"

```



\### Example topics that work well

```python

\# Computer Vision

TOPIC = "few-shot learning for medical image segmentation"



\# NLP

TOPIC = "hallucination detection in large language models"



\# IoT / Systems

TOPIC = "federated learning for edge device anomaly detection"



\# Traffic / Smart Cities

TOPIC = "explainable AI for urban traffic congestion prediction"

```



\### Interactive approval checkpoints

The pipeline pauses 3 times for your input:

1\. \*\*Idea selection\*\* — choose which research idea to pursue

2\. \*\*Experiment approval\*\* — review methodology before code runs

3\. \*\*Final approval\*\* — approve the paper before export



\---



\## 📄 Output



After a successful run you get:



| File | Description |

|------|-------------|

| `outputs/final/paper\_draft.md` | Complete paper draft |

| `outputs/code/experiment.py` | Experiment script |

| `outputs/code/results.json` | Raw metric results |

| `outputs/plots/feature\_importance.png` | XAI feature chart |

| `state/workflow\_state.json` | Full pipeline state |

| `logs/audit.jsonl` | Complete audit trail |



\### Sample paper output structure

Paper Title

Abstract        (\~200 words)

Introduction    (\~400 words)

Related Work    (\~400 words, 8+ citations)

Methodology     (\~500 words, equations)

Experiments     (\~400 words)

Results         (\~400 words, real metrics)

Discussion      (\~400 words)

Conclusion      (\~250 words)

Limitations     (\~200 words)

References      (15-20 verified citations)



\---



\## 🛠 Tech Stack



| Component | Technology |

|-----------|-----------|

| LLM | Groq API — LLaMA 3.3 70B (free tier) |

| Paper Search | arXiv API + Semantic Scholar API |

| ML Experiments | scikit-learn, numpy, pandas |

| Visualization | matplotlib |

| State Management | JSON + TinyDB |

| Audit Logging | jsonlines |

| Schema Validation | Pydantic v2 |

| Version Control | gitpython |



\---



\## 🔒 Safety Guardrails



\- \*\*No fabricated citations\*\* — every reference verified against fetched papers

\- \*\*No invented numbers\*\* — results section uses only actual experiment outputs

\- \*\*No unsafe code\*\* — experiment sandbox has 120s timeout and path whitelist

\- \*\*No accidental commits\*\* — `config.py`, `state/`, `outputs/`, `logs/` all gitignored

\- \*\*Full audit trail\*\* — every agent call logged with timestamp and hash



\---



\## 🗺 Roadmap



\- \[ ] Streamlit web UI for approvals

\- \[ ] LaTeX / PDF export

\- \[ ] Multi-topic batch processing

\- \[ ] Email notification at checkpoints

\- \[ ] Support for additional LLM providers

\- \[ ] Automatic related work expansion

\- \[ ] Integration with Overleaf API



\---



\## 👤 Author



\*\*Ansh Goyal\*\*

B.Tech ECE — Jaypee Institute of Information Technology, Noida

\[GitHub](https://github.com/Ansh-Goyal01) • \[Email](mailto:anshgoyal5500@gmail.com)



\---



\## 📜 License



MIT License — free to use, modify, and distribute.



\---



<div align="center">

Built from scratch in one session · Powered by Groq · 100% free to run

</div>

