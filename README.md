# Research Agent

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-F55036?style=flat-square)](https://console.groq.com)
[![arXiv](https://img.shields.io/badge/Search-arXiv_+_Semantic_Scholar-B31B1B?style=flat-square)](https://arxiv.org)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)
[![Free](https://img.shields.io/badge/Cost-100%25_Free-A855F7?style=flat-square)]()
[![Status](https://img.shields.io/badge/Status-Active-22C55E?style=flat-square)]()

> Change one line. Get a complete IEEE-quality research paper.

```python
TOPIC = "explainable AI for urban traffic congestion prediction"
```

An end-to-end multi-agent system that autonomously searches literature, identifies research gaps, designs and runs real experiments, generates publication-quality figures, and writes a complete research paper draft — with human approval at every critical decision point.

---

## What makes this different

Most AI writing tools generate plausible-sounding text. This doesn't.

- Every citation comes from papers **actually fetched** from arXiv and Semantic Scholar
- Every number in the results section comes from an experiment that **actually ran** on your machine
- Every claim is verified before the paper is written
- Statistical significance is tested with real p-values and confidence intervals
- 6 publication-quality figures are generated automatically including a system architecture diagram

---
## Pipeline

| Stage | Agent | What it does |
|-------|-------|-------------|
| 1 | 🔍 **Scout** | Searches arXiv + Semantic Scholar · primary + secondary queries · 20+ papers |
| 2 | 🔬 **Gap Analyst** | Ranks research gaps by impact and feasibility score |
| 3 | 📚 **Related Work Agent** | Deep synthesis · groups papers into themes · finds contradictions |
| 4 | 💡 **Idea Generator** | Proposes 3 falsifiable hypotheses scored by novelty and impact |
| — | ✋ | **HUMAN APPROVAL #1 — Choose research idea** |
| 5 | 📝 **Title Agent** | Generates method acronym + 3 IEEE-style title variants |
| 6 | 🧪 **Experiment Designer** | Smart dataset selection · CPU-only methodology · baselines |
| — | ✋ | **HUMAN APPROVAL #2 — Review experiment plan** |
| 7 | 🏗️ **Architecture Diagram** | Auto-generates topic-aware system diagram |
| 8 | ⚙️ **Implementation Agent** | Runs RF · GB · XGBoost · LightGBM · 5-fold CV · p-values · 95% CI · 5 figures |
| 9 | 📊 **Results Analyst** | Interprets real experiment output · hypothesis verdict |
| 10 | ✍️ **Paper Writer** | IEEE style guide · 9 sections · strict IMRaD · abstract rewriter |
| — | 🔎 | **Consistency Checker — numbers · citations · section lengths** |
| 11 | 👨‍⚖️ **Reviewer Agent** | Citation audit · number audit · domain relevance · quality score |
| — | ✋ | **HUMAN APPROVAL #3 — Final paper review** |
| — | 📄 | **Export → paper_draft.md + paper.html (figures embedded)** |

## Output

Every run produces these files:

**Final paper:**
| File | Description |
|------|-------------|
| `outputs/final/paper_draft.md` | Complete paper in markdown |
| `outputs/final/paper.html` | Styled paper with all figures embedded — open in browser, print to PDF |

**Figures:**
| File | Figure |
|------|--------|
| `outputs/plots/fig0_architecture.png` | System architecture diagram |
| `outputs/plots/fig1_feature_importance.png` | XAI feature importance |
| `outputs/plots/fig2_model_comparison.png` | Model accuracy comparison |
| `outputs/plots/fig3_cv_distribution.png` | Cross-validation distribution |
| `outputs/plots/fig4_ablation.png` | Ablation study |
| `outputs/plots/fig5_confusion_matrix.png` | Confusion matrix |

**Experiment artifacts:**
| File | Description |
|------|-------------|
| `outputs/code/experiment.py` | Generated and executed experiment script |
| `outputs/code/results.json` | Raw metrics · p-values · confidence intervals |
| `outputs/sections/` | Individual section drafts |

---

## Paper structure

| Section | Details |
|---------|---------|
| **Abstract** | 150-200 words · strict IMRaD · exact metric values |
| **Introduction** | Problem motivation · gap evidence from real papers · numbered contributions |
| **Related Work** | 3 themed subsections · synthesis · contradictions · gaps |
| **Methodology** | System overview · numbered equations · hyperparameters · XAI integration |
| **Experiments** | Dataset statistics · baselines · 5-fold CV · ablation design |
| **Results** | Comparison table · p-values · 95% CI · feature importance top-5 |
| **Discussion** | Hypothesis verdict · direct comparison · practical implications |
| **Conclusion** | Contributions with exact numbers · 3 specific future directions |
| **Limitations** | 3 specific honest limitations with future remedies |
| **References** | 15-20 verified citations from fetched papers only |

## Figures generated

| Figure | Content |
|--------|---------|
| Fig. 1 | System architecture (topic-aware, auto-generated) |
| Fig. 2 | XAI feature importance bar chart |
| Fig. 3 | Model comparison with error bars (RF, GB, XGBoost, LightGBM) |
| Fig. 4 | Cross-validation score distribution (box plot) |
| Fig. 5 | Ablation study — full features vs XAI top-5 |
| Fig. 6 | Confusion matrix (best model) |

Architecture diagram auto-selects style based on topic:
- Traffic / transport → traffic management pipeline
- Medical / clinical → clinical decision pipeline
- Industrial / IoT → predictive maintenance pipeline
- Uncertainty / conformal → conformal prediction pipeline
- General → standard ML pipeline

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
pip install xgboost lightgbm
```

**3. Get a free Groq API key**

[console.groq.com](https://console.groq.com) → Sign up → API Keys → Create key. Free tier: 100k tokens/day. No credit card needed.

**4. Create `config.py`**

This file is gitignored and never committed.

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
MAX_PAPERS = 20
CODE_EXECUTION_TIMEOUT = 180
```

**5. Run**
```bash
python main.py
```

---

## Changing the topic

Edit only these lines in `main.py`:

```python
TOPIC  = "your research topic here"
DOMAIN = "your research domain"
VENUE  = "IEEE Transactions on / CVPR / NeurIPS 2026"
```

Everything else adapts automatically — dataset selection, architecture diagram style, experiment design, paper writing.

**Topics that work well:**

```python
# XAI + Systems
TOPIC = "explainable AI for urban traffic congestion prediction and emergency vehicle prioritization"

# Medical
TOPIC = "conformal prediction for uncertainty quantification in medical image classification"

# Industrial
TOPIC = "predictive maintenance using machine learning and explainable AI for industrial equipment"

# Computer Vision
TOPIC = "real-time structural defect detection using deep learning on edge devices"

# NLP
TOPIC = "hallucination detection and mitigation in large language models"
```

---

## Guardrails

- Citations verified against fetched paper list — hallucinated references blocked
- Metric values cross-checked between experiment output and paper sections
- Consistency checker runs before review — flags number mismatches and short sections
- Experiment runs in sandboxed subprocess with 180s timeout
- Every agent call logged with input/output hashes in `logs/audit.jsonl`
- `config.py`, `state/`, `outputs/`, `logs/` are all gitignored

---

## Token usage

Live token tracker shows usage after every agent:
[Tokens] scout: +1,823 | Today: 1,823/100,000 [█░░░░░░░░░░░░░░░░░░░] 98,177 left
[Tokens] gap_analyst: +1,194 | Today: 3,017/100,000 [░░░░░░░░░░░░░░░░░░░░] 96,983 left

Optimized to use ~20,000-25,000 tokens per full run — 4 complete papers per day on the free tier.

---

## Stack

| Layer | Tool |
|-------|------|
| LLM | Groq API — LLaMA 3.3 70B (free) |
| Paper Search | arXiv API + Semantic Scholar API |
| ML Experiments | scikit-learn · XGBoost · LightGBM · scipy |
| Visualization | matplotlib (6 figures per run) |
| State Management | JSON with snapshot versioning |
| Schema Validation | Pydantic v2 |
| Audit Logging | jsonlines |

No GPU required. 

---

## Roadmap

- [ ] Rate limit auto-retry (no manual intervention)
- [ ] Resume from checkpoint (`--resume` flag)
- [ ] LaTeX export for direct journal submission
- [ ] PDF export via weasyprint
- [ ] Streamlit UI for approvals
- [ ] Batch mode for multiple topics
- [ ] Gemini and Claude backend support

---

## Author

**Ansh Goyal**
B.Tech ECE · Jaypee Institute of Information Technology, Noida
Working on XAI, computer vision, and applied ML research.

[GitHub](https://github.com/Ansh-Goyal01) · [Email](mailto:anshgoyal5500@gmail.com)

---

*Built from scratch · Powered by Groq · 100% free to run · No GPU needed*
