# Research Agent 🔬

A multi-agent system that reads literature, proposes a hypothesis, designs an
experiment, runs it, and drafts a paper.

It is also a case study in a failure mode that multi-agent systems are
structurally prone to, and in fixing it. **The interesting part of this
repository is not that the pipeline writes papers. It is that an earlier
version of it wrote a convincing paper about an experiment that never
happened — and that the current version cannot.**

---

## The failure

The original 9-stage pipeline ran end to end and produced a clean paper. Here
is what actually happened inside it.

**The plan** (`baseline-golden/workflow_state.json`) — the experiment designer
proposed a study of task-prior conditional variational auto-encoders for
few-shot image classification on highly imbalanced classes:

> "The task-prior conditional variational auto-encoder improves the performance
> of few-shot image classification on highly imbalanced classes by at least 10%"

**The code that ran** (`baseline-golden/outputs/code/experiment.py`) — the
implementation agent was simultaneously told "sklearn only, CPU only, under 60
seconds". Those constraints cannot both be satisfied, so it quietly wrote:

```python
from sklearn.datasets import load_iris
...
from sklearn.linear_model import LogisticRegression
```

No VAE. No few-shot setting. No class imbalance. A 150-row flower dataset from
1936.

**The result** (`baseline-golden/outputs/code/results.json`):

```json
{"metrics": [{"metric_name": "accuracy", "mean": 0.9733333333333334, ...}],
 "hypothesis_verdict": "supported"}
```

**The paper** (`baseline-golden/outputs/sections/abstract.md`) reported that
number as confirmation of the VAE hypothesis:

> "The results show a mean accuracy of 0.9733333333333334 and a mean F1 score
> of 0.9630983743924922, indicating the efficacy of our approach. In
> conclusion, our findings support the hypothesis, demonstrating the potential
> of TPCVAEs in few-shot image classification on imbalanced classes"

All four downstream review stages approved it. The complete forensic trace is
preserved in `baseline-golden/` and is the fixture for the regression tests.

### Why it happened

The root cause was **trust topology**, not agent quality. A better prompt or a
stronger model would not have fixed this.

Every audit in the pipeline was an LLM asked to audit another LLM's output.
`reviewer.py` received its metrics from `result_summary` — itself
LLM-generated — labelled "VERIFICATION DATA". It never saw `experiment.py` and
never saw `results.json`. It emitted `citation_audit_passed` and
`number_audit_passed` as booleans describing *its own review*. And the
orchestrator printed the reviewer's "reject" and then asked the human for
approval anyway.

Nothing in the system could fail a run except an unhandled Python exception.
The verification was decorative.

---

## The fix

Four gates, each checking a claim against evidence the LLM did not produce, and
one harness that removes the opportunity to fabricate in the first place.

The design principle throughout: **prefer removing the ability to lie over
detecting the lie.** A check that scans generated source for `load_iris`
catches exactly that one substitution — `fetch_openml`, a local CSV,
`make_classification`, or a dataset name held in a variable all evade it, and a
passing scan manufactures confidence in dimensions nobody checked.

### The harness owns the data

`harness/registry.py` defines the seven datasets the pipeline is permitted to
study (iris, wine, breast_cancer, digits, and three synthetic imbalanced sets),
all of which genuinely run on a laptop CPU in seconds. The generated script
does not load data; it calls `registry.provision(...)`, which loads the data
**and** writes the true fingerprint (shape, class histogram, sha256, seed) to
`outputs/code/manifest.json`.

The manifest is written by trusted harness code, never by the LLM. A script
cannot forge a record that matches the plan while training on something else.
Conformance holds by construction; fingerprinting exists only to catch a script
that bypasses the harness entirely.

Because every registry entry is cheap by construction, "is this plan feasible?"
reduces to a dict lookup rather than a judgment call.

### The harness owns the scoring

`harness/evaluation.py` goes one step further. The generated script no longer
computes metrics at all — it *declares* two sklearn estimators and calls
`evaluate_comparison(...)`. The harness builds the `StratifiedKFold` split,
clones each estimator per fold, fits on train, scores on test, and writes
`results.json` itself.

Two consequences. There is no LLM-authored `hypothesis_verdict` in the results,
so there is nothing to overclaim. And **data leakage becomes structurally
impossible** — fitting happens inside each fold, so even a
`Pipeline([StandardScaler(), ...])` never sees the test fold. (A real run did
submit exactly that pipeline; the scaler was correctly refit per fold.)

### The four gates

| # | Gate | Module | Asks | Wired at |
|---|------|--------|------|----------|
| 1 | Feasibility | `verification/conformance.py` | Is the planned dataset one this machine can actually run? | `orchestrator.py:86` |
| 2 | Conformance | `verification/conformance.py` | Did the data that *ran* match the data that was *planned*? | `orchestrator.py:129` |
| 3 | Significance | `verification/statistics.py` | Do the numbers support the verdict being claimed? | `orchestrator.py:153` |
| 4 | Reporting | `verification/reporting.py` | Does every figure in the paper trace to a number the experiment produced? | `orchestrator.py:192` |

Gate 1 fires before the human approval prompt. Gate 2 reads the manifest after
execution. Gate 3 replaces the LLM's verdict string with a deterministic one.
Gate 4 is the last line of defence: it scans the finished paper for
metric-shaped figures and flags any that match no computed number within
`_TOL=0.005`. That is the gate that catches the 0.9733 abstract.

### The gate policy is deliberately unforgiving

`verification/gate.py` is the only place a finding becomes a halt.

- A `CRITICAL` finding halts **unconditionally**.
- `override_note` is recorded but **never honoured**. An override a human
  grants by typing a sentence is the original bug with paperwork attached.
- A **vacuous pass halts too** — `CheckResult.vacuous` distinguishes "passed
  after examining nothing" from a real pass. A check that examined zero items
  is not evidence of correctness.

The escape hatch for a check that is genuinely wrong is editing
`harness/registry.py` — a reviewable diff in version control, not a runtime
decision made under pressure.

### The verdict is computed, not chosen

`verification/statistics.py` implements a **pre-registered** decision rule —
committed before any results exist, with named constants:

```
ALPHA = 0.05      # paired t-test significance level
MARGIN = 0.01     # smallest effect size worth caring about, on a [0,1] metric
MIN_PAIRED = 3    # below this, no verdict is possible
```

Three-way outcome: **supported** (significant *and* ≥ MARGIN better),
**refuted** (significant *and* ≥ MARGIN worse), **inconclusive** (everything
else). Reported with Cohen's *dz* and a 95% CI.

Inconclusive is meant to be a common outcome. A pipeline that never says "I
don't know" is still lying — just less obviously. Strict `p < 0.05` alone was
rejected during design because on small-n sklearn runs it reports "refuted"
constantly and reads as broken.

The gate halts on **overclaim only**: asserting a definite verdict the rule
does not reach. Claiming "inconclusive" never halts; under-claiming emits a
minor nudge.

---

## What the fixed pipeline produces

The same class of experiment, run honestly. Hypothesis: does
`class_weight="balanced"` improve macro-F1 over unweighted logistic regression
on an imbalanced binary dataset (2000 samples, 1789/211 split)?

Paired 5-fold CV, scored by the harness:

| Condition | macro-F1 (mean ± std) |
|---|---|
| Balanced LR (method) | 0.691 ± 0.016 |
| Unweighted LR (baseline) | 0.713 ± 0.053 |

```
mean_diff  = -0.0224      p = 0.281      dz = -0.557
CI95       = [-0.0725, +0.0276]          n_paired = 5
```

**Computed verdict: inconclusive.** The CI straddles zero; the folk wisdom that
balancing helps was not established here, and the point estimate is slightly
*negative*.

This is the honest analogue of the run that once reported 0.9733 "supported".
It is a less impressive result and a far more trustworthy one — which is the
entire point.

---

## Verification

```bash
python -m pytest          # 62 tests
```

The suite is adversarial rather than confirmatory. Representative cases:

- The archived fabricating plan is **rejected** by the feasibility gate
  (`test_golden_baseline_plan_is_rejected`).
- A harness bypass leaves no manifest → conformance `CRITICAL` → halt.
- A genuine null result claimed as "supported" → significance `CRITICAL` — the
  iris fabrication in verdict form.
- A fabricated 0.95 in a paper whose experiment produced 0.73 → reporting
  `CRITICAL`.
- An `override_note` on a `CRITICAL` finding still halts.
- A `Pipeline` with a scaler produces no leakage.

Several tests are **mutation-guarded**: neutering `is_feasible()` makes the
golden-baseline test fail, which confirms the suite is load-bearing rather than
vacuously green. Both the happy path and the bypass path are additionally
verified end to end through real subprocess runs.

`pytest.ini` disables three unrelated pytest plugins that a shared global
site-packages autoloads — one of them aborts collection before any test runs.
No environment variable is needed.

---

## Setup

1. `pip install -r requirements.txt`
2. Create `config.py` from `config_template.py` and add your Groq API key.
3. Edit the topic in `main.py`, then `python main.py`.

The pipeline pauses for human approval at two stages; gate 1 runs *before* the
first prompt, so an infeasible plan is rejected before a human is asked to
bless it.

**Outputs:** `outputs/final/paper_draft.md`, `outputs/code/experiment.py`,
`outputs/code/results.json`, `outputs/code/manifest.json`, `outputs/plots/`.

**Stack:** Groq (Llama 3.3 70B) · arXiv + Semantic Scholar · sklearn/scipy ·
subprocess sandbox · JSON state store.

---

## Honest limitations

Stated plainly, because a project about verification that overstates its own
guarantees would be self-refuting.

- **Reproducibility is conditional.** Given the declared estimators and
  `GLOBAL_SEED = 42`, the harness produces byte-identical scores. But the
  *estimator declaration* is still LLM-authored and varies between runs — one
  run submitted bare `LogisticRegression`, the next wrapped it in a
  `StandardScaler` pipeline, and the baseline numbers moved accordingly. The
  scoring is deterministic; the experiment is not yet.
- **The reviewer stage is still an LLM reviewing an LLM.** Its
  `citation_audit_passed` / `number_audit_passed` booleans are self-reports.
  They were not removed, but the orchestrator no longer presents them as
  guarantees — they are labelled "(LLM, advisory)", while number conformance is
  reported from gate 4, which can actually halt a run.
- **Single seed, single split.** No multi-seed replication and no
  non-parametric fallback for the t-test. Deliberately deferred as YAGNI until
  there is a reason.
- **The research domain is narrow on purpose** — tabular ML, class imbalance,
  calibration, feature selection. Everything sklearn-and-CPU can genuinely
  answer in seconds. That narrowing is the fix, not a compromise of it: when
  the plan and the compute budget are the same constraint, there is nothing
  left to silently substitute.
- **Cross-run memory is deliberately absent.** Shared state already exists and
  worked perfectly — it faithfully transmitted the fabricated iris result to
  every downstream stage. Persistence before trustworthiness just builds a
  database of confidently-recorded fiction.
- **A fully unattended end-to-end `orchestrator.run()` has not been
  exercised**; it blocks on `input()` at the two approval prompts. Gate logic
  is validated through scoped live drivers and the test suite.
