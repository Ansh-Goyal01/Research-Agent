# memory/style_guide.py  — Research Agent
# IEEE style rules injected into paper_writer system prompt.
# Fixes: removed BNN/MC-dropout references, added modality-aware method description,
# fixed duplicate abstract bug, fixed intro truncation, fixed #### markdown leaking.

IEEE_STYLE_GUIDE = """
WRITING STYLE RULES FOR IEEE JOURNAL PAPERS:

GENERAL:
- Write in third person ("The proposed method", "This paper presents", NOT "We think")
- Every paragraph must have a topic sentence, supporting evidence, and closing sentence
- Never start a sentence with "This shows" or "This means" — be specific
- Avoid phrases: "In this paper we", "It is worth noting", "It should be noted"
- Use active voice: "The model achieves" not "X% was achieved"
- Every claim must be supported by either a citation or experimental evidence
- Do NOT use Markdown heading markers (##, ###, ####) anywhere in your output
- Use plain subsection labels like "A) System Overview" or "A. System Overview"

ABSTRACT (strict IMRaD — write ONCE, do NOT repeat):
- Sentence 1: Background — one sentence on the broader problem
- Sentence 2: Problem — specific gap this paper addresses
- Sentences 3-4: Method — what you propose and how (DO NOT mention Bayesian Neural Networks
  or Monte Carlo dropout unless the implementation actually uses them)
- Sentences 5-6: Results — exact numbers, metrics, dataset name
- Sentence 7: Conclusion — what this means for the field
- Total: 150-200 words maximum
- Write the abstract EXACTLY ONCE — do not repeat it as a standalone section heading

INTRODUCTION:
- Paragraph 1: MUST begin with a complete sentence introducing the broad domain.
  Example: "Large language models have been widely deployed across critical domains..."
  NEVER start with a comma or mid-sentence fragment.
- Paragraph 2: What existing work does and its specific limitations (cite 3+ papers)
- Paragraph 3: What this paper proposes — one clear paragraph
- Paragraph 4: Numbered list of contributions (exactly 3-4 items, formatted as (1)...(2)...(3)...)
- Paragraph 5: Paper organization ("The rest of this paper is organized as follows...")

RELATED WORK:
- Organize into 3 themed subsections using plain labels: "A. [Theme Name]"
- Do NOT use #### or ### markers
- For each cited paper: what they did, what dataset, what metric, what limitation
- End each subsection with: "Unlike these approaches, our method..."
- Never just list papers — synthesize and compare them
- Minimum 8 citations spread across subsections

METHODOLOGY:
- Start with a system overview paragraph describing the full pipeline
- Describe the actual models used — if the pipeline uses TF-IDF + ensemble classifiers,
  say so clearly. Do NOT claim "Bayesian Neural Network" or "Monte Carlo dropout"
  unless those are actually in the experiment code.
- Every equation must be numbered: (1), (2), (3)
- Define every variable immediately after the equation
- Subsections: A) System Overview, B) Data Preprocessing, C) Feature Engineering,
  D) Model Development, E) Mathematical Formulation

RESULTS TABLE FORMAT (always use this exact markdown):
| Model | Accuracy (%) | F1-Score | Precision | Recall | Std |
|-------|-------------|----------|-----------|--------|-----|
| Proposed | **X.XX** | **X.XX** | X.XX | X.XX | ±X.XX |
| Baseline 1 | X.XX | X.XX | X.XX | X.XX | ±X.XX |
- Bold the best result in each column
- Always include standard deviation
- Table caption ABOVE the table: "TABLE I: ..."  (Roman numerals, ALL CAPS TABLE)

DISCUSSION:
- Paragraph 1: Restate hypothesis and whether results support it
- Paragraph 2: Compare your numbers directly with baselines ("outperforms X by Y%")
- Paragraph 3: Why the method works — mechanistic explanation
- Paragraph 4: Practical implications for real deployment
- Paragraph 5: Limitations and when the method might fail

CONCLUSION:
- Never introduce new information
- Restate contributions with exact numbers
- 3 specific future work directions with technical details
- Final sentence: broader impact statement

LIMITATIONS:
- At least 3 limitations
- Each limitation: what it is, why it exists, how future work could address it
- Do not be vague — be specific

CITATION FORMAT:
- Always cite as [N] inline
- When comparing: "Method X [3] achieves Y% on dataset Z, while our approach achieves Y+delta%"
"""

ABSTRACT_TEMPLATE = """
Write the abstract following this EXACT structure (write it ONCE, no heading):
[Background: 1 sentence on the broader problem and its real-world importance]
[Problem: 1 sentence on the specific gap or limitation this paper addresses]
[Method: 2 sentences describing what you propose and how it works technically.
 Describe only models that were actually run in the experiment.]
[Results: 2 sentences with exact metric values, dataset name, comparison to baselines]
[Conclusion: 1 sentence on what this means for the field]
Total must be 150-200 words.
"""

RELATED_WORK_TEMPLATE = """
Write related work with exactly 3 subsections using plain labels (no ## or ### markers):
A. [First theme related to the topic]
- Cite 3-4 papers
- For each: method name, dataset used, performance achieved, key limitation
- End with: "Unlike these methods, our approach..."

B. [Second theme related to the topic]
- Cite 3-4 papers
- For each: method name, dataset used, performance achieved, key limitation
- End with: "In contrast to the above..."

C. [Third theme related to the topic]
- Cite 2-3 papers
- For each: method name, dataset used, performance achieved, key limitation
- End with: "Our work addresses this gap by..."
"""

RESULTS_TABLE_TEMPLATE = """
TABLE I: Comparison of classification performance (5-fold CV). Bold = best per metric.

Present results as a proper markdown comparison table:
| Model | Accuracy (%) | F1-Score | Precision | Recall | Std |
|-------|-------------|----------|-----------|--------|-----|
| **Proposed** | **XX.XX** | **X.XX** | X.XX | X.XX | ±X.XX |
| Baseline 1 | XX.XX | X.XX | X.XX | X.XX | ±X.XX |
| Baseline 2 | XX.XX | X.XX | X.XX | X.XX | ±X.XX |

Bold the best result per column.
After the table write a paragraph analyzing the table.
"""
