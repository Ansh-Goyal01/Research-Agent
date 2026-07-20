import json
from groq import Groq
from memory import state_store, audit_logger
import config

SYSTEM_PROMPT = """You are a strict academic peer reviewer.
Rules:
- Flag any citation not traceable to the provided paper list as a critical issue.
- Flag any number that does not match the result_summary as a critical issue.
- Be thorough but fair.
- Return ONLY valid JSON. No markdown, no explanation."""

def run():
    client = Groq(api_key=config.GROQ_API_KEY)
    paper_draft = state_store.get_state("paper_draft")
    paper_list = state_store.get_state("paper_list")
    result_summary = state_store.get_state("result_summary")

    if not paper_draft:
        raise ValueError("[Reviewer] No paper_draft in state.")

    paper_titles = [p["title"] for p in paper_list["papers"]]
    actual_metrics = result_summary["metrics"]

    print("[Reviewer] Reviewing paper draft...")

    prompt = f"""Review this research paper draft:

ABSTRACT:
{paper_draft.get('abstract', '')}

INTRODUCTION:
{paper_draft.get('introduction', '')[:800]}

RESULTS:
{paper_draft.get('results', '')}

LIMITATIONS:
{paper_draft.get('limitations', '')}

VERIFICATION DATA:
Allowed citations (only these are valid): {json.dumps(paper_titles)}
Actual metric values (must match results section): {json.dumps(actual_metrics)}

Check for:
1. Citations not in the allowed list
2. Numbers that don't match actual metrics
3. Unsupported claims
4. Missing or weak limitations
5. Logic gaps between sections

Return a JSON object with this exact structure:
{{
  "issues": [
    {{
      "severity": "critical",
      "section": "results",
      "description": "description of issue"
    }}
  ],
  "overall_verdict": "accept",
  "required_changes": ["change1", "change2"],
  "citation_audit_passed": true,
  "number_audit_passed": true,
  "hallucination_flags": []
}}

Severity levels: critical, major, minor
Verdict options: accept, revise, reject
Return ONLY the JSON object."""

    response = client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        max_tokens=config.MAX_TOKENS,
        temperature=0.2
    )

    raw_output = response.choices[0].message.content.strip()
    if "```" in raw_output:
        raw_output = raw_output.split("```")[1]
        if raw_output.startswith("json"):
            raw_output = raw_output[4:]

    result = json.loads(raw_output)
    state_store.update_state("reviewer_feedback", result)
    audit_logger.log("reviewer", {"sections_reviewed": list(paper_draft.keys())}, result)
    print(f"[Reviewer] Verdict: {result['overall_verdict']}")
    print(f"[Reviewer] Issues found: {len(result['issues'])}")
    return result