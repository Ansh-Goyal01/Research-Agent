import json
from groq import Groq
from memory import state_store, audit_logger
import config

SYSTEM_PROMPT = """You are a critical research analyst. You identify genuine research gaps from a list of real papers.
Rules:
- Every gap must be traceable to at least one paper in the provided list.
- Do NOT invent gaps that have no evidence in the papers.
- Score impact (1-10) and feasibility (1-10) honestly.
- Return ONLY valid JSON. No markdown, no explanation, no code blocks."""

def run():
    client = Groq(api_key=config.GROQ_API_KEY)
    paper_list = state_store.get_state("paper_list")
    if not paper_list:
        raise ValueError("[GapAnalyst] No paper_list found in state. Run Scout first.")

    papers = paper_list["papers"]
    print(f"[GapAnalyst] Analyzing {len(papers)} papers for gaps...")

    prompt = f"""Analyze these {len(papers)} papers and identify research gaps:

{json.dumps(papers, indent=2)}

Return a JSON object with this exact structure:
{{
  "gaps": [
    {{
      "gap_id": "gap_1",
      "description": "clear description of the gap",
      "supporting_paper_titles": ["title1", "title2"],
      "impact_score": 8.5,
      "feasibility_score": 7.0
    }}
  ],
  "top_gap_id": "gap_1",
  "rationale": "why this is the most important gap"
}}

Identify 3-5 gaps. Return ONLY the JSON object."""

    response = client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        max_tokens=config.MAX_TOKENS,
        temperature=0.3
    )

    raw_output = response.choices[0].message.content.strip()
    if "```" in raw_output:
        raw_output = raw_output.split("```")[1]
        if raw_output.startswith("json"):
            raw_output = raw_output[4:]

    result = json.loads(raw_output)
    state_store.update_state("gap_analysis", result)
    audit_logger.log("gap_analyst", {"papers_count": len(papers)}, result)
    print(f"[GapAnalyst] Found {len(result['gaps'])} gaps. Top gap: {result['top_gap_id']}")
    return result