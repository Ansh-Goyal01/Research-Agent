import json
from groq import Groq
from memory import state_store, audit_logger
import config

SYSTEM_PROMPT = """You are a creative but rigorous research idea generator.
Rules:
- Every idea must directly address a gap from the provided gap analysis.
- Hypotheses must be falsifiable and specific.
- Compute cost must reflect what a student laptop can handle: prefer low or medium.
- Return ONLY valid JSON. No markdown, no explanation, no code blocks."""

def run():
    client = Groq(api_key=config.GROQ_API_KEY)
    gap_analysis = state_store.get_state("gap_analysis")
    paper_list = state_store.get_state("paper_list")
    if not gap_analysis:
        raise ValueError("[IdeaGenerator] No gap_analysis in state. Run GapAnalyst first.")

    print("[IdeaGenerator] Generating research ideas...")

    prompt = f"""Based on this gap analysis:
{json.dumps(gap_analysis, indent=2)}

And these papers for context:
{json.dumps([p["title"] for p in paper_list["papers"]], indent=2)}

Generate 3 research ideas. Return a JSON object with this exact structure:
{{
  "ideas": [
    {{
      "idea_id": "idea_1",
      "hypothesis": "one sentence falsifiable hypothesis",
      "novelty_explanation": "what makes this new compared to existing work",
      "minimum_viable_experiment": "simplest experiment to test this",
      "compute_cost": "low",
      "time_estimate": "1-2 weeks",
      "novelty_score": 8.0,
      "feasibility_score": 9.0,
      "impact_score": 8.5
    }}
  ],
  "recommended_idea_id": "idea_1"
}}

Return ONLY the JSON object."""

    response = client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        max_tokens=config.MAX_TOKENS,
        temperature=0.7
    )

    raw_output = response.choices[0].message.content.strip()
    if "```" in raw_output:
        raw_output = raw_output.split("```")[1]
        if raw_output.startswith("json"):
            raw_output = raw_output[4:]

    result = json.loads(raw_output)
    state_store.update_state("idea_candidates", result)
    audit_logger.log("idea_generator", {"top_gap": gap_analysis["top_gap_id"]}, result)
    print(f"[IdeaGenerator] Generated {len(result['ideas'])} ideas.")
    return result