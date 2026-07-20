import json
from groq import Groq
from memory import state_store, audit_logger
import config

SYSTEM_PROMPT = """You are an expert academic title writer.
Rules:
- Titles must be specific to the actual work, not generic.
- Do not use words like novel, first, state-of-the-art unless justified.
- Each title must be 8-18 words.
- Return ONLY valid JSON. No markdown, no explanation."""

def run(chosen_idea_id=None):
    client = Groq(api_key=config.GROQ_API_KEY)
    ideas = state_store.get_state("idea_candidates")
    if not ideas:
        raise ValueError("[TitleAgent] No idea_candidates in state.")

    if chosen_idea_id:
        idea = next((i for i in ideas["ideas"] if i["idea_id"] == chosen_idea_id), ideas["ideas"][0])
    else:
        idea = next((i for i in ideas["ideas"] if i["idea_id"] == ideas["recommended_idea_id"]), ideas["ideas"][0])

    state_store.update_state("chosen_idea", idea)
    print(f"[TitleAgent] Generating titles for: {idea['hypothesis'][:80]}...")

    prompt = f"""Generate 3 academic paper titles for this research idea:

Hypothesis: {idea['hypothesis']}
Novelty: {idea['novelty_explanation']}
Experiment: {idea['minimum_viable_experiment']}

Return a JSON object with this exact structure:
{{
  "descriptive": "title that states exactly what was done",
  "punchy": "short hook: detailed subtitle format",
  "question_form": "title as a research question?",
  "recommended": "descriptive",
  "rationale": "why this title is best"
}}

Return ONLY the JSON object."""

    response = client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1024,
        temperature=0.7
    )

    raw_output = response.choices[0].message.content.strip()
    if "```" in raw_output:
        raw_output = raw_output.split("```")[1]
        if raw_output.startswith("json"):
            raw_output = raw_output[4:]

    result = json.loads(raw_output)
    state_store.update_state("title_options", result)
    audit_logger.log("title_agent", {"idea_id": idea["idea_id"]}, result)
    print(f"[TitleAgent] Recommended title: {recommended_title(result)}")
    return result


def recommended_title(result):
    """Return the recommended title from a title_agent result.

    The prompt asks for "recommended" to hold a key name ("descriptive"), but
    at temperature 0.7 the model often puts the title itself there instead.
    Both shapes are accepted; indexing straight through crashes on the second.
    """
    recommended = result.get("recommended", "descriptive")
    return result.get(recommended) or recommended