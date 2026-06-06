import json
from groq import Groq
from memory import state_store, audit_logger
import config


def run(chosen_idea_id=None):
    client = Groq(api_key=config.GROQ_API_KEY)
    ideas = state_store.get_state("idea_candidates")
    if not ideas:
        raise ValueError("[TitleAgent] No idea_candidates in state.")
    if chosen_idea_id:
        idea = next((i for i in ideas["ideas"] if i["idea_id"] == chosen_idea_id), ideas["ideas"][0])
    else:
        idea = ideas["ideas"][0]
    state_store.update_state("chosen_idea", idea)
    print("[TitleAgent] Generating titles...")
    prompt = (
        "Generate 3 titles for this research:\n"
        "Hypothesis: " + idea["hypothesis"] + "\n"
        "Novelty: " + idea["novelty_explanation"] + "\n\n"
        "Return a JSON object with exactly these keys:\n"
        "{\n"
        '  "descriptive": "title stating what was done",\n'
        '  "punchy": "hook: subtitle",\n'
        '  "question_form": "title as question?",\n'
        '  "recommended": "descriptive",\n'
        '  "rationale": "why recommended"\n'
        "}\n"
        'The recommended field must be exactly one of: descriptive, punchy, question_form\n'
        "Return ONLY the JSON object."
    )
    response = client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": "You are an academic title writer. Return ONLY valid JSON. The recommended field must be exactly one of: descriptive, punchy, question_form"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1024,
        temperature=0.7
    )
    raw = response.choices[0].message.content.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    result = json.loads(raw)
    rec = result.get("recommended", "descriptive")
    if rec not in ["descriptive", "punchy", "question_form"]:
        result["recommended"] = "descriptive"
        rec = "descriptive"
    state_store.update_state("title_options", result)
    audit_logger.log("title_agent", {"idea_id": idea["idea_id"]}, result)
    print("[TitleAgent] Recommended: " + result.get(rec, ""))
    return result