import json
from groq import Groq
from memory import state_store, audit_logger
from memory.token_tracker import log_usage, print_status
import config


def run(chosen_idea_id=None):
    client = Groq(api_key=config.GROQ_API_KEY)
    ideas = state_store.get_state("idea_candidates")
    input_topic = state_store.get_state("input_topic")
    if not ideas:
        raise ValueError("[TitleAgent] No idea_candidates in state.")

    if chosen_idea_id:
        idea = next((i for i in ideas["ideas"] if i["idea_id"] == chosen_idea_id), ideas["ideas"][0])
    else:
        idea = ideas["ideas"][0]

    state_store.update_state("chosen_idea", idea)

    topic = input_topic.get("topic", "")
    domain = input_topic.get("domain", "")
    venue = input_topic.get("target_venue", "IEEE")

    print("[TitleAgent] Generating titles...")

    prompt = "Generate 3 IEEE-quality paper titles for this research.\n\n"
    prompt += "Topic: " + topic + "\n"
    prompt += "Domain: " + domain + "\n"
    prompt += "Venue: " + venue + "\n"
    prompt += "Hypothesis: " + idea["hypothesis"] + "\n"
    prompt += "Novelty: " + idea["novelty_explanation"] + "\n\n"
    prompt += "IEEE TITLE RULES:\n"
    prompt += "- 8-18 words\n"
    prompt += "- Must include: method/approach + domain + specific contribution\n"
    prompt += "- Never use: novel, first, state-of-the-art, revolutionary\n"
    prompt += "- Descriptive title: state exactly what was done\n"
    prompt += "- Punchy title: create a METHOD ACRONYM then colon then subtitle\n"
    prompt += "  Example: 'XAI-TCNN: Explainable Temporal Convolutional Network for Traffic Prediction'\n"
    prompt += "- Question title: pose as a research question\n\n"
    prompt += "Return ONLY this JSON:\n"
    prompt += "{\n"
    prompt += '  "descriptive": "title stating exactly what was done with domain",\n'
    prompt += '  "punchy": "ACRONYM: Full descriptive subtitle with domain",\n'
    prompt += '  "question_form": "Can X Improve Y in Z Domain?",\n'
    prompt += '  "method_acronym": "ACRONYM used in punchy title",\n'
    prompt += '  "recommended": "punchy",\n'
    prompt += '  "rationale": "why this title is best for ' + venue + '"\n'
    prompt += "}"

    response = client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": (
                "IEEE paper title expert. "
                "Always create a memorable method acronym for the punchy title. "
                "Return ONLY valid JSON."
            )},
            {"role": "user", "content": prompt}
        ],
        max_tokens=800,
        temperature=0.7
    )

    raw = response.choices[0].message.content.strip()
    used = log_usage("title_agent", prompt, raw)
    print_status("title_agent", used)

    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    result = json.loads(raw)

    rec = result.get("recommended", "punchy")
    if rec not in ["descriptive", "punchy", "question_form"]:
        result["recommended"] = "punchy"
        rec = "punchy"

    state_store.update_state("title_options", result)
    audit_logger.log("title_agent", {"idea_id": idea["idea_id"]}, result)

    print("[TitleAgent] Acronym: " + result.get("method_acronym", "N/A"))
    print("[TitleAgent] Recommended: " + result.get(rec, ""))
    return result