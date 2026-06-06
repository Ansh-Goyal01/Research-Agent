import json
from groq import Groq
from memory import state_store, audit_logger
import config

def run():
    client = Groq(api_key=config.GROQ_API_KEY)
    paper_list = state_store.get_state("paper_list")
    if not paper_list:
        raise ValueError("[GapAnalyst] No paper_list in state.")
    papers = paper_list["papers"]
    print("[GapAnalyst] Analyzing " + str(len(papers)) + " papers...")
    prompt = (
        "Analyze these papers and find research gaps:\n\n" +
        json.dumps(papers, indent=2) +
        "\n\nReturn a JSON object with this structure:\n"
        "{\n"
        "  \"gaps\": [\n"
        "    {\n"
        "      \"gap_id\": \"gap_1\",\n"
        "      \"description\": \"description\",\n"
        "      \"supporting_paper_titles\": [\"title1\"],\n"
        "      \"impact_score\": 8.5,\n"
        "      \"feasibility_score\": 7.0\n"
        "    }\n"
        "  ],\n"
        "  \"top_gap_id\": \"gap_1\",\n"
        "  \"rationale\": \"why this gap matters\"\n"
        "}\n"
        "Return ONLY the JSON object."
    )
    response = client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": "You are a research analyst. Return ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=config.MAX_TOKENS,
        temperature=0.3
    )
    raw = response.choices[0].message.content.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    result = json.loads(raw)
    state_store.update_state("gap_analysis", result)
    audit_logger.log("gap_analyst", {"papers": len(papers)}, result)
    print("[GapAnalyst] Found " + str(len(result["gaps"])) + " gaps.")
    return result
