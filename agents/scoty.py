import json
from groq import Groq
from tools.paper_search import search_papers
from memory import state_store, audit_logger
import config


SYSTEM_PROMPT = """You are a rigorous academic literature scout.
For each paper produce:
- abstract_summary: 2-3 sentence summary
- key_contributions: list of 2-4 points
- stated_limitations: list of limitations
Rules:
- Do NOT invent papers
- Return ONLY valid JSON array, no markdown"""


def run(topic_data):
    client = Groq(api_key=config.GROQ_API_KEY)
    topic = topic_data["topic"]
    start_year = topic_data.get("date_range", [2021, 2025])[0]
    max_papers = topic_data.get("max_papers", 25)

    print("[Scout] Searching: " + topic)
    raw_papers = search_papers(topic, max_results=max_papers, start_year=start_year)
    print("[Scout] Found " + str(len(raw_papers)) + " papers. Analyzing...")

    prompt = (
        "Here are real papers fetched from arXiv on: " + topic + "\n\n" +
        json.dumps(raw_papers, indent=2) +
        "\n\nFor each paper return a JSON array where each object has:\n"
        "- title, authors, year, venue, url\n"
        "- abstract_summary (2-3 sentences)\n"
        "- key_contributions (list)\n"
        "- stated_limitations (list)\n"
        "- semantic_scholar_id (null if unknown)\n"
        "Return ONLY the JSON array."
    )

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
        parts = raw_output.split("```")
        raw_output = parts[1]
        if raw_output.startswith("json"):
            raw_output = raw_output[4:]

    try:
        papers = json.loads(raw_output)
    except Exception as e:
        print("[Scout] JSON parse error: " + str(e) + " - using raw results")
        papers = raw_papers

    result = {
        "papers": papers,
        "search_queries_used": [topic],
        "timestamp": str(__import__("datetime").datetime.now())
    }

    state_store.update_state("paper_list", result)
    audit_logger.log("scout", topic_data, result)
    print("[Scout] Done. " + str(len(papers)) + " papers saved.")
    return result