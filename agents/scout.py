import json
from groq import Groq
from tools.paper_search import search_papers
from memory import state_store, audit_logger
from memory.token_tracker import log_usage, print_status
import config


def run(topic_data):
    client = Groq(api_key=config.GROQ_API_KEY)
    topic = topic_data["topic"]
    start_year = topic_data.get("date_range", [2021, 2025])[0]
    max_papers = topic_data.get("max_papers", 20)

    print("[Scout] Searching: " + topic)
    raw_papers = search_papers(topic, max_results=max_papers, start_year=start_year)
    print("[Scout] Found " + str(len(raw_papers)) + " papers.")

    trimmed = []
    for p in raw_papers[:12]:
        trimmed.append({
            "title": p.get("title", ""),
            "authors": p.get("authors", [])[:2],
            "year": p.get("year", 0),
            "venue": p.get("venue", ""),
            "url": p.get("url", ""),
            "abstract": p.get("abstract", "")[:200],
            "semantic_scholar_id": p.get("semantic_scholar_id")
        })

    prompt = (
        "Papers on: " + topic + "\n\n" +
        json.dumps(trimmed, indent=1) +
        "\n\nFor each paper return a JSON array with fields: "
        "title, authors, year, venue, url, abstract_summary (2 sentences), "
        "key_contributions (2 points max), stated_limitations (2 points max), semantic_scholar_id. "
        "Return ONLY the JSON array."
    )

    response = client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": "Literature scout. Return ONLY valid JSON arrays. Be concise."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2048,
        temperature=0.3
    )

    raw_output = response.choices[0].message.content.strip()
    used = log_usage("scout", prompt, raw_output)
    print_status("scout", used)

    if "```" in raw_output:
        raw_output = raw_output.split("```")[1]
        if raw_output.startswith("json"):
            raw_output = raw_output[4:]
    try:
        papers = json.loads(raw_output)
    except Exception:
        papers = trimmed

    result = {
        "papers": papers,
        "search_queries_used": [topic],
        "timestamp": str(__import__("datetime").datetime.now())
    }

    state_store.update_state("paper_list", result)
    audit_logger.log("scout", topic_data, result)
    print("[Scout] Done. " + str(len(papers)) + " papers saved.")
    return result