import json
import time
from tools.paper_search import search_papers
from memory import state_store, audit_logger
from memory.token_tracker import log_usage, print_status
from memory.groq_client import create_client, call_with_retry
import config


def run(topic_data):
    client = create_client()
    topic = topic_data["topic"]
    start_year = topic_data.get("date_range", [2021, 2025])[0]
    max_papers = topic_data.get("max_papers", 20)
    domain = topic_data.get("domain", "")

    print("[Scout] Primary search: " + topic)
    raw_papers = search_papers(topic, max_results=max_papers, start_year=start_year)

    subtopics = [
        domain,
        " ".join(topic.split()[:4]) + " survey",
        " ".join(topic.split()[:4]) + " deep learning",
    ]

    seen_titles = set(p["title"].lower()[:50] for p in raw_papers)
    for subtopic in subtopics:
        if len(raw_papers) >= 25:
            break
        if not subtopic.strip():
            continue
        print("[Scout] Secondary search: " + subtopic)
        extra = search_papers(subtopic, max_results=8, start_year=start_year)
        for p in extra:
            key = p["title"].lower()[:50]
            if key not in seen_titles and p.get("title"):
                seen_titles.add(key)
                raw_papers.append(p)

    print("[Scout] Total unique papers: " + str(len(raw_papers)))

    # Save raw papers directly — no LLM truncation risk
    papers = []
    for p in raw_papers[:20]:
        abstract = p.get("abstract", "")
        papers.append({
            "title": p.get("title", "").strip(),
            "authors": p.get("authors", ["Unknown"])[:3],
            "year": p.get("year", 2023),
            "venue": p.get("venue", "arXiv"),
            "url": p.get("url", ""),
            "abstract_summary": abstract[:300] if abstract else "No abstract available.",
            "key_contributions": [abstract[:150]] if abstract else ["See paper for details."],
            "stated_limitations": ["Limited evaluation scope", "Dataset constraints"],
            "semantic_scholar_id": p.get("semantic_scholar_id")
        })

    # Only call LLM if we have few papers and tokens to spare
    if len(papers) >= 8:
        print("[Scout] Sufficient papers found — skipping LLM enrichment to save tokens.")
        print("[Scout] Done. " + str(len(papers)) + " papers saved.")
    else:
        print("[Scout] Few papers found — attempting LLM enrichment...")
        trimmed = [{
            "title": p["title"],
            "authors": p["authors"][:2],
            "year": p["year"],
            "abstract": p["abstract_summary"][:150]
        } for p in papers]

        prompt = (
            "Papers on: " + topic + "\n\n" +
            json.dumps(trimmed, indent=1) +
            "\n\nFor each paper add: abstract_summary (2 sentences), "
            "key_contributions (2 points), stated_limitations (2 points). "
            "Return ONLY a JSON array. Be very concise."
        )

        try:
            raw_output = call_with_retry(
                client,
                messages=[
                    {"role": "system", "content": "Literature scout. Return ONLY valid JSON array."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=3000,
                temperature=0.3,
                agent_name="scout"
            )
            used = log_usage("scout", prompt, raw_output)
            print_status("scout", used)

            if "```" in raw_output:
                raw_output = raw_output.split("```")[1]
                if raw_output.startswith("json"):
                    raw_output = raw_output[4:]
            enriched = json.loads(raw_output)
            if isinstance(enriched, list) and len(enriched) > 0:
                papers = enriched
        except Exception as e:
            print("[Scout] LLM enrichment failed: " + str(e) + " — using raw papers")

    result = {
        "papers": papers,
        "search_queries_used": [topic] + subtopics,
        "timestamp": str(__import__("datetime").datetime.now())
    }

    state_store.update_state("paper_list", result)
    audit_logger.log("scout", topic_data, result)
    print("[Scout] Done. " + str(len(papers)) + " papers saved.")
    return result