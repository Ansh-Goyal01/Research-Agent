import json
from groq import Groq
from tools.paper_search import search_papers
from memory import state_store, audit_logger
import config

SYSTEM_PROMPT = """You are a rigorous academic literature scout. You receive a list of real papers fetched from arXiv and Semantic Scholar.
Your job is to analyze each paper and extract structured information.
For each paper produce:
- abstract_summary: 2-3 sentence summary of what the paper does
- key_contributions: list of 2-4 bullet points
- stated_limitations: list of limitations the authors mention or that are obvious

Rules:
- Do NOT invent papers. Only work with what is provided.
- Do NOT fabricate URLs, authors, or titles.
- Return ONLY valid JSON. No markdown, no explanation, no code blocks.
- Output must be a JSON array of paper objects."""

def run(topic_data):
    client = Groq(api_key=config.GROQ_API_KEY)
    topic = topic_data["topic"]
    start_year = topic_data.get("date_range", [2021, 2025])[0]
    max_papers = topic_data.get("max_papers", 25)

    print(f"[Scout] Searching papers for: {topic}")
    raw_papers = search_papers(topic, max_results=max_papers, start_year=start_year)
    print(f"[Scout] Found {len(raw_papers)} papers. Analyzing...")

    papers_text = json.dumps(raw_papers, indent=2)

    prompt = f"""Here are real papers fetched from arXiv and Semantic Scholar on the topic: "{topic}"

{papers_text}

For each paper, return a JSON array where each object has these exact fields:
- title (string)
- authors (list of strings)
- year (integer)
- venue (string)
- url (string)
- abstract_summary (string, 2-3 sentences)
- key_contributions (list of strings)
- stated_limitations (list of strings)
- semantic_scholar_id (string or null)

Return ONLY the JSON array. Nothing else."""

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

    try:
        if "```" in raw_output:
            raw_output = raw_output.split("```")[1]
            if raw_output.startswith("json"):
                raw_output = raw_output[4:]
        papers = json.loads(raw_output)
    except Exception as e:
        print(f"[Scout] JSON parse error: {e}")
        papers = raw_papers

    result = {
        "papers": papers,
        "search_queries_used": [topic],
        "timestamp": __import__("datetime").datetime.now().isoformat()
    }

    state_store.update_state("paper_list", result)
    audit_logger.log("scout", topic_data, result)
    print(f"[Scout] Done. {len(papers)} papers saved to state.")
    return result