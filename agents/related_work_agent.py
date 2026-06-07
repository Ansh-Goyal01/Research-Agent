import json
import time
from groq import Groq
from memory import state_store, audit_logger
from memory.token_tracker import log_usage, print_status
import config


def run():
    client = Groq(api_key=config.GROQ_API_KEY)
    paper_list = state_store.get_state("paper_list")
    input_topic = state_store.get_state("input_topic")
    gap_analysis = state_store.get_state("gap_analysis")

    if not paper_list:
        raise ValueError("[RelatedWorkAgent] No paper_list in state.")

    papers = paper_list["papers"]
    topic = input_topic.get("topic", "")
    domain = input_topic.get("domain", "")

    print("[RelatedWorkAgent] Synthesizing " + str(len(papers)) + " papers...")

    papers_summary = []
    for p in papers[:18]:
        papers_summary.append({
            "title": p.get("title", ""),
            "year": p.get("year", 0),
            "venue": p.get("venue", ""),
            "summary": p.get("abstract_summary", p.get("abstract", ""))[:150],
            "limitations": p.get("stated_limitations", [])[:2]
        })

    time.sleep(4)

    prompt = "Analyze these papers and create a structured related work outline.\n\n"
    prompt += "Topic: " + topic + "\n"
    prompt += "Domain: " + domain + "\n\n"
    prompt += "Papers:\n" + json.dumps(papers_summary, indent=1) + "\n\n"
    prompt += "Tasks:\n"
    prompt += "1. Group papers into exactly 3 thematic subsections relevant to: " + topic + "\n"
    prompt += "2. For each subsection identify: theme name, papers that belong, key findings, contradictions between papers, remaining gap\n"
    prompt += "3. Identify which papers directly motivate our work\n"
    prompt += "4. Identify contradictions or debates between papers\n\n"
    prompt += "Return JSON:\n"
    prompt += "{\n"
    prompt += '  "subsections": [\n'
    prompt += '    {\n'
    prompt += '      "theme": "theme name",\n'
    prompt += '      "papers": ["title1", "title2"],\n'
    prompt += '      "key_finding": "what these papers collectively show",\n'
    prompt += '      "contradiction": "any disagreement between papers or null",\n'
    prompt += '      "remaining_gap": "what is still unsolved"\n'
    prompt += '    }\n'
    prompt += '  ],\n'
    prompt += '  "papers_motivating_our_work": ["title1", "title2"],\n'
    prompt += '  "key_contradictions": ["contradiction1"],\n'
    prompt += '  "synthesis_sentence": "one sentence placing our work in context of all above"\n'
    prompt += "}\n"
    prompt += "Return ONLY the JSON object."

    response = client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": (
                "You are a research synthesis expert for " + domain + ". "
                "Identify themes, contradictions and gaps across papers. "
                "Return ONLY valid JSON."
            )},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000,
        temperature=0.3
    )

    raw = response.choices[0].message.content.strip()
    used = log_usage("related_work_agent", prompt, raw)
    print_status("related_work_agent", used)

    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    result = json.loads(raw)
    state_store.update_state("related_work_outline", result)
    audit_logger.log("related_work_agent", {"papers": len(papers)}, result)

    print("[RelatedWorkAgent] Subsections: " + str(len(result.get("subsections", []))))
    for s in result.get("subsections", []):
        print("  - " + s["theme"] + " (" + str(len(s["papers"])) + " papers)")

    return result