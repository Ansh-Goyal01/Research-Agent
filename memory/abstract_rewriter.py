"""
abstract_rewriter.py  — Research Agent
Rewrites the paper_writer abstract to strict IEEE IMRaD format.

Fix: system prompt explicitly bans BNN/MC-dropout claims and guarantees
the rewrite describes only models actually run in the experiment.
"""
import time
from memory.token_tracker import log_usage, print_status
import config


def rewrite(client, abstract, metrics_str, title, topic):
    print("[AbstractRewriter] Rewriting abstract to IMRaD format...")
    time.sleep(4)

    prompt = (
        "Rewrite this abstract to strictly follow IEEE IMRaD format.\n\n"
        "Original abstract:\n" + abstract + "\n\n"
        "Paper title: " + title + "\n"
        "Topic: " + topic + "\n"
        "Actual metrics (must appear verbatim): " + metrics_str + "\n\n"
        "STRICT REQUIREMENTS:\n"
        "- Exactly 150-200 words\n"
        "- Sentence 1: Background (broader problem, real-world importance)\n"
        "- Sentence 2: Problem (specific gap this paper addresses)\n"
        "- Sentences 3-4: Method (what you propose and how it works technically).\n"
        "  IMPORTANT: Describe only models that appear in the metrics string above.\n"
        "  Do NOT mention Bayesian Neural Networks, Monte Carlo dropout, or BNN\n"
        "  unless these exact words appear in the metrics.\n"
        "- Sentences 5-6: Results (exact numbers from the metrics string above)\n"
        "- Sentence 7: Conclusion (what this means for the field)\n"
        "- No citations in abstract\n"
        "- Third person only (never 'we')\n"
        "- Must include at least 2 specific metric values from the metrics string\n"
        "- Do NOT start any sentence with a comma or fragment\n\n"
        "Return ONLY the rewritten abstract text. No labels, no headings, no explanation."
    )

    response = client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": (
                "IEEE abstract writer. Return ONLY the abstract text, "
                "150-200 words, strict IMRaD. Never mention BNN or Monte Carlo dropout "
                "unless the actual experiment used them."
            )},
            {"role": "user", "content": prompt}
        ],
        max_tokens=450,
        temperature=0.3
    )

    rewritten = response.choices[0].message.content.strip()
    used = log_usage("abstract_rewriter", prompt, rewritten)
    print_status("abstract_rewriter", used)

    word_count = len(rewritten.split())
    print("[AbstractRewriter] Word count: " + str(word_count))

    # Hard guard: if word count too low, return original
    if word_count < 50:
        print("[AbstractRewriter] Warning: rewrite too short, keeping original.")
        return abstract

    return rewritten
