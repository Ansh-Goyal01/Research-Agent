import time
from groq import Groq
from memory.token_tracker import log_usage, print_status
import config


def rewrite(client, abstract, metrics_str, title, topic):
    print("[AbstractRewriter] Rewriting abstract to IMRaD format...")
    time.sleep(4)

    prompt = "Rewrite this abstract to strictly follow IEEE IMRaD format.\n\n"
    prompt += "Original abstract:\n" + abstract + "\n\n"
    prompt += "Paper title: " + title + "\n"
    prompt += "Topic: " + topic + "\n"
    prompt += "Actual metrics (must appear): " + metrics_str + "\n\n"
    prompt += "STRICT REQUIREMENTS:\n"
    prompt += "- Exactly 150-200 words\n"
    prompt += "- Sentence 1: Background (broader problem)\n"
    prompt += "- Sentence 2: Problem (specific gap)\n"
    prompt += "- Sentences 3-4: Method (what you propose, how)\n"
    prompt += "- Sentences 5-6: Results (exact numbers from metrics)\n"
    prompt += "- Sentence 7: Conclusion (impact)\n"
    prompt += "- No citations in abstract\n"
    prompt += "- Third person only\n"
    prompt += "- Must include at least 2 specific metric values\n\n"
    prompt += "Return ONLY the rewritten abstract text. No labels, no explanation."

    response = client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": "IEEE abstract writer. Return ONLY the abstract text."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=400,
        temperature=0.3
    )

    rewritten = response.choices[0].message.content.strip()
    used = log_usage("abstract_rewriter", prompt, rewritten)
    print_status("abstract_rewriter", used)

    word_count = len(rewritten.split())
    print("[AbstractRewriter] Word count: " + str(word_count))
    return rewritten