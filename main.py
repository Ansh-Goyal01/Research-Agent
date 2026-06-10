import sys
from orchestrator import run

# ============================================================
# CHANGE ONLY THIS SECTION TO RUN A DIFFERENT PAPER
# ============================================================
TOPIC     = "hallucination detection in large language models using uncertainty quantification and conformal prediction"
DOMAIN    = "natural language processing and AI safety"
VENUE     = "ACL 2026"
YEAR_FROM = 2021
YEAR_TO   = 2025
MAX_PAPERS = 20
# ============================================================

if __name__ == "__main__":
    resume = "--resume" in sys.argv

    topic_data = {
        "topic":        TOPIC,
        "domain":       DOMAIN,
        "target_venue": VENUE,
        "date_range":   [YEAR_FROM, YEAR_TO],
        "max_papers":   MAX_PAPERS
    }

    run(topic_data, resume=resume)