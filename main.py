from orchestrator import run

# ============================================================
# CHANGE ONLY THIS SECTION TO RUN A DIFFERENT PAPER
# ============================================================
TOPIC = "conformal prediction for uncertainty quantification in medical image classification"
DOMAIN = "probabilistic machine learning and medical imaging"
VENUE = "NeurIPS 2026"
YEAR_FROM = 2021
YEAR_TO = 2025
MAX_PAPERS = 20
# ============================================================

if __name__ == "__main__":
    topic_data = {
        "topic": TOPIC,
        "domain": DOMAIN,
        "target_venue": VENUE,
        "date_range": [YEAR_FROM, YEAR_TO],
        "max_papers": MAX_PAPERS
    }
    run(topic_data)