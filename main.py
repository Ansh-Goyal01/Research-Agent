from orchestrator import run

# ============================================================
# CHANGE ONLY THIS SECTION TO RUN A DIFFERENT PAPER
# ============================================================
TOPIC = "explainable AI for urban traffic congestion prediction and emergency vehicle prioritization"
DOMAIN = "intelligent transportation systems and XAI"
VENUE = "IEEE Transactions on Intelligent Transportation Systems"
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