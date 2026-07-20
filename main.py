from orchestrator import run

if __name__ == "__main__":
    topic_data = {
        "topic": "few-shot learning for image classification",
        "domain": "computer vision",
        "target_venue": "CVPR 2026",
        "date_range": [2021, 2025],
        "max_papers": 15
    }
    run(topic_data)