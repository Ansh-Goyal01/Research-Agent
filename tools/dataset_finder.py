import requests
import time


def search_kaggle_datasets(topic, max_results=5):
    try:
        url = "https://www.kaggle.com/api/v1/datasets/list"
        params = {
            "search": topic,
            "sortBy": "votes",
            "maxSize": 500000000,
            "page": 1
        }
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            return []
        data = response.json()
        results = []
        for d in data[:max_results]:
            results.append({
                "name": d.get("title", ""),
                "url": "https://www.kaggle.com/datasets/" + d.get("ref", ""),
                "votes": d.get("totalVotes", 0),
                "size": str(d.get("totalBytes", 0) // 1000000) + "MB",
                "license": d.get("licenseName", "Unknown"),
                "source": "Kaggle"
            })
        return results
    except Exception as e:
        return []


def search_uci_datasets(topic):
    try:
        url = "https://archive.ics.uci.edu/api/datasets/list"
        params = {"search": topic, "numAttributes": ""}
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            return []
        data = response.json()
        results = []
        for d in data.get("datasets", [])[:5]:
            results.append({
                "name": d.get("name", ""),
                "url": "https://archive.ics.uci.edu/dataset/" + str(d.get("id", "")),
                "votes": d.get("numCitations", 0),
                "size": str(d.get("numInstances", "Unknown")) + " samples",
                "license": "CC BY 4.0",
                "source": "UCI"
            })
        return results
    except Exception:
        return []


def search_huggingface_datasets(topic):
    try:
        url = "https://huggingface.co/api/datasets"
        params = {
            "search": topic,
            "sort": "likes",
            "limit": 5
        }
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            return []
        data = response.json()
        results = []
        for d in data[:5]:
            results.append({
                "name": d.get("id", ""),
                "url": "https://huggingface.co/datasets/" + d.get("id", ""),
                "votes": d.get("likes", 0),
                "size": "varies",
                "license": d.get("license", "Unknown"),
                "source": "HuggingFace"
            })
        return results
    except Exception:
        return []


def find_best_dataset(topic, domain):
    print("[DatasetFinder] Searching for best dataset for: " + topic)
    all_datasets = []

    uci = search_uci_datasets(topic)
    print("[DatasetFinder] UCI found: " + str(len(uci)))
    all_datasets.extend(uci)
    time.sleep(0.5)

    hf = search_huggingface_datasets(topic)
    print("[DatasetFinder] HuggingFace found: " + str(len(hf)))
    all_datasets.extend(hf)
    time.sleep(0.5)

    kaggle = search_kaggle_datasets(topic)
    print("[DatasetFinder] Kaggle found: " + str(len(kaggle)))
    all_datasets.extend(kaggle)

    if not all_datasets:
        print("[DatasetFinder] No datasets found, using sklearn fallback")
        return _sklearn_fallback(topic, domain)

    all_datasets.sort(key=lambda x: x.get("votes", 0), reverse=True)

    best = all_datasets[0]
    print("[DatasetFinder] Best dataset: " + best["name"] + " (" + best["source"] + ", votes=" + str(best["votes"]) + ")")
    return best


def _sklearn_fallback(topic, domain):
    topic_lower = topic.lower() + " " + domain.lower()
    if any(w in topic_lower for w in ["medical", "clinical", "health", "cancer"]):
        return {
            "name": "sklearn breast_cancer",
            "url": "sklearn.datasets.load_breast_cancer()",
            "votes": 0,
            "size": "569 samples",
            "license": "BSD",
            "source": "sklearn"
        }
    elif any(w in topic_lower for w in ["traffic", "transport", "congestion"]):
        return {
            "name": "UCI Metro Interstate Traffic Volume",
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz",
            "votes": 0,
            "size": "48MB",
            "license": "CC BY 4.0",
            "source": "UCI"
        }
    elif any(w in topic_lower for w in ["text", "nlp", "language", "sentiment"]):
        return {
            "name": "20 Newsgroups",
            "url": "sklearn.datasets.fetch_20newsgroups()",
            "votes": 0,
            "size": "18846 samples",
            "license": "BSD",
            "source": "sklearn"
        }
    else:
        return {
            "name": "sklearn digits",
            "url": "sklearn.datasets.load_digits()",
            "votes": 0,
            "size": "1797 samples",
            "license": "BSD",
            "source": "sklearn"
        }