import requests
import urllib.parse
import xml.etree.ElementTree as ET
import time
import config


def search_arxiv(query, max_results=10, start_year=2021):
    try:
        params = {
            "search_query": "all:" + query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        url = config.ARXIV_BASE_URL + "?" + urllib.parse.urlencode(params)
        response = requests.get(url, timeout=20)
        if response.status_code != 200:
            print("[Search] arXiv status: " + str(response.status_code))
            return []
        content = response.content
        if not content or len(content) < 10:
            return []
        root = ET.fromstring(content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        papers = []
        for entry in root.findall("atom:entry", ns):
            try:
                year = int(entry.find("atom:published", ns).text[:4])
                if year < start_year:
                    continue
                authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]
                abstract = entry.find("atom:summary", ns).text.strip().replace("\n", " ")[:600]
                papers.append({
                    "title": entry.find("atom:title", ns).text.strip().replace("\n", " "),
                    "authors": authors[:5],
                    "year": year,
                    "venue": "arXiv",
                    "url": entry.find("atom:id", ns).text.strip(),
                    "abstract": abstract,
                    "semantic_scholar_id": None
                })
            except Exception:
                continue
        return papers
    except Exception as e:
        print("[Search] arXiv error: " + str(e))
        return []


def search_semantic_scholar(query, max_results=10):
    try:
        url = config.SEMANTIC_SCHOLAR_BASE_URL + "/paper/search"
        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,authors,year,venue,externalIds,abstract"
        }
        response = requests.get(url, params=params, timeout=20)
        if response.status_code != 200:
            return []
        data = response.json()
        papers = []
        for p in data.get("data", []):
            if not p.get("year") or p["year"] < 2021:
                continue
            if not p.get("title"):
                continue
            papers.append({
                "title": p.get("title", ""),
                "authors": [a["name"] for a in p.get("authors", [])[:5]],
                "year": p.get("year", 0),
                "venue": p.get("venue", "Unknown"),
                "url": "https://www.semanticscholar.org/paper/" + p.get("paperId", ""),
                "abstract": p.get("abstract", "")[:600] if p.get("abstract") else "",
                "semantic_scholar_id": p.get("paperId")
            })
        return papers
    except Exception as e:
        print("[Search] Semantic Scholar error: " + str(e))
        return []


def search_papers(query, max_results=20, start_year=2021):
    all_papers = []
    seen = set()

    queries = [
        query,
        query.split("and")[0].strip(),
        " ".join(query.split()[:4])
    ]

    for q in queries:
        print("[Search] Query: " + q)
        arxiv = search_arxiv(q, max_results=8, start_year=start_year)
        time.sleep(1)
        ss = search_semantic_scholar(q, max_results=8)
        for p in arxiv + ss:
            if not p.get("title"):
                continue
            key = p["title"].lower()[:50]
            if key not in seen:
                seen.add(key)
                all_papers.append(p)
        if len(all_papers) >= max_results:
            break
        time.sleep(1)

    print("[Search] Total unique papers found: " + str(len(all_papers)))
    return all_papers[:max_results]