import requests
import urllib.parse
import xml.etree.ElementTree as ET
import config

def search_arxiv(query, max_results=10, start_year=2021):
    params = {
        "search_query": "all:" + query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }
    url = config.ARXIV_BASE_URL + "?" + urllib.parse.urlencode(params)
    response = requests.get(url, timeout=15)
    root = ET.fromstring(response.content)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    papers = []
    for entry in root.findall("atom:entry", ns):
        try:
            year = int(entry.find("atom:published", ns).text[:4])
            if year < start_year:
                continue
            authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]
            papers.append({
                "title": entry.find("atom:title", ns).text.strip().replace("\n", " "),
                "authors": authors[:5],
                "year": year,
                "venue": "arXiv",
                "url": entry.find("atom:id", ns).text.strip(),
                "abstract": entry.find("atom:summary", ns).text.strip().replace("\n", " ")[:500],
                "semantic_scholar_id": None
            })
        except Exception:
            continue
    return papers

def search_semantic_scholar(query, max_results=10):
    url = config.SEMANTIC_SCHOLAR_BASE_URL + "/paper/search"
    params = {
        "query": query,
        "limit": max_results,
        "fields": "title,authors,year,venue,externalIds,abstract"
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        data = response.json()
        papers = []
        for p in data.get("data", []):
            if not p.get("year") or p["year"] < 2021:
                continue
            papers.append({
                "title": p.get("title", ""),
                "authors": [a["name"] for a in p.get("authors", [])[:5]],
                "year": p.get("year", 0),
                "venue": p.get("venue", "Unknown"),
                "url": "https://www.semanticscholar.org/paper/" + p.get("paperId", ""),
                "abstract": p.get("abstract", "")[:500] if p.get("abstract") else "",
                "semantic_scholar_id": p.get("paperId")
            })
        return papers
    except Exception:
        return []

def search_papers(query, max_results=25, start_year=2021):
    arxiv_results = search_arxiv(query, max_results=max_results//2, start_year=start_year)
    ss_results = search_semantic_scholar(query, max_results=max_results//2)
    seen = set()
    combined = []
    for p in arxiv_results + ss_results:
        key = p["title"].lower()[:50]
        if key not in seen:
            seen.add(key)
            combined.append(p)
    return combined[:max_results]
