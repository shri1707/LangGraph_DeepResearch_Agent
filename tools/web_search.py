from tavily import TavilyClient
import os

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def web_search(query: str, max_results: int = 5):
    results = tavily.search(
        query=query,
        max_results=max_results
    )

    cleaned = []
    for r in results["results"]:
        cleaned.append({
            "url": r["url"],
            "title": r["title"],
            "snippet": r["content"],
            "source": "web",
            "query": query
        })

    return cleaned
