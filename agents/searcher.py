from schemas.state import ResearchState
from tools.web_search import web_search

MAX_RESULTS_PER_QUERY = 4

def search_agent(state: ResearchState):
    all_sources = []

    for query in state["search_queries"]:
        results = web_search(
            query=query,
            max_results=MAX_RESULTS_PER_QUERY
        )
        all_sources.extend(results)

    return {
        "sources": all_sources
    }
