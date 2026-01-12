from graph.research_graph import build_graph
from schemas.state import ResearchState
from pprint import pprint

graph = build_graph()

if __name__ == "__main__":

    query = "Compare self-hosted LLMs vs cloud LLMs for enterprises"

    state = ResearchState(query=query)

    result = graph.invoke(state)

    print("\n--- PLAN OUTPUT ---")
    pprint(result["plan"], width=100)

    print("\n--- SOURCES ---")
    for s in result["sources"][:5]:
        print(s["title"], "→", s["url"])

    print("\n--- EXTRACTED FACTS ---")
    for note in result["notes"][:4]:
        print("\nSOURCE:", note["url"])
        print("\nTitle:", note["title"])
        print("\nsource_type:", note["source_type"])
        for fact in note["facts"][:2]:
            print("-", fact)

    print("\n--- VERIFIED FACTS ---")
    for fact in result["verified_facts"][:4]:
        print("-", fact)

    print("\n--- CONFLICTS ---")
    for conflict in result["conflicts"]:
        print("-", conflict)
    print("\n--- UNCERTAINTY ---")
    for uncertainty in result["uncertain_facts"]:
        print("-", uncertainty)
    print("\n--- FINAL ANSWER ---")
    print(result["final_answer"])