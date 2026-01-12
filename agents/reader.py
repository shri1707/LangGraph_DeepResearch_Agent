import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import List
import requests
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from tools.source_classifier import classify_source_type
load_dotenv()

MAX_FACTS_PER_SOURCE = 5
MAX_SOURCES_TO_READ = 10

class ExtractedFacts(BaseModel):
    facts: List[str] = Field(
        description=(
            "List of explicit factual statements directly stated in the text. "
            "Do NOT infer, summarize, or add opinions."
        )
    )

reader_llm = ChatOpenAI(
    model="mistralai/mistral-7b-instruct",
    temperature=0,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
    ).with_structured_output(ExtractedFacts)

def fetch_page_text(url: str, max_chars: int = 6000) -> str:
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove scripts & styles
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        return text[:max_chars]

    except Exception:
        return ""
    
def reader_agent(state):
    seen_urls = set()
    notes = []

    for source in state["sources"]: # [:MAX_SOURCES_TO_READ]:
        url = source["url"]

        if url in seen_urls or classify_source_type(url) == "forum":
            continue
        seen_urls.add(url)

        page_text = fetch_page_text(url)
        if not page_text:
            continue

        system_message = SystemMessage(
            content=(
                "You are an information extraction agent. "
                "Extract ONLY explicit factual statements from the text. "
                "Do NOT summarize, infer, or add opinions."
            )
        )

        human_message = HumanMessage(content=page_text)

        extracted = reader_llm.invoke([system_message, human_message])

        if not extracted.facts:
            continue

        notes.append({
            "url": url,
            "title": source.get("title"),
            "source_type": classify_source_type(url),
            "facts": extracted.facts[:MAX_FACTS_PER_SOURCE]
        })

    return {"notes": notes}



