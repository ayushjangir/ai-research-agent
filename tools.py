"""
Tools for the AI Research Agent:
1. tavily_search: Live web search via Tavily API
2. summarize_url: Fetch webpage content and summarize with Groq LLM
3. save_report: Save final research report to a local text file
"""
import os
import re
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from tavily import TavilyClient


def get_groq_llm():
    """Helper to get ChatGroq instance with openai/gpt-oss-20b."""
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
    )


@tool
def tavily_search(query: str) -> str:
    """Search the web for up-to-date information on a query using Tavily API."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY is not set. Please add it to your .env file."

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, max_results=3)
        results = response.get("results", [])
        if not results:
            return f"No search results found for: '{query}'."

        formatted = [f"### Web Search Results for: '{query}'\n"]
        for idx, item in enumerate(results, start=1):
            snippet = item.get("content", "").strip()
            formatted.append(
                f"{idx}. **[{item.get('title', 'Link')}]({item.get('url')})**\n"
                f"   - URL: {item.get('url')}\n"
                f"   - Info: {snippet[:400]}...\n"
            )
        return "\n".join(formatted)
    except Exception as e:
        return f"Tavily search error: {e}"


@tool
def summarize_url(url: str) -> str:
    """Fetch the content of a webpage URL and generate a concise summary."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=8)
        resp.raise_for_status()

        # Parse and clean HTML
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = re.sub(r"\s+", " ", soup.get_text()).strip()[:3000]

        if not text:
            return f"Could not extract readable text from {url}."

        llm = get_groq_llm()
        prompt = (
            f"Synthesize a concise, high-density summary of this webpage ({url}):\n\n"
            f"{text}\n\n"
            "Include key facts, metrics, and main takeaways."
        )
        summary = llm.invoke(prompt)
        return f"### Summary of {url}\n\n{summary.content}"
    except Exception as e:
        return f"Failed to summarize URL {url}: {e}"


@tool
def save_report(report_text: str, topic: str = "research_report") -> str:
    """Save the synthesized research report to a local text file in the reports/ folder."""
    os.makedirs("reports", exist_ok=True)
    clean_name = re.sub(r"[^\w\s-]", "", topic).strip().lower().replace(" ", "_") or "research_report"
    filepath = os.path.join("reports", f"{clean_name}.txt")

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report_text)
        return f"SUCCESS: Report saved to {os.path.abspath(filepath)}"
    except Exception as e:
        return f"Error saving report: {e}"
