"""
Tools for the AI Research Agent.

1. tavily_search: Live web search via Tavily API
2. summarize_url: Fetch webpage and summarize with Groq
3. save_report: Save final report to local file
"""

import os
import re
import requests

from bs4 import BeautifulSoup

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from tavily import TavilyClient



# CONFIGURATION


MODEL_NAME = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)

MAX_SEARCH_RESULTS = 3

MAX_SEARCH_SNIPPET_CHARS = 400

MAX_WEBPAGE_CHARS = 1800

MAX_SUMMARY_TOKENS = 500



# GROQ LLM


def get_groq_llm():

    """
    Create Groq LLM for URL summarization.
    """

    return ChatGroq(

        model=MODEL_NAME,

        groq_api_key=os.getenv("GROQ_API_KEY"),

        temperature=0.2,

        max_tokens=MAX_SUMMARY_TOKENS,

    )



# TAVILY SEARCH TOOL


@tool
def tavily_search(query: str) -> str:

    """
    Search the web for up-to-date information
    using Tavily API.
    """

    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:

        return (
            "Error: TAVILY_API_KEY is not set. "
            "Please add it to your .env file."
        )

    try:

        client = TavilyClient(api_key=api_key)

        response = client.search(

            query=query,

            max_results=MAX_SEARCH_RESULTS

        )

        results = response.get("results", [])

        if not results:

            return f"No search results found for: '{query}'."

        formatted = [
            f"### Web Search Results for: '{query}'\n"
        ]

        for idx, item in enumerate(results, start=1):

            title = item.get("title", "Link")

            url = item.get("url", "")

            snippet = item.get("content", "").strip()

            snippet = snippet[:MAX_SEARCH_SNIPPET_CHARS]

            formatted.append(

                f"{idx}. **{title}**\n"

                f"   - URL: {url}\n"

                f"   - Info: {snippet}\n"

            )

        return "\n".join(formatted)

    except Exception as e:

        return f"Tavily search error: {e}"



# URL SUMMARIZATION TOOL

@tool
def summarize_url(url: str) -> str:
    """
    Fetch webpage content and generate a concise summary.
    """

    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        resp = requests.get(
            url,
            headers=headers,
            timeout=8
        )

        resp.raise_for_status()

        # Parse HTML
        soup = BeautifulSoup(
            resp.text,
            "html.parser"
        )

        # Remove unnecessary tags
        for tag in soup([
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
            "form"
        ]):
            tag.decompose()

        # Extract readable text
        text = re.sub(
            r"\s+",
            " ",
            soup.get_text()
        ).strip()

        # Keep webpage input small
        text = text[:1500]

        if not text:
            return f"Could not extract readable text from {url}."

        # Create summarization LLM
        llm = get_groq_llm()

        prompt = (
            "Summarize the following webpage in 5 short bullet points.\n"
            "Include only important facts and metrics.\n"
            "Do not invent information.\n"
            "Keep the summary under 600 characters.\n\n"
            f"URL: {url}\n\n"
            f"Content:\n{text}"
        )

        summary = llm.invoke(prompt)

        # Limit summary returned to the main agent
        summary_text = str(summary.content).strip()[:800]

        return (
            f"### Summary of {url}\n\n"
            f"{summary_text}"
        )

    except requests.RequestException as e:
        return f"Failed to fetch webpage {url}: {str(e)}"

    except Exception as e:
        return f"Failed to summarize webpage {url}: {str(e)}"
    
    

# SAVE REPORT TOOL


@tool
def save_report(report_text: str, topic: str) -> str:
    """
    Save a concise research report to a text file.
    The report must be under 1800 characters.
    """
    try:
        report_text = report_text[:1800]

        reports_dir = "reports"
        os.makedirs(reports_dir, exist_ok=True)

        clean_name = re.sub(r"[^a-zA-Z0-9_-]", "_", topic)
        filepath = os.path.join(
            reports_dir,
            f"{clean_name}.txt"
        )

        with open(filepath, "w", encoding="utf-8") as file:
            file.write(report_text)

        return f"Report saved successfully at: {filepath}"

    except Exception as e:
        return f"Failed to save report: {str(e)}"