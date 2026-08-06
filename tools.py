"""
Tools for the AI Autonomous Research Agent:
1. tavily_search - Perform live web search via Tavily API
2. summarize_url - Fetch web content and synthesize LLM summary
3. save_report - Save research report to a local text file
"""
import os
import re
import requests
from typing import Optional, List, Dict, Any
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from tavily import TavilyClient


def get_tavily_client(api_key: Optional[str] = None) -> Optional[TavilyClient]:
    """Retrieve initialized Tavily client using environment or provided key."""
    key = api_key or os.getenv("TAVILY_API_KEY")
    if not key or key == "your_tavily_api_key_here":
        return None
    try:
        return TavilyClient(api_key=key)
    except Exception as e:
        print(f"[Tools] Failed to initialize TavilyClient: {e}")
        return None


def execute_tavily_search(query: str, api_key: Optional[str] = None, max_results: int = 4) -> str:
    """
    Direct function to search Tavily API.
    Returns structured markdown search results with clear snippets.
    """
    client = get_tavily_client(api_key)
    if not client:
        return (
            "Error: TAVILY_API_KEY is not set or invalid. "
            "Please provide a valid Tavily API key in .env or via Streamlit sidebar."
        )

    try:
        response = client.search(query=query, max_results=max_results, search_depth="advanced")
        results = response.get("results", [])
        if not results:
            return f"No search results found for query: '{query}'."

        formatted = [f"### Web Search Results for: '{query}'\n"]
        for idx, item in enumerate(results, start=1):
            snippet = item.get('content', '').strip()
            if len(snippet) > 600:
                snippet = snippet[:600] + "..."
            formatted.append(
                f"**{idx}. [{item.get('title', 'Untitled')}]({item.get('url')})**\n"
                f"- **Source URL**: {item.get('url')}\n"
                f"- **Key Information**: {snippet}\n"
            )
        return "\n".join(formatted)
    except Exception as e:
        return f"Error executing Tavily web search for '{query}': {str(e)}"


def fetch_and_extract_url_text(url: str, max_chars: int = 3000) -> str:
    """Fetch webpage HTML and return clean extracted main text for detailed summarization."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        response = requests.get(url, headers=headers, timeout=8)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove script, style, nav, footer, header tags
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            element.decompose()

        # Extract remaining text
        text = soup.get_text(separator=" ", strip=True)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        
        if not text:
            return "Could not extract readable text content from the URL."
            
        return text[:max_chars]
    except Exception as e:
        return f"Failed to fetch content from URL ({url}): {str(e)}"


def execute_url_summary(
    url: str, 
    topic: str = "", 
    groq_api_key: Optional[str] = None,
    model_name: str = "llama-3.1-8b-instant"
) -> str:
    """Fetch URL web page content and synthesize a high-density structured summary."""
    extracted_text = fetch_and_extract_url_text(url, max_chars=4000)
    if extracted_text.startswith("Failed to fetch") or extracted_text.startswith("Could not extract"):
        return extracted_text

    key = groq_api_key or os.getenv("GROQ_API_KEY")
    if not key or key == "your_groq_api_key_here":
        return f"Extracted text snippet from {url}:\n{extracted_text[:600]}..."

    try:
        llm = ChatGroq(model=model_name, groq_api_key=key, temperature=0.2, max_tokens=1000)
        prompt = (
            f"Research Topic: {topic or 'General Overview'}\n"
            f"Source URL: {url}\n\n"
            f"Extracted Web Content:\n{extracted_text}\n\n"
            "Task: Synthesize an exceptionally high-quality, structured analytical summary of the webpage. "
            "Organize your summary under these headings:\n"
            "1. **Core Overview**: Key thesis or main announcement.\n"
            "2. **Technical Facts & Metrics**: Concrete data, architectures, numbers, or key arguments.\n"
            "3. **Key Takeaways & Topic Relevance**: Why this matters for the research topic.\n"
            "Keep the summary clear, factual, and high-density."
        )
        response = llm.invoke(prompt)
        return f"### Summary of [{url}]({url})\n\n{response.content}"
    except Exception as e:
        return f"Extracted snippet from {url}:\n{extracted_text[:600]}\n(Summary error: {e})"


def execute_save_report(report_text: str, topic: str, output_dir: str = "reports") -> str:
    """Save the final research report into a .txt / .md file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Sanitize topic string for filename
    clean_topic = re.sub(r"[^\w\s-]", "", topic).strip().lower()
    clean_topic = re.sub(r"[-\s]+", "_", clean_topic) or "research_report"
    
    filename = f"{clean_topic}.txt"
    filepath = os.path.join(output_dir, filename)
    
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report_text)
        abs_path = os.path.abspath(filepath)
        return f"SUCCESS: Research report successfully saved to file: {abs_path}"
    except Exception as e:
        return f"ERROR: Failed to write report file: {str(e)}"


# Define standard LangChain @tool wrappers for agent use
@tool
def tavily_search(query: str) -> str:
    """
    Search the web for up-to-date information on a query using Tavily API.
    Use this tool to find latest articles, data, news, and search results.
    """
    return execute_tavily_search(query)


@tool
def summarize_url(url: str) -> str:
    """
    Fetch the content of a specific web URL and generate a concise summary.
    Use this tool when you have a specific web link that requires deep reading and analysis.
    """
    return execute_url_summary(url)


@tool
def save_report(report_text: str) -> str:
    """
    Save the synthesized final research report to a local text file.
    Only call this tool when the research report is completely written and final.
    """
    # The default topic will be parsed or passed in graph state
    return execute_save_report(report_text, "research_report")
