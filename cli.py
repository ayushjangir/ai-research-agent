"""
Terminal CLI Runner for AI Research Agent.
- Test tools: python cli.py --test-tools
- Run interactive agent: python cli.py
"""
import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from agent.tools import tavily_search, summarize_url, save_report
from agent.graph import build_research_graph

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()


def test_tools():
    """Verify all 3 tools working directly."""
    print("\n--- Testing Individual Tools ---")

    print("\n[1] Testing Tool 1: tavily_search...")
    res1 = tavily_search.invoke({"query": "AI research developments 2026"})
    print(res1[:300] + "...\n")

    print("[2] Testing Tool 2: summarize_url...")
    res2 = summarize_url.invoke({"url": "https://en.wikipedia.org/wiki/Artificial_intelligence"})
    print(res2[:300] + "...\n")

    print("[3] Testing Tool 3: save_report...")
    res3 = save_report.invoke({"report_text": "# Test Report\nAgent is functional.", "topic": "test_report"})
    print(res3 + "\n")

    print("✅ All 3 tools tested successfully!\n")


def run_agent(topic: str):
    """Run full LangGraph agent with state memory and terminal human-in-the-loop confirmation."""
    print(f"\n🚀 Launching Research Agent for topic: '{topic}'\n")

    graph = build_research_graph()
    config = {"configurable": {"thread_id": "cli_session"}}

    initial_state = {
        "messages": [HumanMessage(content=f"Conduct thorough research on '{topic}'. Write a detailed Markdown report.")],
        "research_topic": topic,
        "searched_queries": [],
        "report_draft": "",
        "final_filepath": ""
    }

    state = graph.invoke(initial_state, config)

    print("\n" + "=" * 60)
    print("📋 DRAFT RESEARCH REPORT")
    print("=" * 60)
    report_text = state.get("report_draft") or state["messages"][-1].content
    print(report_text)
    print("=" * 60)

    # Human-in-the-Loop Confirmation
    print("\n🧑‍💻 [Human-in-the-Loop] Confirm saving report?")
    choice = input("Do you approve saving this report to file? (y/n): ").strip().lower()

    if choice in ["y", "yes"]:
        res = save_report.invoke({"report_text": report_text, "topic": topic})
        print(f"\n✅ {res}")
    else:
        print("\n❌ Report save cancelled by user.")


if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY") or not os.getenv("TAVILY_API_KEY"):
        print("⚠️ Missing GROQ_API_KEY or TAVILY_API_KEY in .env file.")
        print("Please configure your .env file with your API keys.")
        sys.exit(1)

    if len(sys.argv) > 1 and sys.argv[1] == "--test-tools":
        test_tools()
        sys.exit(0)

    user_topic = input("Enter research topic (or press Enter for 'Agentic AI 2026'): ").strip()
    if not user_topic:
        user_topic = "Agentic AI 2026"

    run_agent(user_topic)
