"""
LangGraph ReAct Research Agent with State Memory and Tool Calling.
Powered by Groq LLM (openai/gpt-oss-120b).
"""
import os
from typing import Dict, Any, Optional
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import ResearchState
from agent.tools import tavily_search, summarize_url, save_report

# List of tools available to the agent
TOOLS = [tavily_search, summarize_url, save_report]
TOOL_MAP = {t.name: t for t in TOOLS}

SYSTEM_PROMPT = """You are an Autonomous AI Research Analyst.
Your goal is to conduct structured research on the given topic and draft a clear Markdown research report.

### Workflow:
1. Search the web using `tavily_search(query)` for relevant sources.
2. If a specific URL needs deeper reading, call `summarize_url(url)`.
3. Synthesize your findings into a comprehensive report with:
   - **Executive Summary**
   - **Key Insights & Metrics**
   - **Detailed Analysis**
   - **Sources & References**
4. Call `save_report(report_text, topic)` with your final report draft.

Rule: Use only one tool at a time when needed. Once the report is drafted and saved, provide your final response.
"""


def get_llm(groq_api_key: Optional[str] = None):
    """Instantiate ChatGroq model with openai/gpt-oss-20b."""
    key = groq_api_key or os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    return ChatGroq(
        model=model,
        groq_api_key=key,
        temperature=0.2,
    )


def agent_node(state: ResearchState) -> Dict[str, Any]:
    """Agent reasoning node: decides what to do next based on conversation & search memory."""
    llm = get_llm().bind_tools(TOOLS)

    # State Memory: remind the agent what it has already searched so it avoids duplicate queries
    searched = state.get("searched_queries", [])
    memory_context = f"\n[Already Searched]: {', '.join(searched)}\n" if searched else ""

    messages = [SystemMessage(content=SYSTEM_PROMPT + memory_context)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


def tools_node(state: ResearchState) -> Dict[str, Any]:
    """Tools execution node: executes tool calls and updates search memory & report draft."""
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", []) or []

    new_messages = []
    searched_queries = list(state.get("searched_queries", []))
    report_draft = state.get("report_draft", "")
    final_filepath = state.get("final_filepath", "")

    for tc in tool_calls:
        name = tc.get("name")
        args = tc.get("args", {})
        call_id = tc.get("id", "call_id")

        tool_func = TOOL_MAP.get(name)
        if tool_func:
            result = tool_func.invoke(args)

            # Update memory state
            if name == "tavily_search":
                q = args.get("query", "")
                if q and q not in searched_queries:
                    searched_queries.append(q)
            elif name == "save_report":
                report_draft = args.get("report_text", "")
                if "SUCCESS: Report saved to" in str(result):
                    final_filepath = str(result).split("to")[-1].strip()

            new_messages.append(ToolMessage(content=str(result), tool_call_id=call_id))

    return {
        "messages": new_messages,
        "searched_queries": searched_queries,
        "report_draft": report_draft,
        "final_filepath": final_filepath,
    }


def should_continue(state: ResearchState) -> str:
    """Check if the agent requested tools or is finished."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


def build_research_graph(checkpointer: Any = None):
    """Construct and compile the LangGraph ReAct agent workflow."""
    workflow = StateGraph(ResearchState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")

    memory = checkpointer if checkpointer is not None else MemorySaver()
    return workflow.compile(checkpointer=memory)
