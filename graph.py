"""
LangGraph Workflow Graph with ReAct loop, Search State Memory, and Human-In-The-Loop (HITL) confirmation.
"""
import os
import time
from typing import Dict, Any, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from agent.state import ResearchState
from agent.tools import execute_tavily_search, execute_url_summary, execute_save_report


SYSTEM_PROMPT = """You are an Expert Autonomous AI Research Analyst.
Your task is to conduct thorough, structured research on the user's topic and generate a clear, highly informative, professional Markdown research report.

### Available Tools:
1. `tavily_search(query: str)`: Search web for recent information.
2. `summarize_url(url: str)`: Scrape and summarize full webpage content.
3. `save_report(report_text: str)`: Save the final synthesized report.

### Strict Workflow:
1. **Search**: Execute a targeted web search query with `tavily_search`.
2. **Analyze**: Review the search results carefully. If a key URL needs deep reading, call `summarize_url`.
3. **Synthesize Report**: Draft a clear, comprehensive, user-friendly Markdown report containing:
   - 📌 **Executive Summary**: A concise 3-4 sentence overview.
   - 🔑 **Key Insights & Major Developments**: Bulleted key facts, metrics, and trends.
   - 🏗️ **Deep-Dive Analysis**: Detailed breakdown of architecture, methods, or key concepts.
   - ⚖️ **Challenges & Future Outlook**: What obstacles exist and where the field is heading.
   - 📚 **Sources & References**: List of source URLs with titles.
4. **Finalize**: Present the complete report clearly in your response and call `save_report`.

CRITICAL RULE: Call ONLY ONE tool per turn. NEVER call multiple tools simultaneously.
"""


def create_agent_node(groq_api_key: Optional[str] = None, model_name: str = "llama-3.1-8b-instant"):
    """Creates the agent reasoning node that calls Groq LLM with enhanced report synthesis."""
    
    def agent_node(state: ResearchState) -> Dict[str, Any]:
        key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not key or key == "your_groq_api_key_here":
            error_msg = (
                "ERROR: GROQ_API_KEY is missing. Please enter your free Groq API Key "
                "in the sidebar or set it in your .env file."
            )
            return {"messages": [AIMessage(content=error_msg)]}

        llm = ChatGroq(
            model=model_name,
            groq_api_key=key,
            temperature=0.3,
            max_tokens=3000
        )
        
        # Format current search history context
        searched = state.get("searched_queries", [])
        history_context = ""
        if searched:
            history_context = f"\n[Searched Queries So Far]: {', '.join(searched)}\n"
        
        # Keep original prompt + last 4 messages to preserve rich context
        raw_messages = state["messages"]
        trimmed_messages = []
        
        if len(raw_messages) > 5:
            trimmed_messages = [raw_messages[0]] + raw_messages[-4:]
        else:
            trimmed_messages = list(raw_messages)
            
        # Clean / truncate message content safely to max 1500 chars to fit context window
        cleaned_messages = []
        for msg in trimmed_messages:
            if isinstance(msg, ToolMessage):
                if len(str(msg.content)) > 1500:
                    cleaned_messages.append(ToolMessage(content=str(msg.content)[:1500] + "... [truncated]", tool_call_id=msg.tool_call_id))
                else:
                    cleaned_messages.append(msg)
            elif isinstance(msg, AIMessage):
                if len(str(msg.content)) > 2000:
                    cleaned_msg = AIMessage(content=str(msg.content)[:2000] + "... [truncated]", tool_calls=getattr(msg, "tool_calls", []))
                    cleaned_messages.append(cleaned_msg)
                else:
                    cleaned_messages.append(msg)
            else:
                cleaned_messages.append(msg)

        messages = [SystemMessage(content=SYSTEM_PROMPT + history_context)] + cleaned_messages
        
        # Bind tools to LLM
        from agent.tools import tavily_search, summarize_url, save_report
        tools = [tavily_search, summarize_url, save_report]
        llm_with_tools = llm.bind_tools(tools)
        
        # Invoke LLM with rate limit & tool_use_failed recovery
        response = None
        max_retries = 3
        for attempt in range(max_retries):
            try:
                time.sleep(1.5)
                response = llm_with_tools.invoke(messages)
                break
            except Exception as e:
                err_str = str(e).lower()
                if "tool_use_failed" in err_str or "failed to call a function" in err_str or "400" in err_str:
                    print("[Groq Tool Error] Parallel tool calls detected. Falling back to plain LLM reasoning...")
                    # Fallback to standard LLM without tool binding to complete response
                    response = llm.invoke(messages)
                    break
                elif ("413" in err_str or "rate_limit" in err_str or "tpm" in err_str or "tokens" in err_str) and attempt < max_retries - 1:
                    wait_time = 10 * (attempt + 1)
                    print(f"[Groq Rate Limit] TPM limit reached. Sleeping {wait_time}s (Attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    raise e
        
        # Record step history for UI visualization
        t_calls = getattr(response, "tool_calls", []) or []
        parsed_calls = []
        for tc in t_calls:
            if isinstance(tc, dict):
                parsed_calls.append(tc.get("name", "unknown_tool"))
            else:
                parsed_calls.append(getattr(tc, "name", str(tc)))

        step = {
            "node": "Agent Thinking",
            "content": response.content if response else "",
            "tool_calls": parsed_calls
        }
        
        current_steps = list(state.get("step_history", []))
        current_steps.append(step)
        
        return {
            "messages": [response],
            "step_history": current_steps
        }

    return agent_node


def create_tools_node(tavily_api_key: Optional[str] = None, groq_api_key: Optional[str] = None):
    """Creates the tools execution node handling search, summarization, and saving."""
    
    def tools_node(state: ResearchState) -> Dict[str, Any]:
        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", []) or []
        
        new_messages = []
        searched_queries = list(state.get("searched_queries", []))
        urls_summarized = list(state.get("urls_summarized", []))
        report_draft = state.get("report_draft", "")
        final_filepath = state.get("final_filepath", "")
        steps = list(state.get("step_history", []))

        for tc in tool_calls:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
            args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
            call_id = tc.get("id", "call_id") if isinstance(tc, dict) else getattr(tc, "id", "call_id")

            if name == "tavily_search":
                query = args.get("query", "")
                if query not in searched_queries:
                    searched_queries.append(query)
                
                result = execute_tavily_search(query, api_key=tavily_api_key)
                new_messages.append(ToolMessage(content=result, tool_call_id=call_id))
                steps.append({
                    "node": "Tool: Tavily Web Search",
                    "query": query,
                    "result_snippet": result[:400] + "..." if len(result) > 400 else result
                })

            elif name == "summarize_url":
                url = args.get("url", "")
                if url not in urls_summarized:
                    urls_summarized.append(url)
                
                topic = state.get("research_topic", "")
                result = execute_url_summary(url, topic=topic, groq_api_key=groq_api_key)
                new_messages.append(ToolMessage(content=result, tool_call_id=call_id))
                steps.append({
                    "node": "Tool: Summarize URL",
                    "url": url,
                    "result_snippet": result[:400] + "..." if len(result) > 400 else result
                })

            elif name == "save_report":
                report_text = args.get("report_text", "")
                topic = state.get("research_topic", "research_topic")
                report_draft = report_text
                
                result = execute_save_report(report_text, topic)
                if "SUCCESS:" in result:
                    final_filepath = result.split("file: ")[-1].strip()
                
                new_messages.append(ToolMessage(content=result, tool_call_id=call_id))
                steps.append({
                    "node": "Tool: Save Report",
                    "result": result
                })

        return {
            "messages": new_messages,
            "searched_queries": searched_queries,
            "urls_summarized": urls_summarized,
            "report_draft": report_draft,
            "final_filepath": final_filepath,
            "step_history": steps
        }

    return tools_node


def should_continue(state: ResearchState) -> str:
    """Routing function determining whether to call tools, finish, or request approval."""
    messages = state.get("messages", [])
    if not messages:
        return END
        
    last_message = messages[-1]
    tool_calls = getattr(last_message, "tool_calls", []) or []
    
    if tool_calls:
        return "tools"
    
    return END


def build_research_graph(
    groq_api_key: Optional[str] = None, 
    tavily_api_key: Optional[str] = None,
    model_name: str = "llama-3.3-70b-versatile",
    checkpointer: Any = None
):
    """
    Construct and compile the LangGraph ReAct Agent workflow graph.
    """
    workflow = StateGraph(ResearchState)

    # Add Nodes
    workflow.add_node("agent", create_agent_node(groq_api_key, model_name))
    workflow.add_node("tools", create_tools_node(tavily_api_key, groq_api_key))

    # Add Edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")

    # Use memory checkpointer if provided
    memory = checkpointer if checkpointer is not None else MemorySaver()
    return workflow.compile(checkpointer=memory)
