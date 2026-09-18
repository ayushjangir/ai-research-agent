"""
LangGraph ReAct Research Agent.

Powered by Groq GPT-OSS 120B.
"""

import os

from typing import Dict, Any, Optional

from langchain_core.messages import (
    SystemMessage,
    ToolMessage,
)

from langchain_groq import ChatGroq

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from langgraph.checkpoint.memory import MemorySaver

from agent.state import ResearchState

from agent.tools import (
    tavily_search,
    summarize_url,
    save_report,
)



# TOOLS


TOOLS = [

    tavily_search,

    summarize_url,

    save_report,

]

TOOL_MAP = {

    t.name: t

    for t in TOOLS

}



# CONFIGURATION


MODEL_NAME = os.getenv(

    "GROQ_MODEL",

    "openai/gpt-oss-120b"

)

MAX_TOOL_CHARS = 3500

MAX_OUTPUT_TOKENS = 700

MAX_HISTORY_MESSAGES = 8

MAX_AGENT_STEPS = 6



# SYSTEM PROMPT


SYSTEM_PROMPT = """
You are an AI research agent.

Available tools:
- tavily_search
- summarize_url
- save_report

Rules:
- Use only the provided tools.
- Never call browser.open or any browser.* tool.
- Use tavily_search for web research.
- Keep the final report concise.
- The final report must be under 1800 characters.
- Use short paragraphs and bullet points.
- Do not include unnecessary explanations.
- Call save_report only once.
- Pass the complete report as a valid JSON string.
"""


# LLM


def get_llm(

    groq_api_key: Optional[str] = None

):

    """

    Create Groq LLM.

    """

    key = (

        groq_api_key

        or os.getenv("GROQ_API_KEY")

    )

    return ChatGroq(

        model=MODEL_NAME,

        groq_api_key=key,

        temperature=0.2,

        max_tokens=MAX_OUTPUT_TOKENS,

    )



# AGENT NODE


def agent_node(

    state: ResearchState

) -> Dict[str, Any]:

    """

    Agent reasoning node.

    """

    llm = get_llm().bind_tools(TOOLS)

    searched = state.get(

        "searched_queries",

        []

    )

    memory_context = (

        f"\n[Already Searched]: "

        f"{', '.join(searched)}\n"

        if searched

        else ""

    )

    # Keep recent messages
    recent_messages = (

        state["messages"]

        [-MAX_HISTORY_MESSAGES:]

    )

    messages = [

        SystemMessage(

            content=

            SYSTEM_PROMPT

            + memory_context

        )

    ] + recent_messages

    print(

        "MODEL:", MODEL_NAME

    )

    print(

        "MESSAGES:", len(messages)

    )

    response = llm.invoke(messages)

    return {

        "messages": [

            response

        ]

    }



# TOOLS NODE


def tools_node(

    state: ResearchState

) -> Dict[str, Any]:

    """

    Execute tool calls.

    """

    last_message = state["messages"][-1]

    tool_calls = getattr(

        last_message,

        "tool_calls",

        []

    ) or []

    new_messages = []

    searched_queries = list(

        state.get(

            "searched_queries",

            []

        )

    )

    report_draft = state.get(

        "report_draft",

        ""

    )

    final_filepath = state.get(

        "final_filepath",

        ""

    )

    for tc in tool_calls:

        name = tc.get("name")

        args = tc.get("args", {})

        call_id = tc.get(

            "id",

            "call_id"

        )

        tool_func = TOOL_MAP.get(name)

        if not tool_func:

            continue

        print(

            "Executing tool:",

            name

        )

        result = tool_func.invoke(args)

        # Limit tool output
        tool_content = str(result)[

            :MAX_TOOL_CHARS

        ]

        # Update search memory
        if name == "tavily_search":

            q = args.get(

                "query",

                ""

            )

            if (

                q

                and q not in searched_queries

            ):

                searched_queries.append(q)

        elif name == "save_report":

            report_draft = args.get(

                "report_text",

                ""

            )

            if (

                "SUCCESS: Report saved to"

                in str(result)

            ):

                final_filepath = str(

                    result

                ).split(

                    "to",

                    1

                )[-1].strip()

        new_messages.append(

            ToolMessage(

                content=tool_content,

                tool_call_id=call_id

            )

        )

    return {

        "messages": new_messages,

        "searched_queries": searched_queries,

        "report_draft": report_draft,

        "final_filepath": final_filepath,

    }



# ROUTING


def should_continue(

    state: ResearchState

) -> str:

    """

    Decide whether to execute tools
    or finish.

    """

    last_message = state["messages"][-1]

    if getattr(

        last_message,

        "tool_calls",

        None

    ):

        return "tools"

    return END



# BUILD GRAPH


def build_research_graph(

    checkpointer: Any = None

):

    """

    Construct and compile graph.

    """

    workflow = StateGraph(

        ResearchState

    )

    workflow.add_node(

        "agent",

        agent_node

    )

    workflow.add_node(

        "tools",

        tools_node

    )

    workflow.add_edge(

        START,

        "agent"

    )

    workflow.add_conditional_edges(

        "agent",

        should_continue,

        {

            "tools": "tools",

            END: END

        }

    )

    workflow.add_edge(

        "tools",

        "agent"

    )

    memory = (

        checkpointer

        if checkpointer is not None

        else MemorySaver()

    )

    return workflow.compile(

        checkpointer=memory

    )