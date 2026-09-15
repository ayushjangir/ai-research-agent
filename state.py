"""
State definition for the AI Research Agent.
Maintains message history, research topic, search memory, and report status.
"""
from typing import Annotated, List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ResearchState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    research_topic: str
    searched_queries: List[str]
    report_draft: Optional[str]
    final_filepath: Optional[str]
