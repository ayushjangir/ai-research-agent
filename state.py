"""
LangGraph state definition for the AI Autonomous Research Agent.
"""
from typing import Annotated, List, Optional, Dict, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages



class ResearchState(TypedDict):
    """
    State dictionary maintaining agent context across ReAct reasoning steps.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    research_topic: str
    searched_queries: List[str]
    urls_summarized: List[str]
    report_draft: Optional[str]
    is_approved: Optional[bool]
    human_feedback: Optional[str]
    final_filepath: Optional[str]
    step_history: List[Dict[str, Any]]
