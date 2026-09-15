"""
Streamlit Web UI for Autonomous AI Research Agent.
Simple, clean, and easy to understand.
"""
import os
import glob
from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages import HumanMessage
from agent.graph import build_research_graph
from agent.tools import save_report

# Load environment variables (.env)
load_dotenv()

st.set_page_config(page_title="AI Research Agent", page_icon="🔍", layout="wide")

st.title("🤖 Autonomous AI Research Agent")
st.caption("Powered by Groq (`openai/gpt-oss-20b`) + LangGraph + Tavily Search")

# Sidebar Configuration

with st.sidebar:
    st.header("⚙️ Configuration")

    groq_api_key = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Get your free key at https://console.groq.com/keys"
    )
    tavily_api_key = st.text_input(
        "Tavily API Key",
        value=os.getenv("TAVILY_API_KEY", ""),
        type="password",
        help="Get your free key at https://tavily.com"
    )

    st.info("Model: **`openai/gpt-oss-20b`**")

    # Sync keys to environment
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key
    if tavily_api_key:
        os.environ["TAVILY_API_KEY"] = tavily_api_key

    st.divider()
    st.subheader("🔍 LangSmith Tracing")
    enable_tracing = st.checkbox("Enable Tracing", value=os.getenv("LANGCHAIN_TRACING_V2") == "true")
    if enable_tracing:
        langsmith_key = st.text_input("LangSmith API Key", value=os.getenv("LANGCHAIN_API_KEY", ""), type="password")
        if langsmith_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = langsmith_key
            os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "autonomous-research-agent")
            st.caption("✅ Tracing active: [smith.langchain.com](https://smith.langchain.com)")

    st.divider()
    st.subheader("📁 Saved Reports")
    os.makedirs("reports", exist_ok=True)
    saved_files = glob.glob("reports/*.txt")
    if saved_files:
        for f in saved_files:
            fname = os.path.basename(f)
            with open(f, "r", encoding="utf-8") as file:
                st.download_button(f"📥 {fname}", file.read(), file_name=fname, key=f)
    else:
        st.caption("No reports saved yet.")

# Main Research Interface

topic = st.text_input(
    "Enter a research topic:",
    value="Agentic AI Frameworks 2026",
    placeholder="e.g. Quantum Computing Commercialization"
)

can_run = bool(os.getenv("GROQ_API_KEY") and os.getenv("TAVILY_API_KEY"))
if not can_run:
    st.warning("⚠️ Please provide your Groq API Key and Tavily API Key in the sidebar or .env file.")

if st.button("🚀 Start Research", type="primary", disabled=not can_run):
    st.session_state["report_ready"] = False
    st.session_state["report_text"] = ""

    # Build LangGraph workflow
    graph = build_research_graph()
    config = {"configurable": {"thread_id": "streamlit_session"}}

    initial_state = {
        "messages": [HumanMessage(content=f"Research topic: '{topic}'. Write a detailed, structured Markdown report.")],
        "research_topic": topic,
        "searched_queries": [],
        "report_draft": "",
        "final_filepath": ""
    }

    # Stream agent execution steps live
    with st.status("Agent researching topic...", expanded=True) as status:
        final_state = initial_state
        for event in graph.stream(initial_state, config):
            for node_name, output in event.items():
                if node_name == "agent":
                    msg = output["messages"][-1]
                    calls = getattr(msg, "tool_calls", [])
                    if calls:
                        st.write(f"🤔 **Agent decided to call**: `{', '.join([c['name'] for c in calls])}`")
                    else:
                        st.write("✍️ **Agent finalized response**")
                elif node_name == "tools":
                    st.write(f"⚙️ **Tools executed**. Queries so far: {output.get('searched_queries', [])}")
                final_state = output

        status.update(label="Research Complete! ✅", state="complete", expanded=False)

    # Extract report text
    report_text = final_state.get("report_draft") or ""
    if not report_text and "messages" in final_state:
        report_text = final_state["messages"][-1].content

    st.session_state["report_ready"] = True
    st.session_state["report_text"] = report_text
    st.session_state["current_topic"] = topic

# Report Display & Human-in-the-Loop Confirmation

if st.session_state.get("report_ready") and st.session_state.get("report_text"):
    st.subheader("📄 Research Report")
    st.markdown(st.session_state["report_text"])

    st.divider()
    st.subheader("🧑‍💻 Human-in-the-Loop Confirmation")
    st.caption("Review or edit the draft report before approving and saving to disk:")

    edited_report = st.text_area(
        "Edit Report Content:",
        value=st.session_state["report_text"],
        height=350
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("✅ Confirm & Save Final Report", type="primary"):
            current_topic = st.session_state.get("current_topic", "research_report")
            save_result = save_report.invoke({"report_text": edited_report, "topic": current_topic})
            st.success(save_result)
    with col2:
        st.download_button(
            "📥 Download Report (.md)",
            data=edited_report,
            file_name=f"{st.session_state.get('current_topic', 'report')}.md",
            mime="text/markdown"
        )
