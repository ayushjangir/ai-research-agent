"""
Streamlit Web Application for Autonomous Research Platform.
Clean, professional human-centric design.
"""
import os
import glob
from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from agent.graph import build_research_graph
from agent.tools import execute_save_report

# Load environment variables
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="Autonomous Research Platform",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Clean Professional Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Clean Header Container */
    .main-header {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
    }
    
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #f0f6fc;
        margin-bottom: 6px;
    }
    
    .main-subtitle {
        color: #8b949e;
        font-size: 0.95rem;
        margin-bottom: 0;
    }
    
    /* Clean Badge styling */
    .custom-chip {
        display: inline-block;
        background: #1f6feb22;
        color: #58a6ff;
        border: 1px solid #1f6feb66;
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 0.82rem;
        margin: 3px;
    }
    
    .url-chip {
        background: #8957e522;
        color: #d2a8ff;
        border-color: #8957e566;
    }
    
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 12px 16px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State (Blank default API key fields)
if "groq_api_key" not in st.session_state:
    st.session_state["groq_api_key"] = ""
if "tavily_api_key" not in st.session_state:
    st.session_state["tavily_api_key"] = ""
if "research_results" not in st.session_state:
    st.session_state["research_results"] = None
if "is_researching" not in st.session_state:
    st.session_state["is_researching"] = False
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []


# Sidebar Configuration (Clean, No Robot Image)
with st.sidebar:
    st.title("Settings")
    
    st.subheader("API Credentials")
    groq_key_input = st.text_input(
        "Groq API Key",
        value=st.session_state.get("groq_api_key", ""),
        type="password",
        placeholder="Enter your Groq API Key",
        help="Free key available at https://console.groq.com/keys"
    )
    tavily_key_input = st.text_input(
        "Tavily API Key",
        value=st.session_state.get("tavily_api_key", ""),
        type="password",
        placeholder="Enter your Tavily API Key",
        help="Free search key available at https://tavily.com"
    )
    
    st.session_state["groq_api_key"] = groq_key_input
    st.session_state["tavily_api_key"] = tavily_key_input
    
    model_choice = st.selectbox(
        "Language Model",
        options=["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
        index=0
    )
    
    st.divider()
    
    st.subheader("Saved Reports")
    os.makedirs("reports", exist_ok=True)
    saved_files = glob.glob("reports/*.txt")
    if saved_files:
        selected_file = st.selectbox("Select Report", [os.path.basename(f) for f in saved_files])
        if selected_file:
            filepath = os.path.join("reports", selected_file)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            st.download_button(
                label=f"Download {selected_file}",
                data=content,
                file_name=selected_file,
                mime="text/plain"
            )
            with st.expander("Preview Saved Content", expanded=False):
                st.markdown(content)
    else:
        st.caption("No reports saved yet.")


# Main Application Interface
st.markdown("""
<div class="main-header">
    <div class="main-title">Autonomous Research Platform</div>
    <p class="main-subtitle">
        Stateful multi-tool research system with search memory, automated URL synthesis, and interactive Q&A.
    </p>
</div>
""", unsafe_allow_html=True)


# Topic Suggestions
st.markdown("##### Choose a Preset Topic or Type Your Own")
col_s1, col_s2, col_s3 = st.columns(3)
sample_topic = ""
if col_s1.button("Agentic AI Frameworks 2026"):
    sample_topic = "Agentic AI Frameworks 2026"
if col_s2.button("Quantum Computing Commercialization"):
    sample_topic = "Quantum Computing Commercialization 2026"
if col_s3.button("Solid State Battery Breakthroughs"):
    sample_topic = "Solid State Battery Breakthroughs"

topic_query = st.text_input(
    "Research Topic / Inquiry",
    value=sample_topic if sample_topic else "Agentic AI Frameworks 2026",
    placeholder="e.g. Current developments in Fusion Energy 2026"
)


# Validate API keys before starting
can_run = bool(st.session_state["groq_api_key"] and st.session_state["tavily_api_key"])
if not can_run:
    st.warning("Please enter your Groq API Key and Tavily API Key in the sidebar to start research.")

start_btn = st.button("Start Research", type="primary", disabled=not can_run)


if start_btn and can_run:
    st.session_state["is_researching"] = True
    st.session_state["research_results"] = None
    st.session_state["chat_messages"] = []
    
    with st.status("Conducting research...", expanded=True) as status:
        st.write("Initializing state graph with search memory...")
        
        graph = build_research_graph(
            groq_api_key=st.session_state["groq_api_key"],
            tavily_api_key=st.session_state["tavily_api_key"],
            model_name=model_choice
        )
        
        config = {"configurable": {"thread_id": "streamlit_session"}}
        initial_state = {
            "messages": [HumanMessage(content=f"Conduct thorough research on topic: '{topic_query}'. Synthesize a comprehensive Markdown research report.")],
            "research_topic": topic_query,
            "searched_queries": [],
            "urls_summarized": [],
            "report_draft": "",
            "is_approved": False,
            "human_feedback": "",
            "final_filepath": "",
            "step_history": []
        }
        
        st.write("Searching web and analyzing sources...")
        try:
            final_state = graph.invoke(initial_state, config)
            status.update(label="Research Execution Completed", state="complete", expanded=False)
            st.session_state["research_results"] = final_state
        except Exception as e:
            status.update(label="Execution Interrupted", state="error", expanded=True)
            err_msg = str(e)
            if "413" in err_msg or "TPM" in err_msg or "rate_limit" in err_msg:
                st.error(
                    "Groq Free Tier Rate Limit (TPM) Reached.\n\n"
                    "Please wait 15–20 seconds and click 'Start Research' again. "
                    "Using model `llama-3.1-8b-instant` works best!"
                )
            else:
                st.error(f"Execution Error: {err_msg}")
        finally:
            st.session_state["is_researching"] = False


# Display Results
if st.session_state["research_results"]:
    state = st.session_state["research_results"]
    
    st.divider()
    
    last_msg = state["messages"][-1].content if state.get("messages") else ""
    report_draft_text = state.get("report_draft") or last_msg
    searched = state.get("searched_queries", [])
    urls = state.get("urls_summarized", [])
    steps = state.get("step_history", [])
    word_count = len(report_draft_text.split())
    
    st.markdown("### Research Summary Dashboard")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Web Searches", len(searched))
    with col_m2:
        st.metric("URLs Summarized", len(urls))
    with col_m3:
        st.metric("Execution Steps", len(steps))
    with col_m4:
        st.metric("Report Word Count", f"{word_count} words")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Compact Scrollable Memory Box
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("**Searched Queries:**")
        if searched:
            queries_html = "".join([f'<span class="custom-chip">Search: {q}</span>' for q in searched])
            st.markdown(f'<div style="max-height: 80px; overflow-y: auto; padding: 6px; background: #161b22; border: 1px solid #30363d; border-radius: 8px;">{queries_html}</div>', unsafe_allow_html=True)
        else:
            st.caption("No queries logged.")
            
    with col_c2:
        st.markdown("**Summarized Sources:**")
        if urls:
            urls_html = "".join([f'<span class="custom-chip url-chip">Source: {u}</span>' for u in urls])
            st.markdown(f'<div style="max-height: 80px; overflow-y: auto; padding: 6px; background: #161b22; border: 1px solid #30363d; border-radius: 8px;">{urls_html}</div>', unsafe_allow_html=True)
        else:
            st.caption("No direct URLs scraped.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs Interface
    tab_report, tab_chat, tab_sources, tab_history, tab_edit = st.tabs([
        "Executive Report", 
        "Interactive Q&A",
        "Sources & Findings", 
        "Execution Timeline", 
        "Review & Save"
    ])
    
    with tab_report:
        st.markdown("""
        <div style="background: #161b22; border: 1px solid #30363d; border-left: 4px solid #1f6feb; border-radius: 8px; padding: 16px; margin-bottom: 20px;">
            <h4 style="color: #f0f6fc; margin-bottom: 4px; font-weight: 600;">Executive Research Synthesis</h4>
            <p style="color: #8b949e; font-size: 0.9rem; margin-bottom: 0;">Verified multi-source analytical report</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(report_draft_text)
        
        st.divider()
        col_down1, col_down2 = st.columns([2, 1])
        with col_down2:
            st.download_button(
                label="Download Report (.md)",
                data=report_draft_text,
                file_name=f"{topic_query.lower().replace(' ', '_')}_report.md",
                mime="text/markdown",
                use_container_width=True,
                type="primary"
            )

    # Interactive Q&A Chat
    with tab_chat:
        st.markdown("#### Ask Questions About This Report")
        st.caption("Ask questions, clarify points, or request additional explanations based on the generated report.")
        
        for msg in st.session_state["chat_messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        user_doubt = st.chat_input("Type your question here...")
        if user_doubt:
            st.session_state["chat_messages"].append({"role": "user", "content": user_doubt})
            with st.chat_message("user"):
                st.markdown(user_doubt)
                
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    try:
                        from langchain_groq import ChatGroq
                        from langchain_core.messages import SystemMessage, HumanMessage
                        
                        qa_llm = ChatGroq(
                            model=model_choice,
                            groq_api_key=st.session_state["groq_api_key"],
                            temperature=0.3
                        )
                        qa_prompt = (
                            f"You are a helpful Research Assistant answering user questions about the following report:\n\n"
                            f"=== REPORT ===\n{report_draft_text}\n==============\n\n"
                            "Answer clearly, concisely, and accurately."
                        )
                        
                        qa_messages = [SystemMessage(content=qa_prompt)]
                        for m in st.session_state["chat_messages"]:
                            if m["role"] == "user":
                                qa_messages.append(HumanMessage(content=m["content"]))
                            else:
                                qa_messages.append(AIMessage(content=m["content"]))
                                
                        ans = qa_llm.invoke(qa_messages)
                        answer_text = ans.content
                    except Exception as e:
                        answer_text = f"Error: {e}"
                        
                    st.markdown(answer_text)
                    st.session_state["chat_messages"].append({"role": "assistant", "content": answer_text})

    with tab_sources:
        st.markdown("#### Retrieved Sources & Information")
        for idx, step in enumerate(steps, 1):
            if "query" in step or "url" in step or "result_snippet" in step:
                with st.expander(f"Source {idx}: {step.get('node', 'Tool Output')}", expanded=False):
                    if "query" in step:
                        st.info(f"Query: {step['query']}")
                    if "url" in step:
                        st.markdown(f"URL: [{step['url']}]({step['url']})")
                    if "result_snippet" in step:
                        st.markdown(step["result_snippet"])

    with tab_history:
        st.markdown("#### Execution Steps & Reasoning")
        for idx, step in enumerate(steps, 1):
            with st.expander(f"Step {idx}: {step.get('node', 'Agent Step')}", expanded=(idx == len(steps))):
                if "content" in step and step["content"]:
                    st.markdown(f"**Agent Thinking:**\n{step['content']}")
                if "tool_calls" in step and step["tool_calls"]:
                    st.success(f"**Tools Triggered**: {', '.join(step['tool_calls'])}")

    with tab_edit:
        st.markdown("#### Review & Edit Report")
        st.caption("You can edit the synthesized markdown text before saving to disk.")
        edited_report = st.text_area(
            "Report Content Editor",
            value=report_draft_text,
            height=450
        )
        
        if st.button("Approve & Save Report", type="primary", use_container_width=True):
            res = execute_save_report(edited_report, state.get("research_topic", topic_query))
            if "SUCCESS:" in res:
                filepath = res.split("file: ")[-1].strip()
                st.success(f"Report saved to: `{filepath}`")
            else:
                st.error(res)
