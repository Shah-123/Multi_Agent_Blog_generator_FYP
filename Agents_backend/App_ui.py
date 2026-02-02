import streamlit as st
import os
import time
from datetime import date
from dotenv import load_dotenv

# --- CRITICAL IMPORT FOR FIX ---
from langgraph.checkpoint.memory import MemorySaver 
# -------------------------------

from Graph.state import State, Plan
from main import build_graph, refine_plan_with_llm
from validators import TopicValidator

st.set_page_config(page_title="AI Content Factory", page_icon="🚀", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #FF4B4B; color: white; }
    .status-box { padding: 10px; border-radius: 5px; border: 1px solid #ddd; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE SETUP (THE FIX IS HERE)
# ============================================================================
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "1"

# 1. Initialize Memory in Session State so it survives re-runs
if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()

if "plan" not in st.session_state:
    st.session_state.plan = None
if "final_content" not in st.session_state:
    st.session_state.final_content = None
if "phase" not in st.session_state:
    st.session_state.phase = "input"

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    st.info("System: Multi-Agent HITL")
    st.success("✅ Router")
    st.success("✅ Researcher")
    st.success("✅ Orchestrator")
    st.success("✅ Writer (Parallel)")
    st.success("✅ Fact Checker")

# ============================================================================
# MAIN UI
# ============================================================================
st.title("🚀 AI Content Factory")
st.markdown("### Enterprise-Grade Blog Generation with Human-in-the-Loop")

# PHASE 1: INPUT
if st.session_state.phase == "input":
    topic = st.text_input("Enter a Blog Topic:", placeholder="e.g. The Future of AI in Healthcare 2026")
    if st.button("🚀 Start Workflow"):
        if topic:
            val = TopicValidator().validate(topic)
            if not val["valid"]:
                st.error(f"Topic Rejected: {val['reason']}")
            else:
                st.session_state.topic = topic
                st.session_state.phase = "planning"
                st.rerun()

# PHASE 2: RESEARCH & PLANNING
elif st.session_state.phase == "planning":
    st.info(f"🔎 Analyzing Topic: **{st.session_state.topic}**")
    status = st.status("🤖 Agents at work...", expanded=True)
    try:
        # PASS THE SAVED MEMORY
        app = build_graph(memory=st.session_state.memory)
        thread_config = {"configurable": {"thread_id": st.session_state.thread_id}}
        
        initial_state = {"topic": st.session_state.topic, "as_of": date.today().isoformat(), "sections": []}

        for event in app.stream(initial_state, thread_config, stream_mode="values"):
            if "plan" in event and event["plan"]:
                st.session_state.plan = event["plan"]
                status.write("✅ Plan Generated")
        
        status.update(label="✋ Planning Complete! Waiting for Approval.", state="complete", expanded=False)
        st.session_state.phase = "review"
        st.rerun()
    except Exception as e:
        st.error(f"Error during planning: {str(e)}")

# PHASE 3: REVIEW
elif st.session_state.phase == "review":
    st.warning("✋ Human Input Required: Review the Plan")
    plan = st.session_state.plan
    
    with st.container(border=True):
        st.subheader(f"📝 Draft Plan: {plan.blog_title}")
        st.caption(f"Audience: {plan.audience} | Tone: {plan.tone}")
        for task in plan.tasks:
            st.markdown(f"**{task.id+1}. {task.title}**")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve Plan"):
            st.session_state.phase = "writing"
            st.rerun()
    with col2:
        feedback = st.text_input("Request changes:", placeholder="e.g. Add a section on Ethics...")
        if st.button("🔄 Update Plan"):
            if feedback:
                with st.spinner("Updating..."):
                    new_plan = refine_plan_with_llm(plan, feedback)
                    # Update State using SAME memory
                    app = build_graph(memory=st.session_state.memory)
                    thread_config = {"configurable": {"thread_id": st.session_state.thread_id}}
                    app.update_state(thread_config, {"plan": new_plan})
                    st.session_state.plan = new_plan
                    st.success("Updated!")
                    time.sleep(1)
                    st.rerun()

# PHASE 4: WRITING
elif st.session_state.phase == "writing":
    st.info("✍️ Agents are writing...")
    status = st.status("🚀 Writing in progress...", expanded=True)
    try:
        # PASS THE SAVED MEMORY (Critical Step)
        app = build_graph(memory=st.session_state.memory)
        thread_config = {"configurable": {"thread_id": st.session_state.thread_id}}
        
        # Resume with None
        for event in app.stream(None, thread_config, stream_mode="values", recursion_limit=100):
            if "final" in event:
                st.session_state.final_content = event
                status.write("✅ Blog Post Assembled")
            if "linkedin_post" in event:
                status.write("✅ Social Media Created")

        status.update(label="✨ Workflow Complete!", state="complete", expanded=False)
        st.session_state.phase = "done"
        st.rerun()
    except Exception as e:
        st.error(f"Error during writing: {str(e)}")

# PHASE 5: DONE
elif st.session_state.phase == "done":
    result = st.session_state.final_content
    st.balloons()
    st.success("🎉 Success!")
    
    tab1, tab2, tab3 = st.tabs(["📄 Blog", "📱 Social", "🕵️ Audit"])
    with tab1:
        st.markdown(result["final"])
        st.download_button("Download Markdown", result["final"], file_name="blog.md")
    with tab2:
        st.text_area("LinkedIn", result.get("linkedin_post", ""), height=200)
        st.text_area("YouTube", result.get("youtube_script", ""), height=200)
    with tab3:
        st.text(result.get("fact_check_report", ""))
        st.metric("Quality Score", f"{result.get('quality_evaluation', {}).get('final_score', 0)}/10")

    if st.button("Start New"):
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.rerun()