"""
Chat Interface page for the Student Study Coach application.
This page provides an interactive AI agent (StudyCoachAgent) that uses the
student's prediction results to provide personalized coaching, study plans,
and academic support.
"""

import streamlit as st
from agents.study_coach_agent import StudyCoachAgent
from memory.session_memory import SessionMemory
from src.ui_components import UIBuilder

# Unique key for the session memory storage
_SESSION_KEY = "active"

@st.cache_resource
def load_agent():
    """
    Cache the StudyCoachAgent instance to avoid re-initializing the LLM and tools.
    """
    return StudyCoachAgent()

# Initialize agent and UI components
agent = load_agent()
ui = UIBuilder()
ui.load_css()

# ── Navigation Guard ──
# Users must run a prediction on the dashboard before they can access the chat
if "student_data" not in st.session_state or not st.session_state.get("prediction_done"):
    st.markdown("""
    <div style="text-align:center; padding:60px 20px;">
        <div style="font-size:3rem; margin-bottom:12px;">🎓</div>
        <div style="font-size:1.2rem; font-weight:700; color:#e0e7ff; margin-bottom:8px;">No prediction found</div>
        <div style="font-size:0.9rem; color:rgba(255,255,255,0.5); margin-bottom:24px;">
            Run a prediction on the Dashboard first to start your coaching session.
        </div>
    </div>
    """, unsafe_allow_html=True)
    c = st.columns([1, 2, 1])[1]
    with c:
        if st.button("← Go to Dashboard", type="primary", use_container_width=True):
            st.switch_page("pages/dashboard.py")
    st.stop()

# Retrieve student data from session state
student_data = st.session_state.student_data
prediction = student_data.get("prediction", "Unknown")
prob = student_data.get("probability", 0.0)
cluster = student_data.get("cluster", 0)

# ── SessionMemory Management ──
# Single source of truth for the raw conversation history (to be sent to LLM)
if "session_memory" not in st.session_state:
    st.session_state.session_memory = SessionMemory()

session_mem: SessionMemory = st.session_state.session_memory

# ── UI Header and Quick Stats ──
ui.render_chat_profile(prediction, prob, cluster)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Math", f"{student_data.get('math_score', '—')}/100")
m2.metric("Reading", f"{student_data.get('reading_score', '—')}/100")
m3.metric("Writing", f"{student_data.get('writing_score', '—')}/100")
m4.metric("Attendance", f"{student_data.get('attendance_rate', 0)*100:.0f}%")

st.markdown("""
<div style="font-size:0.82rem; color:rgba(255,255,255,0.4); margin:10px 0 20px;">
    💬 Ask me about study tips, a weekly plan, subject help, or motivation.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Chat Display Logic ──
# chat_history stores message dictionaries for the UI display
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    role = msg["role"]
    content = msg["content"]
    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(content)

# ── Chat Input Processing ──
if prompt := st.chat_input("Ask your study coach anything about your studies..."):
    # Log user message
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Generate agent response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            reply, updated_history = agent.chat(
                user_message=prompt,
                student_data=student_data,
                session_history=session_mem.get_history(_SESSION_KEY),
            )
        st.markdown(reply)

    # Log assistant response
    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    
    # Sync the updated history (including system prompts/tool calls) back to memory
    session_mem.set_history(_SESSION_KEY, updated_history)

# ── Reset Conversation Button ──
if st.session_state.chat_history:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([3, 1, 3])
    with btn_col:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            session_mem.clear(_SESSION_KEY)
            st.rerun()

