"""
Dashboard page for the Student Study Coach application.
This page allows users to input their academic and lifestyle data,
get a Pass/Fail prediction, and see personalized recommendations.
"""

import streamlit as st
import pandas as pd
from src.inference import StudentPredictor
from src.ui_components import UIBuilder

@st.cache_resource
def load_predictor():
    """
    Cache the StudentPredictor instance to avoid reloading models on every interaction.
    """
    return StudentPredictor()

# Initialize predictor and UI components
predictor = load_predictor()
ui = UIBuilder()
ui.load_css()

# Check if models are ready before proceeding
if not predictor.is_ready():
    st.error("Models not found. Run `python train_model.py` first.")
    st.stop()

# Header Section
ui.render_hero_banner(
    "🎓 Student Performance Predictor",
    "Enter your academic details below and get an instant AI-powered Pass / Fail prediction."
)

# Input Form
with st.form("student_form"):
    # ── Academic Scores Section ──
    ui.render_section_label("📊 Academic Scores")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        math_score = st.slider("📐 Math Score", 0, 100, 65)
    with sc2:
        reading_score = st.slider("📖 Reading Score", 0, 100, 65)
    with sc3:
        writing_score = st.slider("✍️ Writing Score", 0, 100, 65)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Study Habits Section ──
    ui.render_section_label("📚 Study Habits")
    sh1, sh2, sh3 = st.columns(3)
    with sh1:
        daily_study = st.slider("⏱️ Daily Study Hours", 0.0, 12.0, 3.0, step=0.5)
    with sh2:
        attendance = st.slider("🏫 Attendance Rate", 0.0, 1.0, 0.85, step=0.01,
            help="0.0 = 0%,  1.0 = 100%")
    with sh3:
        parental_education = st.slider("👨‍👩‍🎓 Parental Education Level", 1, 7, 3,
            help="1 = No formal education  →  7 = Postgraduate")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Wellbeing Section ──
    ui.render_section_label("🧘 Wellbeing")
    wb1, wb2, wb3 = st.columns(3)
    with wb1:
        sleep_hours = st.slider("😴 Sleep Hours / Night", 3.0, 12.0, 7.0, step=0.5)
    with wb2:
        stress_level = st.slider("😰 Stress Level", 1, 10, 5,
            help="1 = very low  →  10 = extremely high")
    with wb3:
        motivation = st.slider("🔥 Motivation Score", 0, 100, 60)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🔍 Predict My Outcome", type="primary")

# ── Results Processing ──
if submitted:
    # Prepare input data for prediction
    input_data = {
        "parental_education_level": parental_education,
        "daily_study_hours": daily_study,
        "attendance_rate": attendance,
        "sleep_hours": sleep_hours,
        "stress_level": stress_level,
        "motivation_score": motivation,
        "math_score": math_score,
        "reading_score": reading_score,
        "writing_score": writing_score,
    }
    df = pd.DataFrame([input_data])
    
    # Run prediction, probability estimate, and clustering
    prob, pred, cluster = predictor.predict_bundle(df)

    # Store results in session state for persistence and use in chat
    st.session_state.student_data = {
        **input_data,
        "prediction": "Pass" if pred[0] == 1 else "Fail",
        "probability": float(prob[0]),
        "cluster": int(cluster[0]),
    }
    st.session_state.prediction_done = True
    st.session_state.chat_history = []
    
    # Reset SessionMemory so a new prediction starts a fresh coaching session
    st.session_state.pop("session_memory", None)
    st.session_state.show_chat_button = True

    st.markdown("---")

    # Display prediction visual card
    ui.render_prediction_card(float(prob[0]), int(pred[0]), int(cluster[0]))

    # Display score breakdown and tips side-by-side
    left, right = st.columns([1, 1], gap="large")

    with left:
        ui.render_section_label("📊 Score Breakdown")
        ui.render_score_bars(math_score, reading_score, writing_score)

        # Additional metric chips
        attendance_pct = f"{attendance*100:.0f}%"
        st.markdown(f"""
        <div style="margin-top:18px;">
            <div class="section-label" style="margin-bottom:10px;">📋 Other Metrics</div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
                <div class="stat-chip">
                    <span class="sc-icon">⏱️</span>
                    <div><div class="sc-val">{daily_study}h</div><div class="sc-lbl">Daily Study</div></div>
                </div>
                <div class="stat-chip">
                    <span class="sc-icon">🏫</span>
                    <div><div class="sc-val">{attendance_pct}</div><div class="sc-lbl">Attendance</div></div>
                </div>
                <div class="stat-chip">
                    <span class="sc-icon">😴</span>
                    <div><div class="sc-val">{sleep_hours}h</div><div class="sc-lbl">Sleep / Night</div></div>
                </div>
                <div class="stat-chip">
                    <span class="sc-icon">🔥</span>
                    <div><div class="sc-val">{motivation}</div><div class="sc-lbl">Motivation</div></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        ui.render_section_label("💡 Actionable Tips")
        # Get heuristic recommendations based on input values
        recs = predictor.get_student_recommendations(pd.Series(input_data))
        ui.render_tips(recs)

# ── Call to Action for Chat Interface ──
if st.session_state.get("show_chat_button"):
    st.markdown("""
    <div class="cta-box">
        <div class="cta-title">🤖 Ready to level up?</div>
        <div class="cta-sub">Chat with your AI Study Coach for a personalised 7-day plan and live resources.</div>
    </div>
    """, unsafe_allow_html=True)
    col_btn = st.columns([1, 2, 1])[1]
    with col_btn:
        if st.button("💬 Open Study Coach Chat", type="primary", use_container_width=True):
            st.switch_page("pages/chat_interface.py")

