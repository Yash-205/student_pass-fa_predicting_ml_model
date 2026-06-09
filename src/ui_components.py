"""
UI Components module for the Student Study Coach.
This module defines the UIBuilder class, which handles the injection of custom CSS
and the rendering of premium Streamlit components using HTML/CSS.
"""

import streamlit as st
from typing import List, Optional, Any

class UIBuilder:
    """
    A utility class to build and render custom UI elements in Streamlit.
    Uses raw HTML and inline CSS to create a premium, cohesive look.
    """
    def __init__(self) -> None:

        """Initializes the UI builder with a standard color palette."""

        self.palette = ["#6366f1", "#a78bfa", "#10b981", "#f59e0b", "#ef4444", "#06b6d4"]

    def load_css(self) -> None:
        """
        Injects a large block of custom CSS into the Streamlit app.
        Covers layout, typography, custom cards, buttons, and animations.
        """
        st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        * { font-family: 'Inter', sans-serif !important; }

        /* ── Layout ── */
        .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1100px; }
        h1 { letter-spacing: -0.03em; font-weight: 800; font-size: 2rem !important; }
        h2, h3 { letter-spacing: -0.02em; font-weight: 700; }
        hr { opacity: 0.08; margin: 1.2rem 0; }

        /* ── Hero banner ── */
        .hero-banner {
            background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #1e1b4b 100%);
            border: 1px solid rgba(99,102,241,0.3);
            border-radius: 20px;
            padding: 28px 32px;
            margin-bottom: 24px;
            position: relative;
            overflow: hidden;
        }
        .hero-banner::before {
            content: '';
            position: absolute; top: -40px; right: -40px;
            width: 180px; height: 180px;
            background: radial-gradient(circle, rgba(139,92,246,0.25) 0%, transparent 70%);
            border-radius: 50%;
        }
        .hero-title { font-size: 1.7rem; font-weight: 800; color: #e0e7ff; letter-spacing: -0.03em; }
        .hero-sub { font-size: 0.95rem; color: rgba(224,231,255,0.65); margin-top: 6px; }

        /* ── Section label ── */
        .section-label {
            font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em;
            text-transform: uppercase; color: #a78bfa;
            margin-bottom: 10px; display: flex; align-items: center; gap: 6px;
        }
        .section-label::after {
            content: ''; flex: 1; height: 1px;
            background: linear-gradient(to right, rgba(167,139,250,0.3), transparent);
        }

        /* ── Form card ── */
        .form-section {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 16px;
            padding: 20px 18px 14px;
            margin-bottom: 6px;
        }

        /* ── Prediction card ── */
        .pred-card {
            border-radius: 20px;
            padding: 26px 30px;
            margin-bottom: 20px;
            position: relative;
            overflow: hidden;
        }
        .pred-card-pass {
            background: linear-gradient(135deg, rgba(16,185,129,0.18) 0%, rgba(5,150,105,0.08) 100%);
            border: 1.5px solid rgba(16,185,129,0.45);
        }
        .pred-card-fail {
            background: linear-gradient(135deg, rgba(239,68,68,0.18) 0%, rgba(220,38,38,0.08) 100%);
            border: 1.5px solid rgba(239,68,68,0.45);
        }
        .pred-badge {
            display: inline-block;
            font-size: 0.75rem; font-weight: 700; letter-spacing: 0.08em;
            text-transform: uppercase; padding: 4px 12px;
            border-radius: 99px; margin-bottom: 10px;
        }
        .pred-badge-pass { background: rgba(16,185,129,0.2); color: #34d399; }
        .pred-badge-fail { background: rgba(239,68,68,0.2); color: #f87171; }
        .pred-headline { font-size: 2.4rem; font-weight: 800; letter-spacing: -0.04em; line-height: 1; }
        .pred-headline-pass { color: #10b981; }
        .pred-headline-fail { color: #ef4444; }
        .pred-sub { font-size: 0.88rem; color: rgba(255,255,255,0.55); margin-top: 6px; }
        .prob-row { display: flex; align-items: center; gap: 12px; margin-top: 14px; }
        .prob-label { font-size: 0.82rem; color: rgba(255,255,255,0.6); white-space: nowrap; }
        .prob-bar-wrap { flex: 1; background: rgba(255,255,255,0.08); border-radius: 99px; height: 8px; }
        .prob-bar-inner { height: 8px; border-radius: 99px; transition: width 0.5s ease; }
        .prob-pct { font-size: 1rem; font-weight: 700; white-space: nowrap; }

        /* ── Stat mini-card ── */
        .stat-row { display: flex; gap: 10px; margin-top: 16px; flex-wrap: wrap; }
        .stat-chip {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.09);
            border-radius: 10px; padding: 8px 14px;
            display: flex; align-items: center; gap: 8px;
        }
        .stat-chip .sc-icon { font-size: 1rem; }
        .stat-chip .sc-val { font-size: 0.95rem; font-weight: 600; }
        .stat-chip .sc-lbl { font-size: 0.72rem; color: rgba(255,255,255,0.5); }

        /* ── Score bar ── */
        .score-bar-wrap { margin-bottom: 12px; }
        .score-bar-label {
            display: flex; justify-content: space-between;
            font-size: 0.85rem; margin-bottom: 5px;
        }
        .score-bar-label .sb-name { font-weight: 600; }
        .score-bar-label .sb-val { font-weight: 700; }
        .score-bar-bg { background: rgba(255,255,255,0.07); border-radius: 99px; height: 10px; }
        .score-bar-fill { height: 10px; border-radius: 99px; }

        /* ── Tip card ── */
        .tip-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 12px;
            padding: 12px 14px;
            margin-bottom: 8px;
            display: flex;
            align-items: flex-start;
            gap: 10px;
            font-size: 0.88rem;
            line-height: 1.5;
            transition: border-color 0.2s;
        }
        .tip-icon { font-size: 1.1rem; flex-shrink: 0; margin-top: 1px; }

        /* ── Chat bubbles ── */
        .chat-bubble-wrap { display: flex; margin-bottom: 14px; gap: 10px; }
        .chat-bubble-wrap.user { flex-direction: row-reverse; }
        .chat-avatar {
            width: 34px; height: 34px; border-radius: 50%; flex-shrink: 0;
            display: flex; align-items: center; justify-content: center;
            font-size: 1rem; font-weight: 700;
        }
        .avatar-user { background: linear-gradient(135deg, #6366f1, #8b5cf6); }
        .avatar-ai { background: linear-gradient(135deg, #0891b2, #06b6d4); }
        .chat-bubble {
            max-width: 78%; padding: 12px 16px; border-radius: 16px;
            font-size: 0.9rem; line-height: 1.55;
        }
        .bubble-user {
            background: linear-gradient(135deg, rgba(99,102,241,0.3), rgba(139,92,246,0.2));
            border: 1px solid rgba(99,102,241,0.35);
            border-top-right-radius: 4px;
        }
        .bubble-ai {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.09);
            border-top-left-radius: 4px;
        }
        .bubble-time { font-size: 0.7rem; color: rgba(255,255,255,0.35); margin-top: 5px; }

        /* ── Chat header ── */
        .chat-profile-card {
            background: linear-gradient(135deg, rgba(99,102,241,0.1), rgba(6,182,212,0.06));
            border: 1px solid rgba(99,102,241,0.2);
            border-radius: 16px;
            padding: 18px 22px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 16px;
        }
        .chat-profile-avatar {
            width: 52px; height: 52px; border-radius: 50%;
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            display: flex; align-items: center; justify-content: center;
            font-size: 1.4rem; flex-shrink: 0;
        }
        .chat-profile-name { font-size: 1rem; font-weight: 700; color: #e0e7ff; }
        .chat-profile-sub { font-size: 0.82rem; color: rgba(255,255,255,0.5); margin-top: 2px; }
        .chat-badge {
            margin-left: auto;
            padding: 6px 16px; border-radius: 99px;
            font-size: 0.8rem; font-weight: 700; letter-spacing: 0.04em;
        }
        .chat-badge-pass { background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.3); }
        .chat-badge-fail { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }

        /* ── CTA section ── */
        .cta-box {
            background: linear-gradient(135deg, rgba(99,102,241,0.1), rgba(139,92,246,0.06));
            border: 1px solid rgba(99,102,241,0.25);
            border-radius: 18px;
            padding: 24px 28px;
            text-align: center;
            margin-top: 8px;
        }
        .cta-title { font-size: 1.1rem; font-weight: 700; color: #e0e7ff; margin-bottom: 6px; }
        .cta-sub { font-size: 0.88rem; color: rgba(255,255,255,0.5); margin-bottom: 16px; }

        /* ── Buttons ── */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.6rem 2.2rem !important;
            font-weight: 700 !important;
            font-size: 0.95rem !important;
            letter-spacing: 0.02em !important;
            box-shadow: 0 4px 20px rgba(99,102,241,0.35) !important;
            transition: opacity 0.2s, transform 0.1s !important;
        }
        .stButton > button[kind="primary"]:hover {
            opacity: 0.92 !important;
            transform: translateY(-1px) !important;
        }
        .stButton > button[kind="secondary"] {
            border-radius: 10px !important;
            font-weight: 600 !important;
        }

        /* ── Slider ── */
        .stSlider [data-baseweb="slider"] div[role="slider"] {
            background: #6366f1 !important;
        }

        /* ── Streamlit metric ── */
        [data-testid="stMetric"] {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 14px 18px !important;
        }
        [data-testid="stMetricValue"] { font-weight: 700 !important; }

        /* ── Form submit button ── */
        [data-testid="stFormSubmitButton"] > button {
            background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.65rem 2.5rem !important;
            font-weight: 700 !important;
            font-size: 1rem !important;
            letter-spacing: 0.02em !important;
            box-shadow: 0 4px 20px rgba(99,102,241,0.35) !important;
            width: 100%;
        }

        /* ── Spinner ── */
        .stSpinner > div { border-top-color: #6366f1 !important; }

        /* ── Info/warning boxes ── */
        .stAlert { border-radius: 12px !important; }
        </style>
        """, unsafe_allow_html=True)

    def render_hero_banner(self, title: str, subtitle: str) -> None:
        """Renders a stylish gradient banner with a title and subtitle."""
        st.markdown(f"""
        <div class="hero-banner">
            <div class="hero-title">{title}</div>
            <div class="hero-sub">{subtitle}</div>
        </div>
        """, unsafe_allow_html=True)

    def render_section_label(self, text: str) -> None:
        """Renders a labeled divider to separate different form sections."""
        st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)

    def render_prediction_card(self, proba: float, pred: int, cluster: int) -> None:
        """
        Renders the main prediction result card.
        Shows Pass/Fail status, probability bar, and student cluster.
        """
        pct = proba * 100
        bar_color = "#10b981" if pred == 1 else "#ef4444"
        label = "Predicted to Pass" if pred == 1 else "Predicted to Fail"
        headline = "PASS ✓" if pred == 1 else "FAIL ✗"
        card_cls = "pred-card-pass" if pred == 1 else "pred-card-fail"
        badge_cls = "pred-badge-pass" if pred == 1 else "pred-badge-fail"
        headline_cls = "pred-headline-pass" if pred == 1 else "pred-headline-fail"
        sub = "Great work! Keep maintaining your study habits." if pred == 1 else "Don't worry — your coach has a personalised plan ready for you."
        group_names = ["High Achievers", "Average Performers", "Struggling Students", "At-Risk Group"]
        group = group_names[cluster % len(group_names)]

        st.markdown(f"""
        <div class="pred-card {card_cls}">
            <div class="pred-badge {badge_cls}">{label}</div>
            <div class="pred-headline {headline_cls}">{headline}</div>
            <div class="pred-sub">{sub}</div>
            <div class="prob-row">
                <div class="prob-label">Pass probability</div>
                <div class="prob-bar-wrap">
                    <div class="prob-bar-inner" style="width:{pct:.1f}%; background:{bar_color};"></div>
                </div>
                <div class="prob-pct" style="color:{bar_color};">{pct:.1f}%</div>
            </div>
            <div style="margin-top:10px; font-size:0.8rem; color:rgba(255,255,255,0.4);">
                Performance cluster: {group}
            </div>
        </div>
        """, unsafe_allow_html=True)

    def render_score_bars(self, math: int, reading: int, writing: int) -> None:
        """Renders horizontal bar charts for academic scores with color coding for low scores."""
        scores = [
            ("📐 Math", math, "#6366f1"),
            ("📖 Reading", reading, "#06b6d4"),
            ("✍️ Writing", writing, "#a78bfa"),
        ]
        for name, val, color in scores:
            danger = val < 50
            bar_color = "#ef4444" if danger else color
            st.markdown(f"""
            <div class="score-bar-wrap">
                <div class="score-bar-label">
                    <span class="sb-name">{name}</span>
                    <span class="sb-val" style="color:{bar_color};">{val}/100</span>
                </div>
                <div class="score-bar-bg">
                    <div class="score-bar-fill" style="width:{val}%; background:{bar_color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.75rem; color:rgba(255,255,255,0.35); margin-top:6px;">
            ⚠️ Scores below 50 shown in red
        </div>""", unsafe_allow_html=True)

    def render_tips(self, recs: list) -> None:
        """Renders a list of actionable study tips as decorative cards."""
        icons = ["💡", "📚", "⏰", "🎯", "🧘", "💪", "🌟", "📝"]
        for i, rec in enumerate(recs):
            icon = icons[i % len(icons)]
            st.markdown(f"""
            <div class="tip-card">
                <span class="tip-icon">{icon}</span>
                <span>{rec}</span>
            </div>
            """, unsafe_allow_html=True)

    def render_chat_profile(self, prediction: str, prob: float, cluster: int) -> None:
        """Renders a header for the chat interface showing the student's status."""
        
        badge_cls = "chat-badge-pass" if prediction == "Pass" else "chat-badge-fail"
        badge_text = f"✓ Predicted Pass ({prob*100:.0f}%)" if prediction == "Pass" else f"✗ Predicted Fail ({prob*100:.0f}%)"
        st.markdown(f"""
        <div class="chat-profile-card">
            <div class="chat-profile-avatar">🎓</div>
            <div>
                <div class="chat-profile-name">Your AI Study Coach</div>
                <div class="chat-profile-sub">Personalised guidance based on your prediction</div>
            </div>
            <div class="chat-badge {badge_cls}">{badge_text}</div>
        </div>
        """, unsafe_allow_html=True)

    def render_metric_card(self, value: Any, label: str) -> str:
        """Returns HTML for a small metric card (used in dynamic displays)."""

        return f"""<div class="metric-card"><div class="val">{value}</div><div class="lbl">{label}</div></div>"""

