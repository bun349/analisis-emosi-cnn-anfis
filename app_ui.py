# -*- coding: utf-8 -*-
"""
Streamlit Web Interface: Emotion Intensity Analysis in Instagram Images: A Dual-Branch CNN and ANFIS Approach
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from PIL import Image

from app_utama import analyze_single_image, KELAS_EMOSI

# ==========================================
# PAGE CONFIGURATION & CSS INJECTION
# ==========================================
st.set_page_config(
    page_title="AffectAudit — XAI Affective Classification System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
/* ── FONT IMPORT ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── GLOBAL RESET & BASE ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── MAIN BACKGROUND ── */
.stApp {
    background-color: #0b0f19;
    color: #e2e8f0;
}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background-color: #0f1420;
    border-right: 1px solid #1e2a40;
}

/* ── MAIN HEADER (Kembali ke Rata Kiri Estetik) ── */
.hero-block {
    background: linear-gradient(135deg, #0f1e3a 0%, #0b1628 50%, #12142a 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 40px 48px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-block::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(56,182,255,0.07) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-label {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 3px;
    color: #38b6ff;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.hero-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 32px;
    font-weight: 600;
    color: #f0f6ff;
    line-height: 1.25;
    margin-bottom: 12px;
}
.hero-subtitle {
    font-size: 14px;
    color: #7a90b0;
    font-weight: 300;
    max-width: 780px;
    line-height: 1.6;
}

/* ── BADGES ── */
.badge-row { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 24px; }
.badge {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 1.5px;
    padding: 5px 12px;
    border-radius: 4px;
    text-transform: uppercase;
}
.badge-blue  { background: rgba(56,182,255,0.1);  color: #38b6ff;  border: 1px solid rgba(56,182,255,0.3); }
.badge-teal  { background: rgba(52,211,153,0.1);  color: #34d399;  border: 1px solid rgba(52,211,153,0.3); }
.badge-amber { background: rgba(251,191,36,0.1);  color: #fbbf24;  border: 1px solid rgba(251,191,36,0.3); }

/* ── PANEL SECTION ── */
.panel {
    background: #0f1628;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 20px;
}
.panel-title {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 2.5px;
    color: #38b6ff;
    text-transform: uppercase;
    margin-bottom: 16px;
}

/* ── EMOTION RESULT CARD ── */
.result-card {
    background: linear-gradient(135deg, #0d1f3c 0%, #111827 100%);
    border: 1px solid #2563eb;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
}
.result-label {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: #64748b;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.result-value {
    font-size: 28px;
    font-weight: 600;
    color: #38b6ff;
}

/* ── FUZZIFICATION CARD ── */
.fuzz-card {
    border-radius: 8px;
    padding: 14px 16px;
    text-align: center;
    border: 1.5px solid #1e2d4a;
    background: #0b1224;
    transition: all 0.25s ease;
}
.fuzz-card.active {
    background: rgba(56,182,255,0.08);
    border-color: #38b6ff;
}
.fuzz-card-label {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.fuzz-card-label.active { color: #38b6ff; }
.fuzz-card-label.inactive { color: #3d5070; }
.fuzz-card-value { font-size: 22px; font-weight: 600; }
.fuzz-card-value.active { color: #e2e8f0; }
.fuzz-card-value.inactive { color: #3d5070; }

/* ── FEATURE DIVIDER ── */
.feature-header {
    display: flex;
    align-items: baseline;
    gap: 12px;
    margin: 16px 0 4px;
}
.feature-name {
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    font-weight: 600;
    color: #cbd5e1;
}
.feature-raw {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: #475569;
}
.feature-caption {
    font-size: 12px;
    color: #475569;
    margin-bottom: 10px;
    font-style: italic;
}
.hr-dim { border: none; border-top: 1px solid #1e2d4a; margin: 20px 0; }

/* ── CUSTOM PROGRESS BAR ── */
.progress-track {
    background: #1a2540;
    border-radius: 4px;
    height: 6px;
    margin-top: 8px;
    overflow: hidden;
}
.progress-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #1d4ed8, #38b6ff);
    transition: width 0.6s ease;
}

/* ── INFO BOXES ── */
.info-box {
    background: rgba(56,182,255,0.06);
    border-left: 3px solid #38b6ff;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin-bottom: 20px;
    font-size: 13px;
    color: #94a3b8;
    line-height: 1.6;
}
.warning-box {
    background: rgba(251,191,36,0.06);
    border-left: 3px solid #fbbf24;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin-bottom: 20px;
    font-size: 13px;
    color: #94a3b8;
    line-height: 1.6;
}
.success-box {
    background: rgba(52,211,153,0.06);
    border-left: 3px solid #34d399;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin-bottom: 20px;
    font-size: 13px;
    color: #e2e8f0;
    line-height: 1.6;
}

/* ── BUTTONS ── */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 10px 24px !important;
    letter-spacing: 0.3px;
    transition: all 0.2s !important;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(37,99,235,0.35) !important;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    background: #0f1628 !important;
    border: 1.5px dashed #1e3a5f !important;
    border-radius: 10px !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0b1224;
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #475569 !important;
    border-radius: 6px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    padding: 8px 20px !important;
}
.stTabs [aria-selected="true"] {
    background: #1e2d4a !important;
    color: #38b6ff !important;
}

/* ── HIDE DEFAULT STREAMLIT ELEMENTS ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if 'analysis_result' not in st.session_state:
    st.session_state['analysis_result'] = None
if 'last_filename' not in st.session_state:
    st.session_state['last_filename'] = ""

TEMP_DIR = "temp_uploads"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

def save_uploaded_file(uploaded_file):
    file_path = os.path.join(TEMP_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# ==========================================
# HERO HEADER (Ditambahkan Sub-label Lab)
# ==========================================
st.markdown("""
<div class="hero-block">
    <div class="hero-title">Emotion Intensity Analysis<br>Based on Dual-Branch CNN & ANFIS Architecture</div>
    <div class="hero-subtitle">
        An AI decision transparency audit platform for digital image emotion analysis.
        Integrates convolutional neural networks with adaptive Neuro-Fuzzy inference
        to produce quantitatively explainable predictions.
    </div>
    <div class="badge-row">
        <span class="badge badge-blue">Dual-Branch CNN</span>
        <span class="badge badge-teal">6-Input ANFIS</span>
        <span class="badge badge-amber">Explainable AI (XAI)</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# MAIN ROW: UPLOAD + RESULTS
# ==========================================
col_left, col_right = st.columns([1, 1.6], gap="large")

with col_left:
    st.markdown("""
    <div style="margin-bottom: 16px;">
        <div class="panel-title">Test Image Input</div>
        <p style="font-size:13px; color:#64748b;">Upload a subject image in JPG or PNG format. The system will extract facial features while simultaneously reading the environmental context.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        label="Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        if st.session_state['last_filename'] != uploaded_file.name:
            st.session_state['analysis_result'] = None
            st.session_state['last_filename'] = uploaded_file.name

        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True, caption=f"📁 {uploaded_file.name}")

        st.markdown('<div style="margin-top:16px;"></div>', unsafe_allow_html=True)
        if st.button("▶  Run Inference & XAI Audit", type="primary"):
            with st.spinner('Extracting facial features and visual context...'):
                image_path = save_uploaded_file(uploaded_file)
                try:
                    st.session_state['analysis_result'] = analyze_single_image(image_path)
                except Exception as e:
                    st.error(f"System error: {e}")
    else:
        st.markdown("""
        <div style="text-align:center; padding: 40px 20px; color:#2d4060;">
            <div style="font-size:40px; margin-bottom:12px;">🖼️</div>
            <div style="font-size:13px; font-family: 'Space Mono', monospace; letter-spacing:1px;">
                NO IMAGE UPLOADED
            </div>
        </div>
        """, unsafe_allow_html=True)
        
# ── RIGHT COLUMN ──
with col_right:
    if st.session_state['analysis_result'] is not None:
        analysis_result = st.session_state['analysis_result']

        if isinstance(analysis_result, str):
            st.markdown(f'<div class="warning-box">⚠️ {analysis_result}</div>', unsafe_allow_html=True)
        else:
            result_dict = analysis_result["scores"]
            inputs    = analysis_result["inputs"]
            mu        = analysis_result["mu"]
            sigma     = analysis_result["sigma"]

            dominant_emotion = max(result_dict, key=result_dict.get)
            dominant_score   = result_dict[dominant_emotion]

            # ── FINAL DECISION PANEL (Gunakan .replace('\n', '') untuk cegah parser leak) ──
            final_decision_html = f"""
            <div class="panel">
                <div class="panel-title">Final Classification Decision</div>
                <div style="display:flex; align-items:center; gap:24px; flex-wrap:wrap;">
                    <div class="result-card" style="flex:1; min-width:160px;">
                        <div class="result-label">Predicted Class</div>
                        <div class="result-value">{dominant_emotion}</div>
                    </div>
                    <div class="result-card" style="flex:1; min-width:160px;">
                        <div class="result-label">Confidence Score</div>
                        <div class="result-value">{dominant_score*100:.1f}%</div>
                    </div>
                </div>
            </div>
            """.replace("\n", "")
            st.markdown(final_decision_html, unsafe_allow_html=True)

            # ── INTENSITY DISTRIBUTION (Metode String Sanitasi untuk Amankan Grid Layout) ──
            dist_html = '<div class="panel"><div class="panel-title">Emotion Intensity Distribution (Defuzzification)</div>'

            for kelas, skor in sorted(result_dict.items(), key=lambda x: x[1], reverse=True):
                persen = skor * 100
                is_top = kelas == dominant_emotion
                bar_color = "linear-gradient(90deg, #1d4ed8, #38b6ff)" if is_top else "linear-gradient(90deg, #1e2d4a, #2d4060)"
                label_color = "#e2e8f0" if is_top else "#94a3b8"
                score_color = "#38b6ff" if is_top else "#64748b"
                
                # HTML dirapatkan tanpa baris baru agar tidak dieksekusi sebagai struktur teks markdown terpisah
                dist_html += f'<div style="margin-bottom:12px;"><div style="display:flex; justify-content:space-between; margin-bottom:4px;"><span style="font-size:13px; font-weight:{"600" if is_top else "400"}; color:{label_color};">{kelas}</span><span style="font-family:\'Space Mono\',monospace; font-size:12px; color:{score_color};">{persen:.1f}%</span></div><div class="progress-track"><div class="progress-fill" style="width:{persen:.1f}%; background:{bar_color};"></div></div></div>'
            
            dist_html += "</div>" 
            st.markdown(dist_html, unsafe_allow_html=True)

            # ── EXPLAINABILITY CONCLUSION ──
            if dominant_score > 0.7:
                st.markdown(f"""
                <div class="success-box">
                    <strong>✅ High Confidence (>{dominant_score*100:.0f}%)</strong><br>
                    The Neuro-Fuzzy module detected a strong correlation between micro features (facial expression) and macro representations
                    (environmental brightness/saturation) that consistently supports the <strong>{dominant_emotion}</strong> class.
                    The degree of visual ambiguity is classified as low.
                </div>
                """.replace("\n", ""), unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="warning-box">
                    <strong>⚠️ Marginal Confidence (&lt;70%)</strong><br>
                    The system detected visual ambiguity — a potential contradiction between facial expression
                    and environmental context (e.g., a positive face in a dim color palette). ANFIS acts as an adaptive
                    mediator by proportionally weighing both channels.
                    Predicted class: <strong>{dominant_emotion}</strong>.
                </div>
                """.replace("\n", ""), unsafe_allow_html=True)

    else:
        # ── PLACEHOLDER BEFORE ANALYSIS ──
        st.markdown("""
        <div class="panel" style="height:100%; min-height:380px; display:flex; flex-direction:column; justify-content:center; align-items:center; text-align:center;">
            <div style="font-size:48px; margin-bottom:16px; opacity:0.3;">🔬</div>
            <div class="panel-title" style="text-align:center;">Waiting for Image Input</div>
            <p style="font-size:13px; color:#2d4060; max-width:320px; line-height:1.7;">
                Upload an image via the left panel, then run inference to view the complete AI decision audit results.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# BOTTOM SECTION: XAI VISUAL AUDIT
# ==========================================
if st.session_state['analysis_result'] is not None and not isinstance(st.session_state['analysis_result'], str):
    analysis_result = st.session_state['analysis_result']
    inputs = analysis_result["inputs"]
    mu     = analysis_result["mu"]
    sigma  = analysis_result["sigma"]

    st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Space Mono',monospace; font-size:11px; letter-spacing:3px; color:#38b6ff; text-transform:uppercase; margin-bottom:4px;">
        Explainable AI Visual Audit
    </div>
    <div style="font-size:22px; font-weight:600; color:#e2e8f0; margin-bottom:4px;">Algorithmic Reasoning Trace</div>
    <div style="font-size:13px; color:#64748b; margin-bottom:24px;">
        The following visualization transparently describes how each visual input feature is transformed into linguistic degrees of membership
        through Gaussian membership functions embedded in the ANFIS fuzzification module.
    </div>
    """, unsafe_allow_html=True)

    input_names = [
        "Face: Happy", "Face: Angry", "Face: Neutral", "Face: Sad",
        "Environment: Brightness", "Environment: Saturation"
    ]
    feature_descriptions = {
        "Face: Happy":          "Probability of detecting relaxed and smiling facial muscle patterns from the face CNN channel.",
        "Face: Angry":          "Probability of detecting tension or aggressive wrinkles in the subject's facial region.",
        "Face: Neutral":        "Probability of no dominant expression — indicating a flat or unreadable expression.",
        "Face: Sad":            "Probability of downward facial muscle pulling or lethargic/loss expressions.",
        "Environment: Brightness":"(Visual Psychology) High luminance correlates with positive valence; low luminance with gloominess.",
        "Environment: Saturation": "(Visual Psychology) High saturation reflects high affective energy; low saturation indicates lethargy."
    }
    ling_terms = ["Low", "Medium", "High"]

    tab1, tab2 = st.tabs(["  📊  Degree of Membership Fuzzification  ", "  🕸️  Feature Profile Radar  "])

    with tab1:
        st.markdown("""
        <div class="info-box">
            <strong>Interpretation Guide:</strong> Each block represents one input feature dimension.
            <em>Raw value</em> is the measured value from the network (0–1). Three boxes indicate the degree of membership
            to the linguistic categories Low / Medium / High via a Gaussian function.
            The box with the <span style="color:#38b6ff">blue border</span> is the strongest linguistic conclusion.
        </div>
        """, unsafe_allow_html=True)

        for row in range(0, 6, 2):
            cols_row = st.columns(2, gap="large")
            for ci, i in enumerate([row, row + 1]):
                if i >= 6:
                    break
                with cols_row[ci]:
                    st.markdown(f"""
                    <div style="margin-bottom:6px;">
                        <span class="feature-name" style="font-family:'DM Sans',sans-serif; font-size:14px; font-weight:600; color:#cbd5e1;">{input_names[i]}</span>
                        <span style="font-family:'Space Mono',monospace; font-size:11px; color:#38b6ff; margin-left:10px;">
                            value = {inputs[i]:.3f}
                        </span>
                    </div>
                    <div style="font-size:12px; color:#475569; font-style:italic; margin-bottom:12px;">{feature_descriptions[input_names[i]]}</div>
                    """, unsafe_allow_html=True)

                    persentase = []
                    for j in range(3):
                        deg = np.exp(-0.5 * ((inputs[i] - mu[i, j]) / sigma[i, j])**2)
                        persentase.append(deg * 100)
                    nilai_max = max(persentase)

                    card_cols = st.columns(3)
                    for j in range(3):
                        is_active = persentase[j] == nilai_max
                        state_cls = "active" if is_active else "inactive"
                        card_cols[j].markdown(f"""
                        <div class="fuzz-card {'active' if is_active else ''}">
                            <div class="fuzz-card-label {state_cls}">{ling_terms[j]}</div>
                            <div class="fuzz-card-value {state_cls}">{persentase[j]:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown('<div class="hr-dim"></div>', unsafe_allow_html=True)

    with tab2:
        st.markdown("""
        <div class="info-box">
            The radar chart illustrates the multidimensional profile of input features extracted by the system.
            A wider area indicates the dominance of that feature in influencing the final classification decision.
        </div>
        """, unsafe_allow_html=True)

        r_vals     = list(inputs) + [inputs[0]]
        theta_vals = input_names + [input_names[0]]

        fig_radar = go.Figure(data=go.Scatterpolar(
            r=r_vals,
            theta=theta_vals,
            fill='toself',
            fillcolor='rgba(37,99,235,0.15)',
            marker=dict(color='#38b6ff', size=6),
            line=dict(color='#38b6ff', width=2)
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                angularaxis=dict(
                    tickfont=dict(family='DM Sans', size=12, color='#94a3b8'),
                    linecolor='#1e2d4a',
                    gridcolor='#1e2d4a'
                ),
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickfont=dict(family='Space Mono', size=9, color='#475569'),
                    gridcolor='#1e2d4a',
                    linecolor='#1e2d4a'
                )
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            margin=dict(t=40, b=40, l=60, r=60),
            font=dict(family='DM Sans', color='#94a3b8')
        )

        st.plotly_chart(fig_radar, width='stretch')
