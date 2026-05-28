import streamlit as st
import pandas as pd
import joblib
from src.data_loader import load_data
from src.preprocessing import preprocess
from src.engine import build_training_frame, FEATURE_COLUMNS

st.set_page_config(
    page_title="Serie A Match Predictor",
    page_icon="⚽",
    layout="wide"
)

st.markdown(
    """
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
    <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }
        .sub-title {
            color: #6b7280;
            margin-bottom: 1.2rem;
        }
        .panel {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            padding: 1.1rem 1.2rem;
            box-shadow: 0 6px 24px rgba(0,0,0,0.04);
        }
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 14px;
            margin-top: 14px;
            margin-bottom: 14px;
        }
        .kpi-card {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 16px 18px;
            box-shadow: 0 4px 18px rgba(0,0,0,0.03);
        }
        .kpi-label {
            display: flex;
            align-items: center;
            gap: 10px;
            color: #374151;
            font-size: 0.95rem;
            font-weight: 600;
            margin-bottom: 10px;
        }
        .kpi-value {
            font-size: 1.55rem;
            font-weight: 800;
            color: #111827;
        }
        .feature-box {
            background: #0f172a;
            color: white;
            border-radius: 16px;
            padding: 18px;
        }
        .feature-box h4 {
            margin-top: 0;
            margin-bottom: 12px;
        }
        .muted {
            color: #9ca3af;
            font-size: 0.92rem;
        }
        .badge-line {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }
        .badge-pill {
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(255,255,255,0.06);
            color: #e5e7eb;
            padding: 6px 10px;
            border-radius: 999px;
            font-size: 0.85rem;
        }
        .stButton > button {
            border-radius: 12px;
            padding: 0.6rem 1rem;
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">Serie A Match Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Predizione 1X2 con Elo, forma recente e statistiche storiche.</div>',
    unsafe_allow_html=True
)

@st.cache_resource
def load_artifacts():
    model = joblib.load("models/model.pkl")
    raw = load_data("data/raw")
    raw = preprocess(raw)
    _, tracker = build_training_frame(raw)
    return model, tracker

model, tracker = load_artifacts()
teams = sorted(tracker.states.keys())

col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### Match setup")
    c1, c2 = st.columns(2)
    with c1:
        home = st.selectbox("Home Team", teams)
    with c2:
        away = st.selectbox("Away Team", teams)

    if home == away:
        st.warning("Le squadre devono essere diverse.")
        st.stop()

    features = tracker.make_features(home, away)

    st.markdown(
        """
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label"><i class="bi bi-house-door-fill"></i> Home Elo</div>
                <div class="kpi-value">{home_elo:.1f}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label"><i class="bi bi-arrow-left-right"></i> Elo Diff</div>
                <div class="kpi-value">{elo_diff:.1f}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label"><i class="bi bi-flag-fill"></i> Away Elo</div>
                <div class="kpi-value">{away_elo:.1f}</div>
            </div>
        </div>
        """.format(
            home_elo=tracker.get_state(home).elo,
            elo_diff=features["elo_diff"],
            away_elo=tracker.get_state(away).elo
        ),
        unsafe_allow_html=True
    )

    st.markdown("#### Feature preview")
    st.dataframe(pd.DataFrame([features]), use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown(
        """
        <div class="feature-box">
            <h4><i class="bi bi-bar-chart-line-fill"></i> Model inputs</h4>
            <div class="muted">Le feature sono costruite solo con partite precedenti, così la predizione resta coerente.</div>
            <div class="badge-line">
                <span class="badge-pill">Elo difference</span>
                <span class="badge-pill">Recent form</span>
                <span class="badge-pill">Attack / defense</span>
                <span class="badge-pill">Shots</span>
                <span class="badge-pill">Corners</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Prediction")

    if st.button("Predict", use_container_width=True):
        X = pd.DataFrame([features], columns=FEATURE_COLUMNS)

        proba = model.predict_proba(X)[0]
        classes = model.classes_
        pred = classes[proba.argmax()]

        p_home = 0.0
        p_draw = 0.0
        p_away = 0.0
        for c, p in zip(classes, proba):
            if c == "H":
                p_home = p
            elif c == "D":
                p_draw = p
            elif c == "A":
                p_away = p

        st.markdown("#### Probability split")
        b1, b2, b3 = st.columns(3)
        b1.metric("Home", f"{p_home * 100:.1f}%")
        b2.metric("Draw", f"{p_draw * 100:.1f}%")
        b3.metric("Away", f"{p_away * 100:.1f}%")

        if pred == "H":
            st.success(f"{home} vincente")
        elif pred == "A":
            st.error(f"{away} vincente")
        else:
            st.info("Pareggio")

        st.markdown("#### Raw probabilities")
        st.write(
            {
                "H": round(p_home, 4),
                "D": round(p_draw, 4),
                "A": round(p_away, 4),
            }
        )