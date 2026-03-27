from __future__ import annotations

import streamlit as st
import pandas as pd
import joblib
import os
import gzip
import numpy as np
import time
import io
import base64
from datetime import datetime, timedelta
from src.preprocessing import (
    preprocess_input,
    get_preprocessing_summary,
    VALID_AIRLINES,
    VALID_SOURCES,
    VALID_DESTINATIONS,
)
import google.generativeai as genai
from streamlit_mic_recorder import mic_recorder
from gtts import gTTS
from dotenv import load_dotenv
import speech_recognition as sr

# ─── Environment ──────────────────────────────────────────────────────────────
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    model_gemini = genai.GenerativeModel("gemini-1.5-flash")
else:
    model_gemini = None

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SkyPrice AI | Flight Fare Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Premium CSS ──────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp {
    background: radial-gradient(ellipse at top right, #0f2027, #203a43, #2c5364);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(15, 32, 39, 0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px;
    padding: 8px 20px;
    color: #94a3b8;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
    color: white !important;
}

/* Input overrides */
.stSelectbox label, .stDateInput label, .stTimeInput label, .stRadio label {
    color: #cbd5e1 !important;
    font-size: 0.85rem;
    font-weight: 500;
    letter-spacing: 0.03em;
}

/* Primary button */
.stButton > button[kind="primary"],
.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 3.2em;
    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
    color: white;
    font-weight: 700;
    font-size: 1rem;
    border: none;
    transition: all 0.25s ease;
    box-shadow: 0 4px 15px rgba(59,130,246,0.35);
    letter-spacing: 0.02em;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(59,130,246,0.5);
}

/* Prediction card */
.prediction-card {
    background: linear-gradient(135deg, rgba(30,41,59,0.85), rgba(15,23,42,0.95));
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(59,130,246,0.3);
    border-radius: 20px;
    padding: 36px 28px;
    text-align: center;
    margin-top: 24px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.04);
    animation: fadeSlideUp 0.5s ease;
}
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}

.fare-label {
    color: #64748b;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 6px;
}
.fare-amount {
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.fare-range {
    color: #475569;
    font-size: 0.82rem;
    margin-top: 6px;
}

/* Insight cards */
.insight-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 18px 20px;
    transition: border-color 0.2s;
}
.insight-card:hover {
    border-color: rgba(59,130,246,0.4);
}
.insight-icon { font-size: 1.4rem; margin-bottom: 4px; }
.insight-title { color: #64748b; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; }
.insight-value { color: #f1f5f9; font-size: 1.3rem; font-weight: 700; margin-top: 2px; }
.insight-delta { font-size: 0.78rem; margin-top: 2px; }

/* Tips banner */
.tips-banner {
    background: rgba(16,185,129,0.08);
    border: 1px solid rgba(16,185,129,0.25);
    border-radius: 12px;
    padding: 14px 18px;
    margin-top: 12px;
    font-size: 0.9rem;
    color: #6ee7b7;
}

/* Route summary pill */
.route-pill {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    background: rgba(59,130,246,0.12);
    border: 1px solid rgba(59,130,246,0.25);
    border-radius: 30px;
    padding: 6px 18px;
    font-size: 0.9rem;
    color: #93c5fd;
    font-weight: 600;
    margin-bottom: 20px;
}

/* Metric overrides */
div[data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    color: #38bdf8 !important;
    font-weight: 700 !important;
}
div[data-testid="stMetricDelta"] > div {
    font-size: 0.78rem !important;
}

/* Chat message */
div[data-testid="stChatMessage"] {
    background: rgba(30,41,59,0.6) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
}

/* Divider */
hr { border-color: rgba(255,255,255,0.07) !important; }

/* Warning / info overrides */
div[data-testid="stAlert"] {
    border-radius: 12px !important;
}

/* Progress bar color */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #3b82f6, #818cf8) !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ─── Model Loader ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading prediction engine…")
def load_model():
    MODEL_PATH = "models/flight_price_model.pkl"
    GZ_PATH = "models/flight_price_model.pkl.gz"
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    elif os.path.exists(GZ_PATH):
        with gzip.open(GZ_PATH, "rb") as f:
            return joblib.load(f)
    return None


# ─── AI & Audio Helpers ───────────────────────────────────────────────────────
def build_system_prompt(context: dict | None = None) -> str:
    base = (
        "You are SkyPriceBot, an expert AI travel assistant for the SkyPrice AI platform. "
        "You help users understand flight fares, travel tips, best booking windows, baggage rules, "
        "and airport information for Indian domestic routes. "
        "Always be concise, friendly, and factual. "
        "If asked to predict fares, remind users to use the Price Predictor tab for accurate ML-based estimates."
    )
    if context:
        base += (
            f"\n\nThe user is currently viewing a prediction for: "
            f"Route={context.get('route')}, Airline={context.get('airline')}, "
            f"Class={context.get('travel_class')}, Date={context.get('journey_date')}, "
            f"Dep={context.get('dep_time')}, Predicted Fare=₹{context.get('predicted_fare', 'N/A')}. "
            "Use this context to give better answers when the user asks follow-up questions."
        )
    return base


def get_ai_response(prompt: str, context: dict | None = None) -> str:
    if not model_gemini:
        return "⚠️ Gemini API Key is missing. Please add `GOOGLE_API_KEY` to your `.env` file."
    try:
        full_prompt = build_system_prompt(context) + f"\n\nUser: {prompt}"
        response = model_gemini.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"❌ Error contacting Gemini: {e}"


def text_to_speech(text: str) -> bytes | None:
    try:
        tts = gTTS(text=text, lang="en", slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        return fp.getvalue()
    except Exception as e:
        st.error(f"TTS Error: {e}")
        return None


def autoplay_audio(audio_bytes: bytes):
    b64 = base64.b64encode(audio_bytes).decode()
    st.markdown(
        f'<audio autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>',
        unsafe_allow_html=True,
    )


def transcribe_audio(audio_bytes: bytes) -> str | None:
    """Convert raw audio bytes to text using Google Speech Recognition."""
    try:
        recognizer = sr.Recognizer()
        audio_file = io.BytesIO(audio_bytes)
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        st.warning("🎙️ Couldn't understand the audio. Please try again.")
    except sr.RequestError as e:
        st.error(f"Speech recognition service error: {e}")
    except Exception as e:
        st.error(f"Audio processing error: {e}")
    return None


def price_sentiment(price: float) -> tuple[str, str]:
    """Return (label, color) based on predicted price."""
    if price < 3000:
        return "💚 Great Deal", "#10b981"
    elif price < 6000:
        return "💛 Average", "#f59e0b"
    elif price < 10000:
        return "🟠 Above Average", "#f97316"
    else:
        return "🔴 Premium Fare", "#ef4444"


# ─── Load Model ───────────────────────────────────────────────────────────────
model = load_model()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='text-align:center; padding: 10px 0;'>"
        "<span style='font-size:3rem;'>✈️</span>"
        "<h2 style='margin:8px 0 0; color:#f1f5f9; font-weight:800; letter-spacing:-0.5px;'>SkyPrice AI</h2>"
        "<p style='color:#64748b; font-size:0.8rem; margin:4px 0 0;'>Domestic Flight Fare Intelligence</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.subheader("✈️ Booking Preferences")
    travel_class = st.radio(
        "Travel Class",
        ["Economy", "Business / Premium"],
        index=0,
        help="Business class predictions are optimised for Jet Airways Business.",
    )
    if travel_class == "Business / Premium":
        st.info("💼 Jet Airways Business class gives the best business fare estimates.")

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown(
        "<div style='font-size:0.83rem; color:#64748b; line-height:1.7;'>"
        "🤖 <b style='color:#94a3b8;'>Algorithm:</b> RandomForest Regressor<br>"
        "📈 <b style='color:#94a3b8;'>R² Score:</b> ~85.8%<br>"
        "🗂️ <b style='color:#94a3b8;'>Training Data:</b> Kaggle Flight Dataset<br>"
        "🛫 <b style='color:#94a3b8;'>Routes:</b> 5 origins × 6 destinations"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.caption("v2.1.0 | Built with Streamlit + Google Gemini")

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='margin-bottom:2px; font-size:2.4rem; font-weight:800; "
    "background:linear-gradient(90deg,#38bdf8,#818cf8); "
    "-webkit-background-clip:text; -webkit-text-fill-color:transparent;'>"
    "✈️ SkyPrice AI Hub</h1>"
    "<p style='color:#64748b; font-size:1rem; margin-top:0;'>"
    "Real-time fare prediction · AI travel assistant · Voice commands</p>",
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(["🔮 Price Predictor", "🤖 AI Travel Assistant"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PRICE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if model is None:
        st.error(
            "🚨 **Prediction engine not found.**  \n"
            "Place `flight_price_model.pkl` (or `.pkl.gz`) inside the `models/` directory and restart."
        )
    else:
        # ── Input Section ──────────────────────────────────────────────────
        with st.container():
            col1, col2, col3 = st.columns(3)

            with col1:
                airline = st.selectbox(
                    "🛫 Airline",
                    VALID_AIRLINES,
                    help="Select your preferred carrier",
                )
            with col2:
                source = st.selectbox("📍 Origin City", VALID_SOURCES)
            with col3:
                destination = st.selectbox("🏁 Destination City", VALID_DESTINATIONS)

        st.markdown("<br>", unsafe_allow_html=True)

        col4, col5, col6 = st.columns([1.4, 1, 1])
        with col4:
            journey_date = st.date_input(
                "📅 Departure Date",
                min_value=datetime.now().date(),
                value=(datetime.now() + timedelta(days=7)).date(),
            )
        with col5:
            dep_time = st.time_input(
                "🕐 Departure Time",
                datetime.now().replace(hour=10, minute=0).time(),
            )
        with col6:
            # Days until departure — contextual info
            days_away = (journey_date - datetime.now().date()).days
            if days_away < 0:
                st.warning("⚠️ Date is in the past")
            elif days_away == 0:
                st.info("📌 Departing today")
            elif days_away <= 7:
                st.warning(f"⚡ Departing in {days_away} day{'s' if days_away>1 else ''}")
            else:
                st.success(f"📅 {days_away} days until departure")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Predict Button ─────────────────────────────────────────────────
        predict_col, _ = st.columns([1, 2])
        with predict_col:
            run_prediction = st.button("🔮 Calculate Fare", type="primary", use_container_width=True)

        if run_prediction:
            # Validation
            if source == destination:
                st.warning("⚠️ Origin and destination cannot be the same city.")
            else:
                flight_class_val = 1 if travel_class == "Business / Premium" else 0

                with st.status("🔮 Analysing flight patterns…", expanded=True) as status:
                    st.write("🌐 Fetching historical price data…")
                    time.sleep(0.4)
                    st.write("⚙️ Engineering features…")

                    try:
                        features = preprocess_input(
                            airline, source, destination,
                            journey_date, dep_time, flight_class_val
                        )
                        summary = get_preprocessing_summary(
                            airline, source, destination,
                            journey_date, dep_time, flight_class_val
                        )
                    except ValueError as e:
                        st.error(f"Input error: {e}")
                        st.stop()

                    time.sleep(0.35)
                    st.write("🤖 Running RandomForest Regressor…")
                    prediction = float(model.predict(features)[0])
                    prediction = max(500.0, prediction)  # sanity floor
                    time.sleep(0.25)
                    status.update(label="✅ Analysis complete!", state="complete", expanded=False)

                # Store prediction context for AI assistant
                st.session_state["last_prediction"] = {
                    **summary,
                    "predicted_fare": f"{prediction:,.0f}",
                }

                # ── Prediction Card ────────────────────────────────────────
                low = prediction * 0.90
                high = prediction * 1.12
                sentiment_label, sentiment_color = price_sentiment(prediction)

                st.markdown(
                    f"""
<div class="prediction-card">
    <div class="fare-label">Estimated Fare · {travel_class}</div>
    <div class="fare-amount">₹ {prediction:,.0f}</div>
    <div class="fare-range">Likely range: ₹{low:,.0f} — ₹{high:,.0f}</div>
    <div style="margin-top:14px; padding:8px 16px; display:inline-block;
                background:rgba(255,255,255,0.05); border-radius:8px;
                font-size:0.9rem; color:{sentiment_color}; font-weight:600;">
        {sentiment_label}
    </div>
    <p style="color:#334155; font-size:0.78rem; margin-top:16px; margin-bottom:0;">
        * ML-based estimate. Actual fares depend on availability and booking time.
    </p>
</div>
""",
                    unsafe_allow_html=True,
                )

                # ── Insight Row ────────────────────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                ic1, ic2, ic3, ic4 = st.columns(4)

                with ic1:
                    st.markdown(
                        f'<div class="insight-card">'
                        f'<div class="insight-icon">🎯</div>'
                        f'<div class="insight-title">Model Accuracy</div>'
                        f'<div class="insight-value">85.8%</div>'
                        f'<div class="insight-delta" style="color:#10b981;">R² Score</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with ic2:
                    volatility = "Low" if days_away > 14 else ("Medium" if days_away > 5 else "High")
                    vol_color = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"}[volatility]
                    st.markdown(
                        f'<div class="insight-card">'
                        f'<div class="insight-icon">📉</div>'
                        f'<div class="insight-title">Price Volatility</div>'
                        f'<div class="insight-value">{volatility}</div>'
                        f'<div class="insight-delta" style="color:{vol_color};">{days_away}d until departure</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with ic3:
                    peak_hour = dep_time.hour in range(6, 10) or dep_time.hour in range(18, 22)
                    hour_label = "Peak ⬆️" if peak_hour else "Off-Peak ✅"
                    hour_color = "#ef4444" if peak_hour else "#10b981"
                    st.markdown(
                        f'<div class="insight-card">'
                        f'<div class="insight-icon">🕐</div>'
                        f'<div class="insight-title">Departure Slot</div>'
                        f'<div class="insight-value">{dep_time.strftime("%H:%M")}</div>'
                        f'<div class="insight-delta" style="color:{hour_color};">{hour_label}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with ic4:
                    margin = high - low
                    st.markdown(
                        f'<div class="insight-card">'
                        f'<div class="insight-icon">💰</div>'
                        f'<div class="insight-title">Price Swing</div>'
                        f'<div class="insight-value">₹{margin:,.0f}</div>'
                        f'<div class="insight-delta" style="color:#64748b;">Estimated ± range</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                # ── Booking Tips ───────────────────────────────────────────
                tips = []
                if days_away < 7:
                    tips.append("⚡ Last-minute fares can be 30–40% higher. Book now to lock this price.")
                elif days_away > 30:
                    tips.append("📅 Booking 30–60 days ahead typically gives the best deals.")
                if peak_hour:
                    tips.append("🕗 Morning (6–10 AM) and evening (6–10 PM) slots are pricier. Consider mid-day flights.")
                if flight_class_val == 0 and airline in ("Vistara", "Air India"):
                    tips.append("💺 Full-service carriers often include meals & more baggage allowance at no extra cost.")

                if tips:
                    tips_html = "<br>".join(f"• {t}" for t in tips)
                    st.markdown(
                        f'<div class="tips-banner">💡 <b>Booking Tips</b><br>{tips_html}</div>',
                        unsafe_allow_html=True,
                    )

                # ── Route Summary ──────────────────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("🔍 Prediction Details"):
                    detail_df = pd.DataFrame({
                        "Feature": [
                            "Route", "Airline", "Class", "Journey Date",
                            "Departure", "Days Until Travel"
                        ],
                        "Value": [
                            f"{source} → {destination}",
                            airline,
                            travel_class,
                            str(journey_date),
                            dep_time.strftime("%H:%M"),
                            str(days_away),
                        ],
                    })
                    st.dataframe(detail_df, use_container_width=True, hide_index=True)
                    st.caption(
                        f"Feature vector shape: {features.shape} | "
                        f"Active flags: {int(features.sum(axis=1).iloc[0])}"
                    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — AI TRAVEL ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🤖 Your Personal Travel Assistant")
    st.caption("Ask anything about fares, routes, baggage, airports, or travel tips.")

    if not GOOGLE_API_KEY:
        st.warning(
            "🔑 **AI Assistant is offline.**  \n"
            "Add `GOOGLE_API_KEY=your_key_here` to a `.env` file in the project root and restart."
        )
    else:
        # Context from last prediction
        last_ctx = st.session_state.get("last_prediction")

        if last_ctx:
            st.markdown(
                f'<div class="route-pill">'
                f'📌 Context: {last_ctx["route"]} · {last_ctx["airline"]} · ₹{last_ctx["predicted_fare"]}'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Voice Input
        st.markdown("---")
        with st.expander("🎙️ Voice Input (click to expand)", expanded=False):
            st.caption("Record your question — it will be transcribed automatically.")
            audio = mic_recorder(
                start_prompt="🔴 Start Recording",
                stop_prompt="⏹️ Stop & Send",
                key="recorder",
            )

        user_input = None

        if audio and audio.get("bytes"):
            with st.spinner("Transcribing audio…"):
                user_input = transcribe_audio(audio["bytes"])
            if user_input:
                st.success(f"🎙️ Transcribed: *{user_input}*")

        # Text fallback
        if text_input := st.chat_input("Ask me about flights, fares, or travel tips…"):
            user_input = text_input

        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    response_text = get_ai_response(user_input, context=last_ctx)
                    st.markdown(response_text)

                    # TTS (limit to avoid large payloads)
                    _resp_str: str = str(response_text)
                    tts_text: str = _resp_str[:400].rsplit(" ", 1)[0]  # pyre-ignore[16]
                    audio_data = text_to_speech(tts_text)
                    if audio_data:
                        autoplay_audio(audio_data)

            st.session_state.messages.append({"role": "assistant", "content": response_text})

        # Clear chat button
        if st.session_state.get("messages"):
            if st.button("🗑️ Clear Chat History", use_container_width=False):
                st.session_state.messages = []
                st.rerun()


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
fc1, fc2, fc3 = st.columns([2, 1, 1])
with fc1:
    st.markdown(
        "<p style='color:#475569; font-size:0.85rem;'>"
        "Powered by <b>RandomForestRegressor</b> · <b>Google Gemini 1.5 Flash</b> · "
        "<b>Streamlit</b> | Training data: Kaggle Indian Flight Dataset"
        "</p>",
        unsafe_allow_html=True,
    )
with fc2:
    st.markdown("**🔗 Resources**")
    st.markdown("[GitHub Repository](https://github.com/Prince2005v/SkyPrice-AI)")
with fc3:
    st.markdown("**📬 Support**")
    st.markdown("[Open an Issue](https://github.com/Prince2005v/SkyPrice-AI/issues)")
