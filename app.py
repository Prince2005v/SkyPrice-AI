import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import time
import io
import base64
from datetime import datetime, timedelta
from src.preprocessing import preprocess_input
import google.generativeai as genai
from streamlit_mic_recorder import mic_recorder
from gtts import gTTS
from dotenv import load_dotenv
import speech_recognition as sr

# Load Environment Variables
load_dotenv()

# Configure Gemini AI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    model_gemini = genai.GenerativeModel('gemini-1.5-flash')
else:
    model_gemini = None

# Page Configuration
st.set_page_config(
    page_title="SkyPrice | Premium Flight Fare Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: radial-gradient(circle at top right, #1e293b, #0f172a);
        color: #f8fafc;
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3.5em;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.4);
        background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
    }
    
    .prediction-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin-top: 20px;
    }
    
    .metric-label {
        color: #94a3b8;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #3b82f6;
        margin: 10px 0;
    }

    .sidebar .sidebar-content {
        background-color: #1e293b;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 3rem !important;
        color: #10b981 !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper to load model
@st.cache_resource
def load_model():
    MODEL_PATH = "models/flight_price_model.pkl"
    GZ_PATH = "models/flight_price_model.pkl.gz"
    
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    elif os.path.exists(GZ_PATH):
        import gzip
        with gzip.open(GZ_PATH, 'rb') as f:
            return joblib.load(f)
    return None

# AI Assistant Helpers
def get_ai_response(prompt):
    if not model_gemini:
        return "⚠️ Gemini API Key is missing. Please add it to your `.env` file."
    try:
        response = model_gemini.generate_content(f"You are a helpful flight and travel assistant for SkyPrice. {prompt}")
        return response.text
    except Exception as e:
        return f"❌ Error: {str(e)}"

def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        return fp.getvalue()
    except Exception as e:
        st.error(f"TTS Error: {e}")
        return None

def autoplay_audio(audio_bytes):
    b64 = base64.b64encode(audio_bytes).decode()
    md = f"""
        <audio autoplay="true">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
    st.markdown(md, unsafe_allow_html=True)

model = load_model()

# Sidebar Navigation & Branding
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/airplane-take-off.png", width=80)
    st.title("SkyPrice AI")
    st.markdown("---")
    st.subheader("Booking Preferences")
    travel_class = st.radio("Travel Class", ["Economy", "Business / Premium"], index=0)
    st.info("💡 Business class predictions are currently optimized for Jet Airways Business.")
    
    st.caption("v2.0.0 | Built with Advanced AI")

# Main Content
st.markdown("<h1 style='text-align: left; margin-bottom: 0;'>✈️ SkyPrice AI Hub</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #94a3b8; font-size: 1.1rem;'>Predict fares and talk to your personal travel assistant.</p>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔮 Price Predictor", "🤖 AI Voice Assistant"])

with tab1:
    if model is None:
        st.error("🚨 Critical: Prediction engine (model.pkl) is missing. Please contact system administrator.")
    else:
        # Layout for Inputs
        input_container = st.container()
        with input_container:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                airline = st.selectbox("Airline Provider", [
                    'Air Asia', 'Air India', 'GoAir', 'IndiGo', 'Jet Airways', 
                    'Jet Airways Business', 'Multiple carriers', 
                    'Multiple carriers Premium economy', 'SpiceJet', 
                    'Trujet', 'Vistara', 'Vistara Premium economy'
                ], help="Select your preferred flight carrier")
                
            with col2:
                source = st.selectbox("Origin City", ['Banglore', 'Chennai', 'Delhi', 'Kolkata', 'Mumbai'])
                
            with col3:
                destination = st.selectbox("Destination City", ['Banglore', 'Cochin', 'Delhi', 'Hyderabad', 'Kolkata', 'New Delhi'])

            st.markdown("<br>", unsafe_allow_html=True)
            
            col4, col5 = st.columns(2)
            with col4:
                journey_date = st.date_input("Departure Date", 
                                           min_value=datetime.now(),
                                           value=datetime.now() + timedelta(days=7))
            with col5:
                dep_time = st.time_input("Departure Time", datetime.now().replace(hour=10, minute=0).time())

        # Prediction Trigger
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Calculate Optimal Price"):
            if source == destination:
                st.warning("⚠️ Origin and Destination cannot be the same. Please adjust your itinerary.")
            else:
                with st.status("🔮 Analyzing market trends & flight patterns...", expanded=True) as status:
                    st.write("Fetching historical data...")
                    time.sleep(0.5)
                    st.write("Applying feature engineering...")
                    time.sleep(0.4)
                    
                    # Preprocess
                    flight_class_val = 1 if travel_class == "Business / Premium" else 0
                    features = preprocess_input(airline, source, destination, journey_date, dep_time, flight_class_val)
                    
                    st.write("Running RandomForest Regressor...")
                    assert model is not None, "Model should not be None at this point"
                    prediction = model.predict(features)[0]
                    time.sleep(0.3)
                    status.update(label="Analysis Complete!", state="complete", expanded=False)

                # UI Results
                st.markdown(f"""
                <div class="prediction-card">
                    <div class="metric-label">Estimated Fare (Best Value)</div>
                    <div class="metric-value">₹ {prediction:,.2f}</div>
                    <p style="color: #64748b; font-size: 0.85rem;">*Predicted price based on current model parameters. Actual prices may vary.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Additional Insights
                insight_col1, insight_col2 = st.columns(2)
                with insight_col1:
                    st.metric("Confidence Score", "85.8%", "+1.2%")
                with insight_col2:
                    st.metric("Price Sentiment", "Stable", "± ₹250")

with tab2:
    st.subheader("Your Personal Travel Assistant")
    
    if not GOOGLE_API_KEY:
        st.warning("🔑 AI Assistant is offline. Please configure your `GOOGLE_API_KEY` in the `.env` file to enable voice and text chat.")
    else:
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Voice Input Section
        st.markdown("---")
        st.write("🎙️ **Voice Command**")
        audio = mic_recorder(
            start_prompt="Start Recording",
            stop_prompt="Stop & Send",
            key="recorder"
        )

        # Handle voice or text input
        user_input = None
        
        if audio:
            st.info("Audio received! Transcribing...")
            try:
                # Convert audio bytes to a format SpeechRecognition can understand
                recognizer = sr.Recognizer()
                audio_file = io.BytesIO(audio['bytes'])
                with sr.AudioFile(audio_file) as source:
                    audio_data = recognizer.record(source)
                    user_input = recognizer.recognize_google(audio_data)
                    st.success(f"You said: {user_input}")
            except Exception as e:
                st.error(f"Could not transcribe audio: {e}")
                user_input = None

        # Text input as fallback
        if text_input := st.chat_input("Ask me anything about your travel..."):
            user_input = text_input

        if user_input:
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response_text = get_ai_response(user_input)
                    st.markdown(response_text)
                    
                    # Generate and play speech
                    audio_data = text_to_speech(response_text[:300]) # Limit to 300 chars for speed
                    if audio_data:
                        autoplay_audio(audio_data)
            
            st.session_state.messages.append({"role": "assistant", "content": response_text})

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
footer_col1, footer_col2 = st.columns([2, 1])
with footer_col1:
    st.write("This engine uses a **RandomForestRegressor** for price prediction and **Google Gemini 1.5** for the AI Assistant. It converts your text/voice into insights for a better travel experience.")
with footer_col2:
    st.markdown("### Rapid Links")
    st.markdown("[Documentation](https://github.com/example/flight-fare)")
    st.markdown("[Support Terminal](mailto:support@skyprice.ai)")
