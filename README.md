# ✈️ SkyPrice AI — Flight Fare Predictor

> AI-powered Indian domestic flight fare prediction with a voice-enabled travel assistant.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Model](https://img.shields.io/badge/Model-RandomForestRegressor-green)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🚀 Features

| Feature | Description |
|---|---|
| 🔮 **Fare Prediction** | ML-based price estimate for 12 airlines across 5 origins × 6 destinations |
| 📊 **Price Insights** | Confidence score, price sentiment, volatility indicator, booking tips |
| 🤖 **AI Travel Assistant** | Google Gemini 1.5 Flash-powered chatbot with flight context awareness |
| 🎙️ **Voice Commands** | Microphone recording → Speech-to-Text → AI response → TTS audio playback |
| 🌐 **Premium UI** | Dark glassmorphism design, animated cards, responsive layout |

---

## 🗂️ Project Structure

```
Flight Fare Prediction/
├── app.py                   # Main Streamlit application
├── src/
│   ├── __init__.py
│   └── preprocessing.py     # Feature engineering + input validation
├── models/
│   ├── flight_price_model.pkl.gz   # Trained RandomForest model (compressed)
│   └── feature_columns.pkl         # Feature order reference
├── tests/
│   └── test_preprocess.py   # pytest test suite (20+ tests)
├── notebooks/               # EDA & training notebooks
├── requirements.txt
└── .env                     # API keys (not committed)
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/Prince2005v/SkyPrice-AI.git
cd SkyPrice-AI
```

### 2. Create & activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

> Get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com)

### 5. Run the app

```bash
streamlit run app.py
```

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Algorithm | `RandomForestRegressor` |
| R² Score | ~85.8% |
| Training Data | [Kaggle Flight Price Dataset](https://www.kaggle.com/nikhilmittal/flight-fare-prediction-mh) |
| Features | 25 (class, departure time, route, airline — one-hot encoded) |
| Supported Airlines | Air Asia, Air India, GoAir, IndiGo, Jet Airways, SpiceJet, Trujet, Vistara + premium variants |
| Supported Routes | 5 origins × 6 destinations (Indian domestic) |

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

The test suite covers:
- Feature shape and column order
- One-hot encoding correctness (including reference categories)
- Input validation (invalid airline, source, destination, same city, bad class)
- `datetime` vs `date` input handling
- Parametrized runs over all 12 airlines and all 5 source cities

---

## 🌐 Deployment

The app is deployed on **Streamlit Community Cloud**.

> Live URL: *[Add your Streamlit Cloud URL here]*

To redeploy after changes:
1. Push to the `main` branch on GitHub
2. Streamlit Cloud will auto-redeploy

---

## 📄 License

MIT © [Prince](https://github.com/Prince2005v)
