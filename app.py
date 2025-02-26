import streamlit as st
import pickle
import numpy as np
import requests
from thefuzz import fuzz
import os

# GitHub Raw File URLs
MODEL_URL = "https://raw.githubusercontent.com/YOUR_GITHUB_USERNAME/YOUR_REPO/main/model.pkl"
IMAGE_URL = "https://raw.githubusercontent.com/YOUR_GITHUB_USERNAME/YOUR_REPO/main/image.png"

def download_model():
    model_path = "model.pkl"
    if not os.path.exists(model_path):
        response = requests.get(MODEL_URL)
        if response.status_code == 200:
            with open(model_path, "wb") as f:
                f.write(response.content)

def load_model():
    download_model()
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

def compute_features(sentence1, sentence2):
    features = [
        fuzz.QRatio(sentence1, sentence2),
        fuzz.partial_ratio(sentence1, sentence2),
        fuzz.token_sort_ratio(sentence1, sentence2),
        fuzz.token_set_ratio(sentence1, sentence2)
    ]
    return np.array(features).reshape(1, -1)

# Apply Gradient Background & Image
gradient_css = f"""
<style>
    [data-testid="stAppViewContainer"] {{
        background: url('{IMAGE_URL}');
        background-size: cover;
        background-position: center;
    }}
    [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
    }}
    .title {{
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        color: white;
    }}
    .prediction-box {{
        font-size: 28px; 
        font-weight: bold; 
        padding: 20px; 
        border-radius: 8px; 
        text-align: center;
        background: rgba(255, 255, 255, 0.3);
        backdrop-filter: blur(10px);
        margin-top: 20px;
        color: white;
    }}
</style>
"""
st.markdown(gradient_css, unsafe_allow_html=True)

st.markdown('<div class="title">🔍 Sentence Contradiction Classifier</div>', unsafe_allow_html=True)
st.write("Enter two sentences to determine their relationship: Entailment, Neutral, or Contradiction.")

sentence1 = st.text_input("Enter first sentence:")
sentence2 = st.text_input("Enter second sentence:")

if st.button("🔎 Classify"):
    if not sentence1 or not sentence2:
        st.warning("⚠️ Please enter both sentences.")
    else:
        model = load_model()
        if model is not None:
            try:
                features = compute_features(sentence1, sentence2)
                prediction = model.predict(features)[0]
                labels = ['Entailment', 'Neutral', 'Contradiction']
                result = f'✅ {labels[prediction]}'
            except:
                result = "⚠️ Error in model prediction"
        else:
            avg_score = sum([
                fuzz.QRatio(sentence1, sentence2),
                fuzz.partial_ratio(sentence1, sentence2),
                fuzz.token_sort_ratio(sentence1, sentence2),
                fuzz.token_set_ratio(sentence1, sentence2)
            ]) / 4

            if avg_score > 70:
                result = "✅ Entailment (based on similarity)"
            elif avg_score > 50:
                result = "⚠️ Neutral (based on similarity)"
            else:
                result = "❌ Contradiction (based on similarity)"

        st.markdown(f'<div class="prediction-box">{result}</div>', unsafe_allow_html=True)

st.markdown(
    "**Using model.pkl and image.png from GitHub to ensure easy deployment.**"
)
