import streamlit as st
import joblib
import numpy as np
from thefuzz import fuzz
import os

# Set Page Title with Search Icon in Tab
st.set_page_config(
    page_title="🔍 Sentence Contradiction Classifier",
    layout="centered"
)

def load_model():
    try:
        model_path = "model.joblib"
        if not os.path.exists(model_path):
            return None
        return joblib.load(model_path)
    except Exception:
        return None

def compute_features(sentence1, sentence2):
    return np.array([sentence1, sentence2]).reshape(1, -1)

# Apply Gradient Background
gradient_css = """
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(to right, #6a11cb, #2575fc);
    }
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }
    .title {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        color: white;
    }
    .title span {
        color: yellow;
        font-size: 60px;
    }
    .prediction-box {
        font-size: 28px; 
        font-weight: bold; 
        padding: 20px; 
        border-radius: 8px; 
        text-align: center;
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        margin-top: 20px;
        color: white;
    }
</style>
"""
st.markdown(gradient_css, unsafe_allow_html=True)

st.markdown('<div class="title"><span>S</span>entence Contradiction Classifier</div>', unsafe_allow_html=True)
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
            # Fuzzy matching as a fallback
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
    "**I sampled 3,000 rows from the dataset to reduce the size of the `.joblib` file, as a larger file would make deployment on Streamlit difficult. Otherwise, the results might show better.**"
)
