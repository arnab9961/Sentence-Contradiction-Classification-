import streamlit as st
import joblib
import numpy as np
import os

def load_model():
    model_path = "model.joblib"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def compute_features(sentence1, sentence2):
    return np.array([sentence1, sentence2]).reshape(1, -1)

# Streamlit UI
st.set_page_config(page_title="Sentence Contradiction Classifier")
st.title("🔍 Contradictory, My Dear Watson")
st.write("Enter two sentences to determine their relationship: Entailment, Neutral, or Contradiction.")

# Apply Background Image with Opacity
background_css = """
<style>
[data-testid="stAppViewContainer"] {
    background: url('image.png') no-repeat center center fixed;
    background-size: cover;
}
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(255, 255, 255, 0.6); /* 40% Opacity */
    z-index: -1;
}
</style>
"""
st.markdown(background_css, unsafe_allow_html=True)

# User Input
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
                result = f'**Prediction:** {labels[prediction]}'
            except Exception as e:
                result = f"⚠️ Error in model prediction: {e}"
        else:
            result = "⚠️ Model not found. Please upload a valid model.joblib file."
        
        st.markdown(f'<div style="font-size:24px; font-weight:bold; padding:10px; border-radius:5px; background:#f0f0f0;">{result}</div>', unsafe_allow_html=True)
