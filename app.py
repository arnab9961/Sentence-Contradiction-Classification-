from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

def load_model():
    model_path = "model.joblib"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def compute_features(sentence1, sentence2):
    return np.array([sentence1, sentence2]).reshape(1, -1)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        sentence1 = request.form['sentence1']
        sentence2 = request.form['sentence2']
        
        model = load_model()
        if model is not None:
            try:
                features = compute_features(sentence1, sentence2)
                prediction = model.predict(features)[0]
                labels = ['Entailment', 'Neutral', 'Contradiction']
                result = f'Prediction: {labels[prediction]}'
            except Exception as e:
                result = f"Error in model prediction: {e}"
        else:
            result = "Model not found. Please upload a valid model.joblib file."
    
    return render_template('index.html', result=result)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
