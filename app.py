from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from form
        features = [float(x) for x in request.form.values()]
        scaled_features = scaler.transform([features])
        prediction = model.predict(scaled_features)[0]

        # Map prediction to label
        result_map = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}
        result = result_map[prediction]
        return render_template('index.html', prediction_text=f'Predicted Match Result: {result}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')


if __name__ == "__main__":
    app.run(debug=True)
