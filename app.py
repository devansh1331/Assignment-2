from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model
model_path = 'model/best_heart_disease_model.pkl'
model = joblib.load(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data
    data = request.form
    # Convert data to input format for the model
    features = np.array([[float(data['age']), float(data['cp']), float(data['trestbps']),
                          float(data['chol']), float(data['fbs']), float(data['restecg']),
                          float(data['thalach']), float(data['exang']), float(data['oldpeak']),
                          float(data['slope']), float(data['ca']), float(data['thal'])]])
    
    # Make prediction
    prediction = model.predict(features)
    output = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"

    return render_template('index.html', prediction_text=output)

if __name__ == "__main__":
    app.run(debug=True)
