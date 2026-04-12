from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

import os

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained model, label encoder, and symptom vocabulary using absolute paths
try:
    model_path = os.path.join(BASE_DIR, 'models', 'model.pkl')
    label_path = os.path.join(BASE_DIR, 'models', 'label_encoder.pkl')
    vocab_path = os.path.join(BASE_DIR, 'models', 'symptom_vocab.pkl')
    
    model = joblib.load(model_path)
    label_encoder = joblib.load(label_path)
    symptom_list = joblib.load(vocab_path)
except Exception as e:
    print(f"Error loading model or files: {e}")
    exit(1)

# Route for index page (Home Page)
@app.route('/')
def index():
    return render_template('index.html')

# Route for predict page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        # Get selected symptoms from the form
        symptoms = [request.form.get(f'symptom{i}') for i in range(1, 18)]
        
        # Clean: lowercase, strip, and remove empty values
        symptoms = [s.lower().strip() for s in symptoms if s and s.strip() != '']

        # Create a binary input vector matching symptom_list order and length
        input_vector = [0] * len(symptom_list)
        for symptom in symptoms:
            if symptom in symptom_list:
                idx = symptom_list.index(symptom)
                input_vector[idx] = 1
            else:
                print(f"Unknown symptom: {symptom}")  # Debugging line to handle unknown symptoms

        # Predict encoded label
        try:
            pred_label = model.predict([input_vector])[0]
            # Decode label to disease name
            prediction = label_encoder.inverse_transform([pred_label])[0]
        except Exception as e:
            print(f"Prediction error: {e}")
            prediction = "Prediction failed"

    # Render the template with symptom_list and prediction if any
    return render_template('predict.html', symptom_list=symptom_list, prediction=prediction)

# Route for about page
@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)