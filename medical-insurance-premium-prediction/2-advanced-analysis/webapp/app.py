from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
try:
    model = pickle.load(open('models/model.pkl', 'rb'))
except FileNotFoundError:
    print("Error: Model file not found.")
    model = None

# ROUTES
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/service')
def service():
    return render_template('service.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/feature')
def feature():
    return render_template('feature.html')

@app.route('/health_predict')
def health_page():
    return render_template('health.html')

# PREDICTION ROUTE (Updated for "No Reload")
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'prediction_text': "Error: Model not loaded"})

    try:
        # Get data
        age = float(request.form['age'])
        transplants = int(request.form['transplants'])
        bmi = float(request.form['bmi']) 
        surgeries = int(request.form['surgeries'])
        chronic = int(request.form['chronic'])
        cancer = int(request.form['cancer'])

        features = np.array([[age, transplants, bmi, surgeries, chronic, cancer]])
        
        # Predict
        prediction = model.predict(features)
        output = round(prediction[0], 2)

        # Return data for JavaScript
        return jsonify({'prediction_text': f'Estimated Health Premium for year: INR{output}'})
        
    except Exception as e:
        return jsonify({'prediction_text': f"Error: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)