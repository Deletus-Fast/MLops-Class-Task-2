from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the trained model and scaler
model = joblib.load('house_price_model.pkl')
scaler = joblib.load('scaler.pkl')

# Helper function to validate input
def validate_input(data):
    try:
        # Validate numeric fields
        area = float(data.get('area', 0))
        bedrooms = int(data.get('bedrooms', 0))
        bathrooms = int(data.get('bathrooms', 0))
        stories = int(data.get('stories', 0))
        parking = int(data.get('parking', 0))
        
        # Validate categorical fields (yes/no, furnished/semi/unfurnished)
        if data.get('mainroad', 'no') not in ['yes', 'no']:
            return False
        if data.get('guestroom', 'no') not in ['yes', 'no']:
            return False
        if data.get('basement', 'no') not in ['yes', 'no']:
            return False
        if data.get('hotwaterheating', 'no') not in ['yes', 'no']:
            return False
        if data.get('airconditioning', 'no') not in ['yes', 'no']:
            return False
        if data.get('prefarea', 'no') not in ['yes', 'no']:
            return False
        if data.get('furnishingstatus', 'unfurnished') not in ['furnished', 'semi-furnished', 'unfurnished']:
            return False
        
        return True
    except (ValueError, TypeError):
        return False

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Parse incoming data (expecting JSON)
    data = request.json

    if not validate_input(data):
        return jsonify({'error': 'Invalid input'}), 400
    
    # Extract the features from the request and convert them to match the training data
    features = np.array([
        data.get('area', 0),
        data.get('bedrooms', 0),
        data.get('bathrooms', 0),
        data.get('stories', 0),
        1 if data.get('mainroad', 'no') == 'yes' else 0,
        1 if data.get('guestroom', 'no') == 'yes' else 0,
        1 if data.get('basement', 'no') == 'yes' else 0,
        1 if data.get('hotwaterheating', 'no') == 'yes' else 0,
        1 if data.get('airconditioning', 'no') == 'yes' else 0,
        data.get('parking', 0),
        1 if data.get('prefarea', 'no') == 'yes' else 0,
        2 if data.get('furnishingstatus', 'unfurnished') == 'furnished' 
        else 1 if data.get('furnishingstatus', 'unfurnished') == 'semi-furnished' else 0
    ]).reshape(1, -1)
    
    # Convert features to a DataFrame to retain column names and structure
    columns = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom',
               'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea',
               'furnishingstatus']
    
    input_df = pd.DataFrame(features, columns=columns)
    
    # Scale the features using the loaded scaler
    scaled_features = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(scaled_features)
    
    # Return the prediction as JSON
    return jsonify({'predicted_price': prediction[0]})

# Run the app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
