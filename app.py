# app.py
from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import joblib
import os

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model_file = 'crop_recommendation_model.pkl'
scaler_file = 'scaler.pkl'

# Check if model and scaler exist, else create them
if not os.path.exists(model_file) or not os.path.exists(scaler_file):
    # Load dataset
    df1 = pd.read_csv('datasets/Crop_data1.csv')
    df2 = pd.read_csv('datasets/Crop_data2.csv')
    merged_data = pd.concat([df1, df2], axis=0)
    merged_data.drop_duplicates(subset=['N','P','K','temperature','humidity','ph','rainfall'], inplace=True)
    
    # Features and target
    continuous_columns = merged_data.select_dtypes(include=[float,int]).columns
    target_column = 'label'
    X = merged_data[continuous_columns]
    y = merged_data[target_column]
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model (SVM)
    model = SVC(kernel='linear', probability=True)
    model.fit(X_scaled, y)
    
    # Save model and scaler
    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)
else:
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)

# Homepage route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        features = {
            'N': float(request.form['N']),
            'P': float(request.form['P']),
            'K': float(request.form['K']),
            'temperature': float(request.form['temperature']),
            'humidity': float(request.form['humidity']),
            'ph': float(request.form['ph']),
            'rainfall': float(request.form['rainfall'])
        }
        
        # Convert to DataFrame
        features_df = pd.DataFrame([features])
        scaled_features = scaler.transform(features_df)
        
        # Predict
        prediction = model.predict(scaled_features)[0]
        probability = np.max(model.predict_proba(scaled_features)) * 100
        
        return render_template('index.html', prediction_text=f"Recommended Crop: {prediction} (Confidence: {probability:.2f}%)")
    
    except Exception as e:
        return f"Error: {e}"

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
