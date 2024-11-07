import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

try:
    # Load model and scaler
    MODEL_PATH = r'C:\Users\AdithiyanPV\OneDrive\Desktop\INOSYS SPRINGBOARD OBESITY LEVEL PREDICTION\xgb_top4_features_model.pkl'
    SCALER_PATH = r'C:\Users\AdithiyanPV\OneDrive\Desktop\INOSYS SPRINGBOARD OBESITY LEVEL PREDICTION\scaled_final.pkl'
    
    model = joblib.load(MODEL_PATH)
    pt = joblib.load(SCALER_PATH)
    
    logger.info("Model loaded successfully")
    
    # Log the expected feature names
    if hasattr(model, 'get_booster'):
        feature_names = model.get_booster().feature_names
        logger.info(f"Model expects features: {feature_names}")
except Exception as e:
    logger.error(f"Error loading model or scaler: {str(e)}")
    raise

# Define category mappings
OBESITY_CATEGORIES = [
    'Insufficient Weight',
    'Normal Weight', 
    'Overweight Level I',
    'Overweight Level II',
    'Obesity Type I',
    'Obesity Type II',
    'Obesity Type III'
]

# Update feature names to match exactly what the model expects
EXPECTED_FEATURES = ['FCVC', 'Height', 'Weight']  # Removed 'Gender' if it's not in the model's features

def prepare_input_data(form_data):
    """
    Prepare and validate input data for prediction
    """
    try:
        # Extract and convert form data - maintain uppercase feature names
        input_dict = {
            'Height': float(form_data.get('Height', 0)),
            'Weight': float(form_data.get('Weight', 0)),
            'FCVC': float(form_data.get('FCVC', 1))
        }
        
        logger.debug(f"Extracted form data: {input_dict}")
        
        # Create DataFrame with single row
        input_data = pd.DataFrame([input_dict], columns=EXPECTED_FEATURES)
        
        logger.debug(f"Created input DataFrame: {input_data.to_dict('records')}")
        
        # Validate input ranges            
        if not (1.0 <= input_data['Height'].iloc[0] <= 2.5):
            raise ValueError("Height must be between 1.0 and 2.5 meters")
            
        if not (30 <= input_data['Weight'].iloc[0] <= 200):
            raise ValueError("Weight must be between 30 and 200 kg")
            
        if not (1 <= input_data['FCVC'].iloc[0] <= 3):
            raise ValueError("FCVC must be between 1 and 3")
        
        logger.debug("Input validation successful")
        return input_data
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error in prepare_input_data: {str(e)}")
        raise Exception(f"Error preparing input data: {str(e)}")

def transform_features(input_data):
    """
    Apply feature scaling transformation
    """
    try:
        logger.debug(f"Input data before scaling: {input_data.to_dict('records')}")
        logger.debug(f"Input data columns: {input_data.columns.tolist()}")
        
        # Apply power transformer
        scaled_features = pt.transform(input_data)
        
        # Create DataFrame with scaled values
        scaled_df = pd.DataFrame(
            scaled_features,
            columns=input_data.columns
        )
        
        logger.debug(f"Scaled data: {scaled_df.to_dict('records')}")
        return scaled_df
        
    except Exception as e:
        logger.error(f"Error in transform_features: {str(e)}")
        raise Exception(f"Error in feature transformation: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Log received data
        logger.debug(f"Received form data: {request.form}")
        
        # Prepare input data
        input_data = prepare_input_data(request.form)
        logger.info("Input data prepared successfully")
        
        # Transform features
        scaled_data = transform_features(input_data)
        logger.info("Features transformed successfully")
        
        # Make prediction
        prediction = model.predict(scaled_data)[0]
        logger.info(f"Raw prediction value: {prediction}")
        
        # Map numerical prediction to category
        result = OBESITY_CATEGORIES[int(prediction)]
        logger.info(f"Final prediction: {result}")
        
        return render_template('index.html', result=result)
    
    except ValueError as e:
        logger.error(f"ValueError in prediction: {str(e)}")
        return render_template('index.html', error=str(e))
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return render_template('index.html', error=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True, port=5001)