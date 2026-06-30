import os
import pickle
import numpy as np
from flask import Blueprint, request, jsonify, session

crop_bp = Blueprint('crop', __name__)
MODEL_PATH = os.path.join('models', 'crop_model.pkl')

def load_crop_artifacts():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

@crop_bp.route('/api/crop/recommend', methods=['POST'])
def recommend_crop():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized portal access.'}), 401
        
    artifacts = load_crop_artifacts()
    if not artifacts:
        return jsonify({'error': 'Ensemble model file missing. Please run train_models.py first.'}), 500
        
    data = request.json or request.form
    
    try:
        # Get and sanitize raw inputs safely
        state = data.get('state')
        district = data.get('district')
        season = data.get('season')
        soil_type = data.get('soil_type')
        
        encoders = artifacts['encoders']
        
        # Safe categorical checks to pinpoint encoding issues
        if state not in encoders['state'].classes_:
            return jsonify({'error': f"State '{state}' was not part of the model's training data. Please use Gujarat, Maharashtra, Rajasthan, or Punjab."}), 400
        if district not in encoders['district'].classes_:
            return jsonify({'error': f"District '{district}' was not part of the model's training data. Please use Ahmedabad, Surat, Rajkot, Nagpur, Pune, Jaipur, or Amritsar."}), 400
        if season not in encoders['season'].classes_:
            return jsonify({'error': f"Season '{season}' is invalid."}), 400
        if soil_type not in encoders['soil_type'].classes_:
            return jsonify({'error': f"Soil Type '{soil_type}' is invalid."}), 400

        # Run safe text encoding transforms
        state_enc = encoders['state'].transform([state])[0]
        district_enc = encoders['district'].transform([district])[0]
        season_enc = encoders['season'].transform([season])[0]
        soil_enc = encoders['soil_type'].transform([soil_type])[0]
        
        # Assemble feature vector array safely
        raw_features = np.array([[
            float(data.get('n', 0)),
            float(data.get('p', 0)),
            float(data.get('k', 0)),
            float(data.get('ph', 6.5)),
            float(data.get('temperature', 28.0)),
            float(data.get('humidity', 60.0)),
            float(data.get('rainfall', 800)),
            state_enc,
            district_enc,
            season_enc,
            soil_enc
        ]])
        
        # Standardize features using training scaler matrix configurations
        scaled_features = artifacts['scaler'].transform(raw_features)
        
        model = artifacts['model']
        prob_distribution = model.predict_proba(scaled_features)[0]
        
        top_indices = np.argsort(prob_distribution)[::-1][:5]
        target_encoder = encoders['target']
        
        crop_notes = {
            'Rice': 'Requires abundant watering and structural clayey soils.',
            'Maize': 'Ensure good drainage system and timely Nitrogen applications.',
            'Cotton': 'Thrives best in deep black soils with moderate rainfall.',
            'Wheat': 'Optimal choices during winter cycles with light irrigation.',
            'Moong': 'Great low-moisture alternative for drought periods.'
        }
        
        recommendations = []
        for idx in top_indices:
            crop_name = target_encoder.inverse_transform([idx])[0]
            confidence_score = float(prob_distribution[idx] * 100)
            tips = crop_notes.get(crop_name, "Maintain standard crop rotation balances and monitor active soil wetness.")
            
            recommendations.append({
                "crop": crop_name,
                "confidence": round(confidence_score, 1),
                "tips": tips
            })
            
        return jsonify(recommendations)
        
    except ValueError as val_err:
        return jsonify({'error': f'Value Conversion Error: Verify all numerical inputs are correctly filled. details: {str(val_err)}'}), 400
    except Exception as err:
        return jsonify({'error': f'Internal Server Core Failure: {str(err)}'}), 500