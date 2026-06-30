import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
import xgboost as xgb
import tensorflow as tf

def train_crop_model():
    print("=====================================================================")
    print(" 1. TRAINING CROP RECOMMENDATION ENGINE (crop_model.pkl)")
    print("=====================================================================")
    
    # 1. Synthetic Data Generation (Replicating Kaggle + Data.gov.in Features)
    np.random.seed(42)
    num_samples = 2200
    
    crops = ['Rice', 'Maize', 'Cotton', 'Wheat', 'Moong', 'Jowar', 'Arhar', 'Urad', 
             'Sugarcane', 'Turmeric', 'Groundnut', 'Barley', 'Gram', 'Masoor', 'Mustard',
             'Soyabean', 'Sunflower', 'Ragi', 'Bajra', 'Tobacco', 'Jute', 'Coffee']
    
    states = ['Gujarat', 'Maharashtra', 'Rajasthan', 'Punjab']
    districts = ['Ahmedabad', 'Surat', 'Rajkot', 'Nagpur', 'Pune', 'Jaipur', 'Amritsar']
    seasons = ['Kharif', 'Rabi', 'Zaid']
    soils = ['Black', 'Alluvial', 'Red', 'Laterite', 'Desert', 'Mountain']
    
    data = {
        'N': np.random.uniform(10, 140, num_samples),
        'P': np.random.uniform(10, 140, num_samples),
        'K': np.random.uniform(10, 200, num_samples),
        'ph': np.random.uniform(4.0, 8.5, num_samples),
        'temperature': np.random.uniform(15, 38, num_samples),
        'humidity': np.random.uniform(30, 95, num_samples),
        'rainfall': np.random.uniform(300, 1500, num_samples),
        'state': np.random.choice(states, num_samples),
        'district': np.random.choice(districts, num_samples),
        'season': np.random.choice(seasons, num_samples),
        'soil_type': np.random.choice(soils, num_samples)
    }
    df = pd.DataFrame(data)
    df['crop_name'] = np.random.choice(crops, num_samples)

    # 2. Categorical Mappings & Encoders
    categorical_cols = ['state', 'district', 'season', 'soil_type']
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col])
        encoders[col] = le
        
    target_le = LabelEncoder()
    df['crop_enc'] = target_le.fit_transform(df['crop_name'])
    encoders['target'] = target_le

    features = ['N', 'P', 'K', 'ph', 'temperature', 'humidity', 'rainfall',
                'state_enc', 'district_enc', 'season_enc', 'soil_type_enc']
    
    X = df[features]
    y = df['crop_enc']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Ensemble Voting Setup
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
    gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb_clf = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, eval_metric='mlogloss', random_state=42)
    
    ensemble_model = VotingClassifier(
        estimators=[('rf', rf_clf), ('gb', gb_clf), ('xgb', xgb_clf)],
        voting='soft'
    )
    
    print("Fitting Ensemble Pipeline (RF + XGBoost + Gradient Boosting)...")
    ensemble_model.fit(X_train_scaled, y_train)
    
    accuracy = ensemble_model.score(X_test_scaled, y_test)
    print(f"-> Success! Crop Model Testing Accuracy: {accuracy * 100:.2f}%")
    
    # 4. Serialize Directly according to Project Structure Specification File Name
    artifacts = {
        'model': ensemble_model,
        'scaler': scaler,
        'encoders': encoders,
        'features': features
    }
    
    crop_path = os.path.join('models', 'crop_model.pkl')
    with open(crop_path, 'wb') as f:
        pickle.dump(artifacts, f)
    print(f"Artifact package successfully saved -> {crop_path}\n")


def train_soil_model():
    print("=====================================================================")
    print(" 2. TRAINING DEEP LEARNING SOIL ANALYSIS PORTAL (soil_model.pkl)")
    print("=====================================================================")
    
    # 1. Data Generation Mapping (Replicating Soil Health Card Schema metrics)
    np.random.seed(42)
    num_samples = 1500
    
    crops = ['Rice', 'Wheat', 'Maize', 'Cotton']
    soils = ['Alluvial', 'Black', 'Red', 'Laterite']
    moisture_levels = ['Low', 'Medium', 'High']
    
    data = {
        'intended_crop': np.random.choice(crops, num_samples),
        'soil_type': np.random.choice(soils, num_samples),
        'N': np.random.uniform(0, 150, num_samples),
        'P': np.random.uniform(0, 150, num_samples),
        'K': np.random.uniform(0, 210, num_samples),
        'ph': np.random.uniform(3.5, 9.0, num_samples),
        'organic_carbon': np.random.uniform(0.1, 3.0, num_samples),
        'ec': np.random.uniform(0, 4.0, num_samples),
        'moisture_retention': np.random.choice(moisture_levels, num_samples),
        'def_zn': np.random.choice([0, 1], num_samples),
        'def_fe': np.random.choice([0, 1], num_samples),
        'def_mn': np.random.choice([0, 1], num_samples),
        'def_b': np.random.choice([0, 1], num_samples)
    }
    df = pd.DataFrame(data)
    
    # Generate continuous target outputs
    base_score = 100 - (df['def_zn'] + df['def_fe'] + df['def_mn'] + df['def_b']) * 8
    base_score -= np.abs(df['ph'] - 6.5) * 8
    base_score -= (df['organic_carbon'] < 0.5).astype(int) * 15
    df['health_score'] = np.clip(base_score + np.random.normal(0, 4, num_samples), 10, 100)
    
    # Structural Processing Pipeline
    encoders = {}
    categorical_cols = ['intended_crop', 'soil_type', 'moisture_retention']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col])
        encoders[col] = le
        
    feature_cols = ['N', 'P', 'K', 'ph', 'organic_carbon', 'ec', 
                    'intended_crop_enc', 'soil_type_enc', 'moisture_retention_enc',
                    'def_zn', 'def_fe', 'def_mn', 'def_b']
                    
    X = df[feature_cols]
    y = df['health_score'] / 100.0  # Normalized target for Sigmoid node limits
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. Sequential Neural Network Model Compilation
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(len(feature_cols),)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    
    print("Fitting Keras sequential topology over numerical metrics matrix...")
    model.fit(X_train_scaled, y_train, epochs=15, batch_size=32, validation_split=0.1, verbose=0)
    
    loss, mae = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"-> Success! Neural Network Mean Absolute Error: {mae:.4f}")
    
    # 3. Bundle Keras weight state along with standard processing components into one file
    soil_artifacts = {
        'model_weights': model.get_weights(),
        'scaler': scaler,
        'encoders': encoders,
        'feature_names': feature_cols
    }
    
    soil_path = os.path.join('models', 'soil_model.pkl')
    with open(soil_path, 'wb') as f:
        pickle.dump(soil_artifacts, f)
    print(f"Artifact package successfully saved -> {soil_path}\n")


if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    train_crop_model()
    train_soil_model()
    print("=====================================================================")
    print(" ALL STRUCTURAL PIPELINES SERIALIZED AND ALIGNED MATCHING SYSTEM FILES")
    print("=====================================================================")