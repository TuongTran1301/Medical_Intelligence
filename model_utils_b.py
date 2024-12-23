import pickle
import pandas as pd

def load_model_and_scaler_b():
    with open('models/random_forest_model_b.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler_b.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def predict_health_b(input_data):
    feature_columns = ['Age', 'Gender', 'Total Bilirubin', 'Direct Bilirubin', 'Alkphos', 'SGPT', 'SGOT', 
                       'Total Proteins', 'ALB', 'A/G Ratio']
    input_df = pd.DataFrame([input_data], columns=feature_columns)
    
    model, scaler = load_model_and_scaler_b()
    input_scaled = scaler.transform(input_df)
    probability = model.predict_proba(input_scaled)[0][1]
    return probability, ["Lời khuyên cho mô hình B."]
