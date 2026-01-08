import joblib
import pandas as pd

def predict_xg(new_data_csv):
    model = joblib.load('models/xg_model.joblib')
    features = joblib.load('models/feature_list.joblib')
    
    df = pd.read_csv(new_data_csv)
    # Assume features are already calculated or run the feature script
    X = df[features]
    
    probs = model.predict_proba(X)[:, 1] # Probability of the '1' class (Goal)
    return probs