import joblib
import pandas as pd
from src.features import calculate_shot_features

# 1. Load model & features
model = joblib.load('models/xg_model.joblib')
features = joblib.load('models/feature_list.joblib')

# 2. Create two test shots
# Shot A: Point shot (far, wide angle)
# Shot B: Doorstep shot (close, centered)
test_shots = pd.DataFrame([
    {'xCordAdjusted': 35, 'yCordAdjusted': 25, 'shotRebound': 0, 'shotRush': 0, 'shotType': 'WRIST', 'goal': 0},
    {'xCordAdjusted': 85, 'yCordAdjusted': 2,  'shotRebound': 1, 'shotRush': 0, 'shotType': 'SNAP', 'goal': 0}
])

# 3. Process & Predict
df_processed = calculate_shot_features(test_shots)
# Ensure all columns exist even if not in the test data
for col in features:
    if col not in df_processed.columns:
        df_processed[col] = 0

X = df_processed[features]
probs = model.predict_proba(X)[:, 1]

print(f"Point Shot xG: {probs[0]:.4f}")  # Outputted 0.0155
print(f"Rebound/Crease Shot xG: {probs[1]:.4f}")  # Outputted 0.2605