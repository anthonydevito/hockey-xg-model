import pandas as pd
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, plot_importance
from features import calculate_shot_features

def train():
    # Load shot data
    df = pd.read_csv('data/shots_2018-2024.csv')
    
    # Run dataset with created feature function
    df = calculate_shot_features(df)
    
    # With these features
    features = ['dist_to_net', 'shot_angle', 'shotRebound', 'shotRush'] + \
               [col for col in df.columns if col.startswith('type_')]
    
    X = df[features]
    y = df['goal'] # 1 if goal, 0 otherwise
    
    # Train the actual model
    model = XGBClassifier(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X, y)
    
    # Save and list the model
    joblib.dump(model, 'models/xg_model.joblib')
    joblib.dump(features, 'models/feature_list.joblib')
    print("Model trained and saved to models/xg_model.joblib")

    # Create Feature Importance plot PNG
    plt.figure(figsize=(10, 8))
    plot_importance(model)
    plt.title("Feature Importance: NHL xG Model")
    plt.tight_layout()
    plt.savefig('models/feature_importance.png')
    print("Feature importance plot saved to models/feature_importance.png")

if __name__ == "__main__":
    train()