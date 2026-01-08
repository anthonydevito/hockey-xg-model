import numpy as np
import pandas as pd

def calculate_shot_features(df):
    # Starting with a copy
    df = df.copy()
    
    # Net is at (89, 0) in adjusted coordinates
    df['dist_to_net'] = np.sqrt((89 - df['xCordAdjusted'])**2 + (df['yCordAdjusted'])**2)
    
    # Shot Angle (0 degrees is straight on)
    df['shot_angle'] = np.degrees(np.arctan2(abs(df['yCordAdjusted']), (89 - df['xCordAdjusted'])))
    
    # 'shotRebound' and 'shotRush' are 0/1 in the CSV file
    # Convert 'shotType' into dummy variable for the model
    df = pd.get_dummies(df, columns=['shotType'], prefix='type')
    
    return df