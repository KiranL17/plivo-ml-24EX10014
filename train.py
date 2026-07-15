import os
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from starter.features import load_wav
from feature_extraction import extract_robust_features

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    languages = ["english", "hindi"]
    
    print("===== Starting Model Training Pipeline =====")
    
    X_list = []
    y_list = []
    
    for lang in languages:
        print(f"\nProcessing {lang.upper()} dataset...")
        data_dir = os.path.join(base_dir, "eot_data", lang)
        labels_path = os.path.join(data_dir, "labels.csv")
        
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Missing labels.csv in {data_dir}")
            
        labels_df = pd.read_csv(labels_path)
        
        cache = {}
        for idx, r in labels_df.iterrows():
            path = os.path.join(data_dir, r["audio_file"])
            if path not in cache:
                cache[path] = load_wav(path)
            x, sr = cache[path]
            
            # Extract 110 features
            feat = extract_robust_features(x, sr, float(r["pause_start"]))
            X_list.append(feat)
            y_list.append(1 if r["label"] == "eot" else 0)
            
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"\nCombined dataset shape: {X.shape}")
    print(f"Class counts: EOT = {np.sum(y == 1)}, Hold = {np.sum(y == 0)}")
    
    # Define our best generalizing classifier pipeline
    print("\nFitting model pipeline (StandardScaler + RBF SVC)...")
    clf = SVC(probability=True, class_weight="balanced", C=1.0, kernel="rbf", random_state=42)
    pipeline = make_pipeline(StandardScaler(), clf)
    pipeline.fit(X, y)
    
    # Save the pipeline
    model_path = os.path.join(base_dir, "model.joblib")
    print(f"Saving trained model pipeline to {model_path}...")
    joblib.dump(pipeline, model_path)
    
    print("\nModel training completed successfully!")

if __name__ == "__main__":
    main()
