import os
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from starter.features import load_wav
from improved_features import ImprovedFeatureExtractor

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    languages = ["english", "hindi"]
    
    print("===== Starting Model Training Pipeline (Optimized) =====")
    extractor = ImprovedFeatureExtractor()
    
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
            
            # Extract 130 features
            feat = extractor.extract_features(x, sr, float(r["pause_start"]))
            X_list.append(feat)
            y_list.append(1 if r["label"] == "eot" else 0)
            
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"\nExtracted dataset shape: {X.shape}")
    
    # Run feature selection via Random Forest
    print("Selecting top 80 features via Random Forest importances...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X, y)
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    n_features_to_select = 80
    selected_indices = indices[:n_features_to_select]
    selected_indices = np.sort(selected_indices)
    selected_names = [extractor.feature_names[i] for i in selected_indices]
    
    X_selected = X[:, selected_indices]
    print(f"Reduced dataset shape: {X_selected.shape}")
    
    # Fit the final regularized pipeline on selected features
    print("Fitting model pipeline (StandardScaler + RBF SVC)...")
    clf = SVC(probability=True, class_weight="balanced", C=1.0, kernel="rbf", random_state=42)
    pipeline = make_pipeline(StandardScaler(), clf)
    pipeline.fit(X_selected, y)
    
    # Save the pipeline bundle
    model_path = os.path.join(base_dir, "model.joblib")
    print(f"Saving optimized model bundle to {model_path}...")
    joblib.dump({
        "pipeline": pipeline,
        "selected_indices": selected_indices,
        "feature_names": selected_names
    }, model_path)
    
    print("\nModel training completed successfully!")

if __name__ == "__main__":
    main()
