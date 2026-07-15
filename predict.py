import os
import csv
import argparse
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from starter.features import load_wav
from improved_features import ImprovedFeatureExtractor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()
    
    # 1. Load the pre-trained model artifact bundle
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Please run train.py first.")
        
    print(f"Loading optimized model bundle from {model_path}...")
    model_bundle = joblib.load(model_path)
    
    pipeline = model_bundle["pipeline"]
    selected_indices = model_bundle["selected_indices"]
    
    # 2. Read input labels.csv from data_dir
    labels_csv = os.path.join(args.data_dir, "labels.csv")
    if not os.path.exists(labels_csv):
        raise FileNotFoundError(f"Missing labels.csv in {args.data_dir}")
        
    print(f"Reading labels from {labels_csv}...")
    rows = list(csv.DictReader(open(labels_csv)))
    
    # Use the ImprovedFeatureExtractor class to align with 130 features
    extractor = ImprovedFeatureExtractor()
    
    # 3. Predict for each pause
    cache = {}
    predictions = []
    
    print("Extracting improved features and running inference...")
    for r in rows:
        path = os.path.join(args.data_dir, r["audio_file"])
        if path not in cache:
            cache[path] = load_wav(path)
        x, sr = cache[path]
        
        # Extract 130 features
        raw_feat = extractor.extract_features(x, sr, float(r["pause_start"]))
        
        # Apply the feature selection mask (selects top 80 features)
        feat = raw_feat[selected_indices]
        
        # Predict probability for class 1 (EOT)
        prob = pipeline.predict_proba(feat.reshape(1, -1))[0, 1]
        
        predictions.append({
            "turn_id": r["turn_id"],
            "pause_index": int(r["pause_index"]),
            "p_eot": float(prob)
        })
        
    # 4. Write predictions to out
    print(f"Writing predictions to {args.out}...")
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["turn_id", "pause_index", "p_eot"])
        w.writeheader()
        for p in predictions:
            w.writerow({
                "turn_id": p["turn_id"],
                "pause_index": p["pause_index"],
                "p_eot": f"{p['p_eot']:.4f}"
            })
            
    print("Inference completed successfully!")

if __name__ == "__main__":
    main()
