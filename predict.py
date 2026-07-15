import os
import csv
import argparse
import logging
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from starter.features import load_wav
from improved_features import ImprovedFeatureExtractor

# Configure logging to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Path to the dataset directory containing labels.csv and audio files")
    ap.add_argument("--out", default="predictions.csv", help="Path to write predictions.csv")
    args = ap.parse_args()
    
    logging.info("Starting inference pipeline...")
    
    # 1. Check data directory validity
    if not os.path.exists(args.data_dir):
        logging.error(f"Data directory does not exist: {args.data_dir}")
        return
        
    labels_csv = os.path.join(args.data_dir, "labels.csv")
    if not os.path.exists(labels_csv):
        logging.error(f"Missing labels.csv in dataset directory: {args.data_dir}")
        return
        
    # 2. Load model bundle
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "model.joblib")
    if not os.path.exists(model_path):
        logging.error(f"Trained model artifact bundle not found at: {model_path}. Run train.py first.")
        return
        
    logging.info(f"Loading optimized model bundle from: {model_path}")
    try:
        model_bundle = joblib.load(model_path)
        pipeline = model_bundle["pipeline"]
        selected_indices = model_bundle["selected_indices"]
    except Exception as e:
        logging.error(f"Failed to load or parse model bundle: {e}")
        return
        
    # 3. Read labels.csv
    logging.info(f"Reading labels from: {labels_csv}")
    try:
        with open(labels_csv, mode='r', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        logging.error(f"Failed to read CSV: {e}")
        return
        
    # Initialize improved feature extractor (130 features)
    extractor = ImprovedFeatureExtractor()
    
    # 4. Predict probabilities for each pause
    predictions = []
    audio_cache = {}
    
    logging.info(f"Processing {len(rows)} pauses in dataset...")
    for idx, r in enumerate(rows):
        turn_id = r.get("turn_id")
        pause_index = r.get("pause_index")
        audio_file = r.get("audio_file")
        pause_start_str = r.get("pause_start")
        
        # Validate row values
        if not turn_id or not pause_index or not audio_file or not pause_start_str:
            logging.warning(f"Row {idx+1} is missing fields. Skipping: {r}")
            continue
            
        try:
            pause_start = float(pause_start_str)
        except ValueError:
            logging.warning(f"Row {idx+1} has invalid pause_start '{pause_start_str}'. Skipping.")
            continue
            
        wav_path = os.path.join(args.data_dir, audio_file)
        
        # Load audio with error handling
        if wav_path not in audio_cache:
            if not os.path.exists(wav_path):
                logging.warning(f"Audio file missing: {wav_path}. Defaulting p_eot to 0.5000.")
                predictions.append({
                    "turn_id": turn_id,
                    "pause_index": int(pause_index),
                    "p_eot": 0.5000
                })
                continue
            try:
                audio_cache[wav_path] = load_wav(wav_path)
            except Exception as e:
                logging.warning(f"Failed to load audio {wav_path}: {e}. Defaulting p_eot to 0.5000.")
                predictions.append({
                    "turn_id": turn_id,
                    "pause_index": int(pause_index),
                    "p_eot": 0.5000
                })
                continue
                
        x, sr = audio_cache[wav_path]
        
        # Extract features with error safety
        try:
            raw_feat = extractor.extract_features(x, sr, pause_start)
            # Apply feature selection slice (selects top 80 features)
            feat = raw_feat[selected_indices]
            # Predict prob
            prob = pipeline.predict_proba(feat.reshape(1, -1))[0, 1]
        except Exception as e:
            logging.warning(f"Feature extraction or prediction failed for {audio_file} at {pause_start}s: {e}. Defaulting p_eot to 0.5000.")
            prob = 0.5000
            
        predictions.append({
            "turn_id": turn_id,
            "pause_index": int(pause_index),
            "p_eot": float(prob)
        })
        
    # 5. Write predictions.csv
    logging.info(f"Writing predictions to: {args.out}")
    try:
        # Create parent directories if they do not exist
        out_dir = os.path.dirname(args.out)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
            
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["turn_id", "pause_index", "p_eot"])
            w.writeheader()
            for p in predictions:
                w.writerow({
                    "turn_id": p["turn_id"],
                    "pause_index": p["pause_index"],
                    "p_eot": f"{p['p_eot']:.4f}"
                })
        logging.info("Inference completed successfully!")
    except Exception as e:
        logging.error(f"Failed to write output CSV file: {e}")

if __name__ == "__main__":
    main()
