import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from starter.features import load_wav, speech_before, frame_energy_db, f0_contour
from validation import run_group_kfold_cv

def extract_features(x, sr, pause_start):
    seg = speech_before(x, sr, pause_start, window_s=1.5)
    if len(seg) < sr // 10:
        return np.zeros(3, dtype=np.float32)
    e = frame_energy_db(seg, sr)
    f0 = f0_contour(seg, sr)
    voiced = f0[f0 > 0]
    return np.array([
        e[-5:].mean(),                       # energy right before the pause
        voiced[-3:].mean() if len(voiced) >= 3 else 0.0,   # final pitch
        len(seg) / sr,                       # how much speech context we had
    ], dtype=np.float32)

def main():
    base_dir = "C:/Users/lakka/OneDrive/Desktop/plivo assignment"
    languages = ["english", "hindi"]
    
    for lang in languages:
        print(f"\n===== Profiling {lang.upper()} Dataset =====")
        data_dir = os.path.join(base_dir, "eot_data", lang)
        labels_df = pd.read_csv(os.path.join(data_dir, "labels.csv"))
        
        # Load and extract features
        cache = {}
        X, y, groups = [], [], []
        for idx, r in labels_df.iterrows():
            path = os.path.join(data_dir, r["audio_file"])
            if path not in cache:
                cache[path] = load_wav(path)
            x, sr = cache[path]
            X.append(extract_features(x, sr, float(r["pause_start"])))
            y.append(1 if r["label"] == "eot" else 0)
            groups.append(r["turn_id"])
            
        X = np.array(X)
        y = np.array(y)
        
        # Train baseline classifier (LogisticRegression)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        
        # Run 5-fold cross-validation
        metrics, oof_preds = run_group_kfold_cv(clf, X, y, groups, labels_df, n_splits=5)
        
        print(f"OOF Validation Metrics (5-fold CV):")
        print(f" - mean response delay : {metrics['latency']*1000:.0f} ms")
        print(f" - interrupted turns   : {metrics['cutoff']*100:.1f}%")
        print(f" - operating point     : threshold={metrics['threshold']}, delay={metrics['delay']*1000:.0f} ms")
        print(f" - OOF AUC             : {metrics['auc']:.3f}")
        print(f" - Mean Fold AUC       : {metrics['mean_fold_auc']:.3f}")

if __name__ == "__main__":
    main()
