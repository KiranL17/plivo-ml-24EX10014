import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from starter.features import load_wav
from feature_extraction import extract_robust_features
from validation import run_group_kfold_cv

def evaluate_models(X, y, groups, labels_df):
    classifiers = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", C=0.1),
        "SVC": SVC(probability=True, class_weight="balanced", C=1.0, kernel="rbf", random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5, class_weight="balanced", random_state=42),
        "LightGBM": LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, class_weight="balanced", random_state=42, verbose=-1)
    }
    
    results = {}
    for name, clf in classifiers.items():
        # Wrap in a pipeline with standard scaling
        pipeline = make_pipeline(StandardScaler(), clf)
        
        # Run 5-fold cross-validation
        metrics, _ = run_group_kfold_cv(pipeline, X, y, groups, labels_df, n_splits=5)
        results[name] = metrics
        print(f" - {name:18}: Delay={metrics['latency']*1000:4.0f} ms | Cutoff={metrics['cutoff']*100:4.1f}% | AUC={metrics['auc']:.3f} | FoldAUC={metrics['mean_fold_auc']:.3f}")
        
    return results

def main():
    base_dir = "C:/Users/lakka/OneDrive/Desktop/plivo assignment"
    languages = ["english", "hindi"]
    
    for lang in languages:
        print(f"\n===== Feature Extraction & Model Evaluation for {lang.upper()} =====")
        data_dir = os.path.join(base_dir, "eot_data", lang)
        labels_df = pd.read_csv(os.path.join(data_dir, "labels.csv"))
        
        # Load and extract features
        cache = {}
        X, y, groups = [], [], []
        
        print("Extracting features from audio files...")
        for idx, r in labels_df.iterrows():
            path = os.path.join(data_dir, r["audio_file"])
            if path not in cache:
                cache[path] = load_wav(path)
            x, sr = cache[path]
            
            # Extract robust features (causal, normalized)
            feat = extract_robust_features(x, sr, float(r["pause_start"]))
            X.append(feat)
            y.append(1 if r["label"] == "eot" else 0)
            groups.append(r["turn_id"])
            
        X = np.array(X)
        y = np.array(y)
        groups = np.array(groups)
        
        print(f"Feature matrix shape: {X.shape}")
        
        # Evaluate different models
        print("Evaluating models...")
        evaluate_models(X, y, groups, labels_df)

if __name__ == "__main__":
    main()
