import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold

from starter.features import load_wav
from feature_extraction import extract_robust_features
from validation import evaluate_predictions

def main():
    base_dir = "C:/Users/lakka/OneDrive/Desktop/plivo assignment"
    languages = ["english", "hindi"]
    
    # 1. Extract features
    data = {}
    for lang in languages:
        print(f"Extracting features for {lang.upper()}...")
        data_dir = os.path.join(base_dir, "eot_data", lang)
        labels_df = pd.read_csv(os.path.join(data_dir, "labels.csv"))
        
        cache = {}
        X, y, groups = [], [], []
        for idx, r in labels_df.iterrows():
            path = os.path.join(data_dir, r["audio_file"])
            if path not in cache:
                cache[path] = load_wav(path)
            x, sr = cache[path]
            
            feat = extract_robust_features(x, sr, float(r["pause_start"]))
            X.append(feat)
            y.append(1 if r["label"] == "eot" else 0)
            groups.append(f"{lang}_{r['turn_id']}")
            
        data[lang] = {
            "X": np.array(X),
            "y": np.array(y),
            "groups": np.array(groups),
            "df": labels_df
        }
    
    X_comb = np.concatenate([data["english"]["X"], data["hindi"]["X"]], axis=0)
    y_comb = np.concatenate([data["english"]["y"], data["hindi"]["y"]], axis=0)
    groups_comb = np.concatenate([data["english"]["groups"], data["hindi"]["groups"]], axis=0)
    
    n_eng = len(data["english"]["y"])
    gkf = GroupKFold(n_splits=5)
    
    # Grid of hyperparameters to tune
    # 1. SVC Grid
    svc_params = [
        {"C": 0.1, "kernel": "rbf"},
        {"C": 0.5, "kernel": "rbf"},
        {"C": 1.0, "kernel": "rbf"},
        {"C": 2.0, "kernel": "rbf"},
        {"C": 5.0, "kernel": "rbf"},
        {"C": 0.5, "kernel": "linear"},
        {"C": 1.0, "kernel": "linear"},
    ]
    
    # 2. Logistic Regression Grid
    lr_params = [
        {"C": 0.01, "penalty": "l2"},
        {"C": 0.05, "penalty": "l2"},
        {"C": 0.1, "penalty": "l2"},
        {"C": 0.5, "penalty": "l2"},
        {"C": 1.0, "penalty": "l2"},
    ]
    
    # 3. Random Forest Grid
    rf_params = [
        {"n_estimators": 100, "max_depth": 3},
        {"n_estimators": 150, "max_depth": 4},
        {"n_estimators": 200, "max_depth": 4},
        {"n_estimators": 200, "max_depth": 5},
    ]
    
    print("\n===== Tuning SVC =====")
    for p in svc_params:
        clf = SVC(probability=True, class_weight="balanced", random_state=42, **p)
        pipeline = make_pipeline(StandardScaler(), clf)
        
        oof_preds = np.zeros(len(y_comb))
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_comb, y_comb, groups_comb)):
            pipeline.fit(X_comb[train_idx], y_comb[train_idx])
            oof_preds[val_idx] = pipeline.predict_proba(X_comb[val_idx])[:, 1]
            
        eng_oof = oof_preds[:n_eng]
        hin_oof = oof_preds[n_eng:]
        
        eng_metrics = evaluate_predictions(data["english"]["df"], eng_oof)
        hin_metrics = evaluate_predictions(data["hindi"]["df"], hin_oof)
        
        print(f"SVC C={p['C']} {p['kernel']:6}: EngDelay={eng_metrics['latency']*1000:4.0f} ms (AUC={eng_metrics['auc']:.3f}) | HinDelay={hin_metrics['latency']*1000:4.0f} ms (AUC={hin_metrics['auc']:.3f})")
        
    print("\n===== Tuning LogisticRegression =====")
    for p in lr_params:
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42, **p)
        pipeline = make_pipeline(StandardScaler(), clf)
        
        oof_preds = np.zeros(len(y_comb))
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_comb, y_comb, groups_comb)):
            pipeline.fit(X_comb[train_idx], y_comb[train_idx])
            oof_preds[val_idx] = pipeline.predict_proba(X_comb[val_idx])[:, 1]
            
        eng_oof = oof_preds[:n_eng]
        hin_oof = oof_preds[n_eng:]
        
        eng_metrics = evaluate_predictions(data["english"]["df"], eng_oof)
        hin_metrics = evaluate_predictions(data["hindi"]["df"], hin_oof)
        
        print(f"LR C={p['C']}: EngDelay={eng_metrics['latency']*1000:4.0f} ms (AUC={eng_metrics['auc']:.3f}) | HinDelay={hin_metrics['latency']*1000:4.0f} ms (AUC={hin_metrics['auc']:.3f})")

if __name__ == "__main__":
    main()
