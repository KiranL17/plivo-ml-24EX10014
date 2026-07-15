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
from sklearn.model_selection import GroupKFold

from starter.features import load_wav
from causal_features import FeatureExtractor
from validation import evaluate_predictions

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    languages = ["english", "hindi"]
    
    extractor = FeatureExtractor()
    print(f"Feature names to extract: {len(extractor.feature_names)}")
    
    # 1. Extract features
    data = {}
    for lang in languages:
        print(f"\nExtracting features for {lang.upper()} using FeatureExtractor...")
        data_dir = os.path.join(base_dir, "eot_data", lang)
        labels_df = pd.read_csv(os.path.join(data_dir, "labels.csv"))
        
        cache = {}
        X, y, groups = [], [], []
        for idx, r in labels_df.iterrows():
            path = os.path.join(data_dir, r["audio_file"])
            if path not in cache:
                cache[path] = load_wav(path)
            x, sr = cache[path]
            
            # Use the modular FeatureExtractor class
            feat = extractor.extract_features(x, sr, float(r["pause_start"]))
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
    
    print(f"\nCombined dataset shape: {X_comb.shape}")
    
    # 2. Evaluate Models
    classifiers = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", C=0.05),
        "SVC": SVC(probability=True, class_weight="balanced", C=1.0, kernel="rbf", random_state=42),
    }
    
    gkf = GroupKFold(n_splits=5)
    
    for name, clf in classifiers.items():
        print(f"\n--- {name} ---")
        pipeline = make_pipeline(StandardScaler(), clf)
        
        # OOF CV Evaluation
        oof_preds = np.zeros(len(y_comb))
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_comb, y_comb, groups_comb)):
            pipeline.fit(X_comb[train_idx], y_comb[train_idx])
            oof_preds[val_idx] = pipeline.predict_proba(X_comb[val_idx])[:, 1]
            
        n_eng = len(data["english"]["y"])
        eng_oof = oof_preds[:n_eng]
        hin_oof = oof_preds[n_eng:]
        
        eng_oof_metrics = evaluate_predictions(data["english"]["df"], eng_oof)
        hin_oof_metrics = evaluate_predictions(data["hindi"]["df"], hin_oof)
        print(f" [OOF CV] English: Delay={eng_oof_metrics['latency']*1000:.0f} ms | Cutoff={eng_oof_metrics['cutoff']*100:.1f}% | AUC={eng_oof_metrics['auc']:.3f}")
        print(f" [OOF CV] Hindi:   Delay={hin_oof_metrics['latency']*1000:.0f} ms | Cutoff={hin_oof_metrics['cutoff']*100:.1f}% | AUC={hin_oof_metrics['auc']:.3f}")
        
        # Full Fit (Train) Evaluation
        pipeline.fit(X_comb, y_comb)
        train_preds = pipeline.predict_proba(X_comb)[:, 1]
        
        eng_train = train_preds[:n_eng]
        hin_train = train_preds[n_eng:]
        
        eng_train_metrics = evaluate_predictions(data["english"]["df"], eng_train)
        hin_train_metrics = evaluate_predictions(data["hindi"]["df"], hin_train)
        print(f" [FULL FIT] English: Delay={eng_train_metrics['latency']*1000:.0f} ms | Cutoff={eng_train_metrics['cutoff']*100:.1f}% | AUC={eng_train_metrics['auc']:.3f} | t={eng_train_metrics['threshold']}, d={eng_train_metrics['delay']*1000:.0f}ms")
        print(f" [FULL FIT] Hindi:   Delay={hin_train_metrics['latency']*1000:.0f} ms | Cutoff={hin_train_metrics['cutoff']*100:.1f}% | AUC={hin_train_metrics['auc']:.3f} | t={hin_train_metrics['threshold']}, d={hin_train_metrics['delay']*1000:.0f}ms")

if __name__ == "__main__":
    main()
