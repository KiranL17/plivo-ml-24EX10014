import os
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score

from starter.features import load_wav
from causal_features import FeatureExtractor
from validation import evaluate_predictions

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    languages = ["english", "hindi"]
    
    extractor = FeatureExtractor()
    
    # 1. Load data & extract features
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
    
    print(f"\nCombined Dataset Shape: {X_comb.shape}")
    
    # 2. 5-Fold GroupKFold Cross Validation for Baseline Logistic Regression
    print("\n===== Running 5-Fold CV for Logistic Regression Baseline =====")
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.05, random_state=42)
    pipeline = make_pipeline(StandardScaler(), clf)
    
    gkf = GroupKFold(n_splits=5)
    oof_preds = np.zeros(len(y_comb))
    oof_pred_classes = np.zeros(len(y_comb))
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_comb, y_comb, groups_comb)):
        X_tr, y_tr = X_comb[train_idx], y_comb[train_idx]
        X_val = X_comb[val_idx]
        
        pipeline.fit(X_tr, y_tr)
        oof_preds[val_idx] = pipeline.predict_proba(X_val)[:, 1]
        oof_pred_classes[val_idx] = pipeline.predict(X_val)
        
    # Calculate classification metrics on OOF predictions
    auc = roc_auc_score(y_comb, oof_preds)
    prec = precision_score(y_comb, oof_pred_classes)
    rec = recall_score(y_comb, oof_pred_classes)
    f1 = f1_score(y_comb, oof_pred_classes)
    
    print("\n[Out-Of-Fold CV Metrics]:")
    print(f" - ROC AUC   : {auc:.4f}")
    print(f" - Precision : {prec:.4f}")
    print(f" - Recall    : {rec:.4f}")
    print(f" - F1-Score  : {f1:.4f}")
    
    cm = confusion_matrix(y_comb, oof_pred_classes)
    print("\nConfusion Matrix (OOF):")
    print(cm)
    print(f"True Negative (Hold correctly predicted as Hold): {cm[0,0]}")
    print(f"False Positive (Hold predicted as EOT):          {cm[0,1]}  <-- RISK OF FALSE CUTOFF")
    print(f"False Negative (EOT predicted as Hold):          {cm[1,0]}  <-- RISK OF DELAYED RESPONSE")
    print(f"True Positive (EOT correctly predicted as EOT):  {cm[1,1]}")
    
    # Calculate Assignment Scores
    n_eng = len(data["english"]["y"])
    eng_metrics = evaluate_predictions(data["english"]["df"], oof_preds[:n_eng])
    hin_metrics = evaluate_predictions(data["hindi"]["df"], oof_preds[n_eng:])
    
    print("\n[Out-Of-Fold CV Assignment Scores]:")
    print(f" - English Delay : {eng_metrics['latency']*1000:.0f} ms | Cutoff={eng_metrics['cutoff']*100:.1f}%")
    print(f" - Hindi Delay   : {hin_metrics['latency']*1000:.0f} ms | Cutoff={hin_metrics['cutoff']*100:.1f}%")
    
    # 3. Fit on full data and save
    print("\nTraining Logistic Regression pipeline on full data...")
    pipeline.fit(X_comb, y_comb)
    
    # Save the pipeline
    save_path = os.path.join(base_dir, "logistic_model.joblib")
    print(f"Saving baseline model pipeline to {save_path}...")
    joblib.dump(pipeline, save_path)
    
    # Evaluate full fit score
    train_preds = pipeline.predict_proba(X_comb)[:, 1]
    eng_train = evaluate_predictions(data["english"]["df"], train_preds[:n_eng])
    hin_train = evaluate_predictions(data["hindi"]["df"], train_preds[n_eng:])
    
    print("\n[Full Fit Training Set Scores]:")
    print(f" - English Delay : {eng_train['latency']*1000:.0f} ms | Cutoff={eng_train['cutoff']*100:.1f}% | AUC={eng_train['auc']:.4f}")
    print(f" - Hindi Delay   : {hin_train['latency']*1000:.0f} ms | Cutoff={hin_train['cutoff']*100:.1f}% | AUC={hin_train['auc']:.4f}")

if __name__ == "__main__":
    main()
