import os
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

from starter.features import load_wav
from improved_features import ImprovedFeatureExtractor
from validation import evaluate_predictions

def main():
    base_dir = "C:/Users/lakka/OneDrive/Desktop/plivo assignment"
    output_dir = os.path.join(base_dir, "outputs", "analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    languages = ["english", "hindi"]
    extractor = ImprovedFeatureExtractor()
    print(f"Total features extracted: {len(extractor.feature_names)}")
    
    # 1. Load data & extract features
    data = {}
    for lang in languages:
        print(f"Extracting improved features for {lang.upper()}...")
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
    
    # 2. Feature Importance & Selection using Random Forest
    print("\nCalculating feature importances via Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_comb, y_comb)
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Keep top 80 features
    n_features_to_select = 80
    selected_indices = indices[:n_features_to_select]
    selected_indices = np.sort(selected_indices) # Keep original order for clean indexing
    selected_names = [extractor.feature_names[i] for i in selected_indices]
    
    print(f"\nSelected top {n_features_to_select} features.")
    print("Top 15 Most Important Features:")
    for rank in range(15):
        idx = indices[rank]
        print(f" - Rank {rank+1:2d}: {extractor.feature_names[idx]} (importance = {importances[idx]:.4f})")
        
    # Check if new features were selected
    new_feats = ["ratio_last_voiced_vs_average_voiced_duration", "f0_final_voiced_stability", "flux_ratio_last_200ms_vs_1s", "zcr_ratio_last_200ms_vs_1s"]
    print("\nSelection status of new error-informed features:")
    for nf in new_feats:
        is_selected = nf in selected_names
        importance_score = importances[extractor.feature_names.index(nf)]
        print(f" - {nf}: Selected={is_selected} | Importance={importance_score:.4f}")
        
    # Slice dataset
    X_selected = X_comb[:, selected_indices]
    
    # 3. Generate SHAP Summary Plot
    print("\nGenerating SHAP explanations...")
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_comb)
    
    # For classification, shap_values is a list of arrays (one per class) or 3D array.
    # Typically [class_0_shap, class_1_shap]. We use class 1 (EOT).
    if isinstance(shap_values, list):
        class_1_shap = shap_values[1]
    else:
        # For newer SHAP versions, TreeExplainer returns a single Explanation object or array
        class_1_shap = shap_values[:, :, 1] if len(shap_values.shape) == 3 else shap_values
        
    fig, ax = plt.subplots(figsize=(10, 8))
    # Standard SHAP summary plot
    shap.summary_plot(class_1_shap, X_comb, feature_names=extractor.feature_names, max_display=15, show=False)
    plt.title("SHAP Feature Importance (Class 1: EOT)", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    shap_path = os.path.join(output_dir, "shap_summary_plot.png")
    fig.savefig(shap_path, dpi=150)
    plt.close(fig)
    print(f"Saved SHAP summary plot at: {shap_path}")
    
    # 4. Out-of-Fold CV Evaluation with Selected Features
    print("\n===== Running 5-Fold CV with Selected Features =====")
    clf = SVC(probability=True, class_weight="balanced", C=1.0, kernel="rbf", random_state=42)
    pipeline = make_pipeline(StandardScaler(), clf)
    
    gkf = GroupKFold(n_splits=5)
    oof_preds = np.zeros(len(y_comb))
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_selected, y_comb, groups_comb)):
        pipeline.fit(X_selected[train_idx], y_comb[train_idx])
        oof_preds[val_idx] = pipeline.predict_proba(X_selected[val_idx])[:, 1]
        
    n_eng = len(data["english"]["y"])
    eng_oof = oof_preds[:n_eng]
    hin_oof = oof_preds[n_eng:]
    
    eng_metrics = evaluate_predictions(data["english"]["df"], eng_oof)
    hin_metrics = evaluate_predictions(data["hindi"]["df"], hin_oof)
    
    print("\n[Out-Of-Fold CV Scores with Selected Features]:")
    print(f" - English Delay : {eng_metrics['latency']*1000:.0f} ms | Cutoff={eng_metrics['cutoff']*100:.1f}% | AUC={eng_metrics['auc']:.4f}")
    print(f" - Hindi Delay   : {hin_metrics['latency']*1000:.0f} ms | Cutoff={hin_metrics['cutoff']*100:.1f}% | AUC={hin_metrics['auc']:.4f}")
    
    # 5. Fit final pipeline on full data and serialize
    print("\nTraining final SVC pipeline on all data...")
    pipeline.fit(X_selected, y_comb)
    
    # Evaluate full fit score
    train_preds = pipeline.predict_proba(X_selected)[:, 1]
    eng_train = evaluate_predictions(data["english"]["df"], train_preds[:n_eng])
    hin_train = evaluate_predictions(data["hindi"]["df"], train_preds[n_eng:])
    
    print("\n[Full Fit Training Set Scores with Selected Features]:")
    print(f" - English Delay : {eng_train['latency']*1000:.0f} ms | Cutoff={eng_train['cutoff']*100:.1f}% | AUC={eng_train['auc']:.4f}")
    print(f" - Hindi Delay   : {hin_train['latency']*1000:.0f} ms | Cutoff={hin_train['cutoff']*100:.1f}% | AUC={hin_train['auc']:.4f}")
    
    # Save the model artifact along with selection indices and names
    model_path = os.path.join(base_dir, "model.joblib")
    print(f"\nSaving final optimized model artifact bundle to {model_path}...")
    joblib.dump({
        "pipeline": pipeline,
        "selected_indices": selected_indices,
        "feature_names": selected_names
    }, model_path)
    print("Optimization and retraining completed successfully!")

if __name__ == "__main__":
    main()
