import os
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

import torch
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

from starter.features import load_wav
from causal_features import FeatureExtractor
from validation import evaluate_predictions, run_group_kfold_cv
from model_comparison import PyTorchClassifier

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
    
    n_eng = len(data["english"]["y"])
    gkf = GroupKFold(n_splits=5)
    input_dim = X_comb.shape[1]
    
    # 2. Define hyperparameter configurations to test
    experiments = [
        # Random Forest Configurations
        {"model_name": "Random Forest", "params": {"n_estimators": 150, "max_depth": 3, "class_weight": "balanced", "random_state": 42}, "clf_class": RandomForestClassifier},
        {"model_name": "Random Forest", "params": {"n_estimators": 150, "max_depth": 5, "class_weight": "balanced", "random_state": 42}, "clf_class": RandomForestClassifier},
        {"model_name": "Random Forest", "params": {"n_estimators": 200, "max_depth": 6, "class_weight": "balanced", "random_state": 42}, "clf_class": RandomForestClassifier},
        
        # Extra Trees Configurations
        {"model_name": "Extra Trees", "params": {"n_estimators": 150, "max_depth": 3, "class_weight": "balanced", "random_state": 42}, "clf_class": ExtraTreesClassifier},
        {"model_name": "Extra Trees", "params": {"n_estimators": 150, "max_depth": 5, "class_weight": "balanced", "random_state": 42}, "clf_class": ExtraTreesClassifier},
        {"model_name": "Extra Trees", "params": {"n_estimators": 200, "max_depth": 6, "class_weight": "balanced", "random_state": 42}, "clf_class": ExtraTreesClassifier},
        
        # Gradient Boosting Configurations
        {"model_name": "Gradient Boosting", "params": {"n_estimators": 50, "max_depth": 2, "learning_rate": 0.05, "random_state": 42}, "clf_class": GradientBoostingClassifier},
        {"model_name": "Gradient Boosting", "params": {"n_estimators": 80, "max_depth": 3, "learning_rate": 0.05, "random_state": 42}, "clf_class": GradientBoostingClassifier},
        {"model_name": "Gradient Boosting", "params": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.02, "random_state": 42}, "clf_class": GradientBoostingClassifier},
        
        # MLP Classifier Configurations
        {"model_name": "MLP Classifier", "params": {"hidden_layer_sizes": (32, 8), "alpha": 0.01, "max_iter": 500, "random_state": 42}, "clf_class": MLPClassifier},
        {"model_name": "MLP Classifier", "params": {"hidden_layer_sizes": (64, 16), "alpha": 0.1, "max_iter": 500, "random_state": 42}, "clf_class": MLPClassifier},
        {"model_name": "MLP Classifier", "params": {"hidden_layer_sizes": (128, 32), "alpha": 0.1, "max_iter": 500, "random_state": 42}, "clf_class": MLPClassifier},
        
        # PyTorch SimpleMLP Configurations
        {"model_name": "PyTorch SimpleMLP", "params": {"input_dim": input_dim, "lr": 0.001, "epochs": 80, "batch_size": 32}, "clf_class": PyTorchClassifier},
        {"model_name": "PyTorch SimpleMLP", "params": {"input_dim": input_dim, "lr": 0.003, "epochs": 60, "batch_size": 16}, "clf_class": PyTorchClassifier},
        {"model_name": "PyTorch SimpleMLP", "params": {"input_dim": input_dim, "lr": 0.005, "epochs": 80, "batch_size": 32}, "clf_class": PyTorchClassifier},
    ]
    
    tuning_results = []
    
    print("\n===== Starting Hyperparameter Tuning Sweep =====")
    for idx, exp in enumerate(experiments):
        name = exp["model_name"]
        params = exp["params"]
        clf_class = exp["clf_class"]
        
        print(f"\n[Experiment {idx+1}/15] Tuning {name} with params: {params}")
        
        clf = clf_class(**params)
        pipeline = make_pipeline(StandardScaler(), clf)
        
        oof_preds = np.zeros(len(y_comb))
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_comb, y_comb, groups_comb)):
            X_tr, y_tr = X_comb[train_idx], y_comb[train_idx]
            X_val = X_comb[val_idx]
            
            pipeline.fit(X_tr, y_tr)
            oof_preds[val_idx] = pipeline.predict_proba(X_val)[:, 1]
            
        auc = roc_auc_score(y_comb, oof_preds)
        eng_oof = oof_preds[:n_eng]
        hin_oof = oof_preds[n_eng:]
        
        eng_metrics = evaluate_predictions(data["english"]["df"], eng_oof)
        hin_metrics = evaluate_predictions(data["hindi"]["df"], hin_oof)
        
        tuning_results.append({
            "idx": idx + 1,
            "model": name,
            "params": str(params),
            "auc": auc,
            "eng_delay": eng_metrics["latency"] * 1000,
            "hin_delay": hin_metrics["latency"] * 1000,
            "pipeline": pipeline
        })
        
        print(f" -> Result: OOF AUC={auc:.4f} | EngDelay={eng_metrics['latency']*1000:.0f}ms | HinDelay={hin_metrics['latency']*1000:.0f}ms")

    # Find the best configuration (maximizing OOF AUC)
    tuning_df = pd.DataFrame(tuning_results)
    best_idx = tuning_df["auc"].idxmax()
    best_exp = tuning_results[best_idx]
    
    print("\n===== TUNING SWEEP COMPLETE =====")
    print(f"Best Tuned Model Configuration among target set:")
    print(f" - Model      : {best_exp['model']}")
    print(f" - Params     : {best_exp['params']}")
    print(f" - OOF AUC    : {best_exp['auc']:.4f}")
    print(f" - Eng Delay  : {best_exp['eng_delay']:.0f} ms")
    print(f" - Hin Delay  : {best_exp['hin_delay']:.0f} ms")
    
    # Save the best model configuration trained on all data
    best_pipeline = best_exp["pipeline"]
    print("\nFitting best tuned pipeline on full data...")
    best_pipeline.fit(X_comb, y_comb)
    
    save_path = os.path.join(base_dir, "best_tuned_model.joblib")
    print(f"Saving best tuned model pipeline to {save_path}...")
    joblib.dump(best_pipeline, save_path)
    
    # Save sweep details to CSV
    tuning_df.drop("pipeline", axis=1).to_csv(os.path.join(base_dir, "outputs", "analysis", "hyperparameter_tuning.csv"), index=False)
    print("\nSaved tuning log to outputs/analysis/hyperparameter_tuning.csv")

if __name__ == "__main__":
    main()
