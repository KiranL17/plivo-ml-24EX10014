import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from starter.features import load_wav
from causal_features import FeatureExtractor
from validation import evaluate_predictions, run_group_kfold_cv

# 1. PyTorch Network Definition
class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
    def forward(self, x):
        return self.net(x)

class PyTorchClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim=126, epochs=80, lr=0.003, batch_size=32):
        self.input_dim = input_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.model = None

    def fit(self, X, y):
        # Fix random seeds for reproducibility
        torch.manual_seed(42)
        self.classes_ = np.unique(y)
        self.model = SimpleMLP(self.input_dim)
        
        # Calculate class balance weights
        neg_count = np.sum(y == 0)
        pos_count = np.sum(y == 1)
        w = torch.tensor([1.0, neg_count / pos_count if pos_count > 0 else 1.0], dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=w)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                out = self.model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
        return self
                
    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1).numpy()
        return probs

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    languages = ["english", "hindi"]
    
    extractor = FeatureExtractor()
    
    # 2. Extract features
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
    
    # 3. Setup models dictionary
    input_dim = X_comb.shape[1]
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", C=0.05, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=4, class_weight="balanced", random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=150, max_depth=4, class_weight="balanced", random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=60, max_depth=3, learning_rate=0.05, random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=60, max_depth=3, learning_rate=0.05, class_weight="balanced", random_state=42),
        "Support Vector Machine": SVC(probability=True, class_weight="balanced", C=1.0, kernel="rbf", random_state=42),
        "MLP Classifier": MLPClassifier(hidden_layer_sizes=(64, 16), max_iter=500, alpha=0.01, random_state=42),
        "PyTorch SimpleMLP": PyTorchClassifier(input_dim=input_dim)
    }
    
    comparison_rows = []
    
    print("\n===== Starting Cross-Validation Comparisons =====")
    for name, clf in models.items():
        print(f"\nRunning CV for: {name}...")
        
        # Build evaluation pipeline (wrap in StandardScaler if not using PyTorch since we scale PyTorch inside or outside)
        pipeline = make_pipeline(StandardScaler(), clf)
        
        # Out of fold predictions
        oof_preds = np.zeros(len(y_comb))
        oof_pred_classes = np.zeros(len(y_comb))
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_comb, y_comb, groups_comb)):
            X_tr, y_tr = X_comb[train_idx], y_comb[train_idx]
            X_val = X_comb[val_idx]
            
            pipeline.fit(X_tr, y_tr)
            
            # Predict probabilities
            preds = pipeline.predict_proba(X_val)[:, 1]
            oof_preds[val_idx] = preds
            
            # Predict classes (use 0.5 threshold or model default)
            # For PyTorch we threshold at 0.5
            if name == "PyTorch SimpleMLP":
                oof_pred_classes[val_idx] = (preds >= 0.5).astype(int)
            else:
                oof_pred_classes[val_idx] = pipeline.predict(X_val)
                
        # Calculate OOF classification metrics
        auc = roc_auc_score(y_comb, oof_preds)
        prec = precision_score(y_comb, oof_pred_classes)
        rec = recall_score(y_comb, oof_pred_classes)
        f1 = f1_score(y_comb, oof_pred_classes)
        
        # Calculate Assignment scores
        eng_oof = oof_preds[:n_eng]
        hin_oof = oof_preds[n_eng:]
        
        eng_metrics = evaluate_predictions(data["english"]["df"], eng_oof)
        hin_metrics = evaluate_predictions(data["hindi"]["df"], hin_oof)
        
        comparison_rows.append({
            "Model": name,
            "OOF AUC": f"{auc:.4f}",
            "OOF Precision": f"{prec:.4f}",
            "OOF Recall": f"{rec:.4f}",
            "OOF F1-Score": f"{f1:.4f}",
            "Eng Delay (ms)": f"{eng_metrics['latency']*1000:.0f}",
            "Eng Cutoff (%)": f"{eng_metrics['cutoff']*100:.1f}",
            "Hin Delay (ms)": f"{hin_metrics['latency']*1000:.0f}",
            "Hin Cutoff (%)": f"{hin_metrics['cutoff']*100:.1f}"
        })
        
        print(f" -> AUC={auc:.4f} | EngDelay={eng_metrics['latency']*1000:.0f}ms | HinDelay={hin_metrics['latency']*1000:.0f}ms")

    # 4. Print comparison table
    df_comp = pd.DataFrame(comparison_rows)
    print("\n===== FINAL MODEL COMPARISONS (Out-Of-Fold CV) =====")
    print(df_comp.to_string(index=False))
    
    # Save comparison dataframe to csv
    df_comp.to_csv(os.path.join(base_dir, "outputs", "analysis", "model_comparisons.csv"), index=False)
    
if __name__ == "__main__":
    main()
