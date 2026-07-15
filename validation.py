import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from typing import List, Dict, Tuple, Any
from sklearn.base import BaseEstimator

TIMEOUT_S: float = 1.6
THRESHOLDS: np.ndarray = np.round(np.arange(0.05, 1.0, 0.05), 3)
DELAYS: np.ndarray = np.round(np.arange(0.10, 1.65, 0.05), 3)

def evaluate_predictions(pauses_df: pd.DataFrame, predictions: np.ndarray, budget: float = 0.05) -> Dict[str, Any]:
    """
    Given a DataFrame of pauses and predictions (p_eot), sweep thresholds and delays 
    to find the operating point that minimizes mean delay subject to the budget 
    constraint on interrupted turns.

    Args:
        pauses_df (pd.DataFrame): DataFrame of pauses with turn_id, pause_start, pause_end, label.
        predictions (np.ndarray): 1D array of EOT probabilities.
        budget (float): Strict upper bound budget of interrupted turns (default 0.05).

    Returns:
        Dict[str, Any]: Best operational metrics (latency, cutoff, threshold, delay, auc).
    """
    pauses: List[Dict[str, Any]] = []
    for idx, row in pauses_df.iterrows():
        pauses.append({
            "turn_id": row["turn_id"],
            "dur": float(row["pause_end"]) - float(row["pause_start"]),
            "label": row["label"],
            "p": float(predictions[idx])
        })
        
    best = None
    for t in THRESHOLDS:
        for d in DELAYS:
            turns_cut = set()
            turn_ids = set()
            latencies = []
            
            for pz in pauses:
                turn_ids.add(pz["turn_id"])
                fires = pz["p"] >= t
                if pz["label"] == "hold":
                    if fires and d < pz["dur"]:
                        turns_cut.add(pz["turn_id"])
                else:  # true end of turn
                    latencies.append(d if fires else TIMEOUT_S)
            
            cutoff_rate = len(turns_cut) / max(1, len(turn_ids))
            mean_lat = float(np.mean(latencies)) if latencies else TIMEOUT_S
            
            if cutoff_rate <= budget and (best is None or mean_lat < best["latency"]):
                best = {"latency": mean_lat, "cutoff": cutoff_rate, "threshold": t, "delay": d}
                
    if best is None:
        best = {"latency": TIMEOUT_S, "cutoff": 0.0, "threshold": 1.0, "delay": TIMEOUT_S}
        
    # Calculate AUC
    y = np.array([1 if p["label"] == "eot" else 0 for p in pauses])
    s = np.array([p["p"] for p in pauses])
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), len(y) - y.sum()
    auc = ((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")
    best["auc"] = float(auc)
    
    return best

def run_group_kfold_cv(
    clf: BaseEstimator, 
    X: np.ndarray, 
    y: np.ndarray, 
    groups: np.ndarray, 
    labels_df: pd.DataFrame, 
    n_splits: int = 5
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Run GroupKFold cross-validation, get out-of-fold predictions,
    and compute the out-of-fold validation scores.

    Args:
        clf (BaseEstimator): Scikit-learn estimator classifier to evaluate.
        X (np.ndarray): 2D feature array.
        y (np.ndarray): 1D target label array.
        groups (np.ndarray): 1D array representing grouping keys (e.g. speaker/turn-id).
        labels_df (pd.DataFrame): DataFrame of pauses for evaluation alignment.
        n_splits (int): Cross-validation folds count (default 5).

    Returns:
        Tuple[Dict[str, Any], np.ndarray]: Dict of scoring metrics and out-of-fold predictions.
    """
    gkf = GroupKFold(n_splits=n_splits)
    oof_preds = np.zeros(len(y))
    fold_aucs = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        clf.fit(X_tr, y_tr)
        
        # Predict probability for class 1 (EOT)
        preds = clf.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = preds
        
        # Calculate fold AUC
        y_val_arr = np.array(y_val)
        n1, n0 = y_val_arr.sum(), len(y_val_arr) - y_val_arr.sum()
        if n1 > 0 and n0 > 0:
            from sklearn.metrics import roc_auc_score
            fold_aucs.append(roc_auc_score(y_val_arr, preds))
        else:
            fold_aucs.append(0.5)
            
    # Calculate out-of-fold score
    metrics = evaluate_predictions(labels_df, oof_preds)
    metrics["mean_fold_auc"] = np.mean(fold_aucs)
    
    return metrics, oof_preds
