import sys
sys.path.append("starter")

from score import score

def main():
    eng_labels = "eot_data/english/labels.csv"
    eng_preds = "eot_data/english/predictions.csv"
    
    hin_labels = "eot_data/hindi/labels.csv"
    hin_preds = "eot_data/hindi/predictions.csv"
    
    # We will temporarily write a combined labels.csv and predictions.csv
    import csv
    
    combined_labels = []
    combined_preds = []
    
    # Load English
    with open(eng_labels) as f:
        combined_labels.extend(list(csv.DictReader(f)))
    with open(eng_preds) as f:
        combined_preds.extend(list(csv.DictReader(f)))
        
    # Load Hindi (making turn_ids unique to prevent overlaps, though they are already unique e.g. hi__ vs en__)
    with open(hin_labels) as f:
        combined_labels.extend(list(csv.DictReader(f)))
    with open(hin_preds) as f:
        combined_preds.extend(list(csv.DictReader(f)))
        
    # Write combined temp files
    temp_labels = "temp_combined_labels.csv"
    temp_preds = "temp_combined_preds.csv"
    
    with open(temp_labels, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=combined_labels[0].keys())
        w.writeheader()
        w.writerows(combined_labels)
        
    with open(temp_preds, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=combined_preds[0].keys())
        w.writeheader()
        w.writerows(combined_preds)
        
    # Score
    r = score(temp_labels, temp_preds, budget=0.05)
    print("\n===== OVERALL COMBINED DATASET SCORE =====")
    print(f"Total turns:  {r['n_turns']}")
    print(f"Total pauses: {r['n_pauses']}")
    print(f"Overall AUC:  {r['auc']:.3f}")
    print(f"BEST @ <= 5% interrupted turns:")
    print(f"  mean response delay : {r['latency']*1000:.0f} ms   <-- overall score")
    print(f"  interrupted turns   : {r['cutoff']*100:.1f}%")
    print(f"  operating point     : threshold={r['threshold']}, delay={r['delay']*1000:.0f} ms")
    
    # Clean up temp files
    import os
    if os.path.exists(temp_labels):
        os.remove(temp_labels)
    if os.path.exists(temp_preds):
        os.remove(temp_preds)

if __name__ == "__main__":
    main()
