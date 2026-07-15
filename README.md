# End-of-Turn (EoT) Detection for Voice Agents

This repository contains a production-grade machine learning solution for **End-of-Turn (EoT)** detection at conversational pauses. The model accurately predicts whether a pause is a **Hold** (user is taking a breath or planning, agent should remain silent) or an **End-of-Turn** (user has finished speaking, agent should respond immediately).

We achieve a **71% latency reduction** overall compared to the naive silence-timeout baseline while strictly adhering to a budget of **&le; 5% interrupted turns**.

---

## 1. Project Overview & Objective

In voice agent floor control, timing is critical:
- Speaking too early interrupts the user (False Cutoff).
- Speaking too late results in awkward delays (Latency).

Our model predicts $p_{\text{eot}}$ for each pause using **only causal audio history** in the range $[0, \text{pause\_start}]$. It enables the agent to take the floor as quickly as **100 ms** for clear EOTs while remaining silent during long holds.

---

## 2. Folder Structure

```
├── eot_data/               # Dataset directory
│   ├── english/            # English audio files and labels.csv
│   └── hindi/              # Hindi audio files and labels.csv
├── starter/                # Evaluation & helper scripts
│   ├── score.py            # Official scoring script
│   └── features.py         # Starter feature utilities
├── outputs/                # Figures and analysis reports
│   ├── figures/            # EDA graphs
│   └── analysis/           # Acoustic trajectories and error dashboards
├── train.py                # Model training script
├── predict.py              # CLI inference script
├── causal_features.py      # Base 126 feature extractor class
├── improved_features.py    # Improved 130 feature extractor class
├── baseline_model.py       # Logistic Regression baseline profiler
├── model_comparison.py     # Comparison suite for 8 different ML models
├── hyperparameter_tuning.py# Systematic hyperparameter tuning script
├── error_analysis.py       # Error diagnostic dashboard generator
├── score_overall.py        # Combined evaluation scorer
├── model.joblib            # Serialized optimized SVC model bundle
├── README.md               # Setup and usage guide
├── RUNLOG.md               # Chronological log of experimental runs
└── SUMMARY.html            # Rich dashboard UI summarizing findings
```

---

## 3. Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/KiranL17/plivo-ml-24EX10014.git
   cd "plivo assignment"
   ```

2. **Install Dependencies**:
   Ensure python 3.8+ is installed. Then install requirements:
   ```bash
   pip install numpy pandas librosa scipy scikit-learn joblib matplotlib seaborn torch shap
   ```

---

## 4. How to Run

### A. Model Training
To extract the improved features, perform Random Forest selection, fit the RBF SVC classifier, and save `model.joblib`:
```bash
python train.py
```

### B. Prediction / Inference
To generate predictions for an unseen dataset:
```bash
python predict.py --data_dir eot_data/english --out predictions.csv
```
This generates `predictions.csv` with columns: `turn_id`, `pause_index`, `p_eot`.

### C. Evaluation & Scoring
To run the official score script and calculate the best mean response delay at $\le 5\%$ cutoff rate:
```bash
python starter/score.py --data_dir eot_data/english --pred eot_data/english/predictions.csv
```

---

## 5. Experimental Results

Our optimized **StandardScaler + RBF SVC** pipeline trained on the **top 80 selected features** achieves the following scores:

* **English Dataset**:
  * **Mean response delay**: **430 ms** (73% speedup vs 1600ms baseline)
  * **Interrupted turns**: 5.0%
  * **Evaluation AUC**: 0.966
* **Hindi Dataset**:
  * **Mean response delay**: **265 ms** (69% speedup vs 850ms baseline)
  * **Interrupted turns**: 5.0%
  * **Evaluation AUC**: 0.986
* **Overall Combined**:
  * **Mean response delay**: **348 ms** (71% speedup vs 1225ms baseline)
  * **Interrupted turns**: 5.0%
  * **Evaluation AUC**: 0.976

---

## 6. Future Work
1. **Multimodal ASR Integration**: Combine acoustic predictions with semantic sentence-completeness features from an automatic speech recognition (ASR) model.
2. **Deep Sequential Models**: Train an LSTM, GRU, or 1D CNN in PyTorch directly on raw frame-level speech sequences rather than static statistics.
3. **Data Augmentation**: perturb speed, shift pitch, and inject background noise to increase classifier resilience.
