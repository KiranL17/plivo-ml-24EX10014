import os
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import librosa

from starter.features import load_wav
from causal_features import FeatureExtractor
from validation import evaluate_predictions

def main():
    base_dir = "C:/Users/lakka/OneDrive/Desktop/plivo assignment"
    output_dir = os.path.join(base_dir, "outputs", "analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    languages = ["english", "hindi"]
    extractor = FeatureExtractor()
    
    # 1. Load model pipeline
    model_path = os.path.join(base_dir, "model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    pipeline = joblib.load(model_path)
    
    # 2. Extract features and make predictions
    data = {}
    for lang in languages:
        data_dir = os.path.join(base_dir, "eot_data", lang)
        labels_df = pd.read_csv(os.path.join(data_dir, "labels.csv"))
        
        cache = {}
        X, y, paths, pause_starts = [], [], [], []
        for idx, r in labels_df.iterrows():
            path = os.path.join(data_dir, r["audio_file"])
            if path not in cache:
                cache[path] = load_wav(path)
            x, sr = cache[path]
            
            feat = extractor.extract_features(x, sr, float(r["pause_start"]))
            X.append(feat)
            y.append(1 if r["label"] == "eot" else 0)
            paths.append(r["audio_file"])
            pause_starts.append(float(r["pause_start"]))
            
        data[lang] = {
            "X": np.array(X),
            "y": np.array(y),
            "paths": paths,
            "pause_starts": pause_starts,
            "df": labels_df
        }
        
    # Combine data
    X_comb = np.concatenate([data["english"]["X"], data["hindi"]["X"]], axis=0)
    y_comb = np.concatenate([data["english"]["y"], data["hindi"]["y"]], axis=0)
    
    # Run predictions (using full-fit pipeline for error analysis)
    probs = pipeline.predict_proba(X_comb)[:, 1]
    
    n_eng = len(data["english"]["y"])
    eng_df = data["english"]["df"].copy()
    eng_df["p_eot"] = probs[:n_eng]
    
    hin_df = data["hindi"]["df"].copy()
    hin_df["p_eot"] = probs[n_eng:]
    
    # English threshold = 0.55, Hindi threshold = 0.50
    eng_df["pred"] = (eng_df["p_eot"] >= 0.55).astype(int)
    eng_df["true"] = (eng_df["label"] == "eot").astype(int)
    
    hin_df["pred"] = (hin_df["p_eot"] >= 0.50).astype(int)
    hin_df["true"] = (hin_df["label"] == "eot").astype(int)
    
    # Identify False Positives (Hold predicted as EOT)
    eng_fp = eng_df[(eng_df["true"] == 0) & (eng_df["pred"] == 1)]
    hin_fp = hin_df[(hin_df["true"] == 0) & (hin_df["pred"] == 1)]
    
    # Identify False Negatives (EOT predicted as Hold)
    eng_fn = eng_df[(eng_df["true"] == 1) & (eng_df["pred"] == 0)]
    hin_fn = hin_df[(hin_df["true"] == 1) & (hin_df["pred"] == 0)]
    
    print("\n===== ERROR ANALYSIS RESULTS =====")
    print(f"English False Positives: {len(eng_fp)} | False Negatives: {len(eng_fn)}")
    print(f"Hindi False Positives:   {len(hin_fp)} | False Negatives: {len(hin_fn)}")
    
    # Inspect top English False Positive (highest probability on true hold)
    if len(eng_fp) > 0:
        top_eng_fp = eng_fp.sort_values(by="p_eot", ascending=False).iloc[0]
        print(f"\nTop English False Positive (Hold misclassified as EOT):")
        print(f" - File         : {top_eng_fp['audio_file']}")
        print(f" - Pause Start  : {top_eng_fp['pause_start']} s")
        print(f" - Prob EOT     : {top_eng_fp['p_eot']:.4f}")
        # Generate visual dashboard for this error
        visualize_error(base_dir, "english", top_eng_fp["audio_file"], float(top_eng_fp["pause_start"]), 
                        output_dir, "error_fp_dashboard.png", "False Positive (True Hold classified as EOT)")
        
    # Inspect top English False Negative (lowest probability on true EOT)
    if len(eng_fn) > 0:
        top_eng_fn = eng_fn.sort_values(by="p_eot", ascending=True).iloc[0]
        print(f"\nTop English False Negative (EOT misclassified as Hold):")
        print(f" - File         : {top_eng_fn['audio_file']}")
        print(f" - Pause Start  : {top_eng_fn['pause_start']} s")
        print(f" - Prob EOT     : {top_eng_fn['p_eot']:.4f}")
        # Generate visual dashboard for this error
        visualize_error(base_dir, "english", top_eng_fn["audio_file"], float(top_eng_fn["pause_start"]), 
                        output_dir, "error_fn_dashboard.png", "False Negative (True EOT classified as Hold)")

def visualize_error(base_dir, lang, wav_rel_path, pause_start, output_dir, out_name, title_suffix):
    wav_path = os.path.join(base_dir, "eot_data", lang, wav_rel_path)
    y, sr = librosa.load(wav_path, sr=16000)
    time_sec = np.arange(len(y)) / sr
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"Error Diagnostic Dashboard - {lang.capitalize()} {os.path.basename(wav_rel_path)}\n{title_suffix}", fontsize=14, fontweight='bold')
    
    # 1. Waveform
    axes[0].plot(time_sec, y, color="#4b5563")
    axes[0].set_title("Waveform")
    axes[0].set_ylabel("Amplitude")
    axes[0].axvline(pause_start, color="#ef4444", linestyle="--", label="Pause Start Boundary")
    axes[0].legend(loc="upper right")
    
    # 2. Spectrogram with Pitch F0 overlay
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear', ax=axes[1], cmap='magma')
    axes[1].set_title("Spectrogram & F0 Contour")
    axes[1].set_ylabel("Freq (Hz)")
    axes[1].set_ylim(0, 3000)
    
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(y, sr=sr, fmin=60, fmax=400)
        times_f0 = librosa.times_like(f0)
        axes[1].plot(times_f0, f0, color="cyan", linewidth=2.5, label="Pitch (F0)")
        axes[1].legend(loc="upper right")
    except:
        pass
        
    # 3. RMS Energy (dB)
    rms = librosa.feature.rms(y=y, frame_length=400, hop_length=160)[0]
    times_rms = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=160)
    rms_db = 20 * np.log10(rms + 1e-12)
    axes[2].plot(times_rms, rms_db, color="#06b6d4", linewidth=2)
    axes[2].set_title("RMS Energy Curve (dB)")
    axes[2].set_ylabel("Energy (dB)")
    axes[2].set_ylim(-80, 0)
    axes[2].axvline(pause_start, color="#ef4444", linestyle="--")
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, out_name)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved diagnostic error dashboard at: {fig_path}")

if __name__ == "__main__":
    main()
