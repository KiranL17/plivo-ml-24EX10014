import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import soundfile as sf

def main():
    print("Starting Exploratory Data Analysis (EDA)...")
    
    # 1. Setup paths
    base_dir = "C:/Users/lakka/OneDrive/Desktop/plivo assignment"
    data_dir = os.path.join(base_dir, "eot_data")
    figures_dir = os.path.join(base_dir, "outputs", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    languages = ["english", "hindi"]
    data_summary = {}
    
    # Configure matplotlib style for clean aesthetics
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 16
    })

    # Initialize plot layout
    fig_class, axes_class = plt.subplots(1, 2, figsize=(12, 5))
    fig_hist, axes_hist = plt.subplots(1, 2, figsize=(12, 5))
    fig_box, axes_box = plt.subplots(1, 2, figsize=(12, 5))
    fig_turn, axes_turn = plt.subplots(1, 2, figsize=(12, 5))

    for idx, lang in enumerate(languages):
        print(f"\nAnalyzing {lang.upper()} dataset...")
        labels_path = os.path.join(data_dir, lang, "labels.csv")
        
        # Load dataset
        df = pd.read_csv(labels_path)
        
        # 2. Verify Dataset Integrity
        print(f"[{lang}] Verification & Integrity Checks:")
        print(f" - Missing values in labels.csv:\n{df.isnull().sum().to_string()}")
        
        # Check start < end
        invalid_durations = df[df["pause_start"] >= df["pause_end"]]
        print(f" - Pauses with invalid times (start >= end): {len(invalid_durations)}")
        
        # Calculate pause durations
        df["pause_duration"] = df["pause_end"] - df["pause_start"]
        
        # Check audio file existence and get actual lengths
        audio_durations = {}
        missing_audio = 0
        for audio_file in df["audio_file"].unique():
            audio_path = os.path.join(data_dir, lang, audio_file)
            if not os.path.exists(audio_path):
                missing_audio += 1
            else:
                try:
                    info = sf.info(audio_path)
                    audio_durations[audio_file] = info.duration
                except Exception as e:
                    print(f"   Error reading {audio_file}: {e}")
        
        print(f" - Missing audio files: {missing_audio}")
        print(f" - Total unique audio files checked: {len(audio_durations)}")
        
        # Calculate turn lengths (duration of wav file)
        df["turn_total_duration"] = df["audio_file"].map(audio_durations)
        
        # 3. Calculate statistics
        num_audio_files = df["audio_file"].nunique()
        total_pauses = len(df)
        label_counts = df["label"].value_counts()
        label_dist = df["label"].value_counts(normalize=True) * 100
        
        avg_dur = df["pause_duration"].mean()
        max_dur = df["pause_duration"].max()
        min_dur = df["pause_duration"].min()
        
        print(f"\n[{lang}] Dataset Statistics Summary:")
        print(f" - Unique Audio Files: {num_audio_files}")
        print(f" - Total Annotated Pauses: {total_pauses}")
        print(f" - Label Counts: {label_counts.to_dict()}")
        print(f" - Label Distribution (%): {label_dist.to_dict()}")
        print(f" - Pause Durations (s): Mean={avg_dur:.3f}, Min={min_dur:.3f}, Max={max_dur:.3f}")
        
        # Label-specific stats
        print(f"\n[{lang}] Stats by Label:")
        print(df.groupby("label")["pause_duration"].describe(percentiles=[0.5, 0.75, 0.9, 0.95]))
        
        # 4. Save results to dictionary for report
        data_summary[lang] = {
            "df": df,
            "num_audio": num_audio_files,
            "total_pauses": total_pauses,
            "label_dist": label_dist.to_dict(),
            "avg_dur": avg_dur,
            "min_dur": min_dur,
            "max_dur": max_dur
        }
        
        # 5. Plots
        # A. Class Distribution
        sns.countplot(data=df, x="label", palette="viridis", ax=axes_class[idx], hue="label", legend=False)
        axes_class[idx].set_title(f"{lang.capitalize()} - Class Distribution")
        axes_class[idx].set_xlabel("Label")
        axes_class[idx].set_ylabel("Count")
        
        # B. Pause Duration Histogram
        sns.histplot(data=df, x="pause_duration", bins=20, kde=True, color="teal", ax=axes_hist[idx])
        axes_hist[idx].set_title(f"{lang.capitalize()} - Pause Duration Histogram")
        axes_hist[idx].set_xlabel("Duration (seconds)")
        axes_hist[idx].set_ylabel("Count")
        
        # C. Pause Duration by Label Boxplot
        sns.boxplot(data=df, x="label", y="pause_duration", palette="mako", ax=axes_box[idx], hue="label", legend=False)
        axes_box[idx].set_title(f"{lang.capitalize()} - Pause Duration by Label")
        axes_box[idx].set_xlabel("Label")
        axes_box[idx].set_ylabel("Duration (seconds)")
        
        # D. Turn Length Distribution
        sns.histplot(data=df.drop_duplicates(subset=["audio_file"]), x="turn_total_duration", bins=15, kde=True, color="purple", ax=axes_turn[idx])
        axes_turn[idx].set_title(f"{lang.capitalize()} - Turn Length Distribution")
        axes_turn[idx].set_xlabel("Turn Duration (seconds)")
        axes_turn[idx].set_ylabel("Count")

    # Adjust layout and save figures
    fig_class.tight_layout()
    fig_class.savefig(os.path.join(figures_dir, "class_distribution.png"), dpi=150)
    plt.close(fig_class)
    
    fig_hist.tight_layout()
    fig_hist.savefig(os.path.join(figures_dir, "pause_duration_histogram.png"), dpi=150)
    plt.close(fig_hist)
    
    fig_box.tight_layout()
    fig_box.savefig(os.path.join(figures_dir, "pause_duration_by_label.png"), dpi=150)
    plt.close(fig_box)
    
    fig_turn.tight_layout()
    fig_turn.savefig(os.path.join(figures_dir, "turn_length_distribution.png"), dpi=150)
    plt.close(fig_turn)
    
    print("\nSaved all figures to outputs/figures/ directory.")
    
    # 6. Listen to / Inspect several difficult examples
    print("\n--- Identifying Difficult Examples ---")
    for lang in languages:
        df = data_summary[lang]["df"]
        print(f"\n[{lang.upper()}] Difficult Cases:")
        
        # A. Long Hold Pauses (highly likely to cause false cutoffs)
        long_holds = df[(df["label"] == "hold")].sort_values(by="pause_duration", ascending=False).head(5)
        print("Top 5 Longest Hold Pauses (Risk of False Cutoff):")
        for _, row in long_holds.iterrows():
            print(f" - Turn: {row['turn_id']}, Index: {row['pause_index']}, Duration: {row['pause_duration']:.3f}s (Start: {row['pause_start']}s, End: {row['pause_end']}s)")
            
        # B. Short EOT Pauses (highly likely to hit the 1.6s timeout before firing)
        short_eots = df[(df["label"] == "eot")].sort_values(by="pause_duration", ascending=True).head(5)
        print("Top 5 Shortest EOT Pauses (Risk of Missed/Delayed Response):")
        for _, row in short_eots.iterrows():
            print(f" - Turn: {row['turn_id']}, Index: {row['pause_index']}, Duration: {row['pause_duration']:.3f}s (Start: {row['pause_start']}s, End: {row['pause_end']}s)")
            
    print("\nEDA Completed successfully!")

if __name__ == "__main__":
    main()
