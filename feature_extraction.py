import numpy as np
import librosa
import scipy.stats

def fit_slope(y):
    """Fit a linear regression slope to a 1D array y."""
    n = len(y)
    if n < 2:
        return 0.0
    x = np.arange(n)
    slope, _, _, _, _ = scipy.stats.linregress(x, y)
    return float(slope)

def get_last_voiced_segment(f0_contour):
    """Extract the last contiguous voiced segment from an F0 contour."""
    voiced_indices = np.where(f0_contour > 0)[0]
    if len(voiced_indices) == 0:
        return np.array([])
    
    # Find gaps (difference > 1 frame)
    diffs = np.diff(voiced_indices)
    split_indices = np.where(diffs > 1)[0]
    
    if len(split_indices) == 0:
        last_block = voiced_indices
    else:
        last_block = voiced_indices[split_indices[-1] + 1:]
        
    return f0_contour[last_block]

def extract_robust_features(x, sr, pause_start):
    """
    Extract a rich set of causal features from the audio strictly before pause_start.
    Causality is guaranteed by slicing the audio to x_past = x[:int(pause_start * sr)].
    """
    # 1. Slice audio up to pause_start (guarantees causality)
    end_idx = int(pause_start * sr)
    x_past = x[:end_idx]
    
    # Define total expected features size to ensure shape consistency
    expected_size = 110
    
    # Setup local window segments
    seg_15s = x_past[-int(1.5 * sr):] if len(x_past) >= int(1.5 * sr) else x_past
    seg_05s = x_past[-int(0.5 * sr):] if len(x_past) >= int(0.5 * sr) else x_past
    
    # If the segment is extremely short, return default empty features
    if len(seg_15s) < sr // 10:
        return np.zeros(expected_size, dtype=np.float32)
        
    features = {}
    
    # Feature: Total speech context duration in seconds
    total_dur = len(x_past) / sr
    features["total_duration"] = total_dur
    
    # Hop length = 10ms, Frame length = 25ms
    hop_length = int(0.010 * sr)
    frame_length = int(0.025 * sr)
    
    # 2. Energy features (RMS)
    rms_15s = librosa.feature.rms(y=seg_15s, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db_15s = 20 * np.log10(rms_15s + 1e-12)
    
    rms_05s = librosa.feature.rms(y=seg_05s, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db_05s = 20 * np.log10(rms_05s + 1e-12)
    
    # Energy statistics
    features["energy_mean_15s"] = np.mean(rms_db_15s)
    features["energy_std_15s"] = np.std(rms_db_15s)
    features["energy_max_15s"] = np.max(rms_db_15s)
    features["energy_min_15s"] = np.min(rms_db_15s)
    
    features["energy_mean_05s"] = np.mean(rms_db_05s)
    features["energy_std_05s"] = np.std(rms_db_05s)
    features["energy_mean_100ms"] = np.mean(rms_db_15s[-10:]) if len(rms_db_15s) >= 10 else np.mean(rms_db_15s)
    features["energy_mean_250ms"] = np.mean(rms_db_15s[-25:]) if len(rms_db_15s) >= 25 else np.mean(rms_db_15s)
    
    # Energy slopes
    features["energy_slope_last_10"] = fit_slope(rms_db_15s[-10:])
    features["energy_slope_last_25"] = fit_slope(rms_db_15s[-25:])
    features["energy_slope_last_50"] = fit_slope(rms_db_15s[-50:]) if len(rms_db_15s) >= 50 else fit_slope(rms_db_15s)
    
    # Energy differences & ratios
    features["energy_diff_100ms_vs_500ms"] = features["energy_mean_100ms"] - (np.mean(rms_db_15s[-50:]) if len(rms_db_15s) >= 50 else np.mean(rms_db_15s))
    features["energy_ratio_last_200ms_vs_1s"] = np.mean(rms_db_15s[-20:]) - np.mean(rms_db_15s[:-20]) if len(rms_db_15s) > 20 else 0.0
    
    # 3. Zero Crossing Rate (ZCR) features
    zcr_15s = librosa.feature.zero_crossing_rate(y=seg_15s, frame_length=frame_length, hop_length=hop_length)[0]
    features["zcr_mean_15s"] = np.mean(zcr_15s)
    features["zcr_std_15s"] = np.std(zcr_15s)
    features["zcr_mean_05s"] = np.mean(zcr_15s[-50:]) if len(zcr_15s) >= 50 else np.mean(zcr_15s)
    features["zcr_slope_last_15"] = fit_slope(zcr_15s[-15:])
    
    # 4. Pitch (F0) features
    # Use YIN algorithm (fmin=60, fmax=400)
    try:
        fmin, fmax = 60.0, 400.0
        f0_15s = librosa.yin(y=seg_15s, sr=sr, fmin=fmin, fmax=fmax, 
                             frame_length=int(0.040 * sr), hop_length=hop_length)
        
        # Clean F0 using energy mask
        min_len = min(len(rms_15s), len(f0_15s))
        rms_clean = rms_db_15s[:min_len]
        f0_clean = f0_15s[:min_len].copy()
        f0_clean[rms_clean < -50] = 0.0
        
        # Get voiced frames
        voiced_f0 = f0_clean[f0_clean > 0]
        
        # Last contiguous voiced segment
        last_voiced_seg = get_last_voiced_segment(f0_clean)
        
        if len(voiced_f0) > 0:
            features["f0_mean"] = np.mean(voiced_f0)
            features["f0_std"] = np.std(voiced_f0)
            features["f0_max"] = np.max(voiced_f0)
            features["f0_min"] = np.min(voiced_f0)
            features["f0_last_voiced_mean"] = np.mean(voiced_f0[-5:])
            features["f0_last_voiced_slope"] = fit_slope(voiced_f0[-10:])
            features["voicing_ratio"] = len(voiced_f0) / len(f0_clean)
            features["voicing_ratio_last_05s"] = np.mean(f0_clean[-50:] > 0) if len(f0_clean) >= 50 else features["voicing_ratio"]
        else:
            features["f0_mean"] = 0.0
            features["f0_std"] = 0.0
            features["f0_max"] = 0.0
            features["f0_min"] = 0.0
            features["f0_last_voiced_mean"] = 0.0
            features["f0_last_voiced_slope"] = 0.0
            features["voicing_ratio"] = 0.0
            features["voicing_ratio_last_05s"] = 0.0
            
        if len(last_voiced_seg) > 0:
            features["f0_last_contiguous_slope"] = fit_slope(last_voiced_seg)
        else:
            features["f0_last_contiguous_slope"] = 0.0
            
    except Exception as e:
        features["f0_mean"] = 0.0
        features["f0_std"] = 0.0
        features["f0_max"] = 0.0
        features["f0_min"] = 0.0
        features["f0_last_voiced_mean"] = 0.0
        features["f0_last_voiced_slope"] = 0.0
        features["voicing_ratio"] = 0.0
        features["voicing_ratio_last_05s"] = 0.0
        features["f0_last_contiguous_slope"] = 0.0
        
    # 5. Speaker normalization (turn-level stats)
    rms_full = librosa.feature.rms(y=x_past, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db_full = 20 * np.log10(rms_full + 1e-12)
    turn_energy_mean = np.mean(rms_db_full)
    turn_energy_std = np.std(rms_db_full) + 1e-6
    
    # Speaker normalized local energy
    features["energy_normalized_last_05s"] = (features["energy_mean_05s"] - turn_energy_mean) / turn_energy_std
    features["energy_normalized_last_100ms"] = (features["energy_mean_100ms"] - turn_energy_mean) / turn_energy_std
    features["energy_normalized_last_250ms"] = (features["energy_mean_250ms"] - turn_energy_mean) / turn_energy_std
    
    # Turn level pitch stats
    try:
        f0_full = librosa.yin(y=x_past, sr=sr, fmin=60.0, fmax=400.0, 
                             frame_length=int(0.040 * sr), hop_length=hop_length)
        min_len_full = min(len(rms_full), len(f0_full))
        rms_full_clean = rms_db_full[:min_len_full]
        f0_full_clean = f0_full[:min_len_full].copy()
        f0_full_clean[rms_full_clean < -50] = 0.0
        voiced_full_f0 = f0_full_clean[f0_full_clean > 0]
        
        if len(voiced_full_f0) > 0 and features["f0_mean"] > 0:
            turn_f0_mean = np.mean(voiced_full_f0)
            turn_f0_std = np.std(voiced_full_f0) + 1e-6
            features["f0_normalized_mean"] = (features["f0_mean"] - turn_f0_mean) / turn_f0_std
            features["f0_normalized_last_voiced_mean"] = (features["f0_last_voiced_mean"] - turn_f0_mean) / turn_f0_std
            features["f0_normalized_max"] = (features["f0_max"] - turn_f0_mean) / turn_f0_std
            features["f0_normalized_min"] = (features["f0_min"] - turn_f0_mean) / turn_f0_std
        else:
            features["f0_normalized_mean"] = 0.0
            features["f0_normalized_last_voiced_mean"] = 0.0
            features["f0_normalized_max"] = 0.0
            features["f0_normalized_min"] = 0.0
    except:
        features["f0_normalized_mean"] = 0.0
        features["f0_normalized_last_voiced_mean"] = 0.0
        features["f0_normalized_max"] = 0.0
        features["f0_normalized_min"] = 0.0

    # 6. Spectral Features (MFCCs + Deltas + Double Deltas)
    mfccs = librosa.feature.mfcc(y=seg_15s, sr=sr, n_mfcc=13, n_fft=frame_length, hop_length=hop_length)
    mfcc_deltas = librosa.feature.delta(mfccs)
    mfcc_delta2s = librosa.feature.delta(mfccs, order=2)
    
    # Summary stats for MFCCs (means and stds over the frames)
    for i in range(13):
        features[f"mfcc_mean_{i}"] = np.mean(mfccs[i])
        features[f"mfcc_std_{i}"] = np.std(mfccs[i])
        features[f"mfcc_delta_mean_{i}"] = np.mean(mfcc_deltas[i])
        features[f"mfcc_delta_std_{i}"] = np.std(mfcc_deltas[i])
        features[f"mfcc_delta2_mean_{i}"] = np.mean(mfcc_delta2s[i])
        features[f"mfcc_delta2_std_{i}"] = np.std(mfcc_delta2s[i])
        
    # Convert features dictionary to a sorted list of float values
    feature_keys = sorted(features.keys())
    
    # Sanity check: verify length matches expected_size
    if len(feature_keys) != expected_size:
        # If there is a key mismatch, pad or truncate to maintain size
        feature_vector = np.zeros(expected_size, dtype=np.float32)
        for idx, k in enumerate(feature_keys[:expected_size]):
            feature_vector[idx] = features[k]
        return feature_vector
        
    feature_vector = np.array([features[k] for k in feature_keys], dtype=np.float32)
    return feature_vector
