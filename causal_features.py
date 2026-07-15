import numpy as np
import librosa
import scipy.stats
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class FeatureExtractor:
    """
    A modular, causal feature extraction pipeline for End-of-Turn (EoT) detection.
    
    CRITICAL CAUSALITY RULE:
    - This extractor ONLY accesses audio in the range [0, pause_start].
    - It strictly avoids the current pause's duration (pause_end - pause_start) as using it 
      violates the causality rule (since a live agent cannot know how long a hold pause will last).
    """
    def __init__(self, sr: int = 16000, frame_ms: float = 25.0, hop_ms: float = 10.0, n_mfcc: int = 13) -> None:
        """
        Initialize the FeatureExtractor.

        Args:
            sr (int): Target sample rate (default 16000).
            frame_ms (float): Frame length in milliseconds (default 25.0).
            hop_ms (float): Hop length in milliseconds (default 10.0).
            n_mfcc (int): Number of MFCC coefficients (default 13).
        """
        self.sr = sr
        self.frame_length = int(sr * frame_ms / 1000)
        self.hop_length = int(sr * hop_ms / 1000)
        self.n_mfcc = n_mfcc
        
        # Define the exact names of the features extracted
        self.feature_names: List[str] = self._define_feature_names()
        self.num_features: int = len(self.feature_names)
        
    def _define_feature_names(self) -> List[str]:
        """Define alphabetical list of feature names to guarantee ordering."""
        names = [
            "total_duration",
            "energy_mean_15s", "energy_std_15s", "energy_max_15s", "energy_min_15s",
            "energy_mean_05s", "energy_std_05s", "energy_mean_100ms", "energy_mean_250ms",
            "energy_slope_last_10", "energy_slope_last_25", "energy_slope_last_50",
            "energy_diff_100ms_vs_500ms", "energy_ratio_last_200ms_vs_1s",
            "energy_local_variance",
            "zcr_mean_15s", "zcr_std_15s", "zcr_mean_05s", "zcr_slope_last_15",
            "f0_mean", "f0_std", "f0_max", "f0_min", "f0_variance",
            "f0_last_voiced_mean", "f0_last_voiced_slope", "f0_last_contiguous_slope",
            "voicing_ratio", "voicing_ratio_last_05s", "last_voiced_segment_duration",
            "silence_ratio_15s", "spectral_flux_mean", "spectral_flux_std",
            "spectral_entropy_mean", "spectral_entropy_std",
            "spectral_centroid_mean", "spectral_centroid_std",
            "spectral_bandwidth_mean", "spectral_bandwidth_std",
            "spectral_rolloff_mean", "spectral_rolloff_std",
            "energy_normalized_last_05s", "energy_normalized_last_100ms", "energy_normalized_last_250ms",
            "f0_normalized_mean", "f0_normalized_last_voiced_mean", "f0_normalized_max", "f0_normalized_min"
        ]
        # MFCCs features (13 coefficients * 6 stats = 78)
        for i in range(self.n_mfcc):
            names.extend([
                f"mfcc_mean_{i}", f"mfcc_std_{i}",
                f"mfcc_delta_mean_{i}", f"mfcc_delta_std_{i}",
                f"mfcc_delta2_mean_{i}", f"mfcc_delta2_std_{i}"
            ])
        return sorted(names)
        
    def _fit_slope(self, y: np.ndarray) -> float:
        """Helper to fit a linear regression slope over a 1D array."""
        n = len(y)
        if n < 2:
            return 0.0
        x = np.arange(n)
        slope, _, _, _, _ = scipy.stats.linregress(x, y)
        return float(slope)
        
    def _get_last_voiced_segment(self, f0_contour: np.ndarray) -> Tuple[np.ndarray, float]:
        """Isolate the final contiguous voiced segment of the pitch contour."""
        voiced_indices = np.where(f0_contour > 0)[0]
        if len(voiced_indices) == 0:
            return np.array([]), 0.0
        diffs = np.diff(voiced_indices)
        split_indices = np.where(diffs > 1)[0]
        if len(split_indices) == 0:
            last_block = voiced_indices
        else:
            last_block = voiced_indices[split_indices[-1] + 1:]
        return f0_contour[last_block], len(last_block) * (self.hop_length / self.sr)

    def extract_features(self, x: np.ndarray, sr: int, pause_start: float) -> np.ndarray:
        """
        Extract causal acoustic and prosodic features from the audio history.

        Args:
            x (np.ndarray): Full raw audio waveform.
            sr (int): Target sample rate.
            pause_start (float): The start timestamp of the pause in seconds.

        Returns:
            np.ndarray: 1D feature array aligned with self.feature_names.
        """
        # Ensure sample rate match
        if sr != self.sr:
            x = librosa.resample(x, orig_sr=sr, target_sr=self.sr)
            sr = self.sr
            
        # 1. Strictly enforce causality by slicing audio at pause_start
        end_idx = int(pause_start * sr)
        x_past = x[:end_idx]
        
        # If the audio segment is extremely short, return empty vector
        if len(x_past) < self.frame_length:
            return np.zeros(self.num_features, dtype=np.float32)
            
        # Define local windows (last 1.5s and last 0.5s)
        seg_15s = x_past[-int(1.5 * sr):] if len(x_past) >= int(1.5 * sr) else x_past
        seg_05s = x_past[-int(0.5 * sr):] if len(x_past) >= int(0.5 * sr) else x_past
        
        features: Dict[str, float] = {}
        
        # Feature 1: Total speech context duration in seconds
        features["total_duration"] = len(x_past) / sr
        
        # 2. Extract local Energy (RMS)
        rms_15s = librosa.feature.rms(y=seg_15s, frame_length=self.frame_length, hop_length=self.hop_length)[0]
        rms_db_15s = 20 * np.log10(rms_15s + 1e-12)
        
        rms_05s = librosa.feature.rms(y=seg_05s, frame_length=self.frame_length, hop_length=self.hop_length)[0]
        rms_db_05s = 20 * np.log10(rms_05s + 1e-12)
        
        features["energy_mean_15s"] = float(np.mean(rms_db_15s))
        features["energy_std_15s"] = float(np.std(rms_db_15s))
        features["energy_max_15s"] = float(np.max(rms_db_15s))
        features["energy_min_15s"] = float(np.min(rms_db_15s))
        
        features["energy_mean_05s"] = float(np.mean(rms_db_05s))
        features["energy_std_05s"] = float(np.std(rms_db_05s))
        features["energy_mean_100ms"] = float(np.mean(rms_db_15s[-10:])) if len(rms_db_15s) >= 10 else float(np.mean(rms_db_15s))
        features["energy_mean_250ms"] = float(np.mean(rms_db_15s[-25:])) if len(rms_db_15s) >= 25 else float(np.mean(rms_db_15s))
        
        # Energy Slopes & Variations
        features["energy_slope_last_10"] = self._fit_slope(rms_db_15s[-10:])
        features["energy_slope_last_25"] = self._fit_slope(rms_db_15s[-25:])
        features["energy_slope_last_50"] = self._fit_slope(rms_db_15s[-50:]) if len(rms_db_15s) >= 50 else self._fit_slope(rms_db_15s)
        
        features["energy_diff_100ms_vs_500ms"] = features["energy_mean_100ms"] - (float(np.mean(rms_db_15s[-50:])) if len(rms_db_15s) >= 50 else float(np.mean(rms_db_15s)))
        features["energy_ratio_last_200ms_vs_1s"] = float(np.mean(rms_db_15s[-20:])) - float(np.mean(rms_db_15s[:-20])) if len(rms_db_15s) > 20 else 0.0
        features["energy_local_variance"] = float(np.var(rms_db_05s))
        
        # Silence ratio (ratio of silent frames below -45dB in the last 1.5s)
        features["silence_ratio_15s"] = float(np.mean(rms_db_15s < -45))
        
        # 3. Zero Crossing Rate (ZCR)
        zcr_15s = librosa.feature.zero_crossing_rate(y=seg_15s, frame_length=self.frame_length, hop_length=self.hop_length)[0]
        features["zcr_mean_15s"] = float(np.mean(zcr_15s))
        features["zcr_std_15s"] = float(np.std(zcr_15s))
        features["zcr_mean_05s"] = float(np.mean(zcr_15s[-50:])) if len(zcr_15s) >= 50 else float(np.mean(zcr_15s))
        features["zcr_slope_last_15"] = self._fit_slope(zcr_15s[-15:])
        
        # 4. Pitch (F0) tracking (YIN)
        try:
            fmin, fmax = 60.0, 400.0
            f0_15s = librosa.yin(y=seg_15s, sr=sr, fmin=fmin, fmax=fmax, 
                                 frame_length=int(0.040 * sr), hop_length=self.hop_length)
            
            # Mask unvoiced and silent frames
            min_len = min(len(rms_15s), len(f0_15s))
            rms_clean = rms_db_15s[:min_len]
            f0_clean = f0_15s[:min_len].copy()
            f0_clean[rms_clean < -45] = 0.0
            
            voiced_f0 = f0_clean[f0_clean > 0]
            last_voiced_seg, last_voiced_dur = self._get_last_voiced_segment(f0_clean)
            
            if len(voiced_f0) > 0:
                features["f0_mean"] = float(np.mean(voiced_f0))
                features["f0_std"] = float(np.std(voiced_f0))
                features["f0_max"] = float(np.max(voiced_f0))
                features["f0_min"] = float(np.min(voiced_f0))
                features["f0_variance"] = float(np.var(voiced_f0))
                features["f0_last_voiced_mean"] = float(np.mean(voiced_f0[-5:]))
                features["f0_last_voiced_slope"] = self._fit_slope(voiced_f0[-10:])
                features["voicing_ratio"] = len(voiced_f0) / len(f0_clean)
                features["voicing_ratio_last_05s"] = float(np.mean(f0_clean[-50:] > 0)) if len(f0_clean) >= 50 else features["voicing_ratio"]
                features["last_voiced_segment_duration"] = last_voiced_dur
            else:
                features["f0_mean"] = 0.0
                features["f0_std"] = 0.0
                features["f0_max"] = 0.0
                features["f0_min"] = 0.0
                features["f0_variance"] = 0.0
                features["f0_last_voiced_mean"] = 0.0
                features["f0_last_voiced_slope"] = 0.0
                features["voicing_ratio"] = 0.0
                features["voicing_ratio_last_05s"] = 0.0
                features["last_voiced_segment_duration"] = 0.0
                
            if len(last_voiced_seg) > 0:
                features["f0_last_contiguous_slope"] = self._fit_slope(last_voiced_seg)
            else:
                features["f0_last_contiguous_slope"] = 0.0
        except Exception:
            features["f0_mean"] = 0.0
            features["f0_std"] = 0.0
            features["f0_max"] = 0.0
            features["f0_min"] = 0.0
            features["f0_variance"] = 0.0
            features["f0_last_voiced_mean"] = 0.0
            features["f0_last_voiced_slope"] = 0.0
            features["voicing_ratio"] = 0.0
            features["voicing_ratio_last_05s"] = 0.0
            features["last_voiced_segment_duration"] = 0.0
            features["f0_last_contiguous_slope"] = 0.0
            
        # 5. Causal Speaker Normalization
        rms_full = librosa.feature.rms(y=x_past, frame_length=self.frame_length, hop_length=self.hop_length)[0]
        rms_db_full = 20 * np.log10(rms_full + 1e-12)
        turn_energy_mean = float(np.mean(rms_db_full))
        turn_energy_std = float(np.std(rms_db_full)) + 1e-6
        
        features["energy_normalized_last_05s"] = (features["energy_mean_05s"] - turn_energy_mean) / turn_energy_std
        features["energy_normalized_last_100ms"] = (features["energy_mean_100ms"] - turn_energy_mean) / turn_energy_std
        features["energy_normalized_last_250ms"] = (features["energy_mean_250ms"] - turn_energy_mean) / turn_energy_std
        
        try:
            f0_full = librosa.yin(y=x_past, sr=sr, fmin=60.0, fmax=400.0, 
                                 frame_length=int(0.040 * sr), hop_length=self.hop_length)
            min_len_full = min(len(rms_full), len(f0_full))
            rms_full_clean = rms_db_full[:min_len_full]
            f0_full_clean = f0_full[:min_len_full].copy()
            f0_full_clean[rms_full_clean < -45] = 0.0
            voiced_full_f0 = f0_full_clean[f0_full_clean > 0]
            
            if len(voiced_full_f0) > 0 and features["f0_mean"] > 0:
                turn_f0_mean = float(np.mean(voiced_full_f0))
                turn_f0_std = float(np.std(voiced_full_f0)) + 1e-6
                features["f0_normalized_mean"] = (features["f0_mean"] - turn_f0_mean) / turn_f0_std
                features["f0_normalized_last_voiced_mean"] = (features["f0_last_voiced_mean"] - turn_f0_mean) / turn_f0_std
                features["f0_normalized_max"] = (features["f0_max"] - turn_f0_mean) / turn_f0_std
                features["f0_normalized_min"] = (features["f0_min"] - turn_f0_mean) / turn_f0_std
            else:
                features["f0_normalized_mean"] = 0.0
                features["f0_normalized_last_voiced_mean"] = 0.0
                features["f0_normalized_max"] = 0.0
                features["f0_normalized_min"] = 0.0
        except Exception:
            features["f0_normalized_mean"] = 0.0
            features["f0_normalized_last_voiced_mean"] = 0.0
            features["f0_normalized_max"] = 0.0
            features["f0_normalized_min"] = 0.0

        # 6. Spectral shape features
        centroid = librosa.feature.spectral_centroid(y=seg_15s, sr=sr, n_fft=self.frame_length, hop_length=self.hop_length)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=seg_15s, sr=sr, n_fft=self.frame_length, hop_length=self.hop_length)[0]
        rolloff = librosa.feature.spectral_rolloff(y=seg_15s, sr=sr, n_fft=self.frame_length, hop_length=self.hop_length)[0]
        
        # Spectral Flux (first difference of spectrogram columns)
        S = np.abs(librosa.stft(seg_15s, n_fft=self.frame_length, hop_length=self.hop_length))
        flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0) + 1e-12)
        
        # Spectral Entropy
        S_power = S**2
        S_norm = S_power / (np.sum(S_power, axis=0, keepdims=True) + 1e-12)
        entropy = -np.sum(S_norm * np.log2(S_norm + 1e-12), axis=0)
        
        features["spectral_flux_mean"] = float(np.mean(flux)) if len(flux) > 0 else 0.0
        features["spectral_flux_std"] = float(np.std(flux)) if len(flux) > 0 else 0.0
        features["spectral_entropy_mean"] = float(np.mean(entropy))
        features["spectral_entropy_std"] = float(np.std(entropy))
        
        features["spectral_centroid_mean"] = float(np.mean(centroid))
        features["spectral_centroid_std"] = float(np.std(centroid))
        features["spectral_bandwidth_mean"] = float(np.mean(bandwidth))
        features["spectral_bandwidth_std"] = float(np.std(bandwidth))
        features["spectral_rolloff_mean"] = float(np.mean(rolloff))
        features["spectral_rolloff_std"] = float(np.std(rolloff))
        
        # 7. MFCCs + Deltas + Double Deltas
        mfccs = librosa.feature.mfcc(y=seg_15s, sr=sr, n_mfcc=self.n_mfcc, n_fft=self.frame_length, hop_length=self.hop_length)
        mfcc_deltas = librosa.feature.delta(mfccs)
        mfcc_delta2s = librosa.feature.delta(mfccs, order=2)
        
        for i in range(self.n_mfcc):
            features[f"mfcc_mean_{i}"] = float(np.mean(mfccs[i]))
            features[f"mfcc_std_{i}"] = float(np.std(mfccs[i]))
            features[f"mfcc_delta_mean_{i}"] = float(np.mean(mfcc_deltas[i]))
            features[f"mfcc_delta_std_{i}"] = float(np.std(mfcc_deltas[i]))
            features[f"mfcc_delta2_mean_{i}"] = float(np.mean(mfcc_delta2s[i]))
            features[f"mfcc_delta2_std_{i}"] = float(np.std(mfcc_delta2s[i]))
            
        # 8. Align keys to define feature names list and compile array
        feature_vector = np.zeros(self.num_features, dtype=np.float32)
        for idx, k in enumerate(self.feature_names):
            feature_vector[idx] = features.get(k, 0.0)
            
        return feature_vector
