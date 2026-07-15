import numpy as np
import librosa
import scipy.stats
import warnings
warnings.filterwarnings('ignore')

from causal_features import FeatureExtractor

class ImprovedFeatureExtractor(FeatureExtractor):
    """
    An enhanced causal FeatureExtractor class incorporating error-informed acoustic features:
    1. Vowel elongation ratio (last voiced segment length relative to historical mean)
    2. Final voiced pitch stability (standard deviation of final F0 normalized by context)
    3. Spectral flux dynamics ratio (last 200ms vs context)
    4. ZCR dynamics ratio (last 200ms vs context)
    """
    def __init__(self, sr=16000, frame_ms=25, hop_ms=10, n_mfcc=13):
        super(ImprovedFeatureExtractor, self).__init__(sr=sr, frame_ms=frame_ms, hop_ms=hop_ms, n_mfcc=n_mfcc)
        
    def _define_feature_names(self):
        # Inherit all base names
        names = super(ImprovedFeatureExtractor, self)._define_feature_names()
        # Add the 4 new error-informed features
        names.extend([
            "ratio_last_voiced_vs_average_voiced_duration",
            "f0_final_voiced_stability",
            "flux_ratio_last_200ms_vs_1s",
            "zcr_ratio_last_200ms_vs_1s"
        ])
        return sorted(names)
        
    def extract_features(self, x, sr, pause_start):
        # 1. Base Feature Extraction
        feature_vector = super(ImprovedFeatureExtractor, self).extract_features(x, sr, pause_start)
        
        # Build features dictionary from vector to easily append new values
        features = dict(zip(self.feature_names, feature_vector))
        
        # Enforce causality strictly
        if sr != self.sr:
            x = librosa.resample(x, orig_sr=sr, target_sr=self.sr)
            sr = self.sr
        end_idx = int(pause_start * sr)
        x_past = x[:end_idx]
        
        # If segment is too short, return base feature vector directly
        if len(x_past) < self.frame_length:
            return feature_vector
            
        seg_15s = x_past[-int(1.5 * sr):] if len(x_past) >= int(1.5 * sr) else x_past
        
        # A. F0 & Voiced Segment length metrics
        try:
            f0_full = librosa.yin(y=x_past, sr=sr, fmin=60.0, fmax=400.0, 
                                 frame_length=int(0.040 * sr), hop_length=self.hop_length)
            rms_full = librosa.feature.rms(y=x_past, frame_length=self.frame_length, hop_length=self.hop_length)[0]
            rms_db_full = 20 * np.log10(rms_full + 1e-12)
            
            min_len = min(len(rms_full), len(f0_full))
            f0_full_clean = f0_full[:min_len].copy()
            f0_full_clean[rms_db_full[:min_len] < -45] = 0.0
            
            # Find all contiguous voiced blocks in full history
            voiced_indices = np.where(f0_full_clean > 0)[0]
            if len(voiced_indices) > 0:
                diffs = np.diff(voiced_indices)
                split_indices = np.where(diffs > 1)[0]
                
                # Split indices into contiguous segments
                blocks = np.split(voiced_indices, split_indices + 1)
                block_lengths = [len(b) * (self.hop_length / sr) for b in blocks if len(b) > 0]
                
                # Final block is the last one
                last_block_len = block_lengths[-1]
                avg_block_len = np.mean(block_lengths) if len(block_lengths) > 0 else 0.0
                
                features["ratio_last_voiced_vs_average_voiced_duration"] = last_block_len / (avg_block_len + 1e-6)
                
                # Pitch stability of final voiced segment
                last_block_f0 = f0_full_clean[blocks[-1]]
                features["f0_final_voiced_stability"] = np.std(last_block_f0) / (np.std(f0_full_clean[f0_full_clean > 0]) + 1e-6)
            else:
                features["ratio_last_voiced_vs_average_voiced_duration"] = 0.0
                features["f0_final_voiced_stability"] = 0.0
        except Exception:
            features["ratio_last_voiced_vs_average_voiced_duration"] = 0.0
            features["f0_final_voiced_stability"] = 0.0
            
        # B. ZCR & Flux ratios
        zcr_15s = librosa.feature.zero_crossing_rate(y=seg_15s, frame_length=self.frame_length, hop_length=self.hop_length)[0]
        S = np.abs(librosa.stft(seg_15s, n_fft=self.frame_length, hop_length=self.hop_length))
        flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0) + 1e-12)
        
        # Split into last 200ms (last 20 frames) vs context (prior frames)
        if len(zcr_15s) > 20:
            features["zcr_ratio_last_200ms_vs_1s"] = np.mean(zcr_15s[-20:]) / (np.mean(zcr_15s[:-20]) + 1e-6)
        else:
            features["zcr_ratio_last_200ms_vs_1s"] = 1.0
            
        if len(flux) > 20:
            features["flux_ratio_last_200ms_vs_1s"] = np.mean(flux[-20:]) / (np.mean(flux[:-20]) + 1e-6)
        else:
            features["flux_ratio_last_200ms_vs_1s"] = 1.0
            
        # 3. Align keys to feature name array and return
        final_vector = np.zeros(self.num_features, dtype=np.float32)
        for idx, k in enumerate(self.feature_names):
            final_vector[idx] = features.get(k, 0.0)
            
        return final_vector
