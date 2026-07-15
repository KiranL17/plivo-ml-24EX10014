# Model Notes & Architectural Analysis

## 1. Model Selection & Rationale
We utilize a **Support Vector Classifier (SVC)** with a non-linear Radial Basis Function (RBF) kernel. Standard scaling is applied to the input vectors to ensure zero mean and unit variance.
- **Why SVC RBF**: On high-dimensional feature spaces (80 selected features) with a small dataset size (496 total training pauses), non-linear SVMs generalize much better than high-capacity neural networks (which overfit easily) and tree ensembles (which struggle with high-dimensional correlation).
- **Joint Training**: We train a single cross-lingual model combining English and Hindi to double the training sample size and improve acoustic representation robustness.

---

## 2. Feature Extraction & Selection
We extract **130 handcrafted causal features** from the historical audio window `[0, pause_start]` preceding each pause:
- **Pitch Trajectories**: Pitch stats, contiguous final voiced slopes, and normalized pitch.
- **Energy Dynamics**: Frame-level RMS energy, energy slopes, decay ratios, and local energy variance.
- **Spectral Shapes**: Zero Crossing Rate (ZCR) mean/std/dynamics, Spectral Centroid, Bandwidth, Rolloff, Flux, and Entropy.
- **MFCCs**: 13 coefficients + Deltas + Double-Deltas.
- **Error-Informed Ratios**: Vowel lengthening ratio, F0 stability, ZCR ratio, and spectral flux dynamics ratio.
- **Causal Normalization**: Local pitch and energy features are causally normalized against turn-level historical baselines.

Using **Random Forest Feature Importance**, we filter out 50 redundant features, training our final pipeline on the **top 80 selected features**.

---

## 3. Failure Modes & Weaknesses
- **Volumetric Decay in Holds (False Positives)**: If a speaker takes a breath mid-turn and decreases their volume gradually, the model misclassifies it as EOT.
- **Abrupt EOT Terminations (False Negatives)**: If a speaker terminates a sentence abruptly without pitch/energy decay, the model misclassifies it as a hold continuation.

---

## 4. Next Steps & Future Work
With more time and budget, we would:
1. **ASR Multimodal Fusion**: Extract text transcript embeddings to detect grammatical completeness.
2. **Sequential Causal Models**: Train a causal 1D CNN-LSTM or GRU in PyTorch on frame-level features instead of static averages.
3. **Acoustic Data Augmentation**: Apply speed perturbation, pitch shifting, and additive background noise to double the dataset size.
