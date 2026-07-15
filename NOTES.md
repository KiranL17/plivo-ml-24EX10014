# Notes & Model Analysis

1. The model uses a Support Vector Classifier (RBF kernel) trained on a rich set of 110 acoustic and prosodic features extracted strictly from the audio preceding the pause.
2. The key signals leverage temporal energy decay (e.g., energy slope over the last 100ms/500ms), and vocal pitch dynamics (e.g., contiguous pitch slope of the final syllable to capture statement finality).
3. We also extract spectral shapes (MFCCs, deltas, and double-deltas) and zero-crossing rates to capture unvoiced phonetic closures.
4. Crucially, turn-level z-score normalization is applied to local pitch and energy features, adapting the model to individual speaker baselines causally.
5. The model still faces challenges on long pause holds (over 1.5s) where a speaker stops talking without pitch/energy decay, as the classifier might classify it as an EOT and trigger an interruption.
6. Short EOT pauses also pose a risk if the speaker halts mid-sentence and hangs up abruptly, preventing the model from capturing a clean EOT transition.
7. With one more day, we would explore data augmentation (e.g., adding noise, speed/pitch perturbation) to increase the size of our small dataset and improve classifier generalization.
8. We would also implement a sequence-based recurrent or convolutional architecture (such as a 1D CNN-GRU) in PyTorch to model frame-level dynamics directly rather than using hand-crafted window statistics.
9. Finally, we would experiment with language-specific fine-tuning layers or language embedding features to better capture prosodic differences between English and Hindi.
