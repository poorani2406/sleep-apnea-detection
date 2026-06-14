# sleep-apnea-detection
Detection and Classification of Obstructive Sleep Apnea using Audio Spectrogram Analysis

### Sleep_Apnea_Ensemble_Model.ipynb

Classical machine learning pipeline using handcrafted audio features.

#### Pipeline

Audio Clip
→ Feature Extraction (276 features)
→ SMOTE
→ SVM / Random Forest / XGBoost
→ Ensemble Prediction

#### Features Used

- MFCC + Delta + Delta-Delta
- Mel Spectrogram Statistics
- Spectral Contrast
- Tonnetz
- Spectral Centroid
- Spectral Rolloff
- Zero Crossing Rate
- RMS Energy

#### Models

- SVM (RBF Kernel)
- Random Forest
- XGBoost (Optuna Tuned)

---


## Dataset

PSG-Audio Apnea Dataset

Source:
https://www.kaggle.com/datasets/bryandarquea/psg-audio-apnea-audios

---

## Author Notes

These notebooks were developed for research, benchmarking, and model comparison purposes before deployment into the Streamlit application.
