# Sleep Apnea Detection Web Application

This folder contains the Streamlit application used to perform real-time sleep apnea detection from respiratory audio recordings.

---

## Features

- Upload respiratory audio files
- Automatic feature extraction
- Waveform visualization
- Mel Spectrogram visualization
- Apnea / Non-Apnea prediction
- Confidence score display

---

## Supported Formats

- WAV
- MP3
- NPY

---

## Application Workflow

Audio Upload
→ Preprocessing
→ Feature Extraction
→ Trained Ensemble Model
→ Prediction
→ Visualization

---

## Quick Start

git clone https://github.com/yourusername/sleep-apnea-detection.git

cd sleep-apnea-detection/app

pip install -r requirements.txt

streamlit run apneademo.py
