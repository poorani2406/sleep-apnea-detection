# Research Notebooks

This directory contains all experimentation, model development, benchmarking, and paper replication work performed during the Sleep Apnea Detection project.

---

## Notebooks

### 1. BiLSTM_Attention_Model.ipynb

Deep learning approach using Mel Spectrogram sequences and an attention mechanism.

#### Architecture

Mel Spectrogram
→ Projection Layer
→ BiLSTM
→ Attention Pooling
→ LayerNorm
→ Softmax Classifier

#### Highlights

- Patient-level Cross Validation
- SpecAugment
- WeightedRandomSampler
- Early Stopping
- Youden's J Threshold Optimization

---

### 2. BiLSTM_Paper_Replication.ipynb

Replication of:

Serrano et al.
"Automated Detection of Sleep Apnea from PSG Audio Recordings"

#### Goal

Reproduce published results and compare with current implementations.

#### Key Observation

Dataset class distribution differed from the paper.

To ensure fair comparison:

- AP clips undersampled
- Paper distribution reproduced (30% AP / 70% NAP)

---

## Evaluation Strategy

All experiments prioritize:

- Patient-level splitting
- Prevention of data leakage
- Clinically meaningful metrics

Metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Specificity
- ROC-AUC

---

## Dataset

PSG-Audio Apnea Dataset

Source:
https://www.kaggle.com/datasets/bryandarquea/psg-audio-apnea-audios

---

## Author Notes

These notebooks were developed for research, benchmarking, and model comparison purposes before deployment into the Streamlit application.
