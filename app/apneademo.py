"""
Sleep Apnea Detection — Streamlit Demo
Supports: .mp3, .wav, .npy audio input
"""

import streamlit as st
import numpy as np
import librosa
import joblib
import io
import os
import tempfile
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Sleep Apnea Detector",
    page_icon="🫁",
    layout="centered",
)

# ─────────────────────────────────────────────
#  CONSTANTS (must match training config)
# ─────────────────────────────────────────────
SAMPLE_RATE  = 16_000
CLIP_LENGTH  = 10           # seconds
SAMPLE_PTS   = SAMPLE_RATE * CLIP_LENGTH   # 160,000
N_MFCC       = 40
N_MELS       = 64
MODEL_PATH   = "best_apnea_model.pkl"


# ══════════════════════════════════════════════
#  CORE FUNCTIONS  (replicated from notebook)
# ══════════════════════════════════════════════

def load_audio_bytes(file_bytes: bytes, filename: str) -> np.ndarray:
    """Load mp3 / wav / npy → flat float32 mono array at SAMPLE_RATE."""
    ext = os.path.splitext(filename)[-1].lower()

    if ext == ".npy":
        arr = np.frombuffer(file_bytes, dtype=np.float32)
        try:
            audio = np.load(io.BytesIO(file_bytes), allow_pickle=False).astype(np.float32)
        except Exception:
            audio = arr
        if audio.ndim > 1:
            axis  = 0 if audio.shape[0] < audio.shape[-1] else -1
            audio = audio.mean(axis=axis)
        return audio.ravel()

    elif ext in (".mp3", ".wav"):
        # Write to a temp file because librosa needs a path or file-like object
        suffix = ext
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            audio, _ = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
        finally:
            os.unlink(tmp_path)
        return audio.astype(np.float32)

    else:
        raise ValueError(f"Unsupported format: {ext}")


def preprocess_audio(audio: np.ndarray,
                     target_len: int = SAMPLE_PTS) -> np.ndarray:
    """Wrap-pad / trim to fixed length, then peak-normalise."""
    if len(audio) < target_len:
        repeats = int(np.ceil(target_len / len(audio)))
        audio   = np.tile(audio, repeats)
    audio = audio[:target_len]
    peak  = np.max(np.abs(audio))
    return audio / peak if peak > 1e-6 else audio


def extract_features(audio: np.ndarray,
                     sr:    int = SAMPLE_RATE,
                     n_mfcc: int = N_MFCC,
                     n_mels: int = N_MELS) -> np.ndarray:
    """276-dim feature vector — identical to notebook v3."""
    def stat(m):
        return np.concatenate([m.mean(axis=1), m.std(axis=1)])

    mfcc    = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc_d  = librosa.feature.delta(mfcc)
    mfcc_dd = librosa.feature.delta(mfcc, order=2)

    mel    = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_feat = np.array([mel_db.mean(), mel_db.std()])

    contrast      = librosa.feature.spectral_contrast(y=audio, sr=sr, n_bands=6)
    contrast_feat = stat(contrast)

    harmonic     = librosa.effects.harmonic(audio)
    tonnetz      = librosa.feature.tonnetz(y=harmonic, sr=sr)
    tonnetz_feat = stat(tonnetz)

    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    rolloff  = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    zcr      = librosa.feature.zero_crossing_rate(y=audio)
    rms      = librosa.feature.rms(y=audio)

    scalar_feat = np.array([
        centroid.mean(), centroid.std(),
        rolloff.mean(),  rolloff.std(),
        zcr.mean(),      zcr.std(),
        rms.mean(),      rms.std(),
    ])

    return np.concatenate([
        stat(mfcc), stat(mfcc_d), stat(mfcc_dd),
        mel_feat, contrast_feat, tonnetz_feat, scalar_feat,
    ])


# ══════════════════════════════════════════════
#  MODEL LOADER (cached)
# ══════════════════════════════════════════════

@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


# ══════════════════════════════════════════════
#  VISUALISATION HELPERS
# ══════════════════════════════════════════════

def plot_waveform_and_mel(audio: np.ndarray, sr: int = SAMPLE_RATE):
    """Return a matplotlib figure with waveform + mel spectrogram."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 4))

    # Waveform
    times = np.linspace(0, len(audio) / sr, num=len(audio))
    axes[0].plot(times, audio, linewidth=0.5, color="#2196F3")
    axes[0].set_title("Waveform", fontsize=11)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)

    # Mel spectrogram
    mel    = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(
        mel_db, sr=sr, x_axis="time", y_axis="mel", ax=axes[1], cmap="magma"
    )
    fig.colorbar(img, ax=axes[1], format="%+2.0f dB")
    axes[1].set_title("Mel Spectrogram", fontsize=11)

    plt.tight_layout()
    return fig


def probability_gauge(apnea_prob: float):
    """Return a matplotlib figure showing the apnea probability gauge."""
    fig, ax = plt.subplots(figsize=(4, 2.5))
    color = "#e53935" if apnea_prob >= 0.5 else "#43a047"
    ax.barh(["Apnea probability"], [apnea_prob],
            color=color, height=0.4, edgecolor="white")
    ax.barh(["Apnea probability"], [1 - apnea_prob],
            left=[apnea_prob], color="#e0e0e0", height=0.4, edgecolor="white")
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_title(f"Apnea Probability: {apnea_prob*100:.1f}%", fontweight="bold")
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════

st.title("🫁 Sleep Apnea Detector")
st.caption("Upload a PSG audio clip and the trained model will classify it as **Apnea** or **Non-Apnea**.")

st.divider()

# ── Model status ─────────────────────────────
bundle = load_model(MODEL_PATH)

with st.sidebar:
    st.header("⚙️ Model")
    if bundle is not None:
        st.success(f"Model loaded ✅\n\n**{bundle.get('model_name','Unknown')}**")
        st.caption(f"Feature dim: {bundle.get('feature_dim', '?')}")
    else:
        st.error("No model found ❌")
        st.markdown(
            f"""
            Place your trained model file at:

            ```
            {MODEL_PATH}
            ```

            Run the notebook's `main()` to generate it.
            The file is saved by `save_best_model()` at the end of training.
            """
        )

    st.divider()
    st.header("ℹ️ Config")
    st.json({
        "sample_rate": SAMPLE_RATE,
        "clip_length_sec": CLIP_LENGTH,
        "n_mfcc": N_MFCC,
        "n_mels": N_MELS,
    })

# ── File uploader ─────────────────────────────
uploaded = st.file_uploader(
    "Upload an audio file",
    type=["mp3", "wav", "npy"],
    help="Supported formats: .mp3 · .wav · .npy"
)

if uploaded is not None:
    st.audio(uploaded, format="audio/wav" if uploaded.name.endswith(".wav") else "audio/mpeg")

    with st.spinner("Loading and preprocessing audio…"):
        try:
            raw_audio = load_audio_bytes(uploaded.read(), uploaded.name)
            processed = preprocess_audio(raw_audio)
            st.success(
                f"✅ Loaded `{uploaded.name}` — "
                f"{len(raw_audio)/SAMPLE_RATE:.1f}s  →  preprocessed to {CLIP_LENGTH}s clip"
            )
        except Exception as e:
            st.error(f"Failed to load audio: {e}")
            st.stop()

    # ── Visualise ─────────────────────────────
    st.subheader("📊 Audio Visualisation")
    fig_audio = plot_waveform_and_mel(processed)
    st.pyplot(fig_audio, use_container_width=True)
    plt.close(fig_audio)

    st.divider()

    # ── Predict ───────────────────────────────
    st.subheader("🔍 Prediction")

    if bundle is None:
        st.warning("No model loaded — cannot predict. See sidebar for instructions.")
    else:
        with st.spinner("Extracting features and predicting…"):
            feat  = extract_features(processed).reshape(1, -1)
            feat  = bundle["scaler"].transform(feat)
            pred  = bundle["model"].predict(feat)[0]
            prob  = bundle["model"].predict_proba(feat)[0]   # [non-apnea, apnea]

        apnea_prob    = float(prob[1])
        non_apnea_prob = float(prob[0])
        label = "APNEA" if pred == 1 else "NON-APNEA"

        # Big result badge
        if pred == 1:
            st.error(f"## 🚨 Result: **{label}**")
        else:
            st.success(f"## ✅ Result: **{label}**")

        # Probability columns
        col1, col2 = st.columns(2)
        col1.metric("🟢 Non-Apnea probability", f"{non_apnea_prob*100:.1f}%")
        col2.metric("🔴 Apnea probability",     f"{apnea_prob*100:.1f}%")

        # Gauge bar
        fig_gauge = probability_gauge(apnea_prob)
        st.pyplot(fig_gauge, use_container_width=False)
        plt.close(fig_gauge)

        # Feature vector peek
        with st.expander("🔬 Feature vector (276-dim)", expanded=False):
            feat_np = feat.ravel()
            st.write(f"Shape: `{feat_np.shape}`   |   min: `{feat_np.min():.3f}`   max: `{feat_np.max():.3f}`")
            st.bar_chart(feat_np)

st.divider()
st.caption(
    "Model: Ensemble (SVM + Random Forest + XGBoost) trained on PSG-Audio Apnea dataset. "
    "Features: MFCC + Δ + ΔΔ, Mel spectrogram, Spectral contrast, Tonnetz, ZCR, RMS."
)