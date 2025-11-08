import torch
import torchaudio
import numpy as np
import gradio as gr
from speechbrain.pretrained import EncoderClassifier

from db import save_embedding, load_embedding


# ------------------------------------
# Load ECAPA-TDNN pretrained
# ------------------------------------
model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="ecapa",
    run_opts={"device": "cpu"},  # Mac M1 có thể đổi thành "mps" nếu torch hỗ trợ
)


# ------------------------------------
# Audio utilities
# ------------------------------------
def to_16k_mono(sr: int, wav_np: np.ndarray) -> torch.Tensor:
    """Gradio audio input -> torch tensor [1, T] mono 16k"""
    if wav_np.ndim == 2:
        wav_np = wav_np.mean(axis=1)  # stereo -> mono
    wav = torch.tensor(wav_np, dtype=torch.float32)

    if sr != 16000:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=16000)

    return wav.unsqueeze(0)  # [1, T]


def extract_embedding(audio_tuple) -> np.ndarray:
    """Convert (sr, wav) -> ECAPA embedding (192d vector)"""
    sr, wav_np = audio_tuple
    wav = to_16k_mono(sr, wav_np)

    emb = model.encode_batch(wav)  # tensor [1, 192]
    return emb.squeeze(0).detach().cpu().numpy().astype("float32")


def cosine_similarity(a, b):
    a = a.flatten()
    b = b.flatten()
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


# ------------------------------------
# Business logic (Enroll / Login)
# ------------------------------------
DEFAULT_THRESHOLD = 0.85


def enroll(username, audio):
    if not username:
        return "⚠️ Vui lòng nhập username.", None
    if audio is None:
        return "⚠️ Vui lòng ghi âm giọng nói.", None

    # Handle both filepath (string) and tuple (sr, wav_np) formats
    if isinstance(audio, str):
        print(f"Audio received as filepath: {audio}")
        waveform, sr = torchaudio.load(audio)
        wav_np = waveform.numpy()
        if wav_np.ndim > 1:
            wav_np = wav_np[0]  # Take first channel if stereo
        audio_tuple = (sr, wav_np)
    else:
        sr, wav_np = audio
        audio_tuple = audio

    print(
        f"Audio info: sr={sr}, shape={wav_np.shape}, max={wav_np.max()}, min={wav_np.min()}"
    )

    emb = extract_embedding(audio_tuple)
    save_embedding(username, emb)

    # Normalize audio for playback (ensure it's in [-1, 1] range)
    wav_normalized = wav_np.copy()
    max_val = np.abs(wav_normalized).max()
    if max_val > 0:
        wav_normalized = wav_normalized / max_val

    playback_audio = (sr, wav_normalized)

    return f"✅ Enrolled '{username}' — embedding {emb.shape[0]}D", playback_audio


def login(username, audio, threshold):
    if not username:
        return "⚠️ Vui lòng nhập username.", None
    if audio is None:
        return "⚠️ Vui lòng ghi âm giọng nói.", None

    # Handle both filepath (string) and tuple (sr, wav_np) formats
    if isinstance(audio, str):
        print(f"Audio received as filepath: {audio}")
        waveform, sr = torchaudio.load(audio)
        wav_np = waveform.numpy()
        if wav_np.ndim > 1:
            wav_np = wav_np[0]  # Take first channel if stereo
        audio_tuple = (sr, wav_np)
    else:
        sr, wav_np = audio
        audio_tuple = audio

    print(
        f"Audio info: sr={sr}, shape={wav_np.shape}, max={wav_np.max()}, min={wav_np.min()}"
    )

    stored = load_embedding(username)
    if stored is None:
        # Normalize audio for playback
        wav_normalized = wav_np.copy()
        max_val = np.abs(wav_normalized).max()
        if max_val > 0:
            wav_normalized = wav_normalized / max_val
        return f"❌ Không tìm thấy user '{username}'. Hãy Enroll trước.", (
            sr,
            wav_normalized,
        )

    new_emb = extract_embedding(audio_tuple)
    score = cosine_similarity(stored, new_emb)

    # Normalize audio for playback
    wav_normalized = wav_np.copy()
    max_val = np.abs(wav_normalized).max()
    if max_val > 0:
        wav_normalized = wav_normalized / max_val

    playback_audio = (sr, wav_normalized)

    if score >= threshold:
        return (
            f"✅ SUCCESS — score={score:.3f} ≥ threshold={threshold:.2f}",
            playback_audio,
        )
    else:
        return (
            f"❌ DENIED — score={score:.3f} < threshold={threshold:.2f}",
            playback_audio,
        )


# ------------------------------------
# Gradio UI
# ------------------------------------
with gr.Blocks() as app:
    gr.Markdown(
        "# 🔐 Voice Login — ECAPA-TDNN (SpeechBrain)\nText-independent • Tiếng Việt OK • No fine-tune"
    )

    with gr.Tab("Enroll"):
        u = gr.Textbox(label="Username", placeholder="vd: phatpham")
        a = gr.Audio(
            sources=["microphone", "upload"],
            type="numpy",
            label="Record ~5s",
        )
        out = gr.Textbox(label="Result")
        playback = gr.Audio(label="Playback - Listen to your recording")
        gr.Button("Enroll").click(enroll, inputs=[u, a], outputs=[out, playback])

    with gr.Tab("Login"):
        u2 = gr.Textbox(label="Username")
        a2 = gr.Audio(
            sources=["microphone", "upload"],
            type="numpy",
            label="Record ~3–5s",
        )
        th = gr.Slider(
            0.50, 0.98, value=DEFAULT_THRESHOLD, step=0.01, label="Threshold (cosine)"
        )
        out2 = gr.Textbox(label="Result")
        playback2 = gr.Audio(label="Playback - Listen to your recording")
        gr.Button("Login").click(login, inputs=[u2, a2, th], outputs=[out2, playback2])


app.launch()
