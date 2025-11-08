import torch
import torchaudio
import librosa
import numpy as np
import gradio as gr
import warnings
from speechbrain.inference.speaker import EncoderClassifier
from db import save_multiple_embeddings, load_embedding, list_users

# Suppress PySoundFile and librosa warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")


# ------------------------------------
# Configuration
# ------------------------------------
DEFAULT_THRESHOLD = 0.80
MIN_AUDIO_LENGTH_SEC = 3.0
MAX_AUDIO_LENGTH_SEC = 15.0


# ------------------------------------
# Load ECAPA-TDNN pretrained model
# ------------------------------------
model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="ecapa",
    run_opts={"device": "cpu"},
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


def normalize_audio_length(
    wav: torch.Tensor,
    min_length_sec: float = 3.0,
    max_length_sec: float = 15.0,
    sr: int = 16000,
) -> torch.Tensor:
    """Ensure audio is within min/max length bounds"""
    min_length = int(min_length_sec * sr)
    max_length = int(max_length_sec * sr)
    current_length = wav.shape[-1]

    if current_length < min_length:
        # Pad with zeros if too short
        pad_length = min_length - current_length
        wav = torch.nn.functional.pad(wav, (0, pad_length))
        print(
            f"⚠️ Audio too short ({current_length/sr:.2f}s), padded to {min_length_sec}s"
        )
    elif current_length > max_length:
        # Trim if too long
        wav = wav[..., :max_length]
        print(
            f"⚠️ Audio too long ({current_length/sr:.2f}s), trimmed to {max_length_sec}s"
        )

    return wav


def extract_embedding(audio_tuple) -> np.ndarray:
    """
    Convert audio (sr, wav) to ECAPA-TDNN embedding (192d vector).
    Audio is normalized to be within MIN_AUDIO_LENGTH_SEC and MAX_AUDIO_LENGTH_SEC.
    """
    sr, wav_np = audio_tuple
    wav = to_16k_mono(sr, wav_np)

    # Ensure audio is within reasonable bounds
    wav = normalize_audio_length(
        wav,
        min_length_sec=MIN_AUDIO_LENGTH_SEC,
        max_length_sec=MAX_AUDIO_LENGTH_SEC,
        sr=16000,
    )

    emb = model.encode_batch(wav)  # tensor [1, 192]
    return emb.squeeze(0).detach().cpu().numpy().astype("float32")


def cosine_similarity(a, b):
    """Compute cosine similarity between two normalized vectors"""
    a = a.flatten()
    b = b.flatten()
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


# ------------------------------------
# Business logic (Enroll / Login)
# ------------------------------------


def login(username, audio, threshold):
    if not username:
        return "⚠️ Please enter a username.", None
    if audio is None:
        return "⚠️ Please record your voice.", None

    # Handle both filepath (string) and tuple (sr, wav_np) formats
    if isinstance(audio, str):
        print(f"Audio received as filepath: {audio}")
        # Use librosa to load audio (supports various formats including m4a)
        wav_np, sr = librosa.load(audio, sr=None, mono=True)
        audio_tuple = (sr, wav_np)
    else:
        sr, wav_np = audio
        audio_tuple = audio

    print(
        f"Audio info: sr={sr}, shape={wav_np.shape}, max={wav_np.max()}, min={wav_np.min()}"
    )

    stored_embeddings = load_embedding(username)
    if stored_embeddings is None:
        return f"❌ User '{username}' not found. Please enroll first."

    # Check audio length and provide warning
    audio_length_sec = wav_np.shape[0] / sr
    warning_msg = ""
    if audio_length_sec < 2.0:
        warning_msg = f"\n\n⚠️ Warning: Audio is very short ({audio_length_sec:.1f}s). For better accuracy, record at least 3-5 seconds of speech."
    elif audio_length_sec < 3.0:
        warning_msg = f"\n\n💡 Tip: Audio is short ({audio_length_sec:.1f}s). Recording 3-5+ seconds may improve accuracy."

    new_emb = extract_embedding(audio_tuple)
    print(f"New embedding shape: {new_emb.shape}, norm: {np.linalg.norm(new_emb):.3f}")

    # Compare against all stored embeddings and use the best (maximum) score
    scores = []
    for i, stored_emb in enumerate(stored_embeddings):
        score = cosine_similarity(stored_emb, new_emb)
        scores.append(score)
        print(f"Similarity to stored sample {i+1}: {score:.4f}")

    best_score = max(scores)
    print(f"Best score: {best_score:.4f}, Threshold: {threshold:.2f}")

    if best_score >= threshold:
        return f"✅ SUCCESS — score={best_score:.3f} ≥ threshold={threshold:.2f} (matched sample {scores.index(best_score)+1}/{len(scores)}){warning_msg}"
    else:
        return f"❌ DENIED — score={best_score:.3f} < threshold={threshold:.2f}{warning_msg}"


def enroll(username, audio1, audio2, audio3):
    if not username:
        return "⚠️ Please enter a username.", None

    # Collect all provided audio samples
    audio_samples = []
    for i, audio in enumerate([audio1, audio2, audio3], 1):
        if audio is not None:
            audio_samples.append((i, audio))

    if len(audio_samples) == 0:
        return "⚠️ Please record at least one voice sample.", None

    # Extract embeddings from all samples
    embeddings = []
    short_samples = []
    for idx, audio in audio_samples:
        # Handle both filepath (string) and tuple (sr, wav_np) formats
        if isinstance(audio, str):
            print(f"Sample {idx}: Audio received as filepath: {audio}")
            wav_np, sr = librosa.load(audio, sr=None, mono=True)
            audio_tuple = (sr, wav_np)
        else:
            sr, wav_np = audio
            audio_tuple = audio

        print(
            f"Sample {idx}: sr={sr}, shape={wav_np.shape}, max={wav_np.max()}, min={wav_np.min()}"
        )

        # Check audio length
        audio_length_sec = wav_np.shape[0] / sr
        if audio_length_sec < 3.0:
            short_samples.append((idx, audio_length_sec))

        emb = extract_embedding(audio_tuple)
        embeddings.append(emb)
        print(f"Sample {idx} embedding norm: {np.linalg.norm(emb):.3f}")

    # Store all embeddings separately (not averaged)
    save_multiple_embeddings(username, embeddings)

    # Create warning message if any samples were short
    warning_msg = ""
    if short_samples:
        short_list = ", ".join(
            [f"Sample {idx} ({length:.1f}s)" for idx, length in short_samples]
        )
        warning_msg = f"\n\n⚠️ Warning: Short audio detected - {short_list}. For better verification accuracy, consider re-enrolling with 3-5+ seconds of speech per sample."

    users_list = get_enrolled_users()
    return (
        f"✅ Enrolled '{username}' with {len(embeddings)} sample(s) — {embeddings[0].shape[-1]}D embeddings stored separately{warning_msg}",
        users_list,
    )


def get_enrolled_users():
    """Get formatted list of enrolled users"""
    users = list_users()
    if not users:
        return "No enrollments yet"
    return f"**Enrolled Users ({len(users)}):**\n" + "\n".join(
        f"• {user}" for user in sorted(users)
    )


# ------------------------------------
# Gradio UI
# ------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# 🔐 Voice Login — ECAPA-TDNN (SpeechBrain)")

    with gr.Tab("Login"):
        u2 = gr.Textbox(label="Username", placeholder="e.g: phatpham9")
        a2 = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="Record 3-10s of speech",
        )
        th = gr.Slider(
            0.50, 0.98, value=DEFAULT_THRESHOLD, step=0.01, label="Threshold (cosine)"
        )
        out2 = gr.Textbox(label="Result")
        gr.Button("Login").click(login, inputs=[u2, a2, th], outputs=[out2])

    with gr.Tab("Enroll"):
        u = gr.Textbox(label="Username", placeholder="e.g: phatpham9")
        gr.Markdown("### Record 1-3 voice samples for better accuracy")
        gr.Markdown(
            "*Tip: Use similar length recordings (3-10s each). Avoid very short clips with lots of padding.*"
        )
        a1 = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="Sample 1 (Required) — Record 3-10s",
        )
        a2 = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="Sample 2 (Optional) — Record 3-10s",
        )
        a3 = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="Sample 3 (Optional) — Record 3-10s",
        )
        out = gr.Textbox(label="Result")
        enrolled_list = gr.Markdown(value=get_enrolled_users())
        gr.Button("Enroll").click(
            enroll, inputs=[u, a1, a2, a3], outputs=[out, enrolled_list]
        )


if __name__ == "__main__":
    demo.launch()
