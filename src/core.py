"""
Core utilities for voice authentication system.
Includes model loading, audio processing, and embedding extraction.
"""

import torch
import torchaudio
import numpy as np
import json
from speechbrain.inference.speaker import EncoderClassifier


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
# Initialize Enrollment Texts
# ------------------------------------
def load_enrollment_texts():
    """Load enrollment texts from JSON file"""
    try:
        with open("enrollment_texts.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("voice_enroll_texts", [])
    except FileNotFoundError:
        print("⚠️ enrollment_texts.json not found. Using empty list.")
        return []
    except json.JSONDecodeError as e:
        print(f"⚠️ Error parsing enrollment_texts.json: {e}")
        return []


# Load enrollment texts from JSON file
ENROLLMENT_TEXTS = load_enrollment_texts()
if ENROLLMENT_TEXTS:
    print(f"📝 Loaded {len(ENROLLMENT_TEXTS)} enrollment texts from JSON file")


# ------------------------------------
# Audio utilities
# ------------------------------------
def load_audio_file(filepath: str):
    """Load audio file using torchaudio and return (sr, wav_np)"""
    wav, sr = torchaudio.load(filepath)
    # Convert to mono if stereo
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    # Convert to numpy array
    wav_np = wav.squeeze(0).numpy()
    return int(sr), wav_np


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
