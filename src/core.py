"""
Core utilities for voice authentication system.
Includes model loading, audio processing, and embedding extraction.
"""

import torch
import torchaudio
import numpy as np
import json
import whisper
from jiwer import wer
from speechbrain.inference.speaker import EncoderClassifier


# ------------------------------------
# Configuration
# ------------------------------------
DEFAULT_THRESHOLD = 0.80
MIN_AUDIO_LENGTH_SEC = 5.0  # Increased from 3.0 for better quality
REQUIRED_ENROLLMENT_SAMPLES = 3  # Mandatory number of samples for enrollment
MAX_AUDIO_LENGTH_SEC = 15.0

# Score fusion configuration
TOP_K_SAMPLES = 2  # Use top-2 of 3 samples for averaging
SCORE_WEIGHT_TOP_K = 0.6  # 60% weight on top-k average
SCORE_WEIGHT_CENTROID = 0.4  # 40% weight on centroid

# Anti-spoofing configuration
WER_THRESHOLD = 0.5  # Accept if Word Error Rate < 50%
ENABLE_TEXT_VERIFICATION = True  # Enable/disable text verification


# ------------------------------------
# Load ECAPA-TDNN pretrained model
# ------------------------------------
model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="ecapa",
    run_opts={"device": "cpu"},
)


# ------------------------------------
# Load Whisper model for ASR
# ------------------------------------
print("Loading Whisper model (tiny)...")
whisper_model = whisper.load_model("tiny")
print("✅ Whisper model loaded")


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


def compute_centroid(embeddings: list) -> np.ndarray:
    """
    Compute normalized centroid of multiple embeddings.

    Args:
        embeddings: List of numpy arrays (embeddings)

    Returns:
        Normalized centroid embedding
    """
    # Stack embeddings and compute mean
    embeddings_array = np.stack([emb.flatten() for emb in embeddings])
    centroid = np.mean(embeddings_array, axis=0)

    # Normalize the centroid
    centroid = centroid / (np.linalg.norm(centroid) + 1e-9)

    return centroid.astype("float32")


def compute_weighted_score(
    sample_scores: list, centroid_score: float, top_k: int = TOP_K_SAMPLES
) -> dict:
    """
    Compute final verification score using weighted average of top-k and centroid.

    Args:
        sample_scores: List of similarity scores from individual samples
        centroid_score: Similarity score from centroid
        top_k: Number of top scores to average (default: 2)

    Returns:
        Dictionary with:
        - final_score: Weighted average score
        - top_k_avg: Average of top-k samples
        - top_k_indices: Indices of top-k samples (1-indexed)
        - best_match_score: Best individual sample score
        - best_match_index: Index of best sample (1-indexed)
        - centroid_score: Centroid similarity score
        - strategy_breakdown: Human-readable explanation
    """
    # Sort scores and get top-k
    sorted_indices = sorted(
        range(len(sample_scores)), key=lambda i: sample_scores[i], reverse=True
    )
    top_k_indices = sorted_indices[:top_k]
    top_k_scores = [sample_scores[i] for i in top_k_indices]
    top_k_avg = float(np.mean(top_k_scores))

    # Best match (for comparison)
    best_match_idx = sorted_indices[0]
    best_match_score = sample_scores[best_match_idx]

    # Weighted final score
    final_score = (
        SCORE_WEIGHT_TOP_K * top_k_avg + SCORE_WEIGHT_CENTROID * centroid_score
    )

    # Create strategy breakdown
    top_k_samples_str = ", ".join([f"#{i+1}" for i in sorted(top_k_indices)])
    strategy_breakdown = (
        f"Final score computed as:\n"
        f"  • Top-{top_k} avg (samples {top_k_samples_str}): {top_k_avg:.3f} × {SCORE_WEIGHT_TOP_K:.1f} = {SCORE_WEIGHT_TOP_K * top_k_avg:.3f}\n"
        f"  • Centroid (all samples): {centroid_score:.3f} × {SCORE_WEIGHT_CENTROID:.1f} = {SCORE_WEIGHT_CENTROID * centroid_score:.3f}\n"
        f"  • Final: {final_score:.3f}"
    )

    return {
        "final_score": final_score,
        "top_k_avg": top_k_avg,
        "top_k_indices": [i + 1 for i in sorted(top_k_indices)],  # 1-indexed
        "best_match_score": best_match_score,
        "best_match_index": best_match_idx + 1,  # 1-indexed
        "centroid_score": centroid_score,
        "strategy_breakdown": strategy_breakdown,
    }


def verify_spoken_text(audio_tuple, expected_text: str) -> dict:
    """
    Verify that spoken audio matches expected text using Whisper ASR.
    This provides anti-spoofing protection against replay attacks and TTS.

    Args:
        audio_tuple: (sample_rate, waveform_numpy)
        expected_text: The text that should have been spoken

    Returns:
        Dictionary with:
        - transcription: What was actually said
        - expected: What should have been said
        - wer_score: Word Error Rate (0.0 = perfect, 1.0 = complete mismatch)
        - passed: Boolean indicating if verification passed
        - message: Human-readable result message
    """
    sr, wav_np = audio_tuple

    # Convert to float32 and normalize to [-1, 1] if needed
    if wav_np.dtype != np.float32:
        wav_np = wav_np.astype(np.float32)

    # Ensure mono
    if wav_np.ndim > 1:
        wav_np = wav_np.mean(axis=1)

    # Resample to 16kHz if needed (Whisper expects 16kHz)
    if sr != 16000:
        wav_tensor = torch.from_numpy(wav_np)
        wav_tensor = torchaudio.functional.resample(
            wav_tensor, orig_freq=sr, new_freq=16000
        )
        wav_np = wav_tensor.numpy()

    # Transcribe using Whisper
    try:
        result = whisper_model.transcribe(
            wav_np, language="vi", task="transcribe", fp16=False, verbose=False
        )
        transcription = result["text"].strip()
    except Exception as e:
        print(f"Whisper transcription error: {e}")
        return {
            "transcription": "",
            "expected": expected_text,
            "wer_score": 1.0,
            "passed": False,
            "message": f"⚠️ Transcription failed: {str(e)}",
        }

    # Normalize texts for comparison (lowercase, remove extra spaces)
    expected_normalized = " ".join(expected_text.lower().split())
    transcribed_normalized = " ".join(transcription.lower().split())

    # Calculate Word Error Rate
    if expected_normalized and transcribed_normalized:
        wer_score = wer(expected_normalized, transcribed_normalized)
    else:
        wer_score = 1.0  # Complete mismatch if either is empty

    # Check if passed
    passed = wer_score <= WER_THRESHOLD

    # Create message
    if passed:
        message = (
            f"🔒 Text Verification: ✅ PASSED\n"
            f'Expected: "{expected_text}"\n'
            f'Detected: "{transcription}"\n'
            f"WER: {wer_score:.2f} (threshold: {WER_THRESHOLD:.2f})"
        )
    else:
        message = (
            f"🔒 Text Verification: ❌ FAILED\n"
            f'Expected: "{expected_text}"\n'
            f'Detected: "{transcription}"\n'
            f"WER: {wer_score:.2f} > threshold: {WER_THRESHOLD:.2f}\n\n"
            f"🚨 ANTI-SPOOF: Possible replay attack or wrong text!\n"
            f"Please re-record and read the displayed text carefully."
        )

    print(
        f"Text verification - Expected: '{expected_text}', Got: '{transcription}', WER: {wer_score:.3f}, Passed: {passed}"
    )

    return {
        "transcription": transcription,
        "expected": expected_text,
        "wer_score": wer_score,
        "passed": passed,
        "message": message,
    }


def analyze_audio_quality(wav_np: np.ndarray, sr: int) -> dict:
    """
    Analyze audio quality and provide diagnostic information.

    Args:
        wav_np: Audio waveform as numpy array
        sr: Sample rate

    Returns:
        Dictionary with diagnostic information
    """
    audio_length_sec = wav_np.shape[0] / sr
    max_amplitude = float(np.max(np.abs(wav_np)))

    # Detect if audio is too quiet
    is_quiet = max_amplitude < 0.1

    # Detect if audio is clipped (too loud)
    is_clipped = max_amplitude > 0.95

    # Estimate Signal-to-Noise Ratio (simple energy-based)
    # Use first/last 0.5s as potential silence/noise reference
    noise_samples = min(int(0.5 * sr), len(wav_np) // 4)
    if len(wav_np) > noise_samples * 2:
        noise_level = np.mean(np.abs(wav_np[:noise_samples]))
        signal_level = np.mean(np.abs(wav_np[noise_samples:-noise_samples]))
        snr_estimate = signal_level / (noise_level + 1e-9)
        has_noise = snr_estimate < 3.0  # Low SNR indicates noise
    else:
        snr_estimate = None
        has_noise = False

    return {
        "length_sec": audio_length_sec,
        "max_amplitude": max_amplitude,
        "is_quiet": is_quiet,
        "is_clipped": is_clipped,
        "has_noise": has_noise,
        "snr_estimate": snr_estimate,
    }


def generate_diagnostic_message(diagnostics: dict, context: str = "general") -> str:
    """
    Generate human-readable diagnostic message with suggestions.

    Args:
        diagnostics: Dictionary from analyze_audio_quality()
        context: Either "enrollment" or "login" or "general"

    Returns:
        Formatted diagnostic message with suggestions
    """
    issues = []
    suggestions = []

    # Check length
    if diagnostics["length_sec"] < MIN_AUDIO_LENGTH_SEC:
        issues.append(
            f"Audio too short ({diagnostics['length_sec']:.1f}s, minimum: {MIN_AUDIO_LENGTH_SEC:.0f}s)"
        )
        suggestions.append(
            f"• Record at least {MIN_AUDIO_LENGTH_SEC:.0f} seconds of clear speech"
        )

    # Check amplitude
    if diagnostics["is_quiet"]:
        issues.append(
            f"Audio very quiet (max amplitude: {diagnostics['max_amplitude']:.2f})"
        )
        suggestions.append("• Speak louder or move the microphone closer")

    if diagnostics["is_clipped"]:
        issues.append(
            f"Audio clipped/distorted (max amplitude: {diagnostics['max_amplitude']:.2f})"
        )
        suggestions.append("• Speak softer or move the microphone further away")

    # Check noise
    if diagnostics["has_noise"]:
        issues.append("Background noise detected")
        suggestions.append("• Record in a quieter environment")
        if context == "enrollment":
            suggestions.append(
                "• Consider re-enrolling in a quiet space for better accuracy"
            )

    # Build message
    if not issues:
        return ""

    msg = "\n\n📊 Audio Diagnostics:\n"
    msg += f"• Length: {diagnostics['length_sec']:.1f}s\n"
    msg += f"• Amplitude: {diagnostics['max_amplitude']:.2f}\n"
    if diagnostics["snr_estimate"] is not None:
        msg += f"• Signal quality: {'Good' if diagnostics['snr_estimate'] > 5 else 'Poor'}\n"

    if issues:
        msg += "\n⚠️ Issues detected:\n"
        for issue in issues:
            msg += f"• {issue}\n"

    if suggestions:
        msg += "\n💬 Suggestions:\n"
        for suggestion in suggestions:
            msg += f"{suggestion}\n"

    return msg
