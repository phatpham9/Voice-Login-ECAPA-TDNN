"""
Core utilities for voice authentication system.
Includes model loading, audio processing, and embedding extraction.
"""

import torch
import torchaudio
import numpy as np
import json
import whisper
import webrtcvad
import struct
from jiwer import wer
from speechbrain.inference.speaker import EncoderClassifier


# ------------------------------------
# Configuration
# ------------------------------------
DEFAULT_THRESHOLD = 0.80
MIN_AUDIO_LENGTH_SEC = 5.0  # Increased from 3.0 for better quality
REQUIRED_ENROLLMENT_SAMPLES = 3  # Mandatory number of samples for enrollment
MAX_AUDIO_LENGTH_SEC = 15.0

# Audio preprocessing configuration
AMPLITUDE_NORMALIZATION_TARGET = "peak"  # "peak" or "rms"
PEAK_NORMALIZATION_DB = -3.0  # Target peak level in dBFS (-3dB = ~0.708)
RMS_NORMALIZATION_DB = -23.0  # Target RMS level in dBFS (LUFS-like)
CLIPPING_THRESHOLD = 0.99  # Maximum amplitude before clipping
CLIPPING_METHOD = "clamp"  # "clamp" or "soft_clip"

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
print("✅ ECAPA-TDNN model loaded")


# ------------------------------------
# Load Whisper model for ASR
# ------------------------------------
whisper_model = whisper.load_model("tiny", download_root="whisper")
print("✅ Whisper model (tiny) loaded")


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


def apply_webrtc_vad(
    wav_np: np.ndarray,
    sr: int = 16000,
    frame_duration_ms: int = 30,
    aggressiveness: int = 2,
    padding_duration_ms: int = 300,
) -> np.ndarray:
    """
    Apply WebRTC VAD to extract speech-only segments and remove leading/trailing silence.

    Args:
        wav_np: Audio numpy array (float32, range -1.0 to 1.0)
        sr: Sample rate (must be 8000, 16000, 32000, or 48000)
        frame_duration_ms: Frame size in ms (10, 20, or 30)
        aggressiveness: VAD aggressiveness mode (0-3, higher = more aggressive)
            0 = Quality mode (most permissive, least aggressive)
            1 = Low bitrate mode
            2 = Aggressive mode (recommended for voice authentication)
            3 = Very aggressive mode
        padding_duration_ms: Amount of padding (ms) to add around speech segments

    Returns:
        Speech-only audio segments concatenated as float32 numpy array
    """
    # Validate sample rate
    if sr not in [8000, 16000, 32000, 48000]:
        raise ValueError(f"Sample rate must be 8000, 16000, 32000, or 48000. Got {sr}")

    # Validate frame duration
    if frame_duration_ms not in [10, 20, 30]:
        raise ValueError(
            f"Frame duration must be 10, 20, or 30 ms. Got {frame_duration_ms}"
        )

    vad = webrtcvad.Vad(aggressiveness)

    # Convert float32 to int16 PCM (WebRTC VAD requirement)
    if wav_np.dtype == np.float32 or wav_np.dtype == np.float64:
        # Ensure values are in range [-1.0, 1.0]
        wav_np = np.clip(wav_np, -1.0, 1.0)
        wav_int16 = (wav_np * 32767).astype(np.int16)
    elif wav_np.dtype == np.int16:
        wav_int16 = wav_np
    else:
        # Convert other types to float32 first
        wav_np = wav_np.astype(np.float32)
        wav_np = np.clip(wav_np, -1.0, 1.0)
        wav_int16 = (wav_np * 32767).astype(np.int16)

    # Frame parameters
    frame_length = int(sr * frame_duration_ms / 1000)
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)

    # Split audio into frames
    frames = []
    for i in range(0, len(wav_int16), frame_length):
        frame = wav_int16[i : i + frame_length]
        if len(frame) == frame_length:  # Only process complete frames
            frames.append(frame)

    if not frames:
        print("⚠️ VAD: No complete frames found, returning original audio")
        return wav_np

    # Apply VAD to each frame
    voiced_flags = []
    for frame in frames:
        try:
            # Convert frame to bytes for VAD
            frame_bytes = struct.pack("%dh" % len(frame), *frame)
            is_speech = vad.is_speech(frame_bytes, sr)
            voiced_flags.append(is_speech)
        except Exception as e:
            print(f"⚠️ VAD frame processing error: {e}")
            voiced_flags.append(True)  # Assume speech on error

    # Find speech segments with padding
    speech_frames = []
    num_voiced = sum(voiced_flags)

    if num_voiced == 0:
        print("⚠️ VAD: No speech detected, returning original audio")
        return wav_np

    # Add padding around speech segments
    for i, is_speech in enumerate(voiced_flags):
        if is_speech:
            # Add padding frames before and after
            start_idx = max(0, i - num_padding_frames)
            end_idx = min(len(frames), i + num_padding_frames + 1)
            for j in range(start_idx, end_idx):
                if j not in [idx for idx, frame in speech_frames]:  # Avoid duplicates
                    speech_frames.append((j, frames[j]))

    # Sort by index and extract frames
    speech_frames.sort(key=lambda x: x[0])
    speech_audio_int16 = np.concatenate([frame for _, frame in speech_frames])

    # Convert back to float32
    speech_audio_float32 = speech_audio_int16.astype(np.float32) / 32767.0

    # Calculate statistics
    original_duration = len(wav_np) / sr
    trimmed_duration = len(speech_audio_float32) / sr
    removed_duration = original_duration - trimmed_duration

    print(
        f"🎤 VAD: Removed {removed_duration:.2f}s of silence "
        f"({removed_duration/original_duration*100:.1f}% of audio)"
    )
    print(
        f"   Original: {original_duration:.2f}s → Speech-only: {trimmed_duration:.2f}s"
    )

    return speech_audio_float32


def trim_silence_energy_based(
    wav: torch.Tensor,
    sr: int = 16000,
    top_db: float = 30.0,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> torch.Tensor:
    """
    Remove leading and trailing silence based on energy threshold (fallback method).

    Args:
        wav: Audio tensor [1, T] or [T]
        sr: Sample rate (default 16000)
        top_db: Threshold in dB below peak to consider as silence
        frame_length: Frame size for analysis
        hop_length: Hop size between frames

    Returns:
        Trimmed audio tensor
    """
    wav_2d = wav.squeeze(0) if wav.ndim == 2 else wav

    if wav_2d.shape[0] < frame_length:
        return wav  # Too short to analyze

    # Calculate energy with padding
    frames = wav_2d.unfold(0, frame_length, hop_length)
    energy = torch.sqrt(torch.mean(frames**2, dim=1))

    # Convert to dB scale
    energy_db = 20 * torch.log10(energy + 1e-10)
    threshold = energy_db.max() - top_db

    # Find first and last frames above threshold
    above_threshold = energy_db > threshold
    if not above_threshold.any():
        return wav  # Keep original if all is silence

    start_frame = above_threshold.nonzero()[0].item()
    end_frame = above_threshold.nonzero()[-1].item() + 1

    # Convert frame indices to sample indices
    start_sample = start_frame * hop_length
    end_sample = min(end_frame * hop_length + frame_length, wav_2d.shape[0])

    trimmed = wav_2d[start_sample:end_sample]

    # Calculate removed duration
    original_duration = wav_2d.shape[0] / sr
    trimmed_duration = trimmed.shape[0] / sr
    removed_duration = original_duration - trimmed_duration

    if removed_duration > 0.1:  # Only log if significant
        print(
            f"🎤 Energy-based trim: Removed {removed_duration:.2f}s of silence "
            f"({removed_duration/original_duration*100:.1f}% of audio)"
        )

    return trimmed.unsqueeze(0) if wav.ndim == 2 else trimmed


def normalize_amplitude(
    wav: torch.Tensor,
    target_level: str = "peak",
    peak_db: float = -3.0,
    rms_db: float = -23.0,
) -> torch.Tensor:
    """
    Normalize audio amplitude to target level.

    Args:
        wav: Audio tensor [1, T] or [T]
        target_level: "peak" for peak normalization, "rms" for RMS normalization
        peak_db: Target peak level in dBFS (default -3.0 dBFS)
        rms_db: Target RMS level in dBFS (default -23.0 LUFS-like)

    Returns:
        Normalized audio tensor
    """
    wav_2d = wav.squeeze(0) if wav.ndim == 2 else wav

    if target_level == "peak":
        # Peak normalization to target dBFS
        current_peak = torch.max(torch.abs(wav_2d))
        if current_peak > 1e-6:  # Avoid division by zero
            target_peak = 10 ** (peak_db / 20.0)
            scale = target_peak / current_peak
            wav_normalized = wav_2d * scale

            print(
                f"🔊 Amplitude normalized: Peak {20*torch.log10(current_peak):.1f}dB → {peak_db:.1f}dBFS "
                f"(scale: {scale:.3f}x)"
            )
        else:
            wav_normalized = wav_2d
            print("⚠️ Audio is silent, skipping normalization")

    elif target_level == "rms":
        # RMS normalization
        current_rms = torch.sqrt(torch.mean(wav_2d**2))
        if current_rms > 1e-6:  # Avoid division by zero
            target_rms = 10 ** (rms_db / 20.0)
            scale = target_rms / current_rms
            wav_normalized = wav_2d * scale

            print(
                f"🔊 Amplitude normalized: RMS {20*torch.log10(current_rms):.1f}dB → {rms_db:.1f}dBFS "
                f"(scale: {scale:.3f}x)"
            )
        else:
            wav_normalized = wav_2d
            print("⚠️ Audio is silent, skipping normalization")
    else:
        wav_normalized = wav_2d
        print(f"⚠️ Unknown normalization target: {target_level}, skipping")

    return wav_normalized.unsqueeze(0) if wav.ndim == 2 else wav_normalized


def prevent_clipping(
    wav: torch.Tensor, threshold: float = 0.99, method: str = "clamp"
) -> torch.Tensor:
    """
    Prevent audio clipping/overload.

    Args:
        wav: Audio tensor
        threshold: Clipping threshold (default 0.99)
        method: "clamp" for hard limiting, "soft_clip" for tanh-based soft clipping

    Returns:
        Clipping-prevented audio
    """
    max_val = torch.max(torch.abs(wav))

    if max_val > threshold:
        if method == "clamp":
            # Hard clipping (simple)
            wav_clipped = torch.clamp(wav, -threshold, threshold)
            print(
                f"⚠️ Clipping prevented: Peak {max_val:.3f} clamped to ±{threshold:.2f}"
            )
        elif method == "soft_clip":
            # Soft clipping using tanh
            scale = threshold / 1.0  # Adjust sensitivity
            wav_clipped = threshold * torch.tanh(wav / scale)
            print(
                f"⚠️ Soft clipping applied: Peak {max_val:.3f} → {torch.max(torch.abs(wav_clipped)):.3f}"
            )
        else:
            wav_clipped = wav
    else:
        wav_clipped = wav

    return wav_clipped


def extract_embedding(audio_tuple, return_stats: bool = False):
    """
    Convert audio (sr, wav) to ECAPA-TDNN embedding (192d vector).
    Audio is preprocessed with VAD to remove silence and normalized to be
    within MIN_AUDIO_LENGTH_SEC and MAX_AUDIO_LENGTH_SEC.

    Args:
        audio_tuple: (sample_rate, waveform_numpy)
        return_stats: If True, return (embedding, preprocessing_stats) tuple

    Returns:
        If return_stats=False: numpy array (embedding)
        If return_stats=True: tuple (embedding, preprocessing_stats dict)
    """
    sr, wav_np = audio_tuple

    # Initialize preprocessing stats
    preprocessing_stats = {
        "original_duration": len(wav_np) / sr,
        "vad_method": None,
        "vad_removed_seconds": 0.0,
        "vad_removed_percent": 0.0,
        "amplitude_normalized": False,
        "amplitude_scale": 1.0,
        "original_peak_db": None,
        "normalized_peak_db": None,
        "clipping_applied": False,
    }

    # Step 1: Convert to 16kHz mono
    wav = to_16k_mono(sr, wav_np)
    original_length_16k = wav.shape[-1]

    # Step 2: Apply VAD to remove leading/trailing silence
    # Convert tensor back to numpy for VAD processing
    wav_np_16k = wav.squeeze(0).numpy() if wav.ndim == 2 else wav.numpy()

    try:
        # Use WebRTC VAD (aggressiveness=2 is recommended for voice authentication)
        wav_np_vad = apply_webrtc_vad(
            wav_np_16k,
            sr=16000,
            frame_duration_ms=30,
            aggressiveness=2,
            padding_duration_ms=300,
        )
        # Convert back to tensor
        wav = torch.from_numpy(wav_np_vad).unsqueeze(0)
        preprocessing_stats["vad_method"] = "WebRTC VAD"
        vad_removed = original_length_16k - wav.shape[-1]
        preprocessing_stats["vad_removed_seconds"] = vad_removed / 16000
        preprocessing_stats["vad_removed_percent"] = (
            (vad_removed / original_length_16k) * 100 if original_length_16k > 0 else 0
        )
    except Exception as e:
        print(f"⚠️ VAD failed ({e}), falling back to energy-based trimming")
        # Fallback to energy-based silence trimming
        wav = trim_silence_energy_based(wav, sr=16000, top_db=30.0)
        preprocessing_stats["vad_method"] = "Energy-based"
        vad_removed = original_length_16k - wav.shape[-1]
        preprocessing_stats["vad_removed_seconds"] = vad_removed / 16000
        preprocessing_stats["vad_removed_percent"] = (
            (vad_removed / original_length_16k) * 100 if original_length_16k > 0 else 0
        )

    # Step 3: Normalize amplitude (before length normalization to avoid padding affecting normalization)
    wav_before_norm = wav.clone()
    original_peak = torch.max(torch.abs(wav_before_norm))
    preprocessing_stats["original_peak_db"] = (
        20 * torch.log10(original_peak + 1e-10).item()
    )

    wav = normalize_amplitude(
        wav,
        target_level=AMPLITUDE_NORMALIZATION_TARGET,
        peak_db=PEAK_NORMALIZATION_DB,
        rms_db=RMS_NORMALIZATION_DB,
    )

    normalized_peak = torch.max(torch.abs(wav))
    preprocessing_stats["normalized_peak_db"] = (
        20 * torch.log10(normalized_peak + 1e-10).item()
    )

    if original_peak > 1e-6:
        preprocessing_stats["amplitude_normalized"] = True
        preprocessing_stats["amplitude_scale"] = (
            normalized_peak / original_peak
        ).item()

    # Step 4: Prevent clipping after normalization
    wav_before_clip = wav.clone()
    wav = prevent_clipping(wav, threshold=CLIPPING_THRESHOLD, method=CLIPPING_METHOD)
    preprocessing_stats["clipping_applied"] = not torch.allclose(wav, wav_before_clip)

    # Step 5: Ensure audio is within reasonable bounds (after VAD trimming and normalization)
    wav = normalize_audio_length(
        wav,
        min_length_sec=MIN_AUDIO_LENGTH_SEC,
        max_length_sec=MAX_AUDIO_LENGTH_SEC,
        sr=16000,
    )

    # Step 6: Extract embedding
    emb = model.encode_batch(wav)  # tensor [1, 192]
    embedding = emb.squeeze(0).detach().cpu().numpy().astype("float32")

    if return_stats:
        return embedding, preprocessing_stats
    else:
        return embedding


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
        f"• Top-{top_k} avg (samples {top_k_samples_str}): {top_k_avg:.3f} × {SCORE_WEIGHT_TOP_K:.1f} = {SCORE_WEIGHT_TOP_K * top_k_avg:.3f}\n"
        f"• Centroid (all samples): {centroid_score:.3f} × {SCORE_WEIGHT_CENTROID:.1f} = {SCORE_WEIGHT_CENTROID * centroid_score:.3f}\n"
        f"• Final: {final_score:.3f}"
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
