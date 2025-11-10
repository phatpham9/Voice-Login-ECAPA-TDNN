"""
Enroll UI tab for voice enrollment.
"""

import gradio as gr
import numpy as np
import random
import shutil
from pathlib import Path
from src.core import (
    ENROLLMENT_TEXTS,
    MIN_AUDIO_LENGTH_SEC,
    load_audio_file,
    extract_embedding,
)
from src.database import save_multiple_embeddings


def get_enroll_texts():
    """Get 3 random texts for enrollment"""
    if ENROLLMENT_TEXTS and len(ENROLLMENT_TEXTS) >= 3:
        texts = random.sample(ENROLLMENT_TEXTS, 3)
        return texts[0], texts[1], texts[2]
    elif ENROLLMENT_TEXTS:
        # If less than 3 texts available, use what we have and pad
        texts = ENROLLMENT_TEXTS.copy()
        while len(texts) < 3:
            texts.append("")
        return texts[0], texts[1], texts[2]
    else:
        return "", "", ""


def enroll(username, audio1, audio2, audio3):
    if not username:
        return "⚠️ Please enter a username.", None, None, None

    # Collect all provided audio samples
    audio_samples = []
    for i, audio in enumerate([audio1, audio2, audio3], 1):
        if audio is not None:
            audio_samples.append((i, audio))

    # Enforce 3 mandatory samples
    if len(audio_samples) < 3:
        return (
            f"❌ All 3 voice samples are required for enrollment. You provided {len(audio_samples)}/3 samples. Please record all samples.",
            None,
            None,
            None,
        )

    # Create audio_samples directory if it doesn't exist
    audio_dir = Path("audio_samples")
    audio_dir.mkdir(exist_ok=True)

    # Create user directory
    user_dir = audio_dir / username
    user_dir.mkdir(exist_ok=True)

    # Extract embeddings from all samples
    embeddings = []
    audio_lengths = []
    audio_file_paths = []
    short_samples = []

    for idx, audio in audio_samples:
        # Handle both filepath (string) and tuple (sr, wav_np) formats
        if isinstance(audio, str):
            print(f"Sample {idx}: Audio received as filepath: {audio}")
            # Save audio file to permanent location
            file_ext = Path(audio).suffix or ".wav"
            dest_path = user_dir / f"sample_{idx}{file_ext}"
            shutil.copy2(audio, dest_path)
            audio_file_paths.append(str(dest_path))

            # Use torchaudio to load audio (supports various formats)
            sr, wav_np = load_audio_file(audio)
            audio_tuple = (sr, wav_np)
        else:
            sr, wav_np = audio
            audio_tuple = audio
            # For tuple format, we'll skip saving (temporary recordings)
            audio_file_paths.append(None)

        print(
            f"Sample {idx}: sr={sr}, shape={wav_np.shape}, max={wav_np.max()}, min={wav_np.min()}"
        )

        # Check audio length - enforce minimum length
        audio_length_sec = wav_np.shape[0] / sr
        audio_lengths.append(audio_length_sec)
        if audio_length_sec < MIN_AUDIO_LENGTH_SEC:
            short_samples.append((idx, audio_length_sec))

        emb = extract_embedding(audio_tuple)
        embeddings.append(emb)
        print(f"Sample {idx} embedding norm: {np.linalg.norm(emb):.3f}")

    # Enforce minimum length requirement - reject if any sample is too short
    if short_samples:
        short_list = ", ".join(
            [f"Sample {idx} ({length:.1f}s)" for idx, length in short_samples]
        )
        # Get new random texts for retry
        text1, text2, text3 = get_enroll_texts()
        return (
            f"❌ Enrollment rejected: All samples must be at least {MIN_AUDIO_LENGTH_SEC:.0f} seconds long.\n\n"
            f"Short samples detected: {short_list}\n\n"
            f"Please re-record the short sample(s) with at least {MIN_AUDIO_LENGTH_SEC:.0f} seconds of clear speech.",
            text1,
            text2,
            text3,
        )

    # Store all embeddings separately (centroid will be computed on-the-fly during login)
    save_multiple_embeddings(username, embeddings, audio_lengths, audio_file_paths)

    # Get new random texts for next enrollment
    text1, text2, text3 = get_enroll_texts()

    avg_length = sum(audio_lengths) / len(audio_lengths)
    return (
        f"✅ Enrolled '{username}' successfully!\n\n"
        f"• 3 samples recorded (avg length: {avg_length:.1f}s)\n"
        f"• {embeddings[0].shape[-1]}D embeddings stored",
        text1,
        text2,
        text3,
    )


def create_enroll_tab():
    """Create the Enroll tab UI"""
    with gr.Tab("Enroll"):
        gr.Markdown("### 🎤 Voice Enrollment")

        u = gr.Textbox(label="Username", placeholder="e.g: phatpham9")

        # Get initial random texts
        initial_text1, initial_text2, initial_text3 = get_enroll_texts()

        gr.Markdown(
            f"#### Sample 1 (Required) - Minimum {MIN_AUDIO_LENGTH_SEC:.0f} seconds"
        )
        with gr.Group():
            a1 = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label=f"Record yourself reading the text below ({MIN_AUDIO_LENGTH_SEC:.0f}-10s)",
            )
            text1_display = gr.Textbox(
                label="Text to Read for Sample 1",
                value=initial_text1,
                lines=2,
                interactive=False,
            )
            refresh_text1_btn = gr.Button("🔄 Get New Text", size="sm")

        gr.Markdown(
            f"#### Sample 2 (Required) - Minimum {MIN_AUDIO_LENGTH_SEC:.0f} seconds"
        )
        with gr.Group():
            a2 = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label=f"Record yourself reading the text below ({MIN_AUDIO_LENGTH_SEC:.0f}-10s)",
            )
            text2_display = gr.Textbox(
                label="Text to Read for Sample 2",
                value=initial_text2,
                lines=2,
                interactive=False,
            )
            refresh_text2_btn = gr.Button("🔄 Get New Text", size="sm")

        gr.Markdown(
            f"#### Sample 3 (Required) - Minimum {MIN_AUDIO_LENGTH_SEC:.0f} seconds"
        )
        with gr.Group():
            a3 = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label=f"Record yourself reading the text below ({MIN_AUDIO_LENGTH_SEC:.0f}-10s)",
            )
            text3_display = gr.Textbox(
                label="Text to Read for Sample 3",
                value=initial_text3,
                lines=2,
                interactive=False,
            )
            refresh_text3_btn = gr.Button("🔄 Get New Text", size="sm")

        enroll_btn = gr.Button("Enroll", variant="primary")

        out = gr.Textbox(label="Result")

        enroll_btn.click(
            enroll,
            inputs=[u, a1, a2, a3],
            outputs=[out, text1_display, text2_display, text3_display],
        )

        # Individual refresh buttons for each text
        refresh_text1_btn.click(
            lambda: get_enroll_texts()[0],
            inputs=[],
            outputs=[text1_display],
        )
        refresh_text2_btn.click(
            lambda: get_enroll_texts()[1],
            inputs=[],
            outputs=[text2_display],
        )
        refresh_text3_btn.click(
            lambda: get_enroll_texts()[2],
            inputs=[],
            outputs=[text3_display],
        )
