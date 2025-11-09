"""
Enroll UI tab for voice enrollment.
"""

import gradio as gr
import numpy as np
import random
from src.core import (
    ENROLLMENT_TEXTS,
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

    if len(audio_samples) == 0:
        return "⚠️ Please record at least one voice sample.", None, None, None

    # Extract embeddings from all samples
    embeddings = []
    audio_lengths = []
    short_samples = []
    for idx, audio in audio_samples:
        # Handle both filepath (string) and tuple (sr, wav_np) formats
        if isinstance(audio, str):
            print(f"Sample {idx}: Audio received as filepath: {audio}")
            # Use torchaudio to load audio (supports various formats)
            sr, wav_np = load_audio_file(audio)
            audio_tuple = (sr, wav_np)
        else:
            sr, wav_np = audio
            audio_tuple = audio

        print(
            f"Sample {idx}: sr={sr}, shape={wav_np.shape}, max={wav_np.max()}, min={wav_np.min()}"
        )

        # Check audio length
        audio_length_sec = wav_np.shape[0] / sr
        audio_lengths.append(audio_length_sec)
        if audio_length_sec < 3.0:
            short_samples.append((idx, audio_length_sec))

        emb = extract_embedding(audio_tuple)
        embeddings.append(emb)
        print(f"Sample {idx} embedding norm: {np.linalg.norm(emb):.3f}")

    # Store all embeddings separately (not averaged) with audio lengths
    save_multiple_embeddings(username, embeddings, audio_lengths)

    # Create warning message if any samples were short
    warning_msg = ""
    if short_samples:
        short_list = ", ".join(
            [f"Sample {idx} ({length:.1f}s)" for idx, length in short_samples]
        )
        warning_msg = f"\n\n⚠️ Warning: Short audio detected - {short_list}. For better verification accuracy, consider re-enrolling with 3-5+ seconds of speech per sample."

    # Get new random texts for next enrollment
    text1, text2, text3 = get_enroll_texts()

    return (
        f"✅ Enrolled '{username}' with {len(embeddings)} sample(s) — {embeddings[0].shape[-1]}D embeddings stored separately{warning_msg}",
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

        gr.Markdown("#### Sample 1 (Required)")
        with gr.Group():
            a1 = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Record yourself reading the text below (3-10s)",
            )
            text1_display = gr.Textbox(
                label="Text to Read for Sample 1",
                value=initial_text1,
                lines=2,
                interactive=False,
            )
            refresh_text1_btn = gr.Button("🔄 Get New Text", size="sm")

        gr.Markdown("#### Sample 2 (Optional)")
        with gr.Group():
            a2 = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Record yourself reading the text below (3-10s)",
            )
            text2_display = gr.Textbox(
                label="Text to Read for Sample 2",
                value=initial_text2,
                lines=2,
                interactive=False,
            )
            refresh_text2_btn = gr.Button("🔄 Get New Text", size="sm")

        gr.Markdown("#### Sample 3 (Optional)")
        with gr.Group():
            a3 = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Record yourself reading the text below (3-10s)",
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
