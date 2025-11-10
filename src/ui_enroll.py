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
    ENABLE_TEXT_VERIFICATION,
    load_audio_file,
    extract_embedding,
    analyze_audio_quality,
    generate_diagnostic_message,
    verify_spoken_text,
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


def enroll(username, audio1, audio2, audio3, text1, text2, text3):
    if not username:
        return "⚠️ Please enter a username.", None, None, None, None, None, None

    # Collect all provided audio samples with their expected texts
    audio_samples = []
    expected_texts = [text1, text2, text3]
    for i, (audio, expected_text) in enumerate(
        zip([audio1, audio2, audio3], expected_texts), 1
    ):
        if audio is not None:
            audio_samples.append((i, audio, expected_text))

    # Enforce 3 mandatory samples
    if len(audio_samples) < 3:
        return (
            f"❌ All 3 voice samples are required for enrollment. You provided {len(audio_samples)}/3 samples. Please record all samples.",
            text1,
            text2,
            text3,  # Keep current texts displayed
            text1,
            text2,
            text3,  # Keep state values
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
    all_diagnostics = []
    text_verification_failures = []

    for idx, audio, expected_text in audio_samples:
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

        # Text verification (anti-spoofing)
        if ENABLE_TEXT_VERIFICATION and expected_text:
            text_verify_result = verify_spoken_text(audio_tuple, expected_text)
            if not text_verify_result["passed"]:
                text_verification_failures.append((idx, text_verify_result))

        # Analyze audio quality
        diagnostics = analyze_audio_quality(wav_np, sr)
        all_diagnostics.append((idx, diagnostics))

        # Check audio length - enforce minimum length
        audio_length_sec = wav_np.shape[0] / sr
        audio_lengths.append(audio_length_sec)
        if audio_length_sec < MIN_AUDIO_LENGTH_SEC:
            short_samples.append((idx, audio_length_sec))

        emb = extract_embedding(audio_tuple)
        embeddings.append(emb)
        print(f"Sample {idx} embedding norm: {np.linalg.norm(emb):.3f}")

    # Check text verification failures first (anti-spoofing)
    if text_verification_failures:
        # Get new random texts for retry
        text1, text2, text3 = get_enroll_texts()

        failure_msgs = []
        for idx, verify_result in text_verification_failures:
            failure_msgs.append(f"\nSample {idx}:\n{verify_result['message']}")

        new_text1, new_text2, new_text3 = text1, text2, text3  # Keep current for now
        return (
            f"❌ Enrollment rejected: Text verification failed for {len(text_verification_failures)} sample(s).\n"
            + "\n".join(failure_msgs),
            new_text1,
            new_text2,
            new_text3,  # Display
            new_text1,
            new_text2,
            new_text3,  # State
        )

    # Enforce minimum length requirement - reject if any sample is too short
    if short_samples:
        short_list = ", ".join(
            [f"Sample {idx} ({length:.1f}s)" for idx, length in short_samples]
        )

        # Generate detailed diagnostic messages for short samples
        diagnostic_msgs = []
        for idx, diag in all_diagnostics:
            if any(s[0] == idx for s in short_samples):
                diag_msg = generate_diagnostic_message(diag, context="enrollment")
                if diag_msg:
                    diagnostic_msgs.append(f"Sample {idx}:{diag_msg}")

        # Get new random texts for retry
        new_text1, new_text2, new_text3 = get_enroll_texts()

        base_msg = (
            f"❌ Enrollment rejected: All samples must be at least {MIN_AUDIO_LENGTH_SEC:.0f} seconds long.\n\n"
            f"Short samples detected: {short_list}"
        )

        # Add diagnostic details if available
        if diagnostic_msgs:
            base_msg += "\n" + "\n".join(diagnostic_msgs)

        return (
            base_msg,
            new_text1,
            new_text2,
            new_text3,
            new_text1,
            new_text2,
            new_text3,
        )

    # Store all embeddings separately (centroid will be computed on-the-fly during login)
    save_multiple_embeddings(username, embeddings, audio_lengths, audio_file_paths)

    # Get new random texts for next enrollment
    new_text1, new_text2, new_text3 = get_enroll_texts()

    # Check for quality warnings (even if enrollment succeeded)
    quality_warnings = []
    for idx, diag in all_diagnostics:
        if diag["is_quiet"] or diag["has_noise"] or diag["is_clipped"]:
            quality_warnings.append(idx)

    avg_length = sum(audio_lengths) / len(audio_lengths)
    success_msg = (
        f"✅ Enrolled '{username}' successfully!\n\n"
        f"• 3 samples recorded (avg length: {avg_length:.1f}s)\n"
        f"• {embeddings[0].shape[-1]}D embeddings stored"
    )

    # Add quality warnings if any
    if quality_warnings:
        success_msg += "\n\n⚠️ Quality warnings detected in some samples:"
        for idx, diag in all_diagnostics:
            if idx in quality_warnings:
                diag_msg = generate_diagnostic_message(diag, context="enrollment")
                if diag_msg:
                    success_msg += f"\n\nSample {idx}:{diag_msg}"
        success_msg += "\n\n💡 Consider re-enrolling for better accuracy if login performance is poor."

    return (
        success_msg,
        new_text1,
        new_text2,
        new_text3,
        new_text1,
        new_text2,
        new_text3,
    )


def create_enroll_tab():
    """Create the Enroll tab UI"""
    with gr.Tab("Enroll"):
        gr.Markdown("### 🎤 Voice Enrollment")
        gr.Markdown(
            "**Note:** All 3 samples are required. Each sample must be at least {:.0f} seconds long for optimal verification accuracy.".format(
                MIN_AUDIO_LENGTH_SEC
            )
        )

        u = gr.Textbox(label="Username", placeholder="e.g: phatpham9")

        # Get initial random texts
        initial_text1, initial_text2, initial_text3 = get_enroll_texts()

        # State variables to hold the current texts
        text1_state = gr.State(value=initial_text1)
        text2_state = gr.State(value=initial_text2)
        text3_state = gr.State(value=initial_text3)

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
            inputs=[u, a1, a2, a3, text1_state, text2_state, text3_state],
            outputs=[
                out,
                text1_display,
                text2_display,
                text3_display,
                text1_state,
                text2_state,
                text3_state,
            ],
        )

        # Individual refresh buttons for each text
        def refresh_text1():
            text = get_enroll_texts()[0]
            return text, text

        def refresh_text2():
            text = get_enroll_texts()[1]
            return text, text

        def refresh_text3():
            text = get_enroll_texts()[2]
            return text, text

        refresh_text1_btn.click(
            refresh_text1,
            inputs=[],
            outputs=[text1_display, text1_state],
        )
        refresh_text2_btn.click(
            refresh_text2,
            inputs=[],
            outputs=[text2_display, text2_state],
        )
        refresh_text3_btn.click(
            refresh_text3,
            inputs=[],
            outputs=[text3_display, text3_state],
        )
