"""
Login UI tab for voice authentication.
"""

import gradio as gr
import numpy as np
import random
from src.core import (
    ENROLLMENT_TEXTS,
    DEFAULT_THRESHOLD,
    load_audio_file,
    extract_embedding,
    cosine_similarity,
)
from src.database import load_embedding, log_authentication


def get_login_text():
    """Get a random text for login"""
    if ENROLLMENT_TEXTS:
        return random.choice(ENROLLMENT_TEXTS)
    return "Please record your voice."


def login(username, audio, threshold):
    if not username:
        return "⚠️ Please enter a username.", None
    if audio is None:
        return "⚠️ Please record your voice.", None

    # Handle both filepath (string) and tuple (sr, wav_np) formats
    if isinstance(audio, str):
        print(f"Audio received as filepath: {audio}")
        # Use torchaudio to load audio (supports various formats)
        sr, wav_np = load_audio_file(audio)
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
    matched_sample = scores.index(best_score) + 1
    print(f"Best score: {best_score:.4f}, Threshold: {threshold:.2f}")

    # Log authentication attempt
    success = best_score >= threshold
    log_authentication(username, success, best_score, threshold, matched_sample)

    if success:
        return f"✅ SUCCESS — score={best_score:.3f} ≥ threshold={threshold:.2f} (matched sample {matched_sample}/{len(scores)}){warning_msg}"
    else:
        return f"❌ DENIED — score={best_score:.3f} < threshold={threshold:.2f}{warning_msg}"


def create_login_tab():
    """Create the Login tab UI"""
    with gr.Tab("Login"):
        gr.Markdown("### 🔑 Voice Authentication")

        u2 = gr.Textbox(label="Username", placeholder="e.g: phatpham9")

        with gr.Group():
            a2 = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Record yourself reading the text below (3-10s)",
            )
            login_text_display = gr.Textbox(
                label="Text to Read",
                value=get_login_text(),
                lines=2,
                interactive=False,
                elem_id="login_text",
            )
            refresh_login_text_btn = gr.Button("🔄 Get New Text", size="sm")

        th = gr.Slider(
            0.50, 0.98, value=DEFAULT_THRESHOLD, step=0.01, label="Threshold (cosine)"
        )

        login_btn = gr.Button("Login", variant="primary")

        out = gr.Textbox(label="Result")

        refresh_login_text_btn.click(
            get_login_text, inputs=[], outputs=[login_text_display]
        )
        login_btn.click(login, inputs=[u2, a2, th], outputs=[out])
