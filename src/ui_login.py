"""
Login UI tab for voice authentication.
"""

import gradio as gr
import numpy as np
import random
from src.core import (
    ENROLLMENT_TEXTS,
    DEFAULT_THRESHOLD,
    MIN_AUDIO_LENGTH_SEC,
    load_audio_file,
    extract_embedding,
    cosine_similarity,
    compute_centroid,
    compute_weighted_score,
    analyze_audio_quality,
    generate_diagnostic_message,
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

    # Load stored embeddings
    stored_embeddings = load_embedding(username)
    if stored_embeddings is None:
        return f"❌ User '{username}' not found. Please enroll first."

    # Analyze audio quality
    diagnostics = analyze_audio_quality(wav_np, sr)
    diagnostic_msg = generate_diagnostic_message(diagnostics, context="login")

    new_emb = extract_embedding(audio_tuple)
    print(f"New embedding shape: {new_emb.shape}, norm: {np.linalg.norm(new_emb):.3f}")

    # Compute similarity scores for all stored samples
    sample_scores = []
    for i, stored_emb in enumerate(stored_embeddings):
        score = cosine_similarity(stored_emb, new_emb)
        sample_scores.append(score)
        print(f"Similarity to stored sample {i+1}: {score:.4f}")

    # Compute centroid on-the-fly and compare
    centroid = compute_centroid(stored_embeddings)
    centroid_score = cosine_similarity(centroid, new_emb)
    print(f"Similarity to profile centroid: {centroid_score:.4f}")

    # Compute weighted score using top-k + centroid fusion
    score_result = compute_weighted_score(sample_scores, centroid_score)
    final_score = score_result["final_score"]

    print(f"\n{score_result['strategy_breakdown']}")
    print(f"Threshold: {threshold:.2f}")

    # Log authentication attempt
    success = final_score >= threshold
    log_authentication(
        username, success, final_score, threshold, score_result["best_match_index"]
    )

    # Build detailed result message
    if success:
        result_msg = (
            f"✅ SUCCESS — score={final_score:.3f} ≥ threshold={threshold:.2f}\n\n"
            f"📊 Score Breakdown:\n"
        )

        # Individual sample scores
        for i, score in enumerate(sample_scores):
            marker = "🏆" if i + 1 == score_result["best_match_index"] else "  "
            in_topk = "✓" if (i + 1) in score_result["top_k_indices"] else " "
            result_msg += f"{marker} Sample {i+1}: {score:.3f} [{in_topk}]\n"

        result_msg += (
            f"\n🎯 Verification Strategy:\n"
            f"{score_result['strategy_breakdown']}\n"
            f"\nLegend: [✓] = used in top-k, 🏆 = best match"
        )

        # Add diagnostic info if there are issues (even on success)
        if diagnostic_msg:
            result_msg += diagnostic_msg

        return result_msg
    else:
        result_msg = (
            f"❌ DENIED — score={final_score:.3f} < threshold={threshold:.2f}\n\n"
            f"📊 Score Breakdown:\n"
        )

        # Individual sample scores
        for i, score in enumerate(sample_scores):
            marker = "🏆" if i + 1 == score_result["best_match_index"] else "  "
            in_topk = "✓" if (i + 1) in score_result["top_k_indices"] else " "
            result_msg += f"{marker} Sample {i+1}: {score:.3f} [{in_topk}]\n"

        result_msg += (
            f"\n🎯 Verification Strategy:\n"
            f"{score_result['strategy_breakdown']}\n"
            f"\nLegend: [✓] = used in top-k, 🏆 = best match"
        )

        # Always add diagnostic info on failure to help user troubleshoot
        if diagnostic_msg:
            result_msg += diagnostic_msg
        else:
            # Even if no specific issues detected, provide general guidance
            result_msg += (
                "\n\n💬 Suggestions:\n"
                f"• Ensure you record at least {MIN_AUDIO_LENGTH_SEC:.0f} seconds of clear speech\n"
                "• Speak in a quiet environment\n"
                "• Consider re-enrolling if you continue to have issues"
            )

        return result_msg


def create_login_tab():
    """Create the Login tab UI"""
    with gr.Tab("Login"):
        gr.Markdown("### 🔑 Voice Authentication")

        u2 = gr.Textbox(label="Username", placeholder="e.g: phatpham9")

        with gr.Group():
            a2 = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label=f"Record yourself reading the text below ({MIN_AUDIO_LENGTH_SEC:.0f}-10s recommended)",
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
