"""
Voice Login with ECAPA-TDNN
Main application entry point for Gradio UI.
"""

import gradio as gr
from src.ui_login import create_login_tab
from src.ui_enroll import create_enroll_tab
from src.ui_manage import create_manage_users_tab
from src.ui_statistics import create_statistics_tab
from src.ui_metrics import create_performance_metrics_tab


# ------------------------------------
# Gradio UI
# ------------------------------------
with gr.Blocks() as demo:
    gr.Markdown(
        "# 🔐 Voice Login with ECAPA-TDNN\nA text-independent speaker verification system built with ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation in Time Delay Neural Network) from SpeechBrain."
    )

    create_login_tab()
    create_enroll_tab()
    create_manage_users_tab()
    create_statistics_tab()
    create_performance_metrics_tab()


if __name__ == "__main__":
    demo.launch()
