"""
Performance Metrics UI tab for system performance analysis.
"""

import gradio as gr
from src.core import DEFAULT_THRESHOLD
from src.metrics import (
    collect_test_data,
    generate_metrics_summary,
    generate_roc_curve,
    generate_det_curve,
    generate_far_frr_curve,
    generate_score_distribution,
    generate_confusion_matrix,
)


def create_performance_metrics_tab():
    """Create the Performance Metrics tab UI"""
    with gr.Tab("Metrics"):
        gr.Markdown("### 📈 Performance Metrics Dashboard")
        gr.Markdown(
            """
        This dashboard analyzes the system's performance using authentication history.
        
        **Key Metrics:**
        - **FAR (False Acceptance Rate):** Percentage of impostors incorrectly accepted
        - **FRR (False Rejection Rate):** Percentage of genuine users incorrectly rejected
        - **EER (Equal Error Rate):** Operating point where FAR = FRR (optimal balance)
        - **ROC Curve:** Shows trade-off between true positive rate and false positive rate
        - **DET Curve:** Detection Error Tradeoff curve (log scale for better visualization)
        
        *Note: Requires sufficient authentication history for meaningful analysis.*
        """
        )

        with gr.Row():
            current_threshold_input = gr.Slider(
                0.50,
                0.98,
                value=DEFAULT_THRESHOLD,
                step=0.01,
                label="Current Threshold for Analysis",
            )
            metrics_refresh_btn = gr.Button("🔄 Refresh Metrics", variant="primary")

        metrics_summary = gr.Markdown(label="Metrics Summary")

        with gr.Row():
            with gr.Column():
                roc_plot = gr.Plot(label="ROC Curve")
            with gr.Column():
                det_plot = gr.Plot(label="DET Curve")

        with gr.Row():
            with gr.Column():
                far_frr_plot = gr.Plot(label="FAR/FRR vs Threshold")
            with gr.Column():
                score_dist_plot = gr.Plot(label="Score Distribution")

        confusion_matrix_plot = gr.Plot(label="Confusion Matrix")

        def update_metrics_dashboard(threshold):
            # Collect data
            data = collect_test_data()
            genuine_scores = data["genuine_scores"]
            impostor_scores = data["impostor_scores"]

            # Generate summary
            summary = generate_metrics_summary(
                genuine_scores, impostor_scores, threshold
            )

            # Generate plots
            roc = generate_roc_curve(genuine_scores, impostor_scores)
            det = generate_det_curve(genuine_scores, impostor_scores)
            far_frr = generate_far_frr_curve(genuine_scores, impostor_scores)
            score_dist = generate_score_distribution(
                genuine_scores, impostor_scores, threshold
            )
            confusion = generate_confusion_matrix(
                genuine_scores, impostor_scores, threshold
            )

            return summary, roc, det, far_frr, score_dist, confusion

        # Initial load
        try:
            (
                initial_summary,
                initial_roc,
                initial_det,
                initial_far_frr,
                initial_score_dist,
                initial_confusion,
            ) = update_metrics_dashboard(DEFAULT_THRESHOLD)
            metrics_summary.value = initial_summary
        except Exception as e:
            print(f"Warning: Could not load initial metrics: {e}")

        # Refresh on button click or threshold change
        metrics_refresh_btn.click(
            update_metrics_dashboard,
            inputs=[current_threshold_input],
            outputs=[
                metrics_summary,
                roc_plot,
                det_plot,
                far_frr_plot,
                score_dist_plot,
                confusion_matrix_plot,
            ],
        )

        current_threshold_input.change(
            update_metrics_dashboard,
            inputs=[current_threshold_input],
            outputs=[
                metrics_summary,
                roc_plot,
                det_plot,
                far_frr_plot,
                score_dist_plot,
                confusion_matrix_plot,
            ],
        )
