"""
Statistics UI tab for database statistics and authentication history.
"""

import gradio as gr
from src.database import get_database_stats, get_auth_history


def create_statistics_tab():
    """Create the Statistics tab UI"""
    with gr.Tab("Statistics"):
        gr.Markdown("### 📊 Database Statistics")

        stats_display = gr.Textbox(
            label="System Statistics", lines=10, interactive=False
        )

        auth_history_display = gr.Textbox(
            label="Recent Authentication History (Last 10)", lines=15, interactive=False
        )

        stats_refresh_btn = gr.Button("🔄 Refresh Statistics")

        def display_stats():
            stats = get_database_stats()
            stats_text = f"""**Total Users:** {stats['total_users']}
**Total Embeddings:** {stats['total_embeddings']}
**Total Auth Attempts:** {stats['total_auth_attempts']}
**Successful Attempts:** {stats['successful_attempts']}
**Failed Attempts:** {stats['failed_attempts']}
**Success Rate:** {stats['success_rate']:.1f}%
**Recent Attempts (24h):** {stats['recent_attempts_24h']}"""

            history = get_auth_history(limit=10)
            if history:
                history_text = "\n".join(
                    [
                        f"[{h['timestamp']}] {h['username']}: {'✅ SUCCESS' if h['success'] else '❌ FAILED'} "
                        f"(score={h['score']:.3f}, threshold={h['threshold']:.2f}, sample={h['matched_sample']})"
                        for h in history
                    ]
                )
            else:
                history_text = "No authentication history yet"

            return stats_text, history_text

        # Load initial stats
        initial_stats, initial_history = display_stats()
        stats_display.value = initial_stats
        auth_history_display.value = initial_history

        stats_refresh_btn.click(
            display_stats, inputs=[], outputs=[stats_display, auth_history_display]
        )
