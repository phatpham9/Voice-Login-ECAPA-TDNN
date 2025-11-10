"""
Voice Login with ECAPA-TDNN
Main application entry point for Gradio UI.
"""

import os
import sys
import gradio as gr
from src.database import init_database, list_users
from src.ui_login import create_login_tab
from src.ui_enroll import create_enroll_tab
from src.ui_manage import create_manage_users_tab
from src.ui_statistics import create_statistics_tab


# ------------------------------------
# Initialize database and seed data
# ------------------------------------
def init_app():
    # Initialize database
    init_database()
    print("✅ Database initialized")

    # Check if we need to seed users
    users = list_users()
    if len(users) == 0:
        print("⚠️ No users found. Running seed enrollment...")

        import subprocess

        seed_script = os.path.join(os.path.dirname(__file__), "seed_db", "seed_db.py")

        try:
            result = subprocess.run(
                [sys.executable, seed_script],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(__file__),
            )

            # Print the output
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)

            if result.returncode != 0:
                print(
                    f"⚠️ Seed enrollment exited with code {result.returncode}. Continuing without seed data."
                )
            else:
                # Refresh user list
                users = list_users()
                print(f"✅ Seed enrollment complete. Users: {', '.join(users)}")
        except Exception as e:
            print(
                f"⚠️ Error running seed enrollment: {e}. Continuing without seed data."
            )
    else:
        print(f"✅ Database has {len(users)} user(s): {', '.join(users)}")


# Initialize application
init_app()


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


if __name__ == "__main__":
    demo.launch()
