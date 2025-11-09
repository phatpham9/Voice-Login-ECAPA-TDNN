"""
Manage Users UI tab for user management.
"""

import gradio as gr
import json
import os
from src.database import list_users, get_user_info, delete_user, get_user_samples


def create_manage_users_tab():
    """Create the Manage Users tab UI"""
    with gr.Tab("Manage"):
        gr.Markdown("### 👥 User Management")

        # Get initial user list and first user
        initial_users = list_users()
        initial_user = initial_users[0] if initial_users else None

        with gr.Group():
            user_dropdown = gr.Dropdown(
                choices=initial_users,
                label="Select User",
                interactive=True,
                value=initial_user,
            )
            refresh_btn = gr.Button("🔄 Refresh List", size="sm")

        # Get initial user info and samples
        initial_info = ""
        initial_audio_values = [None, None, None]
        initial_audio_visible = [False, False, False]
        initial_audio_labels = [
            "Sample 1",
            "Sample 2",
            "Sample 3",
        ]

        if initial_user:
            info = get_user_info(initial_user)
            samples = get_user_samples(initial_user)
            if info:
                # Combine info with samples for JSON display
                display_data = {
                    "username": info["username"],
                    "created_at": info["created_at"],
                    "updated_at": info["updated_at"],
                    "sample_count": info["sample_count"],
                    "samples": samples if samples else [],
                }
                initial_info = json.dumps(display_data, indent=2)

                # Set up initial audio components
                if samples:
                    for i in range(min(3, len(samples))):
                        sample = samples[i]
                        if sample["audio_file_path"] and os.path.exists(
                            sample["audio_file_path"]
                        ):
                            initial_audio_values[i] = sample["audio_file_path"]
                            initial_audio_visible[i] = True
                            initial_audio_labels[i] = (
                                f"Sample {sample['sample_number']} ({sample['audio_length_sec']:.2f}s)"
                            )

        user_info_display = gr.Code(
            label="User Information (JSON)",
            language="json",
            lines=15,
            interactive=False,
            value=initial_info,
        )

        # Container for audio samples (supporting up to 3 samples)
        with gr.Group():
            audio1 = gr.Audio(
                label=initial_audio_labels[0],
                value=initial_audio_values[0],
                visible=initial_audio_visible[0],
                interactive=False,
            )
            audio2 = gr.Audio(
                label=initial_audio_labels[1],
                value=initial_audio_values[1],
                visible=initial_audio_visible[1],
                interactive=False,
            )
            audio3 = gr.Audio(
                label=initial_audio_labels[2],
                value=initial_audio_values[2],
                visible=initial_audio_visible[2],
                interactive=False,
            )

        delete_btn = gr.Button("🗑️ Delete User", variant="stop")

        manage_result = gr.Textbox(label="Result")

        def refresh_user_list():
            users = list_users()
            return gr.Dropdown(choices=users)

        def view_user_details(username):
            # Default outputs: json_info + 3 audio components
            default_outputs = [
                "",
                gr.Audio(visible=False),
                gr.Audio(visible=False),
                gr.Audio(visible=False),
            ]

            if not username:
                return default_outputs

            info = get_user_info(username)
            samples = get_user_samples(username)

            if not info:
                return default_outputs

            # Combine info with samples for JSON display
            display_data = {
                "username": info["username"],
                "created_at": info["created_at"],
                "updated_at": info["updated_at"],
                "sample_count": info["sample_count"],
                "samples": samples if samples else [],
            }

            json_info = json.dumps(display_data, indent=2)

            # Prepare audio components (up to 3)
            audio_outputs = [json_info]

            if samples:
                for i in range(3):
                    if i < len(samples):
                        sample = samples[i]
                        if sample["audio_file_path"] and os.path.exists(
                            sample["audio_file_path"]
                        ):
                            label = f"Sample {sample['sample_number']} ({sample['audio_length_sec']:.2f}s)"
                            audio_outputs.append(
                                gr.Audio(
                                    value=sample["audio_file_path"],
                                    label=label,
                                    visible=True,
                                    interactive=False,
                                )
                            )
                        else:
                            audio_outputs.append(gr.Audio(visible=False))
                    else:
                        audio_outputs.append(gr.Audio(visible=False))
            else:
                # No samples, hide all audio components
                for i in range(3):
                    audio_outputs.append(gr.Audio(visible=False))

            return audio_outputs

        def delete_user_action(username):
            if not username:
                return (
                    "⚠️ Please select a user",
                    gr.Dropdown(choices=list_users()),
                    "",
                    gr.Audio(visible=False),
                    gr.Audio(visible=False),
                    gr.Audio(visible=False),
                )

            success = delete_user(username)
            if success:
                users = list_users()
                return (
                    f"✅ User '{username}' deleted successfully",
                    gr.Dropdown(choices=users),
                    "",
                    gr.Audio(visible=False),
                    gr.Audio(visible=False),
                    gr.Audio(visible=False),
                )
            else:
                return (
                    f"❌ Failed to delete user '{username}'",
                    gr.Dropdown(choices=list_users()),
                    "",
                    gr.Audio(visible=False),
                    gr.Audio(visible=False),
                    gr.Audio(visible=False),
                )

        refresh_btn.click(refresh_user_list, inputs=[], outputs=[user_dropdown])

        # Automatically show details when user is selected
        user_dropdown.change(
            view_user_details,
            inputs=[user_dropdown],
            outputs=[user_info_display, audio1, audio2, audio3],
        )

        delete_btn.click(
            delete_user_action,
            inputs=[user_dropdown],
            outputs=[
                manage_result,
                user_dropdown,
                user_info_display,
                audio1,
                audio2,
                audio3,
            ],
        )
