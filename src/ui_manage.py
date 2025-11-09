"""
Manage Users UI tab for user management.
"""

import gradio as gr
from src.database import list_users, get_user_info, delete_user


def create_manage_users_tab():
    """Create the Manage Users tab UI"""
    with gr.Tab("Manage Users"):
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

        # Get initial user info
        initial_info = ""
        if initial_user:
            info = get_user_info(initial_user)
            if info:
                initial_info = f"""**Username:** {info['username']}
**Enrolled:** {info['created_at']}
**Last Updated:** {info['updated_at']}
**Sample Count:** {info['sample_count']}"""

        user_info_display = gr.Textbox(
            label="User Information", lines=5, interactive=False, value=initial_info
        )

        delete_btn = gr.Button("🗑️ Delete User", variant="stop")

        manage_result = gr.Textbox(label="Result")

        def refresh_user_list():
            users = list_users()
            return gr.Dropdown(choices=users)

        def view_user_details(username):
            if not username:
                return ""

            info = get_user_info(username)
            if not info:
                return f"❌ User '{username}' not found"

            details = f"""**Username:** {info['username']}
**Enrolled:** {info['created_at']}
**Last Updated:** {info['updated_at']}
**Sample Count:** {info['sample_count']}"""

            return details

        def delete_user_action(username):
            if not username:
                return "⚠️ Please select a user", gr.Dropdown(choices=list_users()), ""

            success = delete_user(username)
            if success:
                users = list_users()
                return (
                    f"✅ User '{username}' deleted successfully",
                    gr.Dropdown(choices=users),
                    "",
                )
            else:
                return (
                    f"❌ Failed to delete user '{username}'",
                    gr.Dropdown(choices=list_users()),
                    "",
                )

        refresh_btn.click(refresh_user_list, inputs=[], outputs=[user_dropdown])

        # Automatically show details when user is selected
        user_dropdown.change(
            view_user_details, inputs=[user_dropdown], outputs=[user_info_display]
        )

        delete_btn.click(
            delete_user_action,
            inputs=[user_dropdown],
            outputs=[manage_result, user_dropdown, user_info_display],
        )
