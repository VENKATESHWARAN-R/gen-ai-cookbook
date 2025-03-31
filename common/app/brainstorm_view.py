# brainstorm_view.py
import streamlit as st
import api_client
import json
from typing import List, Dict


def render_brainstorm_view():
    """Renders the brainstorming setup and interaction interface."""
    st.title("LLM Brainstorming Session")

    # --- Session State Initialization ---
    if "brainstorm_roles" not in st.session_state:
        # List of dictionaries: [{"role": "Alice", "persona": "Creative thinker"}, ...]
        st.session_state.brainstorm_roles = []
    if "brainstorm_messages" not in st.session_state:
        # List of dictionaries: [{"role": "Alice", "content": "Let's start..."}, ...]
        st.session_state.brainstorm_messages = []
    if "brainstorm_iam" not in st.session_state:
        st.session_state.brainstorm_iam = ""
    if "brainstorm_ai_turns" not in st.session_state:
        st.session_state.brainstorm_ai_turns = 1
    if "brainstorm_result" not in st.session_state:
        st.session_state.brainstorm_result = None

    # --- Configuration UI ---
    st.subheader("Configure Brainstorming Session")
    with st.expander("Setup Roles & Personas", expanded=True):
        # Role Management
        cols = st.columns([2, 3, 1])
        new_role_name = cols[0].text_input("Role Name (e.g., Alice)", key="bs_new_role")
        new_role_persona = cols[1].text_area(
            "Persona Description", key="bs_new_persona", height=50
        )
        if cols[2].button("Add Role", key="bs_add_role", use_container_width=True):
            if new_role_name and new_role_persona:
                # Avoid duplicate role names
                if not any(
                    r["role"] == new_role_name
                    for r in st.session_state.brainstorm_roles
                ):
                    st.session_state.brainstorm_roles.append(
                        {"role": new_role_name, "persona": new_role_persona}
                    )
                    # Clear inputs after adding - need unique keys or different approach
                    # st.session_state.bs_new_role = "" # This doesn't work directly like this in Streamlit loops
                    # st.session_state.bs_new_persona = ""
                    st.rerun()  # Force rerun to update list and clear inputs implicitly
                else:
                    st.warning(f"Role '{new_role_name}' already exists.")
            else:
                st.warning("Please provide both Role Name and Persona.")

        # Display Current Roles
        st.write("**Current Roles:**")
        if not st.session_state.brainstorm_roles:
            st.caption("No roles defined yet.")
        else:
            for i, role_info in enumerate(st.session_state.brainstorm_roles):
                cols_disp = st.columns([1, 2, 1])
                cols_disp[0].write(f"**{role_info['role']}**")
                cols_disp[1].caption(role_info["persona"])
                if cols_disp[2].button(
                    f"Remove {role_info['role']}", key=f"remove_role_{i}"
                ):
                    st.session_state.brainstorm_roles.pop(i)
                    st.rerun()

        # Select 'iam' (who initiates the next AI turn)
        role_names = [r["role"] for r in st.session_state.brainstorm_roles]
        if role_names:
            st.session_state.brainstorm_iam = st.selectbox(
                "'I am' (Role for next AI turn):",
                options=role_names,
                index=(
                    role_names.index(st.session_state.brainstorm_iam)
                    if st.session_state.brainstorm_iam in role_names
                    else 0
                ),
            )
        else:
            st.caption("Add roles to select who starts.")
            st.session_state.brainstorm_iam = ""

    with st.expander("Initial Messages", expanded=True):
        # Initial Message Input
        if role_names:
            msg_cols = st.columns([1, 3, 1])
            msg_role = msg_cols[0].selectbox(
                "Role Speaking:", options=role_names, key="bs_msg_role"
            )
            msg_content = msg_cols[1].text_area(
                "Initial Message Content", key="bs_msg_content", height=50
            )
            if msg_cols[2].button(
                "Add Message", key="bs_add_msg", use_container_width=True
            ):
                if msg_role and msg_content:
                    st.session_state.brainstorm_messages.append(
                        {"role": msg_role, "content": msg_content}
                    )
                    st.rerun()  # Update message list
                else:
                    st.warning("Please select a role and enter message content.")

            # Display Current Initial Messages
            st.write("**Initial Dialogue:**")
            if not st.session_state.brainstorm_messages:
                st.caption("No initial messages added yet.")
            else:
                for i, msg in enumerate(st.session_state.brainstorm_messages):
                    cols_msg_disp = st.columns([3, 1])
                    cols_msg_disp[0].markdown(f"**{msg['role']}:** {msg['content']}")
                    if cols_msg_disp[1].button(f"Remove", key=f"remove_msg_{i}"):
                        st.session_state.brainstorm_messages.pop(i)
                        st.rerun()
        else:
            st.caption("Add roles before adding messages.")

    # AI Turns
    st.session_state.brainstorm_ai_turns = st.number_input(
        "Number of AI-assisted turns to generate:",
        min_value=1,
        max_value=10,
        value=st.session_state.brainstorm_ai_turns,
        step=1,
    )

    # --- Run Brainstorming ---
    st.markdown("---")
    if st.button("Start Brainstorming Session", type="primary"):
        # Validation
        if not st.session_state.brainstorm_roles:
            st.error("Please define at least one role.")
        elif not st.session_state.brainstorm_messages:
            st.error("Please add at least one initial message.")
        elif not st.session_state.brainstorm_iam:
            st.error("Please select the role for the AI's next turn ('I am').")
        else:
            # Prepare payload
            messages = st.session_state.brainstorm_messages
            role_play_configs = st.session_state.brainstorm_roles
            iam = st.session_state.brainstorm_iam
            ai_turns = st.session_state.brainstorm_ai_turns

            with st.spinner("Brainstorming in progress..."):
                # Call API
                result = api_client.call_brainstorm(
                    messages=messages,
                    role=None,  # Role seems ambiguous here based on schema, maybe 'iam' is enough? Check API behavior. Let's assume 'iam' is the primary indicator.
                    iam=iam,
                    role_play_configs=role_play_configs,
                    ai_assisted_turns=ai_turns,
                    # Add kwargs if needed: kwargs={"temperature": 0.7}
                )
                st.session_state.brainstorm_result = result

    # --- Display Results ---
    st.markdown("---")
    st.subheader("Brainstorming Output")
    if st.session_state.brainstorm_result:
        result_data = st.session_state.brainstorm_result
        if isinstance(result_data, dict) and result_data.get("error"):
            st.error(
                f"API Error: {result_data.get('error')} - {result_data.get('detail', 'No details')}"
            )
        elif (
            isinstance(result_data, dict) and "messages" in result_data
        ):  # Assuming result contains the full message history
            # Display the generated conversation
            for message in result_data["messages"]:
                # Ensure message has expected keys
                role = message.get("role", "Unknown Role")
                content = message.get("content", "*No content received*")
                with st.chat_message(role):  # Use chat message for nice formatting
                    st.markdown(content)
        elif result_data:
            # Fallback: Display raw result if structure is unexpected
            st.warning("Unexpected result structure from /brainstorm endpoint.")
            st.json(result_data)
        else:
            st.info("Brainstorming finished, but no result data was received.")

    elif st.button("Clear Brainstorming State"):
        keys_to_clear = [
            "brainstorm_roles",
            "brainstorm_messages",
            "brainstorm_iam",
            "brainstorm_ai_turns",
            "brainstorm_result",
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
