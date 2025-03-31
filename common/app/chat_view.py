# chat_view.py
import json
from typing import Dict, List

import agent_handler  # For DB/RAG agents
import api_client  # For stateless chat
import streamlit as st


def display_chat_messages(chat_history: List[Dict]):
    """Displays chat messages from history."""
    for message in chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def render_chat_view():
    """Renders the main chat interface."""
    st.title("LLM Chat Interface")

    # Agent Selection
    agent_options = ["Stateless Chat", "Database Agent", "RAG Agent"]
    # Initialize session state for selected agent if it doesn't exist
    if "selected_agent" not in st.session_state:
        st.session_state.selected_agent = agent_options[0]

    st.session_state.selected_agent = st.sidebar.selectbox(
        "Select Agent:",
        agent_options,
        index=agent_options.index(st.session_state.selected_agent),  # Persist selection
    )
    st.sidebar.markdown("---")
    st.sidebar.write(f"Current Agent: **{st.session_state.selected_agent}**")
    st.sidebar.markdown("---")
    # Add RAG document upload/management here if needed
    if st.session_state.selected_agent == "RAG Agent":
        st.sidebar.info("RAG Agent selected. (Document handling UI can be added here)")
        # Example: uploaded_files = st.sidebar.file_uploader("Upload Documents for RAG", accept_multiple_files=True)
        # Process uploaded_files and store relevant content/IDs in session state if needed.

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display existing chat messages
    display_chat_messages(st.session_state.chat_history)

    # Chat input
    if prompt := st.chat_input(f"Ask {st.session_state.selected_agent}..."):
        # Add user message to history and display it
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response based on selected agent
        with st.spinner(f"{st.session_state.selected_agent} is thinking..."):
            response_data = None
            error_message = None

            if st.session_state.selected_agent == "Stateless Chat":
                # For stateless, we pass the current history relevant to the API
                api_history = [
                    msg
                    for msg in st.session_state.chat_history
                    if msg["role"] in ["user", "assistant"]
                ]
                response_data = api_client.call_stateless_chat(
                    prompt, chat_history=api_history
                )

            elif st.session_state.selected_agent == "Database Agent":
                # Pass only user/assistant messages for context
                api_history = [
                    msg
                    for msg in st.session_state.chat_history
                    if msg["role"] in ["user", "assistant"]
                ]
                response_data = agent_handler.handle_db_agent_query(
                    prompt, chat_history=api_history
                )

            elif st.session_state.selected_agent == "RAG Agent":
                # Pass history and potentially retrieved documents (if UI added)
                api_history = [
                    msg
                    for msg in st.session_state.chat_history
                    if msg["role"] in ["user", "assistant"]
                ]
                # Example: retrieved_docs = st.session_state.get("retrieved_rag_docs", None)
                retrieved_docs = None  # Placeholder - implement retrieval logic
                response_data = agent_handler.handle_rag_agent_query(
                    prompt, chat_history=api_history, documents=retrieved_docs
                )

            # Process and display response
            if response_data:
                # Check if the response indicates an error from the API client
                if isinstance(response_data, dict) and response_data.get("error"):
                    assistant_response = f"Error: {response_data.get('error')} - {response_data.get('detail', 'No details')}"
                    st.error(assistant_response)
                # Assuming successful response has a 'response' or similar key (ADAPT based on your actual API response structure)
                elif isinstance(response_data, dict):
                    # --- !!! ADAPT THIS PART !!! ---
                    # Look for the actual text content in the response dict.
                    # Common patterns: response_data['response'], response_data['result']['content'], response_data['choices'][0]['message']['content'], etc.
                    assistant_response = response_data.get(
                        "response"
                    )  # EXAMPLE KEY, CHANGE IF NEEDED
                    if assistant_response is None:
                        # Try another common key or inspect the raw response
                        assistant_response = response_data.get(
                            "generated_text"
                        )  # Another example
                    if assistant_response is None:
                        # If still None, show the raw dict for debugging
                        assistant_response = f"Received complex response: ```json\n{json.dumps(response_data, indent=2)}\n``` (Please adapt frontend to parse this structure)"
                        st.warning(
                            "Need to adapt frontend parser for API response structure."
                        )
                else:
                    # Handle unexpected response format (e.g., just a string)
                    assistant_response = str(response_data)

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": assistant_response}
                )
                with st.chat_message("assistant"):
                    st.markdown(assistant_response)
            else:
                # Handle case where API call failed in api_client itself
                error_message = "Failed to get response from the backend."
                st.error(error_message)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": error_message}
                )
