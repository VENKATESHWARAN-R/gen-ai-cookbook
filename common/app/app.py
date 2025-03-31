# app.py
import streamlit as st
from chat_view import render_chat_view
from brainstorm_view import render_brainstorm_view
import api_client  # To potentially check health

# --- Page Configuration (Optional but recommended) ---
st.set_page_config(
    page_title="LLM Interaction Hub",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Main App Logic ---

# Health Check (Optional - good practice)
# Perform once per session or periodically
if "backend_health" not in st.session_state:
    st.session_state.backend_health = "unknown"
    health_status = api_client.call_healthcheck()
    if (
        health_status
        and isinstance(health_status, dict)
        and health_status.get("status") == "healthy"
    ):  # Adapt based on actual healthcheck response
        st.session_state.backend_health = "ok"
    else:
        st.session_state.backend_health = "error"

if st.session_state.backend_health == "error":
    st.sidebar.error("Backend API connection failed. Please check the backend service.")
elif st.session_state.backend_health == "ok":
    st.sidebar.success("Backend API connection successful.")
else:
    st.sidebar.warning("Backend API health unknown.")


# --- Navigation ---
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the interface:", ["Chat", "Brainstorming"])

st.sidebar.markdown("---")  # Separator

# --- Render Selected View ---
if app_mode == "Chat":
    render_chat_view()
elif app_mode == "Brainstorming":
    render_brainstorm_view()

# Add footer or other common elements if needed
st.sidebar.markdown("---")
st.sidebar.info("POC LLM Frontend")
