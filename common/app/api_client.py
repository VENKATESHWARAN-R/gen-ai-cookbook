# api_client.py
import json
from typing import Any, Dict, List, Optional

import requests
from config import BACKEND_API_BASE_URL, BACKEND_API_TIMEOUT


# --- Helper Function ---
def _make_request(
    method: str,
    endpoint: str,
    payload: Optional[Dict] = None,
    params: Optional[Dict] = None,
) -> Optional[Any]:
    """Helper function to make requests to the backend API."""
    url = f"{BACKEND_API_BASE_URL}{endpoint}"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    try:
        if method.upper() == "POST":
            response = requests.post(
                url, headers=headers, json=payload, timeout=BACKEND_API_TIMEOUT
            )
        elif method.upper() == "GET":
            response = requests.get(
                url, headers=headers, params=params, timeout=BACKEND_API_TIMEOUT
            )
        else:
            print(f"Unsupported HTTP method: {method}")
            return None

        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        # Handle cases where response might be empty (e.g., 204 No Content)
        if response.status_code == 204 or not response.content:
            return None
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API request failed for {method} {url}: {e}")
        # Optionally return the error details if the response has JSON content
        try:
            error_detail = response.json()
            print(f"Error details: {error_detail}")
            return {"error": str(e), "detail": error_detail}
        except (json.JSONDecodeError, AttributeError):
            return {
                "error": str(e),
                "detail": (
                    response.text if hasattr(response, "text") else "No response body"
                ),
            }
    except json.JSONDecodeError:
        print(
            f"Failed to decode JSON response from {method} {url}. Response text: {response.text}"
        )
        return {"error": "Invalid JSON response", "detail": response.text}


# --- API Endpoint Functions ---


def call_stateless_chat(
    prompt: str,
    chat_history: Optional[List[Dict]] = None,
    kwargs: Optional[Dict] = None,
) -> Optional[Dict]:
    """Calls the /api/llm/stateless_chat endpoint."""
    endpoint = "/api/llm/stateless_chat"
    payload = {
        "prompt": prompt,
        "chat_history": chat_history or [],
        "kwargs": kwargs or {},
    }
    return _make_request("POST", endpoint, payload=payload)


def call_brainstorm(
    messages: List[Dict],
    role: Optional[str] = None,
    iam: Optional[str] = None,
    role_play_configs: Optional[List[Dict]] = None,
    ai_assisted_turns: int = 0,
    kwargs: Optional[Dict] = None,
) -> Optional[Dict]:
    """Calls the /api/llm/brainstorm endpoint."""
    endpoint = "/api/llm/brainstorm"
    payload = {
        "messages": messages,
        "role": role,
        "iam": iam,
        "role_play_configs": role_play_configs,
        "ai_assisted_turns": ai_assisted_turns,
        "kwargs": kwargs or {},
    }
    return _make_request("POST", endpoint, payload=payload)


def call_generate_response(
    prompt: str,
    system_prompt: Optional[str] = None,
    tools_schema: Optional[
        str
    ] = None,  # Assuming backend expects stringified JSON schema or similar
    documents: Optional[List[Dict]] = None,
    chat_history: Optional[List[Dict]] = None,
    max_new_tokens: Optional[int] = None,
    kwargs: Optional[Dict] = None,
) -> Optional[Dict]:
    """Calls the /api/llm/generate_response endpoint."""
    endpoint = "/api/llm/generate_response"
    payload = {
        "prompt": prompt,
        "system_prompt": system_prompt,
        "tools_schema": tools_schema,
        "documents": documents,
        "chat_history": chat_history or [],
        "max_new_tokens": max_new_tokens,
        "kwargs": kwargs or {},
        "create_chat_session": False,  # Explicitly set based on requirement for chat UI to manage state
    }
    # Clean payload - remove keys with None values if backend doesn't handle them well
    payload = {k: v for k, v in payload.items() if v is not None}

    return _make_request("POST", endpoint, payload=payload)


def call_healthcheck() -> Optional[Dict]:
    """Calls the /api/llm/healthcheck endpoint."""
    return _make_request("GET", "/api/llm/healthcheck")


# Add functions for other endpoints (/get_templates, /examples, /get-token-count) if needed
