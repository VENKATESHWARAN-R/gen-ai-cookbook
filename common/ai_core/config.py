"""
This file is just for defining the configuration of the app.
"""

import os
import logging
from dotenv import load_dotenv

from ai_core.ai_model import LlamaLocal, PhiLocal, GemmaLocal, GeminiApi, QwenLocal

logger = logging.getLogger(__name__)


def load_environment():
    """Load environment variables from a dotenv file."""
    _environment = os.getenv("ENVIRONMENT", "development")
    _dotenv_file = f".env.{_environment}"
    common_env_path = os.path.join(os.path.dirname(__file__), _dotenv_file)
    # Load common environment variables
    load_dotenv(
        dotenv_path=common_env_path, override=False
    )  # Load common settings first

    return _environment


def setup_logging():
    """Initialize the logger based on environment settings."""
    logging_level = os.getenv("LOGGING_LEVEL", "INFO")
    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


logger = setup_logging()

environment = load_environment()

MODEL_REGISTRY = {
    "qwen": QwenLocal,
    "gemma": GemmaLocal,
    "llama": LlamaLocal,
    "phi": PhiLocal,
    "gemini": GeminiApi,
}

ingress_root_path = os.getenv("INGRESS_ROOT_PATH", "")

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", None)

DEFAULT_SYSTEM_PROMPT = os.getenv("DEFAULT_SYSTEM_PROMPT", None)


def load_configs() -> dict:
    """
    Load the configurations from environment variables.
    """
    model_class = os.getenv("MODEL", "phi")
    model_id = os.getenv("MODEL_ID", DEFAULT_MODEL)
    system_prompt = os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)

    output = {
        "model_configs": {
            "model_class": model_class,
            "model_id": model_id,
            "system_prompt": system_prompt,
        },
        "api_configs": {
            "ingress_root_path": ingress_root_path,
        },
    }
    return output


def load_model(model_name: str, model_id: str = None, system_prompt: str = None):
    """
    Loads and returns the requested model.
    If model_id is provided, pass it into the constructor (for local LLMs).
    If no model_id is given (or not applicable), load the default model or use an API key as needed.
    """
    model_class = MODEL_REGISTRY.get(model_name.lower())
    if not model_class:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {list(MODEL_REGISTRY.keys())}"
        )

    if model_name.lower() == "gemini":
        # For Gemini, we rely on the environment variable for the API key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in environment.")
        # Gemini doesn't usually need a model_id, but if you have a version:
        if model_id:
            logger.info(
                f"Note: Gemini API ignoring custom model_id={model_id}"
            )
        return model_class(GOOGLE_API_KEY=api_key)

    else:
        # For local models (GemmaLocal, LlamaLocal, QwenLocal, etc.)
        if model_id:
            return model_class(model_id, system_prompt=system_prompt)
        else:
            # If no model_id was provided, use the default constructor
            return model_class(system_prompt=system_prompt)


configs = load_configs()

logger.info("Environment: %s", environment)
logger.debug("Configs: %s", configs)
logger.info("Configuration loaded successfully.")
