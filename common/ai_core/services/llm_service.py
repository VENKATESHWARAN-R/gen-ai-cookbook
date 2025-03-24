"""
This module provides services for interacting with a language model using FastAPI.
The module includes:
- Initialization and configuration of the language model.
"""

import logging
from fastapi import HTTPException

from ai_core.config import configs, load_model
from ai_core.ai_model.basellm import BaseLLM

logger = logging.getLogger(__name__)

# Load configurations
model_configs = configs.get("model_configs", {})
model_class = model_configs.get("model_class", "gemma")
model_id = model_configs.get("model_id", None)
sys_prompt = model_configs.get("system_prompt", None)

# Load the model
try:
    llm_instance: BaseLLM = load_model(model_class, model_id, sys_prompt)
    logger.info("Model %s loaded successfully.", model_class)
except Exception as e:
    logger.error("Error loading model", exc_info=True)
    raise HTTPException(status_code=500, detail="Model loading failed") from e
