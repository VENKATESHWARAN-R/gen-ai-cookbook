"""
main.py
- creates an instance of the FastAPI app and runs it
"""
import logging
from fastapi import FastAPI
from ai_core.routes import llm_routes, ui_routes
from ai_core.config import configs as app_configs
import uvicorn
#from app.services.llm_service import get_model_instance

logger = logging.getLogger(__name__)

# Load configurations and initialize app
api_configs = app_configs.get("api_configs", {})
model_configs = app_configs.get("model_configs", {})

# Model = get_model_instance()

app = FastAPI(
    title=api_configs.get("title", "AI Core API"),
    description=api_configs.get("description", "API for the Operator model"),
    version=api_configs.get("version", "0.1.0"),
    root_path=api_configs.get("root_path", "")
)

# Include routes
app.include_router(llm_routes.router, prefix="/api/llm", tags=["LLM Inference"])
app.include_router(ui_routes.router, prefix="/llm", tags=["UI"])

# Run the app if this file is executed as the main module.
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
