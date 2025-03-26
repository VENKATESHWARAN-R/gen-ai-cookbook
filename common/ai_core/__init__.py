"""
Setup the AI Core package.

This package contains the AI assistant classes for text generation.

Created on 2025-Mar-23
"""

from .ai_model import RolePlay, Role
from .ai_model import LlamaLocal
from .ai_model import PhiLocal
from .ai_model import GemmaLocal
from .ai_model import GeminiApi
from .ai_model import QwenLocal

__all__ = [
    "RolePlay",
    "Role",
    "LlamaLocal",
    "PhiLocal",
    "GemmaLocal",
    "QwenLocal",
    "GeminiApi",
]
