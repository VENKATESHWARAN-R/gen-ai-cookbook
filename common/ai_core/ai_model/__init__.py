"""
Setup the AI Core package.

This package contains the AI assistant classes for text generation.

Created on 2025-Mar-10
"""

from .basellm import RolePlay, Role
from .llamallm import LlamaLocal
from .phillm import PhiLocal
from .gemmallm import GemmaLocal
from .geminillm import GeminiApi
from .qwenllm import QwenLocal

__all__ = ["RolePlay", "Role", "LlamaLocal", "PhiLocal", "GemmaLocal", "QwenLocal", "GeminiApi"]