"""
Sub class for Phi specifically created for Phi-4-mini-instruct
This inherits the base llm and implements the specific methods for Phi-4-mini-instruct

First Version: 2025-Mar-13
"""

from typing import List
from .basellm import BaseLLM


class PhiLocal(BaseLLM):
    """
    A class to represent Phi local model for text generation.

    Attributes:
    ----------
    model : str
        The model name or path.
    max_history : int, optional
        The maximum number of history entries to keep (default is 5).
    local_files_only : bool, optional
        Flag to indicate if the model is local or remote (default is False).
    tokenizer : AutoTokenizer
        The tokenizer for the model.
    model : AutoModelForCausalLM
        The model for causal language modeling.
    history : list
        The history of text inputs.
    """

    def __init__(
        self,
        model: str = "",
        max_history: int = 10,
        system_prompt: str = None,
        **kwargs,
    ):
        _model = "microsoft/Phi-4-mini-instruct" if not model else model
        super().__init__(_model, max_history, system_prompt, **kwargs)
        self.logger.debug("Default role of the AI assistant: %s", system_prompt)

    @property
    def split_token(self) -> str:
        """Abstract property for split token that must be implemented in subclasses."""
        return "<|assistant|>"

    @property
    def special_tokens(self) -> List[str]:
        """Abstract property for special tokens that must be implemented in subclasses."""
        return [
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
            "<|end|>",
        ]

    @property
    def system_template(self) -> str:
        """System message template"""
        return "<|system|>{system_prompt}<|end|>"

    @property
    def user_turn_template(self) -> str:
        """User turn template"""
        return "<|user|>{user_prompt}<|end|>"

    @property
    def assistant_turn_template(self) -> str:
        """Assistant turn template"""
        return "<|assistant|>{assistant_response}<|end|>"

    @property
    def assistant_template(self) -> str:
        """Assistant response placeholder"""
        return "<|assistant|>"
