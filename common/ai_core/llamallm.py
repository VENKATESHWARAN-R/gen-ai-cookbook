"""
Sub class for Llama specifically created for llama 3.2
This inherits the base llm and implements the specific methods for llama 3.2

First Version: 2025-Mar-13
"""

from typing import List, Dict, Any
from .basellm import BaseLLM


class LlamaLocal(BaseLLM):
    """
    A class to represent Llama local model for text generation.

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
        _model = "meta-llama/Llama-3.2-3B-Instruct" if not model else model
        super().__init__(_model, max_history, system_prompt, **kwargs)
        self.logger.debug("Default role of the AI assistant: %s", system_prompt)

    @property
    def split_token(self) -> str:
        """Abstract property for split token that must be implemented in subclasses."""
        return "<|end_header_id|>"

    @property
    def special_tokens(self) -> List[str]:
        """Abstract property for special tokens that must be implemented in subclasses."""
        return [
            "<|begin_of_text|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eot_id|>",
        ]

    @property
    def bos_token(self) -> str:
        """Abstract property for BOS token that must be implemented in subclasses."""
        return "<|begin_of_text|>"

    @property
    def system_template(self) -> str:
        """System message template"""
        return "<|start_header_id|>system<|end_header_id|> {system_prompt} <|eot_id|>"

    @property
    def user_turn_template(self) -> str:
        """User turn template"""
        return "<|start_header_id|>user<|end_header_id|> {user_prompt} <|eot_id|>"

    @property
    def assistant_turn_template(self) -> str:
        """Assistant turn template"""
        return "<|start_header_id|>assistant<|end_header_id|> {assistant_response} <|eot_id|>"

    @property
    def assistant_template(self) -> str:
        """Assistant response placeholder"""
        return "<|start_header_id|>assistant<|end_header_id|>"
