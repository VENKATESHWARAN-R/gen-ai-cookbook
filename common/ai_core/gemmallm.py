"""
Sub class for Gemma specifically created for Gemma 3
This inherits the base llm and implements the specific methods for gemma 3

First Version: 2025-Mar-13
"""

from typing import List

import torch
from transformers import AutoTokenizer, Gemma3ForCausalLM

from .basellm import BaseLLM


class GemmaLocal(BaseLLM):
    """
    A class to represent Gemma local model for text generation.

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
    model : Gemma3ForCausalLM
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
        _model = "google/gemma-3-4b-it" if not model else model
        super().__init__(_model, max_history, system_prompt, **kwargs)
        self.logger.debug("Default role of the AI assistant: %s", system_prompt)

    def _load_model_and_tokenizer(self, model: str, **kwargs) -> None:
        """
        Loads the tokenizer and model.
        """
        self.logger.info("Initializing tokenizer and model...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model, torch_dtype=torch.bfloat16, **kwargs
            )
            self.model = Gemma3ForCausalLM.from_pretrained(
                model, torch_dtype=torch.bfloat16, **kwargs
            )
            self.model.to(self.device)
        except Exception as e:
            self.logger.error("Error loading model '%s': %s", model, e)
            raise RuntimeError(f"Failed to load model '{model}'") from e

        self.logger.info("Loaded model: %s", model)
        self.logger.info("Model type: %s", type(self.model).__name__)
        self.logger.info("Number of parameters: %s", self.model.num_parameters())
        self.logger.info("Device: %s", self.device.type)

    @property
    def special_tokens(self) -> List[str]:
        """Abstract property for special tokens that must be implemented in subclasses."""
        return [
            "<bos>",
            "<start_of_turn>user",
            "<start_of_turn>model",
            "<start_of_turn>",
            "<end_of_turn>",
        ]

    @property
    def bos_token(self) -> str:
        """Abstract property for BOS token that must be implemented in subclasses."""
        return "<bos>"

    @property
    def system_template(self) -> str:
        """System message template"""
        return "<start_of_turn>user\n{system_prompt}<end_of_turn>"

    @property
    def user_turn_template(self) -> str:
        """User turn template"""
        return "<start_of_turn>user\n{user_prompt}<end_of_turn>"

    @property
    def assistant_turn_template(self) -> str:
        """Assistant turn template"""
        return "<start_of_turn>model\n{assistant_response}<end_of_turn>"

    @property
    def assistant_template(self) -> str:
        """Assistant response placeholder"""
        return "<start_of_turn>model\n"
