"""
Sub class for qwen specifically created for qwen2.5
This inherits the base llm and implements the specific methods for Phi-4-mini-instruct

First Version: 2025-Mar-13
"""

from typing import List
from .basellm import BaseLLM


QWEN_TOOL_CALLING_PROMPT: str = """
You are an expert in function composition. Given a question and available Tools, determine the appropriate function/tool calls.

If no function applies or parameters are missing, indicate that. Do not include extra text.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:

<tools>
{functions_definition}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML 
tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
"""


class QwenLocal(BaseLLM):
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
        _model = "Qwen/Qwen2.5-3B-Instruct" if not model else model
        super().__init__(_model, max_history, system_prompt, **kwargs)
        self.logger.debug("Default role of the AI assistant: %s", system_prompt)

    @property
    def special_tokens(self) -> List[str]:
        """Abstract property for special tokens that must be implemented in subclasses."""
        return [
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
        ]

    @property
    def system_template(self) -> str:
        """System message template"""
        return "<|im_start|>system\n{system_prompt}<|im_end|>"

    @property
    def user_turn_template(self) -> str:
        """User turn template"""
        return "<|im_start|>user\n{user_prompt}<|im_end|>"

    @property
    def assistant_turn_template(self) -> str:
        """Assistant turn template"""
        return "<|im_start|>assistant\n{assistant_response}<|im_end|>"

    @property
    def assistant_template(self) -> str:
        """Assistant response placeholder"""
        return "<|im_start|>assistant\n"

    @property
    def tool_calling_prompt(self) -> str:
        """Tool calling prompt"""
        return QWEN_TOOL_CALLING_PROMPT
