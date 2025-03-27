"""
Sub class for IBM GRanite specifically created for llama 3.2
This inherits the base llm and implements the specific methods for llama 3.2

First Version: 2025-Mar-13
"""

from typing import List
from .basellm import BaseLLM

def get_tool_calling_prompt() -> str:
    """
    Tool calling prompt for Granite model.
    Putting it inside a function to avoid spamming the global namespace.
    """
    return """
You are an expert in function composition. Given a question and available Tools, determine the appropriate function/tool calls.

If no function applies or parameters are missing, indicate that. Do not include extra text.

# Tools
You may call one or more functions to assist with the user query.

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML 
tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call><|end_of_text|>

<|start_of_role|>tools<|end_of_role|>{functions_definition}
"""


class GraniteLocal(BaseLLM):
    """
    A class to represent Granite local model for text generation.

    Attributes:
    ----------
    model : str
        The model name or path.
    max_history : int, optional
        The maximum number of history entries to keep (default is 5).
    system_prompt : str
        The system prompt.
    """

    def __init__(
        self,
        model: str = "",
        max_history: int = 10,
        system_prompt: str = None,
        **kwargs,
    ):
        _model = "ibm-granite/granite-3.2-2b-instruct" if not model else model
        super().__init__(_model, max_history, system_prompt, **kwargs)
        self.logger.debug("Default role of the AI assistant: %s", system_prompt)

    @property
    def special_tokens(self) -> List[str]:
        """Abstract property for special tokens that must be implemented in subclasses."""
        return [
            "<|start_of_role|>",
            "<|end_of_role|>",
            "<|end_of_text|>",
            "<|tool_call|>",
        ]

    @property
    def system_template(self) -> str:
        """System message template"""
        return "<|start_of_role|>system<|end_of_role|>{system_prompt}<|end_of_text|>"

    @property
    def user_turn_template(self) -> str:
        """User turn template"""
        return "<|start_of_role|>user<|end_of_role|>{user_prompt}<|end_of_text|>"

    @property
    def assistant_turn_template(self) -> str:
        """Assistant turn template"""
        return "<|start_of_role|>assistant<|end_of_role|>{assistant_response}<|end_of_text|>"

    @property
    def assistant_template(self) -> str:
        """Assistant response placeholder"""
        return "<|start_of_role|>assistant<|end_of_role|>"
    
    @property
    def tool_calling_prompt(self) -> str:
        """Tool calling prompt"""
        return get_tool_calling_prompt()
