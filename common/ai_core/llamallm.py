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

    def generate_response(
        self,
        prompt: str,
        system_prompt: str = None,
        tools_schema: str = None,
        documents: List[Dict] = None,
        create_chat_session: bool = False,
        chat_history: List[Dict] = None,
        max_new_tokens: int = 120,
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generates text based on the given prompt.

        Parameters:
        ----------
        prompt : str
            The prompt text to generate text from.
        system_prompt : str, optional
            The system prompt text (default is None).
        tools_schema : str, optional
            The schema for the tools prompt (default is None).
        documents : list, optional
            The list of documents for the RAG prompt (default is None).
        create_chat_session : bool, optional
            Flag to indicate if a chat session should be created (default is False).
        chat_history : list, optional
            The chat history for the prompt (default is None).
        max_new_tokens : int, optional
            The maximum length of the generated text (default is 120).
        skip_special_tokens : bool, optional
            Flag to indicate if special tokens should be skipped (default is False).

        Returns:
        -------
        str
            The generated text.
        """
        _chat_history = []
        split_token = "<|end_header_id|>"
        special_tokens = [
            "<|begin_of_text|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eot_id|>",
        ]

        if chat_history:
            _chat_history.extend(chat_history)

        input_prompt = self.format_prompt(
            prompt,
            system_prompt=system_prompt,
            tools_schema=tools_schema,
            documents=documents,
            create_chat_session=create_chat_session,
            chat_history=chat_history,
        )

        model_response = self.generate_text(
            input_prompt,
            max_new_tokens=max_new_tokens,
            skip_special_tokens=skip_special_tokens,
            **kwargs,
        )
        # removing the prompt and special tokens from the model response
        model_response = model_response.split(split_token)[-1]
        for token in special_tokens:
            model_response = model_response.replace(token, "")
        model_response = model_response.strip()
        # Add the user input and model response to the chat history
        _chat_history.append({"role": "user", "content": prompt})
        _chat_history.append({"role": "assistant", "content": model_response})

        return {"response": model_response, "chat_history": _chat_history}

    def chat(
        self,
        prompt: str,
        chat_history: List[Dict] = None,
        clear_session: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Chat with the model.

        Parameters:
        ----------
        prompt : str
            The user prompt.
        clear_session : bool, optional
            Flag to indicate if the session history should be cleared (default is False).

        Returns:
        -------
        dict
            The response and chat history.
        """
        _history_checker: bool = (
            True  # flag to see if the chat history is passed, so we can return the chat history in the response without affecting original
        )
        if clear_session:
            self.clear_history()

        # Initialize chat history if not provided
        if chat_history is None:
            chat_history = []
            _history_checker = False

        # Determine if we need to create a new chat session
        create_chat_session = not self.history and not chat_history

        # If self.history exists, use it as chat_history
        if self.history and not chat_history:
            chat_history = self.history
        # Adding the chat prompt to chat history
        generated_response = self.generate_response(
            prompt,
            create_chat_session=create_chat_session,
            chat_history=chat_history,
            **kwargs,
        )

        extracted_response = generated_response.get(
            "response", "Error generating response"
        )

        # If no chat history is passed, add the user input and model response to the history
        if not _history_checker:
            self.add_to_history(prompt, extracted_response)
            generated_response["chat_history"] = self.history
        else:  # if chat history is passed, return the chat history as is
            generated_response["chat_history"] = chat_history
            generated_response["chat_history"].extend(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": extracted_response},
                ]
            )

        return generated_response

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
