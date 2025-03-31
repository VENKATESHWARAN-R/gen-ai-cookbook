"""
Sub class for Gemini Based on Gemini API's
This inherits the base llm and implements the specific methods for gemini

First Version: 2025-Mar-14
"""

from dataclasses import dataclass
import os
import time
from typing import List, Dict, Any, Union

from google import genai
from google.genai import types

from .basellm import BaseLLM, Role, RolePlay


# Creating dataclass
@dataclass
class Tokenizer:
    """
    Dataclass for Tokenizer, This is created to comply with base class rule
    """

    pad_token_id: int = 1997


class GeminiApi(BaseLLM):
    """
    A class to represent Gemini API for text generation.

    Attributes:
    ----------
    model : str
        The model name.
    max_history : int, optional
        The maximum number of history entries to keep (default is 10).
    """

    def __init__(
        self,
        model: str = "",
        max_history: int = 10,
        system_prompt: str = None,
        **kwargs,
    ):
        _model = "models/gemini-2.0-flash-lite" if not model else model
        # Check if GOOGLE_API_KEY is passed in kwargs
        if "GOOGLE_API_KEY" in kwargs:
            self._api_key = kwargs["GOOGLE_API_KEY"]
        elif "GOOGLE_API_KEY" in os.environ:
            self._api_key = os.environ["GOOGLE_API_KEY"]
        else:
            raise ValueError("GOOGLE_API_KEY is required to use Gemini API")
        self.gen_ai_client = None
        self.chat_client = None
        super().__init__(_model, max_history, system_prompt, **kwargs)
        self.context_length = 131_072
        self.token_limit = 4096
        self.logger.debug("Default role of the AI assistant: %s", system_prompt)

    def _load_model_and_tokenizer(self, model: str, **kwargs) -> None:
        """
        Loads the tokenizer and model.
        """
        self.logger.info("Initializing gemini API...")
        try:
            self.tokenizer: Tokenizer = Tokenizer()
            self.model = model
            self.gen_ai_client = genai.Client(api_key=self._api_key)
            self.chat_client = self.gen_ai_client.chats.create(model=self.model)
            self.logger.info("Loaded model: %s", self.model)
        except Exception as e:
            self.logger.error("Error consiguring model '%s': %s", model, e)
            raise RuntimeError(f"Failed to Configure model '{model}'") from e

    def _create_config_from_kwargs(self, max_output_tokens=None, **kwargs):

        config_data = {}

        if "stop_strings" in kwargs:
            config_data["stop_sequences"] = kwargs["stop_strings"].split(",")

        # Explicitly handle max_output_tokens
        if max_output_tokens is not None:
            config_data["max_output_tokens"] = max_output_tokens

        # Include direct matches from kwargs
        for field in types.GenerateContentConfig.model_fields.keys():
            if field not in config_data and field in kwargs:
                if kwargs[field] is not None:
                    config_data[field] = kwargs[field]

        # Create an instance of GenerateContentConfig
        return types.GenerateContentConfig(**config_data)

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = None,
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> Union[List[Dict], Dict]:
        """
        Generates text based on the given prompt.

        Parameters:
        ----------
        prompt : str
            The prompt text to generate text from.
        max_new_tokens : int, optional
            The maximum length of the generated text.
        skip_special_tokens : bool, optional
            Flag to indicate if special tokens should be skipped (default is True).

        Returns:
        -------
        str or list[str]
            The generated text. if the input is a list, the output will be a list.
        """
        # Pass all input arguments to GenerateContentConfig
        _max_tokens = max_new_tokens or self.token_limit
        config = self._create_config_from_kwargs(max_output_tokens=_max_tokens, **kwargs)

        self.logger.debug("Generating response for prompt: %s", prompt)
        try:
            _start_time = time.time()
            response = self.gen_ai_client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
            generated_text = response.text
            _end_time = time.time()
            _inference_time = _end_time - _start_time
            self.logger.debug("Time taken: %.2f seconds", _inference_time)

        except Exception as e:
            self.logger.error("Error generating response: %s", e)
            return {
                "response": f"Error generating response: {str(e)}",
            }

        self.logger.debug("Generated response: %s", generated_text)
        # print("Generated response: ", generated_text)

        usage_metadata = response.usage_metadata.to_json_dict()
        tool_calls = self._check_tool_calls(generated_text)

        return {
            "response": generated_text,
            "tool_calls": tool_calls,
            "usage_metadata": usage_metadata,
            "response_metadata": {"time_in_seconds": _inference_time},
        }

    def generate_response(
        self,
        prompt: str,
        system_prompt: str = None,
        tools_schema: str = None,
        documents: List[Dict] = None,
        create_chat_session: bool = False,
        chat_history: List[Dict] = None,
        max_new_tokens: int = None,
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
            The maximum length of the generated text.
        skip_special_tokens : bool, optional
            Flag to indicate if special tokens should be skipped (default is False).

        Returns:
        -------
        dict
            The response and chat history.
            e.g. {"response": "Generated response", "chat_history": [{"role": "user", "content": "User input"}, ...]}
        """
        _chat_history = []

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

        _model_response = self.generate_text(
            input_prompt,
            max_new_tokens=max_new_tokens,
            skip_special_tokens=skip_special_tokens,
            **kwargs,
        )
        model_response = _model_response.get("response", "Error generating response")
        _model_response.pop("response", None)

        for token in self.special_tokens:
            model_response = model_response.replace(token, "")
        model_response = model_response.strip()
        # Add the user input and model response to the chat history
        _chat_history.append({"role": Role.USER.value, "content": prompt})
        _chat_history.append({"role": Role.ASSISTANT.value, "content": model_response})

        return {
            "response": model_response,
            "chat_history": _chat_history,
            **_model_response,
        }

    def chat(
        self,
        prompt: str,
        chat_history: List[Dict] = None,
        clear_session: bool = False,
        stateless: bool = False,
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
            e.g. {"response": "Generated response", "chat_history": [{"role": "user", "content": "User input"}, ...]}
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

        # If self.history exists, use it as chat_history
        if self.history and not chat_history:
            chat_history = self.history

        # Converting chat_history to Gemini API format
        transformed_history, system_instruction = self.transform_history_for_gemini(
            history=chat_history, user_prompt=prompt
        )

        # Generate response
        generated_response = self.generate_text(
            transformed_history, system_instruction=system_instruction, **kwargs
        )
        _generated_response = generated_response.get(
            "response", "Error generating response"
        )
        # If no chat history is passed, add the user input and model response to the history
        if not _history_checker and not stateless:
            self.add_to_history(prompt, _generated_response)
            generated_response["chat_history"] = self.history
        else:  # if chat history is passed, return the chat history as is
            generated_response["chat_history"] = chat_history
            generated_response["chat_history"].extend(
                [
                    {"role": Role.USER.value, "content": prompt},
                    {"role": Role.ASSISTANT.value, "content": _generated_response},
                ]
            )

        return generated_response

    def get_token_count(self, text: str) -> int:
        """Approximates token count by splitting text on spaces."""
        return len(text.split()) if text else 0

    def healthcheck(self) -> Dict[str, Any]:
        """
        Performs a health check on the model, verifying its status and configuration.

        Returns:
            Dict[str, Any]: Health check status and model information.

        Example:
            >>> health_status = base_llm.healthcheck()
            >>> print(health_status)
            {"status": "healthy", "model_name": "Gemma3ForCausalLM", "num_parameters": 3880263168, "device": "cuda"}
        """
        try:
            model_response = self.generate_text("This is ahealth check, Just respond with - I'm Healthy")
            if (
                model_response.get("response", "").strip()
                == "Error generating response"
            ):
                raise RuntimeError("Model response error")
            return {"status": "healthy", "response": model_response}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def transform_history_for_gemini(self, history, user_prompt: str = None):
        """Transform the conversation history into the format expected by Google Gemini API."""
        # print("Transforming history for Gemini")
        # print("History: ", history)
        transformed_history = []
        _system_instruction = None
        for message in history:
            if message["role"] == Role.SYSTEM.value:
                _system_instruction = f"{message['content']}"
                continue
            transformed_message = {
                "role": message["role"],
                "parts": [{"text": message["content"]}],
            }
            transformed_history.append(transformed_message)
        if user_prompt:
            transformed_history.append(
                {"role": "user", "parts": [{"text": user_prompt}]}
            )
        return transformed_history, _system_instruction

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
        return ""

    @property
    def system_template(self) -> str:
        """System message template"""
        return "\n{system_prompt}\n USER:"

    @property
    def user_turn_template(self) -> str:
        """User turn template"""
        return "\n{user_prompt}\n"

    @property
    def assistant_turn_template(self) -> str:
        """Assistant turn template"""
        return "\n{assistant_response}\n"

    @property
    def assistant_template(self) -> str:
        """Assistant response placeholder"""
        return "\n"
