"""
Sub class for Gemini Based on Gemini API's
This inherits the base llm and implements the specific methods for gemini

First Version: 2025-Mar-14
"""

import os
import time
from typing import List, Dict, Any, Union

from google import genai
from google.genai import types

from .basellm import BaseLLM, Role, RolePlay


class GemmaLocal(BaseLLM):
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
        _model = "models/gemini-2.0-flash" if not model else model
        # Check if GOOGLE_API_KEY is passed in kwargs
        if "GOOGLE_API_KEY" in kwargs:
            self._API_KEY = kwargs["GOOGLE_API_KEY"]
        elif "GOOGLE_API_KEY" in os.environ:
            self._API_KEY = os.environ["GOOGLE_API_KEY"]
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
            self.tokenizer: Dict = {}
            self.model = model
            self.gen_ai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            self.chat_client = self.gen_ai_client.chats.create(model=self.model)
        except Exception as e:
            self.logger.error("Error consiguring model '%s': %s", model, e)
            raise RuntimeError(f"Failed to Configure model '{model}'") from e
        
    
    def _create_config_from_kwargs(self, max_output_tokens=None, **kwargs):
        # Mapping of expected alternative argument names
        mapping = {
            "stop_strings": "stop_sequences",
        }

        # Prepare data for GenerateContentConfig
        config_data = {
            key: kwargs.get(value, kwargs.get(key))
            for key, value in mapping.items()
        }

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
        prompt: Union[List[str], str],
        max_new_tokens: int = None,
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> Union[List[str], str]:
        """
        Generates text based on the given prompt.

        Parameters:
        ----------
        prompt : str or list[str]
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
        config = self._create_config_from_kwargs(max_new_tokens=_max_tokens, **kwargs)
        
        if isinstance(prompt, str):
            prompt = [prompt]

        self.logger.info("Generating response for prompt: %s", prompt)
        try:
            self.logger.debug("Tokenizing prompt: %s", prompt)
            _start_time = time.time()
            print(f"Generating response for prompt: \n\n {prompt} \n\n")
            response = self.gen_ai_client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
            generated_text = response.text
            _end_time = time.time()
            self.logger.debug("Time taken: %.2f seconds", _end_time - _start_time)

        except Exception as e:
            self.logger.error("Error generating response: %s", e)
            return f"Error generating response: {str(e)}"

        self.logger.debug("Generated response: %s", generated_text)
        print("Generated response: ", generated_text)

        return generated_text

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

        model_response = self.generate_text(
            input_prompt,
            max_new_tokens=max_new_tokens,
            skip_special_tokens=skip_special_tokens,
            **kwargs,
        )

        for token in self.special_tokens:
            model_response = model_response.replace(token, "")
        model_response = model_response.strip()
        # Add the user input and model response to the chat history
        _chat_history.append({"role": Role.USER.value, "content": prompt})
        _chat_history.append({"role": Role.ASSISTANT.value, "content": model_response})

        return {"response": model_response, "chat_history": _chat_history}

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

        return_dict: Dict[str, Any] = {}

        # Initialize chat history if not provided
        if chat_history is None:
            chat_history = []
            _history_checker = False

        # If self.history exists, use it as chat_history
        if self.history and not chat_history:
            chat_history = self.history

        # Converting chat_history to Gemini API format
        transformed_history, system_instruction = self.transform_history_for_gemini(chat_history, prompt)

        # Generate response
        generated_response = self.generate_text(
            transformed_history, system_instruction=system_instruction, **kwargs
        )
        return_dict["response"] = generated_response
        # If no chat history is passed, add the user input and model response to the history
        if not _history_checker and not stateless:
            self.add_to_history(prompt, generated_response)
            return_dict["chat_history"] = self.history
        else:  # if chat history is passed, return the chat history as is
            return_dict["chat_history"] = chat_history
            return_dict["chat_history"].extend(
                [
                    {"role": Role.USER.value, "content": prompt},
                    {"role": Role.ASSISTANT.value, "content": generated_response},
                ]
            )

        return return_dict

    def stateless_chat(
        self,
        prompt: str = "",
        chat_history: List[Dict[str, str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Chat with the model in a stateless manner.

        Parameters:
        ----------
        prompt : str, optional
            The user prompt. if user prompt is not provided, the last user input from chat history will be used (default is "").
        chat_history : list, optional
            The chat history for the prompt (default is None).

        Returns:
        -------
        dict
            The response and chat history.
            e.g. {"response": "Generated response", "chat_history": [{"role": "user", "content": "User input"}, ...]}
        """
        if not prompt and not chat_history:
            return {"response": "No prompt provided", "chat_history": []}
        if chat_history:
            chat_history = self.trim_conversation(chat_history, self.context_length)
        user_input = chat_history.pop()["content"] if not prompt else prompt
        return self.chat(
            user_input, chat_history=chat_history, stateless=True, **kwargs
        )

    def multi_role_chat(
        self,
        messages: List[Dict[str, str]],
        role: str,
        role_play_configs: List[RolePlay] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Chat with the model in a multi-role manner.

        Parameters:
        ----------
        messages : List[Dict[str, str]]
            The list of messages with 'role' and 'content' keys.
        role : str
            The Role for the Ai assistant.
        role_play_configs : List[RolePlay], optional
            The list of role-play configurations (default is None).

        Returns:
        -------
        dict
            The response and chat history.
            e.g. {"response": "Generated response", "chat_history": [{"role": "user", "content": "User input"}, ...]}
        """

        role_play_configs = role_play_configs or self.role_play_configs

        # Extract unique roles from messages
        available_roles = {rp.role for rp in role_play_configs}

        # Get the persona for each role in messages
        role_persona_map = {rp.role: rp.persona for rp in role_play_configs}

        # Construct prompt with role-play personas
        system_instruction = (
            f"You are in a group chatt. Your assigned role is {role}. Always respond in the {role} persona conceisly.\n\n"
            f"Here are the personas for each role:\n\n"
        )
        for persona_role, persona in role_persona_map.items():
            system_instruction += f"- {persona_role}: {persona}\n\n"

        chat_history = self.trim_conversation(messages, self.context_length)

        # Preparing messages for the chat option
        chat_input = [{"role": Role.SYSTEM.value, "content": system_instruction}]

        for message in chat_history:
            _content = message["role"] + ": " + message["content"]
            if message["role"] == role:
                chat_input.append({"role": Role.ASSISTANT.value, "content": _content})
            else:
                chat_input.append({"role": Role.USER.value, "content": _content})

        chat_response = self.stateless_chat(chat_history=chat_input, **kwargs)
        model_response = chat_response.get(
            "response", "Error generating response"
        ).strip()

        # Removing Role Tokens from the response
        for roles in available_roles:
            model_response = (
                model_response.replace(roles + ":", "", 1)
                if model_response.startswith(roles + ":")
                else model_response
            )
        # Append the model response to the chat history
        _messages = messages.copy()
        _messages.append({"role": role, "content": model_response})

        return {"response": model_response, "chat_history": _messages}
    
    def transform_history_for_gemini(history, user_prompt:str = None):
        """Transform the conversation history into the format expected by Google Gemini API."""
        transformed_history = []
        _system_instruction = None
        for message in history:
            if message["role"] == Role.SYSTEM.value:
                _system_instruction = f"{message['content']}"
                continue
            transformed_message = {
                "role": message["role"],
                "parts": [
                    {
                        "text": message["content"]
                    }
                ]
            }
            transformed_history.append(transformed_message)
        if user_prompt:
            transformed_history.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": user_prompt
                        }
                    ]
                }
            )
        return transformed_history, _system_instruction

    def format_prompt(
        self,
        prompt: str,
        system_prompt: str = None,
        tools_schema: str = None,
        documents: List[Dict] = None,
        create_chat_session: bool = False,
        chat_history: List[Dict] = None,
    ) -> str:
        """
        Formats the prompt using the prompt template, handling chat history,
        tools, system prompts, and document-based context.
        """

        system_prompt = system_prompt or self.system_prompt

        # Handle chat history formatting
        if chat_history:
            return self._format_chat_history(prompt, chat_history)

        # Create new chat session formatting
        elif create_chat_session:
            return self._format_new_chat_session(prompt, system_prompt)

        # Format prompt for tool usage
        elif tools_schema:
            return self._format_tools(prompt, tools_schema)

        # Format prompt for RAG (Retrieval-Augmented Generation)
        elif documents:
            return self._format_documents(prompt, documents)

        # Default formatting (with or without system prompt)
        return self._format_default_prompt(prompt, system_prompt)

    def _format_chat_history(self, prompt: str, chat_history: List[Dict]) -> str:
        """
        Formats the prompt including chat history, ensuring system messages are considered.
        """
        self.logger.debug("Formatting prompt with chat history")

        final_prompt: str = self.bos_token

        # Extract system prompt from chat history (if present)
        system_prompt = next(
            (
                msg["content"]
                for msg in chat_history
                if msg["role"] == Role.SYSTEM.value
            ),
            None,
        )
        if system_prompt:
            final_prompt += (
                f"\n{self.system_template.format(system_prompt=system_prompt)}"
            )

        # Format user and assistant exchanges
        for msg in chat_history:
            if msg["role"] == Role.USER.value:
                final_prompt += (
                    f"\n{self.user_turn_template.format(user_prompt=msg['content'])}"
                )
            elif msg["role"] == Role.ASSISTANT.value:
                final_prompt += f"\n{self.assistant_turn_template.format(assistant_response=msg['content'])}"

        # Append current user prompt and assistant placeholder
        final_prompt += f"\n{self.user_turn_template.format(user_prompt=prompt)}"
        final_prompt += f"\n{self.assistant_template}"

        return final_prompt

    def _format_new_chat_session(self, prompt: str, system_prompt: str) -> str:
        """
        Formats the prompt for a new chat session, ensuring the BOS token is set.
        """
        self.logger.debug("Formatting prompt for new chat session")

        final_prompt = self.bos_token
        if system_prompt:
            final_prompt += (
                f"\n{self.system_template.format(system_prompt=system_prompt)}"
            )

        final_prompt += f"\n{self.user_turn_template.format(user_prompt=prompt)}"
        final_prompt += f"\n{self.assistant_template}"

        return final_prompt

    def _format_tools(self, prompt: str, tools_schema: str) -> str:
        """
        Formats the prompt when tools (functions) are available.
        """
        self.logger.debug("Formatting prompt with tool schema")

        try:
            system_prompt = self.tool_calling_prompt.format(
                functions_definition=tools_schema
            )
        except Exception as e:
            self.logger.error("Error formatting tool schema: %s", e)
            system_prompt = self.tool_calling_prompt.replace(
                "{functions_definition}", tools_schema
            )
        return self.tools_prompt_template.format(
            system_prompt=system_prompt, user_prompt=prompt
        )

    def _format_documents(self, prompt: str, documents: List[Dict]) -> str:
        """
        Formats the prompt to include relevant document context.
        """
        self.logger.debug("Formatting prompt with documents")

        required_keys = {"reference", "content"}
        assert all(
            required_keys.issubset(doc.keys()) for doc in documents
        ), "Documents must contain 'reference' and 'content' keys."

        formatted_documents = "\n".join(
            f"**Document {doc.get('reference', 'No Reference available')}**: {doc.get('content', 'No Content available in this document - SKIP THIS')}"
            for doc in documents
        )

        try:
            system_prompt = self.rag_prompt.format(
                documents=formatted_documents, question=prompt
            )
        except Exception as e:
            self.logger.error("Error formatting RAG prompt: %s", e)
            system_prompt = self.rag_prompt.replace(
                "{documents}", formatted_documents
            ).replace("{question}", prompt)

        return self.rag_prompt_template.format(
            system_prompt=system_prompt, user_prompt=prompt
        )

    def _format_default_prompt(self, prompt: str, system_prompt: str) -> str:
        """
        Default prompt formatting when no chat history, tools, or documents are provided.
        """
        self.logger.debug("Formatting default prompt")

        if system_prompt:
            return self.default_prompt_template.format(
                system_prompt=system_prompt, user_prompt=prompt
            )

        return self.non_sys_prompt_template.format(user_prompt=prompt)

    def get_token_count(self, text: str) -> int:
        """
        Gets the token count of the given text.
        """
        return "NOT IMPLEMENTED"

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
        return "\n{system_prompt}\n"

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
