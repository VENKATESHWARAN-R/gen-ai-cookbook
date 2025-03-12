"""
Base class for all language models.
This defines a common interface for all language models.
This is just to streamline for custom and specific usecases.

First Version: 2025-Mar-12
"""

import abc
import logging
import os
from typing import List, Dict, Any
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

RAG_PROMPT: str = """
You are an advanced AI assistant with expertise in retrieving and synthesizing information from provided references. 
Analyze the given documents and answer the question based strictly on their content.

## Context:
You will receive multiple documents, each with a unique identifier. Responses should be derived only from the given documents while maintaining clarity and conciseness. If insufficient information is available, state it explicitly.

## Instructions:
1. **Extract information** only from provided documents.
2. **Cite references** using document identifiers.
3. **Ensure coherence** when summarizing multiple sources.
4. **Avoid speculation** or external knowledge.
5. **State uncertainty** if the answer is unclear.

## Expected Output:
- A **concise and accurate** response with relevant citations.
- A disclaimer if the answer is unavailable.

## Documents:
{documents}

## User's Question:
{question}
"""

TOOL_CALLING_PROMPT: str = """
You are an expert in function composition. Given a question and available functions, determine the appropriate function/tool calls.

If a function is applicable, return it in the format:
[func_name1(param1=value1, param2=value2...), func_name2(params)]

If no function applies or parameters are missing, indicate that. Do not include extra text.

## Available Functions:
{functions_definition}
"""


# Creating a base class for the models, since we will be experimenting with different models which have different requirements
class BaseLLM(abc.ABC):
    """
    Abstract base class for LLM models, defining common functionality.
    """

    def __init__(
        self,
        model: str,
        max_history: int = 25,
        system_prompt: str = "",
        context_length: int = 8000,
        token_limit: int = 4096,
        **kwargs,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.system_prompt = system_prompt
        self.max_history = max_history
        self.context_length = context_length
        self.token_limit = token_limit
        self.history: List[Dict[str, str]] = []

        # Load model and tokenizer
        self.tokenizer = None
        self.model = None
        self.load_model_and_tokenizer(model, **kwargs)

    def load_model_and_tokenizer(self, model: str, **kwargs) -> None:
        """
        Loads the tokenizer and model.
        """
        self.logger.info("Initializing tokenizer and model...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model, torch_dtype=torch.bfloat16, **kwargs
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model, torch_dtype=torch.bfloat16, **kwargs
            )
            self.model.to(self.device)
        except Exception as e:
            self.logger.error(f"Error loading model '{model}': {e}")
            raise RuntimeError(f"Failed to load model '{model}'") from e

        self.logger.info("Loaded model: %s", model)
        self.logger.info("Model type: %s", type(self.model).__name__)
        self.logger.info("Number of parameters: %s", self.model.num_parameters())
        self.logger.info("Device: %s", self.device.type)

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 120,
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> str:
        """
        Generates text based on the given prompt.

        Parameters:
        ----------
        prompt : str
            The prompt text to generate text from.
        max_new_tokens : int, optional
            The maximum length of the generated text (default is 120).
        skip_special_tokens : bool, optional
            Flag to indicate if special tokens should be skipped (default is False).

        Returns:
        -------
        str
            The generated text.
        """
        if not prompt or not prompt.strip():
            self.logger.warning("Received empty prompt, returning empty response.")
            return ""

        self.logger.info("Generating response for prompt: %s", prompt)
        try:
            with torch.inference_mode():
                self.logger.debug("Tokenizing prompt...", prompt)
                print("Tokenizing prompt...", prompt)
                inputs = self.tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                _start_time = time.time()
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs,
                )
                _end_time = time.time()
                self.logger.debug("Time taken: %.2f seconds", _end_time - _start_time)

        except Exception as e:
            self.logger.error("Error generating response: %s", e)
            return f"Error generating response: {str(e)}"

        decoded_output = self.tokenizer.decode(
            outputs[0], skip_special_tokens=skip_special_tokens
        )
        self.logger.debug("Generated response: %s", decoded_output)
        print("Generated response: ", decoded_output)

        return decoded_output

    def get_token_count(self, text: str) -> int:
        """
        Gets the token count of the given text.
        """
        return len(self.tokenizer(text)["input_ids"])

    def trim_conversation(
        self, conversation_history: List[Dict[str, str]], token_limit: int
    ) -> List[Dict[str, str]]:
        """
        Trims the conversation history to fit within the given token limit,
        while retaining system messages and accounting for their token count.

        Parameters:
        ----------
        conversation_history : List[Dict[str, str]]
            List of messages with 'role' and 'content' keys.
        token_limit : int
            The maximum allowed token count.

        Returns:
        -------
        List[Dict[str, str]]
            Trimmed conversation history.
        """
        if not conversation_history or not isinstance(conversation_history, list):
            return []

        total_tokens = 0
        tokenized_history = []
        system_messages = []

        # Iterate through history and separate system messages
        for message in conversation_history:
            role = message.get("role")
            content = message.get("content")
            message_tokens = self.get_token_count(
                content
            )  # Get token count for all messages

            if role == "system":
                system_messages.append(
                    {"role": role, "content": content, "tokens": message_tokens}
                )
            else:
                total_tokens += message_tokens
                tokenized_history.append(
                    {"role": role, "content": content, "tokens": message_tokens}
                )

        # Include system messages in total token count
        system_token_count = sum(msg["tokens"] for msg in system_messages)

        # Adjust token limit to account for system messages
        available_token_budget = token_limit - system_token_count

        # If system messages alone exceed the limit, return only system messages
        if available_token_budget <= 0:
            return [
                {"role": msg["role"], "content": msg["content"]}
                for msg in system_messages
            ]

        # Trim non-system messages to fit within remaining token limit
        while total_tokens > available_token_budget and tokenized_history:
            removed_entry = tokenized_history.pop(0)
            total_tokens -= removed_entry["tokens"]

        # Return trimmed conversation while ensuring system messages are retained
        return [
            {"role": msg["role"], "content": msg["content"]} for msg in system_messages
        ] + [
            {"role": entry["role"], "content": entry["content"]}
            for entry in tokenized_history
        ]

    def clear_history(self) -> None:
        """Clears the stored conversation history."""
        self.history = []

    def add_to_history(self, user_input, model_response) -> None:
        """Adds an interaction to history and maintains max history size."""
        _user = {"role": "user", "content": user_input}
        _assistant = {"role": "assistant", "content": model_response}
        self.history.extend([_user, _assistant])
        if len(self.history) > self.max_history:
            self.history.pop(0)

    # Method for getting the templates
    def get_templates(self) -> Dict[str, str]:
        """
        Get the templates from the model.
        """
        return {
            "user_turn_template": self.user_turn_template,
            "assistant_turn_template": self.assistant_turn_template,
            "assistant_template": self.assistant_template,
            "rag_prompt": self.rag_prompt,
            "rag_prompt_template": self.rag_prompt_template,
            "tools_prompt_template": self.tools_prompt_template,
            "default_prompt_template": self.default_prompt_template,
            "non_sys_prompt_template": self.non_sys_prompt_template,
            "system_prompt_template": self.system_template,
            "tool_calling_prompt": self.tool_calling_prompt,
        }

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

        final_prompt = self.bos_token

        # Extract system prompt from chat history (if present)
        system_prompt = next(
            (msg["content"] for msg in chat_history if msg["role"] == "system"), None
        )
        if system_prompt:
            final_prompt += (
                f"\n{self.system_template.format(system_prompt=system_prompt)}"
            )

        # Format user and assistant exchanges
        for msg in chat_history:
            if msg["role"] == "user":
                final_prompt += (
                    f"\n{self.user_turn_template.format(user_prompt=msg['content'])}"
                )
            elif msg["role"] == "assistant":
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

        system_prompt = self.tool_calling_prompt.format(
            functions_definition=tools_schema
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
            f"**Document {doc['reference']}**: {doc['content']}" for doc in documents
        )

        system_prompt = self.rag_prompt.format(
            documents=formatted_documents, question=prompt
        )
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

    @abc.abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generates a response to the given prompt.
        """
        pass

    @abc.abstractmethod
    def chat(
        self, prompt: str, clear_session: bool = False, **kwargs
    ) -> Dict[str, Any]:
        """
        Abstract method for chatting with the model.
        """
        pass

    @property
    def device(self) -> torch.device:
        """Returns the available device (GPU or CPU)"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def rag_prompt(self) -> str:
        """Retrieves the RAG prompt from environment variables or uses the default."""
        prompt = os.getenv("RAG_PROMPT", RAG_PROMPT)

        if "{documents}" not in prompt:
            self.logger.warning("RAG_PROMPT is missing '{documents}' placeholder!")

        return prompt

    @property
    def tool_calling_prompt(self) -> str:
        """Retrieves the tool calling prompt from environment variables or uses the default."""
        return os.getenv("TOOL_CALLING_PROMPT", TOOL_CALLING_PROMPT)

    @property
    def bos_token(self) -> str:
        """Abstract property for BOS token that must be implemented in subclasses."""
        return ""

    @property
    def system_template(self) -> str:
        """System message template"""
        return "<SYSTEM_INSTRUCTIONS> {system_prompt} \n\n</SYSTEM_INSTRUCTIONS>"

    @property
    def user_turn_template(self) -> str:
        """User turn template"""
        return "<USER> {user_prompt} \n\n</USER>"

    @property
    def assistant_turn_template(self) -> str:
        """Assistant turn template"""
        return "<ASSISTANT> {assistant_response} </ASSISTANT>"

    @property
    def assistant_template(self) -> str:
        """Assistant response placeholder"""
        return "<ASSISTANT> "

    @property
    def rag_prompt_template(self) -> str:
        """Template for RAG-based prompts"""
        return f"{self.bos_token}\n{self.system_template}\n{self.user_turn_template}\n{self.assistant_template}"

    @property
    def tools_prompt_template(self) -> str:
        """Template for tool-using prompts"""
        return f"{self.bos_token}\n{self.system_template}\n{self.user_turn_template}\n{self.assistant_template}"

    @property
    def default_prompt_template(self) -> str:
        """Default fallback prompt template"""
        return f"{self.bos_token}\n{self.system_template}\n{self.user_turn_template}\n{self.assistant_template}"

    @property
    def non_sys_prompt_template(self) -> str:
        """Prompt template when no system prompt is provided"""
        return f"{self.bos_token}\n{self.user_turn_template}\n{self.assistant_template}"

    def __call__(self, prompt: str, **kwargs) -> str:
        """
        Enables direct inference by calling the model instance.
        """
        return self.generate_text(prompt, **kwargs)

    def __repr__(self):
        """
        Official string representation for debugging.
        """
        return f"{self.__class__.__name__}(model={self.model.name_or_path!r}, device={self.device})"

    def __str__(self):
        """
        User-friendly string representation.
        """
        return f"{self.__class__.__name__} running on {self.device.type}, max history: {self.max_history}"

    def __len__(self):
        """
        Returns the number of stored conversation history entries.
        """
        return len(self.history)

    def __getitem__(self, index):
        """
        Retrieves conversation history entries like an array.
        """
        return self.history[index]
