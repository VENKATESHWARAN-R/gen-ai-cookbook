"""
Base class for all language models.
This defines a common interface for all language models.
This is just to streamline for custom and specific usecases.

First Version: 2025-Mar-12
"""

import json
import logging
import re
from typing import List, Dict, Any, Union, Optional, Callable
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .utils import (
    Role,
    RolePlay,
    ChatMessage,
    Document,
    RAG_PROMPT,
    TOOL_CALLING_PROMPT,
    BRAINSTROM_TURN_PROMPT,
)


# Creating a base class for the models,
# since we will be experimenting with different models which have different requirements
class BaseLLM:
    """
    Abstract base class for LLM models, defining common functionality.

    This class provides a common interface for interacting with different
    language models, handling tokenization, generation, chat history, and
    prompt formatting.  It's designed to be subclassed for specific models.

    Args:
        model (str): The name or path of the pre-trained model to load.
        max_history (int, optional): The maximum number of conversation turns
            to store in the history. Defaults to 25.
        system_prompt (str, optional):  A system prompt to provide initial
            instructions to the model. Defaults to "".
        context_length (int, optional): The maximum context length supported by the model.
             Defaults to 8192.
        token_limit (int, optional):  The maximum number of tokens for generation.
            Defaults to 4096.
        **kwargs: Additional keyword arguments passed to the
            `transformers.AutoTokenizer` and `transformers.AutoModelForCausalLM`
            constructors.  This can include parameters like `device_map`, etc.

    Attributes:
        logger (logging.Logger): Logger instance for the class.
        system_prompt (str): The system prompt.
        max_history (int): The maximum number of conversation turns to store.
        context_length (int): The model's context length.
        token_limit (int): The token generation limit.
        history (List[Dict[str, str]]): List to store conversation history.
            Each element is a dictionary with "role" and "content" keys.
        _role_play_configs (List[RolePlay]):  Configuration for role-playing.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for the model.
        model (transformers.PreTrainedModel): The loaded language model.
        _rag_prompt (str):  Prompt template for Retrieval Augmented Generation. Loaded from environment variable RAG_PROMPT if set, otherwise it is None.
        _tool_calling_prompt (str): Prompt template for tool calling. Loaded from environment variable TOOL_CALLING_PROMPT if set, otherwise it is None

    Example:
        >>> model = BaseLLM("model-name")
    """

    def __init__(
        self,
        model: str,
        max_history: int = 25,
        system_prompt: str = "",
        context_length: int = 8192,
        token_limit: int = 4096,
        **kwargs,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.system_prompt = system_prompt
        self.max_history = max_history
        self.context_length = context_length
        self.token_limit = token_limit
        self.history: List[Dict[str, str]] = []
        self._role_play_configs: Optional[List[RolePlay]] = []

        # Load model and tokenizer
        self.tokenizer = None
        self.model = None
        self._rag_prompt = RAG_PROMPT
        self._tool_calling_prompt = TOOL_CALLING_PROMPT
        self._brainstorm_turn_prompt = BRAINSTROM_TURN_PROMPT
        self._load_model_and_tokenizer(model, **kwargs)

        # Check if the tokenizer has a pad token and set it to eos_token if not
        # This is specially needed for Llama models
        try:
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except AttributeError:
            self.logger.warning(
                "Tokenizer does not have a pad_token_id attribute. Skipping setting pad_token."
            )

    def _load_model_and_tokenizer(self, model: str, **kwargs) -> None:
        """
        Internal method to initialize and load the tokenizer and language model from a specified Hugging Face checkpoint.

        Args:
            model (str): Identifier or path to the pre-trained model to load.
            **kwargs: Additional keyword arguments passed directly to Hugging Face's `from_pretrained` methods.

        Raises:
            RuntimeError: If model or tokenizer loading fails due to invalid configurations, missing files, or hardware compatibility issues.

        Side-effects:
            - Loads model and tokenizer into memory.
            - Moves the model to the configured device (CPU/GPU).
            - Sets tokenizer's padding token if missing (uses EOS token).

        Example:
            >>> self._load_model_and_tokenizer("meta-llama/Llama-2-7b-chat-hf")
            # Logs:
            # "Initializing tokenizer and model..."
            # "Loaded model: meta-llama/Llama-2-7b-chat-hf"
            # "Model type: LlamaForCausalLM"
            # "Number of parameters: 7,000,000,000"
            # "Device: cuda"
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
            self.model.eval()
        except Exception as e:
            self.logger.error("Error loading model '%s': %s", model, e)
            raise RuntimeError(f"Failed to load model '{model}'") from e

        self.logger.info("Loaded model: %s", model)
        self.logger.info("Model type: %s", type(self.model).__name__)
        self.logger.info("Number of parameters: %s", self.model.num_parameters())
        self.logger.info("Device: %s", self.device.type)

    def generate_text(
        self,
        prompt: Union[List[str], str],
        max_new_tokens: int = None,
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> Union[List[Dict], Dict]:
        """
        Generates text completion from the given prompt(s).

        Args:
            prompt (Union[List[str], str]): Single or list of prompts to generate responses for.
            max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to class token limit.
            skip_special_tokens (bool, optional): Excludes special tokens (like <EOS>) from the output. Defaults to True.
            **kwargs: Additional generation parameters passed to the underlying model.

        Returns:
            Union[List[Dict], Dict]: Generated text completion(s) as a list of dictionaries or a single dictionary with some extra meta data.

        Example:
            >>> result = llm.generate_text("What is AI?")
            >>> print(result.get('response'))
            "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines..."
        """
        _max_tokens = max_new_tokens or self.token_limit
        # print("Running inference for prompt: ", prompt)
        _prompt = prompt if isinstance(prompt, list) else [str(prompt)]

        self.logger.info("Generating response for prompt: %s", _prompt)
        try:
            with torch.inference_mode():
                self.logger.debug("Tokenizing prompt: %s", prompt)
                model_inputs = self.tokenizer(
                    prompt, padding=True, truncation=True, return_tensors="pt"
                )
                model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

                _start_time = time.time()
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=_max_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs,
                )
                _end_time = time.time()
                # Calculating inference time in seconds
                _inference_time = _end_time - _start_time
                self.logger.debug("Time taken: %.2f seconds", _inference_time)

        except Exception as e:
            self.logger.error("Error generating response: %s", e)
            return f"Error generating response: {str(e)}"

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(
                model_inputs.get("input_ids"), generated_ids
            )
        ]
        decoded_outputs = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=skip_special_tokens
        )
        self.logger.debug("Generated response: %s", decoded_outputs)
        # print("Generated response: ", decoded_outputs)

        usage_metadata = {
            "prompt_token_count": len(model_inputs.get("input_ids")[0]),
            "candidates_token_count": len(generated_ids[0]),
            "total_token_count": len(model_inputs.get("input_ids")[0])
            + len(generated_ids[0]),
            "model_context_length": self.context_length,
            "model_token_limit": self.token_limit,
        }
        tool_calls = [self._check_tool_calls(_) for _ in decoded_outputs]

        response_metadata = {"time_in_seconds": _inference_time}
        model_metadata = {
            "usage_metadata": usage_metadata,
            "response_metadata": response_metadata,
        }

        if len(decoded_outputs) == 1:
            return {
                "response": decoded_outputs[0].strip(),
                "tool_calls": tool_calls[0],
                **model_metadata,
            }
        return {
            "response": [output.strip() for output in decoded_outputs],
            "tool_calls": tool_calls,
            **model_metadata,
        }

    def generate_response(
        self,
        prompt: str,
        system_prompt: str = None,
        tools_schema: str = None,
        documents: Optional[List[Union[Document, Dict[str, Any]]]] = None,
        create_chat_session: bool = False,
        chat_history: Optional[List[Union[ChatMessage, Dict[str, str]]]] = None,
        max_new_tokens: int = None,
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generates a structured and contextualized response from the model, optionally guided by system prompts,
        schemas, document contexts (RAG), and previous conversation history.

        Args:
            prompt (str):
                User's query or prompt to the model.
            system_prompt (Optional[str]):
                Optional high-level instructions or context to guide model responses.
                Defaults to the internal class-defined prompt if None.
            tools_schema (Optional[str]):
                Schema definition for invoking structured tools or API calls through the model.
                Defaults to None (no tool schema).
            documents (Optional[List[Dict[str, Any]]]):
                List of documents to provide context for Retrieval-Augmented Generation (RAG).
                Each document is a dictionary structured per your RAG schema.
                Defaults to None.
            create_chat_session (bool):
                Indicates if this response should initiate a new conversational session context.
                Defaults to False.
            chat_history (Optional[List[Union[ChatMessage, Dict[str, str]]]]):
                Previous messages forming the conversation context, provided as a list of either:
                    - ChatMessage objects, or
                    - Dictionaries with keys 'role' (user/assistant/system) and 'content'.
                Defaults to None.
            max_new_tokens (Optional[int]):
                Maximum number of tokens to generate in the response.
                Defaults to class-configured token limit if None.
            skip_special_tokens (bool):
                If True, omits special tokens from the output.
                Defaults to False.
            **kwargs:
                Additional keyword arguments for fine-tuning model inference parameters.

        Returns:
            Dict[str, Any]:
                A dictionary containing:
                    - 'response' (str): The model-generated response.
                    - 'chat_history' (List[ChatMessage]): Updated conversation history including this interaction.

        Example:
            >>> response = llm.generate_response(
            ...     prompt="Explain relativity simply.",
            ...     system_prompt="You are an experienced science communicator.",
            ...     chat_history=[
            ...         {"role": "user", "content": "Hi there!"},
            ...         {"role": "assistant", "content": "Hello! How can I help?"}
            ...     ],
            ... )
            >>> print(response["response"])
            "Relativity explains how space and time are linked, showing that motion and gravity affect how we perceive time and distances."

        Raises:
            ValidationError:
                Raised by Pydantic if provided chat history inputs don't conform to the required schema.

        Notes:
            - Input messages are automatically validated and converted into `ChatMessage` instances if provided as dictionaries.
            - It's recommended to use clear, concise system prompts to guide the model's behavior effectively.
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

        _model_response: Dict = self.generate_text(
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
        chat_history: Optional[List[ChatMessage]] = None,
        clear_session: bool = False,
        stateless: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Initiates or continues a conversational interaction with stateful or stateless mode.

        Args:
            prompt (str):
                User input or question for the model.
            chat_history (Optional[List[ChatMessage]]):
                A list of prior conversation messages, each clearly defined with:
                    - role: Role Enum (user, assistant, system).
                    - content: Message content as a string.
                If omitted, the internal conversation history is used or initialized fresh.
            clear_session (bool):
                If `True`, clears existing internal conversation history before this interaction.
            stateless (bool):
                If `True`, prevents modifying internal conversation state, useful for isolated queries.
            **kwargs:
                Additional parameters for model inference.

        Returns:
            Dict[str, Any]:
                A dictionary containing:
                    - response (str): Model-generated reply.
                    - chat_history (List[ChatMessage]): Updated history including the latest interaction.

        Example:
            >>> response = llm.chat(
            ...     prompt="Tell me about space exploration.",
            ...     chat_history=[
            ...         ChatMessage(role=Role.USER, content="Hi"),
            ...         ChatMessage(role=Role.ASSISTANT, content="Hello! What can I help you with today?")
            ...     ],
            ...     clear_session=False,
            ...     stateless=False
            ... )
            >>> print(response["response"])
            "Space exploration involves the discovery of celestial structures and the exploration of outer space using spacecraft."

        Notes:
            - Messages provided must adhere to the ChatMessage schema.
            - Internal history management can be explicitly controlled using `clear_session` and `stateless`.
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
        if not _history_checker and not stateless:
            self.add_to_history(prompt, extracted_response)
            generated_response["chat_history"] = self.history
        else:  # if chat history is passed, return the chat history as is
            generated_response["chat_history"] = chat_history
            generated_response["chat_history"].extend(
                [
                    {"role": Role.USER.value, "content": prompt},
                    {"role": Role.ASSISTANT.value, "content": extracted_response},
                ]
            )

        return generated_response

    def stateless_chat(
        self,
        prompt: str = "",
        chat_history: Optional[List[ChatMessage]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generates a stateless model response based solely on provided input and optional context, without affecting internal state.

        Args:
            prompt (str):
                The prompt to generate a response for. If omitted, the last user message from `chat_history` is used.
            chat_history (Optional[List[ChatMessage]]):
                Conversation context as a list of `ChatMessage` instances. Ensures context-aware responses.
            **kwargs:
                Additional keyword arguments for model inference.

        Returns:
            Dict[str, Any]:
                Dictionary with:
                    - response (str): Generated reply.
                    - chat_history (List[ChatMessage]): The provided context extended with this interaction.

        Example:
            >>> response = llm.stateless_chat(
            ...     prompt="What's the highest mountain?",
            ...     chat_history=[
            ...         ChatMessage(role=Role.USER, content="Hello"),
            ...         ChatMessage(role=Role.ASSISTANT, content="Hi there, how can I help?")
            ...     ],
            ... )
            >>> print(response["response"])
            "Mount Everest is the highest mountain on Earth."

        Notes:
            - Ideal for one-off queries or interactions requiring context without persistent changes.
            - The internal session history remains untouched.
        """
        if not prompt and not chat_history:
            return {"response": "No prompt provided", "chat_history": []}
        if chat_history:
            chat_history = self.trim_conversation(chat_history, self.context_length)
        user_input = chat_history.pop()["content"] if not prompt else prompt
        return self.chat(
            user_input, chat_history=chat_history, stateless=True, **kwargs
        )

    def brainstrom(
        self,
        messages: List[Dict[str, str]],
        role: str = "",
        role_play_configs: Optional[List[Union[RolePlay, Dict[str, str]]]] = None,
        iam: str = None,
        ai_assisted_turns: int = 0,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Facilitates a multi-role chat interaction, enabling the model to respond based on specified personas and role configurations.

        Args:
            messages (List[Dict[str, str]]):
                A chronological list of messages exchanged, each message is represented as a dictionary with:
                    - 'role' (str): The role name of the sender.
                    - 'content' (str): The message content.
            role (str):
                The specific role for the model to adopt in this chat turn (e.g., 'manager', 'client', 'developer').
                Defaults to an empty string. if this is empty, the model will generate the next response based on the in the chat history and role_play_configs.
            role_play_configs (Optional[List[Union[RolePlay, Dict[str, str]]]]):
                Optional configurations for role-playing scenarios, specifying personas associated with roles. Each item can be a `RolePlay` instance or a dictionary conforming to:
                    - role (str): Name of the role.
                    - persona (str): Description of the associated personality traits.
                If not provided, the instance's default role-play configurations are used.
            ai_assisted_turns (int):
                Number of AI-assisted turns to generate in the conversation. Defaults to 0.
                If Provided The model will generate the next `ai_assisted_turns` responses based on the provided messages and the model will choose who will chat next.
            iam (str):
                The role of the user who is interacting with the model. This is used to reply to the user when it's the user turn. this only works when uing AI assisted turns
            **kwargs:
                Additional parameters for model inference and generation.

        Returns:
            Dict[str, Any]:
                A dictionary containing:
                    - 'response' (str): The generated response based on the assigned role and persona.
                    - 'chat_history' (List[Dict[str, str]]): Updated message history, including the new response appended.

        Raises:
            pydantic.ValidationError:
                If the provided `role_play_configs` dictionaries don't match the required schema.

        Example:
            >>> messages = [
            ...     {"role": "Dee", "content": "Team, we have a new feature request from Britney. Let's discuss the feasibility and timeline."},
            ...     {"role": "Britney", "content": "Yes, I want to add an **advanced search feature** to our platform. Users are struggling to find relevant items quickly."},
            ...     {"role": "Venkat", "content": "That sounds interesting. Britney, do you have any specific filters in mind? Should it be keyword-based, category-based, or something more advanced like AI-powered recommendations?"},
            ... ]
            >>> role_play_configs = [
            ...     {"role": "Britney", "persona": "A business client who values user experience and is focused on solving real-world problems for customers."},
            ...     {"role": "Venkat", "persona": "A skilled developer with deep technical expertise. He prioritizes efficiency, clean code, and optimal system design."},
            ...     {"role": "John", "persona": "A detail-oriented tester who thrives on finding edge cases and ensuring product stability."},
            ...     {"role": "Dee", "persona": "A meticulous and strategic manager who ensures projects run smoothly. He focuses on business goals, deadlines, and resource allocation."},
            ... ]
            >>> response = base_llm.multi_role_chat(
            ...     messages=messages,
            ...     role="John",
            ...     role_play_configs=role_play_configs
            ... )
            >>> print(response["response"])
            From a testing perspective, we need to ensure the search results are accurate. Venkat, how complex will the AI recommendations be? We might need test cases for various user behaviors.

        Notes:
            - Clients can directly provide dictionaries instead of `RolePlay` instances; these will be automatically validated and converted by Pydantic.
            - Ensure the provided `role` matches one of the roles specified in `role_play_configs` to get the desired persona-based response.
        """

        def format_roles_personas(role_configs: List[RolePlay]) -> str:
            """Formats role configurations into a readable string."""
            return "\n".join(f"- {r.role}: {r.persona}" for r in role_configs)

        def format_conversation_history(history: List[Dict[str, str]]) -> str:
            """Formats the conversation history into a string suitable for model input."""
            role_content_list = [
                f">> {entry['role']}: {entry['content']}" for entry in history
            ]
            role_content = "\n".join(role_content_list)
            role_content += (
                "\n\n---\nThe Last speaker is " + history[-1]["role"]
                if role_content_list
                else ""
            )
            return role_content

        def format_prompt_for_next_role(
            role_configs: List[RolePlay], history: List[Dict[str, str]]
        ) -> str:
            """Formats the prompt used to determine the next role."""
            return self._brainstorm_turn_prompt.format(
                roles_personas=format_roles_personas(role_configs),
                conversation_history=format_conversation_history(history),
            )

        def validate_role_configs(
            configs: List[Union[RolePlay, Dict[str, str]]],
        ) -> List[RolePlay]:
            """Validates and converts role-play configurations."""
            try:
                return [
                    RolePlay(**config) if isinstance(config, dict) else config
                    for config in configs
                ]
            except Exception as e:
                raise ValueError(
                    f"Invalid role-play configuration: {e}. Ensure valid RolePlay instances or dictionaries."
                )

        assert role != iam, "The role for AI and iam should not be the same"
        model_response: str = ""

        role_play_configs = validate_role_configs(
            role_play_configs or self.role_play_configs
        )
        available_roles = [rp.role for rp in role_play_configs]

        chat_history = messages.copy()

        for _ in range(max(ai_assisted_turns, 1)):
            if role:
                current_role = role
                reason = f"Your turn {iam}."
            else:
                current_role, reason = self._determine_next_role(
                    chat_history,
                    available_roles,
                    format_prompt_for_next_role,
                    role_play_configs=role_play_configs,
                )

            if current_role == iam:
                model_response = reason
                break

            system_instruction = (
                f"You are in a group chat. Act as {current_role}.\n"
                f"Try to keep your response concise and aligned with your persona\n"
                f"your persona: {next(rp.persona for rp in role_play_configs if rp.role == current_role)}\n"
                f"Following are the information about the other people and their personas in the group chat:\n"
                f"{format_roles_personas([rp for rp in role_play_configs if rp.role != current_role])}\n"
            )

            trimmed_history = self.trim_conversation(chat_history, self.context_length)

            chat_input = [{"role": Role.SYSTEM.value, "content": system_instruction}]

            for message in trimmed_history:
                prefix: str = f"{message['role']}: "
                chat_input.append(
                    {
                        "role": (
                            Role.ASSISTANT.value
                            if message["role"] == current_role
                            else Role.USER.value
                        ),
                        "content": prefix + message["content"],
                    }
                )

            chat_response: Dict[str, Any] = self.stateless_chat(
                chat_history=chat_input, **kwargs
            )
            model_response: str = chat_response.get(
                "response", "Error generating response"
            ).strip()

            # Clean up the response
            for r in available_roles:
                if model_response.startswith(f"{r}:"):
                    model_response = model_response[len(r) + 1 :].strip()
                    break

            chat_history.append({"role": current_role, "content": model_response})

            if not ai_assisted_turns:
                break
            role = ""  # Reset role for AI to decide next if looping

        return {"response": model_response, "chat_history": chat_history}

    def _get_model_info(self) -> Dict[str, Any]:
        """
        Returns model information such as model name, number of parameters, and device type.

        Returns:
            Dict[str, Any]: Model information dictionary.

        Example:
            >>> model_info = base_llm._get_model_info()
            >>> print(model_info)
            {"model_name": "Gemma3ForCausalLM", "num_parameters": 3880263168, "device": "cuda"}
        """
        return {
            "model_name": self.model._get_name(),
            "num_parameters": self.model.num_parameters(),
            "device": self.device.type,
            "model_version": self.model._version,
        }

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
            model_info = self._get_model_info()
            return {"status": "healthy", **model_info}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def _check_tool_calls(self, text: str) -> Dict:
        """
        Checks for tool calls in the given text and returns the extracted tool calls.

        Args:
            text (str): The text to check for tool calls.

        Returns:
            Dict: A dictionary containing the extracted tool calls.

        Example:
            >>> tool_calls = llm.check_tool_calls("<tool_call>{"functionCall": {"name": "get_user_info", "args": {"user_id": 7890, "special": "black"}}}</tool_call>")
            >>> print(tool_calls)
            {"functionCall": {"name": "get_user_info", "args": {"user_id": 7890, "special": "black"}}}
        """
        try:
            # Regex to extract JSON content inside <tool_call></tool_call>
            match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
            if match:
                tool_call_json = match.group(1).strip()
                return json.loads(tool_call_json)
        except json.JSONDecodeError as e:
            print(f"Error parsing tool call JSON: {e}")

        return {}

    def _extract_from_xml(self, response: str, tag: str) -> str:
        """
        Extracts content inside a specified XML-style tag from the AI response.

        Args:
            response (str): The AI's response containing the XML-style tag.
            tag (str): The name of the XML tag to extract content from.

        Returns:
            str: The extracted content or None if not found.
        """
        pattern = rf"<{tag}>(.*?)</{tag}>"  # Dynamic regex pattern for any tag
        match = re.search(pattern, response, re.IGNORECASE)
        return match.group(1) if match else response

    def _determine_next_role(
        self,
        messages: List[Dict[str, str]],
        available_roles: List[str],
        format_prompt_fn: Callable[[List[RolePlay], List[Dict[str, str]]], str],
        role_play_configs: Optional[List[RolePlay]],
        **kwargs: Any,
    ) -> tuple:
        """Determines the next role based on conversation history.

        Args:
            messages (List[Dict[str, str]]): Conversation history.
            available_roles (List[str]): List of roles available for selection.
            format_prompt_fn (Callable): Function to format prompts for determining next role.
            **kwargs (Any): Additional parameters.

        Returns:
            tuple: The next role and the generated response.
        """
        last_role = messages[-1]["role"]
        if len(available_roles) == 2:
            return next(role for role in available_roles if role != last_role), ""

        next_role_prompt = format_prompt_fn(role_play_configs, messages)
        self.logger.debug("Next role prompt: %s", next_role_prompt)

        _generated_response = self.generate_response(next_role_prompt, **kwargs)["response"]

        generated_role = (
            self._extract_from_xml(_generated_response, "name").split(":")[0].strip()
        )
        refined_response = _generated_response.split("<name>")[0].strip()

        if generated_role in available_roles:
            return generated_role, refined_response

        self.logger.warning(
            "Invalid role '%s' generated. Selecting next available role.",
            generated_role,
        )

        try:
            current_index = available_roles.index(last_role)
            return available_roles[(current_index + 1) % len(available_roles)], ""
        except Exception as e:
            self.logger.error("Error determining next role: %s", e)
            return available_roles[0], ""

    def validate_chat_history(
        self, chat_history: Optional[List[Union[ChatMessage, Dict]]]
    ) -> List[ChatMessage]:
        if not chat_history:
            return []
        return [
            msg if isinstance(msg, ChatMessage) else ChatMessage(**msg)
            for msg in chat_history
        ]

    def get_token_count(self, text: str) -> int:
        """
        Gets the token count of the given text.
        """
        return len(self.tokenizer(text)["input_ids"])

    # Method for getting the templates
    def get_templates(self) -> Dict[str, str]:
        """
        Retrieves the dictionary of prompt templates used by the model.

        Returns:
            Dict[str, str]: A dictionary containing keys for different template types (e.g., 'user_turn_template', 'assistant_turn_template') and their corresponding template strings.

        Example:
            >>> templates = model.get_templates()
            >>> print(templates["user_turn_template"])
            "{role}: {content}\n"
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
        Formats an input prompt by incorporating optional components such as system prompts, tool usage schemas, document context for retrieval-augmented generation (RAG), and chat history.

        Args:
            prompt (str): The primary input prompt or question provided by the user.
            system_prompt (str, optional): A system-level instruction guiding model behavior. Defaults to class-level default.
            tools_schema (str, optional): Schema defining available tools or APIs the model can invoke. Defaults to None.
            documents (List[Union[Document, Dict[str, Any]]], optional): List of relevant documents providing additional context for RAG-based generation. Defaults to None.
            create_chat_session (bool, optional): Flag indicating if a new chat session is to be created. Defaults to False.
            chat_history (List[Dict[str, str]], optional): Previous conversation history for context awareness. Defaults to None.

        Returns:
            str: The fully formatted prompt ready to be used for model inference.

        Example:
            >>> formatted_prompt = self.format_prompt(
            ...     prompt="Tell me about climate change impacts.",
            ...     system_prompt="You are an expert environmental scientist.",
            ...     documents=[
            ...         {"id": "Climate Report 2023", "content": "Climate change impacts are accelerating..."},
            ...     ],
            ... )
            >>> print(formatted_prompt)
            You are an expert environmental scientist.

            Relevant document:
            Climate Report 2023: Climate change impacts are accelerating...

            User: Tell me about climate change impacts.
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

    def trim_conversation(
        self, conversation_history: List[Dict[str, str]], token_limit: int
    ) -> List[Dict[str, str]]:
        """
        Trims conversation history based on token limits.

        Args:
            conversation_history (List[Dict[str, str]]): Full conversation history.
            token_limit (int): Maximum allowable tokens.

        Returns:
            List[Dict[str, str]]: Trimmed conversation history.
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

            if role == Role.SYSTEM.value:
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
        """
        Clears the entire conversation history.
        """
        self.history = []

    def add_to_history(self, user_input, model_response) -> None:
        """
        Adds entries to conversation history, respecting the maximum history limit.

        Args:
            user_input (str): Input provided by the user.
            model_response (str): Response generated by the model.
        """
        _user = {"role": Role.USER.value, "content": user_input}
        _assistant = {"role": Role.ASSISTANT.value, "content": model_response}
        self.history.extend([_user, _assistant])
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def _format_chat_history(self, prompt: str, chat_history: List[Dict]) -> str:
        """
        Formats the prompt including chat history, ensuring system messages are considered.
        """
        self.logger.debug("Formatting prompt with chat history")

        final_prompt: str = self.bos_token

        try:
            _chat_history: List[ChatMessage] = self.validate_chat_history(chat_history)
        except Exception as e:
            raise ValueError(f"Invalid chat history: {e}")

        # Merge consecutive messages of the same role
        merged_history = []
        for msg in _chat_history:
            if merged_history and merged_history[-1].role == msg.role:
                merged_history[
                    -1
                ].content += f"\n\n {msg.content}"  # Merge with space separator
            else:
                merged_history.append(msg)

        # Extract system prompt from merged history (if present)
        system_prompt = next(
            (msg.content for msg in merged_history if msg.role == Role.SYSTEM),
            None,
        )
        if system_prompt:
            final_prompt += (
                f"\n{self.system_template.format(system_prompt=system_prompt)}"
            )

        # Format user and assistant exchanges from merged history
        for msg in merged_history:
            if msg.role == Role.USER:
                final_prompt += (
                    f"\n{self.user_turn_template.format(user_prompt=msg.content)}"
                )
            elif msg.role == Role.ASSISTANT:
                final_prompt += f"\n{self.assistant_turn_template.format(assistant_response=msg.content)}"
            elif msg.role == Role.SYSTEM:
                pass
            else:
                self.logger.warning("Unknown role in chat history: %s", msg.role)

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

        validated_documents: List[Document] = []
        for doc in documents:
            if not isinstance(doc, Document):
                doc = Document(**doc)
            validated_documents.append(doc)

        formatted_documents = "\n".join(
            f"**Document {doc.id }**: {doc.content}" for doc in validated_documents
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

    @property
    def role_play_configs(self) -> List[RolePlay]:
        """Returns the role-play configurations."""
        return self._role_play_configs

    @role_play_configs.setter
    def role_play_configs(self, configs: List[RolePlay]) -> None:
        """Sets the role-play configurations."""
        self._role_play_configs = [
            RolePlay(**cfg) if isinstance(cfg, dict) else cfg for cfg in configs
        ]

    @property
    def device(self) -> torch.device:
        """Returns the available device (GPU or CPU)"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def rag_prompt(self) -> str:
        """Retrieves the RAG prompt from environment variables or uses the default."""
        if "{documents}" not in self._rag_prompt:
            self.logger.warning("RAG prompt does not contain document placeholder")
        return self._rag_prompt

    @rag_prompt.setter
    def rag_prompt(self, rag_prompt: str) -> None:
        """Sets the RAG prompt."""
        if "{documents}" not in rag_prompt:
            self.logger.warning("RAG prompt does not contain document placeholder")
        self._rag_prompt = rag_prompt

    @property
    def tool_calling_prompt(self) -> str:
        """Retrieves the tool calling prompt from environment variables or uses the default."""
        return self._tool_calling_prompt

    @tool_calling_prompt.setter
    def tool_calling_prompt(self, tool_calling_prompt: str) -> None:
        """Sets the tool calling prompt."""
        self._tool_calling_prompt = tool_calling_prompt

    @property
    def special_tokens(self) -> List[str]:
        """Abstract property for special tokens that must be implemented in subclasses."""
        return ["<SYSTEM_INSTRUCTIONS>", "<USER>", "<ASSISTANT>"]

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

    def __call__(self, prompt: Union[List[str], str], **kwargs) -> str:
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
