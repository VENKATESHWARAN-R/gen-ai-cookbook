"""
This module provides utility classes and constants for AI-related functionalities, including role-play configurations,
document handling, and prompt templates for AI interactions.

Classes:
    RolePlay (pydantic.BaseModel): Represents a role and associated persona for role-play scenarios.
    Role (enum.Enum): Defines possible roles within interactions.
    ChatMessage (pydantic.BaseModel): Represents a chat message with a role and content.
    Document (pydantic.BaseModel): Represents a document with an ID, title, content, and optional metadata.

Constants:
    RAG_PROMPT (str): Template for generating responses based on provided documents.
    TOOL_CALLING_PROMPT (str): Template for determining appropriate function/tool calls based on a question and available functions.

"""

from enum import Enum
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional


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

BRAINSTROM_TURN_PROMPT: str = """
You are a helpful AI assistant in a brainstorming session.
Your goal is to determine who should logically respond next, ensuring a smooth and productive discussion.

### Given Team and their Personas:
{roles_personas}

### Conversation History:
{conversation_history}

---
Review the conversation history to determine the logical flow of discussion and determine the next speaker.
Never ask the last speaker to speak again.
If the last speaker asked a question to an available team member, ask that team member to respond.
Now Think concisely step by step and provide the final name inside the xml tags <name>...</name>.
"""


class RolePlay(BaseModel):
    """
    Represents a role and associated persona for role-play scenarios.

    Attributes:
        role (str): The role name.
        persona (str): The personality traits or description associated with the role.
    """

    role: str = Field(..., example="wizard")
    persona: str = Field(..., example="wise and mysterious")


# Creating enum class for the roles
class Role(Enum):
    """
    Defines possible roles within interactions.

    Members:
        USER: Represents the user's input.
        ASSISTANT: Represents the assistant's response.
        SYSTEM: Represents system-level prompts or instructions.
    """

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    role: Role = Field(..., description="Role of the message sender", example="user")
    content: str = Field(
        ..., description="Content of the message", example="Hello, how are you?"
    )


class Document(BaseModel):
    id: str = Field(
        ..., description="Unique identifier for the document", example="doc_001"
    )
    title: Optional[str] = Field(
        None,
        description="Title or brief heading of the document",
        example="Quantum Mechanics Basics",
    )
    content: str = Field(
        ...,
        description="Full content or main text of the document",
        example="Quantum mechanics describes phenomena at atomic scales...",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional metadata associated with the document",
    )
