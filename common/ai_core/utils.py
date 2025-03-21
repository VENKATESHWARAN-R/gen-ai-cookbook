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
You are an AI assistant specialized in interpreting user requests and invoking appropriate tools based on the provided JSON schemas.

Below, within <tools></tools> tags, you'll find JSON schemas defining available tools:

<tools>
{functions_definition}
</tools>

### Instructions:
- Read and understand the user's query.
- Identify if the query matches an available tool.
- Ensure all required parameters for the tool are provided in the query.

### Response Format:

**1. When all required parameters are present:**
AI Thought: Briefly explain your reasoning.
AI Response: <tool_call>{{"name": "function_name", "args": {{args_json}}}}</tool_call>

**2. When parameters are missing:**
AI Thought: Mention clearly which parameter(s) are missing.
AI Response: Politely request the missing information from the user.

**3. When no available tools match:**
AI Thought: Briefly mention the reason why the tools can't handle the query.
AI Response: Inform the user that their request cannot be processed with the available tools.

### Example Scenarios:

**Example 1:**
User Query: Convert 100 USD to EUR.
AI Thought: The convert_currency tool can handle this request, and all required parameters (amount, from_currency, to_currency) are provided.
AI Response: <tool_call>{{"name": "convert_currency", "args": {{"amount": 100, "from_currency": "USD", "to_currency": "EUR"}}}}</tool_call>

**Example 2:**
User Query: Book me a flight from Paris.
AI Thought: The book_flight tool requires destination and date, which are missing.
AI Response: Please provide your destination city and travel date to book the flight.

**Example 3:**
User Query: Who won the football match today?
AI Thought: None of the provided tools can answer queries related to sports results.
AI Response: I'm unable to provide sports results with my current toolset.
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
