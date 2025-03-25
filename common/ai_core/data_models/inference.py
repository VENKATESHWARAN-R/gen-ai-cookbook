"""
This module defines the request and response schemas for the inference endpoint.

Classes:
    InferenceRequest (pydantic.BaseModel): Schema for the inference request, including parameters such as prompt, 
                                           max_new_tokens, temperature, top_p, skip_special_tokens, formated_prompt, 
                                           and system_prompt.
    InferenceResponse (pydantic.BaseModel): Schema for the inference response, containing the generated text.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class GenerateTextRequest(BaseModel):
    prompt: str = Field(..., example="What is the meaning of life?")
    max_new_tokens: Optional[int] = Field(None, example=50)
    skip_special_tokens: Optional[bool] = Field(True, example=True)
    kwargs: Optional[Dict[str, Any]] = Field({}, example={"temperature": 0.7})

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "What is the meaning of life?",
                "max_new_tokens": 50,
                "kwargs": {"temperature": 0.7}
            }
        }

class GenerateResponseRequest(BaseModel):
    prompt: str = Field(..., example="Explain relativity in simple terms.")
    system_prompt: Optional[str] = Field(None, example="You are a science teacher.")
    tools_schema: Optional[str] = Field(None, example="{'function_name': 'explain', 'args': ['concept']}")
    documents: Optional[List[Dict[str, Any]]] = Field(
        None,
        example=[{"id": "Doc1", "content": "Relativity is the theory by Einstein..."}]
    )
    create_chat_session: Optional[bool] = Field(False, example=False)
    chat_history: Optional[List[Dict[str, str]]] = Field(
        None,
        example=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi, how can I help?"}
        ]
    )
    max_new_tokens: Optional[int] = Field(None, example=60)
    skip_special_tokens: Optional[bool] = Field(False, example=False)
    kwargs: Optional[Dict[str, Any]] = Field({}, example={"temperature": 0.8})

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Explain relativity in simple terms.",
                "system_prompt": "You are a science teacher.",
                "tools_schema": "{'function_name': 'explain', 'args': ['concept']}",
                "documents": [{"id": "Doc1", "content": "Relativity is the theory by Einstein..."}],
                "chat_history": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi, how can I help?"}
                ],
                "max_new_tokens": 60,
                "kwargs": {"temperature": 0.8}
            }
        }

class StatelessChatRequest(BaseModel):
    prompt: Optional[str] = Field("", example="What's the capital of France?")
    chat_history: Optional[List[Dict[str, str]]] = Field(
        None,
        example=[
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello! What can I do for you?"}
        ]
    )
    kwargs: Optional[Dict[str, Any]] = Field({}, example={"top_p": 0.9})

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "What's the capital of France?",
                "chat_history": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello! What can I do for you?"}
                ],
                "kwargs": {"top_p": 0.9}
            }
        }

class BrainstormRequest(BaseModel):
    messages: List[Dict[str, str]] = Field(
        ...,
        example=[
            {"role": "Alice", "content": "Let's discuss the future of AI."},
            {"role": "Bob", "content": "I think it will change everything!"}
        ]
    )
    role: Optional[str] = Field("", example="Bob")
    iam: Optional[str] = Field("", example="Alice")
    role_play_configs: Optional[List[Dict[str, str]]] = Field(
        None,
        example=[
            {"role": "Alice", "persona": "An innovative thinker."},
            {"role": "Bob", "persona": "A pragmatic strategist."}
        ]
    )
    ai_assisted_turns: Optional[int] = Field(0, example=1)
    kwargs: Optional[Dict[str, Any]] = Field({}, example={"max_new_tokens": 40})

    class Config:
        json_schema_extra = {
            "example": {
                "messages": [
                    {"role": "Alice", "content": "Let's discuss the future of AI."},
                    {"role": "Bob", "content": "I think it will change everything!"}
                ],
                "role": "Bob",
                "role_play_configs": [
                    {"role": "Alice", "persona": "An innovative thinker."},
                    {"role": "Bob", "persona": "A pragmatic strategist."}
                ],
                "iam": "Alice",
                "ai_assisted_turns": 1,
                "kwargs": {"max_new_tokens": 40}
            }
        }

class TokenCountResponse(BaseModel):
    token_counts: List[int]

class TextBatchInput(BaseModel):
    texts: List[str]