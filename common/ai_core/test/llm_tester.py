# llm_tester.py

import json
from typing import List, Dict, Any, Optional

class LLMTester:
    """
    A test harness for LLMs that provides consistent test routines such as:
    - Simple text generation tests
    - RAG (Retrieval Augmented Generation) tests
    - Tools usage tests
    - Chat / multi-turn conversation tests
    - Role-based brainstorming (role_play_configs)
    """
    def __init__(self, model, model_name: str):
        self.model = model
        self.model_name = model_name
        self.results = []  # we'll collect test results here

    def test_generate_text(self, prompts: List[str], max_new_tokens: int = 128):
        """
        Tests the low-level text generation method.
        """
        responses = self.model.generate_text(prompts, max_new_tokens=max_new_tokens)
        
        # Store results
        self.results.append({
            'test_name': 'test_generate_text',
            'input': prompts,
            'output': responses,
            'model_name': self.model_name
        })
        return responses

    def test_generate_response(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        documents=None,
        tools_schema=None,
        max_new_tokens: int = 128
    ):
        """
        High-level test that checks the model's ability to handle system prompts, RAG, and Tools.
        """
        response = self.model.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            documents=documents,
            tools_schema=tools_schema,
            max_new_tokens=max_new_tokens
        )
        self.results.append({
            'test_name': 'test_generate_response',
            'input': {
                'prompt': prompt,
                'system_prompt': system_prompt,
                'documents': documents,
                'tools_schema': tools_schema
            },
            'output': response,
            'model_name': self.model_name
        })
        return response

    def test_chat(
        self, 
        prompts: List[str], 
        max_new_tokens: int = 128,
        clear_session_between: bool = False
    ):
        """
        Tests multi-turn chat. If clear_session_between is True,
        we create a brand new session for each prompt to see how
        the model responds without context. Otherwise, we maintain context.
        """
        chat_outputs = []
        for i, p in enumerate(prompts):
            if i == 0 or clear_session_between:
                self.model.clear_history()  # reset context
            result = self.model.chat(prompt=p, max_new_tokens=max_new_tokens)
            chat_outputs.append(result)

        self.results.append({
            'test_name': 'test_chat',
            'input': prompts,
            'output': chat_outputs,
            'clear_session_between': clear_session_between,
            'model_name': self.model_name
        })
        return chat_outputs

    def test_stateless_chat(self, chat_history: List[Dict], max_new_tokens: int = 128):
        """
        Tests a stateless chat method. 
        """
        response = self.model.stateless_chat(chat_history=chat_history, max_new_tokens=max_new_tokens)
        self.results.append({
            'test_name': 'test_stateless_chat',
            'input': chat_history,
            'output': response,
            'model_name': self.model_name
        })
        return response

    def test_brainstorm(
        self, 
        messages: List[Dict], 
        role_play_configs: List[Dict],
        max_new_tokens: int = 128,
        ai_assisted_turns: int = 3
    ):
        """
        Tests a role-based brainstorming session.
        """
        response = self.model.brainstrom(
            messages=messages,
            role_play_configs=role_play_configs,
            max_new_tokens=max_new_tokens,
            ai_assisted_turns=ai_assisted_turns
        )
        self.results.append({
            'test_name': 'test_brainstorm',
            'input': {
                'messages': messages,
                'role_play_configs': role_play_configs
            },
            'output': response,
            'model_name': self.model_name
        })
        return response

    def save_results(self, filename: str):
        """
        Saves all test results to a JSON file.
        """
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"[{self.model_name}] Test results saved to {filename}.")
