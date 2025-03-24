# run_tests.py

import sys
import os
import time
import argparse

from dotenv import load_dotenv, find_dotenv
from rich.progress import Progress
from rich.table import Table
from rich.console import Console

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Local imports
from ai_core.ai_model import QwenLocal, GemmaLocal, LlamaLocal, PhiLocal, GeminiApi
from ai_core.test.llm_tester import LLMTester

# Test configs
from ai_core.test.test_configs import (
    SYSTEM_PROMPT, USER_PROMPT1, USER_PROMPT2,
    RAG_PROMPT, TOOL_PROMPT, CHAT_PROMPT1, CHAT_PROMPT2, CHAT_PROMPT3,
    TOOL_PROMPTS, documents, tools_schema,
    role_play_configs, messages
)

console = Console()

# A registry mapping model “keys” to the classes (and any common/default arguments).
MODEL_REGISTRY = {
    "qwen":    QwenLocal,
    "gemma":   GemmaLocal,
    "llama":   LlamaLocal,
    "phi":     PhiLocal,
    "gemini":  GeminiApi,
}

def load_model(model_name: str, model_id: str = None):
    """
    Loads and returns the requested model. 
    If model_id is provided, pass it into the constructor (for local LLMs).
    If no model_id is given (or not applicable), load the default model or use an API key as needed.
    """
    model_class = MODEL_REGISTRY.get(model_name.lower())
    if not model_class:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(MODEL_REGISTRY.keys())}")

    if model_name.lower() == "gemini":
        # For Gemini, we rely on the environment variable for the API key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in environment.")
        # Gemini doesn't usually need a model_id, but if you have a version:
        if model_id:
            console.print(f"[bold yellow]Note:[/bold yellow] Gemini API ignoring custom model_id={model_id}")
        return model_class(GOOGLE_API_KEY=api_key)

    else:
        # For local models (GemmaLocal, LlamaLocal, QwenLocal, etc.)
        if model_id:
            return model_class(model_id)
        else:
            # If no model_id was provided, use the default constructor
            return model_class()

def main():
    load_dotenv(find_dotenv())

    # 1) Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run LLM tests for a single model.")
    parser.add_argument(
        "--model",
        required=True,
        help="Name of the model to test. e.g. qwen, gemma, llama, phi, gemini"
    )
    parser.add_argument(
        "--model_id",
        default=None,
        help="(Optional) Model ID or path/checkpoint. If omitted, default is used."
    )
    args = parser.parse_args()

    # 2) Load model
    console.print(f"[bold green]Loading model:[/bold green] {args.model}")
    model = load_model(args.model, args.model_id)

    # 3) Create LLMTester
    tester = LLMTester(model=model, model_name=args.model)

    # We'll store durations for each test in a dictionary
    test_durations = {}

    # We have a known list of tests to run:
    tests_to_run = [
        "generate_text",
        "generate_response_sys",
        "generate_response_rag",
        "generate_response_tool",
        "chat",
        "tool_prompts_loop",
        "stateless_chat",
        "brainstorm",
    ]

    # Calculate how many sub-tests we really have. 
    # Example: we'll count each item in the `TOOL_PROMPTS` loop individually as well.
    total_sub_tests = (
        len(tests_to_run) 
        + len(TOOL_PROMPTS)  # each tool prompt tested individually
        -1
    )

    # 4) Run the tests with a progress bar
    with Progress(console=console) as progress:
        test_task = progress.add_task("Running LLM tests...", total=total_sub_tests)

        # (A) Test raw text generation
        start_time = time.time()
        prompt_templates = tester.model.get_templates()
        prompt1 = prompt_templates['non_sys_prompt_template'].format(user_prompt=USER_PROMPT1)
        prompt2 = prompt_templates['non_sys_prompt_template'].format(user_prompt=USER_PROMPT2)
        prompt3 = prompt_templates['default_prompt_template'].format(user_prompt=USER_PROMPT1, system_prompt=SYSTEM_PROMPT)
        prompt4 = prompt_templates['default_prompt_template'].format(user_prompt=USER_PROMPT1, system_prompt=SYSTEM_PROMPT)
        prompts = [prompt1, prompt2, prompt3, prompt4]
        _ = tester.test_generate_text(prompts=prompts, max_new_tokens=128)
        end_time = time.time()
        test_durations["generate_text"] = end_time - start_time
        progress.update(test_task, advance=1)

        # (B) Test generate_response with system prompt
        start_time = time.time()
        _ = tester.test_generate_response(prompt=USER_PROMPT1, system_prompt=SYSTEM_PROMPT)
        end_time = time.time()
        test_durations["generate_response_sys"] = end_time - start_time
        progress.update(test_task, advance=1)

        # (C) Test generate_response with RAG
        start_time = time.time()
        _ = tester.test_generate_response(prompt=RAG_PROMPT, documents=documents)
        end_time = time.time()
        test_durations["generate_response_rag"] = end_time - start_time
        progress.update(test_task, advance=1)

        # (D) Test generate_response with tools
        start_time = time.time()
        _ = tester.test_generate_response(prompt=TOOL_PROMPT, tools_schema=tools_schema)
        end_time = time.time()
        test_durations["generate_response_tool"] = end_time - start_time
        progress.update(test_task, advance=1)

        # (E) Test chat
        start_time = time.time()
        chat_prompts = [CHAT_PROMPT1, CHAT_PROMPT2, CHAT_PROMPT3]
        _ = tester.test_chat(chat_prompts, max_new_tokens=128)
        end_time = time.time()
        test_durations["chat"] = end_time - start_time
        progress.update(test_task, advance=1)

        # (F) Test each of the tool prompts separately
        start_time = time.time()
        for tp in TOOL_PROMPTS:
            _ = tester.test_generate_response(prompt=tp, tools_schema=tools_schema, max_new_tokens=256)
            progress.update(test_task, advance=1)
        end_time = time.time()
        test_durations["tool_prompts_loop"] = end_time - start_time

        # (G) Test stateless chat
        start_time = time.time()
        chat_history = [
            {"role": "user", "content": "Hi, can you remember me?"}
        ]
        _ = tester.test_stateless_chat(chat_history, max_new_tokens=128)
        end_time = time.time()
        test_durations["stateless_chat"] = end_time - start_time
        progress.update(test_task, advance=1)

        # (H) Test brainstorming (role-based multi-turn)
        start_time = time.time()
        _ = tester.test_brainstorm(
            messages=messages,
            role_play_configs=role_play_configs,
            max_new_tokens=128,
            ai_assisted_turns=3
        )
        end_time = time.time()
        test_durations["brainstorm"] = end_time - start_time
        progress.update(test_task, advance=1)

    # 5) Save all results to JSON
    results_filename = f"./test_results/{args.model}_test_results.json"
    tester.save_results(results_filename)

    # 6) Print summary of timings
    total_time = sum(test_durations.values())

    console.print(f"\n[bold cyan]Summary of Test Durations for {args.model}[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Test Name")
    table.add_column("Duration (s)", justify="right")

    for test_name, duration in test_durations.items():
        table.add_row(test_name, f"{duration:.2f}")

    table.add_row("TOTAL", f"{total_time:.2f}")
    console.print(table)


if __name__ == "__main__":
    main()
