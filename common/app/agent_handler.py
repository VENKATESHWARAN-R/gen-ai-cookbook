# agent_handler.py
from typing import List, Dict, Optional
import api_client
import json
from db_interface import DatabaseInterface

from datetime import datetime
from config import DB_CONFIG

# Get today's date in desired format
# today_date = datetime.today().strftime("%Y-%m-%d")


# --- Agent Definitions ---
BASE_DB_AGENT_PROMPT = f"""
You're a Data scientist. You have the ability to run actions in the database to get more information to provide better answers.
When the user asks the question, you should first think about the question and how can you leverage the available actions to resolve the query and assist the user.
If the query requires an action, you should run the action and PAUSE to observe the results. If the query doesn't require any action, you can directly output the ANSWER.
You cycle through THOUGHT, ACTION, PAUSE, OBSERVATION. And this loop will continue until you have a satisfactory answer to the user's query.
While responding with final answer include as much information as possible.
You are not allowed to run any action that is not in the list of available actions.

Use following format
THOUGHT: Describe your thoughts about the question you have been asked.
ACTION: run one of the actions available to you - then return PAUSE.
PAUSE - After you return PAUSE, you will receive the result of the action you have run on the next turn as observation.
OBSERVATION: will be the result of running those actions.

IF theobservation retruns error or empty, respond to user that the tools are broken
"""

DB_AGENT_TOOLS_SCHEMA = """You are provided with the following tools (functions). Use them only when necessary and format your action in a special tag for easy parsing.

---

Each tool follows this JSON schema:
[
  {
    "name": "get_user_count",
    "description": "Get the total number of registered users.",
    "parameters": {
      "type": "object",
      "properties": {},
      "required": []
    }
  },
  {
    "name": "get_order_status_count",
    "description": "Get the count of orders grouped by their processing status.",
    "parameters": {
      "type": "object",
      "properties": {},
      "required": []
    }
  },
  {
    "name": "get_user_details",
    "description": "Fetch user details using either their ID or email.",
    "parameters": {
      "type": "object",
      "properties": {
        "user_id": { "type": "integer", "description": "User ID" },
        "email": { "type": "string", "description": "User email" }
      },
      "required": []
    }
  },
  {
    "name": "get_users_by_creation_date",
    "description": "Get users created within a date range.",
    "parameters": {
      "type": "object",
      "properties": {
        "start_date": { "type": "string" },
        "end_date": { "type": "string" }
      },
      "required": ["start_date", "end_date"]
    }
  },
  {
    "name": "get_orders_by_timeline",
    "description": "Retrieve purchased items between two dates.",
    "parameters": {
      "type": "object",
      "properties": {
        "start_date": { "type": "string" },
        "end_date": { "type": "string" }
      },
      "required": ["start_date", "end_date"]
    }
  },
  {
    "name": "get_total_bill_for_month",
    "description": "Get total bill amount for a specified month.",
    "parameters": {
      "type": "object",
      "properties": {
        "year": { "type": "integer" },
        "month": { "type": "integer" }
      },
      "required": ["year", "month"]
    }
  },
  {
    "name": "get_user_total_spending",
    "description": "Retrieve total paid amount for a specific user.",
    "parameters": {
      "type": "object",
      "properties": {
        "user_id": { "type": "integer" },
        "email": { "type": "string" }
      },
      "required": []
    }
  }
]

---

Use the following loop format:

THOUGHT: Reflect on the user's query.
ACTION: Wrap the tool invocation inside <tool_call> XML tags.
PAUSE
OBSERVATION: You'll be given the output of the tool you invoked.
Repeat the loop until you're ready to respond.
ANSWER: Provide your final answer to the user.

The action should look like:
<tool_call>
{"name": "get_total_bill_for_month", "arguments": {"year": 2025, "month": 2}}
</tool_call>

---

"""

DB_AGENT_EXAMPLE_PROMPT = """
### Example Session

QUESTION: How much was spent last month?

THOUGHT: I should fetch the total bill for February 2025.
ACTION:
<tool_call>
{"name": "get_total_bill_for_month", "arguments": {"year": 2025, "month": 2}}
</tool_call>
PAUSE

OBSERVATION: 8450.75

THOUGHT: I now know the spending for February 2025. No further steps are needed.
ANSWER: The total spending for February 2025 was 8450.75.

---

Begin your reasoning loop when the user provides a query.
"""

db_object = DatabaseInterface(db_config=DB_CONFIG)

# creating a dictionary to store the tools
tools_dict = {
    "get_user_count": db_object.get_user_count,
    "get_order_status_count": db_object.get_order_status_count,
    "get_user_details": db_object.get_user_details,
    "get_users_by_creation_date": db_object.get_users_by_creation_date,
    "get_orders_by_timeline": db_object.get_orders_by_timeline,
    "get_total_bill_for_month": db_object.get_total_bill_for_month,
    "get_user_total_spending": db_object.get_user_total_spending,
}

# --- Agent Interaction Logic ---


def handle_db_agent_query(prompt: str, chat_history: List[Dict]) -> Optional[Dict]:
    """
    Sends a prompt to the backend's generate_response endpoint configured for the DB Agent.
    Assumes the backend /generate_response handles the full tool interaction loop internally.
    """
    print(f"Handling DB Agent query: {prompt}")
    # Prepare DB_AGENT system prompt
    DATETIME_PROMPT = (
        "\nCurrent datetime is: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n"
    )
    DB_AGENT_SYSTEM_PROMPT = (
        f"{BASE_DB_AGENT_PROMPT}{DATETIME_PROMPT}{DB_AGENT_TOOLS_SCHEMA}{DB_AGENT_EXAMPLE_PROMPT}"
    )
    response = api_client.call_generate_response(
        prompt=prompt,
        system_prompt=DB_AGENT_SYSTEM_PROMPT,
        kwargs={"stop_strings": "PAUSE"},
        #chat_history=None,
        # Add other relevant params like max_new_tokens if desired
    )
    # Extract the tool call from the response
    model_response = response.get("response", "")
    print(f"Model response: {model_response}")
    model_chat_history = response.get("chat_history", [])
    # Check if the response chat history contains the system prompt add if it doesn't
    if not any(
        msg.get("role") == "system" and msg.get("content") == DB_AGENT_SYSTEM_PROMPT
        for msg in model_chat_history
    ):
        model_chat_history.insert(
            0,
            {
                "role": "system",
                "content": DB_AGENT_SYSTEM_PROMPT,
            },
        )
    # Loop the response back to the model if it contains a tool call
    while "<tool_call>" in model_response:
        # Extract the tool call from the model response
        start_index = model_response.index("<tool_call>") + len("<tool_call>")
        end_index = model_response.index("</tool_call>")
        tool_call_str = model_response[start_index:end_index]
        # Parse the tool call string into a dictionary
        tool_call_dict = json.loads(tool_call_str)
        # Extract the tool name and arguments
        tool_name = tool_call_dict["name"]
        arguments = tool_call_dict["arguments"]
        # Call the appropriate function from tools_dict
        if tool_name in tools_dict:
            result = tools_dict[tool_name](**arguments)
            print(f"Tool '{tool_name}' called with arguments {arguments} returned: {result}")
            response["result"] = result
            # Loop again if there's another tool call
            response = api_client.call_generate_response(
                prompt=f"OBSERVATION: {result}",
                chat_history=model_chat_history,
            )
            model_response = response.get("response", "")
            print(f"Model response after tool call: {model_response}")
            model_chat_history = response.get("chat_history", [])
        else:
            # If the tool name is not recognized, break the loop
            break
    # Final response
    response["response"] = model_response.split("ANSWER:")[-1].strip()
    return response


def handle_rag_agent_query(
    prompt: str, chat_history: List[Dict], documents: Optional[List[Dict]] = None
) -> Optional[Dict]:
    """
    Sends a prompt to the backend's generate_response endpoint configured for the RAG Agent.
    Optionally includes documents for context.
    """
    print(f"Handling RAG Agent query: {prompt}")
    # Here, the 'documents' parameter is key for RAG
    # The actual retrieval step (querying ChromaDB) should ideally happen *before* calling this,
    # potentially orchestrated by the main app or triggered by the user providing context.
    # For simplicity now, we assume 'documents' might be passed in if available.
    response = api_client.call_generate_response(
        prompt=prompt,
        system_prompt=RAG_AGENT_SYSTEM_PROMPT,
        chat_history=chat_history,
        documents=documents,  # Pass retrieved documents here
        tools_schema=RAG_TOOLS_SCHEMA_FOR_API,  # Likely None for RAG unless it has specific tools
    )
    return response
