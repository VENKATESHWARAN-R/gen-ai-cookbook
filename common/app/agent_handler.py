# agent_handler.py
from typing import List, Dict, Optional
import api_client
import json # Needed if you construct complex tools_schema

# --- Agent Definitions ---

DB_AGENT_SYSTEM_PROMPT = """
You are a helpful assistant specialized in answering questions about users, orders, and bills based on a connected database.
Use the available tools to query the database and provide accurate information.
If you cannot answer the question using the tools, state that clearly.
"""

RAG_AGENT_SYSTEM_PROMPT = """
You are a helpful assistant specialized in answering questions based on the provided document context.
Answer the user's query solely based on the information found in the documents.
If the answer is not found in the documents, state that you don't have the information in the provided context.
Cite the document IDs if possible.
"""

# --- Tool Schema Definitions (Example - Adapt based on backend expectations) ---
# This part is crucial and depends HEAVILY on how your backend's
# /generate_response endpoint expects tool information.

# Option 1: Backend expects a predefined string identifier for toolsets
# DB_TOOLS_SCHEMA_ID = "database_tools_v1"
# RAG_TOOLS_SCHEMA_ID = "rag_retrieval_tool_v1"

# Option 2: Backend expects a JSON string representing the tools schema
# (Similar to OpenAI functions - requires knowing the *exact* functions available to the backend agent)
# This would ideally be dynamically fetched or configured.
# Example (replace with actual tool definitions exposed by your backend for the DB agent):
# db_tools = [
#     {
#         "type": "function",
#         "function": {
#             "name": "get_user_count",
#             "description": "Fetch the total number of users.",
#             "parameters": {"type": "object", "properties": {}} # No parameters
#         }
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "get_orders_by_timeline",
#             "description": "Retrieve a summary of items purchased within a specific time range.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
#                     "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)"}
#                 },
#                 "required": ["start_date", "end_date"]
#             }
#         }
#     },
#     # ... add ALL other functions from db_interface.py here ...
# ]
# DB_TOOLS_SCHEMA_STR = json.dumps(db_tools)

# For this implementation, we'll assume Option 1 (or that the backend infers tools from the system prompt)
# If you need Option 2, you'll need to build the schema string `DB_TOOLS_SCHEMA_STR`
DB_TOOLS_SCHEMA_FOR_API = None # Set this if your backend needs explicit schema (e.g., DB_TOOLS_SCHEMA_STR or DB_TOOLS_SCHEMA_ID)
RAG_TOOLS_SCHEMA_FOR_API = None # RAG might not use explicit 'tools' but rely on 'documents'


# --- Agent Interaction Logic ---

def handle_db_agent_query(prompt: str, chat_history: List[Dict]) -> Optional[Dict]:
    """
    Sends a prompt to the backend's generate_response endpoint configured for the DB Agent.
    Assumes the backend /generate_response handles the full tool interaction loop internally.
    """
    print(f"Handling DB Agent query: {prompt}")
    # NOTE: This assumes the backend's /generate_response endpoint is configured
    # to use the database tools when it sees the DB_AGENT_SYSTEM_PROMPT or a specific tools_schema.
    # If the backend expects the frontend to manage the loop, this function becomes much more complex.
    response = api_client.call_generate_response(
        prompt=prompt,
        system_prompt=DB_AGENT_SYSTEM_PROMPT,
        chat_history=chat_history,
        tools_schema=DB_TOOLS_SCHEMA_FOR_API # Pass schema if needed by backend
        # Add other relevant params like max_new_tokens if desired
    )
    return response

def handle_rag_agent_query(prompt: str, chat_history: List[Dict], documents: Optional[List[Dict]] = None) -> Optional[Dict]:
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
        documents=documents, # Pass retrieved documents here
        tools_schema=RAG_TOOLS_SCHEMA_FOR_API # Likely None for RAG unless it has specific tools
    )
    return response