# Description: Configuration file for the data store service.
import os
from dotenv import load_dotenv
from typing import Literal

load_dotenv()

# --- AWS Configuration (Optional) ---
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# --- Local LLM Configuration ---
LLM_ENDPOINT_URL = os.getenv(
    "LOCAL_LLM_URL", "http://localhost:8000/api/llm/generate_response"
)
LLM_HEADERS = {
    "accept": "application/json",
    "Content-Type": "application/json",
}
LLM_TIMEOUT = 300  # seconds
LLM_TARGET_CHUNK_SIZE = 125  # Default size in tokens
LLM_SIZE_UNIT: Literal["tokens", "characters"] = "tokens"
LLM_CHUNK_SEPARATOR: str = "\n---CHUNK_SEPARATOR---\n"
LLM_ADD_CONTEXT: bool = False  # Flag to enable context summarization via LLM
# Define the expected request/response format for your local LLM if needed

# --- ChromaDB Configuration ---
CHROMA_DB_HOST = os.getenv("CHROMA_DB_HOST", "localhost")
CHROMA_DB_PORT = int(os.getenv("CHROMA_DB_PORT", "8001"))
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "my_documents")

# --- Embedding Configuration ---
# Consider using SentenceTransformerEmbedding from langchain_community if preferred
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2"
)

# --- Processing Configuration ---
FALLBACK_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "650"))
FALLBACK_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# --- contextual prompt template ---
CONTEXTUAL_PROMPT_TEMPLATE = """Given the following document:
<document>
{full_doc_content}
</document>

And this specific chunk from the document:
<chunk>
{chunk_content}
</chunk>

Provide a short, succinct context (1-2 sentences) that helps understand where this chunk fits within the larger document. This context will be used to improve search retrieval for the chunk. Focus on the chunk's relationship to surrounding information or its main topic within the document's scope. Respond *only* with the context itself, nothing else.
"""

# --- Other ---
# Temporary directory for downloading S3 files if needed
TEMP_DIR = "/tmp"
