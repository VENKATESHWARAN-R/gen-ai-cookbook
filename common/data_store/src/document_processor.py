"""
Document Processor
"""

import json  # Added for LLM response parsing
import logging
import os
from typing import Any, Dict, List, Literal, Optional, Union  # Added Dict, Any, Literal

import boto3
import requests  # Added for LLM calls
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document

import config as config


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- Custom Exception ---
class LLMError(Exception):
    """Custom exception for LLM API errors."""

    pass


# --- Combined Class ---
class RAGDocumentProcessor:
    """
    Processes documents for RAG applications, handling loading from various sources
    and chunking using either semantic (LLM-based) or recursive methods.
    """

    # --- Class Variables ---
    CONTEXTUAL_PROMPT_TEMPLATE = (
        "Given the following document: "
        "<document>\n{full_doc_content}\n</document>"
        "\n\nAnd this specific chunk from the document: "
        "<chunk>\n{chunk_content}\n</chunk>"
        "\n\nProvide a short, succinct context (1-2 sentences) that helps understand where this chunk fits within the larger document. "
        "This context will be used to improve search retrieval for the chunk. "
        "Focus on the chunk's relationship to surrounding information or its main topic within the document's scope. "
        "Respond *only* with the context itself, nothing else."
    )

    LLM_SPLIT_SYSTEM_PROMPT_TEMPLATE = (
        "You are an expert text analyzer. Your task is to segment the provided text "
        "into semantically coherent chunks. Each chunk should represent a distinct idea or topic. "
        "Aim for chunks around {target_size} {size_unit} long, but prioritize semantic boundaries "
        "over strict size adherence. Output *only* the chunks separated by the delimiter '{separator}'. "
        "Try to stick with the document's original structure and meaning. "
        "Sometimes the document may contain tabular data but the structure might be lost in the text. "
        "In such cases, please try to keep the table data as much as possible and add context about the table and table data in a single chunk. "
        "The document may contain technical jargon or specialized terms. "
        "If you encounter any such terms, please try to keep them in the same chunk. "
    )

    def __init__(
        self,
        # Loading Config
        aws_access_key: Optional[str] = config.AWS_ACCESS_KEY_ID,
        aws_secret_key: Optional[str] = config.AWS_SECRET_ACCESS_KEY,
        aws_region: str = config.AWS_REGION,
        # Recursive Chunking Config (also used as fallback)
        recursive_chunk_size: int = config.FALLBACK_CHUNK_SIZE,
        recursive_chunk_overlap: int = config.FALLBACK_CHUNK_OVERLAP,
        # Semantic Chunking Config (Optional)
        use_semantic_chunking: bool = True,  # Master switch for semantic chunking
        llm_endpoint_url: Optional[str] = config.LLM_ENDPOINT_URL,
        llm_headers: Optional[Dict[str, str]] = config.LLM_HEADERS,
        llm_target_chunk_size: int = config.LLM_TARGET_CHUNK_SIZE,
        llm_size_unit: Literal["tokens", "characters"] = config.LLM_SIZE_UNIT,
        llm_chunk_separator: str = config.LLM_CHUNK_SEPARATOR,
        llm_request_timeout: int = config.LLM_TIMEOUT,
        llm_add_context: bool = config.LLM_ADD_CONTEXT,
    ):
        """Initialize the RAG Document Processor."""
        self.aws_region = aws_region
        self.temp_dir = config.TEMP_DIR  # Store temp dir path

        # --- Initialize S3 Client (Optional) ---
        self.s3_client = None
        if aws_access_key and aws_secret_key:
            try:
                self.s3_client = boto3.client(
                    "s3",
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key,
                    region_name=self.aws_region,
                )
                logger.info("S3 client initialized for region %s.", self.aws_region)
            except Exception as e:
                logger.error("Failed to initialize S3 client: %s", e)
        else:
            logger.warning("AWS credentials not provided. S3 functionality disabled.")

        # --- Initialize Recursive Splitter (Always needed for fallback or primary use) ---
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=recursive_chunk_size,
            chunk_overlap=recursive_chunk_overlap,
            separators=["\n\n", "\n", " ", ".", ""],  # Sensible defaults
            is_separator_regex=False,
        )
        logger.info(
            "Initialized RecursiveCharacterTextSplitter (size=%d, overlap=%d).",
            recursive_chunk_size,
            recursive_chunk_overlap,
        )

        # --- Configure Semantic Chunking (if enabled and configured) ---
        self.use_semantic_chunking = use_semantic_chunking
        self.llm_endpoint_url = llm_endpoint_url
        self.llm_headers = llm_headers.copy() if llm_headers else {}  # Copy headers
        self.llm_target_chunk_size = llm_target_chunk_size
        self.llm_size_unit = llm_size_unit
        self.llm_chunk_separator = llm_chunk_separator
        self.llm_request_timeout = llm_request_timeout
        self.llm_add_context = llm_add_context

        # Validate semantic chunking setup
        if self.use_semantic_chunking and not (
            self.llm_endpoint_url and self.llm_headers
        ):
            logger.warning(
                "Semantic chunking enabled, but LLM endpoint URL or headers are missing. Semantic chunking will be disabled."
            )
            self.use_semantic_chunking = False
        elif self.use_semantic_chunking:
            logger.info(
                "Semantic chunking enabled. LLM Endpoint: %s, Target Size: %d %s, Add Context: %s",
                self.llm_endpoint_url,
                self.llm_target_chunk_size,
                self.llm_size_unit,
                self.llm_add_context,
            )
        else:
            logger.info("Semantic chunking disabled. Using recursive splitting only.")

        # --- Define Supported Loaders ---
        self.loaders_map = {
            ".pdf": PyMuPDFLoader,
            ".docx": Docx2txtLoader,
            ".csv": CSVLoader,
            ".txt": TextLoader,
            ".md": UnstructuredMarkdownLoader,
        }

    # --- Loading Methods (from DocumentProcessor) ---

    def _load_documents_from_loader(self, loader) -> List[Document]:
        """Helper function to load documents using a LangChain loader."""
        try:
            return loader.load()
        except Exception as e:
            source = getattr(
                loader, "file_path", getattr(loader, "web_path", "Unknown Source")
            )
            logger.error(
                "Error loading document %s: %s", source, e, exc_info=True
            )  # Add exc_info
            return []

    def _process_single_file(
        self, file_path: str, original_source: Optional[str] = None
    ) -> List[Document]:
        """Auto-detect file type and process it using the appropriate loader."""
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        loader_class = self.loaders_map.get(ext)

        if loader_class:
            logger.info(
                "Processing file: %s using %s", file_path, loader_class.__name__
            )
            try:
                # Handle specific loader arguments if needed (e.g., CSV encoding)
                if ext == ".csv":
                    # Add encoding='utf-8' or other specific args if necessary
                    loader = loader_class(file_path, encoding="utf-8")
                else:
                    loader = loader_class(file_path)

                docs = self._load_documents_from_loader(loader)

                # Override source metadata if original_source (e.g., S3 URI) is provided
                if original_source:
                    for doc in docs:
                        # Ensure metadata exists
                        if not hasattr(doc, "metadata") or doc.metadata is None:
                            doc.metadata = {}
                        doc.metadata["source"] = original_source
                elif docs and "source" not in docs[0].metadata:  # Add source if missing
                    for doc in docs:
                        if not hasattr(doc, "metadata") or doc.metadata is None:
                            doc.metadata = {}
                        doc.metadata["source"] = file_path  # Default to local path

                return docs
            except Exception as e:
                logger.error(
                    "Failed during processing or loading of file %s: %s",
                    file_path,
                    e,
                    exc_info=True,
                )
                return []
        else:
            logger.warning("Unsupported file format: %s. Skipping.", file_path)
            return []

    def load_from_source(
        self,
        source: Union[str, List[str]],
        source_type: str = "file",
        recursive: bool = True,
    ) -> List[Document]:
        """
        Loads documents from various sources like files, directories, S3, or web pages.

        Args:
            source: Path(s), URL(s), or S3 URI(s) (e.g., "s3://bucket-name/key").
            source_type: 'file', 'directory', 's3', or 'web'.
            recursive: Whether to search directories recursively.

        Returns:
            A list of LangChain Document objects.
        """
        sources = [source] if isinstance(source, str) else source
        all_docs = []

        if source_type == "file":
            for file_path in sources:
                if os.path.isfile(file_path):
                    all_docs.extend(self._process_single_file(file_path))
                else:
                    logger.warning("File not found: %s", file_path)

        elif source_type == "directory":
            for dir_path in sources:
                if not os.path.isdir(dir_path):
                    logger.warning("Directory not found: %s", dir_path)
                    continue
                logger.info(
                    "Scanning directory: %s (recursive=%s)", dir_path, recursive
                )
                for root, _, files in os.walk(dir_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # Check if it's actually a file before processing
                        if os.path.isfile(file_path):
                            all_docs.extend(self._process_single_file(file_path))
                        else:
                            logger.debug(
                                "Skipping non-file item in directory: %s", file_path
                            )
                    if not recursive:
                        break  # Stop after the top-level directory

        elif source_type == "s3":
            if not self.s3_client:
                logger.error("S3 client not initialized. Cannot load from S3.")
                return []

            os.makedirs(self.temp_dir, exist_ok=True)  # Ensure temp dir exists

            for s3_uri in sources:
                try:
                    if not s3_uri.startswith("s3://"):
                        logger.warning(
                            "S3 source '%s' does not start with s3://. Skipping.",
                            s3_uri,
                        )
                        continue

                    s3_path = s3_uri[5:]
                    bucket, key = s3_path.split("/", 1)
                    if not key:  # Check if key is empty (e.g., s3://bucket-name/)
                        logger.warning(
                            "Invalid S3 path (missing key): %s. Skipping.", s3_uri
                        )
                        continue

                    # Create a safe local filename (replace potential path separators in key)
                    safe_filename = key.replace("/", "_")
                    local_file = os.path.join(self.temp_dir, safe_filename)

                    logger.info("Downloading s3://%s/%s to %s", bucket, key, local_file)
                    self.s3_client.download_file(bucket, key, local_file)

                    # Process the downloaded file, passing the original S3 URI as source
                    docs = self._process_single_file(local_file, original_source=s3_uri)
                    all_docs.extend(docs)

                    # Clean up the downloaded file
                    try:
                        os.remove(local_file)
                        logger.debug("Removed temporary file: %s", local_file)
                    except OSError as e:
                        logger.warning(
                            "Failed to remove temporary file %s: %s", local_file, e
                        )

                except Exception as e:
                    logger.error(
                        "Failed to process S3 source %s: %s", s3_uri, e, exc_info=True
                    )
                    # Attempt cleanup even on error
                    if "local_file" in locals() and os.path.exists(local_file):
                        try:
                            os.remove(local_file)
                        except OSError:
                            pass  # Ignore cleanup error if main processing failed

        elif source_type == "web":
            for url in sources:
                logger.info("Loading web page: %s", url)
                try:
                    # Configure WebBaseLoader - consider adding options like bs_get_text_kwargs
                    # header_template = {'User-Agent': 'Mozilla/5.0'} # Example header
                    loader = WebBaseLoader(web_path=url)  # Pass url as web_path
                    docs = self._load_documents_from_loader(loader)
                    # Ensure source metadata is set correctly for web pages
                    for doc in docs:
                        if not hasattr(doc, "metadata") or doc.metadata is None:
                            doc.metadata = {}
                        if "source" not in doc.metadata:
                            doc.metadata["source"] = url
                    all_docs.extend(docs)
                except Exception as e:
                    logger.error(
                        "Failed to load web page %s: %s", url, e, exc_info=True
                    )

        else:
            logger.error(
                "Unsupported source_type: %s. Must be one of 'file', 'directory', 's3', 'web'.",
                source_type,
            )

        logger.info("Loaded %d documents initially from source(s).", len(all_docs))
        return all_docs

    # --- Chunking Methods (incorporating SemanticSplitter logic) ---

    def _call_llm_api(self, payload: Dict[str, Any]) -> str:
        """Helper function to call the configured LLM API endpoint."""
        if not self.llm_endpoint_url:
            raise LLMError("LLM endpoint URL is not configured.")

        try:
            response = requests.post(
                self.llm_endpoint_url,
                headers=self.llm_headers,
                json=payload,
                timeout=self.llm_request_timeout,
            )
            response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)

            response_data = response.json()
            # Adapt based on your actual LLM API response structure
            if "response" in response_data:
                generated_text = response_data["response"]
            elif "generated_text" in response_data:
                generated_text = response_data["generated_text"]
            elif isinstance(response_data, str):  # Handle plain string response
                generated_text = response_data
            else:
                # Try to find a string value if the structure is unknown
                potential_text = [
                    v for v in response_data.values() if isinstance(v, str)
                ]
                if potential_text:
                    generated_text = potential_text[0]  # Take the first string found
                    logger.warning(
                        "LLM response structure unexpected. Using first string value found: %s...",
                        generated_text[:100],
                    )
                else:
                    raise LLMError(
                        f"Could not extract generated text from LLM response: {response_data}"
                    )

            if not isinstance(generated_text, str):
                raise LLMError(
                    f"LLM response content is not a string: {type(generated_text)}"
                )

            return generated_text.strip()

        except requests.exceptions.Timeout as e:
            logger.error(
                "LLM API call timed out after %s seconds.", self.llm_request_timeout
            )
            raise LLMError(
                f"LLM API call timed out ({self.llm_request_timeout}s)."
            ) from e
        except requests.exceptions.RequestException as e:
            logger.error("LLM API request failed: %s", e)
            raise LLMError(f"LLM API request failed: {e}") from e
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(
                "Failed to parse LLM JSON response or access expected key: %s", e
            )
            raise LLMError(f"Failed to parse LLM JSON response: {e}") from e
        except LLMError:  # Re-raise specific LLMError
            raise
        except Exception as e:  # Catch unexpected errors during API call
            logger.error("Unexpected error during LLM API call: %s", e, exc_info=True)
            raise LLMError(f"Unexpected error during LLM API call: {e}") from e

    def _split_with_llm(self, doc: Document) -> List[Document]:
        """Attempts to split a single document using the LLM."""
        source_info = f"source: {doc.metadata.get('source', 'N/A')}, page: {doc.metadata.get('page', 'N/A')}"
        logger.info("Attempting LLM semantic splitting for %s", source_info)
        page_content = doc.page_content
        if not page_content or page_content.isspace():
            logger.warning("Skipping empty page content for %s", source_info)
            return []

        system_prompt = self.LLM_SPLIT_SYSTEM_PROMPT_TEMPLATE.format(
            target_size=self.llm_target_chunk_size,
            size_unit=self.llm_size_unit,
            separator=self.llm_chunk_separator,
        )

        payload = {
            "prompt": page_content,
            "system_prompt": system_prompt,
            "kwargs": {},  # Add any specific LLM params here if needed (e.g., temperature, max_tokens)
            # Example: "kwargs": {"max_tokens": 2048, "temperature": 0.1}
        }

        try:
            generated_text = self._call_llm_api(payload)

            if not generated_text or self.llm_chunk_separator not in generated_text:
                logger.warning(
                    "LLM output validation failed for %s. Output did not contain separator '%s' or was empty. LLM Output: '%s...'",
                    source_info,
                    self.llm_chunk_separator,
                    generated_text[:200],
                )
                raise ValueError(
                    "Invalid LLM response format (separator missing or empty)."
                )

            raw_chunks = generated_text.split(self.llm_chunk_separator)
            raw_chunks = [
                chunk.strip() for chunk in raw_chunks if chunk.strip()
            ]  # Filter empty

            if not raw_chunks:
                logger.warning(
                    "LLM output resulted in zero valid chunks after splitting for %s.",
                    source_info,
                )
                raise ValueError("LLM output resulted in zero valid chunks.")

            logger.info(
                "LLM successfully generated %d raw chunks for %s.",
                len(raw_chunks),
                source_info,
            )
            return self._post_process_chunks(
                raw_chunks, doc, split_method="semantic_llm"
            )

        except (LLMError, ValueError) as e:
            # Logged sufficiently in _call_llm_api or above
            # Re-raise to be caught by the calling method for fallback
            raise e
        except Exception as e:  # Catch unexpected errors during LLM splitting logic
            logger.error(
                "Unexpected error during LLM splitting logic for %s: %s",
                source_info,
                e,
                exc_info=True,
            )
            raise LLMError(
                f"Unexpected error during LLM splitting: {e}"
            ) from e  # Wrap as LLMError to trigger fallback

    def _split_with_recursive(self, doc: Document) -> List[Document]:
        """Splits a single document using the configured RecursiveCharacterTextSplitter."""
        source_info = f"source: {doc.metadata.get('source', 'N/A')}, page: {doc.metadata.get('page', 'N/A')}"
        logger.info("Using recursive splitting for %s", source_info)

        try:
            recursive_chunks_text = self.recursive_splitter.split_text(doc.page_content)
        except Exception as e:
            logger.error(
                "Recursive splitting failed for %s: %s", source_info, e, exc_info=True
            )
            return []  # Return empty list on splitter failure

        # Post-process to create Document objects with metadata
        processed_chunks = self._post_process_chunks(
            recursive_chunks_text, doc, split_method="recursive"
        )

        logger.info(
            "Recursive splitter created %d chunks for %s.",
            len(processed_chunks),
            source_info,
        )
        return processed_chunks

    def _summarize_chunk(self, chunk_text: str, doc_content: str) -> Optional[str]:
        """(Optional) Generates a contextual summary for a chunk using the LLM."""
        if not self.llm_add_context:  # Check if context addition is enabled
            return None
        if not chunk_text or chunk_text.isspace():
            return None
        if not self.llm_endpoint_url:  # Check if LLM is configured
            logger.warning(
                "Cannot generate chunk context summary: LLM endpoint not configured."
            )
            return None

        logger.debug("Requesting context summary for chunk: %s...", chunk_text[:100])

        # Use the dedicated context prompt template
        summary_prompt = self.CONTEXTUAL_PROMPT_TEMPLATE.format(
            full_doc_content=doc_content, chunk_content=chunk_text
        )
        payload = {
            "prompt": summary_prompt,
            "kwargs": {},  # Add LLM parameters if needed (e.g., max_tokens for summary)
            # Example: "kwargs": {"max_tokens": 100}
        }

        try:
            summary = self._call_llm_api(payload)
            logger.debug("Received context summary: %s", summary)
            # Basic validation: check if it's just whitespace or looks like an error message
            if not summary or summary.isspace() or len(summary) < 10:
                logger.warning(
                    "Received potentially invalid summary: '%s'. Skipping context addition for this chunk.",
                    summary,
                )
                return None
            return summary
        except LLMError as e:
            logger.warning(
                "Failed to generate context summary for chunk: %s. Skipping context addition.",
                e,
            )
            return None  # Return None on summary failure

    def _post_process_chunks(
        self, raw_chunks: List[str], original_doc: Document, split_method: str
    ) -> List[Document]:
        """Formats raw text chunks into Document objects with metadata and optional context."""
        processed_chunks = []
        source_metadata = (
            original_doc.metadata.copy()
        )  # Start with original doc metadata

        full_doc_content_for_summary = (
            original_doc.page_content if self.llm_add_context else ""
        )

        for i, chunk_text in enumerate(raw_chunks):
            if not chunk_text or chunk_text.isspace():
                logger.debug("Skipping empty chunk during post-processing.")
                continue  # Skip empty chunks

            chunk_metadata = source_metadata.copy()  # Inherit metadata
            chunk_metadata.update(
                {
                    "chunk_number": i + 1,
                    "split_method": split_method,
                }
            )

            # --- Optional Context Generation ---
            context_summary = None
            if (
                self.llm_add_context and split_method != "recursive"
            ):  # Only add context if LLM was used (or intended)
                context_summary = self._summarize_chunk(
                    chunk_text, full_doc_content_for_summary
                )
                if context_summary:
                    chunk_metadata["chunk_context"] = context_summary
                    chunk_text += f"\n\n[Context of above chunk with whole document: {context_summary}]"
            # --- End Optional Context Generation ---

            processed_chunks.append(
                Document(page_content=chunk_text, metadata=chunk_metadata)
            )

        return processed_chunks

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits a list of documents using the configured method (semantic or recursive).

        Args:
            documents: A list of LangChain Document objects (e.g., from load_from_source).

        Returns:
            A list of new Document objects, each representing a chunk.
        """
        if not documents:
            logger.warning("chunk_documents called with an empty list of documents.")
            return []

        all_chunks = []
        total_docs = len(documents)
        logger.info("Starting chunking process for %d documents...", total_docs)

        for i, doc in enumerate(documents):
            source_info = f"source: {doc.metadata.get('source', 'N/A')}, page: {doc.metadata.get('page', 'N/A')}"
            logger.info("Processing document %d/%d: %s", i + 1, total_docs, source_info)

            if not doc.page_content or doc.page_content.isspace():
                logger.info(
                    "Skipping document %d/%d (%s) due to empty content.",
                    i + 1,
                    total_docs,
                    source_info,
                )
                continue

            doc_chunks = []
            try:
                if self.use_semantic_chunking:
                    try:
                        # Attempt LLM semantic splitting
                        doc_chunks = self._split_with_llm(doc)
                    except (
                        LLMError,
                        ValueError,
                        requests.exceptions.RequestException,
                    ) as e:
                        # Logged inside _split_with_llm or _call_llm_api
                        logger.warning(
                            "Semantic LLM splitting failed for doc %d/%d (%s). Activating fallback recursive splitter. Reason: %s",
                            i + 1,
                            total_docs,
                            source_info,
                            e,
                        )
                        # Use fallback recursive splitter
                        doc_chunks = self._split_with_recursive(doc)
                    except Exception as e:
                        # Catch any other unexpected error during semantic attempt
                        logger.error(
                            "Unexpected error during semantic splitting attempt for doc %d/%d (%s). Error: %s",
                            i + 1,
                            total_docs,
                            source_info,
                            e,
                            exc_info=True,
                        )
                        logger.warning(
                            "Activating fallback recursive splitter due to unexpected error."
                        )
                        doc_chunks = self._split_with_recursive(doc)
                else:
                    # Semantic chunking is disabled, use recursive splitting directly
                    logger.info(
                        "Semantic chunking disabled, using recursive splitting for doc %d/%d.",
                        i + 1,
                        total_docs,
                    )
                    doc_chunks = self._split_with_recursive(doc)

                all_chunks.extend(doc_chunks)
                logger.info(
                    "Finished processing document %d/%d. Generated %d chunks using '%s' method.",
                    i + 1,
                    total_docs,
                    len(doc_chunks),
                    (
                        doc_chunks[0].metadata.get("split_method", "N/A")
                        if doc_chunks
                        else "N/A"
                    ),
                )

            except Exception as e:
                # Catch unexpected errors during the processing loop for a single document
                logger.error(
                    "Unexpected error processing document %d/%d (%s). Skipping this document. Error: %s",
                    i + 1,
                    total_docs,
                    source_info,
                    e,
                    exc_info=True,
                )
                continue  # Skip this document

        logger.info(
            "Completed chunking process. Total chunks generated: %d from %d documents.",
            len(all_chunks),
            total_docs,
        )
        return all_chunks

    # --- Convenience Method ---
    def load_and_chunk(
        self,
        source: Union[str, List[str]],
        source_type: str = "file",
        recursive: bool = True,
    ) -> List[Document]:
        """
        Convenience method to load documents from a source and then chunk them.

        Args:
            source: Path(s), URL(s), or S3 URI(s).
            source_type: 'file', 'directory', 's3', or 'web'.
            recursive: Whether to search directories recursively.

        Returns:
            A list of chunked LangChain Document objects.
        """
        logger.info(
            "Starting load_and_chunk process for source: %s (type: %s)",
            source,
            source_type,
        )
        # Step 1: Load documents
        loaded_docs = self.load_from_source(source, source_type, recursive)

        if not loaded_docs:
            logger.warning("No documents were loaded. Skipping chunking.")
            return []

        # Step 2: Chunk documents
        chunks = self.chunk_documents(loaded_docs)

        logger.info(
            "Load_and_chunk process completed. Generated %d chunks.", len(chunks)
        )
        return chunks


# --- Example Usage ---
if __name__ == "__main__":
    # Create a dummy text file for testing
    os.makedirs("./temp_docs", exist_ok=True)
    with open("./temp_docs/sample.txt", "w", encoding="utf-8") as f:
        f.write(
            "This is the first sentence.\nThis is the second sentence, which is a bit longer.\n\nThis marks a new paragraph.\nIt contains more information.\nFinally, the last sentence."
        )
    with open("./temp_docs/another.txt", "w", encoding="utf-8") as f:
        f.write("Document two starts here.\nIt has only two sentences.")

    # --- Configuration for the example ---
    # Option 1: Use only recursive chunking
    print("\n--- TESTING RECURSIVE CHUNKING ONLY ---")
    recursive_processor = RAGDocumentProcessor(
        use_semantic_chunking=False,  # Explicitly disable semantic
        recursive_chunk_size=50,  # Use small size for demonstration
        recursive_chunk_overlap=10,
    )
    recursive_chunks = recursive_processor.load_and_chunk(
        source="./temp_docs", source_type="directory"
    )
    print(f"Generated {len(recursive_chunks)} recursive chunks.")
    for i, chunk in enumerate(recursive_chunks):
        print(
            f"Chunk {i+1} (Method: {chunk.metadata.get('split_method')}, Source: {chunk.metadata.get('source')}):\n'{chunk.page_content}'"
        )
        print("-" * 20)

    # Option 2: Attempt semantic chunking (will likely fail without a running LLM at the default endpoint)
    # It should fall back to recursive chunking.
    print("\n--- TESTING SEMANTIC CHUNKING (EXPECTING FALLBACK) ---")
    # Ensure LLM endpoint is set, even if it's invalid, to trigger the attempt/fallback
    # Set add_context to True to test that logic as well (it should also fail gracefully)
    semantic_processor = RAGDocumentProcessor(
        use_semantic_chunking=True,
        llm_endpoint_url="http://localhost:12345/nonexistent",  # Invalid endpoint to force failure
        llm_headers={"Content-Type": "application/json"},
        llm_add_context=True,  # Try adding context
        recursive_chunk_size=100,  # Different fallback size for comparison
        recursive_chunk_overlap=20,
    )
    semantic_chunks = semantic_processor.load_and_chunk(
        source="./temp_docs/sample.txt", source_type="file"
    )
    print(f"Generated {len(semantic_chunks)} chunks (semantic attempt).")
    for i, chunk in enumerate(semantic_chunks):
        print(
            f"Chunk {i+1} (Method: {chunk.metadata.get('split_method')}, Context Added: {'chunk_context' in chunk.metadata}, Source: {chunk.metadata.get('source')}):\n'{chunk.page_content}'"
        )
        print("-" * 20)

    # Clean up dummy files
    # import shutil
    # shutil.rmtree("./temp_docs")
    # print("\nCleaned up temporary directory.")
