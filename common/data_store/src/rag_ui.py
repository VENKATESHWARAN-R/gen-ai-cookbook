import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Union

import streamlit as st

try:
    from chroma_store import ChromaVectorStore
    from document_processor import RAGDocumentProcessor
    from embedding_service import SentenceTransformerEmbeddings
    from data_pipeline import DataPipeline
except ImportError as e:
    st.error(
        f"Failed to import pipeline components. Make sure the script can find your modules. Error: {e}"
    )
    st.stop()  # Stop execution if imports fail


# --- Basic Logging Setup ---
# Streamlit manages its own logging, but we can configure the logger used by the pipeline
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- Helper Function ---
def save_uploaded_files(
    uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile],
) -> List[str]:
    """Saves uploaded files to a temporary directory and returns their paths."""
    saved_files_paths = []
    # Create a single temporary directory for all uploaded files in this session/run
    temp_dir = tempfile.mkdtemp(prefix="st_uploads_")
    st.session_state["temp_upload_dir"] = temp_dir

    for uploaded_file in uploaded_files:
        try:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_files_paths.append(file_path)
            logger.info("Saved uploaded file to temporary path: %s", file_path)
        except Exception as e:
            st.error(f"Error saving uploaded file {uploaded_file.name}: {e}")
            logger.error(
                "Error saving uploaded file %s: %s",
                uploaded_file.name,
                e,
                exc_info=True,
            )
    return saved_files_paths


# Initialize embedding model for singleton use
EMBED_PROVIDER = SentenceTransformerEmbeddings()


# Check if the embedding model is loaded correctly
def test_embedding_model():
    """
    Created as a seperate fuinction to not spam the global namespace
    """

    if EMBED_PROVIDER is None:
        # st.error("Failed to load the embedding model.")
        logger.error("Embedding model initialization failed.")
    else:
        logger.info("Embedding model loaded successfully.")
    # Check if ChromaDB connection is valid
    try:
        _vector_store = ChromaVectorStore(
            hostname="localhost",
            port=8001,
            collection_name="rag_collection",
            embedding_provider=EMBED_PROVIDER,
        )
        # Test connection
        _count = _vector_store.get_collection_count()
        if _count is None:
            # st.error("Failed to connect to ChromaDB.")
            logger.error("ChromaDB connection failed.")
        else:
            logger.info("Connected to ChromaDB successfully.")
            # st.success(f"Connected to ChromaDB successfully. Collection count: {_count}")
    except Exception as e:
        # st.error(f"Failed to connect to ChromaDB: {e}")
        logger.error("ChromaDB connection error: %s", e, exc_info=True)


test_embedding_model()

# --- Streamlit App UI ---

st.set_page_config(layout="wide", page_title="RAG Data Pipeline Demo")

st.title("📄 RAG Data Pipeline Demo UI")
st.markdown("Upload documents, configure processing, and ingest data into ChromaDB.")

# --- Sidebar for Configuration ---
st.sidebar.header("⚙️ Pipeline Configuration")

# ChromaDB Connection
st.sidebar.subheader("ChromaDB Connection")
chroma_host = st.sidebar.text_input(
    "Chroma Host", value="localhost"
)  # Replace with your config default
chroma_port = st.sidebar.number_input(
    "Chroma Port", value=8001
)  # Replace with your config default
chroma_collection = st.sidebar.text_input(
    "Collection Name", value="rag_collection"
)  # Replace with your config default

# LLM Connection (Required for Semantic Chunking / Contextualization)
st.sidebar.subheader("LLM Configuration")
llm_endpoint = st.sidebar.text_input(
    "LLM API Endpoint", value="http://localhost:8000/api/llm/generate_response"
)  # Default from your code
# Simple JSON input for headers, could be improved
llm_headers_str = st.sidebar.text_area(
    "LLM API Headers (JSON)",
    value='{\n  "accept": "application/json",\n  "Content-Type": "application/json"\n}',
)
try:
    llm_headers = json.loads(llm_headers_str) if llm_headers_str else None
except json.JSONDecodeError:
    st.sidebar.error("Invalid JSON format for LLM Headers.")
    llm_headers = None


# Embedding Model (Using default SentenceTransformer from your code)
st.sidebar.subheader("Embedding Model")
st.sidebar.info("Using default SentenceTransformer model.")  # Placeholder

# --- Main Area ---

col1, col2 = st.columns(2)

with col1:
    st.header("1. Processing Parameters")
    use_semantic = st.toggle("Use Semantic Chunking (LLM)", value=False)
    #add_context = st.toggle(
    #    "Add Context (LLM)", value=False, disabled=not use_semantic
    #)  # Only enable if semantic is on
    add_context = st.toggle("Add Context (LLM)", value=False) # Enable regardless of chunking method

    # Adjust chunk size/overlap based on whether semantic is chosen?
    # For now, keep them generic but label clearly.
    if use_semantic:
        st.markdown(
            "_(Semantic chunking aims for target size but prioritizes boundaries)_"
        )
        chunk_size = st.slider(
            "Target Chunk Size (Tokens)", 100, 2000, 125
        )  # LLM Target
        chunk_overlap = st.slider(
            "Fallback Chunk Overlap", 0, 500, 50
        )  # Recursive fallback overlap
        fallback_chunk_size = st.slider(
            "Fallback Chunk Size", 100, 4000, 650
        )  # Recursive fallback size
    else:
        st.markdown("_(Using Recursive Character Splitting)_")
        chunk_size = st.slider("Chunk Size", 100, 4000, 650)  # Recursive Size
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 50)  # Recursive Overlap
        # Semantic parameters not needed when it's off
        fallback_chunk_size = chunk_size  # Not directly used when semantic is off

    st.info(
        f"""
    **Current Settings:**
    - Chunking Method: {'Semantic (LLM)' if use_semantic else 'Recursive'}
    - {'Target LLM Size:' if use_semantic else 'Chunk Size:'} {chunk_size}
    - {'Fallback ' if use_semantic else ''}Overlap: {chunk_overlap}
    {'- Fallback Recursive Size: ' + str(fallback_chunk_size) if use_semantic else ''}
    - Add Context (LLM): {'Yes' if add_context else 'No'}
    """
    )


with col2:
    st.header("2. Select Data Source")
    source_type = st.radio(
        "Source Type:",
        options=["File Upload", "Local Directory", "S3 URI", "Web URL"],
        horizontal=True,
        index=0,  # Default to File Upload
    )

    source_input: Union[str, List[str], None] = None
    uploaded_files_list = []

    if source_type == "File Upload":
        uploaded_files_list = st.file_uploader(
            "Upload Files (.pdf, .docx, .csv, .txt, .md)",
            accept_multiple_files=True,
            type=["pdf", "docx", "csv", "txt", "md"],
        )
        if uploaded_files_list:
            st.success(f"Ready to process {len(uploaded_files_list)} uploaded file(s).")

    elif source_type == "Local Directory":
        source_input = st.text_input(
            "Enter full path to the directory:", placeholder="/path/to/your/documents"
        )
        if source_input and not os.path.isdir(source_input):
            st.warning("Please enter a valid directory path.")
        elif source_input:
            st.success(f"Ready to process directory: {source_input}")

    elif source_type == "S3 URI":
        source_input = st.text_input(
            "Enter S3 URI(s) (comma-separated if multiple):",
            placeholder="s3://bucket-name/file.pdf, s3://bucket-name/prefix/",
        )
        if source_input:
            # Simple split, assumes no commas in URIs themselves
            source_input = [
                uri.strip() for uri in source_input.split(",") if uri.strip()
            ]
            st.success(f"Ready to process {len(source_input)} S3 URI(s).")

    elif source_type == "Web URL":
        source_input = st.text_input(
            "Enter Web URL(s) (comma-separated if multiple):",
            placeholder="https://example.com/page1, https://another.com/article",
        )
        if source_input:
            # Simple split
            source_input = [
                url.strip() for url in source_input.split(",") if url.strip()
            ]
            st.success(f"Ready to process {len(source_input)} Web URL(s).")


# --- Processing Trigger ---
st.divider()
st.header("3. Process and Store Data")

process_button = st.button(
    "Start Processing",
    type="primary",
    disabled=(not uploaded_files_list and not source_input),
)

if process_button:
    # --- Input Validation ---
    final_source = None
    final_source_type_str = ""

    if source_type == "File Upload":
        if not uploaded_files_list:
            st.error("Please upload at least one file.")
            st.stop()
        # Save uploaded files to temporary location
        with st.spinner("Saving uploaded files..."):
            final_source = save_uploaded_files(uploaded_files_list)
        if not final_source:
            st.error("Failed to save uploaded files. Cannot proceed.")
            st.stop()
        final_source_type_str = "file"  # Process saved files individually

    elif source_type == "Local Directory":
        if not source_input or not os.path.isdir(str(source_input)):
            st.error("Please provide a valid local directory path.")
            st.stop()
        final_source = str(source_input)
        final_source_type_str = "directory"

    elif source_type == "S3 URI":
        if not source_input:
            st.error("Please provide at least one S3 URI.")
            st.stop()
        final_source = source_input  # Already a list of strings
        final_source_type_str = "s3"

    elif source_type == "Web URL":
        if not source_input:
            st.error("Please provide at least one Web URL.")
            st.stop()
        final_source = source_input  # Already a list of strings
        final_source_type_str = "web"

    # --- Initialize Pipeline Components ---
    status_placeholder = st.empty()
    with st.status("Initializing Pipeline...", expanded=True) as status:
        try:
            st.write("Initializing Document Processor...")
            # Pass parameters from UI to RAGDocumentProcessor
            doc_processor = RAGDocumentProcessor(
                aws_access_key=None,  # Configure AWS via environment or secrets if needed for S3
                aws_secret_key=None,
                # aws_region="your-region", # If needed
                use_semantic_chunking=use_semantic,
                llm_endpoint_url=(
                    llm_endpoint if use_semantic or add_context else None
                ),  # Pass LLM URL if needed
                llm_headers=llm_headers if use_semantic or add_context else None,
                llm_target_chunk_size=(
                    chunk_size if use_semantic else 500
                ),  # Use appropriate size
                # llm_size_unit='tokens', # Keep default or make configurable
                llm_add_context=add_context,
                recursive_chunk_size=(
                    fallback_chunk_size if use_semantic else chunk_size
                ),  # Pass correct size
                recursive_chunk_overlap=chunk_overlap,
            )
            st.write("Document Processor Initialized.")

            st.write("Initializing Embedding Provider...")
            # Assuming default SentenceTransformerEmbeddings is sufficient for demo
            embed_provider = EMBED_PROVIDER
            st.write("Embedding Provider Initialized.")

            st.write("Connecting to Chroma Vector Store...")
            vector_store = ChromaVectorStore(
                hostname=chroma_host,
                port=chroma_port,
                collection_name=chroma_collection,
                embedding_provider=embed_provider,  # Pass the embedder instance
            )
            # Test connection? vector_store.get_collection_count() maybe?
            st.write(
                f"Connected to ChromaDB at {chroma_host}:{chroma_port}, collection: '{chroma_collection}'."
            )

            st.write("Initializing Data Pipeline...")
            # Assuming LocalLLMContextualizer is imported or handled within DataPipeline/RAGDocProcessor
            # If LocalLLMContextualizer needs the endpoint, it should be passed here or obtained from doc_processor
            pipeline = DataPipeline(
                document_processor=doc_processor,
                embedding_provider=embed_provider,
                vector_store=vector_store,
                # contextualizer=LocalLLMContextualizer(llm_endpoint=llm_endpoint, llm_headers=llm_headers) # If needed
            )
            st.write("Data Pipeline Initialized.")
            status.update(
                label="Pipeline Initialized Successfully!",
                state="complete",
                expanded=False,
            )

        except Exception as e:
            logger.error("Pipeline Initialization Failed: %s", e, exc_info=True)
            status.update(
                label="Pipeline Initialization Failed!", state="error", expanded=True
            )
            st.error(f"Error initializing pipeline: {e}")
            st.stop()  # Stop if initialization fails

    # --- Run Processing ---
    st.info(
        f"Starting processing for source: {final_source} (Type: {final_source_type_str})"
    )
    added_count = 0
    # Use st.spinner for the actual processing call
    with st.spinner(f"Processing source: {final_source}... This may take a while."):
        try:
            added_count = pipeline.process_and_store(
                source=final_source,
                source_type=final_source_type_str,
            )
            st.success(
                f"✅ Processing complete! Added {added_count} chunks to the vector store."
            )

        except Exception as e:
            logger.error("Error during pipeline processing: %s", e, exc_info=True)
            st.error(f"❌ Processing failed: {e}")

# --- Vector Store Inspection ---
st.divider()
st.header("4. Inspect Vector Store")

col3, col4 = st.columns([1, 3])

with col3:
    inspect_button = st.button("Show Store Status & Sources")

with col4:
    if inspect_button:
        try:
            # Re-initialize quickly just for inspection
            # (Could use session state to avoid this if performance is critical)
            st.write("Connecting to ChromaDB for inspection...")
            inspect_embed_provider = EMBED_PROVIDER  # Need an embedder instance
            inspect_vector_store = ChromaVectorStore(
                hostname=chroma_host,
                port=chroma_port,
                collection_name=chroma_collection,
                embedding_provider=inspect_embed_provider,
            )

            count = inspect_vector_store.get_collection_count()
            st.metric("Total Items in Collection", value=count)

            if count > 0:
                with st.spinner(f"Fetching sources from {count} items (limit 1000)..."):
                    # Fetch metadata to extract sources. Limit query for performance.
                    limit = 1000
                    results = inspect_vector_store.collection.get(
                        limit=min(count, limit), include=["metadatas"]
                    )
                    sources = set()
                    if results and results.get("metadatas"):
                        for meta in results["metadatas"]:
                            if meta and "source" in meta:
                                sources.add(meta["source"])
                            elif meta and "file_path" in meta:  # Check alternative key
                                sources.add(meta["file_path"])

                    if sources:
                        st.subheader("Unique Sources Found (Sample):")
                        # Display as a list or dataframe
                        # st.dataframe(list(sources), use_container_width=True) # DF might be nicer
                        st.json(list(sources))  # JSON is also clear
                    else:
                        st.info(
                            "Could not extract 'source' metadata from fetched items."
                        )
                    if count > limit:
                        st.warning(
                            f"Only showing sources from the first {limit} items."
                        )

            else:
                st.info("The collection is currently empty.")

        except Exception as e:
            logger.error("Error inspecting vector store: %s", e, exc_info=True)
            st.error(f"Failed to inspect vector store: {e}")
