"""
Data Pipeline
-------------
This module defines the data pipeline class that processes and stores documents
in the vector store, and retrieves relevant documents for a given query.
"""

import logging
from typing import List, Union, Dict, Any

from document_processor import RAGDocumentProcessor
from embedding_service import SentenceTransformerEmbeddings
from chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class DataPipeline:
    def __init__(
        self,
        document_processor: RAGDocumentProcessor = None,
        embedding_provider: SentenceTransformerEmbeddings = None,
        vector_store: ChromaVectorStore = None,
    ):
        """
        Initializes the data pipeline with necessary components.
        If components are not provided, default instances are created.
        """
        self.document_processor = document_processor or RAGDocumentProcessor()
        self.embedding_provider = embedding_provider or SentenceTransformerEmbeddings()
        # Pass the embedding provider to the vector store if creating it here
        self.vector_store = vector_store or ChromaVectorStore(
            embedding_provider=self.embedding_provider
        )
        logger.info("Data Pipeline initialized.")

    def process_and_store(
        self,
        source: Union[str, List[str]],
        source_type: str = "file",  # 'file', 'directory', 's3', 'web'
        contextualize: bool = False,
        max_context_workers: int = 4,
    ) -> int:
        """
        Loads, processes, optionally contextualizes, and stores documents
        from a given source into the vector store.

        Args:
            source: Path(s), URL(s), or S3 key(s).
            source_type: Type of the source ('file', 'directory', 's3', 'web').
            max_context_workers: Number of threads for parallel contextualization.

        Returns:
            The number of chunks successfully added to the vector store.
        """
        logger.info(
            "Starting processing for source: %s (type: %s)", source, source_type
        )

        # 1. Load Documents
        documents = self.document_processor.load_from_source(source, source_type)
        if not documents:
            logger.warning("No documents loaded. Aborting processing.")
            return 0

        logger.info("Loaded %d documents from source.", len(documents))

        # 2. Create Chunks
        chunks_to_embed = self.document_processor.chunk_documents(documents)
        if not chunks_to_embed:
            logger.warning("No chunks created from documents. Aborting processing.")
            return 0

        if not chunks_to_embed:
            logger.error("No chunks available after contextualization step. Aborting.")
            return 0

        # 4. Add to Vector Store (Embeddings handled by ChromaDB via its function)
        initial_count = self.vector_store.get_collection_count()
        self.vector_store.add_documents(chunks_to_embed)
        final_count = self.vector_store.get_collection_count()
        added_count = final_count - initial_count

        logger.info(
            "Processing complete. Added %d chunks to the vector store.", added_count
        )
        return added_count

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves relevant document chunks for a given query.

        Args:
            query: The search query string.
            k: The number of results to retrieve.

        Returns:
            A list of search result dictionaries.
        """
        logger.info("Retrieving top %d results for query: '%s...'", k, query[:50])
        results = self.vector_store.search(query, k=k)

        return results

    def get_vector_store_count(self) -> int:
        """Returns the total number of items in the vector store collection."""
        return self.vector_store.get_collection_count()

    def clear_vector_store(self):
        """Clears all data from the configured vector store collection."""
        self.vector_store.clear_collection()
