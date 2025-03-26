"""
This module provides a service for generating embeddings using the Sentence Transformer model.
"""
from sentence_transformers import SentenceTransformer
from typing import List
import logging
from data_store.src.utils import config

logger = logging.getLogger(__name__)

class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str = config.EMBEDDING_MODEL_NAME):
        """Initializes the Sentence Transformer model."""
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Loaded Sentence Transformer model: {model_name}")
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Embedding dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to load Sentence Transformer model '{model_name}': {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embeddings (each embedding is a list of floats).
        """
        logger.info(f"Generating embeddings for {len(texts)} documents...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        logger.info("Embeddings generated.")
        # Convert numpy arrays to lists of floats
        return [embedding.tolist() for embedding in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """
        Embeds a single query text.

        Args:
            text: The query string.

        Returns:
            The embedding (list of floats).
        """
        logger.debug(f"Generating embedding for query: '{text[:50]}...'")
        embedding = self.model.encode(text)
        return embedding.tolist()

# embedding_model = SentenceTransformerEmbeddings()