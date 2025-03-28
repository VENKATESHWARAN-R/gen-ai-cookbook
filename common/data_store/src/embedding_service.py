"""
This module provides a service for generating embeddings using the Sentence Transformer model.
"""
import logging
from typing import List

from common.data_store.src import config
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str = config.EMBEDDING_MODEL_NAME):
        """Initializes the Sentence Transformer model."""
        try:
            self.model = SentenceTransformer(model_name)
            logger.info("Loaded Sentence Transformer model: %s", model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info("Embedding dimension: %s", self.dimension)
        except Exception as e:
            logger.error("Failed to load Sentence Transformer model '%s': %s", model_name, e)
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embeddings (each embedding is a list of floats).
        """
        logger.info("Generating embeddings for %d documents...", len(texts))
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
        logger.debug("Generating embedding for query: '%s...'", text[:50])
        embedding = self.model.encode(text)
        return embedding.tolist()

# embedding_model = SentenceTransformerEmbeddings()