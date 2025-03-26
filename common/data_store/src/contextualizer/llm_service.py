"""
Local LLM Contextualizer
------------------------
"""
import json
import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import requests
from data_store.src.utils import config
from langchain_core.documents import Document
from tqdm import tqdm

logger = logging.getLogger(__name__)


class LocalLLMContextualizer:
    def __init__(self, api_url: str = config.LOCAL_LLM_URL):
        """
        Initializes the contextualizer with the local LLM API endpoint.

        Args:
            api_url: The URL of your local LLM's completion endpoint.
        """
        self.api_url = api_url
        logger.info("Local LLM Contextualizer configured for URL: %s", api_url)

    def _generate_context_with_llm(
        self, full_doc_content: str, chunk_content: str
    ) -> Optional[str]:
        """Sends request to the local LLM to get context."""
        prompt = config.CONTEXTUAL_PROMPT_TEMPLATE.format(
            full_doc_content=full_doc_content, chunk_content=chunk_content
        )
        logger.debug("Generated prompt: '%s'", prompt)

        headers = {"accept": "application/json", "Content-Type": "application/json"}
        data = {"prompt": prompt}

        try:
            response = requests.post(
                self.api_url, headers=headers, data=json.dumps(data), timeout=60
            )  # Increased timeout
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

            result = response.json()
            context = result.get("response", "").strip()

            if not context:
                logger.warning(
                    "LLM returned empty context for chunk: %s...", chunk_content[:50]
                )
                return None

            logger.debug("Generated context: '%s'", context)
            return context

        except requests.exceptions.RequestException as e:
            logger.error("Error calling local LLM at %s: %s", self.api_url, e)
            return None
        except Exception as e:
            logger.error("Error processing LLM response: %s", e)
            return None

    def _process_chunk_for_context(
        self, full_doc_map: Dict[str, str], chunk: Document
    ) -> Tuple[Document, Optional[str]]:
        """Helper for parallel processing: gets context and prepares augmented chunk."""
        source = chunk.metadata.get("source", "unknown_source")
        full_doc_content = full_doc_map.get(source)

        if not full_doc_content:
            logger.warning(
                "Could not find full document content for source: %s. Skipping contextualization for this chunk.",
                source,
            )
            return chunk, None  # Return original chunk

        original_content = chunk.page_content
        generated_context = self._generate_context_with_llm(
            full_doc_content, original_content
        )

        if generated_context:
            # Prepend context for embedding, store original/context in metadata
            augmented_content = f"{original_content}\n\nContext of chunk over document: {generated_context}"
            augmented_chunk = Document(
                page_content=augmented_content,
                metadata={
                    **chunk.metadata,
                    "original_content": original_content,
                    "generated_context": generated_context,
                },
            )
            return augmented_chunk, generated_context  # Return augmented chunk
        else:
            # If context generation failed, return original chunk
            return chunk, None

    def add_context_to_chunks(
        self, documents: List[Document], chunks: List[Document], max_workers: int = 4
    ) -> List[Document]:
        """
        Adds context to each chunk using the local LLM, processing in parallel.

        Args:
            documents: The original list of full documents (used to get full text).
            chunks: The list of chunked Documents.
            max_workers: Number of parallel threads for calling the LLM.

        Returns:
            A new list of Documents where page_content includes the original content
            plus the generated context (if successful), and metadata is updated.
        """
        if not self.api_url:
            logger.warning("Local LLM URL not configured. Skipping contextualization.")
            return chunks

        start_time = time.time()
        logger.info(
            "Starting contextualization for %d chunks using %d workers...",
            len(chunks),
            max_workers,
        )

        # Create a map of source -> full document content for quick lookup
        logger.debug("Aggregating content from %d page-documents into full_doc_map...", len(documents))
        full_doc_map = defaultdict(str)
        for i, doc in enumerate(documents):
            source = doc.metadata.get("source")
            if source:
                # Append page content. Adding a separator helps the LLM understand page breaks.
                full_doc_map[source] += doc.page_content + "\n\n"
            else:
                 # Assign a unique key if source is missing, though less ideal
                 missing_source_key = f"unknown_source_{i}"
                 full_doc_map[missing_source_key] += doc.page_content + "\n\n"
                 logger.warning("Document at index %d missing 'source' metadata. Using key '%s'. Chunk context might be inaccurate.", i, missing_source_key)

        contextualized_chunks = []
        processed_count = 0
        failed_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks: process_chunk(full_doc_map, chunk)
            future_to_chunk = {
                executor.submit(
                    self._process_chunk_for_context, full_doc_map, chunk
                ): chunk
                for chunk in chunks
            }

            for future in tqdm(
                as_completed(future_to_chunk),
                total=len(chunks),
                desc="Contextualizing Chunks",
            ):
                original_chunk = future_to_chunk[future]
                try:
                    augmented_chunk, generated_context = future.result()
                    contextualized_chunks.append(augmented_chunk)
                    processed_count += 1
                    if generated_context is None and full_doc_map.get(
                        original_chunk.metadata.get("source")
                    ):  # Check if context *should* have been generated
                        failed_count += 1
                except Exception as exc:
                    logger.error(
                        "Chunk from %s generated an exception during contextualization: %s",
                        original_chunk.metadata.get("source", "unknown"),
                        exc,
                    )
                    contextualized_chunks.append(
                        original_chunk
                    )  # Add original chunk back if processing failed
                    failed_count += 1
                    processed_count += 1  # Count as processed even if failed

        end_time = time.time()
        logger.info(
            "Contextualization finished in %.2f seconds.", end_time - start_time
        )
        logger.info(
            "Successfully processed: %d, Failed to generate/add context for: %d chunks.",
            processed_count,
            failed_count,
        )

        return contextualized_chunks
