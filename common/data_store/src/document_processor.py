"""
Document Processor
"""
import logging
import os
from typing import List, Optional, Union

import boto3
from common.data_store.src import config
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (CSVLoader, Docx2txtLoader,
                                                  PyMuPDFLoader, TextLoader,
                                                  UnstructuredMarkdownLoader,
                                                  WebBaseLoader)
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(
        self,
        aws_access_key: Optional[str] = config.AWS_ACCESS_KEY_ID,
        aws_secret_key: Optional[str] = config.AWS_SECRET_ACCESS_KEY,
        aws_region: str = config.AWS_REGION,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
    ):
        """Initialize the document processor."""
        self.s3_client = None
        if aws_access_key and aws_secret_key:
            try:
                self.s3_client = boto3.client(
                    "s3",
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key,
                    region_name=aws_region,
                )
                logger.info("S3 client initialized.")
            except Exception as e:
                logger.error("Failed to initialize S3 client: %s", e)
        else:
            logger.warning("AWS credentials not provided. S3 functionality disabled.")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        # Define supported loaders using a mapping
        self.loaders_map = {
            ".pdf": PyMuPDFLoader,
            ".docx": Docx2txtLoader,
            ".csv": CSVLoader,
            ".txt": TextLoader,
            ".md": UnstructuredMarkdownLoader,
        }

    def _load_documents_from_loader(self, loader) -> List[Document]:
        """Helper function to load documents using a LangChain loader."""
        try:
            return loader.load()
        except Exception as e:
            # Attempt to get source if possible
            source = getattr(loader, 'file_path', getattr(loader, 'web_path', 'Unknown Source'))
            logger.error("Error loading document %s: %s", source, e)
            return []

    def _process_single_file(self, file_path: str) -> List[Document]:
        """Auto-detect file type and process it using the appropriate loader."""
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        loader_class = self.loaders_map.get(ext)

        if loader_class:
            logger.info("Processing file: %s using %s", file_path, loader_class.__name__)
            loader = loader_class(file_path)
            return self._load_documents_from_loader(loader)
        else:
            logger.warning("Unsupported file format: %s", file_path)
            return []

    def load_from_source(self, source: Union[str, List[str]], source_type: str = "file", recursive: bool = True) -> List[Document]:
        """
        Loads documents from various sources like files, directories, S3, or web pages.

        Args:
            source: Path(s), URL(s), or S3 key(s).
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
                logger.info("Scanning directory: %s (recursive=%s)", dir_path, recursive)
                for root, _, files in os.walk(dir_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        all_docs.extend(self._process_single_file(file_path))
                    if not recursive:
                        break # Stop after the top-level directory

        elif source_type == "s3":
            if not self.s3_client:
                logger.error("S3 client not initialized. Cannot load from S3.")
                return []
            for s3_uri in sources: # Expecting s3://bucket-name/key or just bucket-name/key
                try:
                    if s3_uri.startswith("s3://"):
                        s3_uri = s3_uri[5:]
                    bucket, key = s3_uri.split('/', 1)
                    local_file = os.path.join(config.TEMP_DIR, os.path.basename(key))
                    os.makedirs(config.TEMP_DIR, exist_ok=True)
                    logger.info("Downloading s3://%s/%s to %s", bucket, key, local_file)
                    self.s3_client.download_file(bucket, key, local_file)
                    docs = self._process_single_file(local_file)
                    # Add S3 source to metadata
                    for doc in docs:
                        doc.metadata['source'] = f"s3://{bucket}/{key}"
                    all_docs.extend(docs)
                    os.remove(local_file)
                except Exception as e:
                    logger.error("Failed to process S3 file %s: %s", s3_uri, e)

        elif source_type == "web":
             for url in sources:
                 logger.info("Loading web page: %s", url)
                 loader = WebBaseLoader(url)
                 all_docs.extend(self._load_documents_from_loader(loader))

        else:
             logger.error("Unsupported source_type: %s", source_type)

        logger.info("Loaded %d documents initially.", len(all_docs))
        return all_docs

    def create_chunks(self, documents: List[Document]) -> List[Document]:
        """
        Split loaded documents into smaller chunks.

        Args:
            documents: List of LangChain Document objects.

        Returns:
            List of chunked LangChain Document objects.
        """
        if not documents:
            return []
        logger.info(
            "Splitting %d documents into chunks (size=%d, overlap=%d)...",
            len(documents),
            self.text_splitter._chunk_size,
            self.text_splitter._chunk_overlap,
        )
        chunks = self.text_splitter.split_documents(documents)
        logger.info("Created %d chunks.", len(chunks))
        return chunks