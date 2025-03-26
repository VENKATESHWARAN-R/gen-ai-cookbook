"""
Document Processor
"""
import os
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
    UnstructuredPowerPointLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Union, Optional
import boto3
import logging
from data_store.src.utils import config

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
                logger.error(f"Failed to initialize S3 client: {e}")
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
            ".html": UnstructuredHTMLLoader,
            ".htm": UnstructuredHTMLLoader,
            ".pptx": UnstructuredPowerPointLoader,
        }

    def _load_documents_from_loader(self, loader) -> List[Document]:
        """Helper function to load documents using a LangChain loader."""
        try:
            return loader.load()
        except Exception as e:
            # Attempt to get source if possible
            source = getattr(loader, 'file_path', getattr(loader, 'web_path', 'Unknown Source'))
            logger.error(f"Error loading document {source}: {e}")
            return []

    def _process_single_file(self, file_path: str) -> List[Document]:
        """Auto-detect file type and process it using the appropriate loader."""
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        loader_class = self.loaders_map.get(ext)

        if loader_class:
            logger.info(f"Processing file: {file_path} using {loader_class.__name__}")
            loader = loader_class(file_path)
            return self._load_documents_from_loader(loader)
        else:
            logger.warning(f"Unsupported file format: {file_path}")
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
                    logger.warning(f"File not found: {file_path}")

        elif source_type == "directory":
            for dir_path in sources:
                if not os.path.isdir(dir_path):
                    logger.warning(f"Directory not found: {dir_path}")
                    continue
                logger.info(f"Scanning directory: {dir_path} (recursive={recursive})")
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
                    logger.info(f"Downloading s3://{bucket}/{key} to {local_file}")
                    self.s3_client.download_file(bucket, key, local_file)
                    docs = self._process_single_file(local_file)
                    # Add S3 source to metadata
                    for doc in docs:
                        doc.metadata['source'] = f"s3://{bucket}/{key}"
                    all_docs.extend(docs)
                    os.remove(local_file)
                except Exception as e:
                    logger.error(f"Failed to process S3 file {s3_uri}: {e}")

        elif source_type == "web":
             for url in sources:
                 logger.info(f"Loading web page: {url}")
                 loader = WebBaseLoader(url)
                 all_docs.extend(self._load_documents_from_loader(loader))

        else:
             logger.error(f"Unsupported source_type: {source_type}")

        logger.info(f"Loaded {len(all_docs)} documents initially.")
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
        logger.info(f"Splitting {len(documents)} documents into chunks (size={self.text_splitter._chunk_size}, overlap={self.text_splitter._chunk_overlap})...")
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks.")
        return chunks