import os
from langchain.document_loaders import (
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
from typing import List, Dict, Union
import boto3


class DocumentProcessor:
    def __init__(
        self, aws_access_key=None, aws_secret_key=None, aws_region="us-east-1"
    ):
        """Initialize the document processor with optional AWS credentials."""
        self.s3_client = None
        if aws_access_key and aws_secret_key:
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=aws_region,
            )

    def _load_documents(self, loader, source: str) -> List[Dict]:
        """Helper function to load documents and return structured data."""
        documents = loader.load()
        return [
            {
                "source": source,
                "content": doc.page_content,
                "metadata": doc.metadata if hasattr(doc, "metadata") else {},
            }
            for doc in documents
        ]

    def _process_file(self, file_path: str) -> List[Dict]:
        """Auto-detect file type and process it using the appropriate loader."""
        if file_path.endswith(".pdf"):
            return self.load_pdf(file_path)
        elif file_path.endswith(".docx"):
            return self.load_word(file_path)
        elif file_path.endswith(".csv"):
            return self.load_csv(file_path)
        elif file_path.endswith(".txt"):
            return self.load_text(file_path)
        elif file_path.endswith(".md"):
            return self.load_markdown(file_path)
        elif file_path.endswith(".html") or file_path.endswith(".htm"):
            return self.load_html(file_path)
        elif file_path.endswith(".pptx"):
            return self.load_pptx(file_path)
        else:
            return [{"source": file_path, "error": "Unsupported file format"}]

    def load_directory(self, directory_path: str, recursive: bool = True) -> List[Dict]:
        """Load all supported files from a directory and return structured results."""
        results = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                results.extend(self._process_file(file_path))
            if not recursive:
                break  # Stop if recursive processing is disabled
        return results

    def load_pdf(self, file_paths: Union[str, List[str]]) -> List[Dict]:
        """Load PDF files and return structured results."""
        file_paths = [file_paths] if isinstance(file_paths, str) else file_paths
        results = []
        for path in file_paths:
            loader = PyMuPDFLoader(path)
            results.extend(self._load_documents(loader, path))
        return results

    def load_word(self, file_paths: Union[str, List[str]]) -> List[Dict]:
        """Load Word documents (.docx) and return structured results."""
        file_paths = [file_paths] if isinstance(file_paths, str) else file_paths
        results = []
        for path in file_paths:
            loader = Docx2txtLoader(path)
            results.extend(self._load_documents(loader, path))
        return results

    def load_csv(self, file_paths: Union[str, List[str]]) -> List[Dict]:
        """Load CSV files and return structured results."""
        file_paths = [file_paths] if isinstance(file_paths, str) else file_paths
        results = []
        for path in file_paths:
            loader = CSVLoader(path)
            results.extend(self._load_documents(loader, path))
        return results

    def load_text(self, file_paths: Union[str, List[str]]) -> List[Dict]:
        """Load plain text files and return structured results."""
        file_paths = [file_paths] if isinstance(file_paths, str) else file_paths
        results = []
        for path in file_paths:
            loader = TextLoader(path)
            results.extend(self._load_documents(loader, path))
        return results

    def load_markdown(self, file_paths: Union[str, List[str]]) -> List[Dict]:
        """Load Markdown files and return structured results."""
        file_paths = [file_paths] if isinstance(file_paths, str) else file_paths
        results = []
        for path in file_paths:
            loader = UnstructuredMarkdownLoader(path)
            results.extend(self._load_documents(loader, path))
        return results

    def load_html(self, file_paths: Union[str, List[str]]) -> List[Dict]:
        """Load HTML files and return structured results."""
        file_paths = [file_paths] if isinstance(file_paths, str) else file_paths
        results = []
        for path in file_paths:
            loader = UnstructuredHTMLLoader(path)
            results.extend(self._load_documents(loader, path))
        return results

    def load_pptx(self, file_paths: Union[str, List[str]]) -> List[Dict]:
        """Load PowerPoint files and return structured results."""
        file_paths = [file_paths] if isinstance(file_paths, str) else file_paths
        results = []
        for path in file_paths:
            loader = UnstructuredPowerPointLoader(path)
            results.extend(self._load_documents(loader, path))
        return results

    def load_s3_file(self, bucket: str, file_keys: Union[str, List[str]]) -> List[Dict]:
        """Download files from S3 and process them using appropriate loaders."""
        if not self.s3_client:
            raise ValueError("S3 client is not initialized. Provide AWS credentials.")

        file_keys = [file_keys] if isinstance(file_keys, str) else file_keys
        results = []
        for file_key in file_keys:
            local_file = f"/tmp/{file_key.split('/')[-1]}"
            self.s3_client.download_file(bucket, file_key, local_file)
            results.extend(self._process_file(local_file))

        return results

    def load_webpage(self, urls: Union[str, List[str]]) -> List[Dict]:
        """Load content from a web page and return structured results."""
        urls = [urls] if isinstance(urls, str) else urls
        results = []
        for url in urls:
            loader = WebBaseLoader(url)
            results.extend(self._load_documents(loader, url))
        return results

    @staticmethod
    def create_chunks(documents, chunk_size=800, chunk_overlap=200):
        """
        Split documents into overlapping chunks

        Args:
            documents (list): List of documents to split
            chunk_size (int): Size of each chunk in characters
            chunk_overlap (int): Number of characters to overlap between chunks

        Returns:
            list: List of text chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        chunks = text_splitter.split_documents(documents)
        return chunks
