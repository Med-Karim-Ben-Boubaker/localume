import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Iterator, Optional, Callable, Union
from datetime import datetime
import mimetypes
import hashlib
from dataclasses import dataclass, field
import numpy as np

from ..utils.logger import Logger
from ..utils.pdf_extractor import PDFExtractor
from ..utils.text_file_extractor import TextFileExtractor
from ..embeddings.embedding_generator import EmbeddingModel
from ..embeddings.vector_store import VectorStore

@dataclass
class ScannedFile:
    """
    Data class representing a scanned file with its embedding and metadata.
    
    Attributes:
        embedding (np.ndarray): The embedding vector for the file content
        metadata (Dict[str, Any]): File metadata including path, name, and timestamps
        unique_id (int): Unique identifier for the file
    """
    embedding: np.ndarray
    metadata: Dict[str, Any]
    unique_id: int

@dataclass
class ScanResult:
    """
    Data class representing the results of a directory scan.
    
    Attributes:
        scanned_files (List[ScannedFile]): List of scanned files with their data
        scan_time (datetime): When the scan was performed
        scanned_paths (List[str]): List of paths that were scanned
        errors (List[str]): List of any errors encountered during scanning
    """
    scanned_files: List[ScannedFile]
    scan_time: datetime
    scanned_paths: List[str]
    errors: List[str] = field(default_factory=list)

class FileScanner:
    """
    A class to scan directories, extract metadata, and write scan results.
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf": "application/pdf",
        ".txt": "text/plain",
        ".md": "text/markdown",
    }

    def __init__(
        self, 
        vector_store: VectorStore, 
        embedding_model: EmbeddingModel,
        progress_callback: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize FileScanner with vector store and embedding model.
        
        Args:
            vector_store (VectorStore): Vector store for embeddings
            embedding_model (EmbeddingModel): Model for generating embeddings
            progress_callback (Optional[Callable[[str], None]]): Callback function for progress updates
        """
        self.logger = Logger("FileScanner").logger
        self.pdf_extractor = PDFExtractor()
        self.text_extractor = TextFileExtractor()
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.progress_callback = progress_callback

    def _update_progress(self, message: str) -> None:
        """
        Update progress through callback if available.
        
        Args:
            message (str): Progress message to send
        """
        if self.progress_callback:
            self.progress_callback(message)
        self.logger.info(message)

    def _extract_content(self, entry: Union[os.DirEntry, str]) -> Dict[str, Any]:
        """
        Extract text content from a file.
        
        Args:
            entry (Union[os.DirEntry, str]): File entry or path to process
            
        Returns:
            Dict[str, Any]: Dictionary containing extracted text and metadata from extractors
        """
        try:
            # Handle both string paths and DirEntry objects
            file_path = entry.path if isinstance(entry, os.DirEntry) else str(entry)
            mime_type, _ = mimetypes.guess_type(file_path)

            # Extract content based on file type
            if mime_type == "application/pdf":
                # PDF extractor handles both content and metadata
                return self.pdf_extractor.extract_content(file_path)
            elif self.text_extractor.is_supported_file_type(file_path):
                # Text extractor handles its own content extraction
                return self.text_extractor.extract_content(file_path)

            return {}

        except Exception as e:
            self.logger.error(f"Error extracting content from '{file_path}': {str(e)}")
            return {}
    
    def scan_directories_parallel(self, paths: List[str]) -> ScanResult:
        """
        Scan multiple directories in parallel and collect metadata.
        
        Args:
            paths (List[str]): List of directory paths to scan
            
        Returns:
            ScanResult: Results of the parallel scan operation
        """
        all_content: List[ScannedFile] = []
        errors: List[str] = []
        
        self._update_progress("Starting parallel directory scan...")
        
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            futures = {executor.submit(self.scan_directory, path): path for path in paths}
            
            for future in as_completed(futures):
                path = futures[future]
                try:
                    scan_result = future.result()
                    all_content.extend(scan_result.scanned_files)
                    errors.extend(scan_result.errors)
                    self._update_progress(f"Completed scanning directory: {path}")
                except Exception as e:
                    error_msg = f"Error scanning '{path}': {str(e)}"
                    self.logger.error(error_msg)
                    errors.append(error_msg)
                    self._update_progress(error_msg)
        
        self._update_progress(f"Scan complete. Processed {len(all_content)} files.")
        
        return ScanResult(
            scanned_files=all_content,
            scan_time=datetime.now(),
            scanned_paths=paths,
            errors=errors
        )

    def scan_directory(self, root_path: str) -> ScanResult:
        """
        Recursively scan directories and files starting from root_path using depth-first search.
        
        Args:
            root_path (str): The starting directory path to scan
            
        Returns:
            ScanResult: Results of the directory scan
        """
        scanned_files: List[ScannedFile] = []
        errors: List[str] = []

        try:
            root = Path(root_path).resolve()
            
            if not root.exists():
                error_msg = f"Error: Path '{root}' does not exist"
                self.logger.error(error_msg)
                errors.append(error_msg)
                self._update_progress(error_msg)
                return ScanResult([], datetime.now(), [root_path], [error_msg])
                
            self._update_progress(f"Scanning directory: {root}")
            
            with os.scandir(root) as entries:
                for entry in entries:
                    try:
                        abs_path = entry.path
                        file_extension = os.path.splitext(abs_path)[1]

                        if file_extension in self.SUPPORTED_EXTENSIONS:
                            self._update_progress(f"Processing file: {abs_path}")
                            file_content = self._extract_content(entry)

                            extracted_text = file_content.get("extracted_text", "")
                            metadata = file_content.get("metadata", {})

                            if extracted_text:
                                embedding = self.embedding_model.embed_text(extracted_text)
                                unique_id = self.generate_unique_id(abs_path)
                                
                                self.vector_store.add_embedding(
                                    vector=embedding,
                                    metadata=metadata,
                                    unique_id=unique_id
                                )
                                
                                scanned_files.append(ScannedFile(
                                    embedding=embedding,
                                    metadata=metadata,
                                    unique_id=unique_id
                                ))
                                self._update_progress(f"Processed file: {abs_path}")
                            
                        if entry.is_dir():
                            sub_result = self.scan_directory(abs_path)
                            scanned_files.extend(sub_result.scanned_files)
                            errors.extend(sub_result.errors)

                    except PermissionError:
                        error_msg = f"Permission denied: Cannot access '{entry.path}'"
                        self.logger.warning(error_msg)
                        errors.append(error_msg)
                        self._update_progress(error_msg)
                    except Exception as e:
                        error_msg = f"Error processing '{entry.path}': {str(e)}"
                        self.logger.error(error_msg)
                        errors.append(error_msg)
                        self._update_progress(error_msg)

        except Exception as e:
            error_msg = f"Error scanning '{root}': {str(e)}"
            self.logger.error(error_msg)
            errors.append(error_msg)
            self._update_progress(error_msg)

        return ScanResult(
            scanned_files=scanned_files,
            scan_time=datetime.now(),
            scanned_paths=[root_path],
            errors=errors
        )
    
    def scan_file(self, file_path: str) -> ScannedFile:
        """
        Scan a single file and generate its embedding.
        
        Args:
            file_path (str): Path to the file to scan
            
        Returns:
            ScannedFile: Scanned file data including embedding and metadata
        """
        try:
            file_extension = os.path.splitext(file_path)[1]
            if file_extension not in self.SUPPORTED_EXTENSIONS:
                return ScannedFile(
                    embedding=None,
                    metadata={},
                    unique_id=None
                )

            file_content = self._extract_content(file_path)
            if not file_content:
                self.logger.warning(f"No content extracted from file: {file_path}")
                return ScannedFile(
                    embedding=None,
                    metadata={},
                    unique_id=None
                )

            extracted_text = file_content.get("extracted_text", "")
            metadata = file_content.get("metadata", {})

            if not extracted_text:
                return ScannedFile(
                    embedding=None,
                    metadata={},
                    unique_id=None
                )

            embedding = self.embedding_model.embed_text(extracted_text)
            unique_id = self.generate_unique_id(file_path)

            # Remove existing embedding if it exists
            if self.vector_store.check_embedding_exists(unique_id):
                self.vector_store.remove_embedding(unique_id)
            
            # Add new embedding to vector store
            self.vector_store.add_embedding(
                vector=embedding,
                metadata=metadata,
                unique_id=unique_id
            )

            return ScannedFile(
                embedding=embedding,
                metadata=metadata,
                unique_id=unique_id
            )

        except Exception as e:
            self.logger.error(f"Error scanning '{file_path}': {str(e)}")
            return ScannedFile(
                embedding=None,
                metadata={},
                unique_id=None
            )

    def write_scan_results(self, scan_result: ScanResult, output_file: str = "scan_result.log") -> None:
        """
        Write scanning results to a file.
        
        Args:
            scan_result (ScanResult): The results of the scan operation
            output_file (str, optional): Output file name. Defaults to "scan_result.log"
        """
        try:
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            output_path = logs_dir / output_file
            
            with output_path.open('a', encoding='utf-8') as f:
                f.write("File System Scan Results\n\n")
                f.write(f"{'='*50}\n")
                f.write(f"Scan started at: {scan_result.scan_time.isoformat()}\n")
                f.write("Scanned directories:\n")
                for path in scan_result.scanned_paths:
                    f.write(f"- {path}\n")
                f.write(f"{'='*50}\n\n")

                for scanned_file in scan_result.scanned_files:
                    metadata = scanned_file.metadata
                    f.write(f"File: {metadata['filename']}\n")
                    f.write(f"Path: {metadata['file_path']}\n")
                    f.write(f"File Type: {metadata['file_type']}\n")
                    f.write(f"Size: {metadata['size_bytes']} bytes\n")
                    f.write(f"Created: {metadata['created_at']}\n")
                    f.write(f"Last Modified: {metadata['last_modified']}\n")
                    if 'page_count' in metadata and metadata['page_count']:
                        f.write(f"Pages: {metadata['page_count']}\n")
                    f.write(f"ID: {scanned_file.unique_id}\n")
                    f.write("-" * 30 + "\n")

                if scan_result.errors:
                    f.write("\nErrors encountered during scan:\n")
                    for error in scan_result.errors:
                        f.write(f"- {error}\n")

        except Exception as e:
            self.logger.error(f"Error writing to '{output_path}': {str(e)}")

    def generate_unique_id(self, file_path: str) -> int:
        """
        Generates a unique ID based on the file path.
        """
        return int(hashlib.sha256(file_path.encode('utf-8')).hexdigest(), 16) % (10 ** 8)
    
