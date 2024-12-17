import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Iterator
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

    def __init__(self, vector_store: VectorStore, embedding_model: EmbeddingModel):	
        self.logger = Logger("FileScanner").logger
        self.pdf_extractor =  PDFExtractor()
        self.text_extractor = TextFileExtractor()
        self.embedding_model = embedding_model
        self.vector_store = vector_store
    
    def _extract_text(self, entry: os.DirEntry) -> str:
        """
        Extract text from a file
        """
        try:
            file_path = entry.path
            mime_type, _ = mimetypes.guess_type(file_path)

            if mime_type == "application/pdf":
                pdf_content = self.pdf_extractor.extract_content(file_path, n_pages=1)
                return pdf_content.extracted_text
            
            elif self.text_extractor.is_supported_file_type(file_path):
                return self.text_extractor.extract_content(file_path)
            
            else:
                return ""

        except Exception as e:
            self.logger.error(f"Error extracting content from '{file_path}': {str(e)}")
            return ""
    
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
        
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            futures = {executor.submit(self.scan_directory, path): path for path in paths}
            
            for future in as_completed(futures):
                path = futures[future]
                try:
                    scan_result = future.result()
                    all_content.extend(scan_result.scanned_files)
                    errors.extend(scan_result.errors)
                except Exception as e:
                    error_msg = f"Error scanning '{path}': {str(e)}"
                    self.logger.error(error_msg)
                    errors.append(error_msg)
        
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
                return ScanResult([], datetime.now(), [root_path], [error_msg])
                
            with os.scandir(root) as entries:
                for entry in entries:
                    try:
                        abs_path = entry.path
                        file_extension = os.path.splitext(abs_path)[1]

                        if file_extension in self.SUPPORTED_EXTENSIONS:
                            print(f"Extracting content from '{abs_path}'")
                            file_content = self._extract_text(entry)
                            
                            if file_content:
                                embedding = self.embedding_model.embed_text(file_content)
                                unique_id = self.generate_unique_id(abs_path)
                                
                                metadata = {
                                    "file_path": abs_path,
                                    "filename": os.path.basename(abs_path),
                                    "last_modified": datetime.fromtimestamp(entry.stat().st_mtime).isoformat()
                                }
                                
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
                            
                        if entry.is_dir():
                            sub_result = self.scan_directory(abs_path)
                            scanned_files.extend(sub_result.scanned_files)
                            errors.extend(sub_result.errors)

                    except PermissionError:
                        error_msg = f"Permission denied: Cannot access '{entry.path}'"
                        self.logger.warning(error_msg)
                        errors.append(error_msg)
                    except Exception as e:
                        error_msg = f"Error processing '{entry.path}': {str(e)}"
                        self.logger.error(error_msg)
                        errors.append(error_msg)

        except Exception as e:
            error_msg = f"Error scanning '{root}': {str(e)}"
            self.logger.error(error_msg)
            errors.append(error_msg)

        return ScanResult(
            scanned_files=scanned_files,
            scan_time=datetime.now(),
            scanned_paths=[root_path],
            errors=errors
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
                    f.write(f"Last Modified: {metadata['last_modified']}\n")
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
    
    def get_file_metadata(self, entry: os.DirEntry) -> Dict[str, Any]:
        """
        Get file metadata
        """
        return {
            "filename": os.path.basename(entry.path)
        }


