import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Iterator
from datetime import datetime
import mimetypes
import hashlib

from ..utils.logger import Logger
from ..utils.pdf_extractor import PDFExtractor
from ..utils.text_file_extractor import TextFileExtractor
from ..embeddings.embedding_generator import EmbeddingModel
from ..embeddings.vector_store import VectorStore

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
    
    def _extract_content(self, entry: os.DirEntry) -> str:
        """
        Extract content from a file
        """
        try:
            file_path = entry.path
            mime_type, _ = mimetypes.guess_type(file_path)

            if mime_type == "application/pdf":
                return self.pdf_extractor.extract_content(file_path, 1)
            
            elif self.text_extractor.is_supported_file_type(file_path):
                return self.text_extractor.extract_content(file_path)
            
            else:
                return ""

        except Exception as e:
            self.logger.error(f"Error extracting content from '{file_path}': {str(e)}")
            return ""
    
    def scan_directories_parallel(self, paths: List[str]) -> List[Dict[str, Any]]:
        """
        Scan multiple directories in parallel and collect metadata.
        """
        all_content = []
        
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            # Submit scan tasks
            futures = {executor.submit(self.scan_directory, path): path for path in paths}
            
            # Iterate over completed futures
            for future in as_completed(futures):
                path = futures[future]
                try:
                    metadata = future.result()
                    all_content.extend(metadata)
                except Exception as e:
                    self.logger.error(f"Error scanning '{path}': {str(e)}")
        
        return all_content

    def scan_directory(self, root_path: str) -> List[str]:
        """
        Recursively scan directories and files starting from root_path using depth-first search.
        
        Args:
            root_path (str): The starting directory path to scan
        """
        all_content = []

        try:
            # Convert the root_path to absolute path and resolve any symbolic links
            root = Path(root_path).resolve()
            
            # Check if the path exists
            if not root.exists():
                self.logger.error(f"Error: Path '{root}' does not exist")
                return all_content
                
            # Use os.scandir() to iterate through directory entries
            with os.scandir(root) as entries:
                for entry in entries:
                    try:
                        # Get absolute path
                        abs_path = entry.path

                        #check for the file extension
                        file_extension = os.path.splitext(abs_path)[1]

                        if file_extension in self.SUPPORTED_EXTENSIONS:
                            self.logger.info(f"Extracting content from '{abs_path}'")
                            # Get file metadata
                            file_content = self._extract_content(entry)
                            
                            if file_content:
                                embedding = self.embedding_model.embed_text(file_content)
                                unique_id = self.generate_unique_id(abs_path)
                                metadata = {
                                         "file_path": abs_path,
                                         "filename": os.path.basename(abs_path),
                                         "last_modified": datetime.fromtimestamp(entry.stat().st_mtime).isoformat(),
                                     }
                                
                                # Add embedding to vector database
                                self.vector_store.add_embedding(embedding, metadata, unique_id)
                                
                                # Add embedding to list
                                all_content.append({
                                         "embedding": embedding,
                                         "metadata": metadata,
                                         "unique_id": unique_id
                                     })
                                
                            # If entry is a directory, recursively scan it
                            if entry.is_dir():
                                self.scan_directory(abs_path)

                    except PermissionError:
                        self.logger.warning(f"Permission denied: Cannot access '{entry.path}'")
                    except Exception as e:
                        self.logger.error(f"Error processing '{entry.path}': {str(e)}")

        except Exception as e:
            self.logger.error(f"Error scanning '{root}': {str(e)}")

        return all_content

    def write_scan_results(self, content_iterator: Iterator[Dict[str, Any]], paths: List[str], output_file: str = "scan_result.log") -> None:
        """
        Write scanning results to a file.
        """
        try:
            # Ensure logs directory exists
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            output_path = logs_dir / output_file
            
            with output_path.open('a', encoding='utf-8') as f:
                # Write header
                f.write("File System Scan Results\n\n")
                f.write(f"{'='*50}\n")
                f.write(f"Scan started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("Scanned directories:\n")
                for path in paths:
                    f.write(f"- {path}\n")
                f.write(f"{'='*50}\n\n")

                # Write metadata for each file
                for content in content_iterator:
                    f.write(f'{content} \n')

                f.write("\n")
    
        except Exception as e:
            self.logger.error(f"Error writing to '{output_path}': {str(e)}")

    def generate_unique_id(self, file_path: str) -> int:
        """
        Generates a unique ID based on the file path.
        """
        return int(hashlib.sha256(file_path.encode('utf-8')).hexdigest(), 16) % (10 ** 8) 



