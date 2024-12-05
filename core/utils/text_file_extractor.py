from typing import Dict, Any, Optional
import os
import mimetypes
from ..utils.logger import Logger

class TextFileExtractor:
    """
    A class responsible for extracting metadata and content from text-based files.
    Supports file types: .txt, .md, and other plain text formats.
    """

    # Define standard metadata keys that we want to extract
    METADATA_KEYS = {
        "name": "title",
        "path": "path",
        "extension": "extension",
        "file_type": "file_type",
        "line_count": "line_count"
    }
    
    # Supported text file extensions and their mime types
    SUPPORTED_EXTENSIONS = {
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".log": "text/plain",
        ".csv": "text/csv",
        ".json": "application/json",
        ".xml": "application/xml",
    }
    
    def __init__(self):
        self.logger = Logger("FileScanner").logger

    def _extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Dictionary containing file metadata
            
        Raises:
            ValueError: If file type is not supported
            FileNotFoundError: If file doesn't exist
        """
        if not self.is_supported_file_type(file_path):
            raise ValueError(f"Unsupported file type: {file_path}")
            
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        try:
            mime_type, _ = mimetypes.guess_type(file_path)
            
            metadata = {
                self.METADATA_KEYS["name"]: os.path.basename(file_path),
                self.METADATA_KEYS["path"]: file_path,
                self.METADATA_KEYS["extension"]: os.path.splitext(file_path)[1],
                self.METADATA_KEYS["file_type"]: mime_type or "text/plain",
                self.METADATA_KEYS["line_count"]: self._count_lines(file_path)
            }
            
            return metadata
            
        except Exception as e:
            raise RuntimeError(f"Error extracting metadata from '{file_path}': {str(e)}")

    def extract_content(self, file_path: str, max_size_mb: Optional[float] = 50) -> str:
        """
        Extract content from a text file.
        
        Args:
            file_path: Path to the text file
            max_size_mb: Maximum file size in MB to process (default: 10MB)
            
        Returns:
            Extracted text content with metadata
            
        Raises:
            ValueError: If file is too large or type not supported
        """
        if not self.is_supported_file_type(file_path):
            raise ValueError(f"Unsupported file type: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Check file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if max_size_mb and file_size_mb > max_size_mb:
            raise ValueError(f"File too large ({file_size_mb:.2f}MB). Maximum size: {max_size_mb}MB")
            
        try:
            # Get metadata
            metadata = self._extract_metadata(file_path)
            metadata_text = "\n".join(f"{key}: {value}" for key, value in metadata.items() if value is not None)
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return f"{metadata_text}\n\n--- Content ---\n{content}"
                
        except Exception as e:
            raise RuntimeError(f"Error extracting content from '{file_path}': {str(e)}")

    def is_supported_file_type(self, file_path: str) -> bool:
        """
        Check if the file type is supported.
        
        Args:
            file_path: Path to the file
            
        Returns:
            bool: True if the file type is supported
        """
        extension = os.path.splitext(file_path)[1].lower()
        return extension in self.SUPPORTED_EXTENSIONS

    def _count_lines(self, file_path: str) -> int:
        """
        Count the number of lines in a text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            int: Number of lines in the file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        except Exception:
            return 0