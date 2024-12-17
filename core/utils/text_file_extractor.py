from typing import Dict, Any, Optional
import os
import mimetypes
from ..utils.logger import Logger

class TextFileExtractor:
    """
    A class responsible for extracting metadata and content from text-based files.
    Supports file types: .txt, .md, and other plain text formats.
    """

    def _extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a text file.
        
        Args:
            file_path (str): The path to the text file from which metadata will be extracted.
            
        Returns:
            Dict[str, Any]: A dictionary containing the extracted file metadata, including:
                - filename: The name of the file.
                - file_path: The full path to the file.
                - line_count: The total number of lines in the file.
                - creation_time: The time the file was created.
                - file_extension: The extension of the file.
            
        Raises:
            ValueError: If the file type is not supported.
            FileNotFoundError: If the specified file does not exist.
        """ 
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        metadata = {
            "filename": os.path.basename(file_path),
            "file_path": file_path,
            "line_count": sum(1 for _ in open(file_path, 'r', encoding='utf-8')),
            "creation_time": os.path.getctime(file_path),
            "file_extension": os.path.splitext(file_path)[1],
        }

        return metadata
    
    
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
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return {
                "metadata": metadata,
                "content": content.strip()
            }
                
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