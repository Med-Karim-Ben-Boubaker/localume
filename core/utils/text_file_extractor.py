"""
Module for extracting content and metadata from text-based files.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import os
from datetime import datetime
from pathlib import Path

from ..models.file_metadata import FileMetadata

@dataclass
class TextContent:
    """
    Data class representing extracted text content and metadata.
    """
    _metadata: FileMetadata
    _extracted_text: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert the TextContent object to a dictionary."""
        return {
            "metadata": asdict(self._metadata),
            "extracted_text": self._extracted_text
        }

    @property
    def extracted_text(self) -> str:
        """Get the extracted text content."""
        return self._extracted_text
    
    @property
    def metadata(self) -> FileMetadata:
        """Get the complete metadata."""
        return self._metadata

class TextFileExtractor:
    """
    A class to parse text files and extract content and metadata.
    Supports file types: .txt, .md, and other plain text formats.
    """
    
    def _extract_metadata(self, file_path: str) -> FileMetadata:
        """
        Extract metadata from a text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            FileMetadata: Object containing text file metadata
            
        Raises:
            FileNotFoundError: If the file does not exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        file_path = Path(file_path)
        stats = file_path.stat()
        
        return FileMetadata(
            filename=file_path.name,
            file_path=str(file_path),
            file_type=file_path.suffix.lstrip('.'),
            size_bytes=stats.st_size,
            created_at=datetime.fromtimestamp(stats.st_ctime).isoformat(),
            last_modified=datetime.fromtimestamp(stats.st_mtime).isoformat(),
            page_count=None
        )

    def extract_content(self, file_path: str, max_size_mb: Optional[float] = 50) -> Dict[str, Any]:
        """
        Extract text and metadata from a text file.
        
        Args:
            file_path: Path to the text file
            max_size_mb: Maximum file size in MB to process (default: 50MB)
            
        Returns:
            Dict[str, Any]: Dictionary containing metadata and extracted text
            
        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If file is too large or type not supported
            RuntimeError: If there is an error reading the file
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Check file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if max_size_mb and file_size_mb > max_size_mb:
            raise ValueError(f"File too large ({file_size_mb:.2f}MB). Maximum size: {max_size_mb}MB")
            
        try:
            # Extract metadata
            metadata = self._extract_metadata(file_path)
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # Create TextContent object
            text_content = TextContent(
                _metadata=metadata,
                _extracted_text=content
            )
            
            return text_content.to_dict()
                
        except Exception as e:
            raise RuntimeError(f"Error extracting content from '{file_path}': {str(e)}")

    def _count_lines(self, file_path: str) -> int:
        """
        Count the number of lines in a text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            int: Number of lines in the file
            
        Raises:
            RuntimeError: If there is an error reading the file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        except Exception as e:
            raise RuntimeError(f"Error counting lines in '{file_path}': {str(e)}")

def extract_text_data(file_path: str, max_size_mb: Optional[float] = None) -> Dict[str, Any]:
    """
    A simplified function to extract data from a text file.

    Args:
        file_path: Path to the text file
        max_size_mb: Maximum file size in MB to process

    Returns:
        Dict[str, Any]: Dictionary containing metadata and extracted text

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If file is too large
        RuntimeError: If there is an error reading the file
    """
    text_extractor = TextFileExtractor()
    return text_extractor.extract_content(file_path, max_size_mb)