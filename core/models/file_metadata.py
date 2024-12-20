from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

@dataclass
class FileMetadata:
    """
    Unified metadata structure for files in the system.
    
    Attributes:
        file_path (str): Full path to the file
        filename (str): Name of the file
        file_type (str): Type of the file (e.g. "pdf", "txt")
        size_bytes (int): Size of file in bytes
        created_at (str): ISO format timestamp of when the file was first processed
        last_modified (str): ISO format timestamp of last modification
        page_count (Optional[int]): Number of pages (for PDFs)
    """
    file_path: str
    filename: str
    file_type: str
    size_bytes: int
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_modified: str = field(default_factory=lambda: datetime.now().isoformat())
    page_count: Optional[int] = None


    def to_dict(self) -> dict:
        """Convert metadata to dictionary format."""
        return {
            "file_path": self.file_path,
            "filename": self.filename,
            "file_type": self.file_type,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
            "last_modified": self.last_modified
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FileMetadata":
        """Create metadata instance from dictionary."""
        return cls(**data) 