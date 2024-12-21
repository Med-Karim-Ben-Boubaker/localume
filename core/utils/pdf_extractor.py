from dataclasses import dataclass, asdict
from PyPDF2 import PdfReader
from typing import Dict, Any, Optional
import os
from datetime import datetime

from ..models.file_metadata import FileMetadata

@dataclass
class PDFContent:
    """
    Data class representing extracted PDF content and metadata.
    """
    _metadata: FileMetadata
    _extracted_text: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert the PDFContent object to a dictionary."""
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

class PDFExtractor:
    """
    A class to parse PDF files and extract text and metadata.
    """

    def _extract_metadata(self, reader: PdfReader, file_path: str) -> FileMetadata:
        """
        Extract metadata from PDF file and create FileMetadata object.
        
        Args:
            reader (PdfReader): PyPDF2 reader object
            file_path (str): Path to the PDF file
            
        Returns:
            FileMetadata: Object containing PDF metadata
        """
        pdf_info = reader.metadata if reader.metadata else {}
        creation_date = pdf_info.get("/CreationDate", "Unknown")
        
        return FileMetadata(
            filename=os.path.basename(file_path),
            file_path=file_path,
            file_type="pdf",
            size_bytes=os.path.getsize(file_path),
            created_at=creation_date,
            last_modified=datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
            page_count=len(reader.pages)
        )

    def extract_content(self, pdf_path: str, n_pages: Optional[int] = None) -> Dict[str, Any]:
        """
        Extract text and metadata from a PDF file.
        
        Args:
            pdf_path (str): The path to the PDF file
            n_pages (Optional[int]): Number of pages to extract. Extracts all if None
            
        Returns:
            Dict[str, Any]: Dictionary containing metadata and extracted text

        Raises:
            FileNotFoundError: If the PDF file does not exist
            RuntimeError: If there is an error reading the PDF file
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file '{pdf_path}' not found.")
            
        try:
            reader = PdfReader(pdf_path)
            metadata = self._extract_metadata(reader, pdf_path)
            
            extracted_text = []
            pages = reader.pages[:n_pages] if n_pages is not None else reader.pages
            
            for page_num, page in enumerate(pages, 1):
                page_text = page.extract_text()
                if page_text:
                    extracted_text.append(f"--- Page {page_num} ---\n{page_text}")
            
            content = PDFContent(
                _metadata=metadata,
                _extracted_text="".join(extracted_text).strip()
            )
            
            return content.to_dict()
                
        except Exception as e:
            raise RuntimeError(f"Error processing PDF file '{pdf_path}': {str(e)}")

def extract_pdf_data(pdf_path: str, n_pages: Optional[int] = None) -> Dict[str, Any]:
    """
    A simplified function to extract data from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file
        n_pages (Optional[int]): Number of pages to extract. Extracts all if None

    Returns:
        Dict[str, Any]: Dictionary containing metadata and extracted text

    Raises:
        FileNotFoundError: If the PDF file does not exist
        RuntimeError: If there is an error reading the PDF file
    """
    pdf_extractor = PDFExtractor()
    return pdf_extractor.extract_content(pdf_path, n_pages)