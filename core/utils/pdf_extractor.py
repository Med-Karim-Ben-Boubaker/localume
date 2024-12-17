from dataclasses import dataclass
from PyPDF2 import PdfReader
from typing import Dict, Any, Optional
import os

@dataclass
class PDFContent:
    """
    Data class representing extracted PDF content and metadata.
    """
    _metadata: Dict[str, Any]
    _extracted_text: str

    @property
    def extracted_text(self) -> str:
        """Get the extracted text content."""
        return self._extracted_text
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get the complete metadata dictionary."""
        return self._metadata
    
    @property
    def filename(self) -> str:
        """Get the filename from metadata."""
        return self._metadata.get("filename", "Unknown")
    
    @property
    def file_path(self) -> str:
        """Get the file path from metadata."""
        return self._metadata.get("file_path", "Unknown")
    
    @property
    def page_count(self) -> int:
        """Get the page count from metadata."""
        return self._metadata.get("page_count", 0)
    
    @property
    def creation_date(self) -> str:
        """Get the creation date from metadata."""
        return self._metadata.get("creation_date", "Unknown")

class PDFExtractor:
    """
    A class to parse PDF files and extract text and metadata.
    """

    def _extract_metadata(self, reader: PdfReader, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from PDF file.
        
        Args:
            reader (PdfReader): PyPDF2 reader object
            file_path (str): Path to the PDF file
            
        Returns:
            Dict[str, Any]: Dictionary containing PDF metadata
        """
        pdf_info = reader.metadata if reader.metadata else {}
        creation_date = pdf_info.get("/CreationDate", "Unknown")
        
        return {
            "filename": os.path.basename(file_path),
            "file_path": file_path,
            "page_count": len(reader.pages),
            "creation_date": creation_date
        }

    def extract_content(self, pdf_path: str, n_pages: Optional[int] = None) -> PDFContent:
        """
        Extract text and metadata from a PDF file.

        Args:
            pdf_path (str): The path to the PDF file
            n_pages (Optional[int]): Number of pages to extract. Extracts all if None

        Returns:
            PDFContent: Object containing metadata and extracted text

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
            
            return PDFContent(
                _metadata=metadata,
                _extracted_text="".join(extracted_text).strip()
            )

        except Exception as e:
            raise RuntimeError(f"Error processing PDF file '{pdf_path}': {str(e)}")

def extract_pdf_data(pdf_path: str, n_pages: Optional[int] = None) -> PDFContent:
    """
    A simplified function to extract data from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file
        n_pages (Optional[int]): Number of pages to extract. Extracts all if None

    Returns:
        PDFContent: Object containing metadata and extracted text

    Raises:
        FileNotFoundError: If the PDF file does not exist
        RuntimeError: If there is an error reading the PDF file
    """
    pdf_extractor = PDFExtractor()
    return pdf_extractor.extract_content(pdf_path, n_pages)