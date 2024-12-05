from PyPDF2 import PdfReader
from typing import Dict, Any, Optional
import os

class PDFExtractor:
    """
    A class to parse PDF files and extract text and metadata.
    """
    
    # Define standard metadata keys that we want to extract
    METADATA_KEYS = {
        "/Title": "title",
        "/Author": "author",
        "/Subject": "subject",
        "/Producer": "producer",
        "/Creator": "creator",
    }

    def _extract_metadata(self, reader: PdfReader, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from PDF file.
        
        Args:
            reader (PdfReader): PyPDF2 reader object
            file_path (str): Path to the PDF file
            
        Returns:
            Dict[str, Any]: Dictionary containing PDF metadata
        """
        metadata = {}
        
        # Extract PDF-specific metadata
        pdf_info = reader.metadata if reader.metadata else {}
        for pdf_key, meta_key in self.METADATA_KEYS.items():
            value = pdf_info.get(pdf_key, None)

            if value:
                metadata[meta_key] = value
        
        # Add PDF structure information
        metadata.update({
            "page_count": len(reader.pages),
            "file_path": file_path,
            "filename": os.path.basename(file_path)
        })
        
        return metadata

    def extract_content(self, pdf_path: str, n_pages: Optional[int] = None) -> str:
        """
        Extract text and metadata from a PDF file.

        Args:
            pdf_path (str): The path to the PDF file.
            n_pages (Optional[int]): Number of pages to extract. Extracts all if None.

        Returns:
            str: Extracted text from the PDF including metadata.
        
        Raises:
            FileNotFoundError: If the PDF file does not exist.
            RuntimeError: If there is an error reading the PDF file.
        """
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file '{pdf_path}' not found.")
                
            reader = PdfReader(pdf_path)
            
            # Extract metadata
            metadata = self._extract_metadata(reader, pdf_path)
            
            # Extract text content
            text_content = []
            pages = reader.pages[:n_pages] if n_pages is not None else reader.pages
            
            for page_num, page in enumerate(pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text_content.append(f"--- Page {page_num} ---\n{page_text}")
            
            # Combine metadata and text into a structured format
            metadata_text = "\n".join(f"{key}: {value}" for key, value in metadata.items() if value is not None)
            
            full_text = f"{metadata_text}\n\n{''.join(text_content)}"
            
            return full_text.strip()

        except FileNotFoundError:
            raise
        except Exception as e:
            raise RuntimeError(f"Error processing PDF file '{pdf_path}': {str(e)}")