import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from core.utils.pdf_extractor import PDFExtractor

pdf_parser = PDFExtractor()

result = pdf_parser.extract_content("tests/test_files/test.pdf", n_pages=2)
print("Metadata:", result["metadata"])
print("Extracted Text:", result["extracted_text"])
