from typing import List, Dict, Union, BinaryIO
from pathlib import Path
from pypdf import PdfReader


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text by removing extra spaces and empty lines.
    """
    if not text:
        return ""

    lines = text.splitlines()
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    return "\n".join(cleaned_lines)


def extract_text_from_pdf(
    pdf_file: Union[str, Path, BinaryIO],
    source_name: str = "uploaded_pdf"
) -> List[Dict]:
    """
    Extract text from a PDF page by page.

    Returns:
        A list of dictionaries containing:
        - source
        - page_number
        - text
        - character_count
    """
    reader = PdfReader(pdf_file)
    extracted_pages = []

    for page_index, page in enumerate(reader.pages):
        raw_text = page.extract_text()
        cleaned_text = clean_text(raw_text)

        if cleaned_text:
            extracted_pages.append(
                {
                    "source": source_name,
                    "page_number": page_index + 1,
                    "text": cleaned_text,
                    "character_count": len(cleaned_text),
                }
            )

    return extracted_pages


def get_pdf_summary(extracted_pages: List[Dict]) -> Dict:
    """
    Generate a basic summary of extracted PDF text.
    """
    total_pages_with_text = len(extracted_pages)
    total_characters = sum(page["character_count"] for page in extracted_pages)

    return {
        "pages_with_text": total_pages_with_text,
        "total_characters": total_characters,
    }