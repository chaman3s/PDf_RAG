import pdfplumber
import re
from .normalize import normalize # Assuming this handles unicode normalization
from .chunking import build_chunks

def extract_clean_paragraphs(pdf_path):
    """
    Extracts text page-by-page and yields cleaned paragraphs.
    Handles the common PDF issue where sentences are broken by newlines.
    """
    cleaned_paragraphs = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            # 1. Safety check for blank/image-only pages
            raw_text = page.extract_text()
            if not raw_text:
                continue

            # 2. visual-chunk extraction (simple heuristic)
            # We split by double newlines to find "blocks" of text
            raw_blocks = raw_text.split('\n\n')
            
            for block in raw_blocks:
                block = block.strip()
                if not block:
                    continue
                
                # 3. Heal line breaks within a paragraph
                # Replaces single newlines with space, but keeps the block together
                # This fixes: "This is a sentence\nthat was split." -> "This is a sentence that was split."
                cleaned_text = block.replace('\n', ' ')
                
                # Optional: Remove excess whitespace created by the join
                cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
                
                # 4. Apply your custom normalization (unicode, etc)
                if 'normalize' in globals():
                    cleaned_text = normalize(cleaned_text)

                cleaned_paragraphs.append({
                    "text": cleaned_text,
                    "page": i + 1
                })
                
    return cleaned_paragraphs

def get_pdf_chunks(pdf_path):
    """
    Orchestrates the extraction and chunking pipeline.
    """
    # 1. Extract clean, page-aware paragraphs
    paragraphs_data = extract_clean_paragraphs(pdf_path)
    
    # 2. Chunk them using the token-aware semantic chunker
    # Note: build_chunks returns a list of dicts: 
    # [{'content': '...', 'metadata': [...], 'token_count': 123}, ...]
    chunks = build_chunks(paragraphs_data)
    
    # 3. Return directly. Do not re-format or re-count tokens.
    return chunks