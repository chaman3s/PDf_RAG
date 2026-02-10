import pdfplumber

from normalize import normalize
from merge import merge_pages
from chunking import build_chunks
from storage import save_json

pdf_path = "data.pdf"


def extract_pages(pdf_path):
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = normalize(page.extract_text())
            pages.append({
                "page": i + 1,
                "content": text
            })

    return pages


def main():
    pages = extract_pages(pdf_path)

    merged_text = merge_pages(pages)

    paragraphs = [p for p in merged_text.split("\n\n") if p.strip()]

    chunks = build_chunks(paragraphs)

    print("Total chunks:", len(chunks))
    print(chunks[0][:500])
    output = {
        "pdf": pdf_path,
        "page_count": len(pages),
        "raw_pages": pages,
        "cleaned_pages": pages,
        "chunks": [
            {"chunk_id": i + 1, "content": chunk}
            for i, chunk in enumerate(chunks)
        ]
    }

    save_json(output, "output/parsed_output.json")

if __name__ == "__main__":
    main()
