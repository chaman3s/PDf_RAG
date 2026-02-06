def merge_pages(structured_pages):
    corpus_parts = []

    for page in structured_pages:
        if not page["content"]:
            continue

        separator = f"\n\n--- PAGE {page['page']} ---\n\n"
        corpus_parts.append(separator + page["content"])

    return "".join(corpus_parts)
