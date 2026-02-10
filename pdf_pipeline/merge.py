def merge_pages(structured_pages):
    """
    Merges pages into a single string. 
    WARNING: usage of this function discards page-level metadata.
    """
    corpus_parts = []

    for page in structured_pages:
        # Check for None or empty strings
        if not page.get("content"):
            continue

        corpus_parts.append(page["content"])

    # Fix: Use double newline to separate pages clearly
    return "\n\n".join(corpus_parts)