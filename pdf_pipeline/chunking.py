def estimate_tokens(text):
    return int(len(text.split()) * 1.3)


def build_chunks(paragraphs, max_tokens=800, overlap_ratio=0.15):
    chunks = []
    current_chunk = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = estimate_tokens(para)

        if current_tokens + para_tokens > max_tokens:
            chunks.append("\n\n".join(current_chunk))

            overlap_tokens = int(max_tokens * overlap_ratio)
            overlap_chunk = []
            tokens = 0

            for p in reversed(current_chunk):
                t = estimate_tokens(p)
                if tokens + t > overlap_tokens:
                    break
                overlap_chunk.insert(0, p)
                tokens += t

            current_chunk = overlap_chunk
            current_tokens = tokens

        current_chunk.append(para)
        current_tokens += para_tokens

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks
