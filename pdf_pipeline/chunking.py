import tiktoken

def get_encoder(model="cl100k_base"):
    try:
        return tiktoken.get_encoding(model)
    except:
        return tiktoken.get_encoding("cl100k_base")

def build_chunks(paragraphs, max_tokens=300, overlap_tokens=30):
    encoder = get_encoder()
    
    chunks = []
    current_chunk_text = []
    current_metadata = []
    current_tokens_count = 0
    
    for para_data in paragraphs:
        # Normalize input
        if isinstance(para_data, dict):
            para_text = para_data["text"]
            para_meta = {"page": para_data.get("page", 0)}
        else:
            para_text = para_data
            para_meta = {}
            
        # 1. Encode the paragraph ONCE (Fast)
        token_ids = encoder.encode(para_text)
        para_token_count = len(token_ids)
        
        # 2. Handle Oversized Paragraphs (Token Slicing Strategy)
        if para_token_count > max_tokens:
            # If we have data in buffer, flush it first
            if current_chunk_text:
                chunks.append({
                    "content": "\n\n".join(current_chunk_text),
                    "metadata": current_metadata,
                    "token_count": current_tokens_count
                })
                current_chunk_text = []
                current_metadata = []
                current_tokens_count = 0

            # Slice the long paragraph directly using token IDs
            for i in range(0, para_token_count, max_tokens - overlap_tokens):
                chunk_ids = token_ids[i : i + max_tokens]
                chunk_text = encoder.decode(chunk_ids)
                
                # If this isn't the last slice, it effectively "fills" a chunk
                chunks.append({
                    "content": chunk_text,
                    "metadata": [para_meta], # accurate metadata
                    "token_count": len(chunk_ids)
                })
            continue

        # 3. Standard Paragraph Handling
        if current_tokens_count + para_token_count > max_tokens:
            # Commit current chunk
            chunks.append({
                "content": "\n\n".join(current_chunk_text),
                "metadata": current_metadata.copy(),
                "token_count": current_tokens_count
            })
            
            # Create Overlap (Backtracking strategy)
            overlap_buffer_text = []
            overlap_buffer_meta = []
            overlap_running_count = 0
            
            # Walk backwards through current chunk to find overlap text
            for i in range(len(current_chunk_text)-1, -1, -1):
                p_text = current_chunk_text[i]
                p_tokens = len(encoder.encode(p_text)) # acceptable cost here
                
                if overlap_running_count + p_tokens > overlap_tokens:
                    break
                    
                overlap_buffer_text.insert(0, p_text)
                overlap_buffer_meta.insert(0, current_metadata[i])
                overlap_running_count += p_tokens
            
            # Reset current chunk to just the overlap + new paragraph
            current_chunk_text = overlap_buffer_text
            current_metadata = overlap_buffer_meta
            current_tokens_count = overlap_running_count

        # Add new paragraph
        current_chunk_text.append(para_text)
        current_metadata.append(para_meta)
        current_tokens_count += para_token_count

    # 4. Flush remaining buffer
    if current_chunk_text:
        chunks.append({
            "content": "\n\n".join(current_chunk_text),
            "metadata": current_metadata,
            "token_count": current_tokens_count
        })
        
    return chunks