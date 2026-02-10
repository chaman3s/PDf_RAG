import chromadb
from chromadb.config import Settings
import uuid

# Initialize persistent client
client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False)
)

collection = client.get_or_create_collection(name="pdf_chunks")

def dump_chunks(chunks, embeddings, pdf_name):
    """
    Store chunks in ChromaDB with alignment safety and correct metadata types.
    """
    documents = []
    metadatas = []
    ids = []
    valid_embeddings = []
    
    # 1. Zip chunks and embeddings to ensure they stay synchronized
    # If lengths differ, zip stops at the shortest, preventing index errors
    if len(chunks) != len(embeddings):
        print(f"CRITICAL WARNING: Chunk count ({len(chunks)}) and Embedding count ({len(embeddings)}) do not match!")
        # We process only up to the minimum length to avoid crashes
        
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        
        # Extract Content
        text_content = chunk.get("content", "") if isinstance(chunk, dict) else str(chunk)
        
        # 2. Safety Check: Skip empty chunks
        if not text_content or not text_content.strip():
            continue
            
        # Extract Metadata
        meta = {
            "source": pdf_name,
            "chunk_index": i
        }
        
        if isinstance(chunk, dict):
            # 3. FIX: Convert 'pages' list to string for ChromaDB
            meta_list = chunk.get("metadata", [])
            if meta_list:
                # Extract unique page numbers
                page_nums = sorted(list(set(
                    m.get("page") for m in meta_list if isinstance(m, dict) and m.get("page")
                )))
                
                if page_nums:
                    # Store as comma-separated string (e.g., "1,2")
                    meta["pages"] = ",".join(map(str, page_nums))
                    meta["start_page"] = int(page_nums[0]) # Useful for sorting
            
            if "token_count" in chunk:
                meta["token_count"] = int(chunk["token_count"])

        documents.append(text_content)
        metadatas.append(meta)
        valid_embeddings.append(embedding)
        
        # Create a unique ID (Prevent collisions if re-running)
        ids.append(f"{pdf_name}_{i}_{str(uuid.uuid4())[:8]}")

    # 4. Batch Insert (Chroma handles large batches well, but explicit batching is safer for massive docs)
    if documents:
        try:
            collection.add(
                ids=ids,
                documents=documents,
                embeddings=valid_embeddings,
                metadatas=metadatas
            )
            print(f"Successfully added {len(documents)} chunks to ChromaDB.")
        except Exception as e:
            print(f"Error adding to ChromaDB: {e}")
            
    return len(documents)

def retrieve_chunks(query_embedding, n_results=3):
    """
    Query the database.
    """
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results
    except Exception as e:
        print(f"Query failed: {e}")
        return None

def reset_collection():
    try:
        client.delete_collection("pdf_chunks")
    except:
        pass  # Collection might not exist
    global collection
    collection = client.get_or_create_collection(name="pdf_chunks")
    print("Collection reset successfully")