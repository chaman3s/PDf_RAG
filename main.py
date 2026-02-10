import os

from pdf_pipeline.ingest import get_pdf_chunks
from storage.chroma_store import dump_chunks, retrieve_chunks, reset_collection
from Embedding.TextEmbedding import TextEmbedding # Using the optimized class

def main():
    # 1. Setup
    pdf_path = "Resume.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"Error: File '{pdf_path}' not found.")
        return

    print("=== 1. Initialization ===")
    # Initialize optimized embedder (loads model once)
    embedder = TextEmbedding(model_name="all-MiniLM-L6-v2")
    
    # Reset DB for a clean start
    reset_collection()

    # 2. Ingestion Pipeline
    print(f"\n=== 2. Processing '{pdf_path}' ===")
    
    # A. Extract & Chunk
    raw_chunks = get_pdf_chunks(pdf_path)
    print(f"Extracted {len(raw_chunks)} raw chunks.")

    # B. Filter (CRITICAL STEP)
    # Remove empty chunks BEFORE embedding to ensure 1:1 alignment
    valid_chunks = []
    valid_texts = []
    
    for chunk in raw_chunks:
        # Handle both dict and string formats safely
        text = chunk.get("content", "") if isinstance(chunk, dict) else str(chunk)
        
        if text and text.strip():
            valid_chunks.append(chunk)
            valid_texts.append(text)
    
    print(f"Filtered down to {len(valid_chunks)} valid chunks.")

    # C. Embed
    print("Generating embeddings...")
    embeddings = embedder.generate_embeddings(valid_texts, batch_size=32)
    
    # D. Store
    dump_chunks(
        chunks=valid_chunks, 
        embeddings=embeddings, 
        pdf_name="resume_v1"
    )

    # 3. Interactive Chat Loop
    print("\n=== 3. Ready! Ask questions about your PDF (type 'exit' to quit) ===")
    
    while True:
        query = input("\nUser: ").strip()
        if query.lower() in ['exit', 'quit', 'q']:
            break
            
        if not query:
            continue

        # A. Embed Query
        query_vec = embedder.get_query_embedding(query)
        
        # B. Retrieve
        results = retrieve_chunks(query_vec, n_results=3)
        
        # C. Display Results
        print("\n--- Context Found ---")
        if results and results['documents']:
            for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                # Check for page number in metadata
                page_info = f" (Page {meta.get('pages', '?')})" if meta else ""
                
                print(f"\n[Result {i+1}{page_info}]:")
                # Preview first 200 chars to avoid wall of text
                preview = doc[:200].replace('\n', ' ') + "..."
                print(f"\"{preview}\"")
        else:
            print("No relevant information found.")

if __name__ == "__main__":
    main()