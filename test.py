# test_fix.py
import sys
sys.path.append('.')  # Add current directory to path

# Test 1: Check chunk format
from pdf_pipeline.ingest import get_pdf_chunks
chunks = get_pdf_chunks("Resume.pdf")
print(f"Got {len(chunks)} chunks")

# Check first chunk
if chunks:
    print("\nFirst chunk structure:")
    print(f"Type: {type(chunks[0])}")
    if isinstance(chunks[0], dict):
        for key in chunks[0]:
            print(f"  {key}: {type(chunks[0][key])}")
        print(f"  Content preview: {chunks[0].get('content', '')[:100]}...")

# Test 2: Check embedding
from Embedding import TextEmbedding as te
embedder = te.textEmbedding()
test_texts = ["This is a test", "Another test"]
embeddings = embedder.txtToEmbedding(test_texts)
print(f"\nEmbedding test: {len(embeddings)} embeddings, each of length {len(embeddings[0]) if embeddings else 0}")

print("\nIf all tests pass, run main.py")