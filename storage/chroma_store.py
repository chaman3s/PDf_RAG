import chromadb
from chromadb.config import Settings


client = chromadb.Client(
    Settings(
        persist_directory="./chroma_db"
    )
)

collection = client.get_or_create_collection(
    name="pdf_chunks"
)


def dump_chunks(chunks, embeddings, pdf_name):
    ids = [f"{pdf_name}_{i}" for i in range(len(chunks))]

    metadatas = [
        {"pdf": pdf_name, "chunk_id": i}
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )
    
def peek_collection():
    return collection.peek()
