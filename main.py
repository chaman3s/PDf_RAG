from Embedding import TextEmbedding as te
from storage.chroma_store import dump_chunks, peek_collection
from pdf_pipeline.ingest import get_pdf_chunks


def main():
    chunks = get_pdf_chunks("data.pdf")


    embedder = te.textEmbedding()
    embeddings = embedder.txtToEmbedding(chunks)

    dump_chunks(
        chunks=chunks,
        embeddings=embeddings,
        pdf_name="test_pdf"
    )

    print(" Data successfully stored in ChromaDB", peek_collection())

if __name__ == "__main__":
    main()
