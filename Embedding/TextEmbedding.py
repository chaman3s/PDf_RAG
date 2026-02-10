from sentence_transformers import SentenceTransformer

class textEmbedding:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def txtToEmbedding(self, texts):
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def QueryToEmbedding(self, text):
        emb = self.model.encode([text], convert_to_numpy=True)
        return emb[0].tolist()
