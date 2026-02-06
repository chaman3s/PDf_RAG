from sentence_transformers import SentenceTransformer

class textEmbedding:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
    def txtToEmbedding(self,s):
        embeddings = []
        for t in s:
            emb = self.model.encode(t)
            embeddings.append(emb)
        return embeddings 
    def QueryToEmbedding(self,s):
        emb=self.model.encode(s)
        return emb