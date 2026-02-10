from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union, Optional

class TextEmbedding:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        """
        Initialize the embedding model.
        Args:
            model_name: The HuggingFace model string.
            device: 'cpu', 'cuda', or 'mps'. If None, auto-detects.
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded successfully. Dimension: {self.embedding_dim}")
    
    def generate_embeddings(self, texts: Union[str, List[str]], batch_size: int = 32) -> List[List[float]]:
        """
        Unified method for single or list of texts. 
        Auto-handles batching for large lists.
        """
        # 1. Input Normalization
        if isinstance(texts, str):
            texts = [texts]
            
        if not texts:
            return []

        # Filter empty strings safely
        valid_texts = [str(t) for t in texts if t and str(t).strip()]
        if not valid_texts:
            return []

        try:
            # 2. Batch Encoding (Much faster for large datasets)
            embeddings = self.model.encode(
                valid_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True # Crucial for Cosine Similarity
            )
            
            # 3. Return as list (standard python format)
            return embeddings.tolist()
            
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return []

    def get_query_embedding(self, text: str) -> List[float]:
        """Wrapper for single query embedding"""
        result = self.generate_embeddings(text)
        return result[0] if result else []

    @property
    def dimension(self) -> int:
        """Fast dimension check without inference"""
        return self.embedding_dim