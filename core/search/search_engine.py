from typing import List, Dict, Any

from core.embeddings.vector_store import VectorStore
from core.embeddings.embedding_generator import EmbeddingModel

class SearchEngine:
    def __init__(self, vector_store: VectorStore, embedding_model: EmbeddingModel):
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_model.embed_text(query)
        results = self.vector_store.search(query_embedding, top_k)
        return results
