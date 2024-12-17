from typing import List
import numpy as np

from core.embeddings.vector_store import VectorStore, SearchResult
from core.embeddings.embedding_generator import EmbeddingModel

class SearchEngine:
    """
    Search engine that combines embedding generation and vector search capabilities.
    
    This class provides a high-level interface for:
    - Converting text queries to embeddings
    - Performing similarity searches using the vector store
    """

    def __init__(self, vector_store: VectorStore, embedding_model: EmbeddingModel):
        """
        Initialize the search engine.

        Args:
            vector_store (VectorStore): Vector store for similarity search
            embedding_model (EmbeddingModel): Model for generating embeddings
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Search for documents similar to the query text.

        Args:
            query (str): The search query text
            top_k (int, optional): Number of results to return. Defaults to 5.

        Returns:
            List[SearchResult]: List of search results with metadata and distances

        Raises:
            RuntimeError: If search operation fails
        """
        try:
            # Generate embedding for the query text
            query_embedding: np.ndarray = self.embedding_model.embed_text(query)
            
            # Perform vector similarity search
            results: List[SearchResult] = self.vector_store.search(
                query_vector=query_embedding,
                top_k=top_k
            )
            
            return results
            
        except Exception as e:
            raise RuntimeError(f"Search operation failed: {str(e)}")