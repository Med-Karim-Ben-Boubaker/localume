from typing import List, Optional
import numpy as np

from ..embeddings.vector_store import VectorStore, SearchResult
from ..embeddings.embedding_generator import EmbeddingModel
from ..llm.service import GeminiService

class SearchEngine:
    def __init__(
        self, 
        vector_store: VectorStore, 
        embedding_model: EmbeddingModel,
        gemini_service: Optional[GeminiService] = None
    ) -> None:
        """
        Initialize the search engine.

        Args:
            vector_store (VectorStore): Vector store for similarity search
            embedding_model (EmbeddingModel): Model for generating embeddings
            gemini_service (Optional[GeminiService]): Service for query optimization
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.gemini_service = gemini_service

    def search(self, query: str, top_k: int = 5, optimize_query: bool = True) -> List[SearchResult]:
        """
        Search for documents similar to the query text.

        Args:
            query (str): The search query text
            top_k (int, optional): Number of results to return. Defaults to 5.

        Returns:
            List[SearchResult]: List of search results with metadata and distances
        """
        try:
            # Optimize query if Gemini service is available and optimize_query is True
            if self.gemini_service and optimize_query:
                query = self.gemini_service.optimize_search_query(query)

            print(f"Optimized query: {query}")

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