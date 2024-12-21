from typing import List, Optional
import numpy as np
import logging

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
        self.logger = logging.getLogger(__name__)

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Search for documents similar to the query.
        
        Args:
            query (str): The search query
            top_k (int): Number of results to return
            
        Returns:
            List[SearchResult]: List of search results
        """
        try:
            self.logger.info(f"Optimized query: {query}")
            
            # Generate embedding for query
            query_embedding = self.embedding_model.embed_text(query)
            
            # Get results from vector store
            results = self.vector_store.search(query_embedding, top_k)
            
            self.logger.info(f"Found {len(results)} results in vector store")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Search error: {str(e)}")
            return []