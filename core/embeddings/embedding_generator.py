from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Optional

from ..utils.logger import Logger

class EmbeddingModel:
    """
    A class for generating text embeddings using sentence transformers.
    
    Attributes:
        model: The sentence transformer model used for generating embeddings
        logger: Logger instance for error handling
        embeddings: The most recently generated embeddings
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        """
        Initialize the embedding model.

        Args:
            model_name: Name/path of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.logger = Logger("EmbeddingModel").logger
        self.embeddings: Optional[np.ndarray] = None

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embeddings for the input text.

        Args:
            text: The text to generate embeddings for

        Returns:
            A numpy array containing the text embeddings
            
        Raises:
            ValueError: If the input text is empty
        """
        if not text.strip():
            raise ValueError("Input text cannot be empty")
            
        try:
            embedding = self.model.encode(
                text,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False
            )

            self.embeddings = embedding.reshape(1, -1)
            return self.embeddings

        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            # Return zero vector with correct dimensions as fallback
            return np.zeros((1, self.model.get_sentence_embedding_dimension()))