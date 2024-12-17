from dataclasses import dataclass, field
from pathlib import Path
import faiss
import numpy as np
from typing import List, Dict, Any, Optional, Union
import os
import pickle
from datetime import datetime

@dataclass(frozen=True)
class SearchResult:
    """
    Immutable data class representing a single search result from the vector store.
    
    Attributes:
        metadata (Dict[str, Any]): Metadata associated with the search result
        distance (float): Distance/similarity score
        unique_id (int): Unique identifier for the result
    """
    metadata: Dict[str, Any]
    distance: float
    unique_id: int

@dataclass
class VectorMetadata:
    """
    Data class representing metadata for a vector in the store.
    
    Attributes:
        file_path (str): Path to the source file
        filename (str): Name of the file
        created_at (str): Timestamp when the vector was added
        last_modified (str): Last modification timestamp
    """
    file_path: str
    filename: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_modified: str = field(default_factory=lambda: datetime.now().isoformat())

class VectorStore:
    """
    Manages storage and retrieval of embeddings using FAISS with ID mapping.
    
    This class provides functionality to:
    - Add and remove embeddings with associated metadata
    - Search for similar vectors
    - Save and load the index to/from disk
    """

    def __init__(
        self, 
        dimension: int, 
        index_path: Union[str, Path] = "faiss.index", 
        id_map_path: Union[str, Path] = "id_map.pkl"
    ) -> None:
        """
        Initialize the vector store.

        Args:
            dimension (int): Dimensionality of the vectors
            index_path (Union[str, Path]): Path to save/load the FAISS index
            id_map_path (Union[str, Path]): Path to save/load the ID mapping

        Raises:
            ValueError: If dimension is less than 1
        """
        if dimension < 1:
            raise ValueError("Dimension must be a positive integer")

        self.dimension = dimension
        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))
        self.id_map_path = Path(id_map_path)
        self.index_path = Path(index_path)
        self.id_map: Dict[int, VectorMetadata] = {}

        self._initialize_store()

    def _initialize_store(self) -> None:
        """Initialize the store by loading existing index and ID map or creating new ones."""
        if self.index_path.exists() and self.id_map_path.exists():
            self._load_existing_store()
        else:
            self._create_new_store()

    def _load_existing_store(self) -> None:
        """Load existing index and ID map from disk."""
        try:
            self.index = faiss.read_index(str(self.index_path))
            with open(self.id_map_path, 'rb') as f:
                self.id_map = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load existing store: {str(e)}")

    def _create_new_store(self) -> None:
        """Create and save new index and ID map."""
        try:
            faiss.write_index(self.index, str(self.index_path))
            self.save_id_map()
        except Exception as e:
            raise RuntimeError(f"Failed to create new store: {str(e)}")

    def add_embedding(self, vector: np.ndarray, metadata: Dict[str, Any], unique_id: int) -> None:
        """
        Add an embedding to the index with unique ID.

        Args:
            vector (np.ndarray): The embedding vector to add as a numpy array
            metadata (Dict[str, Any]): Metadata associated with the vector
            unique_id (int): Unique identifier for the vector

        Raises:
            ValueError: If vector dimension doesn't match index dimension
            RuntimeError: If failed to add embedding
        """
        try:
            # Ensure vector is 2D array with shape (1, dimension)
            np_vector = vector.astype(np.float32).reshape(1, -1)
            if np_vector.shape[1] != self.dimension:
                raise RuntimeError(
                    f"Vector dimension {np_vector.shape[1]} does not match index dimension {self.dimension}"
                )
            
            np_id = np.array([unique_id], dtype=np.int64)
            self.index.add_with_ids(np_vector, np_id)
            
            vector_metadata = VectorMetadata(
                file_path=metadata.get("file_path", ""),
                filename=metadata.get("filename", ""),
                last_modified=datetime.now().isoformat()
            )
            
            self.id_map[unique_id] = vector_metadata
            self.save_id_map()

        except Exception as e:
            raise RuntimeError(f"Failed to add embedding: {str(e)}")

    def search(self, query_vector: List[float], top_k: int = 5) -> List[SearchResult]:
        """
        Search for the top_k most similar embeddings to the query vector.

        Args:
            query_vector (List[float]): The query vector to search for
            top_k (int): Number of results to return (default: 5)

        Returns:
            List[SearchResult]: List of search results with metadata and distances

        Raises:
            ValueError: If query_vector dimension doesn't match index dimension
            RuntimeError: If search operation fails
        """
        try:
            np_query = np.array(query_vector, dtype=np.float32).reshape(1, -1)
            if np_query.shape[1] != self.dimension:
                raise ValueError("Query vector dimension mismatch")

            distances, indices = self.index.search(np_query, top_k)
            
            results: List[SearchResult] = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx != -1 and idx in self.id_map:  # -1 indicates no match found
                    metadata = self.id_map[idx]
                    metadata_dict = {
                        "file_path": metadata.file_path,
                        "filename": metadata.filename,
                        "created_at": metadata.created_at,
                        "last_modified": metadata.last_modified
                    }
                    results.append(SearchResult(
                        metadata=metadata_dict,
                        distance=float(distance),
                        unique_id=int(idx)
                    ))
            return results

        except Exception as e:
            raise RuntimeError(f"Search operation failed: {str(e)}")

    def remove_embedding(self, unique_id: int) -> None:
        """
        Remove an embedding from the index based on its unique ID.

        Args:
            unique_id (int): The unique identifier of the embedding to remove

        Raises:
            RuntimeError: If removal operation fails
        """
        try:
            faiss_IDs = np.array([unique_id], dtype=np.int64)
            self.index.remove_ids(faiss_IDs)
            self.id_map.pop(unique_id, None)
            self.save_id_map()
        except Exception as e:
            raise RuntimeError(f"Failed to remove embedding: {str(e)}")

    def save_id_map(self) -> None:
        """
        Save the ID map to disk.

        Raises:
            RuntimeError: If saving fails
        """
        try:
            with open(self.id_map_path, 'wb') as f:
                pickle.dump(self.id_map, f)
        except Exception as e:
            raise RuntimeError(f"Failed to save ID map: {str(e)}")