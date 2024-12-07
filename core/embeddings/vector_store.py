import faiss
import numpy as np
from typing import List, Dict, Any
import os
import pickle

class VectorStore:
    """
    Manages storage and retrieval of embeddings using FAISS with ID mapping.
    """
    def __init__(self, dimension: int, index_path: str = "faiss.index", id_map_path: str = "id_map.pkl"):
        self.dimension = dimension
        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))
        self.id_map_path = id_map_path
        self.id_map = {}  # Map from unique ID to metadata
        
        if os.path.exists(index_path) and os.path.exists(id_map_path):
            self.index = faiss.read_index(index_path)
            with open(id_map_path, 'rb') as f:
                self.id_map = pickle.load(f)
        else:
            faiss.write_index(self.index, index_path)
            self.save_id_map()

    def add_embedding(self, vector: List[float], metadata: Dict[str, Any], unique_id: int):
        """
        Adds a single embedding to the index with a unique ID.
        """
        np_vector = np.array(vector).astype('float32').reshape(1, -1)
        self.index.add_with_ids(np_vector, np.array([unique_id]))
        self.id_map[unique_id] = metadata
        self.save_id_map()

    def add_embeddings(self, vectors: List[List[float]], metadatas: List[Dict[str, Any]], unique_ids: List[int]):
        """
        Adds multiple embeddings to the index with unique IDs.
        """
        np_vectors = np.array(vectors).astype('float32')
        np_ids = np.array(unique_ids)
        self.index.add_with_ids(np_vectors, np_ids)
        for uid, metadata in zip(unique_ids, metadatas):
            self.id_map[uid] = metadata
        self.save_id_map()

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches for the top_k most similar embeddings to the query vector.
        """
        np_query = np.array(query_vector).astype('float32').reshape(1, -1)
        distances, indices = self.index.search(np_query, top_k)
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx in self.id_map:
                metadata = self.id_map[idx]
                results.append({
                    "metadata": metadata,
                    "distance": distance
                })
        return results

    def remove_embedding(self, unique_id: int):
        """
        Removes an embedding from the index based on its unique ID.
        """
        faiss_IDs = np.array([unique_id], dtype='int64')
        self.index.remove_ids(faiss_IDs)
        if unique_id in self.id_map:
            del self.id_map[unique_id]
        self.save_id_map()

    def save_index(self, index_path: str = "faiss.index"):
        """
        Saves the FAISS index to disk.
        """
        faiss.write_index(self.index, index_path)
        self.save_id_map()

    def load_index(self, index_path: str = "faiss.index", id_map_path: str = "id_map.pkl"):
        """
        Loads the FAISS index from disk.
        """
        if os.path.exists(index_path) and os.path.exists(id_map_path):
            self.index = faiss.read_index(index_path)
            with open(id_map_path, 'rb') as f:
                self.id_map = pickle.load(f)

    def save_id_map(self):
        """
        Saves the id_map to disk.
        """
        with open(self.id_map_path, 'wb') as f:
            pickle.dump(self.id_map, f)