"""
Vector storage system using FAISS for the AI Knowledge Copilot.

This module provides a wrapper around FAISS (Facebook AI Similarity Search)
for efficient storage and retrieval of embedding vectors. It supports adding,
searching, deleting vectors, and persisting the index to disk.
"""
from typing import List, Tuple, Optional
import numpy as np
import faiss
import pickle
import os
from pathlib import Path


class VectorStore:
    """
    Wrapper around FAISS for vector storage and similarity search.
    
    This class provides a high-level interface for managing embedding vectors
    using FAISS. It supports efficient k-nearest neighbor search, vector
    addition/deletion, and index persistence.
    
    Attributes:
        dimension: Dimensionality of the embedding vectors
        index_type: Type of FAISS index to use (default: "Flat")
        index: The FAISS index instance
        id_map: Mapping from string IDs to integer indices in FAISS
        reverse_id_map: Mapping from integer indices to string IDs
    """
    
    def __init__(self, dimension: int, index_type: str = "Flat"):
        """
        Initialize the VectorStore with a FAISS index.
        
        Args:
            dimension: Dimensionality of embedding vectors (e.g., 384 for all-MiniLM-L6-v2)
            index_type: Type of FAISS index to use (default: "Flat" for IndexFlatL2)
                       Options:
                       - "Flat": Exact search using L2 distance (IndexFlatL2)
                       - "IVFFlat": Faster approximate search for large datasets
        
        Raises:
            ValueError: If dimension <= 0 or index_type is unsupported
        
        Validates:
            - Requirements 9.1: Create or load a FAISS index for vector storage
        """
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        
        if index_type not in ["Flat", "IVFFlat"]:
            raise ValueError(f"Unsupported index_type: {index_type}. Use 'Flat' or 'IVFFlat'")
        
        self.dimension = dimension
        self.index_type = index_type
        
        # Initialize FAISS index
        if index_type == "Flat":
            # IndexFlatL2 uses L2 (Euclidean) distance for exact search
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "IVFFlat":
            # For IVFFlat, we need a quantizer and number of clusters
            # This is more complex and typically used for production
            quantizer = faiss.IndexFlatL2(dimension)
            n_clusters = 100  # Default number of clusters
            self.index = faiss.IndexIVFFlat(quantizer, dimension, n_clusters)
        
        # Maintain mappings between string IDs and integer indices
        self.id_map: dict[str, int] = {}
        self.reverse_id_map: dict[int, str] = {}
        self._next_index = 0
    
    def add_vectors(self, vectors: np.ndarray, ids: List[str]) -> None:
        """
        Add embedding vectors to the FAISS index with associated IDs.
        
        This method inserts new vectors into the index and maintains the
        mapping between string IDs and FAISS integer indices.
        
        Args:
            vectors: Numpy array of shape (n, dimension) containing embeddings
            ids: List of string IDs corresponding to each vector
        
        Raises:
            ValueError: If vectors shape doesn't match (n, dimension) or
                       if len(ids) != len(vectors) or if any ID already exists
        
        Validates:
            - Requirements 2.5: Store embeddings in vector database
            - Requirements 9.2: Insert embeddings into FAISS index with IDs
        
        Example:
            >>> store = VectorStore(dimension=384)
            >>> vectors = np.random.rand(5, 384).astype('float32')
            >>> ids = ["chunk_1", "chunk_2", "chunk_3", "chunk_4", "chunk_5"]
            >>> store.add_vectors(vectors, ids)
        """
        # Validate inputs
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2D array, got shape {vectors.shape}")
        
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"vectors dimension {vectors.shape[1]} doesn't match "
                f"index dimension {self.dimension}"
            )
        
        if len(ids) != len(vectors):
            raise ValueError(
                f"Number of IDs ({len(ids)}) must match number of vectors ({len(vectors)})"
            )
        
        # Check for duplicate IDs
        for id_str in ids:
            if id_str in self.id_map:
                raise ValueError(f"ID '{id_str}' already exists in the index")
        
        # Convert to float32 if needed (FAISS requires float32)
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        
        # Add vectors to FAISS index
        self.index.add(vectors)
        
        # Update ID mappings
        for i, id_str in enumerate(ids):
            faiss_index = self._next_index + i
            self.id_map[id_str] = faiss_index
            self.reverse_id_map[faiss_index] = id_str
        
        self._next_index += len(ids)

    def search(
        self, 
        query_vector: np.ndarray, 
        k: int = 10,
        threshold: Optional[float] = None
    ) -> Tuple[List[float], List[str]]:
        """
        Perform k-nearest neighbor search for a query vector.
        
        This method finds the k most similar vectors in the index to the
        query vector using L2 distance. Optionally filters results by
        a distance threshold.
        
        Args:
            query_vector: Query embedding vector of shape (dimension,) or (1, dimension)
            k: Number of nearest neighbors to retrieve (default: 10)
            threshold: Optional maximum distance threshold for filtering results
        
        Returns:
            Tuple of (distances, ids):
            - distances: List of L2 distances to the k nearest neighbors
            - ids: List of string IDs corresponding to the nearest neighbors
        
        Raises:
            ValueError: If query_vector dimension doesn't match index dimension
                       or if k <= 0
        
        Validates:
            - Requirements 3.2: Perform similarity search in vector database
            - Requirements 3.3: Retrieve top-k most relevant chunks
            - Requirements 9.3: Use FAISS to find nearest neighbors efficiently
        
        Example:
            >>> store = VectorStore(dimension=384)
            >>> # ... add vectors ...
            >>> query = np.random.rand(384).astype('float32')
            >>> distances, ids = store.search(query, k=5)
            >>> print(f"Found {len(ids)} results")
        """
        # Validate inputs
        if k <= 0:
            raise ValueError("k must be positive")
        
        # Handle empty index
        if self.index.ntotal == 0:
            return [], []
        
        # Reshape query vector if needed
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        if query_vector.shape[1] != self.dimension:
            raise ValueError(
                f"query_vector dimension {query_vector.shape[1]} doesn't match "
                f"index dimension {self.dimension}"
            )
        
        # Convert to float32 if needed
        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype(np.float32)
        
        # Limit k to the number of vectors in the index
        k = min(k, self.index.ntotal)
        
        # Perform search
        distances, indices = self.index.search(query_vector, k)
        
        # Convert to lists (FAISS returns 2D arrays)
        distances = distances[0].tolist()
        indices = indices[0].tolist()
        
        # Map integer indices back to string IDs
        result_ids = []
        result_distances = []
        
        for dist, idx in zip(distances, indices):
            # FAISS returns -1 for missing results when k > ntotal
            if idx == -1:
                continue
            
            # Apply threshold filter if specified
            if threshold is not None and dist > threshold:
                continue
            
            # Get string ID from reverse mapping
            if idx in self.reverse_id_map:
                result_ids.append(self.reverse_id_map[idx])
                result_distances.append(dist)
        
        return result_distances, result_ids
    
    def delete_vectors(self, ids: List[str]) -> None:
        """
        Remove vectors from the index by their IDs.
        
        Note: FAISS IndexFlatL2 doesn't support efficient deletion.
        This implementation rebuilds the index without the deleted vectors.
        For production with frequent deletions, consider using IndexIDMap.
        
        Args:
            ids: List of string IDs to delete
        
        Raises:
            ValueError: If any ID doesn't exist in the index
        
        Validates:
            - Requirements 11.1: Remove chunks and embeddings from vector database
        
        Example:
            >>> store = VectorStore(dimension=384)
            >>> # ... add vectors ...
            >>> store.delete_vectors(["chunk_1", "chunk_3"])
        """
        # Validate that all IDs exist
        for id_str in ids:
            if id_str not in self.id_map:
                raise ValueError(f"ID '{id_str}' not found in the index")
        
        # If deleting all vectors, just reset the index
        if len(ids) == len(self.id_map):
            self._reset_index()
            return
        
        # Get indices to delete
        indices_to_delete = {self.id_map[id_str] for id_str in ids}
        
        # Collect vectors and IDs to keep
        vectors_to_keep = []
        ids_to_keep = []
        
        # Iterate through all current vectors
        for idx in range(self._next_index):
            if idx not in indices_to_delete and idx in self.reverse_id_map:
                # Reconstruct vector from index
                # Note: This is inefficient but necessary for IndexFlatL2
                # In production, consider storing vectors separately or using IndexIDMap
                id_str = self.reverse_id_map[idx]
                
                # We need to extract the vector from FAISS
                # For IndexFlatL2, we can access the internal data
                vector = self.index.reconstruct(int(idx))
                vectors_to_keep.append(vector)
                ids_to_keep.append(id_str)
        
        # Reset and rebuild index
        self._reset_index()
        
        if vectors_to_keep:
            vectors_array = np.array(vectors_to_keep, dtype=np.float32)
            self.add_vectors(vectors_array, ids_to_keep)
    
    def save_index(self, path: str) -> None:
        """
        Save the FAISS index and ID mappings to disk.
        
        This method persists both the FAISS index and the ID mappings
        to enable loading the index later without losing data.
        
        Args:
            path: Directory path where index files will be saved
        
        Raises:
            IOError: If unable to write to the specified path
        
        Validates:
            - Requirements 9.4: Persist changes to disk for durability
        
        Example:
            >>> store = VectorStore(dimension=384)
            >>> # ... add vectors ...
            >>> store.save_index("./data/faiss_index")
        """
        # Create directory if it doesn't exist
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_file = path_obj / "index.faiss"
        faiss.write_index(self.index, str(index_file))
        
        # Save metadata (ID mappings, dimension, index_type, next_index)
        metadata = {
            'id_map': self.id_map,
            'reverse_id_map': self.reverse_id_map,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'next_index': self._next_index
        }
        
        metadata_file = path_obj / "metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
    
    def load_index(self, path: str) -> None:
        """
        Load a FAISS index and ID mappings from disk.
        
        This method restores a previously saved index, including all
        vectors and ID mappings.
        
        Args:
            path: Directory path where index files are stored
        
        Raises:
            FileNotFoundError: If index files don't exist at the specified path
            ValueError: If loaded dimension doesn't match current dimension
        
        Validates:
            - Requirements 9.1: Create or load a FAISS index
            - Requirements 9.4: Persist changes to disk for durability
        
        Example:
            >>> store = VectorStore(dimension=384)
            >>> store.load_index("./data/faiss_index")
        """
        path_obj = Path(path)
        
        # Check if files exist
        index_file = path_obj / "index.faiss"
        metadata_file = path_obj / "metadata.pkl"
        
        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found: {index_file}")
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        # Load metadata
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        # Validate dimension matches
        if metadata['dimension'] != self.dimension:
            raise ValueError(
                f"Loaded dimension {metadata['dimension']} doesn't match "
                f"current dimension {self.dimension}"
            )
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_file))
        
        # Restore metadata
        self.id_map = metadata['id_map']
        self.reverse_id_map = metadata['reverse_id_map']
        self.index_type = metadata['index_type']
        self._next_index = metadata['next_index']
    
    def _reset_index(self) -> None:
        """
        Reset the index to empty state.
        
        This is an internal helper method used by delete_vectors.
        """
        # Recreate empty index
        if self.index_type == "Flat":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IVFFlat":
            quantizer = faiss.IndexFlatL2(self.dimension)
            n_clusters = 100
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, n_clusters)
        
        # Clear mappings
        self.id_map = {}
        self.reverse_id_map = {}
        self._next_index = 0
    
    def get_vector_count(self) -> int:
        """
        Get the total number of vectors in the index.
        
        Returns:
            Number of vectors currently stored in the index
        """
        return self.index.ntotal
    
    def get_vector_by_id(self, id_str: str) -> Optional[np.ndarray]:
        """
        Retrieve a vector by its string ID.
        
        Args:
            id_str: String ID of the vector to retrieve
        
        Returns:
            The vector as a numpy array, or None if ID not found
        """
        if id_str not in self.id_map:
            return None
        
        idx = self.id_map[id_str]
        vector = self.index.reconstruct(int(idx))
        return vector
