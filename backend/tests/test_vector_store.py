"""
Unit tests for the VectorStore class.

This module tests the FAISS vector storage functionality including
adding, searching, deleting vectors, and persistence operations.
"""
import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from hypothesis import given, strategies as st
from src.vector_store import VectorStore


class TestVectorStoreInitialization:
    """Tests for VectorStore initialization."""
    
    def test_init_with_valid_dimension(self):
        """Test initialization with valid dimension."""
        store = VectorStore(dimension=384)
        assert store.dimension == 384
        assert store.index_type == "Flat"
        assert store.get_vector_count() == 0
    
    def test_init_with_invalid_dimension(self):
        """Test initialization with invalid dimension raises ValueError."""
        with pytest.raises(ValueError, match="dimension must be positive"):
            VectorStore(dimension=0)
        
        with pytest.raises(ValueError, match="dimension must be positive"):
            VectorStore(dimension=-10)
    
    def test_init_with_unsupported_index_type(self):
        """Test initialization with unsupported index type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported index_type"):
            VectorStore(dimension=384, index_type="InvalidType")


class TestVectorStoreAddVectors:
    """Tests for adding vectors to the store."""
    
    def test_add_single_vector(self):
        """Test adding a single vector."""
        store = VectorStore(dimension=384)
        vector = np.random.rand(1, 384).astype('float32')
        ids = ["chunk_1"]
        
        store.add_vectors(vector, ids)
        
        assert store.get_vector_count() == 1
        assert "chunk_1" in store.id_map
    
    def test_add_multiple_vectors(self):
        """Test adding multiple vectors at once."""
        store = VectorStore(dimension=384)
        vectors = np.random.rand(5, 384).astype('float32')
        ids = ["chunk_1", "chunk_2", "chunk_3", "chunk_4", "chunk_5"]
        
        store.add_vectors(vectors, ids)
        
        assert store.get_vector_count() == 5
        for id_str in ids:
            assert id_str in store.id_map
    
    def test_add_vectors_wrong_dimension(self):
        """Test adding vectors with wrong dimension raises ValueError."""
        store = VectorStore(dimension=384)
        vectors = np.random.rand(3, 512).astype('float32')  # Wrong dimension
        ids = ["chunk_1", "chunk_2", "chunk_3"]
        
        with pytest.raises(ValueError, match="doesn't match index dimension"):
            store.add_vectors(vectors, ids)
    
    def test_add_vectors_mismatched_ids(self):
        """Test adding vectors with mismatched number of IDs raises ValueError."""
        store = VectorStore(dimension=384)
        vectors = np.random.rand(3, 384).astype('float32')
        ids = ["chunk_1", "chunk_2"]  # Only 2 IDs for 3 vectors
        
        with pytest.raises(ValueError, match="Number of IDs.*must match"):
            store.add_vectors(vectors, ids)
    
    def test_add_duplicate_id(self):
        """Test adding vector with duplicate ID raises ValueError."""
        store = VectorStore(dimension=384)
        vector1 = np.random.rand(1, 384).astype('float32')
        store.add_vectors(vector1, ["chunk_1"])
        
        vector2 = np.random.rand(1, 384).astype('float32')
        with pytest.raises(ValueError, match="already exists"):
            store.add_vectors(vector2, ["chunk_1"])
    
    def test_add_vectors_auto_converts_dtype(self):
        """Test that vectors are automatically converted to float32."""
        store = VectorStore(dimension=384)
        vectors = np.random.rand(2, 384).astype('float64')  # float64
        ids = ["chunk_1", "chunk_2"]
        
        # Should not raise an error, auto-converts to float32
        store.add_vectors(vectors, ids)
        assert store.get_vector_count() == 2


class TestVectorStoreSearch:
    """Tests for searching vectors in the store."""
    
    def test_search_returns_correct_count(self):
        """Test that search returns correct number of results."""
        store = VectorStore(dimension=384)
        vectors = np.random.rand(10, 384).astype('float32')
        ids = [f"chunk_{i}" for i in range(10)]
        store.add_vectors(vectors, ids)
        
        query = np.random.rand(384).astype('float32')
        distances, result_ids = store.search(query, k=5)
        
        assert len(distances) == 5
        assert len(result_ids) == 5
    
    def test_search_with_k_greater_than_total(self):
        """Test search when k is greater than total vectors."""
        store = VectorStore(dimension=384)
        vectors = np.random.rand(3, 384).astype('float32')
        ids = ["chunk_1", "chunk_2", "chunk_3"]
        store.add_vectors(vectors, ids)
        
        query = np.random.rand(384).astype('float32')
        distances, result_ids = store.search(query, k=10)
        
        # Should return only 3 results
        assert len(distances) == 3
        assert len(result_ids) == 3
    
    def test_search_empty_index(self):
        """Test search on empty index returns empty results."""
        store = VectorStore(dimension=384)
        query = np.random.rand(384).astype('float32')
        
        distances, result_ids = store.search(query, k=5)
        
        assert len(distances) == 0
        assert len(result_ids) == 0
    
    def test_search_with_invalid_k(self):
        """Test search with invalid k raises ValueError."""
        store = VectorStore(dimension=384)
        query = np.random.rand(384).astype('float32')
        
        with pytest.raises(ValueError, match="k must be positive"):
            store.search(query, k=0)
        
        with pytest.raises(ValueError, match="k must be positive"):
            store.search(query, k=-5)
    
    def test_search_with_wrong_dimension(self):
        """Test search with wrong dimension query raises ValueError."""
        store = VectorStore(dimension=384)
        vectors = np.random.rand(5, 384).astype('float32')
        ids = [f"chunk_{i}" for i in range(5)]
        store.add_vectors(vectors, ids)
        
        query = np.random.rand(512).astype('float32')  # Wrong dimension
        
        with pytest.raises(ValueError, match="doesn't match index dimension"):
            store.search(query, k=3)
    
    def test_search_returns_valid_ids(self):
        """Test that search returns valid IDs that exist in the store."""
        store = VectorStore(dimension=384)
        vectors = np.random.rand(5, 384).astype('float32')
        ids = ["chunk_1", "chunk_2", "chunk_3", "chunk_4", "chunk_5"]
        store.add_vectors(vectors, ids)
        
        query = vectors[0]  # Use first vector as query
        distances, result_ids = store.search(query, k=3)
        
        # All returned IDs should be in the original ID list
        for result_id in result_ids:
            assert result_id in ids
    
    def test_search_with_threshold(self):
        """Test search with distance threshold filtering."""
        store = VectorStore(dimension=384)
        vectors = np.random.rand(5, 384).astype('float32')
        ids = [f"chunk_{i}" for i in range(5)]
        store.add_vectors(vectors, ids)
        
        # Use one of the stored vectors as query (distance should be ~0)
        query = vectors[0]
        distances, result_ids = store.search(query, k=5, threshold=0.1)
        
        # Should return at least the exact match
        assert len(result_ids) >= 1
        # All distances should be below threshold
        for dist in distances:
            assert dist <= 0.1


class TestVectorStoreDeleteVectors:
    """Tests for deleting vectors from the store."""
    
    def test_delete_single_vector(self):
        """Test deleting a single vector."""
        store = VectorStore(dimension=384)
        vectors = np.random.rand(5, 384).astype('float32')
        ids = [f"chunk_{i}" for i in range(5)]
        store.add_vectors(vectors, ids)
        
        store.delete_vectors(["chunk_2"])
        
        assert store.get_vector_count() == 4
        assert "chunk_2" not in store.id_map
    
    def test_delete_multiple_vectors(self):
        """Test deleting multiple vectors."""
        store = VectorStore(dimension=384)
        vectors = np.random.rand(5, 384).astype('float32')
        ids = [f"chunk_{i}" for i in range(5)]
        store.add_vectors(vectors, ids)
        
        store.delete_vectors(["chunk_1", "chunk_3", "chunk_4"])
        
        assert store.get_vector_count() == 2
        assert "chunk_1" not in store.id_map
        assert "chunk_3" not in store.id_map
        assert "chunk_4" not in store.id_map
    
    def test_delete_nonexistent_id(self):
        """Test deleting non-existent ID raises ValueError."""
        store = VectorStore(dimension=384)
        vectors = np.random.rand(3, 384).astype('float32')
        ids = ["chunk_1", "chunk_2", "chunk_3"]
        store.add_vectors(vectors, ids)
        
        with pytest.raises(ValueError, match="not found in the index"):
            store.delete_vectors(["chunk_999"])
    
    def test_delete_all_vectors(self):
        """Test deleting all vectors resets the index."""
        store = VectorStore(dimension=384)
        vectors = np.random.rand(5, 384).astype('float32')
        ids = [f"chunk_{i}" for i in range(5)]
        store.add_vectors(vectors, ids)
        
        store.delete_vectors(ids)
        
        assert store.get_vector_count() == 0
        assert len(store.id_map) == 0


class TestVectorStorePersistence:
    """Tests for saving and loading the index."""
    
    def test_save_and_load_index(self):
        """Test saving and loading index preserves data."""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create and populate store
            store1 = VectorStore(dimension=384)
            vectors = np.random.rand(5, 384).astype('float32')
            ids = [f"chunk_{i}" for i in range(5)]
            store1.add_vectors(vectors, ids)
            
            # Save index
            store1.save_index(temp_dir)
            
            # Load into new store
            store2 = VectorStore(dimension=384)
            store2.load_index(temp_dir)
            
            # Verify data is preserved
            assert store2.get_vector_count() == 5
            for id_str in ids:
                assert id_str in store2.id_map
            
            # Verify search works on loaded index
            query = vectors[0]
            distances, result_ids = store2.search(query, k=3)
            assert len(result_ids) == 3
        
        finally:
            # Clean up
            shutil.rmtree(temp_dir)
    
    def test_load_nonexistent_index(self):
        """Test loading from non-existent path raises FileNotFoundError."""
        store = VectorStore(dimension=384)
        
        with pytest.raises(FileNotFoundError):
            store.load_index("/nonexistent/path")
    
    def test_load_index_dimension_mismatch(self):
        """Test loading index with different dimension raises ValueError."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create and save with dimension 384
            store1 = VectorStore(dimension=384)
            vectors = np.random.rand(3, 384).astype('float32')
            ids = ["chunk_1", "chunk_2", "chunk_3"]
            store1.add_vectors(vectors, ids)
            store1.save_index(temp_dir)
            
            # Try to load with different dimension
            store2 = VectorStore(dimension=512)
            with pytest.raises(ValueError, match="doesn't match current dimension"):
                store2.load_index(temp_dir)
        
        finally:
            shutil.rmtree(temp_dir)
    
    def test_save_creates_directory(self):
        """Test that save_index creates directory if it doesn't exist."""
        temp_dir = tempfile.mkdtemp()
        save_path = Path(temp_dir) / "nested" / "path"
        
        try:
            store = VectorStore(dimension=384)
            vectors = np.random.rand(2, 384).astype('float32')
            ids = ["chunk_1", "chunk_2"]
            store.add_vectors(vectors, ids)
            
            # Should create nested directories
            store.save_index(str(save_path))
            
            assert save_path.exists()
            assert (save_path / "index.faiss").exists()
            assert (save_path / "metadata.pkl").exists()
        
        finally:
            shutil.rmtree(temp_dir)


class TestVectorStoreHelperMethods:
    """Tests for helper methods."""
    
    def test_get_vector_count(self):
        """Test get_vector_count returns correct count."""
        store = VectorStore(dimension=384)
        assert store.get_vector_count() == 0
        
        vectors = np.random.rand(7, 384).astype('float32')
        ids = [f"chunk_{i}" for i in range(7)]
        store.add_vectors(vectors, ids)
        
        assert store.get_vector_count() == 7
    
    def test_get_vector_by_id(self):
        """Test retrieving vector by ID."""
        store = VectorStore(dimension=384)
        vectors = np.random.rand(3, 384).astype('float32')
        ids = ["chunk_1", "chunk_2", "chunk_3"]
        store.add_vectors(vectors, ids)
        
        retrieved = store.get_vector_by_id("chunk_2")
        
        assert retrieved is not None
        assert retrieved.shape == (384,)
        # Should be close to original vector (within floating point precision)
        np.testing.assert_allclose(retrieved, vectors[1], rtol=1e-5)
    
    def test_get_vector_by_nonexistent_id(self):
        """Test retrieving vector with non-existent ID returns None."""
        store = VectorStore(dimension=384)
        vectors = np.random.rand(2, 384).astype('float32')
        ids = ["chunk_1", "chunk_2"]
        store.add_vectors(vectors, ids)
        
        retrieved = store.get_vector_by_id("chunk_999")
        
        assert retrieved is None



class TestVectorStorePropertyTests:
    """Property-based tests for VectorStore using Hypothesis."""
    
    # Feature: ai-knowledge-copilot, Property 7: Vector storage round-trip
    @given(
        dimension=st.integers(min_value=2, max_value=512),
        num_vectors=st.integers(min_value=1, max_value=50),
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_vector_storage_round_trip(self, dimension, num_vectors, seed):
        """
        Property 7: Vector storage round-trip
        
        For any embedding stored in the vector database with an ID,
        retrieving by that ID should return the same embedding
        (within floating-point precision).
        
        **Validates: Requirements 2.5**
        """
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Create vector store
        store = VectorStore(dimension=dimension)
        
        # Generate random vectors
        vectors = np.random.rand(num_vectors, dimension).astype('float32')
        ids = [f"chunk_{i}" for i in range(num_vectors)]
        
        # Add vectors to store
        store.add_vectors(vectors, ids)
        
        # Verify round-trip: retrieve each vector by ID and compare
        for i, id_str in enumerate(ids):
            retrieved_vector = store.get_vector_by_id(id_str)
            
            # Vector should be retrieved successfully
            assert retrieved_vector is not None, f"Failed to retrieve vector with ID {id_str}"
            
            # Vector should have correct dimension
            assert retrieved_vector.shape == (dimension,), \
                f"Retrieved vector has wrong shape: {retrieved_vector.shape}, expected ({dimension},)"
            
            # Retrieved vector should match original (within floating-point precision)
            np.testing.assert_allclose(
                retrieved_vector,
                vectors[i],
                rtol=1e-5,
                atol=1e-7,
                err_msg=f"Round-trip failed for vector {id_str}: retrieved vector doesn't match original"
            )
    
    # Feature: ai-knowledge-copilot, Property 27: FAISS index persistence round-trip
    @given(
        dimension=st.integers(min_value=2, max_value=512),
        num_vectors=st.integers(min_value=1, max_value=50),
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_faiss_index_persistence_round_trip(self, dimension, num_vectors, seed):
        """
        Property 27: FAISS index persistence round-trip
        
        For any FAISS index with vectors, saving to disk and reloading
        should preserve all vectors and enable identical search results.
        
        **Validates: Requirements 9.4**
        """
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Create temporary directory for persistence
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create vector store and add vectors
            store1 = VectorStore(dimension=dimension)
            vectors = np.random.rand(num_vectors, dimension).astype('float32')
            ids = [f"chunk_{i}" for i in range(num_vectors)]
            store1.add_vectors(vectors, ids)
            
            # Perform a search on the original store
            query_vector = np.random.rand(dimension).astype('float32')
            k = min(5, num_vectors)
            original_distances, original_ids = store1.search(query_vector, k=k)
            
            # Save the index to disk
            store1.save_index(temp_dir)
            
            # Create a new store and load the saved index
            store2 = VectorStore(dimension=dimension)
            store2.load_index(temp_dir)
            
            # Verify vector count is preserved
            assert store2.get_vector_count() == num_vectors, \
                f"Vector count mismatch after load: expected {num_vectors}, got {store2.get_vector_count()}"
            
            # Verify all IDs are preserved
            for id_str in ids:
                assert id_str in store2.id_map, \
                    f"ID {id_str} not found in loaded index"
            
            # Verify all vectors are preserved (round-trip check)
            for i, id_str in enumerate(ids):
                original_vector = vectors[i]
                loaded_vector = store2.get_vector_by_id(id_str)
                
                assert loaded_vector is not None, \
                    f"Failed to retrieve vector {id_str} from loaded index"
                
                np.testing.assert_allclose(
                    loaded_vector,
                    original_vector,
                    rtol=1e-5,
                    atol=1e-7,
                    err_msg=f"Vector {id_str} changed after save/load cycle"
                )
            
            # Verify search results are identical on loaded index
            loaded_distances, loaded_ids = store2.search(query_vector, k=k)
            
            assert len(loaded_distances) == len(original_distances), \
                f"Search result count mismatch: expected {len(original_distances)}, got {len(loaded_distances)}"
            
            assert loaded_ids == original_ids, \
                f"Search result IDs differ after load: expected {original_ids}, got {loaded_ids}"
            
            # Distances should be identical (within floating-point precision)
            np.testing.assert_allclose(
                loaded_distances,
                original_distances,
                rtol=1e-5,
                atol=1e-7,
                err_msg="Search distances differ after save/load cycle"
            )
        
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir)
    
    # Feature: ai-knowledge-copilot, Property 9: Top-k retrieval count
    @given(
        dimension=st.integers(min_value=2, max_value=512),
        num_vectors=st.integers(min_value=0, max_value=100),
        k=st.integers(min_value=1, max_value=150),
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_top_k_retrieval_count(self, dimension, num_vectors, k, seed):
        """
        Property 9: Top-k retrieval count
        
        For any query and k value, the number of retrieved chunks
        should be min(k, total_chunks_in_index).
        
        **Validates: Requirements 3.3**
        """
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Create vector store
        store = VectorStore(dimension=dimension)
        
        # Add vectors if num_vectors > 0
        if num_vectors > 0:
            vectors = np.random.rand(num_vectors, dimension).astype('float32')
            ids = [f"chunk_{i}" for i in range(num_vectors)]
            store.add_vectors(vectors, ids)
        
        # Generate a random query vector
        query_vector = np.random.rand(dimension).astype('float32')
        
        # Perform search with k
        distances, result_ids = store.search(query_vector, k=k)
        
        # Expected count is min(k, num_vectors)
        expected_count = min(k, num_vectors)
        
        # Verify the number of retrieved chunks matches expected count
        assert len(result_ids) == expected_count, \
            f"Top-k retrieval count mismatch: expected {expected_count} results " \
            f"(min(k={k}, total={num_vectors})), but got {len(result_ids)}"
        
        # Verify distances list has same length
        assert len(distances) == expected_count, \
            f"Distances count mismatch: expected {expected_count}, got {len(distances)}"
        
        # Verify all returned IDs are valid (exist in the store)
        for result_id in result_ids:
            assert result_id in store.id_map, \
                f"Invalid ID returned: {result_id} not in store"
    
    # Feature: ai-knowledge-copilot, Property 10: Retrieval score ordering
    @given(
        dimension=st.integers(min_value=2, max_value=512),
        num_vectors=st.integers(min_value=2, max_value=100),
        k=st.integers(min_value=2, max_value=50),
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_retrieval_score_ordering(self, dimension, num_vectors, k, seed):
        """
        Property 10: Retrieval score ordering
        
        For any retrieval results, the similarity scores should be in
        non-increasing order (highest similarity first).
        
        Note: FAISS returns L2 distances, where smaller values indicate
        higher similarity. Therefore, distances should be in non-decreasing
        order (smallest/best distance first).
        
        **Validates: Requirements 3.4**
        """
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Create vector store
        store = VectorStore(dimension=dimension)
        
        # Generate and add random vectors
        vectors = np.random.rand(num_vectors, dimension).astype('float32')
        ids = [f"chunk_{i}" for i in range(num_vectors)]
        store.add_vectors(vectors, ids)
        
        # Generate a random query vector
        query_vector = np.random.rand(dimension).astype('float32')
        
        # Perform search with k
        distances, result_ids = store.search(query_vector, k=k)
        
        # Skip test if no results returned (empty index case)
        if len(distances) < 2:
            return
        
        # Verify distances are in non-decreasing order
        # (smaller L2 distance = higher similarity = better match)
        for i in range(len(distances) - 1):
            assert distances[i] <= distances[i + 1], \
                f"Retrieval score ordering violated: distance at index {i} ({distances[i]}) " \
                f"is greater than distance at index {i + 1} ({distances[i + 1]}). " \
                f"Distances should be in non-decreasing order (best match first). " \
                f"Full distances: {distances}"
