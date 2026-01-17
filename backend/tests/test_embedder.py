"""
Tests for the EmbeddingGenerator class.

This module contains unit tests and property-based tests for the embedding
generation functionality.
"""
import pytest
import numpy as np
from hypothesis import given, strategies as st, settings
from src.embedder import EmbeddingGenerator


class TestEmbeddingGenerator:
    """Unit tests for EmbeddingGenerator class."""
    
    @pytest.fixture
    def embedder(self):
        """Create an EmbeddingGenerator instance for testing."""
        return EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    
    def test_initialization_default_model(self):
        """Test that EmbeddingGenerator initializes with default model."""
        embedder = EmbeddingGenerator()
        assert embedder.model_name == "all-MiniLM-L6-v2"
        assert embedder.model is not None
    
    def test_initialization_custom_model(self):
        """Test that EmbeddingGenerator initializes with custom model."""
        embedder = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
        assert embedder.model_name == "all-MiniLM-L6-v2"
        assert embedder.model is not None
    
    def test_embed_text_returns_numpy_array(self, embedder):
        """Test that embed_text returns a numpy array."""
        text = "This is a test sentence."
        embedding = embedder.embed_text(text)
        assert isinstance(embedding, np.ndarray)
    
    def test_embed_text_correct_dimension(self, embedder):
        """Test that embed_text returns correct embedding dimension."""
        text = "This is a test sentence."
        embedding = embedder.embed_text(text)
        expected_dim = embedder.get_embedding_dimension()
        assert embedding.shape == (expected_dim,)
        assert embedding.shape == (384,)  # all-MiniLM-L6-v2 dimension
    
    def test_embed_text_empty_string_raises_error(self, embedder):
        """Test that embed_text raises error for empty string."""
        with pytest.raises(ValueError, match="text cannot be empty"):
            embedder.embed_text("")
    
    def test_embed_text_whitespace_only_raises_error(self, embedder):
        """Test that embed_text raises error for whitespace-only string."""
        with pytest.raises(ValueError, match="text cannot be empty"):
            embedder.embed_text("   ")
    
    def test_embed_batch_returns_numpy_array(self, embedder):
        """Test that embed_batch returns a numpy array."""
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        embeddings = embedder.embed_batch(texts)
        assert isinstance(embeddings, np.ndarray)
    
    def test_embed_batch_correct_shape(self, embedder):
        """Test that embed_batch returns correct shape."""
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        embeddings = embedder.embed_batch(texts)
        expected_dim = embedder.get_embedding_dimension()
        assert embeddings.shape == (len(texts), expected_dim)
        assert embeddings.shape == (3, 384)
    
    def test_embed_batch_single_text(self, embedder):
        """Test that embed_batch works with a single text."""
        texts = ["Single sentence."]
        embeddings = embedder.embed_batch(texts)
        assert embeddings.shape == (1, 384)
    
    def test_embed_batch_empty_list_raises_error(self, embedder):
        """Test that embed_batch raises error for empty list."""
        with pytest.raises(ValueError, match="texts list cannot be empty"):
            embedder.embed_batch([])
    
    def test_embed_batch_all_empty_strings_raises_error(self, embedder):
        """Test that embed_batch raises error when all texts are empty."""
        with pytest.raises(ValueError, match="all texts in the list are empty"):
            embedder.embed_batch(["", "  ", ""])
    
    def test_embed_batch_custom_batch_size(self, embedder):
        """Test that embed_batch works with custom batch size."""
        texts = ["Text " + str(i) for i in range(10)]
        embeddings = embedder.embed_batch(texts, batch_size=5)
        assert embeddings.shape == (10, 384)
    
    def test_get_embedding_dimension(self, embedder):
        """Test that get_embedding_dimension returns correct value."""
        dimension = embedder.get_embedding_dimension()
        assert isinstance(dimension, int)
        assert dimension == 384  # all-MiniLM-L6-v2 dimension
    
    def test_embed_text_consistency(self, embedder):
        """Test that embedding the same text twice produces similar results."""
        text = "Consistency test sentence."
        embedding1 = embedder.embed_text(text)
        embedding2 = embedder.embed_text(text)
        # Embeddings should be very close (within floating point precision)
        np.testing.assert_array_almost_equal(embedding1, embedding2, decimal=5)
    
    def test_embed_batch_vs_single(self, embedder):
        """Test that batch embedding produces same results as single embedding."""
        texts = ["First text.", "Second text."]
        
        # Embed individually
        embedding1 = embedder.embed_text(texts[0])
        embedding2 = embedder.embed_text(texts[1])
        
        # Embed as batch
        batch_embeddings = embedder.embed_batch(texts)
        
        # Compare results
        np.testing.assert_array_almost_equal(embedding1, batch_embeddings[0], decimal=5)
        np.testing.assert_array_almost_equal(embedding2, batch_embeddings[1], decimal=5)
    
    def test_different_texts_produce_different_embeddings(self, embedder):
        """Test that different texts produce different embeddings."""
        text1 = "The cat sat on the mat."
        text2 = "The dog ran in the park."
        
        embedding1 = embedder.embed_text(text1)
        embedding2 = embedder.embed_text(text2)
        
        # Embeddings should be different
        assert not np.array_equal(embedding1, embedding2)
        
        # But they should have the same dimension
        assert embedding1.shape == embedding2.shape
    
    def test_similar_texts_produce_similar_embeddings(self, embedder):
        """Test that similar texts produce similar embeddings (high cosine similarity)."""
        text1 = "The cat is sleeping."
        text2 = "A cat is sleeping."
        text3 = "The weather is nice today."
        
        embedding1 = embedder.embed_text(text1)
        embedding2 = embedder.embed_text(text2)
        embedding3 = embedder.embed_text(text3)
        
        # Calculate cosine similarity
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        sim_1_2 = cosine_similarity(embedding1, embedding2)
        sim_1_3 = cosine_similarity(embedding1, embedding3)
        
        # Similar texts should have higher similarity than dissimilar texts
        assert sim_1_2 > sim_1_3
        assert sim_1_2 > 0.8  # Very similar texts should have high similarity
    
    # Task 4.4: Unit tests for batch processing
    def test_batch_embedding_with_small_batch_size(self, embedder):
        """Test batch embedding with small batch size (batch_size=1)."""
        texts = ["Text one.", "Text two.", "Text three.", "Text four."]
        embeddings = embedder.embed_batch(texts, batch_size=1)
        
        # Should process all texts
        assert embeddings.shape == (4, 384)
        
        # Each embedding should have correct dimension
        for i in range(len(texts)):
            assert embeddings[i].shape == (384,)
    
    def test_batch_embedding_with_medium_batch_size(self, embedder):
        """Test batch embedding with medium batch size (batch_size=16)."""
        texts = ["Sentence number " + str(i) for i in range(20)]
        embeddings = embedder.embed_batch(texts, batch_size=16)
        
        # Should process all texts
        assert embeddings.shape == (20, 384)
        
        # Verify dimensions match model output
        expected_dim = embedder.get_embedding_dimension()
        assert embeddings.shape[1] == expected_dim
    
    def test_batch_embedding_with_large_batch_size(self, embedder):
        """Test batch embedding with large batch size (batch_size=64)."""
        texts = ["Document chunk " + str(i) for i in range(50)]
        embeddings = embedder.embed_batch(texts, batch_size=64)
        
        # Should process all texts
        assert embeddings.shape == (50, 384)
        
        # Verify dimensions match model output
        expected_dim = embedder.get_embedding_dimension()
        assert embeddings.shape[1] == expected_dim
    
    def test_batch_embedding_batch_size_larger_than_input(self, embedder):
        """Test batch embedding when batch_size is larger than number of texts."""
        texts = ["Text A", "Text B", "Text C"]
        embeddings = embedder.embed_batch(texts, batch_size=100)
        
        # Should still process all texts correctly
        assert embeddings.shape == (3, 384)
        
        # Verify dimensions match model output
        expected_dim = embedder.get_embedding_dimension()
        assert embeddings.shape[1] == expected_dim
    
    def test_batch_embedding_various_batch_sizes_consistency(self, embedder):
        """Test that different batch sizes produce identical embeddings."""
        texts = ["Consistency test " + str(i) for i in range(12)]
        
        # Embed with different batch sizes
        embeddings_batch_2 = embedder.embed_batch(texts, batch_size=2)
        embeddings_batch_4 = embedder.embed_batch(texts, batch_size=4)
        embeddings_batch_6 = embedder.embed_batch(texts, batch_size=6)
        
        # All should produce identical results
        np.testing.assert_array_almost_equal(embeddings_batch_2, embeddings_batch_4, decimal=5)
        np.testing.assert_array_almost_equal(embeddings_batch_4, embeddings_batch_6, decimal=5)
    
    def test_batch_embedding_dimension_verification(self, embedder):
        """Test that all embeddings in batch have correct dimensions."""
        texts = ["Text " + str(i) for i in range(25)]
        embeddings = embedder.embed_batch(texts, batch_size=8)
        
        expected_dim = embedder.get_embedding_dimension()
        
        # Verify overall shape
        assert embeddings.shape == (25, expected_dim)
        
        # Verify each individual embedding dimension
        for i in range(len(texts)):
            assert embeddings[i].shape == (expected_dim,)
            assert len(embeddings[i]) == expected_dim
    
    def test_batch_embedding_large_dataset(self, embedder):
        """Test batch embedding with a larger dataset."""
        texts = ["Document chunk number " + str(i) + " with some content." for i in range(100)]
        embeddings = embedder.embed_batch(texts, batch_size=32)
        
        # Should process all texts
        assert embeddings.shape == (100, 384)
        
        # Verify dimensions match model output
        expected_dim = embedder.get_embedding_dimension()
        assert embeddings.shape[1] == expected_dim
        
        # Verify all embeddings are valid
        assert not np.isnan(embeddings).any()
        assert not np.isinf(embeddings).any()
    
    def test_batch_embedding_preserves_order(self, embedder):
        """Test that batch embedding preserves the order of input texts."""
        texts = ["First", "Second", "Third", "Fourth", "Fifth"]
        
        # Embed as batch
        batch_embeddings = embedder.embed_batch(texts)
        
        # Embed individually
        individual_embeddings = [embedder.embed_text(t) for t in texts]
        
        # Order should be preserved
        for i in range(len(texts)):
            np.testing.assert_array_almost_equal(
                batch_embeddings[i],
                individual_embeddings[i],
                decimal=5
            )


# Module-level embedder for property tests to avoid reloading model
_test_embedder = None

def get_test_embedder():
    """Get or create a shared embedder instance for property tests."""
    global _test_embedder
    if _test_embedder is None:
        _test_embedder = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    return _test_embedder


# Property-Based Tests
class TestEmbeddingGeneratorProperties:
    """Property-based tests for EmbeddingGenerator."""
    
    @given(text=st.text(min_size=1, max_size=1000))
    @settings(max_examples=20, deadline=None)
    def test_property_embedding_dimension_consistency(self, text):
        """
        Property 8: Query embedding dimension consistency
        
        For any query string, the generated embedding should have the same
        dimension as document embeddings.
        
        **Validates: Requirements 3.1**
        """
        # Skip if text is only whitespace
        if not text.strip():
            return
        
        embedder = get_test_embedder()
        embedding = embedder.embed_text(text)
        expected_dim = embedder.get_embedding_dimension()
        
        # Embedding should have correct dimension
        assert embedding.shape == (expected_dim,)
        assert len(embedding) == expected_dim
    
    @given(text=st.text(min_size=1, max_size=1000))
    @settings(max_examples=100, deadline=None)
    def test_property_embedding_generation_completeness(self, text):
        """
        Property 6: Embedding generation completeness
        
        For any chunk, an embedding vector should be generated with dimension
        matching the embedding model's output dimension.
        
        **Validates: Requirements 2.4**
        """
        # Skip if text is only whitespace
        if not text.strip():
            return
        
        embedder = get_test_embedder()
        embedding = embedder.embed_text(text)
        expected_dim = embedder.get_embedding_dimension()
        
        # Embedding should be generated (not None)
        assert embedding is not None
        
        # Embedding should be a numpy array
        assert isinstance(embedding, np.ndarray)
        
        # Embedding should have correct dimension
        assert embedding.shape == (expected_dim,)
        assert len(embedding) == expected_dim
        
        # Embedding should be valid (no NaN or Inf)
        assert not np.isnan(embedding).any()
        assert not np.isinf(embedding).any()
        
        # Embedding should contain non-zero values (semantic meaning captured)
        assert np.any(embedding != 0)
    
    @given(
        texts=st.lists(
            st.text(min_size=1, max_size=500),
            min_size=1,
            max_size=10
        )
    )
    @settings(max_examples=20, deadline=None)
    def test_property_batch_embedding_completeness(self, texts):
        """
        Property 6: Embedding generation completeness (batch variant)
        
        For any set of chunks, embedding vectors should be generated with dimension
        matching the embedding model's output dimension.
        
        **Validates: Requirements 2.4**
        """
        # Filter out empty texts
        valid_texts = [t for t in texts if t.strip()]
        if not valid_texts:
            return
        
        embedder = get_test_embedder()
        embeddings = embedder.embed_batch(valid_texts)
        expected_dim = embedder.get_embedding_dimension()
        
        # Should generate embedding for each text
        assert embeddings.shape[0] == len(valid_texts)
        
        # Each embedding should have correct dimension
        assert embeddings.shape[1] == expected_dim
        
        # All embeddings should be valid (no NaN or Inf)
        assert not np.isnan(embeddings).any()
        assert not np.isinf(embeddings).any()
    
    @given(text=st.text(min_size=1, max_size=1000))
    @settings(max_examples=15, deadline=None)
    def test_property_embedding_determinism(self, text):
        """
        Property: Embedding determinism
        
        For any text, embedding it multiple times should produce the same result.
        """
        # Skip if text is only whitespace
        if not text.strip():
            return
        
        embedder = get_test_embedder()
        embedding1 = embedder.embed_text(text)
        embedding2 = embedder.embed_text(text)
        
        # Embeddings should be identical (within floating point precision)
        np.testing.assert_array_almost_equal(embedding1, embedding2, decimal=5)
    
    @given(
        texts=st.lists(
            st.text(min_size=1, max_size=200),
            min_size=2,
            max_size=5
        )
    )
    @settings(max_examples=15, deadline=None)
    def test_property_batch_vs_individual_consistency(self, texts):
        """
        Property: Batch vs individual consistency
        
        For any list of texts, batch embedding should produce the same results
        as individual embedding.
        """
        # Filter out empty texts
        valid_texts = [t for t in texts if t.strip()]
        if len(valid_texts) < 2:
            return
        
        embedder = get_test_embedder()
        
        # Embed individually
        individual_embeddings = [embedder.embed_text(t) for t in valid_texts]
        
        # Embed as batch
        batch_embeddings = embedder.embed_batch(valid_texts)
        
        # Results should be very close
        for i, individual_emb in enumerate(individual_embeddings):
            np.testing.assert_array_almost_equal(
                individual_emb,
                batch_embeddings[i],
                decimal=5
            )
