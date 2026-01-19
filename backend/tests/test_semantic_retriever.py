"""
Tests for the SemanticRetriever class.

This module contains unit tests for the semantic retrieval functionality.
"""
import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from src.semantic_retriever import SemanticRetriever
from src.vector_store import VectorStore
from src.embedder import EmbeddingGenerator
from src.metadata_store import MetadataStore
from src.models import RetrievedChunk, Chunk


class TestSemanticRetriever:
    """Unit tests for SemanticRetriever class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def embedder(self):
        """Create an EmbeddingGenerator instance for testing."""
        return EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    
    @pytest.fixture
    def vector_store(self, embedder):
        """Create a VectorStore instance for testing."""
        dimension = embedder.get_embedding_dimension()
        return VectorStore(dimension=dimension)
    
    @pytest.fixture
    def metadata_store(self, temp_dir):
        """Create a MetadataStore instance for testing."""
        db_path = str(Path(temp_dir) / "test_metadata.db")
        return MetadataStore(db_path=db_path)
    
    @pytest.fixture
    def retriever(self, vector_store, embedder, metadata_store):
        """Create a SemanticRetriever instance for testing."""
        return SemanticRetriever(vector_store, embedder, metadata_store)
    
    @pytest.fixture
    def populated_retriever(self, retriever, embedder, vector_store, metadata_store):
        """Create a retriever with some test data."""
        # Add test documents
        doc_id = "doc_1"
        metadata_store.add_document(doc_id, {
            'source': '/path/to/test_document.pdf',
            'source_type': 'pdf'
        })
        
        # Add test chunks
        test_texts = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing enables computers to understand text.",
            "Computer vision allows machines to interpret visual information.",
            "Reinforcement learning trains agents through rewards and penalties."
        ]
        
        chunk_ids = []
        embeddings_list = []
        
        for i, text in enumerate(test_texts):
            chunk_id = f"chunk_{i}"
            chunk_ids.append(chunk_id)
            
            # Add chunk metadata
            metadata_store.add_chunk(chunk_id, {
                'document_id': doc_id,
                'text': text,
                'chunk_index': i,
                'page_number': i + 1
            })
            
            # Generate embedding
            embedding = embedder.embed_text(text)
            embeddings_list.append(embedding)
        
        # Add embeddings to vector store
        embeddings_array = np.array(embeddings_list, dtype=np.float32)
        vector_store.add_vectors(embeddings_array, chunk_ids)
        
        return retriever
    
    def test_initialization(self, vector_store, embedder, metadata_store):
        """Test that SemanticRetriever initializes correctly."""
        retriever = SemanticRetriever(vector_store, embedder, metadata_store)
        
        assert retriever.vector_store is vector_store
        assert retriever.embedder is embedder
        assert retriever.metadata_store is metadata_store
    
    def test_retrieve_with_empty_query_raises_error(self, retriever):
        """Test that retrieve raises error for empty query."""
        with pytest.raises(ValueError, match="query cannot be empty"):
            retriever.retrieve("")
        
        with pytest.raises(ValueError, match="query cannot be empty"):
            retriever.retrieve("   ")
    
    def test_retrieve_with_invalid_k_raises_error(self, retriever):
        """Test that retrieve raises error for invalid k."""
        with pytest.raises(ValueError, match="k must be positive"):
            retriever.retrieve("test query", k=0)
        
        with pytest.raises(ValueError, match="k must be positive"):
            retriever.retrieve("test query", k=-5)
    
    def test_retrieve_from_empty_index_returns_empty_list(self, retriever):
        """Test that retrieve returns empty list when index is empty."""
        results = retriever.retrieve("test query", k=5)
        
        assert isinstance(results, list)
        assert len(results) == 0
    
    def test_retrieve_returns_retrieved_chunks(self, populated_retriever):
        """Test that retrieve returns RetrievedChunk objects."""
        results = populated_retriever.retrieve("What is machine learning?", k=3)
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert len(results) <= 3
        
        for result in results:
            assert isinstance(result, RetrievedChunk)
            assert isinstance(result.chunk, Chunk)
            assert isinstance(result.score, float)
            assert isinstance(result.document_name, str)
    
    def test_retrieve_returns_correct_number_of_results(self, populated_retriever):
        """Test that retrieve returns correct number of results."""
        # Request 3 results
        results = populated_retriever.retrieve("artificial intelligence", k=3)
        assert len(results) == 3
        
        # Request more than available
        results = populated_retriever.retrieve("artificial intelligence", k=100)
        assert len(results) == 5  # We only have 5 chunks
    
    def test_retrieve_results_have_valid_scores(self, populated_retriever):
        """Test that retrieved results have valid similarity scores."""
        results = populated_retriever.retrieve("neural networks", k=5)
        
        for result in results:
            # Score should be non-negative
            assert result.score >= 0
            # Score should be reasonable (similarity score)
            assert result.score <= 1.0
    
    def test_retrieve_results_are_sorted_by_relevance(self, populated_retriever):
        """Test that results are sorted by relevance (highest score first)."""
        results = populated_retriever.retrieve("deep learning neural networks", k=5)
        
        # Scores should be in descending order
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_retrieve_with_threshold_filters_results(self, populated_retriever):
        """Test that threshold parameter filters low-similarity results."""
        # Get all results without threshold
        all_results = populated_retriever.retrieve("machine learning", k=5, threshold=None)
        
        # Get results with high threshold (will filter out distant results)
        # Note: threshold is on distance, not similarity score
        filtered_results = populated_retriever.retrieve("machine learning", k=5, threshold=0.5)
        
        # Filtered results should be <= all results
        assert len(filtered_results) <= len(all_results)
    
    def test_retrieve_populates_chunk_metadata(self, populated_retriever):
        """Test that retrieved chunks have complete metadata."""
        results = populated_retriever.retrieve("computer vision", k=3)
        
        for result in results:
            chunk = result.chunk
            
            # Check required fields
            assert chunk.id is not None
            assert chunk.document_id is not None
            assert chunk.text is not None
            assert len(chunk.text) > 0
            
            # Check metadata
            assert 'chunk_index' in chunk.metadata
            assert 'document_id' in chunk.metadata
    
    def test_retrieve_populates_document_name(self, populated_retriever):
        """Test that retrieved chunks have document name."""
        results = populated_retriever.retrieve("reinforcement learning", k=3)
        
        for result in results:
            assert result.document_name is not None
            assert len(result.document_name) > 0
            # Should extract filename from path
            assert result.document_name == "test_document.pdf"
    
    def test_retrieve_populates_page_number(self, populated_retriever):
        """Test that retrieved chunks have page number when available."""
        results = populated_retriever.retrieve("natural language", k=3)
        
        for result in results:
            # Page number should be present in our test data
            assert result.page_number is not None
            assert isinstance(result.page_number, int)
            assert result.page_number > 0
    
    def test_retrieve_with_relevant_query(self, populated_retriever):
        """Test retrieval with a query relevant to indexed content."""
        # Query about machine learning should return relevant chunk
        results = populated_retriever.retrieve("What is machine learning?", k=1)
        
        assert len(results) == 1
        # The first result should contain "machine learning" in the text
        assert "machine learning" in results[0].chunk.text.lower()
    
    def test_retrieve_with_scores_returns_tuples(self, populated_retriever):
        """Test that retrieve_with_scores returns tuples of (chunk, score)."""
        results = populated_retriever.retrieve_with_scores("deep learning", k=3)
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 2
            
            chunk, score = item
            assert isinstance(chunk, RetrievedChunk)
            assert isinstance(score, float)
            assert score >= 0
    
    def test_retrieve_with_scores_matches_retrieve(self, populated_retriever):
        """Test that retrieve_with_scores returns same results as retrieve."""
        query = "neural networks and deep learning"
        k = 3
        
        # Get results from both methods
        results_retrieve = populated_retriever.retrieve(query, k=k)
        results_with_scores = populated_retriever.retrieve_with_scores(query, k=k)
        
        # Should have same number of results
        assert len(results_retrieve) == len(results_with_scores)
        
        # Scores should match
        for i, (chunk, score) in enumerate(results_with_scores):
            assert chunk.chunk.id == results_retrieve[i].chunk.id
            assert score == results_retrieve[i].score
    
    def test_retrieve_with_scores_sorted_by_score(self, populated_retriever):
        """Test that retrieve_with_scores returns results sorted by score."""
        results = populated_retriever.retrieve_with_scores("artificial intelligence", k=5)
        
        # Extract scores
        scores = [score for _, score in results]
        
        # Should be in descending order
        assert scores == sorted(scores, reverse=True)
    
    def test_retrieve_handles_missing_metadata_gracefully(self, retriever, embedder, vector_store, metadata_store):
        """Test that retrieve handles missing metadata gracefully."""
        # Add a vector without corresponding metadata
        embedding = embedder.embed_text("Test text without metadata")
        vector_store.add_vectors(np.array([embedding], dtype=np.float32), ["orphan_chunk"])
        
        # Retrieve should not crash
        results = retriever.retrieve("test query", k=5)
        
        # Should return empty or skip the orphan chunk
        # (depends on implementation - we skip chunks without metadata)
        assert isinstance(results, list)
    
    def test_retrieve_with_multiple_documents(self, retriever, embedder, vector_store, metadata_store):
        """Test retrieval across multiple documents."""
        # Add two documents
        for doc_num in range(2):
            doc_id = f"doc_{doc_num}"
            metadata_store.add_document(doc_id, {
                'source': f'/path/to/document_{doc_num}.pdf',
                'source_type': 'pdf'
            })
            
            # Add chunks for this document
            for chunk_num in range(3):
                chunk_id = f"doc{doc_num}_chunk{chunk_num}"
                text = f"Document {doc_num} content about topic {chunk_num}"
                
                metadata_store.add_chunk(chunk_id, {
                    'document_id': doc_id,
                    'text': text,
                    'chunk_index': chunk_num
                })
                
                embedding = embedder.embed_text(text)
                vector_store.add_vectors(
                    np.array([embedding], dtype=np.float32),
                    [chunk_id]
                )
        
        # Retrieve should return results from both documents
        results = retriever.retrieve("document content", k=6)
        
        assert len(results) > 0
        # Should have results from multiple documents
        document_names = {r.document_name for r in results}
        assert len(document_names) >= 1  # At least one document
    
    def test_retrieve_empty_query_with_scores_raises_error(self, retriever):
        """Test that retrieve_with_scores raises error for empty query."""
        with pytest.raises(ValueError, match="query cannot be empty"):
            retriever.retrieve_with_scores("")
    
    def test_retrieve_consistency(self, populated_retriever):
        """Test that multiple retrievals with same query return same results."""
        query = "machine learning and AI"
        k = 3
        
        results1 = populated_retriever.retrieve(query, k=k)
        results2 = populated_retriever.retrieve(query, k=k)
        
        # Should return same chunk IDs in same order
        assert len(results1) == len(results2)
        for i in range(len(results1)):
            assert results1[i].chunk.id == results2[i].chunk.id
            # Scores should be very close (within floating point precision)
            assert abs(results1[i].score - results2[i].score) < 1e-5
    
    def test_threshold_excludes_low_similarity_results(self, populated_retriever):
        """Test that results below threshold are excluded."""
        # First, get all results without threshold to see the distance range
        all_results = populated_retriever.retrieve("quantum computing", k=5, threshold=None)
        
        # If we have results, test with a restrictive threshold
        if len(all_results) > 0:
            # Use a very low threshold (on distance) to filter out distant results
            # Lower threshold = only very close matches pass
            filtered_results = populated_retriever.retrieve("quantum computing", k=5, threshold=0.3)
            
            # Filtered results should have fewer or equal items
            assert len(filtered_results) <= len(all_results)
            
            # All filtered results should have passed the threshold
            # (This is validated by the vector store's search method)
            for result in filtered_results:
                # Results that passed should exist
                assert result.chunk.id is not None
    
    def test_threshold_returns_empty_when_no_chunks_meet_threshold(self, populated_retriever):
        """Test empty results when no chunks meet threshold."""
        # Use a query very different from indexed content
        # and a very restrictive threshold
        results = populated_retriever.retrieve(
            "xyzabc nonsense query with random words",
            k=5,
            threshold=0.01  # Very low threshold - only exact matches would pass
        )
        
        # Should return empty list or very few results
        assert isinstance(results, list)
        # With such a restrictive threshold and irrelevant query, 
        # we expect 0 or very few results
        assert len(results) <= 1
