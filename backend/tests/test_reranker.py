"""
Tests for the Reranker class.

This module contains unit tests and property-based tests for the reranking
functionality using cross-encoder models.
"""
import pytest
import numpy as np
from hypothesis import given, strategies as st, settings
from src.reranker import Reranker
from src.models import RetrievedChunk, Chunk


class TestReranker:
    """Unit tests for Reranker class."""
    
    @pytest.fixture
    def reranker(self):
        """Create a Reranker instance for testing."""
        return Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample RetrievedChunk objects for testing."""
        chunks = []
        texts = [
            "Machine learning is a subset of artificial intelligence that focuses on learning from data.",
            "The weather today is sunny with a high of 75 degrees.",
            "Deep learning uses neural networks with multiple layers to learn representations.",
            "Pizza is a popular Italian dish made with dough, sauce, and cheese.",
            "Natural language processing enables computers to understand and generate human language."
        ]
        
        for i, text in enumerate(texts):
            chunk = Chunk(
                id=f"chunk_{i}",
                document_id=f"doc_{i % 2}",  # Alternate between two documents
                text=text,
                embedding=None,
                metadata={
                    'chunk_index': i,
                    'page_number': i + 1,
                    'document_id': f"doc_{i % 2}"
                }
            )
            
            retrieved_chunk = RetrievedChunk(
                chunk=chunk,
                score=0.5 + i * 0.1,  # Arbitrary initial scores
                document_name=f"document_{i % 2}.pdf",
                page_number=i + 1
            )
            
            chunks.append(retrieved_chunk)
        
        return chunks
    
    def test_initialization(self):
        """Test that Reranker initializes correctly."""
        reranker = Reranker()
        
        assert reranker.model is not None
        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    def test_initialization_with_custom_model(self):
        """Test initialization with custom model name."""
        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        reranker = Reranker(model_name=model_name)
        
        assert reranker.model_name == model_name
        assert reranker.model is not None
    
    def test_score_pairs_with_empty_query_raises_error(self, reranker):
        """Test that score_pairs raises error for empty query."""
        with pytest.raises(ValueError, match="query cannot be empty"):
            reranker.score_pairs("", ["some text"])
        
        with pytest.raises(ValueError, match="query cannot be empty"):
            reranker.score_pairs("   ", ["some text"])
    
    def test_score_pairs_with_empty_texts_raises_error(self, reranker):
        """Test that score_pairs raises error for empty texts list."""
        with pytest.raises(ValueError, match="texts list cannot be empty"):
            reranker.score_pairs("test query", [])
    
    def test_score_pairs_returns_scores(self, reranker):
        """Test that score_pairs returns relevance scores."""
        query = "What is machine learning?"
        texts = [
            "Machine learning is a subset of AI.",
            "The weather is nice today."
        ]
        
        scores = reranker.score_pairs(query, texts)
        
        assert isinstance(scores, list)
        assert len(scores) == len(texts)
        
        for score in scores:
            assert isinstance(score, float)
    
    def test_score_pairs_relevance_ordering(self, reranker):
        """Test that score_pairs gives higher scores to more relevant texts."""
        query = "What is machine learning?"
        texts = [
            "Machine learning is a subset of artificial intelligence.",
            "The weather forecast predicts rain tomorrow.",
            "Deep learning is a type of machine learning using neural networks."
        ]
        
        scores = reranker.score_pairs(query, texts)
        
        # First and third texts are about ML, should have higher scores than weather
        assert scores[0] > scores[1]
        assert scores[2] > scores[1]
    
    def test_score_pairs_with_single_text(self, reranker):
        """Test score_pairs with a single text."""
        query = "artificial intelligence"
        texts = ["AI is the simulation of human intelligence."]
        
        scores = reranker.score_pairs(query, texts)
        
        assert len(scores) == 1
        assert isinstance(scores[0], float)
    
    def test_rerank_with_empty_query_raises_error(self, reranker, sample_chunks):
        """Test that rerank raises error for empty query."""
        with pytest.raises(ValueError, match="query cannot be empty"):
            reranker.rerank("", sample_chunks)
        
        with pytest.raises(ValueError, match="query cannot be empty"):
            reranker.rerank("   ", sample_chunks)
    
    def test_rerank_with_empty_chunks_raises_error(self, reranker):
        """Test that rerank raises error for empty chunks list."""
        with pytest.raises(ValueError, match="chunks list cannot be empty"):
            reranker.rerank("test query", [])
    
    def test_rerank_with_invalid_top_n_raises_error(self, reranker, sample_chunks):
        """Test that rerank raises error for invalid top_n."""
        with pytest.raises(ValueError, match="top_n must be positive"):
            reranker.rerank("test query", sample_chunks, top_n=0)
        
        with pytest.raises(ValueError, match="top_n must be positive"):
            reranker.rerank("test query", sample_chunks, top_n=-5)
    
    def test_rerank_returns_retrieved_chunks(self, reranker, sample_chunks):
        """Test that rerank returns RetrievedChunk objects."""
        query = "What is machine learning?"
        
        reranked = reranker.rerank(query, sample_chunks, top_n=3)
        
        assert isinstance(reranked, list)
        assert len(reranked) > 0
        
        for chunk in reranked:
            assert isinstance(chunk, RetrievedChunk)
            assert isinstance(chunk.chunk, Chunk)
            assert isinstance(chunk.score, float)
    
    def test_rerank_returns_correct_number_of_chunks(self, reranker, sample_chunks):
        """Test that rerank returns correct number of chunks."""
        query = "machine learning"
        
        # Request 3 chunks
        reranked = reranker.rerank(query, sample_chunks, top_n=3)
        assert len(reranked) == 3
        
        # Request more than available
        reranked = reranker.rerank(query, sample_chunks, top_n=100)
        assert len(reranked) == len(sample_chunks)
    
    def test_rerank_updates_scores(self, reranker, sample_chunks):
        """Test that rerank updates scores with reranking scores."""
        query = "machine learning and AI"
        
        # Get original scores
        original_scores = [chunk.score for chunk in sample_chunks]
        
        # Rerank
        reranked = reranker.rerank(query, sample_chunks, top_n=5)
        
        # Reranked scores should be different from original
        reranked_scores = [chunk.score for chunk in reranked]
        
        # At least some scores should be different
        # (unless by coincidence they're the same, which is unlikely)
        assert reranked_scores != original_scores[:len(reranked_scores)]
    
    def test_rerank_preserves_metadata(self, reranker, sample_chunks):
        """Test that rerank preserves all chunk metadata."""
        query = "deep learning neural networks"
        
        reranked = reranker.rerank(query, sample_chunks, top_n=3)
        
        for reranked_chunk in reranked:
            # Find original chunk by ID
            original_chunk = next(
                c for c in sample_chunks if c.chunk.id == reranked_chunk.chunk.id
            )
            
            # Check that metadata is preserved
            assert reranked_chunk.chunk.id == original_chunk.chunk.id
            assert reranked_chunk.chunk.document_id == original_chunk.chunk.document_id
            assert reranked_chunk.chunk.text == original_chunk.chunk.text
            assert reranked_chunk.chunk.metadata == original_chunk.chunk.metadata
            assert reranked_chunk.document_name == original_chunk.document_name
            assert reranked_chunk.page_number == original_chunk.page_number
    
    def test_rerank_orders_by_relevance(self, reranker, sample_chunks):
        """Test that rerank orders chunks by relevance to query."""
        query = "machine learning and artificial intelligence"
        
        reranked = reranker.rerank(query, sample_chunks, top_n=5)
        
        # Scores should be in descending order
        scores = [chunk.score for chunk in reranked]
        assert scores == sorted(scores, reverse=True)
        
        # Most relevant chunks should be first
        # (chunks about ML/AI should rank higher than weather/pizza)
        top_chunk_text = reranked[0].chunk.text.lower()
        assert any(keyword in top_chunk_text for keyword in ['machine learning', 'artificial intelligence', 'deep learning', 'natural language'])
    
    def test_rerank_improves_relevance_ordering(self, reranker):
        """Test that reranking improves relevance ordering."""
        # Create chunks with intentionally poor initial ordering
        texts = [
            "Pizza is delicious.",  # Irrelevant
            "Machine learning is a subset of AI.",  # Relevant
            "The weather is sunny.",  # Irrelevant
            "Deep learning uses neural networks.",  # Relevant
            "I like cats."  # Irrelevant
        ]
        
        chunks = []
        for i, text in enumerate(texts):
            chunk = Chunk(
                id=f"chunk_{i}",
                document_id="doc_1",
                text=text,
                embedding=None,
                metadata={'chunk_index': i, 'document_id': 'doc_1'}
            )
            
            # Give higher initial scores to irrelevant chunks
            retrieved_chunk = RetrievedChunk(
                chunk=chunk,
                score=1.0 - i * 0.1,  # Decreasing scores
                document_name="test.pdf",
                page_number=i + 1
            )
            
            chunks.append(retrieved_chunk)
        
        query = "What is machine learning?"
        
        # Rerank
        reranked = reranker.rerank(query, chunks, top_n=3)
        
        # After reranking, relevant chunks should be at the top
        top_texts = [chunk.chunk.text.lower() for chunk in reranked]
        
        # At least one of the top 2 should be about ML/AI
        assert any('machine learning' in text or 'deep learning' in text for text in top_texts[:2])
    
    def test_rerank_with_single_chunk(self, reranker):
        """Test reranking with a single chunk."""
        chunk = Chunk(
            id="chunk_1",
            document_id="doc_1",
            text="Machine learning is a subset of AI.",
            embedding=None,
            metadata={'chunk_index': 0, 'document_id': 'doc_1'}
        )
        
        retrieved_chunk = RetrievedChunk(
            chunk=chunk,
            score=0.8,
            document_name="test.pdf",
            page_number=1
        )
        
        query = "What is machine learning?"
        
        reranked = reranker.rerank(query, [retrieved_chunk], top_n=1)
        
        assert len(reranked) == 1
        assert reranked[0].chunk.id == "chunk_1"
    
    def test_rerank_top_n_less_than_chunks(self, reranker, sample_chunks):
        """Test that rerank correctly limits results to top_n."""
        query = "artificial intelligence"
        
        # Request fewer chunks than available
        top_n = 2
        reranked = reranker.rerank(query, sample_chunks, top_n=top_n)
        
        assert len(reranked) == top_n
        
        # Should return the most relevant chunks
        scores = [chunk.score for chunk in reranked]
        assert scores == sorted(scores, reverse=True)
    
    def test_rerank_consistency(self, reranker, sample_chunks):
        """Test that reranking is consistent across multiple calls."""
        query = "machine learning and neural networks"
        top_n = 3
        
        # Rerank twice
        reranked1 = reranker.rerank(query, sample_chunks, top_n=top_n)
        reranked2 = reranker.rerank(query, sample_chunks, top_n=top_n)
        
        # Should return same chunks in same order
        assert len(reranked1) == len(reranked2)
        
        for i in range(len(reranked1)):
            assert reranked1[i].chunk.id == reranked2[i].chunk.id
            # Scores should be very close (within floating point precision)
            assert abs(reranked1[i].score - reranked2[i].score) < 1e-5
    
    def test_rerank_preserves_chunk_text(self, reranker, sample_chunks):
        """Test that chunk text is not modified during reranking."""
        query = "test query"
        
        # Store original texts
        original_texts = {chunk.chunk.id: chunk.chunk.text for chunk in sample_chunks}
        
        # Rerank
        reranked = reranker.rerank(query, sample_chunks, top_n=5)
        
        # Check that texts are unchanged
        for chunk in reranked:
            assert chunk.chunk.text == original_texts[chunk.chunk.id]
    
    def test_rerank_preserves_document_info(self, reranker, sample_chunks):
        """Test that document information is preserved."""
        query = "test query"
        
        # Store original document info
        original_info = {
            chunk.chunk.id: (chunk.document_name, chunk.page_number)
            for chunk in sample_chunks
        }
        
        # Rerank
        reranked = reranker.rerank(query, sample_chunks, top_n=5)
        
        # Check that document info is unchanged
        for chunk in reranked:
            original_name, original_page = original_info[chunk.chunk.id]
            assert chunk.document_name == original_name
            assert chunk.page_number == original_page
    
    def test_score_pairs_with_multiple_texts(self, reranker):
        """Test score_pairs with multiple texts."""
        query = "machine learning"
        texts = [
            "Machine learning is a field of AI.",
            "Deep learning is a subset of machine learning.",
            "The weather is nice.",
            "Natural language processing uses machine learning.",
            "Pizza is delicious."
        ]
        
        scores = reranker.score_pairs(query, texts)
        
        assert len(scores) == len(texts)
        
        # ML-related texts should have higher scores
        ml_indices = [0, 1, 3]  # Indices of ML-related texts
        non_ml_indices = [2, 4]  # Indices of non-ML texts
        
        avg_ml_score = sum(scores[i] for i in ml_indices) / len(ml_indices)
        avg_non_ml_score = sum(scores[i] for i in non_ml_indices) / len(non_ml_indices)
        
        # Average ML score should be higher
        assert avg_ml_score > avg_non_ml_score
