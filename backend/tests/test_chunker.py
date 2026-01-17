"""
Unit tests for the TextChunker class.

Tests chunking logic, overlap handling, metadata preservation, and edge cases.
"""
import pytest
import sys
from pathlib import Path
from hypothesis import given, strategies as st, settings

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunker import TextChunker
from src.models import Chunk


class TestTextChunkerInitialization:
    """Tests for TextChunker initialization and validation."""
    
    def test_default_initialization(self):
        """Test creating TextChunker with default parameters."""
        chunker = TextChunker()
        assert chunker.chunk_size == 512
        assert chunker.overlap == 50
    
    def test_custom_initialization(self):
        """Test creating TextChunker with custom parameters."""
        chunker = TextChunker(chunk_size=256, overlap=25)
        assert chunker.chunk_size == 256
        assert chunker.overlap == 25
    
    def test_invalid_chunk_size_zero(self):
        """Test that chunk_size of 0 raises ValueError."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            TextChunker(chunk_size=0, overlap=10)
    
    def test_invalid_chunk_size_negative(self):
        """Test that negative chunk_size raises ValueError."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            TextChunker(chunk_size=-100, overlap=10)
    
    def test_invalid_overlap_negative(self):
        """Test that negative overlap raises ValueError."""
        with pytest.raises(ValueError, match="overlap must be non-negative"):
            TextChunker(chunk_size=100, overlap=-10)
    
    def test_invalid_overlap_greater_than_chunk_size(self):
        """Test that overlap >= chunk_size raises ValueError."""
        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            TextChunker(chunk_size=100, overlap=100)
        
        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            TextChunker(chunk_size=100, overlap=150)


class TestTextChunkerEdgeCases:
    """Tests for edge cases in text chunking."""
    
    def test_empty_text(self):
        """Test chunking empty text returns empty list."""
        chunker = TextChunker(chunk_size=100, overlap=10)
        chunks = chunker.chunk_text("", document_id="doc1")
        assert chunks == []
    
    def test_whitespace_only_text(self):
        """Test chunking whitespace-only text returns empty list."""
        chunker = TextChunker(chunk_size=100, overlap=10)
        chunks = chunker.chunk_text("   \n\t  ", document_id="doc1")
        assert chunks == []
    
    def test_text_shorter_than_chunk_size(self):
        """Test chunking text shorter than chunk_size returns single chunk."""
        chunker = TextChunker(chunk_size=100, overlap=10)
        text = "This is a short text."
        chunks = chunker.chunk_text(text, document_id="doc1")
        
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].document_id == "doc1"
        assert chunks[0].metadata['chunk_index'] == 0
    
    def test_text_exactly_chunk_size(self):
        """Test chunking text exactly equal to chunk_size."""
        chunker = TextChunker(chunk_size=20, overlap=5)
        text = "12345678901234567890"  # Exactly 20 characters
        chunks = chunker.chunk_text(text, document_id="doc1")
        
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert len(chunks[0].text) == 20
    
    def test_text_one_character_over_chunk_size(self):
        """Test chunking text one character over chunk_size creates two chunks."""
        chunker = TextChunker(chunk_size=20, overlap=5)
        text = "123456789012345678901"  # 21 characters
        chunks = chunker.chunk_text(text, document_id="doc1")
        
        assert len(chunks) == 2
        assert chunks[0].text == "12345678901234567890"  # First 20 chars
        assert chunks[1].text == "678901"  # Last 6 chars (from position 15 due to overlap)
    
    def test_text_exact_multiple_of_chunk_size(self):
        """Test chunking text that is an exact multiple of chunk_size."""
        chunker = TextChunker(chunk_size=20, overlap=5)
        text = "A" * 40  # Exactly 2x chunk_size
        chunks = chunker.chunk_text(text, document_id="doc1")
        
        # With overlap=5, we expect:
        # Chunk 0: chars 0-19 (20 chars)
        # Chunk 1: chars 15-34 (20 chars) 
        # Chunk 2: chars 30-39 (10 chars)
        assert len(chunks) == 3
        assert len(chunks[0].text) == 20
        assert len(chunks[1].text) == 20
        assert len(chunks[2].text) == 10
        
        # Verify overlap between chunks
        assert chunks[0].text[-5:] == chunks[1].text[:5]
        assert chunks[1].text[-5:] == chunks[2].text[:5]
    
    def test_text_exact_multiple_with_zero_overlap(self):
        """Test chunking text that is exact multiple with zero overlap."""
        chunker = TextChunker(chunk_size=20, overlap=0)
        text = "A" * 60  # Exactly 3x chunk_size
        chunks = chunker.chunk_text(text, document_id="doc1")
        
        # With zero overlap, should create exactly 3 chunks
        assert len(chunks) == 3
        assert all(len(chunk.text) == 20 for chunk in chunks)
        
        # Verify no overlap (chunks are adjacent)
        full_text = "".join(chunk.text for chunk in chunks)
        assert full_text == text


class TestTextChunkerBasicFunctionality:
    """Tests for basic chunking functionality."""
    
    def test_chunk_text_creates_chunks(self):
        """Test that chunk_text creates Chunk objects."""
        chunker = TextChunker(chunk_size=50, overlap=10)
        text = "This is a test document that will be split into multiple chunks for testing purposes."
        chunks = chunker.chunk_text(text, document_id="doc1")
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
    
    def test_chunk_text_preserves_document_id(self):
        """Test that all chunks have the correct document_id."""
        chunker = TextChunker(chunk_size=50, overlap=10)
        text = "This is a test document that will be split into multiple chunks."
        chunks = chunker.chunk_text(text, document_id="doc123")
        
        assert all(chunk.document_id == "doc123" for chunk in chunks)
    
    def test_chunk_text_assigns_sequential_indices(self):
        """Test that chunks have sequential chunk_index values."""
        chunker = TextChunker(chunk_size=30, overlap=5)
        text = "This is a longer text that will definitely be split into multiple chunks for testing."
        chunks = chunker.chunk_text(text, document_id="doc1")
        
        for i, chunk in enumerate(chunks):
            assert chunk.metadata['chunk_index'] == i
    
    def test_chunk_text_respects_chunk_size(self):
        """Test that all chunks (except possibly the last) respect chunk_size."""
        chunker = TextChunker(chunk_size=50, overlap=10)
        text = "A" * 200  # 200 characters
        chunks = chunker.chunk_text(text, document_id="doc1")
        
        # All chunks except the last should be exactly chunk_size
        for chunk in chunks[:-1]:
            assert len(chunk.text) == 50
        
        # Last chunk can be <= chunk_size
        assert len(chunks[-1].text) <= 50
    
    def test_chunk_text_preserves_base_metadata(self):
        """Test that base metadata is preserved in all chunks."""
        chunker = TextChunker(chunk_size=50, overlap=10)
        text = "This is a test document with metadata."
        base_metadata = {
            "page_number": 5,
            "source": "test.pdf",
            "author": "Test Author"
        }
        chunks = chunker.chunk_text(text, document_id="doc1", metadata=base_metadata)
        
        for chunk in chunks:
            assert chunk.metadata['page_number'] == 5
            assert chunk.metadata['source'] == "test.pdf"
            assert chunk.metadata['author'] == "Test Author"
            assert 'chunk_index' in chunk.metadata


class TestTextChunkerOverlap:
    """Tests for overlap functionality."""
    
    def test_chunks_have_overlap(self):
        """Test that consecutive chunks have overlapping content."""
        chunker = TextChunker(chunk_size=20, overlap=5)
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"  # 32 characters
        chunks = chunker.chunk_text(text, document_id="doc1")
        
        # Should create 2 chunks with overlap
        assert len(chunks) >= 2
        
        # Check overlap between first two chunks
        chunk1_text = chunks[0].text
        chunk2_text = chunks[1].text
        
        # Last 5 characters of chunk1 should match first 5 of chunk2
        overlap_from_chunk1 = chunk1_text[-5:]
        overlap_from_chunk2 = chunk2_text[:5]
        
        assert overlap_from_chunk1 == overlap_from_chunk2
    
    def test_overlap_size_is_correct(self):
        """Test that overlap size matches configuration."""
        chunker = TextChunker(chunk_size=30, overlap=10)
        text = "A" * 100  # 100 characters
        chunks = chunker.chunk_text(text, document_id="doc1")
        
        # Check overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            chunk1_text = chunks[i].text
            chunk2_text = chunks[i + 1].text
            
            # Last 10 characters of chunk1 should match first 10 of chunk2
            if len(chunk1_text) >= 10 and len(chunk2_text) >= 10:
                overlap_from_chunk1 = chunk1_text[-10:]
                overlap_from_chunk2 = chunk2_text[:10]
                assert overlap_from_chunk1 == overlap_from_chunk2
    
    def test_zero_overlap(self):
        """Test chunking with zero overlap."""
        chunker = TextChunker(chunk_size=20, overlap=0)
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"  # 32 characters
        chunks = chunker.chunk_text(text, document_id="doc1")
        
        # With zero overlap, chunks should be adjacent but not overlapping
        assert len(chunks) == 2
        assert chunks[0].text == "ABCDEFGHIJKLMNOPQRST"  # First 20
        assert chunks[1].text == "UVWXYZ012345"  # Remaining 12


class TestChunkWithOverlap:
    """Tests for the chunk_with_overlap method (simpler version)."""
    
    def test_chunk_with_overlap_returns_strings(self):
        """Test that chunk_with_overlap returns list of strings."""
        chunker = TextChunker(chunk_size=50, overlap=10)
        text = "This is a test document that will be split into chunks."
        chunks = chunker.chunk_with_overlap(text)
        
        assert isinstance(chunks, list)
        assert all(isinstance(chunk, str) for chunk in chunks)
    
    def test_chunk_with_overlap_empty_text(self):
        """Test chunk_with_overlap with empty text."""
        chunker = TextChunker(chunk_size=50, overlap=10)
        chunks = chunker.chunk_with_overlap("")
        assert chunks == []
    
    def test_chunk_with_overlap_short_text(self):
        """Test chunk_with_overlap with text shorter than chunk_size."""
        chunker = TextChunker(chunk_size=100, overlap=10)
        text = "Short text"
        chunks = chunker.chunk_with_overlap(text)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_chunk_with_overlap_has_overlap(self):
        """Test that chunk_with_overlap creates overlapping chunks."""
        chunker = TextChunker(chunk_size=20, overlap=5)
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        chunks = chunker.chunk_with_overlap(text)
        
        # Check overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            overlap_from_chunk1 = chunks[i][-5:]
            overlap_from_chunk2 = chunks[i + 1][:5]
            assert overlap_from_chunk1 == overlap_from_chunk2


class TestChunkIdGeneration:
    """Tests for chunk ID generation."""
    
    def test_chunk_ids_are_unique(self):
        """Test that each chunk gets a unique ID."""
        chunker = TextChunker(chunk_size=30, overlap=5)
        text = "This is a test document that will be split into multiple chunks."
        chunks = chunker.chunk_text(text, document_id="doc1")
        
        chunk_ids = [chunk.id for chunk in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))  # All IDs are unique
    
    def test_chunk_ids_contain_document_id(self):
        """Test that chunk IDs contain the document ID."""
        chunker = TextChunker(chunk_size=30, overlap=5)
        text = "This is a test document."
        chunks = chunker.chunk_text(text, document_id="doc123")
        
        for chunk in chunks:
            assert "doc123" in chunk.id
    
    def test_chunk_ids_contain_chunk_index(self):
        """Test that chunk IDs contain the chunk index."""
        chunker = TextChunker(chunk_size=30, overlap=5)
        text = "This is a test document that will be split into multiple chunks."
        chunks = chunker.chunk_text(text, document_id="doc1")
        
        for i, chunk in enumerate(chunks):
            assert f"chunk_{i}" in chunk.id



# ============================================================================
# Property-Based Tests
# ============================================================================

class TestChunkerProperties:
    """Property-based tests for TextChunker using Hypothesis."""
    
    # Feature: ai-knowledge-copilot, Property 3: Chunk size constraint
    @given(
        text=st.text(min_size=100, max_size=5000),
        chunk_size=st.integers(min_value=50, max_value=500),
        overlap=st.integers(min_value=0, max_value=49)
    )
    @settings(max_examples=100)
    def test_chunk_size_constraint(self, text, chunk_size, overlap):
        """
        Property 3: Chunk size constraint
        
        For any document and configured chunk size, all generated chunks 
        (except possibly the last) should not exceed the specified size.
        
        **Validates: Requirements 2.1**
        """
        # Ensure overlap is less than chunk_size
        if overlap >= chunk_size:
            overlap = chunk_size - 1
        
        # Skip empty or whitespace-only text
        if not text or not text.strip():
            return
        
        # Create chunker and chunk the text
        chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
        chunks = chunker.chunk_text(text, document_id="test_doc")
        
        # Property: All chunks except the last should not exceed chunk_size
        for i, chunk in enumerate(chunks[:-1]):
            assert len(chunk.text) <= chunk_size, (
                f"Chunk {i} has length {len(chunk.text)}, "
                f"which exceeds chunk_size {chunk_size}"
            )
        
        # The last chunk can be any size <= chunk_size
        if chunks:
            assert len(chunks[-1].text) <= chunk_size, (
                f"Last chunk has length {len(chunks[-1].text)}, "
                f"which exceeds chunk_size {chunk_size}"
            )
    
    # Feature: ai-knowledge-copilot, Property 4: Chunk overlap preservation
    @given(
        text=st.text(min_size=200, max_size=5000),
        chunk_size=st.integers(min_value=100, max_value=500),
        overlap=st.integers(min_value=10, max_value=99)
    )
    @settings(max_examples=100)
    def test_chunk_overlap_preservation(self, text, chunk_size, overlap):
        """
        Property 4: Chunk overlap preservation
        
        For any two consecutive chunks from the same document, there should be 
        overlapping text content of approximately the configured overlap size.
        
        **Validates: Requirements 2.2**
        """
        # Ensure overlap is less than chunk_size
        if overlap >= chunk_size:
            overlap = chunk_size - 1
        
        # Skip empty or whitespace-only text
        if not text or not text.strip():
            return
        
        # Create chunker and chunk the text
        chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
        chunks = chunker.chunk_text(text, document_id="test_doc")
        
        # If we only have one chunk, no overlap to test
        if len(chunks) <= 1:
            return
        
        # Property: For any two consecutive chunks, there should be overlap
        for i in range(len(chunks) - 1):
            chunk1_text = chunks[i].text
            chunk2_text = chunks[i + 1].text
            
            # The overlap should be approximately the configured overlap size
            # We check if the end of chunk1 matches the beginning of chunk2
            
            # Calculate the expected overlap size
            # It should be min(overlap, len(chunk1_text), len(chunk2_text))
            expected_overlap = min(overlap, len(chunk1_text), len(chunk2_text))
            
            # Extract the overlapping portions
            overlap_from_chunk1 = chunk1_text[-expected_overlap:]
            overlap_from_chunk2 = chunk2_text[:expected_overlap]
            
            # Assert that the overlapping portions match
            assert overlap_from_chunk1 == overlap_from_chunk2, (
                f"Chunks {i} and {i+1} do not have proper overlap. "
                f"Expected overlap size: {expected_overlap}, "
                f"Chunk {i} end: '{overlap_from_chunk1}', "
                f"Chunk {i+1} start: '{overlap_from_chunk2}'"
            )
