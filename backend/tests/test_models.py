"""
Unit tests for data models.

Tests validation rules, serialization, and type hints for all Pydantic models.
"""
import pytest
import numpy as np
from datetime import datetime
from hypothesis import given, strategies as st
import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    Document,
    Chunk,
    RetrievedChunk,
    Citation,
    QueryResponse,
    APIKeyConfig,
    DocumentDetails
)


class TestDocument:
    """Tests for Document model."""
    
    def test_valid_document_creation(self):
        """Test creating a valid document."""
        doc = Document(
            id="doc1",
            source="/path/to/file.pdf",
            source_type="pdf",
            text="Sample document text",
            metadata={"title": "Test Document"}
        )
        assert doc.id == "doc1"
        assert doc.source_type == "pdf"
        assert doc.text == "Sample document text"
        assert doc.metadata["title"] == "Test Document"
        assert isinstance(doc.created_at, datetime)
    
    def test_invalid_source_type(self):
        """Test that invalid source_type raises ValueError."""
        with pytest.raises(ValueError, match="source_type must be 'pdf' or 'web'"):
            Document(
                id="doc1",
                source="/path/to/file.txt",
                source_type="txt",
                text="Sample text"
            )
    
    def test_empty_text_validation(self):
        """Test that empty text raises ValueError."""
        with pytest.raises(ValueError, match="text content cannot be empty"):
            Document(
                id="doc1",
                source="/path/to/file.pdf",
                source_type="pdf",
                text=""
            )
    
    def test_whitespace_only_text_validation(self):
        """Test that whitespace-only text raises ValueError."""
        with pytest.raises(ValueError, match="text content cannot be empty"):
            Document(
                id="doc1",
                source="/path/to/file.pdf",
                source_type="pdf",
                text="   \n\t  "
            )
    
    def test_web_source_type(self):
        """Test creating a document with web source type."""
        doc = Document(
            id="doc2",
            source="https://example.com/article",
            source_type="web",
            text="Web content text"
        )
        assert doc.source_type == "web"
        assert doc.source == "https://example.com/article"


class TestChunk:
    """Tests for Chunk model."""
    
    def test_valid_chunk_creation(self):
        """Test creating a valid chunk."""
        chunk = Chunk(
            id="chunk1",
            document_id="doc1",
            text="This is a chunk of text",
            metadata={"chunk_index": 0, "page_number": 1}
        )
        assert chunk.id == "chunk1"
        assert chunk.document_id == "doc1"
        assert chunk.text == "This is a chunk of text"
        assert chunk.metadata["chunk_index"] == 0
        assert chunk.embedding is None
    
    def test_chunk_with_embedding(self):
        """Test creating a chunk with an embedding vector."""
        embedding = np.array([0.1, 0.2, 0.3, 0.4])
        chunk = Chunk(
            id="chunk1",
            document_id="doc1",
            text="Text with embedding",
            embedding=embedding,
            metadata={"chunk_index": 0}
        )
        assert chunk.embedding is not None
        assert np.array_equal(chunk.embedding, embedding)
    
    def test_empty_chunk_text_validation(self):
        """Test that empty chunk text raises ValueError."""
        with pytest.raises(ValueError, match="chunk text cannot be empty"):
            Chunk(
                id="chunk1",
                document_id="doc1",
                text="",
                metadata={"chunk_index": 0}
            )
    
    def test_metadata_completeness_validation(self):
        """Test that metadata must contain chunk_index."""
        with pytest.raises(ValueError, match="metadata must contain 'chunk_index'"):
            Chunk(
                id="chunk1",
                document_id="doc1",
                text="Some text",
                metadata={"page_number": 1}  # Missing chunk_index
            )
    
    def test_chunk_serialization_with_embedding(self):
        """Test that chunk with embedding can be serialized."""
        embedding = np.array([0.1, 0.2, 0.3])
        chunk = Chunk(
            id="chunk1",
            document_id="doc1",
            text="Text",
            embedding=embedding,
            metadata={"chunk_index": 0}
        )
        data = chunk.model_dump()
        assert isinstance(data['embedding'], list)
        assert data['embedding'] == [0.1, 0.2, 0.3]
    
    def test_chunk_deserialization_with_embedding(self):
        """Test that chunk with embedding list can be deserialized."""
        data = {
            'id': 'chunk1',
            'document_id': 'doc1',
            'text': 'Text',
            'embedding': [0.1, 0.2, 0.3],
            'metadata': {'chunk_index': 0}
        }
        chunk = Chunk.model_validate(data)
        assert isinstance(chunk.embedding, np.ndarray)
        assert np.array_equal(chunk.embedding, np.array([0.1, 0.2, 0.3]))


class TestRetrievedChunk:
    """Tests for RetrievedChunk model."""
    
    def test_valid_retrieved_chunk_creation(self):
        """Test creating a valid retrieved chunk."""
        chunk = Chunk(
            id="chunk1",
            document_id="doc1",
            text="Retrieved text",
            metadata={"chunk_index": 0}
        )
        retrieved = RetrievedChunk(
            chunk=chunk,
            score=0.85,
            document_name="test.pdf",
            page_number=5
        )
        assert retrieved.chunk.id == "chunk1"
        assert retrieved.score == 0.85
        assert retrieved.document_name == "test.pdf"
        assert retrieved.page_number == 5
    
    def test_retrieved_chunk_without_page_number(self):
        """Test creating a retrieved chunk without page number."""
        chunk = Chunk(
            id="chunk1",
            document_id="doc1",
            text="Text",
            metadata={"chunk_index": 0}
        )
        retrieved = RetrievedChunk(
            chunk=chunk,
            score=0.75,
            document_name="web_article"
        )
        assert retrieved.page_number is None
    
    def test_negative_score_validation(self):
        """Test that negative score raises ValueError."""
        chunk = Chunk(
            id="chunk1",
            document_id="doc1",
            text="Text",
            metadata={"chunk_index": 0}
        )
        with pytest.raises(ValueError, match="score must be non-negative"):
            RetrievedChunk(
                chunk=chunk,
                score=-0.5,
                document_name="test.pdf"
            )


class TestCitation:
    """Tests for Citation model."""
    
    def test_valid_citation_creation(self):
        """Test creating a valid citation."""
        citation = Citation(
            document_name="research_paper.pdf",
            page_number=10,
            excerpt="This is the relevant excerpt from the document.",
            chunk_id="chunk123"
        )
        assert citation.document_name == "research_paper.pdf"
        assert citation.page_number == 10
        assert citation.excerpt == "This is the relevant excerpt from the document."
        assert citation.chunk_id == "chunk123"
    
    def test_citation_without_page_number(self):
        """Test creating a citation without page number (e.g., for web content)."""
        citation = Citation(
            document_name="web_article",
            excerpt="Excerpt from web content.",
            chunk_id="chunk456"
        )
        assert citation.page_number is None
    
    def test_empty_document_name_validation(self):
        """Test that empty document_name raises ValueError."""
        with pytest.raises(ValueError, match="document_name is required and cannot be empty"):
            Citation(
                document_name="",
                excerpt="Some excerpt",
                chunk_id="chunk1"
            )
    
    def test_whitespace_document_name_validation(self):
        """Test that whitespace-only document_name raises ValueError."""
        with pytest.raises(ValueError, match="document_name is required and cannot be empty"):
            Citation(
                document_name="   ",
                excerpt="Some excerpt",
                chunk_id="chunk1"
            )
    
    def test_empty_excerpt_validation(self):
        """Test that empty excerpt raises ValueError."""
        with pytest.raises(ValueError, match="excerpt is required and cannot be empty"):
            Citation(
                document_name="test.pdf",
                excerpt="",
                chunk_id="chunk1"
            )
    
    def test_whitespace_excerpt_validation(self):
        """Test that whitespace-only excerpt raises ValueError."""
        with pytest.raises(ValueError, match="excerpt is required and cannot be empty"):
            Citation(
                document_name="test.pdf",
                excerpt="   \n  ",
                chunk_id="chunk1"
            )


class TestQueryResponse:
    """Tests for QueryResponse model."""
    
    def test_valid_query_response_creation(self):
        """Test creating a valid query response."""
        citation = Citation(
            document_name="test.pdf",
            page_number=5,
            excerpt="Relevant excerpt",
            chunk_id="chunk1"
        )
        response = QueryResponse(
            answer="This is the answer to your question.",
            citations=[citation],
            retrieved_chunks=10,
            confidence=0.85
        )
        assert response.answer == "This is the answer to your question."
        assert len(response.citations) == 1
        assert response.retrieved_chunks == 10
        assert response.confidence == 0.85
    
    def test_query_response_without_citations(self):
        """Test creating a query response without citations."""
        response = QueryResponse(
            answer="I don't have enough information to answer this question.",
            citations=[],
            retrieved_chunks=0,
            confidence=0.0
        )
        assert len(response.citations) == 0
        assert response.retrieved_chunks == 0
    
    def test_confidence_range_validation_too_high(self):
        """Test that confidence > 1 raises ValueError."""
        with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
            QueryResponse(
                answer="Answer",
                citations=[],
                retrieved_chunks=5,
                confidence=1.5
            )
    
    def test_confidence_range_validation_too_low(self):
        """Test that confidence < 0 raises ValueError."""
        with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
            QueryResponse(
                answer="Answer",
                citations=[],
                retrieved_chunks=5,
                confidence=-0.1
            )
    
    def test_negative_retrieved_chunks_validation(self):
        """Test that negative retrieved_chunks raises ValueError."""
        with pytest.raises(ValueError, match="retrieved_chunks must be non-negative"):
            QueryResponse(
                answer="Answer",
                citations=[],
                retrieved_chunks=-1,
                confidence=0.5
            )


class TestAPIKeyConfig:
    """Tests for APIKeyConfig model."""
    
    def test_valid_openai_config(self):
        """Test creating a valid OpenAI configuration."""
        config = APIKeyConfig(
            provider="openai",
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1000
        )
        assert config.provider == "openai"
        assert config.model_name == "gpt-3.5-turbo"
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
    
    def test_valid_huggingface_config(self):
        """Test creating a valid Hugging Face configuration."""
        config = APIKeyConfig(
            provider="huggingface",
            model_name="meta-llama/Llama-2-7b-chat-hf"
        )
        assert config.provider == "huggingface"
        assert config.temperature == 0.1  # Default value
        assert config.max_tokens == 500  # Default value
    
    def test_invalid_provider_validation(self):
        """Test that invalid provider raises ValueError."""
        with pytest.raises(ValueError, match="provider must be 'openai' or 'huggingface'"):
            APIKeyConfig(
                provider="anthropic",
                model_name="claude-2"
            )
    
    def test_temperature_too_high_validation(self):
        """Test that temperature > 2 raises ValueError."""
        with pytest.raises(ValueError, match="temperature must be between 0 and 2"):
            APIKeyConfig(
                provider="openai",
                model_name="gpt-3.5-turbo",
                temperature=2.5
            )
    
    def test_temperature_negative_validation(self):
        """Test that negative temperature raises ValueError."""
        with pytest.raises(ValueError, match="temperature must be between 0 and 2"):
            APIKeyConfig(
                provider="openai",
                model_name="gpt-3.5-turbo",
                temperature=-0.1
            )
    
    def test_max_tokens_zero_validation(self):
        """Test that max_tokens = 0 raises ValueError."""
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            APIKeyConfig(
                provider="openai",
                model_name="gpt-3.5-turbo",
                max_tokens=0
            )
    
    def test_max_tokens_negative_validation(self):
        """Test that negative max_tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            APIKeyConfig(
                provider="openai",
                model_name="gpt-3.5-turbo",
                max_tokens=-100
            )


class TestDocumentDetails:
    """Tests for DocumentDetails model."""
    
    def test_valid_document_details_creation(self):
        """Test creating valid document details."""
        upload_date = datetime.now()
        details = DocumentDetails(
            id="doc1",
            name="research_paper.pdf",
            source_type="pdf",
            upload_date=upload_date,
            chunk_count=25,
            total_size=1024000
        )
        assert details.id == "doc1"
        assert details.name == "research_paper.pdf"
        assert details.source_type == "pdf"
        assert details.upload_date == upload_date
        assert details.chunk_count == 25
        assert details.total_size == 1024000
    
    def test_web_source_type_details(self):
        """Test creating document details for web content."""
        details = DocumentDetails(
            id="doc2",
            name="article",
            source_type="web",
            upload_date=datetime.now(),
            chunk_count=10,
            total_size=50000
        )
        assert details.source_type == "web"
    
    def test_invalid_source_type_validation(self):
        """Test that invalid source_type raises ValueError."""
        with pytest.raises(ValueError, match="source_type must be 'pdf' or 'web'"):
            DocumentDetails(
                id="doc1",
                name="file.txt",
                source_type="txt",
                upload_date=datetime.now(),
                chunk_count=5,
                total_size=1000
            )
    
    def test_negative_chunk_count_validation(self):
        """Test that negative chunk_count raises ValueError."""
        with pytest.raises(ValueError, match="chunk_count must be non-negative"):
            DocumentDetails(
                id="doc1",
                name="test.pdf",
                source_type="pdf",
                upload_date=datetime.now(),
                chunk_count=-5,
                total_size=1000
            )
    
    def test_negative_total_size_validation(self):
        """Test that negative total_size raises ValueError."""
        with pytest.raises(ValueError, match="total_size must be non-negative"):
            DocumentDetails(
                id="doc1",
                name="test.pdf",
                source_type="pdf",
                upload_date=datetime.now(),
                chunk_count=10,
                total_size=-1000
            )



# Property-Based Tests
# Feature: ai-knowledge-copilot

class TestChunkMetadataProperties:
    """Property-based tests for Chunk metadata completeness."""
    
    # Feature: ai-knowledge-copilot, Property 5: Chunk metadata completeness
    @given(
        chunk_id=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        document_id=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        text=st.text(min_size=1, max_size=1000).filter(lambda x: x.strip()),
        chunk_index=st.integers(min_value=0, max_value=10000),
        page_number=st.integers(min_value=1, max_value=1000),
        source=st.text(min_size=1, max_size=200).filter(lambda x: x.strip())
    )
    def test_chunk_metadata_completeness(
        self, 
        chunk_id: str, 
        document_id: str, 
        text: str, 
        chunk_index: int,
        page_number: int,
        source: str
    ):
        """
        Property 5: Chunk metadata completeness
        
        For any generated chunk, the metadata should contain document_id, 
        source information, and chunk_index fields.
        
        Validates: Requirements 2.3
        """
        # Create metadata with required fields
        metadata = {
            'chunk_index': chunk_index,
            'page_number': page_number,
            'source': source
        }
        
        # Create chunk - should succeed with complete metadata
        chunk = Chunk(
            id=chunk_id,
            document_id=document_id,
            text=text,
            metadata=metadata
        )
        
        # Verify all required metadata fields are present
        assert 'chunk_index' in chunk.metadata, "metadata must contain 'chunk_index'"
        assert chunk.metadata['chunk_index'] == chunk_index
        
        # Verify document_id is accessible (it's a separate field, not in metadata)
        assert chunk.document_id == document_id
        
        # Verify source information is preserved in metadata
        assert 'source' in chunk.metadata
        assert chunk.metadata['source'] == source
    
    # Feature: ai-knowledge-copilot, Property 5: Chunk metadata completeness
    @given(
        chunk_id=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        document_id=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        text=st.text(min_size=1, max_size=1000).filter(lambda x: x.strip())
    )
    def test_chunk_metadata_missing_chunk_index_fails(
        self,
        chunk_id: str,
        document_id: str,
        text: str
    ):
        """
        Property 5: Chunk metadata completeness (negative test)
        
        For any chunk without chunk_index in metadata, validation should fail.
        
        Validates: Requirements 2.3
        """
        # Create metadata WITHOUT chunk_index
        metadata = {
            'page_number': 1,
            'source': 'test.pdf'
        }
        
        # Attempt to create chunk - should fail validation
        with pytest.raises(ValueError, match="metadata must contain 'chunk_index'"):
            Chunk(
                id=chunk_id,
                document_id=document_id,
                text=text,
                metadata=metadata
            )


class TestCitationProperties:
    """Property-based tests for Citation field completeness."""
    
    # Feature: ai-knowledge-copilot, Property 17: Citation field completeness
    @given(
        document_name=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
        excerpt=st.text(min_size=1, max_size=500).filter(lambda x: x.strip()),
        chunk_id=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        page_number=st.one_of(st.none(), st.integers(min_value=1, max_value=10000))
    )
    def test_citation_field_completeness(
        self,
        document_name: str,
        excerpt: str,
        chunk_id: str,
        page_number: int
    ):
        """
        Property 17: Citation field completeness
        
        For any citation in a response, it should include document_name 
        and excerpt fields at minimum.
        
        Validates: Requirements 5.3
        """
        # Create citation with all required fields
        citation = Citation(
            document_name=document_name,
            page_number=page_number,
            excerpt=excerpt,
            chunk_id=chunk_id
        )
        
        # Verify required fields are present and non-empty
        assert citation.document_name is not None
        assert citation.document_name.strip() != ""
        assert citation.excerpt is not None
        assert citation.excerpt.strip() != ""
        assert citation.chunk_id is not None
        
        # Verify the values match what was provided
        assert citation.document_name == document_name
        assert citation.excerpt == excerpt
        assert citation.chunk_id == chunk_id
        assert citation.page_number == page_number
    
    # Feature: ai-knowledge-copilot, Property 17: Citation field completeness
    @given(
        excerpt=st.text(min_size=1, max_size=500).filter(lambda x: x.strip()),
        chunk_id=st.text(min_size=1, max_size=50).filter(lambda x: x.strip())
    )
    def test_citation_missing_document_name_fails(
        self,
        excerpt: str,
        chunk_id: str
    ):
        """
        Property 17: Citation field completeness (negative test)
        
        For any citation without document_name, validation should fail.
        
        Validates: Requirements 5.3
        """
        # Attempt to create citation with empty document_name - should fail
        with pytest.raises(ValueError, match="document_name is required and cannot be empty"):
            Citation(
                document_name="",
                excerpt=excerpt,
                chunk_id=chunk_id
            )
        
        # Attempt to create citation with whitespace-only document_name - should fail
        with pytest.raises(ValueError, match="document_name is required and cannot be empty"):
            Citation(
                document_name="   ",
                excerpt=excerpt,
                chunk_id=chunk_id
            )
    
    # Feature: ai-knowledge-copilot, Property 17: Citation field completeness
    @given(
        document_name=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
        chunk_id=st.text(min_size=1, max_size=50).filter(lambda x: x.strip())
    )
    def test_citation_missing_excerpt_fails(
        self,
        document_name: str,
        chunk_id: str
    ):
        """
        Property 17: Citation field completeness (negative test)
        
        For any citation without excerpt, validation should fail.
        
        Validates: Requirements 5.3
        """
        # Attempt to create citation with empty excerpt - should fail
        with pytest.raises(ValueError, match="excerpt is required and cannot be empty"):
            Citation(
                document_name=document_name,
                excerpt="",
                chunk_id=chunk_id
            )
        
        # Attempt to create citation with whitespace-only excerpt - should fail
        with pytest.raises(ValueError, match="excerpt is required and cannot be empty"):
            Citation(
                document_name=document_name,
                excerpt="   \n\t  ",
                chunk_id=chunk_id
            )
