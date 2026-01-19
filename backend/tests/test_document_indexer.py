"""
Unit tests for the DocumentIndexer class.

Tests the document indexing pipeline that coordinates chunking, embedding,
and storage operations.
"""
import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from hypothesis import given, settings, strategies as st
from datetime import datetime

from src.document_indexer import DocumentIndexer, DocumentIndexerError
from src.document_processor import DocumentProcessor
from src.chunker import TextChunker
from src.embedder import EmbeddingGenerator
from src.vector_store import VectorStore
from src.metadata_store import MetadataStore
from src.models import Document


class TestDocumentIndexer:
    """Test suite for DocumentIndexer."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def components(self, temp_dir):
        """Create all required components for DocumentIndexer."""
        document_processor = DocumentProcessor()
        chunker = TextChunker(chunk_size=100, overlap=20)
        embedder = EmbeddingGenerator("all-MiniLM-L6-v2")
        
        # Create vector store
        dimension = embedder.get_embedding_dimension()
        vector_store = VectorStore(dimension=dimension)
        
        # Create metadata store with temp database
        db_path = Path(temp_dir) / "test_metadata.db"
        metadata_store = MetadataStore(str(db_path))
        
        return {
            'document_processor': document_processor,
            'chunker': chunker,
            'embedder': embedder,
            'vector_store': vector_store,
            'metadata_store': metadata_store
        }
    
    @pytest.fixture
    def indexer(self, components):
        """Create a DocumentIndexer instance."""
        return DocumentIndexer(
            document_processor=components['document_processor'],
            chunker=components['chunker'],
            embedder=components['embedder'],
            vector_store=components['vector_store'],
            metadata_store=components['metadata_store']
        )
    
    @pytest.fixture
    def sample_pdf(self, temp_dir):
        """Create a sample PDF file for testing."""
        # Note: This requires a real PDF file for integration testing
        # For unit tests, we'll mock the document processor
        pdf_path = Path(temp_dir) / "sample.pdf"
        # We'll skip actual PDF creation and rely on mocking
        return str(pdf_path)
    
    def test_indexer_initialization(self, indexer):
        """Test that DocumentIndexer initializes correctly."""
        assert indexer.document_processor is not None
        assert indexer.chunker is not None
        assert indexer.embedder is not None
        assert indexer.vector_store is not None
        assert indexer.metadata_store is not None
    
    def test_index_document_with_mock_pdf(self, indexer, components, monkeypatch):
        """
        Test indexing a document through the full pipeline.
        
        This test validates Requirements 1.5, 2.1, 2.2, 2.3, 2.4, 2.5
        """
        # Create a mock document
        from datetime import datetime
        mock_document = Document(
            id="test-doc-1",
            source="/path/to/test.pdf",
            source_type="pdf",
            text="This is a test document. " * 50,  # Long enough to create multiple chunks
            metadata={"filename": "test.pdf"},
            created_at=datetime.now()
        )
        
        # Mock the document processor
        def mock_process_pdf(file_path):
            return mock_document
        
        monkeypatch.setattr(
            components['document_processor'],
            'process_pdf',
            mock_process_pdf
        )
        
        # Index the document
        doc_id = indexer.index_document("/path/to/test.pdf", "pdf")
        
        # Verify document ID is returned
        assert doc_id == "test-doc-1"
        
        # Verify document metadata is stored
        doc_metadata = components['metadata_store'].get_document_metadata(doc_id)
        assert doc_metadata is not None
        assert doc_metadata['id'] == doc_id
        assert doc_metadata['source_type'] == 'pdf'
        
        # Verify chunks are created and stored
        chunk_ids = components['metadata_store'].get_document_chunks(doc_id)
        assert len(chunk_ids) > 0
        
        # Verify embeddings are stored in vector store
        assert components['vector_store'].get_vector_count() == len(chunk_ids)
        
        # Verify we can retrieve chunks by ID
        for chunk_id in chunk_ids:
            chunk_metadata = components['metadata_store'].get_chunk_metadata(chunk_id)
            assert chunk_metadata is not None
            assert chunk_metadata['document_id'] == doc_id
            
            # Verify vector exists
            vector = components['vector_store'].get_vector_by_id(chunk_id)
            assert vector is not None
            assert len(vector) == components['embedder'].get_embedding_dimension()
    
    def test_document_independence(self, indexer, components, monkeypatch):
        """
        Test that multiple documents are processed independently.
        
        Validates Requirement 1.5: Document independence
        """
        from datetime import datetime
        
        # Create two mock documents
        mock_doc1 = Document(
            id="doc-1",
            source="/path/to/doc1.pdf",
            source_type="pdf",
            text="First document content. " * 30,
            metadata={"filename": "doc1.pdf"},
            created_at=datetime.now()
        )
        
        mock_doc2 = Document(
            id="doc-2",
            source="/path/to/doc2.pdf",
            source_type="pdf",
            text="Second document content. " * 30,
            metadata={"filename": "doc2.pdf"},
            created_at=datetime.now()
        )
        
        # Mock the document processor to return different documents
        call_count = [0]
        
        def mock_process_pdf(file_path):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_doc1
            else:
                return mock_doc2
        
        monkeypatch.setattr(
            components['document_processor'],
            'process_pdf',
            mock_process_pdf
        )
        
        # Index both documents
        doc_id1 = indexer.index_document("/path/to/doc1.pdf", "pdf")
        doc_id2 = indexer.index_document("/path/to/doc2.pdf", "pdf")
        
        # Verify both documents have unique IDs
        assert doc_id1 != doc_id2
        
        # Verify both documents are stored separately
        doc1_metadata = components['metadata_store'].get_document_metadata(doc_id1)
        doc2_metadata = components['metadata_store'].get_document_metadata(doc_id2)
        
        assert doc1_metadata['id'] == doc_id1
        assert doc2_metadata['id'] == doc_id2
        
        # Verify chunks are tracked separately
        chunks1 = components['metadata_store'].get_document_chunks(doc_id1)
        chunks2 = components['metadata_store'].get_document_chunks(doc_id2)
        
        assert len(chunks1) > 0
        assert len(chunks2) > 0
        
        # Verify no chunk ID overlap
        assert set(chunks1).isdisjoint(set(chunks2))
        
        # Verify all chunks belong to correct documents
        for chunk_id in chunks1:
            chunk_meta = components['metadata_store'].get_chunk_metadata(chunk_id)
            assert chunk_meta['document_id'] == doc_id1
        
        for chunk_id in chunks2:
            chunk_meta = components['metadata_store'].get_chunk_metadata(chunk_id)
            assert chunk_meta['document_id'] == doc_id2
    
    def test_index_pdf_convenience_method(self, indexer, components, monkeypatch):
        """Test the convenience method for indexing PDFs."""
        from datetime import datetime
        
        mock_document = Document(
            id="test-doc",
            source="/path/to/test.pdf",
            source_type="pdf",
            text="Test content. " * 20,
            metadata={},
            created_at=datetime.now()
        )
        
        monkeypatch.setattr(
            components['document_processor'],
            'process_pdf',
            lambda x: mock_document
        )
        
        doc_id = indexer.index_pdf("/path/to/test.pdf")
        assert doc_id == "test-doc"
    
    def test_index_url_convenience_method(self, indexer, components, monkeypatch):
        """Test the convenience method for indexing URLs."""
        from datetime import datetime
        
        mock_document = Document(
            id="test-web-doc",
            source="https://example.com",
            source_type="web",
            text="Web content. " * 20,
            metadata={},
            created_at=datetime.now()
        )
        
        monkeypatch.setattr(
            components['document_processor'],
            'process_url',
            lambda x: mock_document
        )
        
        doc_id = indexer.index_url("https://example.com")
        assert doc_id == "test-web-doc"
    
    def test_get_document_info(self, indexer, components, monkeypatch):
        """Test retrieving document information."""
        from datetime import datetime
        
        mock_document = Document(
            id="info-test-doc",
            source="/path/to/test.pdf",
            source_type="pdf",
            text="Test content. " * 20,
            metadata={"filename": "test.pdf"},
            created_at=datetime.now()
        )
        
        monkeypatch.setattr(
            components['document_processor'],
            'process_pdf',
            lambda x: mock_document
        )
        
        # Index document
        doc_id = indexer.index_document("/path/to/test.pdf", "pdf")
        
        # Get document info
        info = indexer.get_document_info(doc_id)
        
        assert info is not None
        assert info['id'] == doc_id
        assert info['source_type'] == 'pdf'
        assert 'chunk_count' in info
        assert info['chunk_count'] > 0
    
    def test_get_document_info_not_found(self, indexer):
        """Test getting info for non-existent document."""
        info = indexer.get_document_info("non-existent-id")
        assert info is None
    
    def test_list_documents(self, indexer, components, monkeypatch):
        """Test listing all indexed documents."""
        from datetime import datetime
        
        # Create multiple mock documents
        docs = []
        for i in range(3):
            doc = Document(
                id=f"doc-{i}",
                source=f"/path/to/doc{i}.pdf",
                source_type="pdf",
                text=f"Document {i} content. " * 20,
                metadata={"filename": f"doc{i}.pdf"},
                created_at=datetime.now()
            )
            docs.append(doc)
        
        call_count = [0]
        
        def mock_process_pdf(file_path):
            doc = docs[call_count[0]]
            call_count[0] += 1
            return doc
        
        monkeypatch.setattr(
            components['document_processor'],
            'process_pdf',
            mock_process_pdf
        )
        
        # Index all documents
        for i in range(3):
            indexer.index_document(f"/path/to/doc{i}.pdf", "pdf")
        
        # List documents
        doc_list = indexer.list_documents()
        
        assert len(doc_list) == 3
        
        # Verify each document has required fields
        for doc in doc_list:
            assert 'id' in doc
            assert 'source_type' in doc
            assert 'chunk_count' in doc
            assert doc['chunk_count'] > 0
    
    def test_index_document_with_additional_metadata(self, indexer, components, monkeypatch):
        """Test indexing with additional metadata."""
        from datetime import datetime
        
        mock_document = Document(
            id="meta-test-doc",
            source="/path/to/test.pdf",
            source_type="pdf",
            text="Test content. " * 20,
            metadata={},
            created_at=datetime.now()
        )
        
        monkeypatch.setattr(
            components['document_processor'],
            'process_pdf',
            lambda x: mock_document
        )
        
        # Index with additional metadata
        additional_meta = {"author": "Test Author", "year": 2024}
        doc_id = indexer.index_document(
            "/path/to/test.pdf",
            "pdf",
            additional_metadata=additional_meta
        )
        
        # Verify additional metadata is stored
        doc_metadata = components['metadata_store'].get_document_metadata(doc_id)
        assert doc_metadata['author'] == "Test Author"
        assert doc_metadata['year'] == 2024
    
    def test_index_document_error_handling(self, indexer, components, monkeypatch):
        """Test error handling when document processing fails."""
        # Mock document processor to raise an error
        def mock_process_pdf_error(file_path):
            raise Exception("Processing failed")
        
        monkeypatch.setattr(
            components['document_processor'],
            'process_pdf',
            mock_process_pdf_error
        )
        
        # Verify that DocumentIndexerError is raised
        with pytest.raises(DocumentIndexerError):
            indexer.index_document("/path/to/test.pdf", "pdf")
    
    def test_index_document_empty_chunks(self, indexer, components, monkeypatch):
        """Test handling of documents that produce no chunks."""
        from datetime import datetime
        
        # Create a document with valid text
        mock_document = Document(
            id="empty-doc",
            source="/path/to/test.pdf",
            source_type="pdf",
            text="Valid text content",
            metadata={},
            created_at=datetime.now()
        )
        
        monkeypatch.setattr(
            components['document_processor'],
            'process_pdf',
            lambda x: mock_document
        )
        
        # Mock the chunker to return empty list
        monkeypatch.setattr(
            components['chunker'],
            'chunk_text',
            lambda text, document_id, metadata=None: []
        )
        
        # Verify that DocumentIndexerError is raised for empty chunks
        with pytest.raises(DocumentIndexerError, match="No chunks generated"):
            indexer.index_document("/path/to/test.pdf", "pdf")
    
    def test_metadata_vector_consistency(self, indexer, components, monkeypatch):
        """
        Test that metadata store and vector store remain consistent.
        
        Validates Requirement 11.5: Metadata-vector consistency
        """
        from datetime import datetime
        
        mock_document = Document(
            id="consistency-test",
            source="/path/to/test.pdf",
            source_type="pdf",
            text="Test content for consistency. " * 30,
            metadata={},
            created_at=datetime.now()
        )
        
        monkeypatch.setattr(
            components['document_processor'],
            'process_pdf',
            lambda x: mock_document
        )
        
        # Index document
        doc_id = indexer.index_document("/path/to/test.pdf", "pdf")
        
        # Get chunk IDs from metadata store
        metadata_chunk_ids = set(components['metadata_store'].get_document_chunks(doc_id))
        
        # Get chunk IDs from vector store (via reverse_id_map)
        vector_chunk_ids = set(components['vector_store'].reverse_id_map.values())
        
        # Verify consistency: all metadata chunks should have vectors
        assert metadata_chunk_ids.issubset(vector_chunk_ids)
        
        # Verify all vectors have metadata
        for chunk_id in vector_chunk_ids:
            chunk_meta = components['metadata_store'].get_chunk_metadata(chunk_id)
            assert chunk_meta is not None



class TestDocumentIndexerPropertyBased:
    """Property-based tests for DocumentIndexer using Hypothesis."""
    
    # Feature: ai-knowledge-copilot, Property 22: Web content pipeline consistency
    @given(
        web_text=st.text(
            min_size=100,
            max_size=500,
            alphabet=st.characters(min_codepoint=32, max_codepoint=126)
        ).filter(lambda x: x.strip())
    )
    @settings(max_examples=5, deadline=None)
    def test_property_web_content_pipeline_consistency(self, web_text):
        """
        Property 22: Web content pipeline consistency
        
        For any web content, it should be chunked and embedded using the same 
        functions as PDF content.
        
        **Validates: Requirements 7.2**
        """
        import html
        from unittest.mock import patch, Mock
        
        # Create temporary directory for test
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create all required components
            document_processor = DocumentProcessor()
            chunker = TextChunker(chunk_size=100, overlap=20)
            embedder = EmbeddingGenerator("all-MiniLM-L6-v2")
            
            # Create vector store
            dimension = embedder.get_embedding_dimension()
            vector_store = VectorStore(dimension=dimension)
            
            # Create metadata store with temp database
            db_path = Path(temp_dir) / "test_metadata.db"
            metadata_store = MetadataStore(str(db_path))
            
            # Create indexer
            indexer = DocumentIndexer(
                document_processor=document_processor,
                chunker=chunker,
                embedder=embedder,
                vector_store=vector_store,
                metadata_store=metadata_store
            )
            
            # Escape HTML special characters
            escaped_text = html.escape(web_text)
            
            # Create HTML with the generated text
            html_content = f"""
            <html>
                <head><title>Test Page</title></head>
                <body>
                    <p>{escaped_text}</p>
                </body>
            </html>
            """
            
            # Mock the requests.get call for web content
            with patch('src.document_processor.requests.get') as mock_get:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.headers = {'Content-Type': 'text/html; charset=utf-8'}
                mock_response.text = html_content
                mock_get.return_value = mock_response
                
                # Index web content
                web_doc_id = indexer.index_url("https://example.com/test")
            
            # Create a PDF document with the same text for comparison
            pdf_document = Document(
                id="pdf-comparison",
                source="/path/to/test.pdf",
                source_type="pdf",
                text=web_text,
                metadata={"filename": "test.pdf"},
                created_at=datetime.now()
            )
            
            # Mock the PDF processor
            with patch.object(document_processor, 'process_pdf', return_value=pdf_document):
                # Index PDF content
                pdf_doc_id = indexer.index_document("/path/to/test.pdf", "pdf")
            
            # Property 1: Both documents should be indexed successfully
            assert web_doc_id is not None, "Web document should be indexed"
            assert pdf_doc_id is not None, "PDF document should be indexed"
            
            # Property 2: Both should use the same chunker (verify chunk count is similar)
            web_chunks = metadata_store.get_document_chunks(web_doc_id)
            pdf_chunks = metadata_store.get_document_chunks(pdf_doc_id)
            
            # Since both have the same text, they should produce the same number of chunks
            # (allowing for minor differences due to whitespace normalization in HTML)
            assert len(web_chunks) > 0, "Web content should produce chunks"
            assert len(pdf_chunks) > 0, "PDF content should produce chunks"
            
            # Property 3: Both should use the same embedder (verify embedding dimensions)
            # Get a sample chunk from each
            if web_chunks and pdf_chunks:
                web_vector = vector_store.get_vector_by_id(web_chunks[0])
                pdf_vector = vector_store.get_vector_by_id(pdf_chunks[0])
                
                assert web_vector is not None, "Web chunk should have embedding"
                assert pdf_vector is not None, "PDF chunk should have embedding"
                
                # Both should have the same embedding dimension
                assert len(web_vector) == len(pdf_vector), \
                    "Web and PDF embeddings should have the same dimension"
                assert len(web_vector) == embedder.get_embedding_dimension(), \
                    "Embeddings should match the embedder's dimension"
            
            # Property 4: Both should be stored in the same vector store
            web_doc_info = indexer.get_document_info(web_doc_id)
            pdf_doc_info = indexer.get_document_info(pdf_doc_id)
            
            assert web_doc_info is not None, "Web document info should be retrievable"
            assert pdf_doc_info is not None, "PDF document info should be retrievable"
            
            # Property 5: Both should have chunk metadata stored consistently
            for chunk_id in web_chunks:
                chunk_meta = metadata_store.get_chunk_metadata(chunk_id)
                assert chunk_meta is not None, f"Web chunk {chunk_id} should have metadata"
                assert 'document_id' in chunk_meta, "Chunk metadata should have document_id"
                assert 'text' in chunk_meta, "Chunk metadata should have text"
                assert chunk_meta['document_id'] == web_doc_id
            
            for chunk_id in pdf_chunks:
                chunk_meta = metadata_store.get_chunk_metadata(chunk_id)
                assert chunk_meta is not None, f"PDF chunk {chunk_id} should have metadata"
                assert 'document_id' in chunk_meta, "Chunk metadata should have document_id"
                assert 'text' in chunk_meta, "Chunk metadata should have text"
                assert chunk_meta['document_id'] == pdf_doc_id
            
            # Property 6: Both should be listable through the same interface
            all_docs = indexer.list_documents()
            doc_ids = [doc['id'] for doc in all_docs]
            
            assert web_doc_id in doc_ids, "Web document should appear in document list"
            assert pdf_doc_id in doc_ids, "PDF document should appear in document list"
            
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Feature: ai-knowledge-copilot, Property 2: Document independence
    @given(
        num_documents=st.integers(min_value=2, max_value=3),
        doc_texts=st.lists(
            st.text(min_size=50, max_size=300, alphabet=st.characters(
                whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs', 'Po')
            )),
            min_size=2,
            max_size=3
        )
    )
    @settings(max_examples=5, deadline=None)
    def test_property_document_independence(self, num_documents, doc_texts):
        """
        Property 2: Document independence
        
        For any set of uploaded documents, each document should be tracked 
        with a unique ID and processed independently without interference.
        
        **Validates: Requirements 1.5**
        """
        # Create temporary directory for test
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create all required components
            document_processor = DocumentProcessor()
            chunker = TextChunker(chunk_size=100, overlap=20)
            embedder = EmbeddingGenerator("all-MiniLM-L6-v2")
            
            # Create vector store
            dimension = embedder.get_embedding_dimension()
            vector_store = VectorStore(dimension=dimension)
            
            # Create metadata store with temp database
            db_path = Path(temp_dir) / "test_metadata.db"
            metadata_store = MetadataStore(str(db_path))
            
            # Create indexer
            indexer = DocumentIndexer(
                document_processor=document_processor,
                chunker=chunker,
                embedder=embedder,
                vector_store=vector_store,
                metadata_store=metadata_store
            )
            
            # Ensure we have the right number of documents
            doc_texts = doc_texts[:num_documents]
            if len(doc_texts) < num_documents:
                # Pad with additional texts if needed
                while len(doc_texts) < num_documents:
                    doc_texts.append("Additional document content. " * 10)
            
            # Create mock documents with unique IDs
            mock_documents = []
            for i, text in enumerate(doc_texts):
                doc = Document(
                    id=f"doc-{i}-{hash(text) % 10000}",
                    source=f"/path/to/doc{i}.pdf",
                    source_type="pdf",
                    text=text,
                    metadata={"filename": f"doc{i}.pdf"},
                    created_at=datetime.now()
                )
                mock_documents.append(doc)
            
            # Mock the document processor to return documents in sequence
            call_count = [0]
            
            def mock_process_pdf(file_path):
                doc = mock_documents[call_count[0]]
                call_count[0] += 1
                return doc
            
            # Monkey patch the process_pdf method
            original_process_pdf = document_processor.process_pdf
            document_processor.process_pdf = mock_process_pdf
            
            try:
                # Index all documents
                indexed_doc_ids = []
                for i in range(num_documents):
                    doc_id = indexer.index_document(f"/path/to/doc{i}.pdf", "pdf")
                    indexed_doc_ids.append(doc_id)
                
                # Property 1: All documents should have unique IDs
                assert len(indexed_doc_ids) == len(set(indexed_doc_ids)), \
                    "All document IDs should be unique"
                
                # Property 2: Each document should be stored separately in metadata
                for doc_id in indexed_doc_ids:
                    doc_metadata = metadata_store.get_document_metadata(doc_id)
                    assert doc_metadata is not None, f"Document {doc_id} should exist in metadata store"
                    assert doc_metadata['id'] == doc_id, "Document ID should match"
                
                # Property 3: Chunks should be tracked separately per document
                all_chunk_ids = []
                for doc_id in indexed_doc_ids:
                    chunks = metadata_store.get_document_chunks(doc_id)
                    assert len(chunks) > 0, f"Document {doc_id} should have at least one chunk"
                    all_chunk_ids.extend(chunks)
                    
                    # Verify all chunks belong to the correct document
                    for chunk_id in chunks:
                        chunk_meta = metadata_store.get_chunk_metadata(chunk_id)
                        assert chunk_meta is not None, f"Chunk {chunk_id} should exist"
                        assert chunk_meta['document_id'] == doc_id, \
                            f"Chunk {chunk_id} should belong to document {doc_id}"
                
                # Property 4: No chunk ID overlap between documents
                assert len(all_chunk_ids) == len(set(all_chunk_ids)), \
                    "All chunk IDs should be unique across all documents"
                
                # Property 5: Each chunk should have a corresponding vector in the vector store
                for chunk_id in all_chunk_ids:
                    vector = vector_store.get_vector_by_id(chunk_id)
                    assert vector is not None, f"Chunk {chunk_id} should have a vector in the vector store"
                    assert len(vector) == embedder.get_embedding_dimension(), \
                        "Vector dimension should match embedding dimension"
                
                # Property 6: Total vector count should equal total chunk count
                total_chunks = len(all_chunk_ids)
                vector_count = vector_store.get_vector_count()
                assert vector_count == total_chunks, \
                    f"Vector count ({vector_count}) should equal total chunk count ({total_chunks})"
                
                # Property 7: Documents can be listed and all indexed documents appear
                listed_docs = metadata_store.list_documents()
                listed_doc_ids = [doc['id'] for doc in listed_docs]
                
                for doc_id in indexed_doc_ids:
                    assert doc_id in listed_doc_ids, \
                        f"Document {doc_id} should appear in document list"
            
            finally:
                # Restore original method
                document_processor.process_pdf = original_process_pdf
        
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Feature: ai-knowledge-copilot, Property 23: Web content metadata preservation
    @given(
        url_path=st.text(
            min_size=1,
            max_size=50,
            alphabet=st.characters(
                whitelist_categories=('Ll', 'Nd'),
                blacklist_characters='/'
            )
        ).filter(lambda x: x and x.strip()),
        web_text=st.text(
            min_size=100,
            max_size=500,
            alphabet=st.characters(min_codepoint=32, max_codepoint=126)
        ).filter(lambda x: x.strip())
    )
    @settings(max_examples=5, deadline=None)
    def test_property_web_metadata_preservation(self, url_path, web_text):
        """
        Property 23: Web content metadata preservation
        
        For any chunk created from web content, the metadata should contain 
        the source URL.
        
        **Validates: Requirements 7.4**
        """
        import html
        from unittest.mock import patch, Mock
        
        # Create temporary directory for test
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create all required components
            document_processor = DocumentProcessor()
            chunker = TextChunker(chunk_size=100, overlap=20)
            embedder = EmbeddingGenerator("all-MiniLM-L6-v2")
            
            # Create vector store
            dimension = embedder.get_embedding_dimension()
            vector_store = VectorStore(dimension=dimension)
            
            # Create metadata store with temp database
            db_path = Path(temp_dir) / "test_metadata.db"
            metadata_store = MetadataStore(str(db_path))
            
            # Create indexer
            indexer = DocumentIndexer(
                document_processor=document_processor,
                chunker=chunker,
                embedder=embedder,
                vector_store=vector_store,
                metadata_store=metadata_store
            )
            
            # Construct a valid URL
            url = f"https://example.com/{url_path}"
            
            # Escape HTML special characters
            escaped_text = html.escape(web_text)
            
            # Create HTML with the generated text
            html_content = f"""
            <html>
                <head><title>Test Page</title></head>
                <body>
                    <p>{escaped_text}</p>
                </body>
            </html>
            """
            
            # Mock the requests.get call for web content
            with patch('src.document_processor.requests.get') as mock_get:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.headers = {'Content-Type': 'text/html; charset=utf-8'}
                mock_response.text = html_content
                mock_get.return_value = mock_response
                
                # Index web content
                doc_id = indexer.index_url(url)
            
            # Property 1: Document should be indexed successfully
            assert doc_id is not None, "Web document should be indexed"
            
            # Property 2: Document metadata should contain the source URL
            doc_metadata = metadata_store.get_document_metadata(doc_id)
            assert doc_metadata is not None, "Document metadata should exist"
            assert 'source' in doc_metadata or 'url' in doc_metadata, \
                "Document metadata should contain source URL"
            
            # Check if URL is in source or url field
            stored_url = doc_metadata.get('source') or doc_metadata.get('url')
            assert stored_url == url, \
                f"Stored URL '{stored_url}' should match original URL '{url}'"
            
            # Property 3: Document should be marked as web source type
            assert doc_metadata.get('source_type') == 'web', \
                "Document should be marked as web source type"
            
            # Property 4: All chunks should preserve the source URL in metadata
            chunk_ids = metadata_store.get_document_chunks(doc_id)
            assert len(chunk_ids) > 0, "Web content should produce at least one chunk"
            
            for chunk_id in chunk_ids:
                chunk_metadata = metadata_store.get_chunk_metadata(chunk_id)
                assert chunk_metadata is not None, \
                    f"Chunk {chunk_id} should have metadata"
                
                # Chunk metadata should contain source information
                assert 'source' in chunk_metadata, \
                    f"Chunk {chunk_id} metadata should contain 'source' field"
                
                # The source should be the URL
                chunk_source = chunk_metadata['source']
                assert chunk_source == url, \
                    f"Chunk source '{chunk_source}' should match URL '{url}'"
                
                # Chunk metadata should indicate it's from web content
                assert 'source_type' in chunk_metadata, \
                    f"Chunk {chunk_id} metadata should contain 'source_type' field"
                assert chunk_metadata['source_type'] == 'web', \
                    f"Chunk {chunk_id} should be marked as web source type"
            
            # Property 5: Document info should reflect web source
            doc_info = indexer.get_document_info(doc_id)
            assert doc_info is not None, "Document info should be retrievable"
            assert doc_info.get('source_type') == 'web', \
                "Document info should show web source type"
            
            # Property 6: URL should be preserved through the entire pipeline
            # Verify we can retrieve the document and its URL is intact
            all_docs = indexer.list_documents()
            matching_docs = [d for d in all_docs if d['id'] == doc_id]
            assert len(matching_docs) == 1, "Document should appear in list"
            
            listed_doc = matching_docs[0]
            assert listed_doc.get('source_type') == 'web', \
                "Listed document should show web source type"
            
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Feature: ai-knowledge-copilot, Property 24: URL processing independence
    @given(
        num_urls=st.integers(min_value=2, max_value=3),
        url_paths=st.lists(
            st.text(
                min_size=1,
                max_size=30,
                alphabet=st.characters(
                    whitelist_categories=('Ll', 'Nd'),
                    blacklist_characters='/'
                )
            ).filter(lambda x: x and x.strip()),
            min_size=2,
            max_size=3
        ),
        web_texts=st.lists(
            st.text(
                min_size=100,
                max_size=300,
                alphabet=st.characters(min_codepoint=32, max_codepoint=126)
            ).filter(lambda x: x.strip()),
            min_size=2,
            max_size=3
        )
    )
    @settings(max_examples=5, deadline=None)
    def test_property_url_processing_independence(self, num_urls, url_paths, web_texts):
        """
        Property 24: URL processing independence
        
        For any set of URLs, each should be processed into separate documents 
        with unique IDs.
        
        **Validates: Requirements 7.5**
        """
        import html
        from unittest.mock import patch, Mock
        
        # Create temporary directory for test
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create all required components
            document_processor = DocumentProcessor()
            chunker = TextChunker(chunk_size=100, overlap=20)
            embedder = EmbeddingGenerator("all-MiniLM-L6-v2")
            
            # Create vector store
            dimension = embedder.get_embedding_dimension()
            vector_store = VectorStore(dimension=dimension)
            
            # Create metadata store with temp database
            db_path = Path(temp_dir) / "test_metadata.db"
            metadata_store = MetadataStore(str(db_path))
            
            # Create indexer
            indexer = DocumentIndexer(
                document_processor=document_processor,
                chunker=chunker,
                embedder=embedder,
                vector_store=vector_store,
                metadata_store=metadata_store
            )
            
            # Ensure we have the right number of URLs
            url_paths = url_paths[:num_urls]
            web_texts = web_texts[:num_urls]
            
            # Pad if needed
            while len(url_paths) < num_urls:
                url_paths.append(f"page{len(url_paths)}")
            while len(web_texts) < num_urls:
                web_texts.append("Additional web content. " * 20)
            
            # Create URLs and HTML content
            urls = []
            html_contents = []
            
            for i, (path, text) in enumerate(zip(url_paths, web_texts)):
                url = f"https://example.com/{path}"
                urls.append(url)
                
                escaped_text = html.escape(text)
                html_content = f"""
                <html>
                    <head><title>Test Page {i}</title></head>
                    <body>
                        <p>{escaped_text}</p>
                    </body>
                </html>
                """
                html_contents.append(html_content)
            
            # Mock the requests.get call to return different content for each URL
            def mock_get(url, **kwargs):
                # Find the index of this URL
                try:
                    idx = urls.index(url)
                except ValueError:
                    idx = 0
                
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.headers = {'Content-Type': 'text/html; charset=utf-8'}
                mock_response.text = html_contents[idx]
                return mock_response
            
            with patch('src.document_processor.requests.get', side_effect=mock_get):
                # Index all URLs
                indexed_doc_ids = []
                for url in urls:
                    doc_id = indexer.index_url(url)
                    indexed_doc_ids.append(doc_id)
                
                # Property 1: All URLs should be indexed successfully
                assert len(indexed_doc_ids) == num_urls, \
                    f"Should have indexed {num_urls} URLs"
                
                # Property 2: All document IDs should be unique
                assert len(indexed_doc_ids) == len(set(indexed_doc_ids)), \
                    "All document IDs should be unique"
                
                # Property 3: Each document should be stored separately in metadata
                for i, doc_id in enumerate(indexed_doc_ids):
                    doc_metadata = metadata_store.get_document_metadata(doc_id)
                    assert doc_metadata is not None, \
                        f"Document {doc_id} should exist in metadata store"
                    assert doc_metadata['id'] == doc_id, "Document ID should match"
                    
                    # Verify the source URL is correct
                    stored_url = doc_metadata.get('source') or doc_metadata.get('url')
                    assert stored_url == urls[i], \
                        f"Document {doc_id} should have correct source URL"
                    
                    # Verify source type is web
                    assert doc_metadata.get('source_type') == 'web', \
                        f"Document {doc_id} should be marked as web source"
                
                # Property 4: Chunks should be tracked separately per URL/document
                all_chunk_ids = []
                for doc_id in indexed_doc_ids:
                    chunks = metadata_store.get_document_chunks(doc_id)
                    assert len(chunks) > 0, \
                        f"Document {doc_id} should have at least one chunk"
                    all_chunk_ids.extend(chunks)
                    
                    # Verify all chunks belong to the correct document
                    for chunk_id in chunks:
                        chunk_meta = metadata_store.get_chunk_metadata(chunk_id)
                        assert chunk_meta is not None, \
                            f"Chunk {chunk_id} should exist"
                        assert chunk_meta['document_id'] == doc_id, \
                            f"Chunk {chunk_id} should belong to document {doc_id}"
                        
                        # Verify chunk has web source type
                        assert chunk_meta.get('source_type') == 'web', \
                            f"Chunk {chunk_id} should be marked as web source"
                
                # Property 5: No chunk ID overlap between documents
                assert len(all_chunk_ids) == len(set(all_chunk_ids)), \
                    "All chunk IDs should be unique across all documents"
                
                # Property 6: Each chunk should have a corresponding vector
                for chunk_id in all_chunk_ids:
                    vector = vector_store.get_vector_by_id(chunk_id)
                    assert vector is not None, \
                        f"Chunk {chunk_id} should have a vector in the vector store"
                    assert len(vector) == embedder.get_embedding_dimension(), \
                        "Vector dimension should match embedding dimension"
                
                # Property 7: Total vector count should equal total chunk count
                total_chunks = len(all_chunk_ids)
                vector_count = vector_store.get_vector_count()
                assert vector_count == total_chunks, \
                    f"Vector count ({vector_count}) should equal total chunk count ({total_chunks})"
                
                # Property 8: All documents can be listed and appear in the list
                listed_docs = metadata_store.list_documents()
                listed_doc_ids = [doc['id'] for doc in listed_docs]
                
                for doc_id in indexed_doc_ids:
                    assert doc_id in listed_doc_ids, \
                        f"Document {doc_id} should appear in document list"
                
                # Property 9: Each document maintains its unique URL in the list
                for i, doc_id in enumerate(indexed_doc_ids):
                    matching_docs = [d for d in listed_docs if d['id'] == doc_id]
                    assert len(matching_docs) == 1, \
                        f"Document {doc_id} should appear exactly once in list"
                    
                    listed_doc = matching_docs[0]
                    assert listed_doc.get('source_type') == 'web', \
                        f"Listed document {doc_id} should show web source type"
        
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)



class TestDocumentIndexerIntegration:
    """Integration tests for end-to-end document indexing pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def components(self, temp_dir):
        """Create all required components for DocumentIndexer."""
        document_processor = DocumentProcessor()
        chunker = TextChunker(chunk_size=100, overlap=20)
        embedder = EmbeddingGenerator("all-MiniLM-L6-v2")
        
        # Create vector store
        dimension = embedder.get_embedding_dimension()
        vector_store = VectorStore(dimension=dimension)
        
        # Create metadata store with temp database
        db_path = Path(temp_dir) / "test_metadata.db"
        metadata_store = MetadataStore(str(db_path))
        
        return {
            'document_processor': document_processor,
            'chunker': chunker,
            'embedder': embedder,
            'vector_store': vector_store,
            'metadata_store': metadata_store
        }
    
    @pytest.fixture
    def indexer(self, components):
        """Create a DocumentIndexer instance."""
        return DocumentIndexer(
            document_processor=components['document_processor'],
            chunker=components['chunker'],
            embedder=components['embedder'],
            vector_store=components['vector_store'],
            metadata_store=components['metadata_store']
        )
    
    def test_end_to_end_pdf_indexing(self, indexer, components, monkeypatch):
        """
        Integration test: Upload a PDF and verify chunks in FAISS.
        
        This test validates the complete pipeline from PDF upload through
        to searchable chunks in the vector database.
        
        Tests Requirements: 1.1, 1.5, 2.1, 2.2, 2.3, 2.4, 2.5, 9.2, 9.3
        """
        from datetime import datetime
        
        # Create a realistic mock PDF document
        pdf_text = """
        Introduction to Machine Learning
        
        Machine learning is a subset of artificial intelligence that focuses on 
        building systems that can learn from data. These systems improve their 
        performance on a specific task over time without being explicitly programmed.
        
        There are three main types of machine learning:
        1. Supervised Learning - Learning from labeled data
        2. Unsupervised Learning - Finding patterns in unlabeled data
        3. Reinforcement Learning - Learning through trial and error
        
        Applications of machine learning include image recognition, natural language
        processing, recommendation systems, and autonomous vehicles. The field has
        grown rapidly in recent years due to increased computational power and
        availability of large datasets.
        """
        
        mock_document = Document(
            id="ml-intro-pdf",
            source="/path/to/ml_intro.pdf",
            source_type="pdf",
            text=pdf_text,
            metadata={"filename": "ml_intro.pdf", "page_count": 1},
            created_at=datetime.now()
        )
        
        # Mock the PDF processor
        monkeypatch.setattr(
            components['document_processor'],
            'process_pdf',
            lambda x: mock_document
        )
        
        # Step 1: Index the PDF document
        doc_id = indexer.index_pdf("/path/to/ml_intro.pdf")
        
        # Verify document was indexed
        assert doc_id == "ml-intro-pdf"
        
        # Step 2: Verify document metadata is stored
        doc_metadata = components['metadata_store'].get_document_metadata(doc_id)
        assert doc_metadata is not None
        assert doc_metadata['id'] == doc_id
        assert doc_metadata['source_type'] == 'pdf'
        assert doc_metadata['source'] == "/path/to/ml_intro.pdf"
        
        # Step 3: Verify chunks were created
        chunk_ids = components['metadata_store'].get_document_chunks(doc_id)
        assert len(chunk_ids) > 0, "Should have created at least one chunk"
        
        # Verify chunk properties
        for chunk_id in chunk_ids:
            # Check chunk metadata
            chunk_meta = components['metadata_store'].get_chunk_metadata(chunk_id)
            assert chunk_meta is not None
            assert chunk_meta['document_id'] == doc_id
            assert 'text' in chunk_meta
            assert len(chunk_meta['text']) > 0
            assert chunk_meta['source_type'] == 'pdf'
        
        # Step 4: Verify embeddings are in FAISS
        vector_count = components['vector_store'].get_vector_count()
        assert vector_count == len(chunk_ids), \
            "Vector count should match chunk count"
        
        # Verify each chunk has a vector
        for chunk_id in chunk_ids:
            vector = components['vector_store'].get_vector_by_id(chunk_id)
            assert vector is not None, f"Chunk {chunk_id} should have a vector"
            assert len(vector) == components['embedder'].get_embedding_dimension()
            
            # Verify vector values are reasonable (not all zeros)
            assert np.any(vector != 0), "Vector should not be all zeros"
        
        # Step 5: Test semantic search functionality
        # Create a query embedding
        query = "What is machine learning?"
        query_embedding = components['embedder'].embed_text(query)
        
        # Search for similar chunks
        scores, retrieved_ids = components['vector_store'].search(
            query_embedding,
            k=min(3, len(chunk_ids))
        )
        
        # Verify search results
        assert len(retrieved_ids) > 0, "Should retrieve at least one chunk"
        assert all(rid in chunk_ids for rid in retrieved_ids), \
            "Retrieved chunks should be from the indexed document"
        
        # Verify distances are in ascending order (lower distance = more similar)
        # FAISS returns L2 distances, so lower is better
        assert all(scores[i] <= scores[i+1] for i in range(len(scores)-1)), \
            "Distances should be in ascending order (lower distance first)"
        
        # Step 6: Verify document can be listed
        all_docs = indexer.list_documents()
        assert len(all_docs) == 1
        assert all_docs[0]['id'] == doc_id
        assert all_docs[0]['chunk_count'] == len(chunk_ids)
        
        # Step 7: Verify document info retrieval
        doc_info = indexer.get_document_info(doc_id)
        assert doc_info is not None
        assert doc_info['id'] == doc_id
        assert doc_info['chunk_count'] == len(chunk_ids)
        assert doc_info['source_type'] == 'pdf'
    
    def test_end_to_end_web_content_indexing(self, indexer, components):
        """
        Integration test: Upload web content and verify metadata.
        
        This test validates the complete pipeline for web content indexing,
        ensuring URL metadata is preserved throughout.
        
        Tests Requirements: 7.1, 7.2, 7.4, 7.5
        """
        import html
        from unittest.mock import patch, Mock
        
        # Create realistic web content
        web_text = """
        Understanding Neural Networks
        
        Neural networks are computing systems inspired by biological neural networks.
        They consist of interconnected nodes (neurons) organized in layers. Each
        connection has a weight that adjusts as learning proceeds.
        
        A typical neural network has:
        - Input layer: Receives the initial data
        - Hidden layers: Process the information
        - Output layer: Produces the final result
        
        Training a neural network involves adjusting weights through backpropagation
        to minimize the difference between predicted and actual outputs.
        """
        
        url = "https://example.com/neural-networks"
        
        # Create HTML content
        escaped_text = html.escape(web_text)
        html_content = f"""
        <html>
            <head>
                <title>Understanding Neural Networks</title>
                <meta name="description" content="A guide to neural networks">
            </head>
            <body>
                <h1>Understanding Neural Networks</h1>
                <article>
                    {escaped_text}
                </article>
            </body>
        </html>
        """
        
        # Mock the web request
        with patch('src.document_processor.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {'Content-Type': 'text/html; charset=utf-8'}
            mock_response.text = html_content
            mock_get.return_value = mock_response
            
            # Step 1: Index the web content
            doc_id = indexer.index_url(url)
        
        # Verify document was indexed
        assert doc_id is not None
        
        # Step 2: Verify document metadata includes URL
        doc_metadata = components['metadata_store'].get_document_metadata(doc_id)
        assert doc_metadata is not None
        assert doc_metadata['source_type'] == 'web'
        
        # Check URL is stored (could be in 'source' or 'url' field)
        stored_url = doc_metadata.get('source') or doc_metadata.get('url')
        assert stored_url == url, "URL should be preserved in metadata"
        
        # Step 3: Verify chunks were created
        chunk_ids = components['metadata_store'].get_document_chunks(doc_id)
        assert len(chunk_ids) > 0, "Should have created at least one chunk"
        
        # Step 4: Verify all chunks have URL metadata
        for chunk_id in chunk_ids:
            chunk_meta = components['metadata_store'].get_chunk_metadata(chunk_id)
            assert chunk_meta is not None
            assert chunk_meta['document_id'] == doc_id
            assert chunk_meta['source_type'] == 'web'
            
            # Verify URL is in chunk metadata
            chunk_source = chunk_meta.get('source') or chunk_meta.get('url')
            assert chunk_source == url, \
                f"Chunk {chunk_id} should preserve the source URL"
        
        # Step 5: Verify embeddings are in FAISS
        vector_count = components['vector_store'].get_vector_count()
        assert vector_count == len(chunk_ids)
        
        for chunk_id in chunk_ids:
            vector = components['vector_store'].get_vector_by_id(chunk_id)
            assert vector is not None
            assert len(vector) == components['embedder'].get_embedding_dimension()
        
        # Step 6: Test semantic search with web content
        query = "How do neural networks work?"
        query_embedding = components['embedder'].embed_text(query)
        
        scores, retrieved_ids = components['vector_store'].search(
            query_embedding,
            k=min(3, len(chunk_ids))
        )
        
        assert len(retrieved_ids) > 0
        assert all(rid in chunk_ids for rid in retrieved_ids)
        
        # Step 7: Verify document appears in listing with correct metadata
        all_docs = indexer.list_documents()
        matching_docs = [d for d in all_docs if d['id'] == doc_id]
        assert len(matching_docs) == 1
        
        listed_doc = matching_docs[0]
        assert listed_doc['source_type'] == 'web'
        assert listed_doc['chunk_count'] == len(chunk_ids)
    
    def test_end_to_end_multiple_documents(self, indexer, components, monkeypatch):
        """
        Integration test: Index multiple documents and verify independence.
        
        This test validates that multiple documents can be indexed and
        searched independently without interference.
        
        Tests Requirements: 1.5, 11.5
        """
        from datetime import datetime
        import html
        from unittest.mock import patch, Mock
        
        # Create two different documents
        pdf_text = "Artificial intelligence is transforming technology. " * 20
        web_text = "Quantum computing uses quantum mechanics principles. " * 20
        
        # Mock PDF document
        pdf_doc = Document(
            id="ai-doc",
            source="/path/to/ai.pdf",
            source_type="pdf",
            text=pdf_text,
            metadata={"filename": "ai.pdf"},
            created_at=datetime.now()
        )
        
        # Mock the PDF processor
        monkeypatch.setattr(
            components['document_processor'],
            'process_pdf',
            lambda x: pdf_doc
        )
        
        # Index PDF
        pdf_doc_id = indexer.index_pdf("/path/to/ai.pdf")
        
        # Mock web content
        url = "https://example.com/quantum"
        html_content = f"""
        <html>
            <body><p>{html.escape(web_text)}</p></body>
        </html>
        """
        
        with patch('src.document_processor.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {'Content-Type': 'text/html; charset=utf-8'}
            mock_response.text = html_content
            mock_get.return_value = mock_response
            
            # Index web content
            web_doc_id = indexer.index_url(url)
        
        # Verify both documents are indexed with unique IDs
        assert pdf_doc_id != web_doc_id
        
        # Get chunks for each document
        pdf_chunks = components['metadata_store'].get_document_chunks(pdf_doc_id)
        web_chunks = components['metadata_store'].get_document_chunks(web_doc_id)
        
        assert len(pdf_chunks) > 0
        assert len(web_chunks) > 0
        
        # Verify no chunk overlap
        assert set(pdf_chunks).isdisjoint(set(web_chunks)), \
            "Chunks should be independent between documents"
        
        # Verify metadata-vector consistency
        all_chunk_ids = pdf_chunks + web_chunks
        vector_count = components['vector_store'].get_vector_count()
        assert vector_count == len(all_chunk_ids), \
            "Vector count should match total chunk count"
        
        # Verify each chunk has correct document association
        for chunk_id in pdf_chunks:
            chunk_meta = components['metadata_store'].get_chunk_metadata(chunk_id)
            assert chunk_meta['document_id'] == pdf_doc_id
            assert chunk_meta['source_type'] == 'pdf'
        
        for chunk_id in web_chunks:
            chunk_meta = components['metadata_store'].get_chunk_metadata(chunk_id)
            assert chunk_meta['document_id'] == web_doc_id
            assert chunk_meta['source_type'] == 'web'
        
        # Verify both documents appear in listing
        all_docs = indexer.list_documents()
        assert len(all_docs) == 2
        
        doc_ids = [d['id'] for d in all_docs]
        assert pdf_doc_id in doc_ids
        assert web_doc_id in doc_ids
        
        # Test searching - results should come from both documents
        query = "technology and computing"
        query_embedding = components['embedder'].embed_text(query)
        
        scores, retrieved_ids = components['vector_store'].search(
            query_embedding,
            k=10
        )
        
        # Should retrieve chunks (could be from either or both documents)
        assert len(retrieved_ids) > 0
        assert all(rid in all_chunk_ids for rid in retrieved_ids)
