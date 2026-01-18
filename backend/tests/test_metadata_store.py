"""
Unit tests for MetadataStore class.

Tests CRUD operations, cascade deletion, and document listing functionality.
"""
import pytest
import tempfile
import os
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metadata_store import MetadataStore


class TestMetadataStore:
    """Tests for MetadataStore class."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        # Create a temporary file
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        
        # Yield the path for use in tests
        yield path
        
        # Cleanup after test
        if os.path.exists(path):
            os.remove(path)
    
    @pytest.fixture
    def store(self, temp_db):
        """Create a MetadataStore instance with temporary database."""
        return MetadataStore(db_path=temp_db)
    
    def test_init_creates_database(self, temp_db):
        """Test that initializing MetadataStore creates the database file."""
        store = MetadataStore(db_path=temp_db)
        assert os.path.exists(temp_db)
    
    def test_add_document(self, store):
        """Test adding a document to the metadata store."""
        metadata = {
            'source': '/path/to/document.pdf',
            'source_type': 'pdf',
            'created_at': datetime.now().isoformat(),
            'title': 'Test Document'
        }
        
        store.add_document('doc1', metadata)
        
        # Verify document was added
        doc = store.get_document_metadata('doc1')
        assert doc is not None
        assert doc['id'] == 'doc1'
        assert doc['source'] == '/path/to/document.pdf'
        assert doc['source_type'] == 'pdf'
        assert doc['title'] == 'Test Document'
    
    def test_add_document_with_datetime_object(self, store):
        """Test adding a document with datetime object (not string)."""
        created_at = datetime.now()
        metadata = {
            'source': '/path/to/document.pdf',
            'source_type': 'pdf',
            'created_at': created_at,
            'title': 'Test Document'
        }
        
        store.add_document('doc1', metadata)
        
        # Verify document was added and datetime was converted
        doc = store.get_document_metadata('doc1')
        assert doc is not None
        assert doc['id'] == 'doc1'
        assert 'created_at' in doc
    
    def test_add_chunk(self, store):
        """Test adding a chunk to the metadata store."""
        # First add a document
        doc_metadata = {
            'source': '/path/to/document.pdf',
            'source_type': 'pdf',
            'created_at': datetime.now().isoformat()
        }
        store.add_document('doc1', doc_metadata)
        
        # Add a chunk
        chunk_metadata = {
            'document_id': 'doc1',
            'text': 'This is chunk text',
            'chunk_index': 0,
            'page_number': 1
        }
        store.add_chunk('chunk1', chunk_metadata)
        
        # Verify chunk was added
        chunk = store.get_chunk_metadata('chunk1')
        assert chunk is not None
        assert chunk['id'] == 'chunk1'
        assert chunk['document_id'] == 'doc1'
        assert chunk['text'] == 'This is chunk text'
        assert chunk['chunk_index'] == 0
        assert chunk['page_number'] == 1
    
    def test_get_chunk_metadata_not_found(self, store):
        """Test retrieving metadata for non-existent chunk."""
        chunk = store.get_chunk_metadata('nonexistent')
        assert chunk is None
    
    def test_get_document_metadata_not_found(self, store):
        """Test retrieving metadata for non-existent document."""
        doc = store.get_document_metadata('nonexistent')
        assert doc is None
    
    def test_get_document_chunks(self, store):
        """Test retrieving all chunks for a document."""
        # Add document
        doc_metadata = {
            'source': '/path/to/document.pdf',
            'source_type': 'pdf',
            'created_at': datetime.now().isoformat()
        }
        store.add_document('doc1', doc_metadata)
        
        # Add multiple chunks
        for i in range(3):
            chunk_metadata = {
                'document_id': 'doc1',
                'text': f'Chunk {i} text',
                'chunk_index': i
            }
            store.add_chunk(f'chunk{i}', chunk_metadata)
        
        # Retrieve chunks
        chunk_ids = store.get_document_chunks('doc1')
        assert len(chunk_ids) == 3
        assert 'chunk0' in chunk_ids
        assert 'chunk1' in chunk_ids
        assert 'chunk2' in chunk_ids
    
    def test_get_document_chunks_empty(self, store):
        """Test retrieving chunks for document with no chunks."""
        # Add document without chunks
        doc_metadata = {
            'source': '/path/to/document.pdf',
            'source_type': 'pdf',
            'created_at': datetime.now().isoformat()
        }
        store.add_document('doc1', doc_metadata)
        
        # Retrieve chunks
        chunk_ids = store.get_document_chunks('doc1')
        assert len(chunk_ids) == 0
    
    def test_delete_document_cascade(self, store):
        """Test that deleting a document also deletes its chunks (cascade)."""
        # Add document
        doc_metadata = {
            'source': '/path/to/document.pdf',
            'source_type': 'pdf',
            'created_at': datetime.now().isoformat()
        }
        store.add_document('doc1', doc_metadata)
        
        # Add chunks
        for i in range(3):
            chunk_metadata = {
                'document_id': 'doc1',
                'text': f'Chunk {i} text',
                'chunk_index': i
            }
            store.add_chunk(f'chunk{i}', chunk_metadata)
        
        # Verify chunks exist
        assert len(store.get_document_chunks('doc1')) == 3
        
        # Delete document
        store.delete_document('doc1')
        
        # Verify document is deleted
        assert store.get_document_metadata('doc1') is None
        
        # Verify chunks are also deleted (cascade)
        assert len(store.get_document_chunks('doc1')) == 0
        assert store.get_chunk_metadata('chunk0') is None
        assert store.get_chunk_metadata('chunk1') is None
        assert store.get_chunk_metadata('chunk2') is None
    
    def test_list_documents_empty(self, store):
        """Test listing documents when none exist."""
        documents = store.list_documents()
        assert len(documents) == 0
    
    def test_list_documents(self, store):
        """Test listing all documents."""
        # Add multiple documents
        for i in range(3):
            metadata = {
                'source': f'/path/to/document{i}.pdf',
                'source_type': 'pdf',
                'created_at': datetime.now().isoformat(),
                'title': f'Document {i}'
            }
            store.add_document(f'doc{i}', metadata)
        
        # List documents
        documents = store.list_documents()
        assert len(documents) == 3
        
        # Verify all documents are present
        doc_ids = [doc['id'] for doc in documents]
        assert 'doc0' in doc_ids
        assert 'doc1' in doc_ids
        assert 'doc2' in doc_ids
        
        # Verify metadata is included
        for doc in documents:
            assert 'source' in doc
            assert 'source_type' in doc
            assert 'created_at' in doc
            assert 'title' in doc
    
    def test_list_documents_ordered_by_created_at(self, store):
        """Test that documents are listed in reverse chronological order."""
        import time
        
        # Add documents with slight time delays
        for i in range(3):
            metadata = {
                'source': f'/path/to/document{i}.pdf',
                'source_type': 'pdf',
                'created_at': datetime.now().isoformat(),
                'title': f'Document {i}'
            }
            store.add_document(f'doc{i}', metadata)
            time.sleep(0.01)  # Small delay to ensure different timestamps
        
        # List documents
        documents = store.list_documents()
        
        # Verify order (most recent first)
        assert documents[0]['id'] == 'doc2'
        assert documents[1]['id'] == 'doc1'
        assert documents[2]['id'] == 'doc0'
    
    def test_clear_all(self, store):
        """Test clearing all documents and chunks."""
        # Add documents and chunks
        for i in range(2):
            doc_metadata = {
                'source': f'/path/to/document{i}.pdf',
                'source_type': 'pdf',
                'created_at': datetime.now().isoformat()
            }
            store.add_document(f'doc{i}', doc_metadata)
            
            chunk_metadata = {
                'document_id': f'doc{i}',
                'text': f'Chunk text',
                'chunk_index': 0
            }
            store.add_chunk(f'chunk{i}', chunk_metadata)
        
        # Verify data exists
        assert len(store.list_documents()) == 2
        
        # Clear all
        store.clear_all()
        
        # Verify everything is deleted
        assert len(store.list_documents()) == 0
        assert store.get_chunk_metadata('chunk0') is None
        assert store.get_chunk_metadata('chunk1') is None
    
    def test_update_document(self, store):
        """Test updating a document (INSERT OR REPLACE)."""
        # Add initial document
        metadata = {
            'source': '/path/to/document.pdf',
            'source_type': 'pdf',
            'created_at': datetime.now().isoformat(),
            'title': 'Original Title'
        }
        store.add_document('doc1', metadata)
        
        # Update document
        updated_metadata = {
            'source': '/path/to/document.pdf',
            'source_type': 'pdf',
            'created_at': datetime.now().isoformat(),
            'title': 'Updated Title'
        }
        store.add_document('doc1', updated_metadata)
        
        # Verify document was updated
        doc = store.get_document_metadata('doc1')
        assert doc['title'] == 'Updated Title'
        
        # Verify only one document exists
        documents = store.list_documents()
        assert len(documents) == 1
    
    def test_update_chunk(self, store):
        """Test updating a chunk (INSERT OR REPLACE)."""
        # Add document
        doc_metadata = {
            'source': '/path/to/document.pdf',
            'source_type': 'pdf',
            'created_at': datetime.now().isoformat()
        }
        store.add_document('doc1', doc_metadata)
        
        # Add initial chunk
        chunk_metadata = {
            'document_id': 'doc1',
            'text': 'Original text',
            'chunk_index': 0
        }
        store.add_chunk('chunk1', chunk_metadata)
        
        # Update chunk
        updated_metadata = {
            'document_id': 'doc1',
            'text': 'Updated text',
            'chunk_index': 0
        }
        store.add_chunk('chunk1', updated_metadata)
        
        # Verify chunk was updated
        chunk = store.get_chunk_metadata('chunk1')
        assert chunk['text'] == 'Updated text'
        
        # Verify only one chunk exists
        chunks = store.get_document_chunks('doc1')
        assert len(chunks) == 1


class TestDocumentListingCompleteness:
    """Property-based tests for document listing completeness using Hypothesis."""
    
    # Feature: ai-knowledge-copilot, Property 31: Document listing completeness
    def test_document_listing_completeness(self):
        """
        Property 31: Document listing completeness
        
        For any set of indexed documents, the list operation should return all document IDs.
        
        **Validates: Requirements 11.2**
        """
        from hypothesis import given, strategies as st, settings
        
        @given(
            num_documents=st.integers(min_value=0, max_value=20),
            seed=st.integers(min_value=0, max_value=10000)
        )
        @settings(deadline=500, max_examples=50)
        def run_test(num_documents, seed):
            # Create fresh temporary database for this iteration
            fd, temp_db_path = tempfile.mkstemp(suffix='.db')
            os.close(fd)
            
            try:
                # Create fresh metadata store
                metadata_store = MetadataStore(db_path=temp_db_path)
                
                # Track all document IDs we add
                expected_doc_ids = set()
                
                # Add documents with various metadata
                for doc_idx in range(num_documents):
                    doc_id = f"doc_{doc_idx}_{seed}"
                    expected_doc_ids.add(doc_id)
                    
                    # Create document with varying metadata
                    doc_metadata = {
                        'source': f'/path/to/document{doc_idx}.pdf',
                        'source_type': 'pdf' if doc_idx % 2 == 0 else 'web',
                        'created_at': datetime.now().isoformat(),
                        'title': f'Document {doc_idx}',
                        'author': f'Author {doc_idx % 3}',
                        'tags': ['tag1', 'tag2'] if doc_idx % 3 == 0 else []
                    }
                    
                    metadata_store.add_document(doc_id, doc_metadata)
                
                # List all documents
                listed_documents = metadata_store.list_documents()
                
                # Extract document IDs from the list
                actual_doc_ids = set(doc['id'] for doc in listed_documents)
                
                # VERIFY: All added documents are in the list
                assert actual_doc_ids == expected_doc_ids, \
                    f"Document listing completeness violated:\n" \
                    f"Expected document IDs: {sorted(expected_doc_ids)}\n" \
                    f"Actual document IDs: {sorted(actual_doc_ids)}\n" \
                    f"Missing documents: {expected_doc_ids - actual_doc_ids}\n" \
                    f"Extra documents: {actual_doc_ids - expected_doc_ids}"
                
                # VERIFY: Count matches
                assert len(listed_documents) == num_documents, \
                    f"Expected {num_documents} documents in list, got {len(listed_documents)}"
                
                # VERIFY: Each document has required fields
                for doc in listed_documents:
                    assert 'id' in doc, f"Document missing 'id' field: {doc}"
                    assert 'source' in doc, f"Document missing 'source' field: {doc}"
                    assert 'source_type' in doc, f"Document missing 'source_type' field: {doc}"
                    assert 'created_at' in doc, f"Document missing 'created_at' field: {doc}"
                
                # VERIFY: No duplicates in the list
                doc_ids_list = [doc['id'] for doc in listed_documents]
                assert len(doc_ids_list) == len(set(doc_ids_list)), \
                    f"Duplicate document IDs found in list: {doc_ids_list}"
            
            finally:
                # Clean up temporary database
                if os.path.exists(temp_db_path):
                    os.remove(temp_db_path)
        
        # Run the property test
        run_test()


class TestMetadataVectorConsistency:
    """Property-based tests for metadata-vector consistency using Hypothesis."""
    
    # Feature: ai-knowledge-copilot, Property 33: Metadata-vector consistency
    def test_metadata_vector_consistency_add_operations(self):
        """
        Property 33: Metadata-vector consistency
        
        For any operation (add, delete, update), the set of chunk IDs in metadata
        storage should match the set of vector IDs in FAISS.
        
        This test validates consistency after ADD operations.
        
        **Validates: Requirements 11.5**
        """
        from hypothesis import given, strategies as st, settings
        import numpy as np
        from src.vector_store import VectorStore
        import shutil
        
        @given(
            num_documents=st.integers(min_value=1, max_value=5),
            chunks_per_doc=st.integers(min_value=1, max_value=10),
            dimension=st.integers(min_value=2, max_value=128),
            seed=st.integers(min_value=0, max_value=10000)
        )
        @settings(deadline=500)  # Increase deadline for slower operations
        def run_test(num_documents, chunks_per_doc, dimension, seed):
            # Set random seed for reproducibility
            np.random.seed(seed)
            
            # Create fresh temporary database for this iteration
            fd, temp_db_path = tempfile.mkstemp(suffix='.db')
            os.close(fd)
            
            try:
                # Create fresh stores for each test iteration
                metadata_store = MetadataStore(db_path=temp_db_path)
                vector_store = VectorStore(dimension=dimension)
                
                # Track all chunk IDs we add
                all_chunk_ids = []
                
                # Add documents and chunks
                for doc_idx in range(num_documents):
                    doc_id = f"doc_{doc_idx}"
                    
                    # Add document to metadata store
                    doc_metadata = {
                        'source': f'/path/to/document{doc_idx}.pdf',
                        'source_type': 'pdf',
                        'created_at': datetime.now().isoformat(),
                        'title': f'Document {doc_idx}'
                    }
                    metadata_store.add_document(doc_id, doc_metadata)
                    
                    # Add chunks for this document
                    chunk_ids = []
                    chunk_vectors = []
                    
                    for chunk_idx in range(chunks_per_doc):
                        chunk_id = f"chunk_{doc_idx}_{chunk_idx}"
                        chunk_ids.append(chunk_id)
                        all_chunk_ids.append(chunk_id)
                        
                        # Add chunk to metadata store
                        chunk_metadata = {
                            'document_id': doc_id,
                            'text': f'Chunk {chunk_idx} text from document {doc_idx}',
                            'chunk_index': chunk_idx,
                            'page_number': chunk_idx // 3 + 1
                        }
                        metadata_store.add_chunk(chunk_id, chunk_metadata)
                        
                        # Generate random embedding vector
                        vector = np.random.rand(dimension).astype('float32')
                        chunk_vectors.append(vector)
                    
                    # Add all chunk vectors to vector store
                    vectors_array = np.array(chunk_vectors, dtype='float32')
                    vector_store.add_vectors(vectors_array, chunk_ids)
                
                # VERIFY CONSISTENCY: Get all chunk IDs from metadata store
                metadata_chunk_ids = set()
                for doc_idx in range(num_documents):
                    doc_id = f"doc_{doc_idx}"
                    doc_chunks = metadata_store.get_document_chunks(doc_id)
                    metadata_chunk_ids.update(doc_chunks)
                
                # Get all vector IDs from vector store
                vector_ids = set(vector_store.id_map.keys())
                
                # Assert that the sets are identical
                assert metadata_chunk_ids == vector_ids, \
                    f"Metadata-vector consistency violated after ADD operations:\n" \
                    f"Metadata chunk IDs: {sorted(metadata_chunk_ids)}\n" \
                    f"Vector IDs: {sorted(vector_ids)}\n" \
                    f"Missing in vectors: {metadata_chunk_ids - vector_ids}\n" \
                    f"Missing in metadata: {vector_ids - metadata_chunk_ids}"
                
                # Verify count matches
                assert len(metadata_chunk_ids) == len(all_chunk_ids), \
                    f"Expected {len(all_chunk_ids)} chunks in metadata, got {len(metadata_chunk_ids)}"
                
                assert vector_store.get_vector_count() == len(all_chunk_ids), \
                    f"Expected {len(all_chunk_ids)} vectors in store, got {vector_store.get_vector_count()}"
            
            finally:
                # Clean up temporary database
                if os.path.exists(temp_db_path):
                    os.remove(temp_db_path)
        
        # Run the property test
        run_test()
    
    def test_metadata_vector_consistency_delete_operations(self):
        """
        Property 33: Metadata-vector consistency (DELETE operations)
        
        For any operation (add, delete, update), the set of chunk IDs in metadata
        storage should match the set of vector IDs in FAISS.
        
        This test validates consistency after DELETE operations.
        
        **Validates: Requirements 11.5**
        """
        from hypothesis import given, strategies as st, settings
        import numpy as np
        from src.vector_store import VectorStore
        import shutil
        
        @given(
            num_documents=st.integers(min_value=2, max_value=5),
            chunks_per_doc=st.integers(min_value=2, max_value=8),
            docs_to_delete=st.integers(min_value=1, max_value=2),
            dimension=st.integers(min_value=2, max_value=128),
            seed=st.integers(min_value=0, max_value=10000)
        )
        @settings(deadline=500)
        def run_test(num_documents, chunks_per_doc, docs_to_delete, dimension, seed):
            # Ensure we don't try to delete more documents than we have
            docs_to_delete = min(docs_to_delete, num_documents - 1)
            
            # Set random seed for reproducibility
            np.random.seed(seed)
            
            # Create fresh temporary database for this iteration
            fd, temp_db_path = tempfile.mkstemp(suffix='.db')
            os.close(fd)
            
            try:
                # Create fresh stores
                metadata_store = MetadataStore(db_path=temp_db_path)
                vector_store = VectorStore(dimension=dimension)
                
                # Add documents and chunks
                for doc_idx in range(num_documents):
                    doc_id = f"doc_{doc_idx}"
                    
                    # Add document to metadata store
                    doc_metadata = {
                        'source': f'/path/to/document{doc_idx}.pdf',
                        'source_type': 'pdf',
                        'created_at': datetime.now().isoformat()
                    }
                    metadata_store.add_document(doc_id, doc_metadata)
                    
                    # Add chunks
                    chunk_ids = []
                    chunk_vectors = []
                    
                    for chunk_idx in range(chunks_per_doc):
                        chunk_id = f"chunk_{doc_idx}_{chunk_idx}"
                        chunk_ids.append(chunk_id)
                        
                        chunk_metadata = {
                            'document_id': doc_id,
                            'text': f'Chunk {chunk_idx} text',
                            'chunk_index': chunk_idx
                        }
                        metadata_store.add_chunk(chunk_id, chunk_metadata)
                        
                        vector = np.random.rand(dimension).astype('float32')
                        chunk_vectors.append(vector)
                    
                    vectors_array = np.array(chunk_vectors, dtype='float32')
                    vector_store.add_vectors(vectors_array, chunk_ids)
                
                # Delete some documents
                for doc_idx in range(docs_to_delete):
                    doc_id = f"doc_{doc_idx}"
                    
                    # Get chunk IDs before deletion
                    chunk_ids_to_delete = metadata_store.get_document_chunks(doc_id)
                    
                    # Delete from metadata store (cascade deletes chunks)
                    metadata_store.delete_document(doc_id)
                    
                    # Delete from vector store
                    if chunk_ids_to_delete:
                        vector_store.delete_vectors(chunk_ids_to_delete)
                
                # VERIFY CONSISTENCY after deletions
                metadata_chunk_ids = set()
                for doc_idx in range(num_documents):
                    doc_id = f"doc_{doc_idx}"
                    doc_chunks = metadata_store.get_document_chunks(doc_id)
                    metadata_chunk_ids.update(doc_chunks)
                
                vector_ids = set(vector_store.id_map.keys())
                
                # Assert consistency
                assert metadata_chunk_ids == vector_ids, \
                    f"Metadata-vector consistency violated after DELETE operations:\n" \
                    f"Metadata chunk IDs: {sorted(metadata_chunk_ids)}\n" \
                    f"Vector IDs: {sorted(vector_ids)}\n" \
                    f"Missing in vectors: {metadata_chunk_ids - vector_ids}\n" \
                    f"Missing in metadata: {vector_ids - metadata_chunk_ids}"
                
                # Verify expected count
                expected_chunks = (num_documents - docs_to_delete) * chunks_per_doc
                assert len(metadata_chunk_ids) == expected_chunks, \
                    f"Expected {expected_chunks} chunks after deletion, got {len(metadata_chunk_ids)}"
                
                assert vector_store.get_vector_count() == expected_chunks, \
                    f"Expected {expected_chunks} vectors after deletion, got {vector_store.get_vector_count()}"
            
            finally:
                # Clean up temporary database
                if os.path.exists(temp_db_path):
                    os.remove(temp_db_path)
        
        # Run the property test
        run_test()
    
    def test_metadata_vector_consistency_mixed_operations(self):
        """
        Property 33: Metadata-vector consistency (MIXED operations)
        
        For any operation (add, delete, update), the set of chunk IDs in metadata
        storage should match the set of vector IDs in FAISS.
        
        This test validates consistency after a sequence of ADD and DELETE operations.
        
        **Validates: Requirements 11.5**
        """
        from hypothesis import given, strategies as st, settings
        import numpy as np
        from src.vector_store import VectorStore
        import shutil
        
        @given(
            initial_docs=st.integers(min_value=2, max_value=4),
            chunks_per_doc=st.integers(min_value=2, max_value=6),
            docs_to_delete=st.integers(min_value=1, max_value=2),
            new_docs=st.integers(min_value=1, max_value=3),
            dimension=st.integers(min_value=2, max_value=128),
            seed=st.integers(min_value=0, max_value=10000)
        )
        @settings(deadline=500)
        def run_test(initial_docs, chunks_per_doc, docs_to_delete, new_docs, dimension, seed):
            # Ensure we don't delete more than we have
            docs_to_delete = min(docs_to_delete, initial_docs - 1)
            
            np.random.seed(seed)
            
            # Create fresh temporary database for this iteration
            fd, temp_db_path = tempfile.mkstemp(suffix='.db')
            os.close(fd)
            
            try:
                metadata_store = MetadataStore(db_path=temp_db_path)
                vector_store = VectorStore(dimension=dimension)
                
                # Phase 1: Add initial documents
                for doc_idx in range(initial_docs):
                    doc_id = f"doc_{doc_idx}"
                    doc_metadata = {
                        'source': f'/path/to/document{doc_idx}.pdf',
                        'source_type': 'pdf',
                        'created_at': datetime.now().isoformat()
                    }
                    metadata_store.add_document(doc_id, doc_metadata)
                    
                    chunk_ids = []
                    chunk_vectors = []
                    for chunk_idx in range(chunks_per_doc):
                        chunk_id = f"chunk_{doc_idx}_{chunk_idx}"
                        chunk_ids.append(chunk_id)
                        
                        chunk_metadata = {
                            'document_id': doc_id,
                            'text': f'Chunk {chunk_idx} text',
                            'chunk_index': chunk_idx
                        }
                        metadata_store.add_chunk(chunk_id, chunk_metadata)
                        chunk_vectors.append(np.random.rand(dimension).astype('float32'))
                    
                    vector_store.add_vectors(np.array(chunk_vectors), chunk_ids)
                
                # Phase 2: Delete some documents
                for doc_idx in range(docs_to_delete):
                    doc_id = f"doc_{doc_idx}"
                    chunk_ids_to_delete = metadata_store.get_document_chunks(doc_id)
                    metadata_store.delete_document(doc_id)
                    if chunk_ids_to_delete:
                        vector_store.delete_vectors(chunk_ids_to_delete)
                
                # Phase 3: Add new documents
                for doc_idx in range(initial_docs, initial_docs + new_docs):
                    doc_id = f"doc_{doc_idx}"
                    doc_metadata = {
                        'source': f'/path/to/document{doc_idx}.pdf',
                        'source_type': 'pdf',
                        'created_at': datetime.now().isoformat()
                    }
                    metadata_store.add_document(doc_id, doc_metadata)
                    
                    chunk_ids = []
                    chunk_vectors = []
                    for chunk_idx in range(chunks_per_doc):
                        chunk_id = f"chunk_{doc_idx}_{chunk_idx}"
                        chunk_ids.append(chunk_id)
                        
                        chunk_metadata = {
                            'document_id': doc_id,
                            'text': f'Chunk {chunk_idx} text',
                            'chunk_index': chunk_idx
                        }
                        metadata_store.add_chunk(chunk_id, chunk_metadata)
                        chunk_vectors.append(np.random.rand(dimension).astype('float32'))
                    
                    vector_store.add_vectors(np.array(chunk_vectors), chunk_ids)
                
                # VERIFY CONSISTENCY after mixed operations
                metadata_chunk_ids = set()
                all_docs = metadata_store.list_documents()
                for doc in all_docs:
                    doc_chunks = metadata_store.get_document_chunks(doc['id'])
                    metadata_chunk_ids.update(doc_chunks)
                
                vector_ids = set(vector_store.id_map.keys())
                
                # Assert consistency
                assert metadata_chunk_ids == vector_ids, \
                    f"Metadata-vector consistency violated after MIXED operations:\n" \
                    f"Metadata chunk IDs: {sorted(metadata_chunk_ids)}\n" \
                    f"Vector IDs: {sorted(vector_ids)}\n" \
                    f"Missing in vectors: {metadata_chunk_ids - vector_ids}\n" \
                    f"Missing in metadata: {vector_ids - metadata_chunk_ids}"
                
                # Verify expected count
                expected_docs = initial_docs - docs_to_delete + new_docs
                expected_chunks = expected_docs * chunks_per_doc
                assert len(metadata_chunk_ids) == expected_chunks, \
                    f"Expected {expected_chunks} chunks after mixed operations, got {len(metadata_chunk_ids)}"
                
                assert vector_store.get_vector_count() == expected_chunks, \
                    f"Expected {expected_chunks} vectors after mixed operations, got {vector_store.get_vector_count()}"
            
            finally:
                # Clean up temporary database
                if os.path.exists(temp_db_path):
                    os.remove(temp_db_path)
        
        # Run the property test
        run_test()
