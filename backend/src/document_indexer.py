"""
Document indexing pipeline for the AI Knowledge Copilot.

This module provides the DocumentIndexer class that orchestrates the full
document processing pipeline: document processing → chunking → embedding → storage.
"""
import logging
from typing import List, Optional
from datetime import datetime

from .document_processor import DocumentProcessor
from .chunker import TextChunker
from .embedder import EmbeddingGenerator
from .vector_store import VectorStore
from .metadata_store import MetadataStore
from .models import Document, Chunk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentIndexerError(Exception):
    """Base exception for document indexing errors."""
    pass


class DocumentIndexer:
    """
    Orchestrates the full document indexing pipeline.
    
    This class coordinates document processing, chunking, embedding generation,
    and storage in both the vector database and metadata store. It ensures
    document independence and handles multiple documents without interference.
    
    Validates Requirements 1.5, 2.1, 2.2, 2.3, 2.4, 2.5
    """
    
    def __init__(
        self,
        document_processor: DocumentProcessor,
        chunker: TextChunker,
        embedder: EmbeddingGenerator,
        vector_store: VectorStore,
        metadata_store: MetadataStore
    ):
        """
        Initialize the DocumentIndexer with all required components.
        
        Args:
            document_processor: Component for extracting text from documents
            chunker: Component for splitting text into chunks
            embedder: Component for generating embeddings
            vector_store: FAISS-based vector storage
            metadata_store: SQLite-based metadata storage
        """
        self.document_processor = document_processor
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.metadata_store = metadata_store
        
        logger.info("DocumentIndexer initialized")
    
    def index_document(
        self,
        source: str,
        source_type: str,
        additional_metadata: Optional[dict] = None
    ) -> str:
        """
        Index a document through the full pipeline.
        
        This method orchestrates the complete indexing process:
        1. Process document (extract text)
        2. Chunk text into overlapping pieces
        3. Generate embeddings for each chunk
        4. Store embeddings in FAISS vector database
        5. Store metadata in SQLite database
        
        Each document is tracked with a unique ID and processed independently
        to ensure no interference between documents (Requirement 1.5).
        
        Args:
            source: File path (for PDF) or URL (for web content)
            source_type: Type of source ("pdf" or "web")
            additional_metadata: Optional additional metadata to store
        
        Returns:
            Document ID of the indexed document
        
        Raises:
            DocumentIndexerError: If any step of the pipeline fails
        
        Validates:
            - Requirements 1.5: Document independence
            - Requirements 2.1: Configurable chunk size
            - Requirements 2.2: Chunk overlap
            - Requirements 2.3: Metadata preservation
            - Requirements 2.4: Embedding generation
            - Requirements 2.5: Vector storage
        
        Example:
            >>> indexer = DocumentIndexer(...)
            >>> doc_id = indexer.index_document("paper.pdf", "pdf")
            >>> print(f"Indexed document: {doc_id}")
        """
        try:
            logger.info(f"Starting indexing pipeline for {source_type}: {source}")
            
            # Step 1: Process document to extract text
            logger.info("Step 1/5: Processing document...")
            document = self._process_document(source, source_type)
            
            # Add any additional metadata
            if additional_metadata:
                document.metadata.update(additional_metadata)
            
            # Step 2: Store document metadata
            logger.info("Step 2/5: Storing document metadata...")
            self._store_document_metadata(document)
            
            # Step 3: Chunk the document text
            logger.info("Step 3/5: Chunking document text...")
            chunks = self._chunk_document(document)
            
            if not chunks:
                raise DocumentIndexerError(
                    f"No chunks generated from document: {source}"
                )
            
            logger.info(f"Generated {len(chunks)} chunks")
            
            # Step 4: Generate embeddings for all chunks
            logger.info("Step 4/5: Generating embeddings...")
            chunks_with_embeddings = self._generate_embeddings(chunks)
            
            # Step 5: Store chunks in vector database and metadata store
            logger.info("Step 5/5: Storing chunks in vector database and metadata store...")
            self._store_chunks(chunks_with_embeddings)
            
            logger.info(
                f"Successfully indexed document {document.id} "
                f"with {len(chunks)} chunks"
            )
            
            return document.id
            
        except Exception as e:
            logger.error(f"Error indexing document {source}: {str(e)}")
            raise DocumentIndexerError(
                f"Failed to index document: {str(e)}"
            ) from e
    
    def _process_document(self, source: str, source_type: str) -> Document:
        """
        Process document to extract text.
        
        Args:
            source: File path or URL
            source_type: "pdf" or "web"
        
        Returns:
            Document object with extracted text
        
        Raises:
            DocumentIndexerError: If document processing fails
        """
        try:
            if source_type == "pdf":
                document = self.document_processor.process_pdf(source)
            elif source_type == "web":
                document = self.document_processor.process_url(source)
            else:
                raise DocumentIndexerError(
                    f"Unsupported source type: {source_type}"
                )
            
            return document
            
        except Exception as e:
            raise DocumentIndexerError(
                f"Document processing failed: {str(e)}"
            ) from e
    
    def _store_document_metadata(self, document: Document) -> None:
        """
        Store document metadata in the metadata store.
        
        Args:
            document: Document object to store
        
        Raises:
            DocumentIndexerError: If metadata storage fails
        """
        try:
            metadata = {
                'source': document.source,
                'source_type': document.source_type,
                'created_at': document.created_at.isoformat(),
                **document.metadata
            }
            
            self.metadata_store.add_document(document.id, metadata)
            
        except Exception as e:
            raise DocumentIndexerError(
                f"Failed to store document metadata: {str(e)}"
            ) from e
    
    def _chunk_document(self, document: Document) -> List[Chunk]:
        """
        Chunk document text into overlapping pieces.
        
        Args:
            document: Document to chunk
        
        Returns:
            List of Chunk objects
        
        Raises:
            DocumentIndexerError: If chunking fails
        """
        try:
            # Prepare base metadata for chunks
            base_metadata = {
                'source': document.source,
                'source_type': document.source_type
            }
            
            # Add page_number if available in document metadata
            if 'page_number' in document.metadata:
                base_metadata['page_number'] = document.metadata['page_number']
            
            # Chunk the text
            chunks = self.chunker.chunk_text(
                text=document.text,
                document_id=document.id,
                metadata=base_metadata
            )
            
            return chunks
            
        except Exception as e:
            raise DocumentIndexerError(
                f"Chunking failed: {str(e)}"
            ) from e
    
    def _generate_embeddings(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Generate embeddings for all chunks.
        
        Args:
            chunks: List of chunks to embed
        
        Returns:
            List of chunks with embeddings added
        
        Raises:
            DocumentIndexerError: If embedding generation fails
        """
        try:
            # Extract text from all chunks
            chunk_texts = [chunk.text for chunk in chunks]
            
            # Generate embeddings in batch for efficiency
            embeddings = self.embedder.embed_batch(chunk_texts)
            
            # Add embeddings to chunks
            for i, chunk in enumerate(chunks):
                chunk.embedding = embeddings[i]
            
            return chunks
            
        except Exception as e:
            raise DocumentIndexerError(
                f"Embedding generation failed: {str(e)}"
            ) from e
    
    def _store_chunks(self, chunks: List[Chunk]) -> None:
        """
        Store chunks in both vector database and metadata store.
        
        This ensures consistency between the vector index and metadata storage
        (Requirement 11.5).
        
        Args:
            chunks: List of chunks with embeddings to store
        
        Raises:
            DocumentIndexerError: If storage fails
        """
        try:
            # Prepare data for vector store
            chunk_ids = [chunk.id for chunk in chunks]
            embeddings = [chunk.embedding for chunk in chunks]
            
            # Convert list of embeddings to numpy array
            import numpy as np
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Store in vector database
            self.vector_store.add_vectors(embeddings_array, chunk_ids)
            
            # Store metadata for each chunk
            for chunk in chunks:
                chunk_metadata = {
                    'document_id': chunk.document_id,
                    'text': chunk.text,
                    **chunk.metadata
                }
                self.metadata_store.add_chunk(chunk.id, chunk_metadata)
            
            logger.info(f"Stored {len(chunks)} chunks in vector database and metadata store")
            
        except Exception as e:
            raise DocumentIndexerError(
                f"Chunk storage failed: {str(e)}"
            ) from e
    
    def index_pdf(self, file_path: str, additional_metadata: Optional[dict] = None) -> str:
        """
        Convenience method to index a PDF file.
        
        Args:
            file_path: Path to the PDF file
            additional_metadata: Optional additional metadata
        
        Returns:
            Document ID
        """
        return self.index_document(file_path, "pdf", additional_metadata)
    
    def index_url(self, url: str, additional_metadata: Optional[dict] = None) -> str:
        """
        Convenience method to index web content from a URL.
        
        Args:
            url: URL to fetch and index
            additional_metadata: Optional additional metadata
        
        Returns:
            Document ID
        """
        return self.index_document(url, "web", additional_metadata)
    
    def get_document_info(self, doc_id: str) -> Optional[dict]:
        """
        Get information about an indexed document.
        
        Args:
            doc_id: Document ID
        
        Returns:
            Dictionary with document information, or None if not found
        """
        try:
            doc_metadata = self.metadata_store.get_document_metadata(doc_id)
            if not doc_metadata:
                return None
            
            # Get chunk count
            chunk_ids = self.metadata_store.get_document_chunks(doc_id)
            doc_metadata['chunk_count'] = len(chunk_ids)
            
            return doc_metadata
            
        except Exception as e:
            logger.error(f"Error getting document info for {doc_id}: {str(e)}")
            return None
    
    def list_documents(self) -> List[dict]:
        """
        List all indexed documents.
        
        Returns:
            List of document metadata dictionaries
        """
        try:
            documents = self.metadata_store.list_documents()
            
            # Add chunk count for each document
            for doc in documents:
                chunk_ids = self.metadata_store.get_document_chunks(doc['id'])
                doc['chunk_count'] = len(chunk_ids)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            return []
