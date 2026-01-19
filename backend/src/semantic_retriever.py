"""
Semantic retrieval system for the AI Knowledge Copilot.

This module provides semantic search functionality by combining embedding
generation with FAISS vector search. It retrieves relevant chunks based on
query similarity and supports threshold filtering.
"""
from typing import List, Tuple, Optional
import numpy as np
from src.vector_store import VectorStore
from src.embedder import EmbeddingGenerator
from src.metadata_store import MetadataStore
from src.models import RetrievedChunk, Chunk


class SemanticRetriever:
    """
    Performs semantic retrieval using embeddings and vector search.
    
    This class coordinates query embedding generation, vector similarity search,
    and metadata retrieval to return relevant chunks with their context.
    
    Attributes:
        vector_store: VectorStore instance for similarity search
        embedder: EmbeddingGenerator for converting queries to embeddings
        metadata_store: MetadataStore for retrieving chunk metadata
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: EmbeddingGenerator,
        metadata_store: MetadataStore
    ):
        """
        Initialize the SemanticRetriever.
        
        Args:
            vector_store: VectorStore instance for similarity search
            embedder: EmbeddingGenerator for query embedding
            metadata_store: MetadataStore for chunk metadata retrieval
        
        Validates:
            - Requirements 3.1: Convert query to embedding
            - Requirements 3.2: Perform similarity search
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.metadata_store = metadata_store
    
    def retrieve(
        self,
        query: str,
        k: int = 10,
        threshold: Optional[float] = None
    ) -> List[RetrievedChunk]:
        """
        Retrieve relevant chunks for a query.
        
        This method converts the query to an embedding, searches the vector
        database for similar chunks, retrieves their metadata, and returns
        RetrievedChunk objects with all necessary information.
        
        Args:
            query: The user's natural language query
            k: Number of top results to retrieve (default: 10)
            threshold: Optional similarity threshold for filtering results.
                      Results with distance > threshold are excluded.
        
        Returns:
            List of RetrievedChunk objects, sorted by relevance (best first)
        
        Raises:
            ValueError: If query is empty or k <= 0
        
        Validates:
            - Requirements 3.1: Convert query into embedding vector
            - Requirements 3.2: Perform similarity search in vector database
            - Requirements 3.3: Retrieve top-k most relevant chunks
            - Requirements 3.4: Return results ranked by similarity score
            - Requirements 3.5: Inform user when no relevant information found
        
        Example:
            >>> retriever = SemanticRetriever(vector_store, embedder, metadata_store)
            >>> results = retriever.retrieve("What is machine learning?", k=5)
            >>> for chunk in results:
            ...     print(f"Score: {chunk.score}, Doc: {chunk.document_name}")
        """
        # Validate inputs
        if not query or not query.strip():
            raise ValueError("query cannot be empty")
        
        if k <= 0:
            raise ValueError("k must be positive")
        
        # Convert query to embedding (Requirement 3.1)
        query_embedding = self.embedder.embed_text(query)
        
        # Perform similarity search (Requirement 3.2, 3.3)
        distances, chunk_ids = self.vector_store.search(
            query_embedding,
            k=k,
            threshold=threshold
        )
        
        # Handle empty results (Requirement 3.5)
        if not chunk_ids:
            return []
        
        # Retrieve metadata for each chunk and construct RetrievedChunk objects
        retrieved_chunks = []
        
        for distance, chunk_id in zip(distances, chunk_ids):
            # Get chunk metadata from metadata store
            chunk_metadata = self.metadata_store.get_chunk_metadata(chunk_id)
            
            if chunk_metadata is None:
                # Skip chunks that don't have metadata (shouldn't happen in normal operation)
                continue
            
            # Get document metadata to extract document name
            document_id = chunk_metadata.get('document_id', '')
            doc_metadata = self.metadata_store.get_document_metadata(document_id)
            
            # Extract document name from source
            document_name = ''
            if doc_metadata:
                source = doc_metadata.get('source', '')
                # Extract filename from path or use URL
                if source:
                    document_name = source.split('/')[-1] if '/' in source else source
            
            # Create Chunk object
            chunk = Chunk(
                id=chunk_id,
                document_id=document_id,
                text=chunk_metadata.get('text', ''),
                embedding=None,  # We don't need to load the embedding
                metadata=chunk_metadata
            )
            
            # Convert distance to similarity score
            # FAISS returns L2 distance, we convert to similarity (lower distance = higher similarity)
            # For better UX, we can use: similarity = 1 / (1 + distance)
            similarity_score = 1.0 / (1.0 + distance)
            
            # Create RetrievedChunk object
            retrieved_chunk = RetrievedChunk(
                chunk=chunk,
                score=similarity_score,
                document_name=document_name,
                page_number=chunk_metadata.get('page_number')
            )
            
            retrieved_chunks.append(retrieved_chunk)
        
        # Results are already sorted by distance (best first) from FAISS
        # Requirement 3.4: Return results ranked by similarity score
        return retrieved_chunks
    
    def retrieve_with_scores(
        self,
        query: str,
        k: int = 10
    ) -> List[Tuple[RetrievedChunk, float]]:
        """
        Retrieve chunks with explicit similarity scores.
        
        This is a convenience method that returns tuples of (chunk, score)
        for cases where the score needs to be accessed separately.
        
        Args:
            query: The user's natural language query
            k: Number of top results to retrieve (default: 10)
        
        Returns:
            List of tuples (RetrievedChunk, score), sorted by score (best first)
        
        Raises:
            ValueError: If query is empty or k <= 0
        
        Example:
            >>> retriever = SemanticRetriever(vector_store, embedder, metadata_store)
            >>> results = retriever.retrieve_with_scores("What is AI?", k=3)
            >>> for chunk, score in results:
            ...     print(f"Score: {score:.3f}, Text: {chunk.chunk.text[:50]}")
        """
        # Use the main retrieve method
        retrieved_chunks = self.retrieve(query, k=k, threshold=None)
        
        # Return as tuples of (chunk, score)
        return [(chunk, chunk.score) for chunk in retrieved_chunks]
