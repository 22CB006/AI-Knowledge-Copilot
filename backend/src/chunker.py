"""
Text chunking system for the AI Knowledge Copilot.

This module provides functionality to split documents into overlapping chunks
while preserving metadata and handling edge cases.
"""
from typing import List, Dict, Any
import uuid
from src.models import Chunk


class TextChunker:
    """
    Splits text into overlapping chunks with configurable size and overlap.
    
    This class handles the chunking of documents into smaller pieces suitable
    for embedding and retrieval. It maintains overlap between consecutive chunks
    to preserve context and tracks metadata for each chunk.
    
    Attributes:
        chunk_size: Maximum number of characters per chunk
        overlap: Number of characters to overlap between consecutive chunks
    """
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """
        Initialize the TextChunker with configurable parameters.
        
        Args:
            chunk_size: Maximum number of characters per chunk (default: 512)
            overlap: Number of characters to overlap between chunks (default: 50)
        
        Raises:
            ValueError: If chunk_size <= 0 or overlap < 0 or overlap >= chunk_size
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")
        
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(
        self, 
        text: str, 
        document_id: str,
        metadata: Dict[str, Any] = None
    ) -> List[Chunk]:
        """
        Split text into overlapping chunks with metadata preservation.
        
        This method implements the core chunking logic, handling:
        - Text shorter than chunk size (returns single chunk)
        - Empty text (returns empty list)
        - Overlap between consecutive chunks
        - Metadata preservation with chunk_index
        
        Args:
            text: The text to be chunked
            document_id: ID of the parent document
            metadata: Optional base metadata to include in all chunks
        
        Returns:
            List of Chunk objects with preserved metadata
        
        Validates:
            - Requirements 2.1: Configurable chunk size
            - Requirements 2.2: Overlap between consecutive chunks
            - Requirements 2.3: Metadata preservation (document_id, chunk_index)
        """
        # Handle empty text edge case
        if not text or not text.strip():
            return []
        
        # Initialize base metadata
        base_metadata = metadata.copy() if metadata else {}
        
        # Handle text shorter than chunk size edge case
        if len(text) <= self.chunk_size:
            chunk_metadata = base_metadata.copy()
            chunk_metadata['chunk_index'] = 0
            
            chunk = Chunk(
                id=self._generate_chunk_id(document_id, 0),
                document_id=document_id,
                text=text,
                metadata=chunk_metadata
            )
            return [chunk]
        
        # Split text into overlapping chunks
        chunks = []
        chunk_index = 0
        start = 0
        
        while start < len(text):
            # Calculate end position for this chunk
            end = min(start + self.chunk_size, len(text))
            
            # Extract chunk text
            chunk_text = text[start:end]
            
            # Create metadata for this chunk
            chunk_metadata = base_metadata.copy()
            chunk_metadata['chunk_index'] = chunk_index
            
            # Create Chunk object
            chunk = Chunk(
                id=self._generate_chunk_id(document_id, chunk_index),
                document_id=document_id,
                text=chunk_text,
                metadata=chunk_metadata
            )
            chunks.append(chunk)
            
            # Move to next chunk position with overlap
            # If we've reached the end, break
            if end >= len(text):
                break
            
            # Move start position forward by (chunk_size - overlap)
            start = start + self.chunk_size - self.overlap
            chunk_index += 1
        
        return chunks
    
    def chunk_with_overlap(self, text: str) -> List[str]:
        """
        Split text into overlapping string chunks without metadata.
        
        This is a simpler version that returns just the text chunks
        without creating Chunk objects. Useful for testing or when
        metadata is not needed.
        
        Args:
            text: The text to be chunked
        
        Returns:
            List of text strings representing chunks
        """
        # Handle empty text
        if not text or not text.strip():
            return []
        
        # Handle text shorter than chunk size
        if len(text) <= self.chunk_size:
            return [text]
        
        # Split text into overlapping chunks
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            
            # Break if we've reached the end
            if end >= len(text):
                break
            
            # Move to next position with overlap
            start = start + self.chunk_size - self.overlap
        
        return chunks
    
    def _generate_chunk_id(self, document_id: str, chunk_index: int) -> str:
        """
        Generate a unique ID for a chunk.
        
        Args:
            document_id: ID of the parent document
            chunk_index: Index of the chunk within the document
        
        Returns:
            Unique chunk ID string
        """
        return f"{document_id}_chunk_{chunk_index}_{uuid.uuid4().hex[:8]}"
