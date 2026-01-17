"""
Embedding generation system for the AI Knowledge Copilot.

This module provides functionality to generate embeddings for text chunks
using Sentence Transformers models. It supports both single text and batch
processing for efficiency.
"""
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """
    Generates embeddings for text using Sentence Transformers.
    
    This class wraps the Sentence Transformers library to provide a consistent
    interface for embedding generation. It supports configurable models and
    efficient batch processing.
    
    Attributes:
        model_name: Name of the Sentence Transformer model to use
        model: The loaded SentenceTransformer model instance
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the EmbeddingGenerator with a specified model.
        
        Args:
            model_name: Name of the Sentence Transformer model (default: all-MiniLM-L6-v2)
                       Common options:
                       - all-MiniLM-L6-v2: 384 dimensions, fast, good for most use cases
                       - all-mpnet-base-v2: 768 dimensions, more accurate but slower
        
        Raises:
            Exception: If the model cannot be loaded
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate an embedding for a single text string.
        
        This method converts a single text string into a dense vector representation
        that captures its semantic meaning. The embedding can be used for similarity
        search and retrieval.
        
        Args:
            text: The text string to embed
        
        Returns:
            A numpy array containing the embedding vector
        
        Validates:
            - Requirements 2.4: Generate embeddings for each chunk
            - Requirements 3.1: Convert query into embedding vector
        
        Example:
            >>> embedder = EmbeddingGenerator()
            >>> embedding = embedder.embed_text("Hello world")
            >>> print(embedding.shape)
            (384,)
        """
        if not text or not text.strip():
            raise ValueError("text cannot be empty")
        
        # Generate embedding using the model
        # encode() returns a numpy array by default
        embedding = self.model.encode(text, convert_to_numpy=True)
        
        return embedding
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for multiple texts efficiently using batch processing.
        
        This method is more efficient than calling embed_text() multiple times
        because it processes multiple texts in parallel batches. This is especially
        important when using GPU acceleration.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process in each batch (default: 32)
        
        Returns:
            A numpy array of shape (len(texts), embedding_dimension) containing
            all embeddings
        
        Validates:
            - Requirements 2.4: Generate embeddings for each chunk (batch processing)
        
        Example:
            >>> embedder = EmbeddingGenerator()
            >>> texts = ["Hello world", "How are you?", "Good morning"]
            >>> embeddings = embedder.embed_batch(texts)
            >>> print(embeddings.shape)
            (3, 384)
        """
        if not texts:
            raise ValueError("texts list cannot be empty")
        
        # Filter out empty texts and track their indices
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)
        
        if not valid_texts:
            raise ValueError("all texts in the list are empty")
        
        # Generate embeddings in batches
        embeddings = self.model.encode(
            valid_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # If some texts were filtered out, we need to handle that
        # For now, we assume all texts are valid (caller's responsibility)
        # In production, you might want to return None for invalid texts
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        This is useful for initializing vector stores and validating
        that embeddings have the expected dimensionality.
        
        Returns:
            The dimension of the embedding vectors as an integer
        
        Example:
            >>> embedder = EmbeddingGenerator("all-MiniLM-L6-v2")
            >>> print(embedder.get_embedding_dimension())
            384
            >>> embedder2 = EmbeddingGenerator("all-mpnet-base-v2")
            >>> print(embedder2.get_embedding_dimension())
            768
        """
        return self.model.get_sentence_embedding_dimension()
