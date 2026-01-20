"""
Reranking engine for the AI Knowledge Copilot.

This module provides reranking functionality using cross-encoder models
to improve the relevance of retrieved chunks. Cross-encoders score
query-document pairs more accurately than bi-encoders used in initial retrieval.
"""
from typing import List
from sentence_transformers import CrossEncoder
from src.models import RetrievedChunk


class Reranker:
    """
    Reranks retrieved chunks using a cross-encoder model.
    
    Cross-encoders process query-document pairs jointly, providing more
    accurate relevance scores than the bi-encoder embeddings used in
    initial retrieval. This helps reduce false positives and improve
    answer quality.
    
    Attributes:
        model: CrossEncoder model for scoring query-document pairs
        model_name: Name of the cross-encoder model being used
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the Reranker with a cross-encoder model.
        
        Args:
            model_name: Name of the cross-encoder model to use.
                       Default: "cross-encoder/ms-marco-MiniLM-L-6-v2"
                       This model is trained on MS MARCO passage ranking dataset
                       and provides good balance between speed and accuracy.
        
        Validates:
            - Requirements 4.1: Apply reranking model to retrieved chunks
        
        Example:
            >>> reranker = Reranker()
            >>> # or with custom model
            >>> reranker = Reranker("cross-encoder/ms-marco-MiniLM-L-12-v2")
        """
        self.model_name = model_name
        self.model = CrossEncoder(model_name)
    
    def score_pairs(self, query: str, texts: List[str]) -> List[float]:
        """
        Score query-text pairs for relevance.
        
        This method computes relevance scores for each text given the query.
        Higher scores indicate higher relevance.
        
        Args:
            query: The user's query string
            texts: List of text strings to score against the query
        
        Returns:
            List of relevance scores (floats), one per text.
            Higher scores indicate higher relevance.
        
        Raises:
            ValueError: If query is empty or texts list is empty
        
        Validates:
            - Requirements 4.2: Score each chunk based on relevance to query
        
        Example:
            >>> reranker = Reranker()
            >>> query = "What is machine learning?"
            >>> texts = ["ML is a subset of AI", "The weather is nice"]
            >>> scores = reranker.score_pairs(query, texts)
            >>> print(scores)  # [0.85, 0.12] (example scores)
        """
        # Validate inputs
        if not query or not query.strip():
            raise ValueError("query cannot be empty")
        
        if not texts:
            raise ValueError("texts list cannot be empty")
        
        # Create query-text pairs
        pairs = [[query, text] for text in texts]
        
        # Score all pairs (Requirement 4.2)
        scores = self.model.predict(pairs)
        
        # Convert numpy array to list of floats
        return scores.tolist()
    
    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_n: int = 5
    ) -> List[RetrievedChunk]:
        """
        Rerank retrieved chunks by relevance to the query.
        
        This method takes chunks from initial retrieval, scores them using
        the cross-encoder model, reorders them by relevance, and returns
        the top-n most relevant chunks.
        
        Args:
            query: The user's query string
            chunks: List of RetrievedChunk objects from initial retrieval
            top_n: Number of top chunks to return after reranking (default: 5)
        
        Returns:
            List of RetrievedChunk objects, reranked and limited to top_n.
            Chunks are sorted by reranking score (highest first).
            All original metadata is preserved.
        
        Raises:
            ValueError: If query is empty, chunks list is empty, or top_n <= 0
        
        Validates:
            - Requirements 4.1: Apply reranking model to retrieved chunks
            - Requirements 4.2: Score each chunk based on relevance to query
            - Requirements 4.3: Reorder chunks by reranking score
            - Requirements 4.4: Select top-n chunks for context augmentation
            - Requirements 4.5: Maintain all original metadata for each chunk
        
        Example:
            >>> reranker = Reranker()
            >>> # Assume we have retrieved_chunks from semantic retrieval
            >>> reranked = reranker.rerank("What is AI?", retrieved_chunks, top_n=3)
            >>> for chunk in reranked:
            ...     print(f"Score: {chunk.score}, Text: {chunk.chunk.text[:50]}")
        """
        # Validate inputs
        if not query or not query.strip():
            raise ValueError("query cannot be empty")
        
        if not chunks:
            raise ValueError("chunks list cannot be empty")
        
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        
        # Extract texts from chunks
        texts = [chunk.chunk.text for chunk in chunks]
        
        # Score all query-chunk pairs (Requirements 4.1, 4.2)
        reranking_scores = self.score_pairs(query, texts)
        
        # Create list of (chunk, reranking_score) tuples
        chunks_with_scores = list(zip(chunks, reranking_scores))
        
        # Sort by reranking score in descending order (Requirement 4.3)
        chunks_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top-n chunks (Requirement 4.4)
        top_chunks_with_scores = chunks_with_scores[:top_n]
        
        # Normalize scores to be non-negative for RetrievedChunk validation
        # We use min-max normalization to map scores to [0, 1] range
        # This preserves the relative ordering while ensuring non-negative scores
        if top_chunks_with_scores:
            scores_only = [score for _, score in top_chunks_with_scores]
            min_score = min(scores_only)
            max_score = max(scores_only)
            
            # Handle case where all scores are the same
            score_range = max_score - min_score
            if score_range == 0:
                normalized_scores = [1.0] * len(scores_only)
            else:
                normalized_scores = [
                    (score - min_score) / score_range
                    for score in scores_only
                ]
        else:
            normalized_scores = []
        
        # Update the score field in RetrievedChunk objects with normalized reranking scores
        # while preserving all other metadata (Requirement 4.5)
        reranked_chunks = []
        for (chunk, _), normalized_score in zip(top_chunks_with_scores, normalized_scores):
            # Create a new RetrievedChunk with updated score
            # All other fields (chunk, document_name, page_number) are preserved
            reranked_chunk = RetrievedChunk(
                chunk=chunk.chunk,  # Preserve original chunk with all metadata
                score=float(normalized_score),  # Update with normalized reranking score
                document_name=chunk.document_name,  # Preserve document name
                page_number=chunk.page_number  # Preserve page number
            )
            reranked_chunks.append(reranked_chunk)
        
        return reranked_chunks
