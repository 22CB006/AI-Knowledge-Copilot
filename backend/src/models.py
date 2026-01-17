"""
Data models for the AI Knowledge Copilot system.

This module defines all core data structures using Pydantic for validation,
serialization, and type safety.
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict, model_validator
import numpy as np


class Document(BaseModel):
    """
    Represents a document in the system (PDF or web content).
    
    Attributes:
        id: Unique identifier for the document
        source: File path or URL of the document
        source_type: Type of source ("pdf" or "web")
        text: Extracted text content from the document
        metadata: Additional metadata (e.g., title, author, page count)
        created_at: Timestamp when the document was created/uploaded
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    id: str = Field(..., description="Unique document identifier")
    source: str = Field(..., description="File path or URL")
    source_type: str = Field(..., description="Source type: 'pdf' or 'web'")
    text: str = Field(..., description="Extracted text content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    
    @field_validator('source_type')
    @classmethod
    def validate_source_type(cls, v: str) -> str:
        """Validate that source_type is either 'pdf' or 'web'."""
        if v not in ['pdf', 'web']:
            raise ValueError("source_type must be 'pdf' or 'web'")
        return v
    
    @field_validator('text')
    @classmethod
    def validate_text_not_empty(cls, v: str) -> str:
        """Validate that text content is not empty."""
        if not v or not v.strip():
            raise ValueError("text content cannot be empty")
        return v


class Chunk(BaseModel):
    """
    Represents a chunk of text from a document.
    
    Attributes:
        id: Unique identifier for the chunk
        document_id: ID of the parent document
        text: Text content of the chunk
        embedding: Optional embedding vector for the chunk
        metadata: Additional metadata (page_number, chunk_index, etc.)
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    text: str = Field(..., description="Chunk text content")
    embedding: Optional[np.ndarray] = Field(default=None, description="Embedding vector")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    
    @field_validator('text')
    @classmethod
    def validate_text_not_empty(cls, v: str) -> str:
        """Validate that chunk text is not empty."""
        if not v or not v.strip():
            raise ValueError("chunk text cannot be empty")
        return v
    
    @field_validator('metadata')
    @classmethod
    def validate_metadata_completeness(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that metadata contains required fields: document_id, chunk_index
        
        This validates Requirement 2.3 (chunk metadata completeness)
        Note: document_id is already a separate field, but we check
        for chunk_index in metadata as per the design.
        """
        if 'chunk_index' not in v:
            raise ValueError("metadata must contain 'chunk_index'")
        return v
    
    @classmethod
    def model_validate(cls, obj: Any, **kwargs) -> 'Chunk':
        """
        Custom deserialization to handle numpy arrays.
        """
        if isinstance(obj, dict):
            if 'embedding' in obj and obj['embedding'] is not None:
                if isinstance(obj['embedding'], list):
                    obj['embedding'] = np.array(obj['embedding'])
        return super().model_validate(obj, **kwargs)
    
    def model_dump(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Custom serialization to handle numpy arrays.
        """
        data = super().model_dump(**kwargs)
        if self.embedding is not None:
            data['embedding'] = self.embedding.tolist()
        return data


class RetrievedChunk(BaseModel):
    """
    Represents a chunk retrieved from the vector database with similarity score.
    
    Attributes:
        chunk: The retrieved chunk
        score: Similarity or relevance score
        document_name: Name of the source document
        page_number: Optional page number in the source document
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    chunk: Chunk = Field(..., description="Retrieved chunk")
    score: float = Field(..., description="Similarity/relevance score")
    document_name: str = Field(..., description="Source document name")
    page_number: Optional[int] = Field(default=None, description="Page number in source")
    
    @field_validator('score')
    @classmethod
    def validate_score_range(cls, v: float) -> float:
        """Validate that score is non-negative."""
        if v < 0:
            raise ValueError("score must be non-negative")
        return v


class Citation(BaseModel):
    """
    Represents a citation to a source document.
    
    Attributes:
        document_name: Name of the cited document
        page_number: Optional page number
        excerpt: Relevant text excerpt from the source
        chunk_id: ID of the chunk being cited
    """
    document_name: str = Field(..., description="Cited document name")
    page_number: Optional[int] = Field(default=None, description="Page number")
    excerpt: str = Field(..., description="Relevant text excerpt from source")
    chunk_id: str = Field(..., description="ID of the chunk being cited")
    
    @model_validator(mode='after')
    def validate_citation_completeness(self) -> 'Citation':
        """
        Validate that citation has required fields (document_name and excerpt).
        
        This validates Requirement 5.3 (citation field completeness)
        """
        if not self.document_name or not self.document_name.strip():
            raise ValueError("document_name is required and cannot be empty")
        if not self.excerpt or not self.excerpt.strip():
            raise ValueError("excerpt is required and cannot be empty")
        return self


class QueryResponse(BaseModel):
    """
    Represents the response to a user query.
    
    Attributes:
        answer: Generated answer text
        citations: List of citations supporting the answer
        retrieved_chunks: Number of chunks retrieved
        confidence: Confidence score for the answer
    """
    answer: str = Field(..., description="Generated answer text")
    citations: List[Citation] = Field(default_factory=list, description="Supporting citations")
    retrieved_chunks: int = Field(..., description="Number of chunks retrieved")
    confidence: float = Field(..., description="Confidence score")
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence_range(cls, v: float) -> float:
        """Validate that confidence is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("confidence must be between 0 and 1")
        return v
    
    @field_validator('retrieved_chunks')
    @classmethod
    def validate_retrieved_chunks_non_negative(cls, v: int) -> int:
        """Validate that retrieved_chunks is non-negative."""
        if v < 0:
            raise ValueError("retrieved_chunks must be non-negative")
        return v


class APIKeyConfig(BaseModel):
    """
    Configuration for API keys and LLM settings.
    
    Attributes:
        provider: LLM provider ("openai" or "huggingface")
        model_name: Name of the model to use
        temperature: Temperature parameter for generation
        max_tokens: Maximum tokens to generate
    """
    provider: str = Field(..., description="LLM provider: 'openai' or 'huggingface'")
    model_name: str = Field(..., description="Model name")
    temperature: float = Field(default=0.1, description="Temperature parameter")
    max_tokens: int = Field(default=500, description="Maximum tokens to generate")
    
    @field_validator('provider')
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate that provider is either 'openai' or 'huggingface'."""
        if v not in ['openai', 'huggingface']:
            raise ValueError("provider must be 'openai' or 'huggingface'")
        return v
    
    @field_validator('temperature')
    @classmethod
    def validate_temperature_range(cls, v: float) -> float:
        """Validate that temperature is between 0 and 2."""
        if not 0 <= v <= 2:
            raise ValueError("temperature must be between 0 and 2")
        return v
    
    @field_validator('max_tokens')
    @classmethod
    def validate_max_tokens_positive(cls, v: int) -> int:
        """Validate that max_tokens is positive."""
        if v <= 0:
            raise ValueError("max_tokens must be positive")
        return v


class DocumentDetails(BaseModel):
    """
    Detailed information about a document.
    
    Attributes:
        id: Document identifier
        name: Document name
        source_type: Type of source ("pdf" or "web")
        upload_date: Date when document was uploaded
        chunk_count: Number of chunks created from the document
        total_size: Total size of the document in bytes
    """
    id: str = Field(..., description="Document identifier")
    name: str = Field(..., description="Document name")
    source_type: str = Field(..., description="Source type: 'pdf' or 'web'")
    upload_date: datetime = Field(..., description="Upload timestamp")
    chunk_count: int = Field(..., description="Number of chunks")
    total_size: int = Field(..., description="Total size in bytes")
    
    @field_validator('source_type')
    @classmethod
    def validate_source_type(cls, v: str) -> str:
        """Validate that source_type is either 'pdf' or 'web'."""
        if v not in ['pdf', 'web']:
            raise ValueError("source_type must be 'pdf' or 'web'")
        return v
    
    @field_validator('chunk_count')
    @classmethod
    def validate_chunk_count_non_negative(cls, v: int) -> int:
        """Validate that chunk_count is non-negative."""
        if v < 0:
            raise ValueError("chunk_count must be non-negative")
        return v
    
    @field_validator('total_size')
    @classmethod
    def validate_total_size_non_negative(cls, v: int) -> int:
        """Validate that total_size is non-negative."""
        if v < 0:
            raise ValueError("total_size must be non-negative")
        return v
