"""
LLM Generator for the AI Knowledge Copilot.

This module provides LLM-based answer generation with hallucination control.
It supports multiple backends (OpenAI and Hugging Face) and implements
context-grounded prompts to ensure answers are based only on provided documents.
"""
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import os
from src.models import RetrievedChunk, Citation


@dataclass
class AnswerWithCitations:
    """
    Container for an answer with its citations.
    
    Attributes:
        answer: The generated answer text
        citations: List of citations extracted from the answer
        raw_response: The raw response from the LLM
    """
    answer: str
    citations: List[Citation]
    raw_response: str


class LLMGenerator:
    """
    Generates answers using LLM with hallucination control.
    
    This class supports multiple LLM backends (OpenAI and Hugging Face)
    and implements context-grounded prompts to ensure answers are based
    only on the provided document chunks. It includes hallucination control
    instructions and citation generation.
    
    Attributes:
        model_type: Type of model ("openai" or "huggingface")
        model_name: Name of the specific model to use
        temperature: Temperature parameter for generation (0.0-2.0)
        max_tokens: Maximum tokens to generate
        client: The LLM client (OpenAI or Hugging Face)
    """
    
    # Prompt template with hallucination control instructions
    PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based ONLY on the provided context.

IMPORTANT INSTRUCTIONS:
- Answer ONLY using information from the context below
- If the answer cannot be found in the context, say "I don't have enough information to answer this question."
- Do NOT use external knowledge or make assumptions
- Always cite your sources using [Source: document_name, page X] format
- If multiple sources support your answer, cite all of them
- Be concise and accurate

Context:
{context}

Question: {query}

Answer:"""
    
    def __init__(
        self,
        model_type: str,
        model_name: str,
        temperature: float = 0.1,
        max_tokens: int = 500,
        api_key: Optional[str] = None
    ):
        """
        Initialize the LLM Generator.
        
        Args:
            model_type: Type of model - "openai" or "huggingface"
            model_name: Name of the specific model (e.g., "gpt-3.5-turbo", "meta-llama/Llama-2-7b-chat-hf")
            temperature: Temperature for generation (0.0-2.0). Lower = more deterministic.
            max_tokens: Maximum tokens to generate in the response
            api_key: Optional API key. If not provided, will use environment variable.
        
        Raises:
            ValueError: If model_type is not "openai" or "huggingface"
            ValueError: If temperature is not in valid range
            ValueError: If max_tokens is not positive
            ImportError: If required libraries are not installed
        
        Validates:
            - Requirements 10.1: Allow selection between OpenAI and open-source models
            - Requirements 10.2: Use OpenAI API for text generation
            - Requirements 10.4: Maintain consistent API interfaces
        
        Example:
            >>> # OpenAI
            >>> generator = LLMGenerator("openai", "gpt-3.5-turbo")
            >>> # Hugging Face
            >>> generator = LLMGenerator("huggingface", "meta-llama/Llama-2-7b-chat-hf")
        """
        # Validate inputs
        if model_type not in ["openai", "huggingface"]:
            raise ValueError("model_type must be 'openai' or 'huggingface'")
        
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        
        self.model_type = model_type
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize the appropriate client
        if model_type == "openai":
            self._init_openai_client(api_key)
        else:  # huggingface
            self._init_huggingface_client(api_key)
    
    def _init_openai_client(self, api_key: Optional[str] = None):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: Optional API key. If not provided, uses OPENAI_API_KEY env var.
        
        Raises:
            ImportError: If openai library is not installed
            ValueError: If API key is not provided and not in environment
        
        Validates:
            - Requirements 10.2: Use OpenAI API for text generation
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai library is required for OpenAI models. "
                "Install it with: pip install openai"
            )
        
        # Get API key from parameter or environment
        final_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not final_api_key:
            raise ValueError(
                "OpenAI API key must be provided either as parameter or "
                "via OPENAI_API_KEY environment variable"
            )
        
        self.client = OpenAI(api_key=final_api_key)
    
    def _init_huggingface_client(self, api_key: Optional[str] = None):
        """
        Initialize Hugging Face client for local or API-based models.
        
        Args:
            api_key: Optional Hugging Face API token for gated models
        
        Raises:
            ImportError: If transformers or torch libraries are not installed
        
        Validates:
            - Requirements 10.3: Support open-source models (Llama, Mistral)
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch
        except ImportError:
            raise ImportError(
                "transformers and torch libraries are required for Hugging Face models. "
                "Install them with: pip install transformers torch"
            )
        
        # Store for later use
        self._hf_api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        
        # Initialize tokenizer and model
        # Note: This loads the model locally. For production, consider using
        # Hugging Face Inference API or a model server for better performance.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=self._hf_api_key
        )
        
        # Determine device (GPU if available, else CPU)
        device = 0 if torch.cuda.is_available() else -1
        
        # Create text generation pipeline
        self.client = pipeline(
            "text-generation",
            model=self.model_name,
            tokenizer=self.tokenizer,
            device=device,
            token=self._hf_api_key
        )
    
    def _construct_prompt(self, query: str, context: List[RetrievedChunk]) -> str:
        """
        Construct a prompt with context and hallucination control instructions.
        
        Args:
            query: The user's query
            context: List of retrieved chunks to use as context
        
        Returns:
            Formatted prompt string with context and query
        
        Validates:
            - Requirements 5.1: Construct prompt containing query and retrieved context
            - Requirements 6.1: Instruct LLM to answer only from provided context
        
        Example:
            >>> prompt = generator._construct_prompt("What is AI?", chunks)
            >>> assert "What is AI?" in prompt
            >>> assert "based ONLY on the provided context" in prompt
        """
        # Format context from chunks
        context_parts = []
        for i, retrieved_chunk in enumerate(context, 1):
            chunk = retrieved_chunk.chunk
            doc_name = retrieved_chunk.document_name
            page_num = retrieved_chunk.page_number
            
            # Format: [Source 1: document_name, page X]
            source_info = f"[Source {i}: {doc_name}"
            if page_num is not None:
                source_info += f", page {page_num}"
            source_info += "]"
            
            context_parts.append(f"{source_info}\n{chunk.text}\n")
        
        context_text = "\n".join(context_parts)
        
        # Fill in the template (Requirements 5.1, 6.1)
        prompt = self.PROMPT_TEMPLATE.format(
            context=context_text,
            query=query
        )
        
        return prompt
    
    def _validate_prompt_construction(self, prompt: str, query: str, context: List[RetrievedChunk]) -> bool:
        """
        Validate that the prompt contains all required elements.
        
        Args:
            prompt: The constructed prompt
            query: The original query
            context: The context chunks
        
        Returns:
            True if prompt is valid, False otherwise
        
        Validates:
            - Requirements 5.1: Prompt contains query and all context chunks
            - Requirements 6.1: Prompt contains grounding instructions
        """
        # Check query is present
        if query not in prompt:
            return False
        
        # Check grounding instructions are present (Requirement 6.1)
        grounding_keywords = [
            "based ONLY on the provided context",
            "Do NOT use external knowledge"
        ]
        if not all(keyword in prompt for keyword in grounding_keywords):
            return False
        
        # Check all context chunks are present (Requirement 5.1)
        for chunk in context:
            if chunk.chunk.text not in prompt:
                return False
        
        return True
    
    def generate_answer(self, query: str, context: List[RetrievedChunk]) -> str:
        """
        Generate an answer based on the query and context.
        
        This method constructs a context-grounded prompt and generates an answer
        using the configured LLM. The prompt includes hallucination control
        instructions to ensure the answer is based only on the provided context.
        
        Args:
            query: The user's query
            context: List of retrieved chunks to use as context
        
        Returns:
            Generated answer string
        
        Raises:
            ValueError: If query is empty or context is empty
            Exception: If LLM generation fails
        
        Validates:
            - Requirements 5.1: Construct prompt with query and context
            - Requirements 6.1: Instruct LLM to answer only from context
            - Requirements 10.4: Maintain consistent interface across model types
        
        Example:
            >>> generator = LLMGenerator("openai", "gpt-3.5-turbo")
            >>> answer = generator.generate_answer("What is AI?", retrieved_chunks)
            >>> print(answer)
        """
        # Validate inputs
        if not query or not query.strip():
            raise ValueError("query cannot be empty")
        
        if not context:
            raise ValueError("context cannot be empty")
        
        # Construct prompt with hallucination control (Requirements 5.1, 6.1)
        prompt = self._construct_prompt(query, context)
        
        # Validate prompt construction
        if not self._validate_prompt_construction(prompt, query, context):
            raise ValueError("Prompt construction failed validation")
        
        # Generate answer using appropriate backend (Requirement 10.4)
        if self.model_type == "openai":
            answer = self._generate_openai(prompt)
        else:  # huggingface
            answer = self._generate_huggingface(prompt)
        
        return answer
    
    def _generate_openai(self, prompt: str) -> str:
        """
        Generate answer using OpenAI API.
        
        Args:
            prompt: The formatted prompt
        
        Returns:
            Generated answer string
        
        Raises:
            Exception: If API call fails
        
        Validates:
            - Requirements 10.2: Use OpenAI API for text generation
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based only on provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            answer = response.choices[0].message.content.strip()
            return answer
        
        except Exception as e:
            raise Exception(f"OpenAI API call failed: {str(e)}")
    
    def _generate_huggingface(self, prompt: str) -> str:
        """
        Generate answer using Hugging Face model.
        
        Args:
            prompt: The formatted prompt
        
        Returns:
            Generated answer string
        
        Raises:
            Exception: If generation fails
        
        Validates:
            - Requirements 10.3: Support open-source models
        """
        try:
            # Generate using pipeline
            outputs = self.client(
                prompt,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True if self.temperature > 0 else False,
                return_full_text=False  # Only return generated text, not prompt
            )
            
            answer = outputs[0]['generated_text'].strip()
            return answer
        
        except Exception as e:
            raise Exception(f"Hugging Face generation failed: {str(e)}")
    
    def generate_with_citations(
        self,
        query: str,
        context: List[RetrievedChunk]
    ) -> AnswerWithCitations:
        """
        Generate an answer with extracted citations.
        
        This method generates an answer and extracts citation information
        from the response. Citations are parsed from the [Source: ...] format
        in the generated text.
        
        Args:
            query: The user's query
            context: List of retrieved chunks to use as context
        
        Returns:
            AnswerWithCitations object containing answer, citations, and raw response
        
        Raises:
            ValueError: If query is empty or context is empty
            Exception: If generation fails
        
        Validates:
            - Requirements 5.2: Include citations referencing source documents
            - Requirements 5.3: Include document name, page number, and excerpt
            - Requirements 5.5: Provide citations for all relevant sources
        
        Example:
            >>> generator = LLMGenerator("openai", "gpt-3.5-turbo")
            >>> result = generator.generate_with_citations("What is AI?", chunks)
            >>> print(result.answer)
            >>> for citation in result.citations:
            ...     print(f"Source: {citation.document_name}, Page: {citation.page_number}")
        """
        # Generate answer
        answer = self.generate_answer(query, context)
        
        # Extract citations from the answer (Requirements 5.2, 5.3, 5.5)
        citations = self._extract_citations(answer, context)
        
        return AnswerWithCitations(
            answer=answer,
            citations=citations,
            raw_response=answer
        )
    
    def _extract_citations(
        self,
        answer: str,
        context: List[RetrievedChunk]
    ) -> List[Citation]:
        """
        Extract citations from the generated answer.
        
        This method parses citation markers in the format [Source: document_name, page X]
        or [Source X: document_name, page Y] and creates Citation objects.
        
        Args:
            answer: The generated answer text
            context: The context chunks used for generation
        
        Returns:
            List of Citation objects extracted from the answer
        
        Validates:
            - Requirements 5.3: Citations include document name, page number, and excerpt
        """
        import re
        
        citations = []
        
        # Pattern to match [Source: document_name, page X] or [Source X: document_name, page Y]
        # This pattern captures optional source number, document name, and optional page number
        pattern = r'\[Source\s*(?:\d+)?\s*:\s*([^,\]]+)(?:,\s*page\s*(\d+))?\]'
        
        matches = re.finditer(pattern, answer, re.IGNORECASE)
        
        for match in matches:
            doc_name = match.group(1).strip()
            page_num_str = match.group(2)
            page_num = int(page_num_str) if page_num_str else None
            
            # Find the corresponding chunk from context
            matching_chunk = None
            for retrieved_chunk in context:
                if retrieved_chunk.document_name == doc_name:
                    if page_num is None or retrieved_chunk.page_number == page_num:
                        matching_chunk = retrieved_chunk
                        break
            
            # If we found a matching chunk, create a citation
            if matching_chunk:
                # Extract a relevant excerpt (first 200 characters of chunk text)
                excerpt = matching_chunk.chunk.text[:200]
                if len(matching_chunk.chunk.text) > 200:
                    excerpt += "..."
                
                citation = Citation(
                    document_name=doc_name,
                    page_number=page_num,
                    excerpt=excerpt,
                    chunk_id=matching_chunk.chunk.id
                )
                citations.append(citation)
        
        return citations
