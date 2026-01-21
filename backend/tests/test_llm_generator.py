"""
Unit tests for the LLM Generator.

Tests cover initialization, prompt construction, answer generation,
and citation extraction for both OpenAI and Hugging Face backends.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.llm_generator import LLMGenerator, AnswerWithCitations
from src.models import RetrievedChunk, Chunk, Citation


# Test fixtures
@pytest.fixture
def sample_chunks():
    """Create sample retrieved chunks for testing."""
    chunk1 = Chunk(
        id="chunk1",
        document_id="doc1",
        text="Artificial Intelligence (AI) is the simulation of human intelligence by machines.",
        embedding=None,
        metadata={"chunk_index": 0, "page_number": 1}
    )
    
    chunk2 = Chunk(
        id="chunk2",
        document_id="doc1",
        text="Machine learning is a subset of AI that enables systems to learn from data.",
        embedding=None,
        metadata={"chunk_index": 1, "page_number": 2}
    )
    
    retrieved1 = RetrievedChunk(
        chunk=chunk1,
        score=0.9,
        document_name="AI_Basics.pdf",
        page_number=1
    )
    
    retrieved2 = RetrievedChunk(
        chunk=chunk2,
        score=0.8,
        document_name="AI_Basics.pdf",
        page_number=2
    )
    
    return [retrieved1, retrieved2]


class TestLLMGeneratorInitialization:
    """Test LLM Generator initialization."""
    
    def test_init_with_valid_openai_params(self):
        """Test initialization with valid OpenAI parameters."""
        with patch('openai.OpenAI'):
            generator = LLMGenerator(
                model_type="openai",
                model_name="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=500,
                api_key="test-key"
            )
            
            assert generator.model_type == "openai"
            assert generator.model_name == "gpt-3.5-turbo"
            assert generator.temperature == 0.7
            assert generator.max_tokens == 500
    
    def test_init_with_invalid_model_type(self):
        """Test initialization fails with invalid model type."""
        with pytest.raises(ValueError, match="model_type must be"):
            LLMGenerator(
                model_type="invalid",
                model_name="gpt-3.5-turbo",
                api_key="test-key"
            )
    
    def test_init_with_invalid_temperature(self):
        """Test initialization fails with invalid temperature."""
        with pytest.raises(ValueError, match="temperature must be between"):
            LLMGenerator(
                model_type="openai",
                model_name="gpt-3.5-turbo",
                temperature=3.0,
                api_key="test-key"
            )
    
    def test_init_with_invalid_max_tokens(self):
        """Test initialization fails with non-positive max_tokens."""
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            LLMGenerator(
                model_type="openai",
                model_name="gpt-3.5-turbo",
                max_tokens=0,
                api_key="test-key"
            )
    
    def test_init_openai_without_api_key(self):
        """Test OpenAI initialization fails without API key."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI API key must be provided"):
                LLMGenerator(
                    model_type="openai",
                    model_name="gpt-3.5-turbo"
                )
    
    def test_init_openai_with_env_api_key(self):
        """Test OpenAI initialization succeeds with environment API key."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'env-test-key'}):
            with patch('openai.OpenAI') as mock_openai:
                generator = LLMGenerator(
                    model_type="openai",
                    model_name="gpt-3.5-turbo"
                )
                
                # Verify OpenAI was initialized with the env key
                mock_openai.assert_called_once_with(api_key='env-test-key')


class TestPromptConstruction:
    """Test prompt construction with hallucination control."""
    
    def test_construct_prompt_contains_query(self, sample_chunks):
        """Test that constructed prompt contains the query."""
        with patch('openai.OpenAI'):
            generator = LLMGenerator(
                model_type="openai",
                model_name="gpt-3.5-turbo",
                api_key="test-key"
            )
            
            query = "What is artificial intelligence?"
            prompt = generator._construct_prompt(query, sample_chunks)
            
            assert query in prompt
    
    def test_construct_prompt_contains_context(self, sample_chunks):
        """Test that constructed prompt contains all context chunks."""
        with patch('openai.OpenAI'):
            generator = LLMGenerator(
                model_type="openai",
                model_name="gpt-3.5-turbo",
                api_key="test-key"
            )
            
            query = "What is AI?"
            prompt = generator._construct_prompt(query, sample_chunks)
            
            # Check all chunk texts are in the prompt
            for chunk in sample_chunks:
                assert chunk.chunk.text in prompt
    
    def test_construct_prompt_contains_grounding_instructions(self, sample_chunks):
        """Test that prompt contains hallucination control instructions."""
        with patch('openai.OpenAI'):
            generator = LLMGenerator(
                model_type="openai",
                model_name="gpt-3.5-turbo",
                api_key="test-key"
            )
            
            query = "What is AI?"
            prompt = generator._construct_prompt(query, sample_chunks)
            
            # Check for grounding instructions (Requirement 6.1)
            assert "based ONLY on the provided context" in prompt
            assert "Do NOT use external knowledge" in prompt
    
    def test_construct_prompt_includes_source_info(self, sample_chunks):
        """Test that prompt includes source document and page information."""
        with patch('openai.OpenAI'):
            generator = LLMGenerator(
                model_type="openai",
                model_name="gpt-3.5-turbo",
                api_key="test-key"
            )
            
            query = "What is AI?"
            prompt = generator._construct_prompt(query, sample_chunks)
            
            # Check source information is included
            assert "AI_Basics.pdf" in prompt
            assert "page 1" in prompt
            assert "page 2" in prompt
    
    def test_validate_prompt_construction_success(self, sample_chunks):
        """Test prompt validation succeeds for valid prompt."""
        with patch('openai.OpenAI'):
            generator = LLMGenerator(
                model_type="openai",
                model_name="gpt-3.5-turbo",
                api_key="test-key"
            )
            
            query = "What is AI?"
            prompt = generator._construct_prompt(query, sample_chunks)
            
            is_valid = generator._validate_prompt_construction(prompt, query, sample_chunks)
            assert is_valid is True
    
    def test_validate_prompt_construction_missing_query(self, sample_chunks):
        """Test prompt validation fails when query is missing."""
        with patch('openai.OpenAI'):
            generator = LLMGenerator(
                model_type="openai",
                model_name="gpt-3.5-turbo",
                api_key="test-key"
            )
            
            query = "What is AI?"
            prompt = "This prompt doesn't contain the query"
            
            is_valid = generator._validate_prompt_construction(prompt, query, sample_chunks)
            assert is_valid is False


class TestAnswerGeneration:
    """Test answer generation functionality."""
    
    def test_generate_answer_with_empty_query(self, sample_chunks):
        """Test that generation fails with empty query."""
        with patch('openai.OpenAI'):
            generator = LLMGenerator(
                model_type="openai",
                model_name="gpt-3.5-turbo",
                api_key="test-key"
            )
            
            with pytest.raises(ValueError, match="query cannot be empty"):
                generator.generate_answer("", sample_chunks)
    
    def test_generate_answer_with_empty_context(self):
        """Test that generation fails with empty context."""
        with patch('openai.OpenAI'):
            generator = LLMGenerator(
                model_type="openai",
                model_name="gpt-3.5-turbo",
                api_key="test-key"
            )
            
            with pytest.raises(ValueError, match="context cannot be empty"):
                generator.generate_answer("What is AI?", [])
    
    def test_generate_answer_openai_success(self, sample_chunks):
        """Test successful answer generation with OpenAI."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "AI is the simulation of human intelligence. [Source: AI_Basics.pdf, page 1]"
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch('openai.OpenAI'):
            generator = LLMGenerator(
                model_type="openai",
                model_name="gpt-3.5-turbo",
                api_key="test-key"
            )
            generator.client = mock_client
            
            answer = generator.generate_answer("What is AI?", sample_chunks)
            
            assert "AI is the simulation of human intelligence" in answer
            assert mock_client.chat.completions.create.called
    
    def test_generate_answer_openai_api_failure(self, sample_chunks):
        """Test handling of OpenAI API failure."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        with patch('openai.OpenAI'):
            generator = LLMGenerator(
                model_type="openai",
                model_name="gpt-3.5-turbo",
                api_key="test-key"
            )
            generator.client = mock_client
            
            with pytest.raises(Exception, match="OpenAI API call failed"):
                generator.generate_answer("What is AI?", sample_chunks)


class TestCitationExtraction:
    """Test citation extraction from generated answers."""
    
    def test_extract_citations_with_valid_format(self, sample_chunks):
        """Test citation extraction with valid [Source: ...] format."""
        with patch('openai.OpenAI'):
            generator = LLMGenerator(
                model_type="openai",
                model_name="gpt-3.5-turbo",
                api_key="test-key"
            )
            
            answer = "AI is the simulation of human intelligence. [Source: AI_Basics.pdf, page 1]"
            citations = generator._extract_citations(answer, sample_chunks)
            
            assert len(citations) > 0
            assert citations[0].document_name == "AI_Basics.pdf"
            assert citations[0].page_number == 1
            assert citations[0].excerpt is not None
    
    def test_extract_citations_with_numbered_sources(self, sample_chunks):
        """Test citation extraction with numbered sources [Source 1: ...]."""
        with patch('openai.OpenAI'):
            generator = LLMGenerator(
                model_type="openai",
                model_name="gpt-3.5-turbo",
                api_key="test-key"
            )
            
            answer = "AI is important. [Source 1: AI_Basics.pdf, page 1] ML is a subset. [Source 2: AI_Basics.pdf, page 2]"
            citations = generator._extract_citations(answer, sample_chunks)
            
            assert len(citations) == 2
            assert citations[0].document_name == "AI_Basics.pdf"
            assert citations[1].document_name == "AI_Basics.pdf"
    
    def test_extract_citations_without_page_number(self, sample_chunks):
        """Test citation extraction when page number is not specified."""
        with patch('openai.OpenAI'):
            generator = LLMGenerator(
                model_type="openai",
                model_name="gpt-3.5-turbo",
                api_key="test-key"
            )
            
            answer = "AI is important. [Source: AI_Basics.pdf]"
            citations = generator._extract_citations(answer, sample_chunks)
            
            assert len(citations) > 0
            assert citations[0].document_name == "AI_Basics.pdf"
            assert citations[0].page_number is None or isinstance(citations[0].page_number, int)
    
    def test_extract_citations_no_matches(self, sample_chunks):
        """Test citation extraction when no citations are present."""
        with patch('openai.OpenAI'):
            generator = LLMGenerator(
                model_type="openai",
                model_name="gpt-3.5-turbo",
                api_key="test-key"
            )
            
            answer = "This answer has no citations."
            citations = generator._extract_citations(answer, sample_chunks)
            
            assert len(citations) == 0


class TestGenerateWithCitations:
    """Test generate_with_citations method."""
    
    def test_generate_with_citations_returns_answer_and_citations(self, sample_chunks):
        """Test that generate_with_citations returns both answer and citations."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "AI is the simulation of human intelligence. [Source: AI_Basics.pdf, page 1]"
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch('openai.OpenAI'):
            generator = LLMGenerator(
                model_type="openai",
                model_name="gpt-3.5-turbo",
                api_key="test-key"
            )
            generator.client = mock_client
            
            result = generator.generate_with_citations("What is AI?", sample_chunks)
            
            assert isinstance(result, AnswerWithCitations)
            assert result.answer is not None
            assert isinstance(result.citations, list)
            assert result.raw_response is not None
    
    def test_generate_with_citations_extracts_multiple_citations(self, sample_chunks):
        """Test extraction of multiple citations from answer."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = (
            "AI is the simulation of human intelligence [Source: AI_Basics.pdf, page 1]. "
            "ML is a subset of AI [Source: AI_Basics.pdf, page 2]."
        )
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch('openai.OpenAI'):
            generator = LLMGenerator(
                model_type="openai",
                model_name="gpt-3.5-turbo",
                api_key="test-key"
            )
            generator.client = mock_client
            
            result = generator.generate_with_citations("What is AI and ML?", sample_chunks)
            
            # Should extract citations for both sources
            assert len(result.citations) >= 1


class TestModelInterfaceConsistency:
    """Test that both OpenAI and Hugging Face maintain consistent interfaces."""
    
    def test_openai_interface_consistency(self, sample_chunks):
        """Test OpenAI generator has consistent interface."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test answer"
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch('openai.OpenAI'):
            generator = LLMGenerator(
                model_type="openai",
                model_name="gpt-3.5-turbo",
                api_key="test-key"
            )
            generator.client = mock_client
            
            # Test that the interface methods exist and work
            answer = generator.generate_answer("Test query", sample_chunks)
            assert isinstance(answer, str)
            
            result = generator.generate_with_citations("Test query", sample_chunks)
            assert isinstance(result, AnswerWithCitations)
    
    def test_huggingface_interface_consistency(self, sample_chunks):
        """Test Hugging Face generator has consistent interface."""
        # Mock the Hugging Face pipeline
        mock_pipeline = Mock()
        mock_pipeline.return_value = [{'generated_text': 'Test answer'}]
        
        with patch('transformers.AutoTokenizer'):
            with patch('transformers.pipeline', return_value=mock_pipeline):
                generator = LLMGenerator(
                    model_type="huggingface",
                    model_name="gpt2",  # Small model for testing
                    api_key="test-key"
                )
                
                # Test that the interface methods exist and work
                answer = generator.generate_answer("Test query", sample_chunks)
                assert isinstance(answer, str)
                
                result = generator.generate_with_citations("Test query", sample_chunks)
                assert isinstance(result, AnswerWithCitations)
