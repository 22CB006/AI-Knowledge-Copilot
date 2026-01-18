# Implementation Plan

- [x] 1. Set up project structure and dependencies





  - Create monorepo structure with backend/ and frontend/ directories
  - Set up Python virtual environment for backend
  - Install backend dependencies: FastAPI, LangChain, FAISS, Sentence Transformers
  - Install document processing libraries: PyPDF2, BeautifulSoup4, requests
  - Install testing frameworks: pytest, hypothesis
  - Initialize Next.js project with TypeScript and Tailwind CSS
  - Install frontend dependencies: axios, react-query, zustand, react-dropzone
  - Create directory structure: backend/src/, backend/tests/, frontend/src/, frontend/components/
  - Set up configuration management (config.yaml and environment variables)
  - Configure CORS in FastAPI for frontend communication
  - _Requirements: 8.1, 12.2, 13.1_

- [x] 2. Implement core data models







  - [x] 2.1 Create data model classes using Pydantic

    - Define Document, Chunk, RetrievedChunk, Citation, QueryResponse models
    - Add validation rules and type hints
    - Implement serialization methods
    - _Requirements: 2.3, 5.3_
  - [x] 2.2 Write property test for data model validation

    - **Property 5: Chunk metadata completeness**
    - **Validates: Requirements 2.3**
  - [x] 2.3 Write property test for citation completeness

    - **Property 17: Citation field completeness**
    - **Validates: Requirements 5.3**

- [x] 3. Implement text chunking system
  - [x] 3.1 Create TextChunker class with configurable size and overlap
    - Implement chunk_text() method to split text into chunks
    - Implement overlap logic between consecutive chunks
    - Preserve metadata (document_id, chunk_index) for each chunk
    - Handle edge cases (text shorter than chunk size, empty text)
    - _Requirements: 2.1, 2.2, 2.3_
  - [x] 3.2 Write property test for chunk size constraint

    - **Property 3: Chunk size constraint**
    - **Validates: Requirements 2.1**
  - [x] 3.3 Write property test for chunk overlap

    - **Property 4: Chunk overlap preservation**
    - **Validates: Requirements 2.2**
  - [x] 3.4 Write unit tests for edge cases

    - Test chunking with text shorter than chunk size
    - Test chunking with exact multiples of chunk size
    - Test empty text handling

- [x] 4. Implement embedding generation
  - [x] 4.1 Create EmbeddingGenerator class using Sentence Transformers
    - Initialize with configurable model (default: all-MiniLM-L6-v2)
    - Implement embed_text() for single text
    - Implement embed_batch() for efficient batch processing
    - Add get_embedding_dimension() method
    - _Requirements: 2.4, 3.1_
  - [x] 4.2 Write property test for embedding generation

    - **Property 6: Embedding generation completeness**
    - **Validates: Requirements 2.4**
  - [x] 4.3 Write property test for query embedding consistency

    - **Property 8: Query embedding dimension consistency**
    - **Validates: Requirements 3.1**
  - [x] 4.4 Write unit tests for batch processing

    - Test batch embedding with various batch sizes
    - Verify embedding dimensions match model output

- [ ] 5. Implement FAISS vector store
  - [x] 5.1 Create VectorStore class wrapping FAISS
    - Initialize FAISS index (IndexFlatL2 for development)
    - Implement add_vectors() to insert embeddings with IDs
    - Implement search() for k-nearest neighbor retrieval
    - Implement delete_vectors() for removing embeddings
    - Implement save_index() and load_index() for persistence
    - _Requirements: 2.5, 3.2, 3.3, 9.1, 9.2, 9.3, 9.4_
  - [x] 5.2 Write property test for vector storage round-trip

    - **Property 7: Vector storage round-trip**
    - **Validates: Requirements 2.5**
  - [x] 5.3 Write property test for FAISS persistence

    - **Property 27: FAISS index persistence round-trip**
    - **Validates: Requirements 9.4**
  - [x] 5.4 Write property test for top-k retrieval

    - **Property 9: Top-k retrieval count**
    - **Validates: Requirements 3.3**
  - [x] 5.5 Write property test for score ordering

    - **Property 10: Retrieval score ordering**
    - **Validates: Requirements 3.4**

- [x] 6. Implement metadata storage
  - [x] 6.1 Create MetadataStore class using SQLite
    - Design schema: documents table, chunks table with foreign keys
    - Implement add_document() and add_chunk() methods
    - Implement get_chunk_metadata() and get_document_chunks()
    - Implement delete_document() with cascade deletion
    - Implement list_documents() to return all indexed documents
    - _Requirements: 2.3, 11.1, 11.2, 11.5_
  - [x] 6.2 Write property test for metadata-vector consistency

    - **Property 33: Metadata-vector consistency**
    - **Validates: Requirements 11.5**
  - [x] 6.3 Write property test for document listing

    - **Property 31: Document listing completeness**
    - **Validates: Requirements 11.2**
  - [x] 6.4 Write unit tests for CRUD operations

    - Test adding and retrieving documents
    - Test cascade deletion of chunks when document is deleted
    - Test querying chunks by document ID

- [x] 7. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Implement document processing
  - [x] 8.1 Create DocumentProcessor class for PDF and web content
    - Implement process_pdf() using PyPDF2 or pdfplumber
    - Implement process_url() using requests and BeautifulSoup
    - Implement extract_text() to handle both sources
    - Add error handling for invalid files and failed web requests
    - _Requirements: 1.1, 1.3, 7.1_
  - [x] 8.2 Write property test for text extraction

    - **Property 1: Text extraction completeness**
    - **Validates: Requirements 1.1**
  - [x] 8.3 Write property test for web content extraction

    - **Property 21: Web content extraction**
    - **Validates: Requirements 7.1**
  - [x] 8.4 Write unit tests for error handling

    - Test handling of corrupted PDFs
    - Test handling of invalid URLs
    - Test handling of non-HTML web content

- [ ] 9. Implement document indexing pipeline
  - [ ] 9.1 Create DocumentIndexer to coordinate chunking, embedding, and storage
    - Implement index_document() that orchestrates the full pipeline
    - Process document → chunk → embed → store in FAISS + metadata
    - Track document IDs and ensure independence
    - Handle multiple documents without interference
    - _Requirements: 1.5, 2.1, 2.2, 2.3, 2.4, 2.5_
  - [ ]* 9.2 Write property test for document independence
    - **Property 2: Document independence**
    - **Validates: Requirements 1.5**
  - [ ]* 9.3 Write property test for web content pipeline consistency
    - **Property 22: Web content pipeline consistency**
    - **Validates: Requirements 7.2**
  - [ ]* 9.4 Write property test for web metadata preservation
    - **Property 23: Web content metadata preservation**
    - **Validates: Requirements 7.4**
  - [ ]* 9.5 Write property test for URL processing independence
    - **Property 24: URL processing independence**
    - **Validates: Requirements 7.5**
  - [ ]* 9.6 Write integration test for end-to-end indexing
    - Test uploading a PDF and verifying chunks in FAISS
    - Test uploading web content and verifying metadata

- [ ] 10. Implement semantic retrieval
  - [ ] 10.1 Create SemanticRetriever class
    - Implement retrieve() to convert query to embedding and search FAISS
    - Implement retrieve_with_scores() to return chunks with similarity scores
    - Add similarity threshold filtering
    - Handle empty results gracefully
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_
  - [ ]* 10.2 Write property test for retrieval score ordering (already covered in 5.5)
  - [ ]* 10.3 Write unit tests for threshold filtering
    - Test that results below threshold are excluded
    - Test empty results when no chunks meet threshold

- [ ] 11. Implement reranking engine
  - [ ] 11.1 Create Reranker class using cross-encoder model
    - Initialize with cross-encoder model (ms-marco-MiniLM-L-6-v2)
    - Implement rerank() to score and reorder chunks
    - Implement score_pairs() for query-chunk relevance scoring
    - Select top-n chunks after reranking
    - Preserve all metadata through reranking
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_
  - [ ]* 11.2 Write property test for reranking score assignment
    - **Property 11: Reranking score assignment**
    - **Validates: Requirements 4.2**
  - [ ]* 11.3 Write property test for reranking score ordering
    - **Property 12: Reranking score ordering**
    - **Validates: Requirements 4.3**
  - [ ]* 11.4 Write property test for top-n selection
    - **Property 13: Top-n selection after reranking**
    - **Validates: Requirements 4.4**
  - [ ]* 11.5 Write property test for metadata preservation
    - **Property 14: Metadata preservation through reranking**
    - **Validates: Requirements 4.5**

- [ ] 12. Implement LLM generator with hallucination control
  - [ ] 12.1 Create LLMGenerator class supporting multiple backends
    - Implement generate_answer() with context-grounded prompts
    - Implement generate_with_citations() to include source references
    - Add prompt template with hallucination control instructions
    - Support OpenAI API integration
    - Add configuration for model selection
    - _Requirements: 5.1, 5.2, 6.1, 10.1, 10.2, 10.4_
  - [ ]* 12.2 Write property test for prompt construction
    - **Property 15: Prompt construction completeness**
    - **Validates: Requirements 5.1**
  - [ ]* 12.3 Write property test for grounding instructions
    - **Property 19: Grounding instruction presence**
    - **Validates: Requirements 6.1**
  - [ ]* 12.4 Write property test for model interface consistency
    - **Property 29: Model interface consistency**
    - **Validates: Requirements 10.4**
  - [ ]* 12.5 Write unit tests for OpenAI integration
    - Test successful generation with OpenAI
    - Test error handling for API failures

- [ ] 13. Add open-source LLM support
  - [ ] 13.1 Extend LLMGenerator to support Hugging Face models
    - Add support for loading local models (Llama, Mistral)
    - Implement model switching logic
    - Handle model-specific parameters
    - Ensure consistent interface across model types
    - _Requirements: 10.3, 10.5_
  - [ ]* 13.2 Write property test for model configuration
    - **Property 28: Model configuration acceptance**
    - **Validates: Requirements 10.1**
  - [ ]* 13.3 Write unit tests for open-source model integration
    - Test loading and running a small open-source model
    - Test parameter handling for different model types

- [ ] 14. Implement citation builder
  - [ ] 14.1 Create CitationBuilder class
    - Implement build_citations() to extract source references
    - Implement format_citation() with document name, page, excerpt
    - Map citations to original chunks
    - Validate that all citations are traceable
    - Handle multiple sources in answers
    - _Requirements: 5.2, 5.3, 5.5_
  - [ ]* 14.2 Write property test for citation presence
    - **Property 16: Citation presence in responses**
    - **Validates: Requirements 5.2**
  - [ ]* 14.3 Write property test for multiple source citations
    - **Property 18: Multiple source citation**
    - **Validates: Requirements 5.5**
  - [ ]* 14.4 Write unit tests for citation formatting
    - Test citation format with all required fields
    - Test citation extraction from generated text

- [ ] 15. Implement query orchestration
  - [ ] 15.1 Create QueryOrchestrator to coordinate retrieval → reranking → generation
    - Implement query() method that runs full pipeline
    - Coordinate SemanticRetriever, Reranker, LLMGenerator, CitationBuilder
    - Handle low confidence scenarios
    - Return QueryResponse with answer and citations
    - _Requirements: 3.1, 3.2, 4.1, 5.1, 6.3_
  - [ ]* 15.2 Write property test for low confidence indication
    - **Property 20: Low confidence indication**
    - **Validates: Requirements 6.3**
  - [ ]* 15.3 Write integration test for full query pipeline
    - Test end-to-end: index document → query → get answer with citations
    - Verify citations reference correct source documents

- [ ] 16. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 17. Implement FastAPI REST API with CORS
  - [ ] 17.1 Create FastAPI application with endpoints and CORS middleware
    - Implement POST /documents/upload for PDF upload
    - Implement POST /documents/url for web content indexing
    - Implement GET /documents to list all documents
    - Implement GET /documents/{doc_id} for document details
    - Implement DELETE /documents/{doc_id} to delete documents
    - Implement POST /query for asking questions
    - Implement GET /health for health checks
    - Add request validation using Pydantic models
    - Configure CORS to allow frontend origin
    - _Requirements: 8.1, 8.2, 8.4, 13.1_
  - [ ]* 17.2 Write property test for API input validation
    - **Property 25: API input validation**
    - **Validates: Requirements 8.2**
  - [ ]* 17.3 Write property test for API response format
    - **Property 26: API response format**
    - **Validates: Requirements 8.4**
  - [ ]* 17.4 Write unit tests for each endpoint
    - Test successful document upload
    - Test document listing
    - Test query endpoint with valid input
    - Test error responses for invalid inputs

- [ ] 18. Implement document management operations
  - [ ] 18.1 Add document deletion and update functionality
    - Implement delete operation that removes chunks from FAISS and metadata
    - Implement update operation that re-processes and re-indexes
    - Implement clear operation to reset entire index
    - Ensure consistency between FAISS and metadata store
    - _Requirements: 11.1, 11.3, 11.4, 11.5_
  - [ ]* 18.2 Write property test for deletion completeness
    - **Property 30: Document deletion completeness**
    - **Validates: Requirements 11.1**
  - [ ]* 18.3 Write property test for document update
    - **Property 32: Document update replacement**
    - **Validates: Requirements 11.3**
  - [ ]* 18.4 Write unit tests for clear operation
    - Test that clearing removes all documents and resets index

- [ ] 19. Implement comprehensive error handling and logging
  - [ ] 19.1 Add error handling throughout the application
    - Add try-catch blocks for all external operations
    - Implement graceful degradation (e.g., skip reranking on failure)
    - Add retry logic for LLM API calls (3 retries with backoff)
    - Add retry logic for web requests (2 retries)
    - Return user-friendly error messages
    - _Requirements: 12.1, 12.3_
  - [ ] 19.2 Set up logging infrastructure
    - Configure Python logging with appropriate levels
    - Log all errors with stack traces
    - Log key operations (document upload, queries) with timing
    - Add monitoring metrics collection
    - _Requirements: 12.2, 12.5_
  - [ ]* 19.3 Write property test for error logging
    - **Property 34: Error logging completeness**
    - **Validates: Requirements 12.1**
  - [ ]* 19.4 Write property test for operation logging
    - **Property 35: Operation logging**
    - **Validates: Requirements 12.2**
  - [ ]* 19.5 Write property test for exception handling
    - **Property 36: Exception handling gracefully**
    - **Validates: Requirements 12.3**
  - [ ]* 19.6 Write unit tests for retry logic
    - Test LLM API retry with mock failures
    - Test web request retry behavior

- [ ] 20. Add configuration and environment setup
  - [ ] 20.1 Create configuration system for backend and frontend
    - Create backend config.yaml with all configurable parameters
    - Add environment variable support for sensitive data (API keys)
    - Implement config loading and validation
    - Add defaults for all optional parameters
    - Create frontend .env.local with API URL
    - Document all configuration options
    - _Requirements: 10.1, 10.5_

- [ ] 21. Implement API key management system
  - [ ] 21.1 Create APIKeyManager class with encryption
    - Implement save_key() with Fernet encryption
    - Implement get_key() with decryption
    - Implement validate_key() to test API keys
    - Store keys securely in encrypted file or database
    - Add API endpoints: POST /settings/api-key, GET /settings/api-key, POST /settings/test-connection
    - _Requirements: 14.1, 14.2, 14.4, 14.5_
  - [ ]* 21.2 Write property test for API key validation
    - **Property 39: API key validation**
    - **Validates: Requirements 14.2**
  - [ ]* 21.3 Write property test for API key persistence
    - **Property 40: API key persistence**
    - **Validates: Requirements 14.4**
  - [ ]* 21.4 Write unit tests for encryption/decryption
    - Test key encryption and decryption round-trip
    - Test invalid key handling

- [ ] 22. Checkpoint - Ensure all backend tests pass
  - Ensure all backend tests pass, ask the user if questions arise.

- [ ] 23. Build Next.js frontend foundation
  - [ ] 23.1 Set up Next.js app structure with Tailwind CSS
    - Configure Tailwind CSS with custom theme
    - Create MainLayout component with navigation
    - Create Sidebar component with route links
    - Set up routing for pages: /, /query, /documents, /settings, /history
    - Create global state management with Zustand
    - Set up axios instance with base URL configuration
    - _Requirements: 13.1_
  - [ ]* 23.2 Write unit tests for layout components
    - Test MainLayout renders navigation correctly
    - Test Sidebar shows active route highlighting

- [ ] 24. Build query interface
  - [ ] 24.1 Create QueryPage and related components
    - Create QueryInput component with textarea and submit button
    - Create AnswerDisplay component with markdown rendering
    - Create CitationCard component to show sources
    - Implement query submission to backend API
    - Add loading states during query processing
    - Display answers with formatted citations
    - Add error handling and display
    - _Requirements: 13.3, 13.4, 13.5_
  - [ ]* 24.2 Write property test for answer display
    - **Property 38: Frontend displays answers with citations**
    - **Validates: Requirements 13.3**
  - [ ]* 24.3 Write unit tests for query components
    - Test QueryInput validates empty submissions
    - Test AnswerDisplay renders markdown correctly
    - Test CitationCard displays all citation fields

- [ ] 25. Build document management interface
  - [ ] 25.1 Create DocumentsPage and upload components
    - Create DocumentCard component to display document metadata
    - Create DocumentUploadZone with drag-and-drop using react-dropzone
    - Implement file upload to backend with progress tracking
    - Display list of all documents from API
    - Add delete functionality with confirmation dialog
    - Add search and filter controls
    - Show upload status and processing stages
    - _Requirements: 13.2, 15.1, 15.2, 15.3, 15.4, 15.5, 17.1, 17.2, 17.3, 17.4, 17.5_
  - [ ]* 25.2 Write property test for upload status display
    - **Property 37: Frontend displays upload status**
    - **Validates: Requirements 13.2**
  - [ ]* 25.3 Write property test for document list display
    - **Property 41: Document list completeness in UI**
    - **Validates: Requirements 15.1, 15.2**
  - [ ]* 25.4 Write unit tests for document components
    - Test DocumentCard renders metadata correctly
    - Test DocumentUploadZone accepts PDF files only
    - Test delete confirmation dialog appears

- [ ] 26. Build settings page for API key management
  - [ ] 26.1 Create SettingsPage and APIKeyForm component
    - Create APIKeyForm with secure input fields (masked)
    - Add provider selection dropdown (OpenAI / Open-source)
    - Add model parameter inputs (temperature, max_tokens)
    - Implement save API key functionality
    - Add test connection button to validate keys
    - Display success/error messages
    - Show security notice about key storage
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5_
  - [ ]* 26.2 Write unit tests for settings components
    - Test APIKeyForm validates input before submission
    - Test provider selection updates available models
    - Test masked input for API keys

- [ ] 27. Build query history feature
  - [ ] 27.1 Create HistoryPage with local storage integration
    - Implement query history storage in browser local storage
    - Create history list component with chronological display
    - Add expandable query cards showing full answers
    - Implement clear history functionality
    - Add click to view full query details
    - _Requirements: 16.1, 16.2, 16.3, 16.4, 16.5_
  - [ ]* 27.2 Write property test for history persistence
    - **Property 42: Query history persistence**
    - **Validates: Requirements 16.1, 16.2**
  - [ ]* 27.3 Write unit tests for history components
    - Test history saves to local storage on query
    - Test history displays chronologically
    - Test clear history removes all entries

- [ ] 28. Add progress indicators and real-time feedback
  - [ ] 28.1 Create ProgressIndicator component
    - Create multi-stage progress bar component
    - Show stages: extract → chunk → embed → index
    - Highlight current stage during processing
    - Display chunk count and statistics on completion
    - Add to document upload flow
    - _Requirements: 17.1, 17.2, 17.3, 17.4, 17.5_
  - [ ]* 28.2 Write unit tests for progress component
    - Test ProgressIndicator shows all stages
    - Test current stage highlighting

- [ ] 29. Add UI polish and error handling
  - [ ] 29.1 Create utility components and error boundaries
    - Create LoadingSpinner component
    - Create Toast notification system
    - Create Modal and ConfirmDialog components
    - Add ErrorBoundary to catch React errors
    - Implement consistent error message display
    - Add responsive design for mobile devices
    - _Requirements: 13.4, 13.5_
  - [ ]* 29.2 Write unit tests for utility components
    - Test Toast displays messages correctly
    - Test Modal opens and closes
    - Test ErrorBoundary catches errors

- [ ] 30. Create example scripts and documentation
  - [ ] 30.1 Write example usage scripts and comprehensive documentation
    - Create backend example_upload.py to demonstrate document upload
    - Create backend example_query.py to demonstrate querying
    - Create backend example_full_pipeline.py for end-to-end demo
    - Add comments explaining each step
    - Write README.md with installation steps for both backend and frontend
    - Document configuration options
    - Document API endpoints with examples
    - Add frontend usage guide with screenshots
    - Add troubleshooting section
    - Include learning resources for key concepts (embeddings, RAG, FAISS)
    - Create deployment guide for production

- [ ] 31. Final checkpoint - Ensure all tests pass
  - Ensure all backend and frontend tests pass, ask the user if questions arise.
  - Run full test suite with coverage report
  - Verify all 42 correctness properties are tested
  - Test with real PDFs and queries
  - Test full user flow: upload → query → view history → manage settings
