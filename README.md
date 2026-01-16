# AI Knowledge Copilot

A production-grade Retrieval-Augmented Generation (RAG) system that enables users to query PDFs, documents, and web content with high accuracy. The system implements semantic search, reranking, citation-based answers, and hallucination control to provide reliable, verifiable responses.

## Features

- 📄 **Document Processing**: Upload and index PDF documents and web content
- 🔍 **Semantic Search**: Find relevant information using embeddings and vector similarity
- 🎯 **Reranking**: Improve result relevance with cross-encoder models
- 📚 **Citations**: Get answers with source references for verification
- 🛡️ **Hallucination Control**: Responses grounded only in your documents
- 🤖 **Flexible LLM Support**: Use OpenAI or open-source models (Llama, Mistral)
- 🎨 **Modern UI**: Clean, responsive Next.js interface with Tailwind CSS

## Architecture

This is a monorepo containing:
- **backend/**: FastAPI server with RAG pipeline (Python)
- **frontend/**: Next.js web application (TypeScript + React)

## Quick Start

### Backend Setup

1. Navigate to backend directory:
```bash
cd backend
```

2. Create and activate virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
copy .env.example .env
# Edit .env with your settings
```

5. Run the server:
```bash
uvicorn src.main:app --reload
```

Backend will be available at http://localhost:8000

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Configure environment:
```bash
copy .env.local.example .env.local
# Edit .env.local if needed
```

4. Run the development server:
```bash
npm run dev
```

Frontend will be available at http://localhost:3000

## Technology Stack

### Backend
- **FastAPI**: Modern Python web framework
- **LangChain**: LLM application framework
- **FAISS**: Vector similarity search
- **Sentence Transformers**: Text embeddings
- **PyPDF2/pdfplumber**: PDF processing
- **BeautifulSoup4**: Web scraping
- **pytest + hypothesis**: Testing

### Frontend
- **Next.js 14**: React framework
- **TypeScript**: Type safety
- **Tailwind CSS**: Styling
- **React Query**: API state management
- **Zustand**: Global state
- **React Dropzone**: File uploads

## Project Structure

```
.
├── backend/
│   ├── src/
│   │   ├── __init__.py
│   │   ├── main.py          # FastAPI application
│   │   └── config.py        # Configuration management
│   ├── tests/               # Backend tests
│   ├── data/                # FAISS index and metadata storage
│   ├── requirements.txt     # Python dependencies
│   ├── config.yaml          # Application configuration
│   └── .env.example         # Environment variables template
│
├── frontend/
│   ├── app/
│   │   ├── layout.tsx       # Root layout
│   │   ├── page.tsx         # Home page
│   │   └── globals.css      # Global styles
│   ├── components/          # React components
│   ├── package.json         # Node dependencies
│   ├── tsconfig.json        # TypeScript config
│   ├── tailwind.config.ts   # Tailwind config
│   └── .env.local.example   # Environment variables template
│
└── .kiro/
    └── specs/               # Feature specifications
```

## Configuration

### Backend Configuration (config.yaml)

- **Embedding model**: Sentence transformer model for embeddings
- **Chunking**: Chunk size and overlap settings
- **Retrieval**: Top-k results and similarity threshold
- **Reranking**: Cross-encoder model and top-n selection
- **LLM**: Provider, temperature, and token limits
- **Storage**: Paths for FAISS index and metadata

### Environment Variables

Backend (.env):
- `OPENAI_API_KEY`: OpenAI API key (optional, can be set via UI)
- `ENCRYPTION_KEY`: Key for encrypting stored API keys
- `CORS_ORIGINS`: Allowed frontend origins

Frontend (.env.local):
- `NEXT_PUBLIC_API_URL`: Backend API URL
- `NEXT_PUBLIC_APP_NAME`: Application name

## Testing

### Backend Tests
```bash
cd backend
pytest                    # Run all tests
pytest --cov=src tests/  # Run with coverage
pytest -v tests/         # Verbose output
```

### Frontend Tests
```bash
cd frontend
npm test                 # Run all tests
npm run test:watch      # Watch mode
```

## Development

### Backend Development
- API documentation available at http://localhost:8000/docs
- Health check endpoint: http://localhost:8000/health
- Hot reload enabled with `--reload` flag

### Frontend Development
- Hot reload enabled by default
- TypeScript type checking
- ESLint for code quality

## API Endpoints

- `POST /documents/upload` - Upload PDF document
- `POST /documents/url` - Index web content
- `GET /documents` - List all documents
- `GET /documents/{doc_id}` - Get document details
- `DELETE /documents/{doc_id}` - Delete document
- `POST /query` - Ask a question
- `GET /health` - Health check
- `POST /settings/api-key` - Save API key
- `GET /settings/api-key` - Get current provider
- `POST /settings/test-connection` - Test LLM connection

## 📚 About This Project

This is an educational project developed as part of college coursework to learn about:
- Retrieval-Augmented Generation (RAG) systems
- Vector databases and semantic search
- Modern web development with FastAPI and Next.js
- LLM integration and prompt engineering
- Production-grade software architecture

**Note for Students:** This code is shared for learning and reference purposes. If you're working on a similar assignment, please use this to understand concepts and approaches, but develop your own implementation. Direct copying violates academic integrity policies.

**Note for Recruiters/Employers:** This project demonstrates my understanding of AI/ML systems, full-stack development, and software engineering best practices.

## Contributing

This project follows a spec-driven development approach. See `.kiro/specs/` for detailed requirements, design, and implementation tasks.

All code and documentation © 2025. All rights reserved.
