# Setup Guide - AI Knowledge Copilot

This guide will help you set up the AI Knowledge Copilot development environment.

## Prerequisites

### Required Software
- **Python 3.8+** (Python 3.13.5 recommended)
- **Node.js 18+** and npm
- **Git** (for version control)

### Optional
- **CUDA** (for GPU acceleration with FAISS and embeddings)
- **OpenAI API Key** (for using OpenAI models)

## Quick Setup (Automated)

### Windows
```bash
setup.bat
```

### Linux/Mac
```bash
chmod +x setup.sh
./setup.sh
```

The setup script will:
1. Create Python virtual environment
2. Install backend dependencies
3. Install frontend dependencies
4. Generate encryption key
5. Create configuration files

## Manual Setup

### 1. Backend Setup

#### Create Virtual Environment
```bash
cd backend
python -m venv venv
```

#### Activate Virtual Environment
**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

#### Install Dependencies
```bash
pip install -r requirements.txt
```

#### Configure Environment
```bash
# Windows
copy .env.example .env

# Linux/Mac
cp .env.example .env
```

#### Generate Encryption Key
```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Add the generated key to `backend/.env`:
```
ENCRYPTION_KEY=your-generated-key-here
```

#### Create Data Directory
```bash
mkdir data
```

### 2. Frontend Setup

#### Install Dependencies
```bash
cd frontend
npm install
```

#### Configure Environment
```bash
# Windows
copy .env.local.example .env.local

# Linux/Mac
cp .env.local.example .env.local
```

Edit `frontend/.env.local` if your backend runs on a different port:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Configuration

### Backend Configuration (backend/config.yaml)

```yaml
embedding:
  model: all-MiniLM-L6-v2  # Sentence transformer model
  dimension: 384

chunking:
  chunk_size: 512          # Tokens per chunk
  overlap: 50              # Overlap between chunks

retrieval:
  top_k: 10                # Number of chunks to retrieve
  similarity_threshold: 0.5

reranking:
  model: cross-encoder/ms-marco-MiniLM-L-6-v2
  top_n: 5                 # Chunks after reranking

llm:
  provider: openai         # or huggingface
  temperature: 0.7
  max_tokens: 500

storage:
  faiss_index_path: ./data/faiss_index
  metadata_db_path: ./data/metadata.db
```

### Environment Variables

#### Backend (.env)
```bash
# OpenAI API Key (optional - can be set via UI)
OPENAI_API_KEY=sk-your-key-here

# Encryption key for API key storage
ENCRYPTION_KEY=your-encryption-key-here

# CORS Origins (comma-separated)
CORS_ORIGINS=http://localhost:3000

# Environment
ENVIRONMENT=development
```

#### Frontend (.env.local)
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_APP_NAME=AI Knowledge Copilot
```

## Running the Application

### Start Backend Server

```bash
cd backend
# Activate virtual environment first
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

uvicorn src.main:app --reload
```

Backend will be available at:
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### Start Frontend Server

```bash
cd frontend

```

Frontend will be available at:
- Web UI: http://localhost:3000

## Verification

### Verify Setup
```bash
python verify_setup.py
```

This will check that all directories and files are in place.

### Test Backend
```bash
cd backend
pytest
```

### Test Frontend
```bash
cd frontend
npm test
```

## Troubleshooting

### Backend Issues

**Issue: ModuleNotFoundError**
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt` again

**Issue: FAISS installation fails**
- Try `pip install faiss-cpu` separately
- For GPU support: `pip install faiss-gpu`

**Issue: Torch installation is slow**
- Torch is large (~2GB). Be patient or use a faster mirror
- For CPU-only: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

**Issue: Config file not found**
- Ensure you're running from the backend directory
- Check that `config.yaml` exists

### Frontend Issues

**Issue: npm install fails**
- Clear npm cache: `npm cache clean --force`
- Delete `node_modules` and `package-lock.json`, then retry

**Issue: Port 3000 already in use**
- Change port: `npm run dev -- -p 3001`
- Or kill the process using port 3000

**Issue: TypeScript errors**
- Run `npm install` to ensure all types are installed
- Check `tsconfig.json` is present

### General Issues

**Issue: CORS errors**
- Ensure `CORS_ORIGINS` in backend `.env` includes your frontend URL
- Check that both servers are running

**Issue: API connection fails**
- Verify backend is running on port 8000
- Check `NEXT_PUBLIC_API_URL` in frontend `.env.local`
- Test backend health: `curl http://localhost:8000/health`

## Development Workflow

1. **Start Backend**: Terminal 1
   ```bash
   cd backend
   venv\Scripts\activate  # or source venv/bin/activate
   uvicorn src.main:app --reload
   ```

2. **Start Frontend**: Terminal 2
   ```bash
   cd frontend
   npm run dev
   ```

3. **Run Tests**: Terminal 3
   ```bash
   # Backend tests
   cd backend
   pytest

   # Frontend tests
   cd frontend
   npm test
   ```

## Next Steps

After setup is complete:

1. **Configure API Keys**: Visit http://localhost:3000/settings (once implemented)
2. **Upload Documents**: Test document upload functionality
3. **Run Queries**: Try asking questions about your documents
4. **Review Logs**: Check backend console for processing details

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Next.js Documentation](https://nextjs.org/docs)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
- [LangChain Documentation](https://python.langchain.com/)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the main README.md
3. Check the specification documents in `.kiro/specs/`
