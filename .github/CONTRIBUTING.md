# Contributing to AI Knowledge Copilot

Thank you for your interest in this educational project!

## For Students

This is a college project created for learning purposes. If you're working on a similar assignment:
- ✅ Use this as a reference to understand concepts
- ✅ Learn from the architecture and implementation
- ❌ Don't copy code directly for your assignments
- ❌ Respect academic integrity policies

## Setup for Development

1. **Clone the repository:**
   ```bash
   git clone https://github.com/22CB006/AI-Knowledge-Copilot.git
   cd AI-Knowledge-Copilot
   ```

2. **Follow setup instructions:**
   - See [SETUP.md](../SETUP.md) for detailed setup guide
   - Run `setup.bat` (Windows) or `./setup.sh` (Linux/Mac)

3. **Configure environment:**
   - Copy `.env.example` to `.env` in backend
   - Copy `.env.local.example` to `.env.local` in frontend
   - Add your API keys and configuration

## What Gets Committed

### ✅ DO Commit:
- Source code (`*.py`, `*.ts`, `*.tsx`)
- Configuration templates (`.env.example`, `.env.local.example`)
- Documentation (`README.md`, `SETUP.md`)
- Package files (`requirements.txt`, `package.json`)
- Tests (`test_*.py`, `*.test.ts`)
- Setup scripts (`setup.bat`, `setup.sh`)

### ❌ DON'T Commit:
- Environment files (`.env`, `.env.local`)
- Virtual environments (`venv/`, `node_modules/`)
- Generated data (`data/`, `*.db`, `*.faiss`)
- IDE settings (`.vscode/`, `.idea/`)
- API keys or secrets
- Build artifacts (`.next/`, `dist/`, `build/`)
- Log files (`*.log`)

## Project Structure

```
AI-Knowledge-Copilot/
├── backend/              # Python FastAPI backend
│   ├── src/             # Source code
│   ├── tests/           # Tests
│   ├── data/            # Generated data (not committed)
│   └── venv/            # Virtual environment (not committed)
├── frontend/            # Next.js frontend
│   ├── app/             # Next.js app directory
│   ├── components/      # React components
│   └── node_modules/    # Dependencies (not committed)
└── .kiro/specs/         # Project specifications
```

## Questions?

For questions about this project, please open an issue on GitHub.
