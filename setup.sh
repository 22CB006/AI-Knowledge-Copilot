#!/bin/bash

echo "========================================"
echo "AI Knowledge Copilot - Setup Script"
echo "========================================"
echo ""

echo "[1/4] Setting up backend..."
cd backend

echo "Creating Python virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "Error: Failed to create virtual environment"
    exit 1
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing backend dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install backend dependencies"
    exit 1
fi

echo "Creating .env file from example..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Please edit backend/.env with your configuration"
fi

echo "Creating data directory..."
mkdir -p data

cd ..

echo ""
echo "[2/4] Setting up frontend..."
cd frontend

echo "Installing frontend dependencies..."
npm install
if [ $? -ne 0 ]; then
    echo "Error: Failed to install frontend dependencies"
    exit 1
fi

echo "Creating .env.local file from example..."
if [ ! -f .env.local ]; then
    cp .env.local.example .env.local
fi

cd ..

echo ""
echo "[3/4] Generating encryption key..."
python3 -c "from cryptography.fernet import Fernet; print('ENCRYPTION_KEY=' + Fernet.generate_key().decode())" > encryption_key.txt
echo "Encryption key saved to encryption_key.txt"
echo "Please add this to your backend/.env file"

echo ""
echo "[4/4] Setup complete!"
echo ""
echo "========================================"
echo "Next Steps:"
echo "========================================"
echo "1. Edit backend/.env with your configuration"
echo "2. Add the encryption key from encryption_key.txt to backend/.env"
echo "3. Start backend: cd backend && source venv/bin/activate && uvicorn src.main:app --reload"
echo "4. Start frontend: cd frontend && npm run dev"
echo "5. Open http://localhost:3000 in your browser"
echo "========================================"
