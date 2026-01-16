@echo off
echo ========================================
echo AI Knowledge Copilot - Setup Script
echo ========================================
echo.

echo [1/4] Setting up backend...
cd backend

echo Creating Python virtual environment...
python -m venv venv
if errorlevel 1 (
    echo Error: Failed to create virtual environment
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing backend dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install backend dependencies
    exit /b 1
)

echo Creating .env file from example...
if not exist .env (
    copy .env.example .env
    echo Please edit backend\.env with your configuration
)

echo Creating data directory...
if not exist data mkdir data

cd ..

echo.
echo [2/4] Setting up frontend...
cd frontend

echo Installing frontend dependencies...
call npm install
if errorlevel 1 (
    echo Error: Failed to install frontend dependencies
    exit /b 1
)

echo Creating .env.local file from example...
if not exist .env.local (
    copy .env.local.example .env.local
)

cd ..

echo.
echo [3/4] Generating encryption key...
python -c "from cryptography.fernet import Fernet; print('ENCRYPTION_KEY=' + Fernet.generate_key().decode())" > encryption_key.txt
echo Encryption key saved to encryption_key.txt
echo Please add this to your backend\.env file

echo.
echo [4/4] Setup complete!
echo.
echo ========================================
echo Next Steps:
echo ========================================
echo 1. Edit backend\.env with your configuration
echo 2. Add the encryption key from encryption_key.txt to backend\.env
echo 3. Start backend: cd backend ^&^& venv\Scripts\activate ^&^& uvicorn src.main:app --reload
echo 4. Start frontend: cd frontend ^&^& npm run dev
echo 5. Open http://localhost:3000 in your browser
echo ========================================
