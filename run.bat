@echo off
echo 🚀 Starting Quora Duplicate Question Detector

REM Check if virtual environment exists
if not exist "venv" (
    echo ❌ Virtual environment not found. Please run setup.bat first.
    pause
    exit /b 1
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if models exist
if not exist "models" (
    echo ⚠️ Models not found. Running quick training...
    python quick_train.py
)

REM Start the app
echo 🌐 Starting Streamlit app...
streamlit run app.py --server.port 8501 