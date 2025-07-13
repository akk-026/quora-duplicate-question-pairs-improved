@echo off
echo 🚀 Setting up Quora Duplicate Question Detector

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo ✅ Python is installed

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📚 Installing dependencies...
pip install -r requirements_simple.txt

REM Download NLTK data
echo 📥 Downloading NLTK data...
python -c "import nltk; nltk.download('stopwords')"

echo ✅ Setup completed successfully!
echo.
echo 🎯 To activate the virtual environment:
echo    venv\Scripts\activate.bat
echo.
echo 🚀 To run the app:
echo    streamlit run app.py --server.port 8501
echo.
echo 🧪 To test the system:
echo    python test_system.py
pause 