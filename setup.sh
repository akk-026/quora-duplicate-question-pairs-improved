#!/bin/bash

echo "🚀 Setting up Quora Duplicate Question Detector"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✅ Python version: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing dependencies..."
pip install -r requirements_simple.txt

# Download NLTK data
echo "📥 Downloading NLTK data..."
python3 -c "import nltk; nltk.download('stopwords')"

echo "✅ Setup completed successfully!"
echo ""
echo "🎯 To activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "🚀 To run the app:"
echo "   streamlit run app.py --server.port 8501"
echo ""
echo "🧪 To test the system:"
echo "   python test_system.py" 