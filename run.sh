#!/bin/bash

echo "🚀 Starting Quora Duplicate Question Detector"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if models exist
if [ ! -d "models" ]; then
    echo "⚠️ Models not found. Running quick training..."
    python quick_train.py
fi

# Start the app
echo "🌐 Starting Streamlit app..."
streamlit run app.py --server.port 8501 