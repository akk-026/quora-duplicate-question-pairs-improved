# 🔍 Quora Duplicate Question Detector

A machine learning system that detects duplicate questions using transformers and ensemble models. Built with Streamlit for easy deployment.

## 🚀 Quick Start

### 1. Setup
```bash
git clone <your-repo-url>
cd quora-duplicate
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate.bat
pip install -r requirements.txt
```

### 2. Train Models
```bash
python fast_train.py
```

### 3. Run App
```bash
streamlit run app.py --server.port 8501
```

Visit `http://localhost:8501` to use the app.

## 📊 Performance

- **Accuracy**: 85.07% (XGBoost), 81.40% (Random Forest)
- **Model Size**: < 100MB (Streamlit Cloud compatible)
- **Features**: 789 features including sentence embeddings, TF-IDF, and semantic similarity

## 🔧 Features

- **Single Prediction**: Analyze individual question pairs
- **Batch Analysis**: Process multiple questions at once
- **Model Comparison**: Compare predictions across models
- **Advanced NLP**: Sentence transformers for semantic understanding
- **Multiple Models**: Random Forest and XGBoost ensemble

## 📁 Project Structure

```
quora-duplicate/
├── app.py                 # Streamlit frontend
├── data_processor.py      # Data processing and feature extraction
├── fast_train.py          # Model training script
├── requirements.txt       # Python dependencies
├── train.csv             # Quora dataset
└── models/               # Trained models
    ├── random_forest.pkl
    ├── xgboost.pkl
    ├── scaler.pkl
    └── processor.pkl
```

## 🌐 Deployment

### Streamlit Cloud
1. Push to GitHub
2. Connect to Streamlit Cloud
3. Set main file to `app.py`
4. Deploy!

### Local
```bash
pip install -r requirements.txt
python fast_train.py
streamlit run app.py --server.port 8501
```

## 📈 Model Details

- **Sentence Embeddings**: 384-dimensional vectors from `all-MiniLM-L6-v2`
- **Advanced Features**: TF-IDF similarity, semantic similarity, word overlap
- **Ensemble Models**: Random Forest + XGBoost with hyperparameter tuning

---

**Made with ❤️ for the ML community** 