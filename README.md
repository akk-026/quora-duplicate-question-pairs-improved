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

## 🌐 Streamlit Cloud Deployment

Choose the appropriate version based on your needs:

### Option 1: No Training (Recommended for Cloud)
```bash
streamlit run app_no_train.py --server.port 8501
```
- **No training required** - uses rule-based approach
- **Instant deployment** - no timeouts
- **Simple but effective** - fuzzy string matching + word overlap

### Option 2: Ultra-Fast Training
```bash
streamlit run app_ultra_fast.py --server.port 8501
```
- **Minimal training** - 1,000 samples, simple models
- **Fast deployment** - ~30 seconds training time
- **Lightweight models** - 50 estimators, shallow trees

### Option 3: Cloud-Optimized Training
```bash
streamlit run app_cloud.py --server.port 8501
```
- **Reduced training** - 5,000 samples, optimized models
- **Moderate speed** - ~2-3 minutes training time
- **Balanced performance** - 100 estimators, moderate depth

### Option 4: Full Features (Local Only)
```bash
streamlit run app.py --server.port 8501
```
- **Full training** - 15,000 samples, complex models
- **Best performance** - 85%+ accuracy
- **Local deployment only** - may timeout on cloud

## 📊 Performance Comparison

| Version | Training Time | Accuracy | Cloud Compatible |
|---------|---------------|----------|------------------|
| `app_no_train.py` | 0 seconds | ~75% | ✅ Yes |
| `app_ultra_fast.py` | ~30 seconds | ~78% | ✅ Yes |
| `app_cloud.py` | ~2-3 minutes | ~80% | ✅ Yes |
| `app.py` | ~5-10 minutes | ~85% | ❌ No |

## 🔧 Features

- **Single Prediction**: Analyze individual question pairs
- **Batch Analysis**: Process multiple questions at once
- **Model Comparison**: Compare predictions across models
- **Advanced NLP**: Sentence transformers for semantic understanding
- **Multiple Models**: Random Forest and XGBoost ensemble

## 📁 Project Structure

```
quora-duplicate/
├── app.py                 # Full-featured Streamlit app (local only)
├── app_cloud.py           # Cloud-optimized version
├── app_ultra_fast.py      # Ultra-fast training version
├── app_no_train.py        # No-training rule-based version
├── app_simple.py          # Simplified version
├── data_processor.py      # Data processing and feature extraction
├── fast_train.py          # Model training script
├── requirements.txt       # Python dependencies
├── .streamlit/
│   └── config.toml       # Streamlit configuration
├── models/                # Trained models
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── scaler.pkl
│   └── processor.pkl
└── train.csv             # Dataset
```

## 🌐 Deployment

### Streamlit Cloud (Recommended)
1. Push to GitHub
2. Connect to Streamlit Cloud
3. Set main file to `app_no_train.py` (instant deployment)
4. Deploy!

### Local Development
```bash
pip install -r requirements.txt
python fast_train.py
streamlit run app.py --server.port 8501
```

## 📈 Model Details

- **Sentence Embeddings**: 384-dimensional vectors from `all-MiniLM-L6-v2`
- **Advanced Features**: TF-IDF similarity, semantic similarity, word overlap
- **Ensemble Models**: Random Forest + XGBoost with hyperparameter tuning
- **Rule-based**: Fuzzy string matching + word overlap (no training)

---

🌐 Live Demo

The ultra-fast version of the Quora Duplicate Question Detector is deployed and available for live use at:

🔗 Live App: quora-duplicate-question-pairs-improved

This deployment uses app_ultrafast.py, optimized for real-time predictions with minimal latency. The model is preloaded with efficient caching to ensure smooth user experience even with multiple requests.

**Made with ❤️ for the ML community** 