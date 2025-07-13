# 🚀 Deployment Guide

## Streamlit Cloud Deployment

### Prerequisites
1. GitHub repository with your code
2. Streamlit Cloud account

### Steps
1. **Push to GitHub**: Ensure all files are committed and pushed
2. **Connect to Streamlit Cloud**: 
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
3. **Configure App**:
   - **Main file path**: `app.py` (or `app_simple.py` for simplified version)
   - **Python version**: 3.9 or 3.10
4. **Deploy**: Click "Deploy app"

### Troubleshooting

#### ModuleNotFoundError
If you see `ModuleNotFoundError: No module named 'joblib'`:

1. **Check requirements.txt**: Ensure it exists and has all dependencies
2. **Verify file structure**:
   ```
   your-repo/
   ├── app.py
   ├── data_processor.py
   ├── requirements.txt
   ├── models/
   │   ├── random_forest.pkl
   │   ├── xgboost.pkl
   │   ├── scaler.pkl
   │   └── processor.pkl
   └── train.csv
   ```

3. **Alternative**: Use `app_simple.py` which has better error handling

#### Model Loading Issues
If models fail to load:

1. **Ensure models exist**: Run `python fast_train.py` locally first
2. **Check file sizes**: Models should be in `models/` directory
3. **Verify file permissions**: All files should be readable

#### Memory Issues
If app crashes due to memory:

1. **Use smaller models**: The current setup uses `all-MiniLM-L6-v2` (384 dimensions)
2. **Reduce batch size**: Process fewer samples at once
3. **Use simplified app**: `app_simple.py` has reduced memory usage

### Local Testing

Before deploying, test locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Train models
python fast_train.py

# Test app
streamlit run app.py --server.port 8501
```

### File Structure for Deployment

```
quora-duplicate/
├── app.py                 # Main Streamlit app
├── app_simple.py          # Simplified version (backup)
├── data_processor.py      # Data processing module
├── fast_train.py          # Training script
├── requirements.txt       # Dependencies
├── .streamlit/
│   └── config.toml       # Streamlit config
├── models/                # Trained models
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── scaler.pkl
│   └── processor.pkl
└── train.csv             # Dataset
```

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| ModuleNotFoundError | Check requirements.txt exists and has correct dependencies |
| Model loading fails | Ensure models/ directory exists with trained models |
| Memory errors | Use app_simple.py or reduce model complexity |
| Port conflicts | Change port in config.toml or use different port |

### Performance Tips

1. **Use caching**: The app uses `@st.cache_resource` for model loading
2. **Optimize imports**: Only import what's needed
3. **Reduce model size**: Current models are <100MB total
4. **Use simplified UI**: `app_simple.py` has fewer features but better performance 