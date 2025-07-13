# 🚀 Deployment Guide

## Streamlit Cloud Deployment

### Prerequisites
1. GitHub account
2. Streamlit Cloud account (free at https://share.streamlit.io/)

### Steps

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to https://share.streamlit.io/
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set the main file path to: `app.py`
   - Click "Deploy"

### Important Notes

- **Model Training**: The app will automatically train models on first run
- **Memory Limit**: Models are optimized to stay under 100MB
- **Dependencies**: All required packages are in `requirements.txt`
- **Data**: The `train.csv` file is included in the repository

### Environment Variables (Optional)

You can set these in Streamlit Cloud if needed:
- `STREAMLIT_SERVER_PORT`: 8501
- `STREAMLIT_SERVER_ADDRESS`: 0.0.0.0

### Troubleshooting

1. **Models not loading**: The app will automatically train models on first run
2. **Memory issues**: The quick training uses a smaller dataset
3. **Dependency issues**: All packages are pinned to specific versions

### Local vs Cloud

- **Local**: Use `./run.sh` or `run.bat` for easy setup
- **Cloud**: Automatic deployment from GitHub
- **Port**: Local uses 8501, Cloud uses default Streamlit port 