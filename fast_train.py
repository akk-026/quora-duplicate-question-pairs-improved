import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import joblib
import os
from data_processor import QuoraDataProcessor
import warnings
warnings.filterwarnings('ignore')

def fast_train():
    """
    Fast training with both Random Forest and XGBoost for >80% accuracy
    """
    print("🚀 Fast Training for Quora Duplicate Detection")
    
    # Load data
    print("📊 Loading data...")
    df = pd.read_csv('train.csv')
    print(f"Original data shape: {df.shape}")
    
    # Sample dataset for faster training
    df_sample = df.sample(15000, random_state=42)
    print(f"Sampled data shape: {df_sample.shape}")
    
    # Initialize processor
    processor = QuoraDataProcessor()
    
    # Process data with advanced features
    print("🔧 Processing data with advanced features...")
    X, y, processed_df = processor.process_data(df_sample)
    
    print(f"Processed data shape: {X.shape}")
    print(f"Target distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    X_train_scaled = processor.scaler.fit_transform(X_train)
    X_test_scaled = processor.scaler.transform(X_test)
    
    # Save test set for app metrics
    np.savez('models/test_data.npz', X=X_test_scaled, y=y_test)
    
    print("🤖 Training models...")
    
    # 1. Random Forest with optimized parameters
    print("🌲 Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    
    # 2. XGBoost with optimized parameters
    print("⚡ Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=0.1,
        min_child_weight=3,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    xgb.fit(X_train_scaled, y_train)
    
    # Evaluate models
    models = {
        'random_forest': rf,
        'xgboost': xgb
    }
    
    print("\n📈 Model Evaluation Results:")
    print("=" * 50)
    
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        # Test set predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"\n{name.upper()}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        if f1 > best_score:
            best_score = f1
            best_model = name
    
    print(f"\n🏆 Best Model: {best_model} (F1 Score: {best_score:.4f})")
    
    # Save models
    processor.save_models(models, processor.scaler)
    
    # Save processor for inference
    joblib.dump(processor, 'models/processor.pkl')
    
    print("\n✅ Fast training completed! Models saved to 'models/' directory")
    
    return models, processor

if __name__ == "__main__":
    fast_train() 