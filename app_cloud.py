import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Set environment variable to avoid tokenizer warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Set page config
st.set_page_config(
    page_title="Quora Duplicate Question Detector",
    page_icon="🔍",
    layout="wide"
)

def quick_train_models():
    """Quick training with smaller dataset for cloud deployment"""
    try:
        from data_processor import QuoraDataProcessor
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        
        st.info("🤖 Training models with smaller dataset for faster deployment...")
        
        # Load smaller dataset
        df = pd.read_csv('train.csv')
        df_sample = df.sample(5000, random_state=42)  # Much smaller sample
        
        # Initialize processor
        processor = QuoraDataProcessor()
        
        # Process data
        X, y, _ = processor.process_data(df_sample)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = processor.scaler.fit_transform(X_train)
        X_test_scaled = processor.scaler.transform(X_test)
        
        # Save test data
        np.savez('models/test_data.npz', X=X_test_scaled, y=y_test)
        
        # Train Random Forest (faster)
        st.text("Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=100,  # Reduced
            max_depth=10,       # Reduced
            random_state=42,
            n_jobs=1           # Single thread for cloud
        )
        rf.fit(X_train_scaled, y_train)
        
        # Train XGBoost (faster)
        st.text("Training XGBoost...")
        xgb = XGBClassifier(
            n_estimators=100,   # Reduced
            max_depth=6,        # Reduced
            learning_rate=0.1,
            random_state=42,
            n_jobs=1           # Single thread for cloud
        )
        xgb.fit(X_train_scaled, y_train)
        
        # Save models
        models = {
            'random_forest': rf,
            'xgboost': xgb
        }
        
        os.makedirs('models', exist_ok=True)
        for name, model in models.items():
            joblib.dump(model, f'models/{name}.pkl')
        
        joblib.dump(processor.scaler, 'models/scaler.pkl')
        joblib.dump(processor, 'models/processor.pkl')
        
        st.success("✅ Models trained successfully!")
        return True
        
    except Exception as e:
        st.error(f"❌ Error training models: {str(e)}")
        return False

def train_models_if_needed():
    """Train models if they don't exist"""
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Check if models exist
    required_files = ['random_forest.pkl', 'xgboost.pkl', 'scaler.pkl', 'processor.pkl']
    models_exist = all(os.path.exists(os.path.join(model_dir, f)) for f in required_files)
    
    if not models_exist:
        return quick_train_models()
    
    return True

@st.cache_resource
def load_models():
    """Load trained models and processor"""
    try:
        # First, ensure models are trained
        if not train_models_if_needed():
            return None, None, None
        
        models = {}
        model_dir = 'models'
        
        # Load models
        for file in os.listdir(model_dir):
            if file.endswith('.pkl') and file != 'scaler.pkl' and file != 'processor.pkl':
                name = file.replace('.pkl', '')
                models[name] = joblib.load(os.path.join(model_dir, file))
        
        # Load scaler and processor
        scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        processor = joblib.load(os.path.join(model_dir, 'processor.pkl'))
        
        return models, scaler, processor
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None, None

def predict_duplicate(question1, question2, models, scaler, processor):
    """Predict if two questions are duplicates"""
    try:
        # Preprocess questions
        q1_processed = processor.preprocess_text(question1)
        q2_processed = processor.preprocess_text(question2)
        
        # Get embeddings
        q1_embedding = processor.get_embeddings([q1_processed])
        q2_embedding = processor.get_embeddings([q2_processed])
        
        # Create a temporary dataframe for feature extraction
        temp_df = pd.DataFrame({
            'question1': [q1_processed],
            'question2': [q2_processed]
        })
        
        # Extract features
        temp_df = processor.extract_basic_features(temp_df)
        temp_df = processor.extract_word_features(temp_df)
        temp_df = processor.extract_token_features(temp_df)
        temp_df = processor.extract_length_features(temp_df)
        temp_df = processor.extract_fuzzy_features(temp_df)
        temp_df = processor.extract_advanced_features(temp_df)
        
        # Combine features
        feature_columns = ['cwc_min', 'cwc_max', 'csc_min', 'csc_max', 'ctc_min', 'ctc_max',
                          'last_word_eq', 'first_word_eq', 'abs_len_diff', 'mean_len',
                          'token_set_ratio', 'token_sort_ratio', 'fuzz_ratio', 
                          'fuzz_partial_ratio', 'longest_substr_ratio', 'word_share',
                          'tfidf_similarity', 'semantic_similarity', 'exact_word_match_ratio',
                          'partial_word_match_ratio', 'question_type_similarity']
        
        other_features = temp_df[feature_columns].values
        
        # Combine all features
        X = np.hstack((q1_embedding, q2_embedding, other_features))
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Get predictions from all models
        predictions = {}
        probabilities = {}
        
        for name, model in models.items():
            pred = model.predict(X_scaled)[0]
            prob = model.predict_proba(X_scaled)[0]
            
            predictions[name] = pred
            probabilities[name] = prob[1] if len(prob) > 1 else prob[0]
        
        return predictions, probabilities, {
            'q1_processed': q1_processed,
            'q2_processed': q2_processed,
            'features': temp_df.iloc[0].to_dict()
        }
        
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        return None, None, None

def main():
    st.title("🔍 Quora Duplicate Question Detector")
    st.markdown("*Cloud-optimized version with faster training*")
    
    # Load models
    with st.spinner("Loading models..."):
        models, scaler, processor = load_models()
    
    if models is None:
        st.stop()
    
    # Model selection
    selected_model = st.sidebar.selectbox(
        "Choose Model:",
        list(models.keys()),
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    st.sidebar.info(f"**Selected Model:** {selected_model.replace('_', ' ').title()}")
    
    # Main interface
    st.header("Single Question Pair Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        question1 = st.text_area(
            "Question 1:",
            placeholder="Enter your first question here...",
            height=150
        )
    
    with col2:
        question2 = st.text_area(
            "Question 2:",
            placeholder="Enter your second question here...",
            height=150
        )
    
    if st.button("🔍 Analyze Questions", type="primary"):
        if question1.strip() and question2.strip():
            with st.spinner("Analyzing questions..."):
                predictions, probabilities, details = predict_duplicate(
                    question1, question2, models, scaler, processor
                )
            
            if predictions:
                # Display results
                st.header("📊 Analysis Results")
                
                # Main prediction
                main_pred = predictions[selected_model]
                main_prob = probabilities[selected_model]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Prediction",
                        "DUPLICATE" if main_pred == 1 else "NOT DUPLICATE",
                        delta=f"{main_prob:.1%} confidence"
                    )
                
                with col2:
                    st.metric(
                        "Confidence",
                        f"{main_prob:.1%}",
                        delta="High" if main_prob > 0.8 else "Medium" if main_prob > 0.6 else "Low"
                    )
                
                with col3:
                    st.metric(
                        "Model Used",
                        selected_model.replace('_', ' ').title()
                    )
                
                # Model comparison
                st.subheader("📈 All Model Predictions")
                
                model_results = []
                for name, pred in predictions.items():
                    model_results.append({
                        'Model': name.replace('_', ' ').title(),
                        'Prediction': 'Duplicate' if pred == 1 else 'Not Duplicate',
                        'Confidence': f"{probabilities[name]:.1%}",
                        'Confidence_Value': probabilities[name]
                    })
                
                results_df = pd.DataFrame(model_results)
                
                # Create bar chart
                fig = px.bar(
                    results_df,
                    x='Model',
                    y='Confidence_Value',
                    color='Confidence_Value',
                    title="Model Confidence Comparison",
                    color_continuous_scale='RdYlGn'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Results table
                st.dataframe(results_df.drop('Confidence_Value', axis=1), use_container_width=True)
        else:
            st.warning("Please enter both questions to analyze.")

if __name__ == "__main__":
    main() 