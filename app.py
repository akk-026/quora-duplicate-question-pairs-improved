import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from data_processor import QuoraDataProcessor

# Page configuration
st.set_page_config(
    page_title="Quora Duplicate Question Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
    }
    .duplicate {
        background-color: #d4edda;
        border-color: #28a745;
    }
    .not-duplicate {
        background-color: #f8d7da;
        border-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load trained models and processor"""
    try:
        models = {}
        model_dir = 'models'
        
        if not os.path.exists(model_dir):
            st.error("❌ Models not found! Please run the training script first.")
            return None, None
        
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
    # Header
    st.markdown('<h1 class="main-header">🔍 Quora Duplicate Question Detector</h1>', unsafe_allow_html=True)
    
    # Load models
    with st.spinner("Loading models..."):
        models, scaler, processor = load_models()
    
    if models is None:
        st.stop()
    
    # Sidebar
    st.sidebar.markdown("## 🎯 Model Selection")
    selected_model = st.sidebar.selectbox(
        "Choose Model:",
        list(models.keys()),
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    st.sidebar.markdown("## 📊 Model Info")
    st.sidebar.info(f"**Selected Model:** {selected_model.replace('_', ' ').title()}")
    st.sidebar.info(f"**Available Models:** {len(models)}")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📊 Batch Analysis", "📈 Model Comparison"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">Single Question Pair Analysis</h2>', unsafe_allow_html=True)
        
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
                    st.markdown("## 📊 Analysis Results")
                    
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
                    
                    # Prediction box
                    prediction_class = "duplicate" if main_pred == 1 else "not-duplicate"
                    st.markdown(f"""
                    <div class="prediction-box {prediction_class}">
                        <h3>🎯 Final Prediction</h3>
                        <p><strong>Result:</strong> {'DUPLICATE' if main_pred == 1 else 'NOT DUPLICATE'}</p>
                        <p><strong>Confidence:</strong> {main_prob:.1%}</p>
                        <p><strong>Model:</strong> {selected_model.replace('_', ' ').title()}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Model comparison
                    st.markdown("### 📈 All Model Predictions")
                    
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
                    
                    # Feature analysis
                    if st.checkbox("🔍 Show Feature Analysis"):
                        st.markdown("### 🔧 Feature Analysis")
                        
                        features = details['features']
                        feature_df = pd.DataFrame([
                            {'Feature': k, 'Value': v} 
                            for k, v in features.items()
                        ])
                        
                        # Create feature importance visualization
                        fig = px.bar(
                            feature_df,
                            x='Feature',
                            y='Value',
                            title="Extracted Features",
                            color='Value',
                            color_continuous_scale='Blues'
                        )
                        fig.update_layout(height=400, xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.dataframe(feature_df, use_container_width=True)
                
            else:
                st.warning("⚠️ Please enter both questions to analyze.")
    
    with tab2:
        st.markdown('<h2 class="sub-header">Batch Analysis</h2>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload CSV file with 'question1' and 'question2' columns:",
            type=['csv']
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                if 'question1' in df.columns and 'question2' in df.columns:
                    st.success(f"✅ Loaded {len(df)} question pairs")
                    
                    if st.button("🔍 Analyze All Questions"):
                        progress_bar = st.progress(0)
                        results = []
                        
                        for idx, row in df.iterrows():
                            predictions, probabilities, _ = predict_duplicate(
                                row['question1'], row['question2'], models, scaler, processor
                            )
                            
                            if predictions:
                                results.append({
                                    'Question 1': row['question1'][:100] + "..." if len(row['question1']) > 100 else row['question1'],
                                    'Question 2': row['question2'][:100] + "..." if len(row['question2']) > 100 else row['question2'],
                                    'Prediction': 'Duplicate' if predictions[selected_model] == 1 else 'Not Duplicate',
                                    'Confidence': f"{probabilities[selected_model]:.1%}"
                                })
                            
                            progress_bar.progress((idx + 1) / len(df))
                        
                        results_df = pd.DataFrame(results)
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name="duplicate_analysis_results.csv",
                            mime="text/csv"
                        )
                
                else:
                    st.error("❌ CSV must contain 'question1' and 'question2' columns")
            
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    
    with tab3:
        st.markdown('<h2 class="sub-header">Model Performance Comparison</h2>', unsafe_allow_html=True)
        
        # Model comparison metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Models", len(models))
        
        with col2:
            st.metric("Best Model", selected_model.replace('_', ' ').title())
        
        with col3:
            st.metric("Model Type", "Ensemble")
        
        # Model comparison chart
        if models:
            model_names = list(models.keys())
            model_names_formatted = [name.replace('_', ' ').title() for name in model_names]

            # Dynamically compute metrics for each model
            accuracy_scores = []
            precision_scores = []
            recall_scores = []
            from sklearn.metrics import accuracy_score, precision_score, recall_score
            import joblib
            import numpy as np
            import os
            # Load test data and labels if available
            test_data_path = os.path.join('models', 'test_data.npz')
            if os.path.exists(test_data_path):
                arr = np.load(test_data_path)
                X_test_scaled = arr['X']
                y_test = arr['y']
                for name in model_names:
                    model = models[name]
                    y_pred = model.predict(X_test_scaled)
                    accuracy_scores.append(accuracy_score(y_test, y_pred))
                    precision_scores.append(precision_score(y_test, y_pred))
                    recall_scores.append(recall_score(y_test, y_pred))
            else:
                # Fallback: show N/A if no test data
                accuracy_scores = [None] * len(model_names)
                precision_scores = [None] * len(model_names)
                recall_scores = [None] * len(model_names)

            metrics_df = pd.DataFrame({
                'Model': model_names_formatted,
                'Accuracy': accuracy_scores,
                'Precision': precision_scores,
                'Recall': recall_scores
            })
            
            # Create comparison chart
            fig = go.Figure()
            
            for metric in ['Accuracy', 'Precision', 'Recall']:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=metrics_df['Model'],
                    y=metrics_df[metric],
                    text=[f'{val:.3f}' for val in metrics_df[metric]],
                    textposition='auto'
                ))
            
            fig.update_layout(
                title="Model Performance Comparison",
                barmode='group',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(metrics_df, use_container_width=True)

if __name__ == "__main__":
    main() 