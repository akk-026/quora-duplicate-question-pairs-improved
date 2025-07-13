import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Quora Duplicate Question Detector",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def load_models():
    """Load trained models and processor"""
    try:
        models = {}
        model_dir = 'models'
        
        if not os.path.exists(model_dir):
            st.error("❌ Models not found! Please run the training script first.")
            return None, None, None
        
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