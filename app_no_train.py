import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
from fuzzywuzzy import fuzz
import re

# Set environment variable to avoid tokenizer warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Set page config
st.set_page_config(
    page_title="Quora Duplicate Question Detector",
    page_icon="🔍",
    layout="wide"
)

def preprocess_text(text):
    """Simple text preprocessing"""
    if pd.isna(text):
        return ""
    
    text = str(text).lower().strip()
    text = re.sub(r'\W', ' ', text).strip()
    return text

def calculate_similarity(q1, q2):
    """Calculate similarity between two questions"""
    q1_processed = preprocess_text(q1)
    q2_processed = preprocess_text(q2)
    
    # Fuzzy string matching
    fuzzy_ratio = fuzz.ratio(q1_processed, q2_processed)
    fuzzy_partial = fuzz.partial_ratio(q1_processed, q2_processed)
    fuzzy_token_sort = fuzz.token_sort_ratio(q1_processed, q2_processed)
    fuzzy_token_set = fuzz.token_set_ratio(q1_processed, q2_processed)
    
    # Word overlap
    q1_words = set(q1_processed.split())
    q2_words = set(q2_processed.split())
    
    if len(q1_words) == 0 or len(q2_words) == 0:
        word_overlap = 0.0
    else:
        intersection = len(q1_words & q2_words)
        union = len(q1_words | q2_words)
        word_overlap = intersection / union if union > 0 else 0.0
    
    # Length similarity
    len_diff = abs(len(q1_processed) - len(q2_processed))
    max_len = max(len(q1_processed), len(q2_processed))
    length_similarity = 1 - (len_diff / max_len) if max_len > 0 else 0.0
    
    # Combined similarity score
    similarity_score = (
        fuzzy_ratio * 0.3 +
        fuzzy_partial * 0.2 +
        fuzzy_token_sort * 0.2 +
        fuzzy_token_set * 0.2 +
        word_overlap * 100 * 0.1
    ) / 100
    
    return similarity_score, {
        'fuzzy_ratio': fuzzy_ratio,
        'fuzzy_partial': fuzzy_partial,
        'fuzzy_token_sort': fuzzy_token_sort,
        'fuzzy_token_set': fuzzy_token_set,
        'word_overlap': word_overlap,
        'length_similarity': length_similarity
    }

def predict_duplicate_simple(question1, question2):
    """Simple rule-based duplicate detection"""
    similarity_score, features = calculate_similarity(question1, question2)
    
    # Simple threshold-based prediction
    is_duplicate = similarity_score > 0.7
    
    return {
        'prediction': is_duplicate,
        'confidence': similarity_score,
        'features': features,
        'q1_processed': preprocess_text(question1),
        'q2_processed': preprocess_text(question2)
    }

def main():
    st.title("🔍 Quora Duplicate Question Detector")
    st.markdown("*Simple rule-based version - no training required*")
    
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
                result = predict_duplicate_simple(question1, question2)
            
            if result:
                # Display results
                st.header("📊 Analysis Results")
                
                # Main prediction
                main_pred = result['prediction']
                main_prob = result['confidence']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Prediction",
                        "DUPLICATE" if main_pred else "NOT DUPLICATE",
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
                        "Method",
                        "Rule-based"
                    )
                
                # Feature analysis
                st.subheader("🔧 Feature Analysis")
                
                features = result['features']
                feature_df = pd.DataFrame([
                    {'Feature': k.replace('_', ' ').title(), 'Value': v} 
                    for k, v in features.items()
                ])
                
                # Create feature visualization
                fig = px.bar(
                    feature_df,
                    x='Feature',
                    y='Value',
                    title="Similarity Features",
                    color='Value',
                    color_continuous_scale='RdYlGn'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature table
                st.dataframe(feature_df, use_container_width=True)
                
                # Explanation
                st.subheader("📝 How it works")
                st.markdown("""
                This simple rule-based approach uses:
                - **Fuzzy String Matching**: Compares text similarity
                - **Word Overlap**: Measures common words between questions
                - **Length Similarity**: Compares question lengths
                - **Combined Score**: Weighted average of all features
                
                **Threshold**: Questions with >70% similarity are considered duplicates.
                """)
        else:
            st.warning("Please enter both questions to analyze.")

if __name__ == "__main__":
    main() 