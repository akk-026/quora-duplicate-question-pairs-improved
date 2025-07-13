import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
import distance
import joblib
import os

# Set environment variable to avoid tokenizer warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class QuoraDataProcessor:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the data processor with a sentence transformer model
        model_name: Pre-trained model to use for embeddings
        """
        self.model = SentenceTransformer(model_name)
        self.scaler = MinMaxScaler()
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        
        # Contractions dictionary
        self.contractions = {
            "ain't": "am not", "aren't": "are not", "can't": "can not", "can't've": "can not have",
            "'cause": "because", "could've": "could have", "couldn't": "could not",
            "couldn't've": "could not have", "didn't": "did not", "doesn't": "does not",
            "don't": "do not", "hadn't": "had not", "hadn't've": "had not have",
            "hasn't": "has not", "haven't": "have not", "he'd": "he would",
            "he'd've": "he would have", "he'll": "he will", "he'll've": "he will have",
            "he's": "he is", "how'd": "how did", "how'd'y": "how do you",
            "how'll": "how will", "how's": "how is", "i'd": "i would",
            "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have",
            "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
            "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have",
            "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not",
            "might've": "might have", "mightn't": "might not", "mightn't've": "might not have",
            "must've": "must have", "mustn't": "must not", "mustn't've": "must not have",
            "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock",
            "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
            "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would",
            "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have",
            "she's": "she is", "should've": "should have", "shouldn't": "should not",
            "shouldn't've": "should not have", "so've": "so have", "so's": "so as",
            "that'd": "that would", "that'd've": "that would have", "that's": "that is",
            "there'd": "there would", "there'd've": "there would have", "there's": "there is",
            "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
            "they'll've": "they will have", "they're": "they are", "they've": "they have",
            "to've": "to have", "wasn't": "was not", "we'd": "we would",
            "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
            "we're": "we are", "we've": "we have", "weren't": "were not",
            "what'll": "what will", "what'll've": "what will have", "what're": "what are",
            "what's": "what is", "what've": "what have", "when's": "when is",
            "when've": "when have", "where'd": "where did", "where's": "where is",
            "where've": "where have", "who'll": "who will", "who'll've": "who will have",
            "who's": "who is", "who've": "who have", "why's": "why is",
            "why've": "why have", "will've": "will have", "won't": "will not",
            "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
            "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
            "y'all'd've": "you all would have", "y'all're": "you all are",
            "y'all've": "you all have", "you'd": "you would", "you'd've": "you would have",
            "you'll": "you will", "you'll've": "you will have", "you're": "you are",
            "you've": "you have"
        }

    def preprocess_text(self, text):
        """
        Comprehensive text preprocessing
        """
        if pd.isna(text):
            return ""
        
        text = str(text).lower().strip()
        
        # Replace special characters
        text = text.replace('%', ' percent')
        text = text.replace('$', ' dollar ')
        text = text.replace('₹', ' rupee ')
        text = text.replace('€', ' euro ')
        text = text.replace('@', ' at ')
        text = text.replace('[math]', '')
        
        # Replace large numbers
        text = text.replace(',000,000,000 ', 'b ')
        text = text.replace(',000,000 ', 'm ')
        text = text.replace(',000 ', 'k ')
        text = re.sub(r'([0-9]+)000000000', r'\1b', text)
        text = re.sub(r'([0-9]+)000000', r'\1m', text)
        text = re.sub(r'([0-9]+)000', r'\1k', text)
        
        # Expand contractions
        text_decontracted = []
        for word in text.split():
            if word in self.contractions:
                word = self.contractions[word]
            text_decontracted.append(word)
        
        text = ' '.join(text_decontracted)
        text = text.replace("'ve", " have")
        text = text.replace("n't", " not")
        text = text.replace("'re", " are")
        text = text.replace("'ll", " will")
        
        # Remove HTML tags
        text = BeautifulSoup(text, 'html.parser').get_text()
        
        # Remove punctuation
        text = re.sub(r'\W', ' ', text).strip()
        
        return text

    def extract_advanced_features(self, df):
        """
        Extract advanced text features including TF-IDF and semantic similarity
        """
        # TF-IDF features
        print("Computing TF-IDF features...")
        all_texts = df['question1'].tolist() + df['question2'].tolist()
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
        
        # Split back to q1 and q2
        q1_tfidf = tfidf_matrix[:len(df)]
        q2_tfidf = tfidf_matrix[len(df):]
        
        # TF-IDF similarity
        tfidf_similarities = []
        for i in range(len(df)):
            similarity = cosine_similarity(q1_tfidf[i:i+1], q2_tfidf[i:i+1])[0][0]
            tfidf_similarities.append(similarity)
        
        df['tfidf_similarity'] = tfidf_similarities
        
        # Semantic similarity using sentence embeddings
        print("Computing semantic similarity...")
        q1_embeddings = self.get_embeddings(df['question1'].tolist())
        q2_embeddings = self.get_embeddings(df['question2'].tolist())
        
        semantic_similarities = []
        for i in range(len(df)):
            similarity = cosine_similarity(q1_embeddings[i:i+1], q2_embeddings[i:i+1])[0][0]
            semantic_similarities.append(similarity)
        
        df['semantic_similarity'] = semantic_similarities
        
        # Advanced word overlap features
        df['exact_word_match_ratio'] = df.apply(self.exact_word_match_ratio, axis=1)
        df['partial_word_match_ratio'] = df.apply(self.partial_word_match_ratio, axis=1)
        df['question_type_similarity'] = df.apply(self.question_type_similarity, axis=1)
        
        return df

    def exact_word_match_ratio(self, row):
        """Calculate exact word match ratio"""
        q1_words = set(str(row['question1']).lower().split())
        q2_words = set(str(row['question2']).lower().split())
        
        if len(q1_words) == 0 or len(q2_words) == 0:
            return 0.0
        
        intersection = len(q1_words & q2_words)
        union = len(q1_words | q2_words)
        
        return intersection / union if union > 0 else 0.0

    def partial_word_match_ratio(self, row):
        """Calculate partial word match ratio"""
        q1_words = str(row['question1']).lower().split()
        q2_words = str(row['question2']).lower().split()
        
        if len(q1_words) == 0 or len(q2_words) == 0:
            return 0.0
        
        partial_matches = 0
        for word1 in q1_words:
            for word2 in q2_words:
                if word1 in word2 or word2 in word1:
                    partial_matches += 1
                    break
        
        return partial_matches / len(q1_words) if len(q1_words) > 0 else 0.0

    def question_type_similarity(self, row):
        """Check if questions are asking for the same type of information"""
        q1 = str(row['question1']).lower()
        q2 = str(row['question2']).lower()
        
        # Question words
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
        
        q1_type = None
        q2_type = None
        
        for word in question_words:
            if word in q1:
                q1_type = word
                break
        
        for word in question_words:
            if word in q2:
                q2_type = word
                break
        
        return 1.0 if q1_type == q2_type else 0.0

    def extract_basic_features(self, df):
        """
        Extract basic text features
        """
        df['q1_len'] = df['question1'].str.len()
        df['q2_len'] = df['question2'].str.len()
        df['q1_num_words'] = df['question1'].apply(lambda x: len(str(x).split()))
        df['q2_num_words'] = df['question2'].apply(lambda x: len(str(x).split()))
        
        return df

    def extract_word_features(self, df):
        """
        Extract word-level features
        """
        def common_words(row):
            w1 = set(str(row['question1']).lower().split())
            w2 = set(str(row['question2']).lower().split())
            return len(w1 & w2)
        
        def total_words(row):
            w1 = set(str(row['question1']).lower().split())
            w2 = set(str(row['question2']).lower().split())
            return len(w1) + len(w2)
        
        df['word_common'] = df.apply(common_words, axis=1)
        df['word_total'] = df.apply(total_words, axis=1)
        df['word_share'] = df['word_common'] / df['word_total'].replace(0, 1)
        
        return df

    def extract_token_features(self, df):
        """
        Extract token-level features
        """
        def fetch_token_features(row):
            q1 = str(row['question1'])
            q2 = str(row['question2'])
            
            SAFE_DIV = 0.0001
            token_features = [0.0] * 8
            
            q1_tokens = q1.split()
            q2_tokens = q2.split()
            
            if len(q1_tokens) == 0 or len(q2_tokens) == 0:
                return token_features
            
            # Non-stopwords
            q1_words = set([word for word in q1_tokens if word not in self.stop_words])
            q2_words = set([word for word in q2_tokens if word not in self.stop_words])
            
            # Stopwords
            q1_stops = set([word for word in q1_tokens if word in self.stop_words])
            q2_stops = set([word for word in q2_tokens if word in self.stop_words])
            
            # Common words
            common_word_count = len(q1_words & q2_words)
            common_stop_count = len(q1_stops & q2_stops)
            common_token_count = len(set(q1_tokens) & set(q2_tokens))
            
            token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
            token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
            token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
            token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
            token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
            token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
            token_features[6] = int(q1_tokens[-1] == q2_tokens[-1]) if q1_tokens and q2_tokens else 0
            token_features[7] = int(q1_tokens[0] == q2_tokens[0]) if q1_tokens and q2_tokens else 0
            
            return token_features
        
        token_features = df.apply(fetch_token_features, axis=1)
        df['cwc_min'] = [x[0] for x in token_features]
        df['cwc_max'] = [x[1] for x in token_features]
        df['csc_min'] = [x[2] for x in token_features]
        df['csc_max'] = [x[3] for x in token_features]
        df['ctc_min'] = [x[4] for x in token_features]
        df['ctc_max'] = [x[5] for x in token_features]
        df['last_word_eq'] = [x[6] for x in token_features]
        df['first_word_eq'] = [x[7] for x in token_features]
        
        return df

    def extract_length_features(self, df):
        """
        Extract length-based features
        """
        def fetch_length_features(row):
            q1 = str(row['question1'])
            q2 = str(row['question2'])
            
            length_features = [0.0] * 3
            
            q1_tokens = q1.split()
            q2_tokens = q2.split()
            
            if len(q1_tokens) == 0 or len(q2_tokens) == 0:
                return length_features
            
            length_features[0] = abs(len(q1_tokens) - len(q2_tokens))
            length_features[1] = (len(q1_tokens) + len(q2_tokens)) / 2
            
            try:
                strs = list(distance.lcsubstrings(q1, q2))
                length_features[2] = len(strs[0]) / (min(len(q1), len(q2)) + 1)
            except:
                length_features[2] = 0.0
            
            return length_features
        
        length_features = df.apply(fetch_length_features, axis=1)
        df['abs_len_diff'] = [x[0] for x in length_features]
        df['mean_len'] = [x[1] for x in length_features]
        df['longest_substr_ratio'] = [x[2] for x in length_features]
        
        return df

    def extract_fuzzy_features(self, df):
        """
        Extract fuzzy matching features
        """
        def fetch_fuzzy_features(row):
            q1 = str(row['question1'])
            q2 = str(row['question2'])
            
            fuzzy_features = [0.0] * 4
            
            try:
                fuzzy_features[0] = fuzz.QRatio(q1, q2)
                fuzzy_features[1] = fuzz.partial_ratio(q1, q2)
                fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)
                fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)
            except:
                pass
            
            return fuzzy_features
        
        fuzzy_features = df.apply(fetch_fuzzy_features, axis=1)
        df['fuzz_ratio'] = [x[0] for x in fuzzy_features]
        df['fuzz_partial_ratio'] = [x[1] for x in fuzzy_features]
        df['token_sort_ratio'] = [x[2] for x in fuzzy_features]
        df['token_set_ratio'] = [x[3] for x in fuzzy_features]
        
        return df

    def get_embeddings(self, texts):
        """
        Get sentence embeddings using transformers
        """
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings

    def process_data(self, df, sample_size=None):
        """
        Complete data processing pipeline with advanced features
        """
        print("Loading and preprocessing data...")
        
        # Sample data if specified
        if sample_size:
            df = df.sample(sample_size, random_state=42)
        
        # Remove null values
        df = df.dropna()
        
        # Preprocess text
        df['question1'] = df['question1'].apply(self.preprocess_text)
        df['question2'] = df['question2'].apply(self.preprocess_text)
        
        # Extract features
        df = self.extract_basic_features(df)
        df = self.extract_word_features(df)
        df = self.extract_token_features(df)
        df = self.extract_length_features(df)
        df = self.extract_fuzzy_features(df)
        df = self.extract_advanced_features(df)
        
        print("Getting sentence embeddings...")
        # Get embeddings
        q1_embeddings = self.get_embeddings(df['question1'].tolist())
        q2_embeddings = self.get_embeddings(df['question2'].tolist())
        
        # Combine embeddings with other features
        feature_columns = ['cwc_min', 'cwc_max', 'csc_min', 'csc_max', 'ctc_min', 'ctc_max',
                          'last_word_eq', 'first_word_eq', 'abs_len_diff', 'mean_len',
                          'token_set_ratio', 'token_sort_ratio', 'fuzz_ratio', 
                          'fuzz_partial_ratio', 'longest_substr_ratio', 'word_share',
                          'tfidf_similarity', 'semantic_similarity', 'exact_word_match_ratio',
                          'partial_word_match_ratio', 'question_type_similarity']
        
        other_features = df[feature_columns].values
        
        # Combine all features
        X = np.hstack((q1_embeddings, q2_embeddings, other_features))
        y = df['is_duplicate'].values
        
        return X, y, df

    def save_models(self, models, scaler, output_dir='models'):
        """
        Save trained models and scaler
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for name, model in models.items():
            joblib.dump(model, os.path.join(output_dir, f'{name}.pkl'))
        
        joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
        print(f"Models saved to {output_dir}/")

    def load_models(self, input_dir='models'):
        """
        Load trained models and scaler
        """
        models = {}
        for file in os.listdir(input_dir):
            if file.endswith('.pkl') and file != 'scaler.pkl':
                name = file.replace('.pkl', '')
                models[name] = joblib.load(os.path.join(input_dir, file))
        
        scaler = joblib.load(os.path.join(input_dir, 'scaler.pkl'))
        return models, scaler 