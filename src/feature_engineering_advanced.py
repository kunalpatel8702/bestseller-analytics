"""
Advanced Feature Engineering Module
Includes NLP, Frequency Encoding, and Target Transformation.
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

class AdvancedFeatureEngineer:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.df = None
        self.tfidf_vectorizer = None
        
    def load_data(self):
        self.df = pd.read_csv(self.input_path)
        print(f"[INFO] Loaded data: {len(self.df)} records")
        
    def add_nlp_features(self):
        """Extract features from text fields (Title)."""
        print("[INFO] Adding NLP features...")
        
        # 1. Structural features
        self.df['title_len_char'] = self.df['title'].astype(str).apply(len)
        self.df['title_len_word'] = self.df['title'].astype(str).apply(lambda x: len(x.split()))
        
        # 2. Keyword Flags (Heuristic)
        keywords = ['Guide', 'Ultimate', 'Beginner', 'Advanced', 'Masterclass', 'Strategies']
        for kw in keywords:
            self.df[f'title_has_{kw}'] = self.df['title'].astype(str).apply(lambda x: 1 if kw in x else 0)
            
        print(f"[INFO] Added {len(keywords)} keyword flags")
        
    def add_categorical_encodings(self):
        """Advanced encoding for high-cardinality features."""
        print("[INFO] Adding advanced encodings...")
        
        # 1. Frequency Encoding (Count Encoding)
        for col in ['author', 'publisher']:
            freq_map = self.df[col].value_counts().to_dict()
            self.df[f'{col}_freq'] = self.df[col].map(freq_map)
            
        # 2. Target Encoding (Mean Price by Category)
        # Note: In production, compute this on TRAIN set only to avoid leakage.
        # Since this is a demo pipeline, we apply it globally but simulate standard practice.
        for col in ['genre', 'author']:
            target_mean = self.df.groupby(col)['price'].transform('mean')
            self.df[f'{col}_target_price'] = target_mean
            
    def create_success_target(self):
        """Create binary target for 'Success' model."""
        # Define success as Top 25% Sales Rank (Lower rank is better)
        threshold = self.df['sales_rank'].quantile(0.25)
        self.df['is_success'] = (self.df['sales_rank'] <= threshold).astype(int)
        
        print(f"[INFO] Created 'is_success' target (Threshold Rank: {threshold})")
        
    def run(self):
        self.load_data()
        self.add_nlp_features()
        self.add_categorical_encodings()
        self.create_success_target()
        
        # Save
        self.df.to_csv(self.output_path, index=False)
        print(f"[SUCCESS] Saved engineered data with {len(self.df.columns)} columns")
        return True

if __name__ == "__main__":
    fe = AdvancedFeatureEngineer("data/raw.csv", "data/features.csv")
    fe.run()
