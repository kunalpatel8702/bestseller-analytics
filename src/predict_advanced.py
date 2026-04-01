"""
Advanced Prediction and Causal Optimization Module (v2)
- Price Model: Predicts market price.
- Optimization: Simulates price to maximize Expected Value (Price * P(Success)).
"""
import pandas as pd
import numpy as np
import joblib
import shap
import catboost as cb

class AdvancedPredictor:
    def __init__(self):
        self.price_model = None
        self.success_model = None
        self.feature_names_price = None
        self.feature_names_success = None
        self.data_path = "data/features.csv"
        self.genre_stats = None
        self.author_freq = None
        
    def load_artifacts(self):
        """Load all models and artifacts."""
        try:
            self.price_model = joblib.load("models/price_model_advanced.pkl")
            self.success_model = joblib.load("models/success_model_advanced.pkl")
            
            # Load feature columns from training data header to ensure alignment
            # We need to know which features belong to which model
            # Re-read data to get columns (robust way is to save feature names during training)
            # For now, we infer from data/features.csv structure
            df = pd.read_csv("data/features.csv", nrows=1)
            all_cols = [c for c in df.columns if c not in ['price', 'sales_rank', 'is_success']]
            
            self.feature_names_price = all_cols
            self.feature_names_success = all_cols + ['price'] # Price is feature for Success Model
            
            # Load stats
            full_df = pd.read_csv("data/features.csv")
            self.genre_stats = full_df.groupby('genre')['price'].mean().to_dict()
            self.author_freq = full_df['author'].value_counts().to_dict()
            
            print("[INFO] Advanced Predictor artifacts loaded successfully.")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load artifacts: {str(e)}")
            return False

    def prepare_base_features(self, book_dict):
        """Prepare base features (dictionaries/encodings) common to both models."""
        title = str(book_dict.get('title', 'Unknown Title'))
        genre = book_dict.get('genre', 'Non-Fiction')
        author = book_dict.get('author', 'Unknown')
        publisher = book_dict.get('publisher', 'Unknown')
        
        # NLP Features
        keywords = ['Guide', 'Ultimate', 'Beginner', 'Advanced', 'Masterclass', 'Strategies']
        has_keywords = {f'title_has_{kw}': (1 if kw in title else 0) for kw in keywords}
        
        # Encodings
        author_freq = self.author_freq.get(author, 1)
        genre_target_price = self.genre_stats.get(genre, 15.0)
        author_target_price = genre_target_price 
        
        row = {
            'title': title, # CatBoost handles text/cat
            'author': author,
            'genre': genre,
            'publisher': publisher,
            'rating': float(book_dict.get('rating', 4.0)),
            'reviews_count': int(book_dict.get('reviews_count', 0)),
            'year': int(book_dict.get('year', 2024)),
            'title_len_char': len(title),
            'title_len_word': len(title.split()),
            'author_freq': author_freq,
            'genre_target_price': genre_target_price,
            'author_target_price': author_target_price
        }
        row.update(has_keywords)
        
        # DataFrame conversion
        df = pd.DataFrame([row])
        
        # Fill missing numeric cols
        for col in self.feature_names_price:
            if col not in df.columns:
                df[col] = 0
                
        return df

    def predict_market_price(self, book_dict):
        """Predict the 'Market Price' (Regression)."""
        df = self.prepare_base_features(book_dict)
        df_price = df[self.feature_names_price]
        return self.price_model.predict(df_price)[0]

    def optimize_price(self, book_dict, min_price=4.99, max_price=50.0, steps=20):
        """
        Simulate success probability across price range.
        Returns optimal price and curve data.
        """
        base_df = self.prepare_base_features(book_dict)
        
        prices = np.linspace(min_price, max_price, steps)
        probs = []
        expected_values = []
        
        # Create batch dataframe for efficiency
        # Duplicate base row 'steps' times
        batch_df = pd.concat([base_df]*steps, ignore_index=True)
        batch_df['price'] = prices # Inject simulated price
        
        # Ensure correct column order
        batch_df = batch_df[self.feature_names_success]
        
        # Predict success probabilities for batch
        probs = self.success_model.predict_proba(batch_df)[:, 1]
        
        # Calculate heuristics
        expected_values = prices * probs
        
        # Find optimal
        optimal_idx = np.argmax(expected_values)
        optimal_price = prices[optimal_idx]
        optimal_prob = probs[optimal_idx]
        max_ev = expected_values[optimal_idx]
        
        market_price = self.predict_market_price(book_dict)
        
        return {
            'optimal_price': round(optimal_price, 2),
            'market_price': round(market_price, 2),
            'max_expected_value': round(max_ev, 2),
            'success_probability': round(optimal_prob, 2),
            'curve': {
                'prices': prices.tolist(),
                'probs': probs.tolist(),
                'expected_values': expected_values.tolist()
            }
        }
        
    def explain_prediction(self, book_dict):
        """Local SHAP explanation for the Price Model."""
        df = self.prepare_base_features(book_dict)
        df = df[self.feature_names_price]
        
        explainer = shap.TreeExplainer(self.price_model)
        shap_values = explainer(df)
        return shap_values

if __name__ == "__main__":
    # Test
    ap = AdvancedPredictor()
    if ap.load_artifacts():
        test_book = {'title': 'The Ultimate Guide to Machine Learning', 'genre': 'Technology', 'rating': 4.5}
        res = ap.optimize_price(test_book)
        print(f"Optimal Price: ${res['optimal_price']}")
        print(f"Market Price: ${res['market_price']}")
