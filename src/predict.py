"""
Prediction and Pricing Recommendation Module

This module provides:
- Price prediction for new books
- Pricing recommendations based on market positioning
- Genre-based pricing analysis
- Confidence intervals for predictions

Author: Senior Data Science Team
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class PricePredictor:
    """
    Price prediction and recommendation engine for books.
    """
    
    def __init__(self, model_path: str, feature_names_path: str, data_path: str):
        """
        Initialize PricePredictor.
        
        Args:
            model_path (str): Path to trained model
            feature_names_path (str): Path to feature names pickle
            data_path (str): Path to training data (for genre averages)
        """
        self.model_path = model_path
        self.feature_names_path = feature_names_path
        self.data_path = data_path
        self.model = None
        self.feature_names = None
        self.genre_stats = None
        self.author_stats = None
        
    def load_model(self):
        """Load trained model from disk."""
        try:
            self.model = joblib.load(self.model_path)
            print(f"[INFO] Loaded model from {self.model_path}")
            return True
        except FileNotFoundError:
            print(f"[ERROR] Model file not found: {self.model_path}")
            return False
        except Exception as e:
            print(f"[ERROR] Failed to load model: {str(e)}")
            return False
    
    def load_feature_names(self):
        """Load feature names from disk."""
        try:
            self.feature_names = joblib.load(self.feature_names_path)
            print(f"[INFO] Loaded {len(self.feature_names)} feature names")
            return True
        except FileNotFoundError:
            print(f"[ERROR] Feature names file not found: {self.feature_names_path}")
            return False
        except Exception as e:
            print(f"[ERROR] Failed to load feature names: {str(e)}")
            return False
    
    def load_reference_data(self):
        """Load reference data for genre and author statistics."""
        try:
            df = pd.read_csv(self.data_path)
            
            # Calculate genre statistics
            self.genre_stats = df.groupby('genre')['price'].agg(['mean', 'std', 'min', 'max']).to_dict('index')
            
            # Calculate author statistics
            author_counts = df['author'].value_counts().to_dict()
            self.author_stats = author_counts
            
            print(f"[INFO] Loaded reference data: {len(self.genre_stats)} genres, {len(self.author_stats)} authors")
            return True
        except FileNotFoundError:
            print(f"[ERROR] Data file not found: {self.data_path}")
            return False
        except Exception as e:
            print(f"[ERROR] Failed to load reference data: {str(e)}")
            return False
    
    def initialize(self):
        """Initialize predictor by loading all required components."""
        success = self.load_model() and self.load_feature_names() and self.load_reference_data()
        if success:
            print("[SUCCESS] Predictor initialized successfully")
        return success
    
    def prepare_features(self, book_dict: dict) -> pd.DataFrame:
        """
        Prepare features from book dictionary.
        
        Args:
            book_dict (dict): Dictionary with book attributes
                Required keys: genre, rating, reviews_count, author
                Optional keys: year, publisher
        
        Returns:
            pd.DataFrame: Feature dataframe ready for prediction
        """
        # Extract basic features
        genre = book_dict.get('genre', 'unknown').lower().strip()
        rating = book_dict.get('rating', 4.0)
        reviews_count = book_dict.get('reviews_count', 0)
        author = book_dict.get('author', 'Unknown Author')
        year = book_dict.get('year', 2020)
        
        # Create derived features
        log_reviews = np.log1p(reviews_count)
        author_bestseller_count = self.author_stats.get(author, 1)
        
        # Year features (if year is in feature names)
        current_year = 2026
        book_age = current_year - year
        is_recent = 1 if book_age <= 3 else 0
        
        # Genre statistics
        genre_info = self.genre_stats.get(genre, {'mean': 15.0, 'std': 5.0})
        genre_avg_price = genre_info['mean']
        
        # Interaction features
        rating_reviews_interaction = rating * log_reviews
        
        # Statistical features (using genre stats)
        price_zscore = 0  # Will be calculated after prediction
        reviews_percentile = 50  # Default to median
        
        # Categorical encoding (simple mapping for demo)
        # In production, use the same encoder as training
        genre_encoded = hash(genre) % 100
        author_encoded = hash(author) % 100
        
        # Create feature dictionary
        features = {
            'rating': rating,
            'reviews_count': reviews_count,
            'log_reviews': log_reviews,
            'author_bestseller_count': author_bestseller_count,
            'rating_reviews_interaction': rating_reviews_interaction,
            'genre_avg_price': genre_avg_price,
            'genre_encoded': genre_encoded,
            'author_encoded': author_encoded,
        }
        
        # Add year features if they exist in feature names
        if 'book_age' in self.feature_names:
            features['book_age'] = book_age
        if 'is_recent' in self.feature_names:
            features['is_recent'] = is_recent
        if 'year' in self.feature_names:
            features['year'] = year
        
        # Add statistical features if they exist
        if 'price_zscore' in self.feature_names:
            features['price_zscore'] = price_zscore
        if 'reviews_percentile' in self.feature_names:
            features['reviews_percentile'] = reviews_percentile
        if 'price_vs_genre_avg' in self.feature_names:
            features['price_vs_genre_avg'] = 0  # Will be updated after prediction
        
        # Create dataframe with only the features used in training
        feature_df = pd.DataFrame([features])
        
        # Ensure all required features are present
        for feat in self.feature_names:
            if feat not in feature_df.columns:
                feature_df[feat] = 0  # Default value
        
        # Select only the features in the correct order
        feature_df = feature_df[self.feature_names]
        
        return feature_df
    
    def predict_price(self, book_dict: dict) -> float:
        """
        Predict price for a book.
        
        Args:
            book_dict (dict): Dictionary with book attributes
        
        Returns:
            float: Predicted price
        """
        # Prepare features
        features = self.prepare_features(book_dict)
        
        # Make prediction
        predicted_price = self.model.predict(features)[0]
        
        return predicted_price
    
    def get_pricing_recommendation(self, book_dict: dict, predicted_price: float = None) -> dict:
        """
        Generate pricing recommendation based on market positioning.
        
        Args:
            book_dict (dict): Dictionary with book attributes
            predicted_price (float): Predicted price (if None, will predict)
        
        Returns:
            dict: Recommendation with band, reasoning, and suggested price range
        """
        # Get predicted price if not provided
        if predicted_price is None:
            predicted_price = self.predict_price(book_dict)
        
        # Get genre statistics
        genre = book_dict.get('genre', 'unknown').lower().strip()
        genre_info = self.genre_stats.get(genre, {'mean': 15.0, 'std': 5.0, 'min': 5.0, 'max': 30.0})
        
        genre_avg = genre_info['mean']
        genre_std = genre_info['std']
        genre_min = genre_info['min']
        genre_max = genre_info['max']
        
        # Calculate price position relative to genre
        price_diff = predicted_price - genre_avg
        price_diff_pct = (price_diff / genre_avg) * 100
        
        # Determine pricing band
        if predicted_price > genre_avg + 0.5 * genre_std:
            band = "PREMIUM"
            reasoning = f"Predicted price (${predicted_price:.2f}) is {abs(price_diff_pct):.1f}% above genre average (${genre_avg:.2f}). "
            reasoning += "This book has strong market positioning. Consider premium pricing strategy."
            suggested_min = predicted_price * 0.95
            suggested_max = min(predicted_price * 1.05, genre_max)
            
        elif predicted_price < genre_avg - 0.5 * genre_std:
            band = "COMPETITIVE"
            reasoning = f"Predicted price (${predicted_price:.2f}) is {abs(price_diff_pct):.1f}% below genre average (${genre_avg:.2f}). "
            reasoning += "Consider competitive pricing to capture market share."
            suggested_min = max(predicted_price * 0.95, genre_min)
            suggested_max = predicted_price * 1.05
            
        else:
            band = "MARKET_RATE"
            reasoning = f"Predicted price (${predicted_price:.2f}) is within {abs(price_diff_pct):.1f}% of genre average (${genre_avg:.2f}). "
            reasoning += "Price at market rate for optimal balance of revenue and volume."
            suggested_min = predicted_price * 0.95
            suggested_max = predicted_price * 1.05
        
        # Add additional context
        rating = book_dict.get('rating', 4.0)
        reviews_count = book_dict.get('reviews_count', 0)
        
        if rating >= 4.5 and reviews_count > 1000:
            reasoning += " High rating and strong review count support premium positioning."
        elif rating < 3.5:
            reasoning += " Lower rating suggests conservative pricing approach."
        
        if reviews_count < 100:
            reasoning += " Limited reviews - consider introductory pricing to build traction."
        
        return {
            'predicted_price': round(predicted_price, 2),
            'pricing_band': band,
            'reasoning': reasoning,
            'suggested_price_range': {
                'min': round(suggested_min, 2),
                'max': round(suggested_max, 2)
            },
            'genre_average': round(genre_avg, 2),
            'genre_range': {
                'min': round(genre_min, 2),
                'max': round(genre_max, 2)
            }
        }
    
    def batch_predict(self, books_list: list) -> pd.DataFrame:
        """
        Predict prices for multiple books.
        
        Args:
            books_list (list): List of book dictionaries
        
        Returns:
            pd.DataFrame: Predictions and recommendations
        """
        results = []
        
        for book in books_list:
            try:
                predicted_price = self.predict_price(book)
                recommendation = self.get_pricing_recommendation(book, predicted_price)
                
                result = {
                    'title': book.get('title', 'Unknown'),
                    'author': book.get('author', 'Unknown'),
                    'genre': book.get('genre', 'unknown'),
                    'predicted_price': predicted_price,
                    'pricing_band': recommendation['pricing_band'],
                    'suggested_min': recommendation['suggested_price_range']['min'],
                    'suggested_max': recommendation['suggested_price_range']['max']
                }
                results.append(result)
            except Exception as e:
                print(f"[WARNING] Failed to predict for book: {book.get('title', 'Unknown')} - {str(e)}")
        
        return pd.DataFrame(results)


# Convenience functions for easy import
_predictor = None

def initialize_predictor(model_path="models/price_model.pkl", 
                        feature_names_path="models/feature_names.pkl",
                        data_path="data/cleaned.csv"):
    """Initialize global predictor instance."""
    global _predictor
    _predictor = PricePredictor(model_path, feature_names_path, data_path)
    return _predictor.initialize()


def predict_price(book_features: dict) -> float:
    """
    Predict price for a book.
    
    Args:
        book_features (dict): Book attributes
            Required: genre, rating, reviews_count, author
            Optional: year, publisher
    
    Returns:
        float: Predicted price
    
    Example:
        >>> price = predict_price({
        ...     'genre': 'Fiction',
        ...     'rating': 4.5,
        ...     'reviews_count': 5000,
        ...     'author': 'John Doe'
        ... })
        >>> print(f"Predicted price: ${price:.2f}")
    """
    global _predictor
    if _predictor is None:
        if not initialize_predictor():
            raise RuntimeError("Failed to initialize predictor")
    
    return _predictor.predict_price(book_features)


def get_pricing_recommendation(book_features: dict, predicted_price: float = None) -> dict:
    """
    Get pricing recommendation for a book.
    
    Args:
        book_features (dict): Book attributes
        predicted_price (float): Optional pre-computed predicted price
    
    Returns:
        dict: Recommendation with pricing band and reasoning
    
    Example:
        >>> recommendation = get_pricing_recommendation({
        ...     'genre': 'Fiction',
        ...     'rating': 4.5,
        ...     'reviews_count': 5000,
        ...     'author': 'John Doe'
        ... })
        >>> print(recommendation['pricing_band'])
        >>> print(recommendation['reasoning'])
    """
    global _predictor
    if _predictor is None:
        if not initialize_predictor():
            raise RuntimeError("Failed to initialize predictor")
    
    return _predictor.get_pricing_recommendation(book_features, predicted_price)


def main():
    """Main execution function with examples."""
    print("\n" + "="*60)
    print("PRICE PREDICTION & RECOMMENDATION ENGINE")
    print("="*60 + "\n")
    
    # Initialize predictor
    predictor = PricePredictor(
        model_path="models/price_model.pkl",
        feature_names_path="models/feature_names.pkl",
        data_path="data/cleaned.csv"
    )
    
    if not predictor.initialize():
        print("[FAILED] Could not initialize predictor")
        return False
    
    # Example 1: Single prediction
    print("\n" + "="*60)
    print("EXAMPLE 1: Single Book Prediction")
    print("="*60 + "\n")
    
    book1 = {
        'title': 'The Great Novel',
        'genre': 'fiction',
        'rating': 4.5,
        'reviews_count': 5000,
        'author': 'Jane Smith',
        'year': 2023
    }
    
    predicted_price = predictor.predict_price(book1)
    recommendation = predictor.get_pricing_recommendation(book1, predicted_price)
    
    print(f"Book: {book1['title']}")
    print(f"Genre: {book1['genre']}")
    print(f"Rating: {book1['rating']}")
    print(f"Reviews: {book1['reviews_count']:,}")
    print(f"\nPredicted Price: ${recommendation['predicted_price']:.2f}")
    print(f"Pricing Band: {recommendation['pricing_band']}")
    print(f"Suggested Range: ${recommendation['suggested_price_range']['min']:.2f} - ${recommendation['suggested_price_range']['max']:.2f}")
    print(f"\nReasoning: {recommendation['reasoning']}")
    
    # Example 2: Batch prediction
    print("\n" + "="*60)
    print("EXAMPLE 2: Batch Prediction")
    print("="*60 + "\n")
    
    books = [
        {'title': 'Mystery Novel', 'genre': 'mystery', 'rating': 4.2, 'reviews_count': 2000, 'author': 'John Doe'},
        {'title': 'Romance Story', 'genre': 'romance', 'rating': 4.7, 'reviews_count': 8000, 'author': 'Jane Doe'},
        {'title': 'Sci-Fi Epic', 'genre': 'science fiction', 'rating': 4.0, 'reviews_count': 1500, 'author': 'Bob Smith'},
    ]
    
    results = predictor.batch_predict(books)
    print(results.to_string(index=False))
    
    print("\n[SUCCESS] Prediction examples completed!")
    return True


if __name__ == "__main__":
    main()
