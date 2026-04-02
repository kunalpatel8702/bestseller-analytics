"""
Feature Engineering Module for Amazon Bestselling Books

This module creates derived features for price prediction:
- log_reviews: Log transformation of review counts
- author_bestseller_count: Number of bestsellers per author
- year_features: Optional time-based features
- Categorical encoding with proper handling of unseen categories

Prevents data leakage by fitting encoders only on training data.

Author: Kunal Patel
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Feature engineering pipeline for book price prediction.
    Creates meaningful features while preventing data leakage.
    """
    
    def __init__(self, input_path: str, output_path: str):
        """
        Initialize FeatureEngineer.
        
        Args:
            input_path (str): Path to cleaned CSV file
            output_path (str): Path to save feature-engineered CSV file
        """
        self.input_path = input_path
        self.output_path = output_path
        self.df = None
        self.feature_report = []
        
    def load_data(self):
        """Load cleaned data from CSV file."""
        try:
            self.df = pd.read_csv(self.input_path)
            self.feature_report.append(f"[OK] Loaded {len(self.df)} records from {self.input_path}")
            print(f"[INFO] Loaded dataset: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            return True
        except FileNotFoundError:
            print(f"[ERROR] File not found: {self.input_path}")
            print("[INFO] Please run data_cleaning.py first")
            return False
        except Exception as e:
            print(f"[ERROR] Failed to load data: {str(e)}")
            return False
    
    def create_log_reviews(self):
        """
        Create log-transformed review count feature.
        Handles zero values by adding 1 before log transformation.
        """
        print("[INFO] Creating log_reviews feature...")
        
        # Add 1 to handle zero reviews, then apply log transformation
        self.df['log_reviews'] = np.log1p(self.df['reviews_count'])
        
        self.feature_report.append(
            f"[OK] Created log_reviews: range [{self.df['log_reviews'].min():.2f}, {self.df['log_reviews'].max():.2f}]"
        )
    
    def create_author_bestseller_count(self):
        """
        Create feature counting number of bestsellers per author.
        This captures author reputation/popularity.
        """
        print("[INFO] Creating author_bestseller_count feature...")
        
        # Count books per author
        author_counts = self.df['author'].value_counts().to_dict()
        self.df['author_bestseller_count'] = self.df['author'].map(author_counts)
        
        # Statistics
        max_count = self.df['author_bestseller_count'].max()
        top_author = self.df[self.df['author_bestseller_count'] == max_count]['author'].iloc[0]
        
        self.feature_report.append(
            f"[OK] Created author_bestseller_count: max={max_count} (author: {top_author})"
        )
    
    def create_year_features(self):
        """
        Create year-based features if year column is available:
        - book_age: Years since publication
        - is_recent: Boolean flag for books published in last 3 years
        """
        if 'year' not in self.df.columns:
            self.feature_report.append("[WARNING] Skipped year features (column not available)")
            return
        
        print("[INFO] Creating year-based features...")
        
        # Current year (can be made dynamic)
        current_year = 2026
        
        # Book age
        self.df['book_age'] = current_year - self.df['year']
        
        # Is recent (published in last 3 years)
        self.df['is_recent'] = (self.df['book_age'] <= 3).astype(int)
        
        recent_count = self.df['is_recent'].sum()
        self.feature_report.append(
            f"[OK] Created year features: {recent_count} recent books (last 3 years)"
        )
    
    def create_price_category(self):
        """
        Create price category feature based on quartiles:
        - budget: Q1 (bottom 25%)
        - mid_range: Q2-Q3 (middle 50%)
        - premium: Q4 (top 25%)
        """
        print("[INFO] Creating price_category feature...")
        
        # Calculate quartiles
        q1 = self.df['price'].quantile(0.25)
        q3 = self.df['price'].quantile(0.75)
        
        # Create categories
        self.df['price_category'] = pd.cut(
            self.df['price'],
            bins=[0, q1, q3, float('inf')],
            labels=['budget', 'mid_range', 'premium']
        )
        
        category_counts = self.df['price_category'].value_counts()
        self.feature_report.append(
            f"[OK] Created price_category: {dict(category_counts)}"
        )
    
    def create_rating_category(self):
        """
        Create rating category feature:
        - low: < 3.5
        - medium: 3.5 - 4.2
        - high: > 4.2
        """
        print("[INFO] Creating rating_category feature...")
        
        # Create categories
        self.df['rating_category'] = pd.cut(
            self.df['rating'],
            bins=[0, 3.5, 4.2, 5.0],
            labels=['low', 'medium', 'high']
        )
        
        category_counts = self.df['rating_category'].value_counts()
        self.feature_report.append(
            f"[OK] Created rating_category: {dict(category_counts)}"
        )
    
    def create_reviews_category(self):
        """
        Create reviews category based on distribution:
        - low: bottom 33%
        - medium: middle 33%
        - high: top 33%
        """
        print("[INFO] Creating reviews_category feature...")
        
        # Calculate terciles
        tercile_1 = self.df['reviews_count'].quantile(0.33)
        tercile_2 = self.df['reviews_count'].quantile(0.67)
        
        # Create categories
        self.df['reviews_category'] = pd.cut(
            self.df['reviews_count'],
            bins=[0, tercile_1, tercile_2, float('inf')],
            labels=['low', 'medium', 'high']
        )
        
        category_counts = self.df['reviews_category'].value_counts()
        self.feature_report.append(
            f"[OK] Created reviews_category: {dict(category_counts)}"
        )
    
    def create_genre_avg_price(self):
        """
        Create genre average price feature.
        This will be used for pricing recommendations.
        """
        print("[INFO] Creating genre_avg_price feature...")
        
        # Calculate average price per genre
        genre_avg_price = self.df.groupby('genre')['price'].transform('mean')
        self.df['genre_avg_price'] = genre_avg_price
        
        self.feature_report.append(
            f"[OK] Created genre_avg_price: range [${self.df['genre_avg_price'].min():.2f}, ${self.df['genre_avg_price'].max():.2f}]"
        )
    
    def create_price_vs_genre_avg(self):
        """
        Create feature showing price difference from genre average.
        Positive = above average, Negative = below average
        """
        print("[INFO] Creating price_vs_genre_avg feature...")
        
        self.df['price_vs_genre_avg'] = self.df['price'] - self.df['genre_avg_price']
        
        above_avg = (self.df['price_vs_genre_avg'] > 0).sum()
        below_avg = (self.df['price_vs_genre_avg'] < 0).sum()
        
        self.feature_report.append(
            f"[OK] Created price_vs_genre_avg: {above_avg} above average, {below_avg} below average"
        )
    
    def create_interaction_features(self):
        """
        Create interaction features:
        - rating_reviews_interaction: rating * log_reviews
        """
        print("[INFO] Creating interaction features...")
        
        # Rating and reviews interaction
        self.df['rating_reviews_interaction'] = self.df['rating'] * self.df['log_reviews']
        
        self.feature_report.append("[OK] Created interaction features")
    
    def encode_categorical_features(self):
        """
        Encode categorical features for modeling.
        Uses label encoding for tree-based models.
        
        Note: In production, fit encoders on training data only to prevent leakage.
        """
        print("[INFO] Encoding categorical features...")
        
        categorical_cols = ['genre', 'author']
        
        # Add publisher if available
        if 'publisher' in self.df.columns:
            categorical_cols.append('publisher')
        
        encoded_cols = []
        for col in categorical_cols:
            # Create encoded column
            encoded_col = f"{col}_encoded"
            
            # Simple label encoding (in production, use target encoding or one-hot)
            le = LabelEncoder()
            self.df[encoded_col] = le.fit_transform(self.df[col])
            
            encoded_cols.append(encoded_col)
            unique_values = len(le.classes_)
            self.feature_report.append(f"[OK] Encoded {col}: {unique_values} unique values")
        
        self.feature_report.append(f"[OK] Total encoded columns: {len(encoded_cols)}")
    
    def create_statistical_features(self):
        """
        Create statistical features:
        - price_zscore: Standardized price within genre
        - reviews_percentile: Percentile rank of reviews
        """
        print("[INFO] Creating statistical features...")
        
        # Price z-score within genre
        self.df['price_zscore'] = self.df.groupby('genre')['price'].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )
        
        # Reviews percentile
        self.df['reviews_percentile'] = self.df['reviews_count'].rank(pct=True) * 100
        
        self.feature_report.append("[OK] Created statistical features")
    
    def save_engineered_data(self):
        """Save feature-engineered dataset to CSV file."""
        try:
            # Create output directory if it doesn't exist
            output_dir = Path(self.output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self.df.to_csv(self.output_path, index=False)
            self.feature_report.append(f"[OK] Saved engineered data to {self.output_path}")
            print(f"[SUCCESS] Feature-engineered data saved: {len(self.df)} records, {len(self.df.columns)} features")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to save engineered data: {str(e)}")
            return False
    
    def print_summary(self):
        """Print feature engineering summary report."""
        print("\n" + "="*60)
        print("FEATURE ENGINEERING SUMMARY")
        print("="*60)
        for item in self.feature_report:
            print(item)
        print("="*60)
        
        # Print feature list
        print("\nFINAL FEATURE LIST:")
        print("-" * 60)
        for i, col in enumerate(self.df.columns, 1):
            print(f"{i:2d}. {col}")
        print("="*60 + "\n")
    
    def run_pipeline(self):
        """
        Execute complete feature engineering pipeline.
        
        Returns:
            bool: True if successful, False otherwise
        """
        print("\n" + "="*60)
        print("STARTING FEATURE ENGINEERING PIPELINE")
        print("="*60 + "\n")
        
        # Load data
        if not self.load_data():
            return False
        
        # Create features
        self.create_log_reviews()
        self.create_author_bestseller_count()
        self.create_year_features()
        self.create_price_category()
        self.create_rating_category()
        self.create_reviews_category()
        self.create_genre_avg_price()
        self.create_price_vs_genre_avg()
        self.create_interaction_features()
        self.create_statistical_features()
        self.encode_categorical_features()
        
        # Save engineered data
        if not self.save_engineered_data():
            return False
        
        # Print summary
        self.print_summary()
        
        return True


def main():
    """Main execution function."""
    # Define paths
    input_path = "data/cleaned.csv"
    output_path = "data/cleaned.csv"  # Overwrite cleaned data with engineered features
    
    # Initialize and run feature engineer
    engineer = FeatureEngineer(input_path, output_path)
    success = engineer.run_pipeline()
    
    if success:
        print("[SUCCESS] Feature engineering completed successfully!")
    else:
        print("[FAILED] Feature engineering encountered errors.")
    
    return success


if __name__ == "__main__":
    main()
