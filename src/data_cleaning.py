"""
Data Cleaning Module for Amazon Bestselling Books

This module handles all data preprocessing tasks including:
- Price standardization (remove currency symbols, convert to float)
- Reviews count conversion to integer
- Genre text standardization
- Duplicate removal
- Missing value handling
- Data validation and quality checks

Author: Senior Data Science Team
"""

import pandas as pd
import numpy as np
import re
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')


class DataCleaner:
    """
    Robust data cleaning pipeline for book dataset.
    Handles various data quality issues and standardizes formats.
    """
    
    def __init__(self, input_path: str, output_path: str):
        """
        Initialize DataCleaner.
        
        Args:
            input_path (str): Path to raw CSV file
            output_path (str): Path to save cleaned CSV file
        """
        self.input_path = input_path
        self.output_path = output_path
        self.df = None
        self.cleaning_report = []
        
    def load_data(self):
        """Load raw data from CSV file."""
        try:
            self.df = pd.read_csv(self.input_path)
            self.cleaning_report.append(f"[OK] Loaded {len(self.df)} records from {self.input_path}")
            print(f"[INFO] Loaded dataset: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            return True
        except FileNotFoundError:
            print(f"[ERROR] File not found: {self.input_path}")
            return False
        except Exception as e:
            print(f"[ERROR] Failed to load data: {str(e)}")
            return False
    
    def validate_required_columns(self):
        """
        Validate that required columns exist in the dataset.
        
        Required columns: title, author, genre, price, rating, reviews_count
        Optional columns: year, publisher
        """
        required_cols = ['title', 'author', 'genre', 'price', 'rating', 'reviews_count']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        
        if missing_cols:
            print(f"[ERROR] Missing required columns: {missing_cols}")
            print(f"[INFO] Available columns: {list(self.df.columns)}")
            return False
        
        self.cleaning_report.append(f"[OK] All required columns present")
        
        # Check for optional columns
        optional_cols = ['year', 'publisher']
        available_optional = [col for col in optional_cols if col in self.df.columns]
        if available_optional:
            self.cleaning_report.append(f"[OK] Optional columns found: {available_optional}")
        
        return True
    
    def clean_price(self):
        """
        Clean price column:
        - Remove currency symbols ($, £, €, etc.)
        - Remove commas
        - Convert to float
        - Handle missing values
        """
        print("[INFO] Cleaning price column...")
        original_count = len(self.df)
        
        # Convert to string first to handle various formats
        self.df['price'] = self.df['price'].astype(str)
        
        # Remove currency symbols and commas
        self.df['price'] = self.df['price'].apply(
            lambda x: re.sub(r'[^\d.]', '', x) if pd.notna(x) else x
        )
        
        # Convert to float
        self.df['price'] = pd.to_numeric(self.df['price'], errors='coerce')
        
        # Handle missing or invalid prices
        missing_price = self.df['price'].isna().sum()
        if missing_price > 0:
            # Fill with median price (robust to outliers)
            median_price = self.df['price'].median()
            self.df['price'].fillna(median_price, inplace=True)
            self.cleaning_report.append(
                f"[WARNING] Filled {missing_price} missing prices with median: ${median_price:.2f}"
            )
        
        # Remove negative or zero prices (data quality issue)
        invalid_prices = (self.df['price'] <= 0).sum()
        if invalid_prices > 0:
            self.df = self.df[self.df['price'] > 0]
            self.cleaning_report.append(f"[WARNING] Removed {invalid_prices} records with invalid prices")
        
        self.cleaning_report.append(
            f"[OK] Price cleaned: ${self.df['price'].min():.2f} - ${self.df['price'].max():.2f}"
        )
    
    def clean_reviews_count(self):
        """
        Clean reviews_count column:
        - Remove commas and other separators
        - Convert to integer
        - Handle missing values
        """
        print("[INFO] Cleaning reviews_count column...")
        
        # Convert to string first
        self.df['reviews_count'] = self.df['reviews_count'].astype(str)
        
        # Remove commas and other non-digit characters
        self.df['reviews_count'] = self.df['reviews_count'].apply(
            lambda x: re.sub(r'[^\d]', '', x) if pd.notna(x) else '0'
        )
        
        # Convert to integer
        self.df['reviews_count'] = pd.to_numeric(self.df['reviews_count'], errors='coerce').fillna(0).astype(int)
        
        # Handle negative values
        negative_reviews = (self.df['reviews_count'] < 0).sum()
        if negative_reviews > 0:
            self.df.loc[self.df['reviews_count'] < 0, 'reviews_count'] = 0
            self.cleaning_report.append(f"[WARNING] Fixed {negative_reviews} negative review counts")
        
        self.cleaning_report.append(
            f"[OK] Reviews cleaned: {self.df['reviews_count'].min()} - {self.df['reviews_count'].max():,}"
        )
    
    def clean_rating(self):
        """
        Clean rating column:
        - Convert to float
        - Ensure values are between 0 and 5
        - Handle missing values
        """
        print("[INFO] Cleaning rating column...")
        
        # Convert to numeric
        self.df['rating'] = pd.to_numeric(self.df['rating'], errors='coerce')
        
        # Handle missing ratings
        missing_ratings = self.df['rating'].isna().sum()
        if missing_ratings > 0:
            median_rating = self.df['rating'].median()
            self.df['rating'].fillna(median_rating, inplace=True)
            self.cleaning_report.append(
                f"[WARNING] Filled {missing_ratings} missing ratings with median: {median_rating:.1f}"
            )
        
        # Ensure ratings are in valid range (0-5)
        invalid_ratings = ((self.df['rating'] < 0) | (self.df['rating'] > 5)).sum()
        if invalid_ratings > 0:
            self.df.loc[self.df['rating'] < 0, 'rating'] = 0
            self.df.loc[self.df['rating'] > 5, 'rating'] = 5
            self.cleaning_report.append(f"[WARNING] Fixed {invalid_ratings} out-of-range ratings")
        
        self.cleaning_report.append(
            f"[OK] Rating cleaned: {self.df['rating'].min():.1f} - {self.df['rating'].max():.1f}"
        )
    
    def clean_genre(self):
        """
        Standardize genre text:
        - Convert to lowercase
        - Strip whitespace
        - Handle missing values
        """
        print("[INFO] Cleaning genre column...")
        
        # Convert to string and standardize
        self.df['genre'] = self.df['genre'].astype(str).str.strip().str.lower()
        
        # Replace common variations
        genre_mapping = {
            'non-fiction': 'nonfiction',
            'non fiction': 'nonfiction',
            'sci-fi': 'science fiction',
            'scifi': 'science fiction',
        }
        self.df['genre'] = self.df['genre'].replace(genre_mapping)
        
        # Handle missing genres
        missing_genres = (self.df['genre'] == 'nan') | (self.df['genre'] == '')
        if missing_genres.sum() > 0:
            self.df.loc[missing_genres, 'genre'] = 'unknown'
            self.cleaning_report.append(f"[WARNING] Filled {missing_genres.sum()} missing genres with 'unknown'")
        
        unique_genres = self.df['genre'].nunique()
        self.cleaning_report.append(f"[OK] Genre standardized: {unique_genres} unique genres")
    
    def clean_text_columns(self):
        """
        Clean text columns (title, author, publisher if available):
        - Strip whitespace
        - Handle missing values
        - Standardize format
        """
        print("[INFO] Cleaning text columns...")
        
        # Clean title
        self.df['title'] = self.df['title'].astype(str).str.strip()
        missing_titles = (self.df['title'] == 'nan') | (self.df['title'] == '')
        if missing_titles.sum() > 0:
            self.df = self.df[~missing_titles]
            self.cleaning_report.append(f"[WARNING] Removed {missing_titles.sum()} records with missing titles")
        
        # Clean author
        self.df['author'] = self.df['author'].astype(str).str.strip()
        missing_authors = (self.df['author'] == 'nan') | (self.df['author'] == '')
        if missing_authors.sum() > 0:
            self.df.loc[missing_authors, 'author'] = 'Unknown Author'
            self.cleaning_report.append(f"[WARNING] Filled {missing_authors.sum()} missing authors")
        
        # Clean publisher if available
        if 'publisher' in self.df.columns:
            self.df['publisher'] = self.df['publisher'].astype(str).str.strip()
            missing_publishers = (self.df['publisher'] == 'nan') | (self.df['publisher'] == '')
            if missing_publishers.sum() > 0:
                self.df.loc[missing_publishers, 'publisher'] = 'Unknown Publisher'
        
        self.cleaning_report.append("[OK] Text columns cleaned")
    
    def clean_year(self):
        """
        Clean year column if available:
        - Convert to integer
        - Validate reasonable range
        - Handle missing values
        """
        if 'year' not in self.df.columns:
            return
        
        print("[INFO] Cleaning year column...")
        
        # Convert to numeric
        self.df['year'] = pd.to_numeric(self.df['year'], errors='coerce')
        
        # Validate year range (1900 - current year + 1)
        current_year = 2026  # Can be updated dynamically
        invalid_years = ((self.df['year'] < 1900) | (self.df['year'] > current_year + 1)).sum()
        
        if invalid_years > 0:
            median_year = self.df['year'].median()
            self.df.loc[(self.df['year'] < 1900) | (self.df['year'] > current_year + 1), 'year'] = median_year
            self.cleaning_report.append(f"[WARNING] Fixed {invalid_years} invalid years")
        
        # Fill missing years with median
        missing_years = self.df['year'].isna().sum()
        if missing_years > 0:
            median_year = self.df['year'].median()
            self.df['year'].fillna(median_year, inplace=True)
            self.cleaning_report.append(f"[WARNING] Filled {missing_years} missing years with median")
        
        self.df['year'] = self.df['year'].astype(int)
        self.cleaning_report.append(f"[OK] Year cleaned: {self.df['year'].min()} - {self.df['year'].max()}")
    
    def remove_duplicates(self):
        """
        Remove duplicate records based on title and author.
        Keep the first occurrence.
        """
        print("[INFO] Removing duplicates...")
        
        original_count = len(self.df)
        self.df.drop_duplicates(subset=['title', 'author'], keep='first', inplace=True)
        duplicates_removed = original_count - len(self.df)
        
        if duplicates_removed > 0:
            self.cleaning_report.append(f"[WARNING] Removed {duplicates_removed} duplicate records")
        else:
            self.cleaning_report.append("[OK] No duplicates found")
    
    def generate_warnings(self):
        """
        Generate automatic warnings for data quality issues.
        """
        print("\n" + "="*60)
        print("DATA QUALITY WARNINGS")
        print("="*60)
        
        warnings_list = []
        
        # Small dataset warning
        if len(self.df) < 100:
            warning = f"[WARNING] SMALL DATASET: Only {len(self.df)} samples. Model performance may be limited."
            warnings_list.append(warning)
            print(warning)
        
        # Skewed distribution warning
        price_skew = self.df['price'].skew()
        if abs(price_skew) > 1:
            warning = f"[WARNING] SKEWED PRICE DISTRIBUTION: Skewness = {price_skew:.2f}. Consider log transformation."
            warnings_list.append(warning)
            print(warning)
        
        reviews_skew = self.df['reviews_count'].skew()
        if abs(reviews_skew) > 1:
            warning = f"[WARNING] SKEWED REVIEWS DISTRIBUTION: Skewness = {reviews_skew:.2f}. Log transformation recommended."
            warnings_list.append(warning)
            print(warning)
        
        # Missing important features
        if 'year' not in self.df.columns:
            warning = "[WARNING] MISSING FEATURE: 'year' column not available. Temporal analysis limited."
            warnings_list.append(warning)
            print(warning)
        
        if 'publisher' not in self.df.columns:
            warning = "[WARNING] MISSING FEATURE: 'publisher' column not available. Publisher analysis not possible."
            warnings_list.append(warning)
            print(warning)
        
        # Genre imbalance
        genre_counts = self.df['genre'].value_counts()
        if len(genre_counts) > 0:
            max_genre_pct = (genre_counts.iloc[0] / len(self.df)) * 100
            if max_genre_pct > 50:
                warning = f"[WARNING] GENRE IMBALANCE: {genre_counts.index[0]} represents {max_genre_pct:.1f}% of data."
                warnings_list.append(warning)
                print(warning)
        
        if not warnings_list:
            print("[OK] No major data quality issues detected")
        
        print("="*60 + "\n")
        
        return warnings_list
    
    def save_cleaned_data(self):
        """Save cleaned dataset to CSV file."""
        try:
            # Create output directory if it doesn't exist
            output_dir = Path(self.output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self.df.to_csv(self.output_path, index=False)
            self.cleaning_report.append(f"[OK] Saved cleaned data to {self.output_path}")
            print(f"[SUCCESS] Cleaned data saved: {len(self.df)} records")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to save cleaned data: {str(e)}")
            return False
    
    def print_summary(self):
        """Print cleaning summary report."""
        print("\n" + "="*60)
        print("DATA CLEANING SUMMARY")
        print("="*60)
        for item in self.cleaning_report:
            print(item)
        print("="*60 + "\n")
    
    def run_pipeline(self):
        """
        Execute complete data cleaning pipeline.
        
        Returns:
            bool: True if successful, False otherwise
        """
        print("\n" + "="*60)
        print("STARTING DATA CLEANING PIPELINE")
        print("="*60 + "\n")
        
        # Load data
        if not self.load_data():
            return False
        
        # Validate columns
        if not self.validate_required_columns():
            return False
        
        # Clean each column
        self.clean_price()
        self.clean_reviews_count()
        self.clean_rating()
        self.clean_genre()
        self.clean_text_columns()
        self.clean_year()
        
        # Remove duplicates
        self.remove_duplicates()
        
        # Generate warnings
        self.generate_warnings()
        
        # Save cleaned data
        if not self.save_cleaned_data():
            return False
        
        # Print summary
        self.print_summary()
        
        return True


def main():
    """Main execution function."""
    # Define paths
    input_path = "data/raw.csv"
    output_path = "data/cleaned.csv"
    
    # Initialize and run cleaner
    cleaner = DataCleaner(input_path, output_path)
    success = cleaner.run_pipeline()
    
    if success:
        print("[SUCCESS] Data cleaning completed successfully!")
    else:
        print("[FAILED] Data cleaning encountered errors.")
    
    return success


if __name__ == "__main__":
    main()
