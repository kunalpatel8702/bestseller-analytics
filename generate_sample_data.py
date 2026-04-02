"""
Enhanced Sample Data Generator for Advanced Analytics

This script creates a highly realistic dataset with:
- Text-rich titles (for NLP)
- Sales rank (for success prediction)
- Complex price-success relationships (causality)
- Anomalies and outliers (for detection)
- Duplicates (for cleaning)

Author: Kunal Patel
"""

import pandas as pd
import numpy as np
import random
from pathlib import Path

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

NUM_SAMPLES = 1000  # Increased for better modeling
OUTPUT_FILE = Path("data/raw.csv")

# Semantic components for title generation
TITLE_PREFIXES = [
    "The Ultimate", "Complete", "Essential", "Advanced", "Modern", "Beginner's",
    "Hidden", "Lost", "Secret", "Forgotten", "Dark", "Bright", "Future", "Past"
]

TITLE_TOPICS = [
    "Python", "Machine Learning", "Data Science", "AI", "Cooking", "Baking",
    "Gardening", "Meditation", "Fitness", "History", "War", "Love", "Mystery",
    "Space", "Time", "Money", "Business", "Marketing", "Leadership", "Design"
]

TITLE_SUFFIXES = [
    "Guide", "Handbook", "Bible", "Masterclass", "Story", "Chronicles",
    "Strategies", "Secrets", "Principles", "Methods", "System", "Framework",
    "Workbook", "Journal", "Collection", "Anthology", "Saga", "Trilogy"
]

AUTHORS = [
    "James Clear", "Michelle Obama", "Colleen Hoover", "Robert Kiyosaki", 
    "Stephen King", "J.K. Rowling", "Malcolm Gladwell", "Yuval Noah Harari",
    "Brene Brown", "Simon Sinek", "Tim Ferriss", "Dale Carnegie",
    "Agatha Christie", "Neil Gaiman", "George R.R. Martin", 
    "Walter Isaacson", "Tara Westover", "Delia Owens", "Alex Michaelides",
    "Brand New Author"  # For cold-start testing
]

GENRES = [
    "Non-Fiction", "Fiction", "Business", "Self-Help", "Science", 
    "Technology", "Cooking", "History", "Romance", "Mystery", 
    "Fantasy", "Science Fiction", "Biography", "Health"
]

PUBLISHERS = [
    "Penguin Random House", "HarperCollins", "Simon & Schuster", 
    "Macmillan", "Hachette", "Indie Pub", "Amazon KDP", "O'Reilly Media"
]

def generate_title(genre):
    """Generate a realistic title based on genre."""
    if random.random() < 0.7:
        # Standard structure: Prefix Topic Suffix
        return f"{random.choice(TITLE_PREFIXES)} {random.choice(TITLE_TOPICS)} {random.choice(TITLE_SUFFIXES)}"
    elif random.random() < 0.9:
        # Simple: Topic Suffix
        return f"{random.choice(TITLE_TOPICS)} {random.choice(TITLE_SUFFIXES)}"
    else:
        # Complex: Prefix Topic: Suffix
        return f"{random.choice(TITLE_PREFIXES)} {random.choice(TITLE_TOPICS)}: The {random.choice(TITLE_SUFFIXES)}"

def generate_dataset():
    """Generate the synthetic dataset."""
    print(f"Generating {NUM_SAMPLES} enhanced book records...")
    
    data = []
    
    for _ in range(NUM_SAMPLES):
        genre = random.choice(GENRES)
        author = random.choice(AUTHORS)
        publisher = random.choice(PUBLISHERS)
        year = random.randint(2010, 2025)
        
        # Title with embedded signals
        title = generate_title(genre)
        
        # Base price logic
        base_price = 15.0
        if genre in ["Technology", "Business", "Science"]:
            base_price += 10.0
        if publisher == "O'Reilly Media":
            base_price += 15.0
        if "Ultimate" in title or "Bible" in title:
            base_price += 5.0
        
        # Add random variation
        price = max(4.99, np.random.normal(base_price, 5.0))
        
        # Rank Logic (Lower is better)
        # Driven by: Author reputation, recent year, reasonable price
        rank_score = (
            (100 if author in AUTHORS[:5] else 50) + 
            (20 if year >= 2023 else 0) +
            (30 if "Guide" in title else 0)
        )
        # Price elasticity: very high/low price hurts rank
        if price > 40 or price < 7:
            rank_score -= 20
        
        # Add randomness
        rank_score += np.random.normal(0, 20)
        rank_score = max(1, rank_score)
        
        # Convert score to rank (inverse relation)
        sales_rank = max(1, int(100000 / rank_score))
        
        # Reviews count correlated with rank
        reviews_count = int(max(0, np.random.lognormal(mean=np.log(rank_score * 10), sigma=1.0)))
        
        # Rating (biased high for bestsellers)
        rating = min(5.0, max(1.0, np.random.normal(4.2, 0.5)))
        if sales_rank < 1000:
            rating = min(5.0, rating + 0.3)
            
        data.append({
            'title': title,
            'author': author,
            'genre': genre,
            'publisher': publisher,
            'price': round(price, 2),
            'rating': round(rating, 1),
            'reviews_count': reviews_count,
            'sales_rank': sales_rank,  # NEW FEATURE
            'year': year
        })
    
    df = pd.DataFrame(data)
    
    # Introduce Data Quality Issues (for point #7)
    
    # 1. Missing values
    mask = np.random.choice([True, False], size=len(df), p=[0.05, 0.95])
    df.loc[mask, 'publisher'] = np.nan
    
    # 2. Outliers
    outlier_idx = np.random.choice(df.index, size=5, replace=False)
    df.loc[outlier_idx, 'price'] = 500.0  # Extreme price
    
    # 3. Duplicates
    duplicates = df.sample(n=20)
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")
    print("\nSample Data:")
    print(df.head())

if __name__ == "__main__":
    generate_dataset()
