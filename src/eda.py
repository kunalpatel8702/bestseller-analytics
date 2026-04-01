"""
Exploratory Data Analysis (EDA) Script for Amazon Bestselling Books

This script generates:
- Comprehensive visualizations
- Statistical summaries
- Top 10 insights automatically
- All plots saved to reports/visualizations/

Author: Senior Data Science Team
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Set plot style
sns.set_style('whitegrid')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class EDAAnalyzer:
    """
    Comprehensive EDA pipeline for book dataset.
    """
    
    def __init__(self, data_path: str, output_dir: str):
        """
        Initialize EDA Analyzer.
        
        Args:
            data_path (str): Path to cleaned CSV file
            output_dir (str): Directory to save outputs
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.viz_dir = self.output_dir / "visualizations"
        self.df = None
        self.insights = []
        
    def load_data(self):
        """Load cleaned data from CSV file."""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"[INFO] Loaded dataset: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            return True
        except FileNotFoundError:
            print(f"[ERROR] File not found: {self.data_path}")
            return False
        except Exception as e:
            print(f"[ERROR] Failed to load data: {str(e)}")
            return False
    
    def create_output_dirs(self):
        """Create output directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Output directory: {self.viz_dir}")
    
    def plot_price_distribution(self):
        """Plot price distribution."""
        print("[INFO] Creating price distribution plot...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(self.df['price'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0].axvline(self.df['price'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${self.df["price"].mean():.2f}')
        axes[0].axvline(self.df['price'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: ${self.df["price"].median():.2f}')
        axes[0].set_xlabel('Price ($)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Price Distribution', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Box plot
        axes[1].boxplot(self.df['price'], vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2))
        axes[1].set_ylabel('Price ($)', fontsize=12)
        axes[1].set_title('Price Box Plot', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'price_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate insight
        skewness = self.df['price'].skew()
        self.insights.append(
            f"Price distribution has a mean of ${self.df['price'].mean():.2f} and median of ${self.df['price'].median():.2f}. "
            f"Skewness of {skewness:.2f} indicates {'right-skewed' if skewness > 0 else 'left-skewed'} distribution."
        )
    
    def plot_rating_distribution(self):
        """Plot rating distribution."""
        print("[INFO] Creating rating distribution plot...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(self.df['rating'], bins=20, edgecolor='black', alpha=0.7, color='coral')
        axes[0].axvline(self.df['rating'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {self.df["rating"].mean():.2f}')
        axes[0].set_xlabel('Rating', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Rating Distribution', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Count plot by rating category
        rating_categories = pd.cut(self.df['rating'], bins=[0, 3.5, 4.2, 5.0], labels=['Low', 'Medium', 'High'])
        rating_counts = rating_categories.value_counts().sort_index()
        axes[1].bar(rating_counts.index, rating_counts.values, color=['#ff6b6b', '#ffd93d', '#6bcf7f'], edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Rating Category', fontsize=12)
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].set_title('Books by Rating Category', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'rating_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate insight
        high_rated = (self.df['rating'] >= 4.2).sum()
        pct_high_rated = (high_rated / len(self.df)) * 100
        self.insights.append(
            f"{high_rated} books ({pct_high_rated:.1f}%) have ratings >= 4.2, indicating strong overall quality among bestsellers."
        )
    
    def plot_genre_frequency(self):
        """Plot genre frequency."""
        print("[INFO] Creating genre frequency plot...")
        
        genre_counts = self.df['genre'].value_counts().head(10)
        
        plt.figure(figsize=(12, 6))
        bars = plt.barh(genre_counts.index, genre_counts.values, color='teal', edgecolor='black', alpha=0.7)
        plt.xlabel('Number of Books', fontsize=12)
        plt.ylabel('Genre', fontsize=12)
        plt.title('Top 10 Genres by Frequency', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2, f' {int(width)}', 
                    ha='left', va='center', fontsize=10)
        
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'genre_frequency.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate insight
        top_genre = genre_counts.index[0]
        top_genre_pct = (genre_counts.iloc[0] / len(self.df)) * 100
        self.insights.append(
            f"'{top_genre}' is the most popular genre with {genre_counts.iloc[0]} books ({top_genre_pct:.1f}% of dataset)."
        )
    
    def plot_price_vs_reviews(self):
        """Plot price vs reviews scatter plot."""
        print("[INFO] Creating price vs reviews plot...")
        
        plt.figure(figsize=(12, 6))
        plt.scatter(self.df['reviews_count'], self.df['price'], alpha=0.5, edgecolors='k', linewidths=0.5, c=self.df['rating'], cmap='viridis')
        plt.colorbar(label='Rating')
        plt.xlabel('Reviews Count', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.title('Price vs Reviews Count (colored by Rating)', fontsize=14, fontweight='bold')
        plt.xscale('log')  # Log scale for better visualization
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'price_vs_reviews.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate correlation
        correlation = self.df['price'].corr(self.df['reviews_count'])
        self.insights.append(
            f"Price and reviews count have a correlation of {correlation:.3f}, suggesting "
            f"{'positive' if correlation > 0 else 'negative'} relationship."
        )
    
    def plot_price_vs_rating(self):
        """Plot price vs rating scatter plot."""
        print("[INFO] Creating price vs rating plot...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot
        axes[0].scatter(self.df['rating'], self.df['price'], alpha=0.5, edgecolors='k', linewidths=0.5, color='purple')
        axes[0].set_xlabel('Rating', fontsize=12)
        axes[0].set_ylabel('Price ($)', fontsize=12)
        axes[0].set_title('Price vs Rating', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Box plot by rating category
        rating_categories = pd.cut(self.df['rating'], bins=[0, 3.5, 4.2, 5.0], labels=['Low', 'Medium', 'High'])
        data_by_category = [self.df[rating_categories == cat]['price'].values for cat in ['Low', 'Medium', 'High']]
        bp = axes[1].boxplot(data_by_category, labels=['Low', 'Medium', 'High'], patch_artist=True)
        for patch, color in zip(bp['boxes'], ['#ff6b6b', '#ffd93d', '#6bcf7f']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1].set_xlabel('Rating Category', fontsize=12)
        axes[1].set_ylabel('Price ($)', fontsize=12)
        axes[1].set_title('Price by Rating Category', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'price_vs_rating.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate correlation
        correlation = self.df['price'].corr(self.df['rating'])
        self.insights.append(
            f"Price and rating have a correlation of {correlation:.3f}. "
            f"{'Higher-rated books tend to be priced higher' if correlation > 0.1 else 'Rating has minimal impact on price'}."
        )
    
    def plot_correlation_heatmap(self):
        """Plot correlation heatmap for numeric features."""
        print("[INFO] Creating correlation heatmap...")
        
        # Select numeric columns
        numeric_cols = ['price', 'rating', 'reviews_count']
        if 'year' in self.df.columns:
            numeric_cols.append('year')
        
        # Calculate correlation matrix
        corr_matrix = self.df[numeric_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Find strongest correlation
        corr_with_price = corr_matrix['price'].drop('price').abs()
        strongest_feature = corr_with_price.idxmax()
        strongest_corr = corr_matrix.loc[strongest_feature, 'price']
        self.insights.append(
            f"'{strongest_feature}' has the strongest correlation with price ({strongest_corr:.3f})."
        )
    
    def plot_genre_price_analysis(self):
        """Plot average price by genre."""
        print("[INFO] Creating genre price analysis...")
        
        genre_price = self.df.groupby('genre')['price'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = range(len(genre_price))
        ax.bar(x, genre_price['mean'], yerr=genre_price['std'], capsize=5, 
               color='steelblue', edgecolor='black', alpha=0.7, error_kw={'linewidth': 2})
        ax.set_xticks(x)
        ax.set_xticklabels(genre_price.index, rotation=45, ha='right')
        ax.set_xlabel('Genre', fontsize=12)
        ax.set_ylabel('Average Price ($)', fontsize=12)
        ax.set_title('Average Price by Genre (Top 10)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'genre_price_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate insight
        highest_price_genre = genre_price.index[0]
        lowest_price_genre = genre_price.index[-1]
        self.insights.append(
            f"'{highest_price_genre}' has the highest average price (${genre_price.loc[highest_price_genre, 'mean']:.2f}), "
            f"while '{lowest_price_genre}' has the lowest (${genre_price.loc[lowest_price_genre, 'mean']:.2f})."
        )
    
    def plot_author_analysis(self):
        """Plot top authors by book count."""
        print("[INFO] Creating author analysis...")
        
        author_counts = self.df['author'].value_counts().head(10)
        
        plt.figure(figsize=(12, 6))
        bars = plt.barh(author_counts.index, author_counts.values, color='orange', edgecolor='black', alpha=0.7)
        plt.xlabel('Number of Bestsellers', fontsize=12)
        plt.ylabel('Author', fontsize=12)
        plt.title('Top 10 Authors by Bestseller Count', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2, f' {int(width)}', 
                    ha='left', va='center', fontsize=10)
        
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'author_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate insight
        top_author = author_counts.index[0]
        top_author_count = author_counts.iloc[0]
        self.insights.append(
            f"'{top_author}' has the most bestsellers with {top_author_count} books in the dataset."
        )
    
    def plot_year_trends(self):
        """Plot trends over years if year column exists."""
        if 'year' not in self.df.columns:
            print("[INFO] Skipping year trends (column not available)")
            return
        
        print("[INFO] Creating year trends plot...")
        
        year_stats = self.df.groupby('year').agg({
            'price': 'mean',
            'rating': 'mean',
            'title': 'count'
        }).rename(columns={'title': 'count'})
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Book count by year
        axes[0].bar(year_stats.index, year_stats['count'], color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Year', fontsize=12)
        axes[0].set_ylabel('Number of Books', fontsize=12)
        axes[0].set_title('Bestsellers by Publication Year', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Average price and rating by year
        ax2 = axes[1]
        ax2.plot(year_stats.index, year_stats['price'], marker='o', linewidth=2, color='green', label='Avg Price')
        ax2.set_xlabel('Year', fontsize=12)
        ax2.set_ylabel('Average Price ($)', fontsize=12, color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        ax2.grid(True, alpha=0.3)
        
        ax3 = ax2.twinx()
        ax3.plot(year_stats.index, year_stats['rating'], marker='s', linewidth=2, color='orange', label='Avg Rating')
        ax3.set_ylabel('Average Rating', fontsize=12, color='orange')
        ax3.tick_params(axis='y', labelcolor='orange')
        
        axes[1].set_title('Average Price and Rating by Year', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'year_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate insight
        recent_years = year_stats.loc[year_stats.index >= 2020]
        if len(recent_years) > 0:
            avg_recent_price = recent_years['price'].mean()
            self.insights.append(
                f"Books published since 2020 have an average price of ${avg_recent_price:.2f}."
            )
    
    def generate_statistical_summary(self):
        """Generate statistical summary."""
        print("[INFO] Generating statistical summary...")
        
        summary = self.df[['price', 'rating', 'reviews_count']].describe()
        
        # Additional insights
        self.insights.append(
            f"Dataset contains {len(self.df)} books across {self.df['genre'].nunique()} genres "
            f"from {self.df['author'].nunique()} authors."
        )
        
        # Price insights
        expensive_books = (self.df['price'] > self.df['price'].quantile(0.9)).sum()
        self.insights.append(
            f"Top 10% of books are priced above ${self.df['price'].quantile(0.9):.2f}, "
            f"representing {expensive_books} premium titles."
        )
    
    def save_insights(self):
        """Save top 10 insights to text file."""
        print("[INFO] Saving insights...")
        
        insights_path = self.output_dir / 'insights.txt'
        
        with open(insights_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("TOP 10 INSIGHTS: AMAZON BESTSELLING BOOKS\n")
            f.write("="*60 + "\n\n")
            
            for i, insight in enumerate(self.insights[:10], 1):
                f.write(f"{i}. {insight}\n\n")
            
            f.write("="*60 + "\n")
            f.write("END OF INSIGHTS\n")
            f.write("="*60 + "\n")
        
        print(f"[SUCCESS] Insights saved to {insights_path}")
    
    def run_pipeline(self):
        """Execute complete EDA pipeline."""
        print("\n" + "="*60)
        print("STARTING EXPLORATORY DATA ANALYSIS")
        print("="*60 + "\n")
        
        # Load data
        if not self.load_data():
            return False
        
        # Create output directories
        self.create_output_dirs()
        
        # Generate visualizations
        self.plot_price_distribution()
        self.plot_rating_distribution()
        self.plot_genre_frequency()
        self.plot_price_vs_reviews()
        self.plot_price_vs_rating()
        self.plot_correlation_heatmap()
        self.plot_genre_price_analysis()
        self.plot_author_analysis()
        self.plot_year_trends()
        
        # Generate statistical summary
        self.generate_statistical_summary()
        
        # Save insights
        self.save_insights()
        
        print("\n" + "="*60)
        print("EDA COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"\nVisualizations saved to: {self.viz_dir}")
        print(f"Insights saved to: {self.output_dir / 'insights.txt'}")
        print(f"\nTotal visualizations created: {len(list(self.viz_dir.glob('*.png')))}")
        print("="*60 + "\n")
        
        return True


def main():
    """Main execution function."""
    # Define paths
    data_path = "data/cleaned.csv"
    output_dir = "reports"
    
    # Initialize and run EDA
    analyzer = EDAAnalyzer(data_path, output_dir)
    success = analyzer.run_pipeline()
    
    if success:
        print("[SUCCESS] EDA completed successfully!")
    else:
        print("[FAILED] EDA encountered errors.")
    
    return success


if __name__ == "__main__":
    main()
