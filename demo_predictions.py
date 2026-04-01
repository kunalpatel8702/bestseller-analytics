"""
Demonstration Script: Price Prediction and Recommendations

This script demonstrates how to use the trained model for predictions.
"""

from src.predict import PricePredictor

def main():
    print("\n" + "="*70)
    print(" "*15 + "PRICE PREDICTION DEMO")
    print("="*70 + "\n")
    
    # Initialize predictor
    predictor = PricePredictor(
        model_path="models/price_model.pkl",
        feature_names_path="models/feature_names.pkl",
        data_path="data/cleaned.csv"
    )
    
    if not predictor.initialize():
        print("[ERROR] Failed to initialize predictor")
        return
    
    print("\n" + "="*70)
    print("EXAMPLE PREDICTIONS")
    print("="*70 + "\n")
    
    # Example books
    examples = [
        {
            'title': 'The Great Mystery Novel',
            'genre': 'mystery',
            'rating': 4.5,
            'reviews_count': 5000,
            'author': 'Jane Detective',
            'year': 2023
        },
        {
            'title': 'Romance in Paris',
            'genre': 'romance',
            'rating': 4.7,
            'reviews_count': 10000,
            'author': 'Love Writer',
            'year': 2024
        },
        {
            'title': 'History of Ancient Rome',
            'genre': 'history',
            'rating': 4.3,
            'reviews_count': 2000,
            'author': 'Professor Smith',
            'year': 2022
        },
        {
            'title': 'Sci-Fi Adventure',
            'genre': 'science fiction',
            'rating': 4.0,
            'reviews_count': 3000,
            'author': 'Future Author',
            'year': 2024
        }
    ]
    
    for i, book in enumerate(examples, 1):
        print(f"Book {i}: {book['title']}")
        print("-" * 70)
        print(f"  Genre: {book['genre']}")
        print(f"  Rating: {book['rating']}")
        print(f"  Reviews: {book['reviews_count']:,}")
        print(f"  Author: {book['author']}")
        print(f"  Year: {book['year']}")
        
        # Get prediction and recommendation
        predicted_price = predictor.predict_price(book)
        recommendation = predictor.get_pricing_recommendation(book, predicted_price)
        
        print(f"\n  PREDICTED PRICE: ${recommendation['predicted_price']:.2f}")
        print(f"  PRICING BAND: {recommendation['pricing_band']}")
        print(f"  SUGGESTED RANGE: ${recommendation['suggested_price_range']['min']:.2f} - ${recommendation['suggested_price_range']['max']:.2f}")
        print(f"  GENRE AVERAGE: ${recommendation['genre_average']:.2f}")
        print(f"\n  REASONING:")
        print(f"  {recommendation['reasoning']}")
        print("\n" + "="*70 + "\n")
    
    # Batch prediction
    print("="*70)
    print("BATCH PREDICTION SUMMARY")
    print("="*70 + "\n")
    
    results = predictor.batch_predict(examples)
    print(results.to_string(index=False))
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
