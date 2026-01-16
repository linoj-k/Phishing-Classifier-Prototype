"""
Prediction Interface for Phishing URL Classifier

Classify raw URLs as phishing or legitimate using the trained Random Forest model.
Automatically extracts features from URLs and makes predictions.
"""

import sys
import argparse
from phishing_classifier import PhishingClassifier
from url_features import URLFeatureExtractor
import numpy as np


def classify_url(url, classifier, extractor):
    """
    Classify a single URL
    
    Args:
        url: Raw URL string
        classifier: Trained PhishingClassifier instance
        extractor: URLFeatureExtractor instance
        
    Returns:
        prediction: 0 (legitimate) or 1 (phishing)
        probability: Phishing probability (0-1)
    """
    # Extract features
    features = extractor.extract_as_array(url)
    features_array = np.array(features).reshape(1, -1)
    
    # Make prediction
    prediction, probability = classifier.predict(features_array)
    
    return prediction[0], probability[0]


def print_result(url, prediction, probability):
    """Print classification result in a formatted way"""
    print("\n" + "="*70)
    print(f"URL: {url}")
    print("-"*70)
    
    if prediction == 1:
        status = "⚠️  PHISHING"
        color = "RED"
    else:
        status = "✓  LEGITIMATE"
        color = "GREEN"
    
    print(f"Classification: {status}")
    print(f"Phishing Probability: {probability:.2%}")
    
    # Risk level
    if probability >= 0.8:
        risk = "VERY HIGH"
    elif probability >= 0.6:
        risk = "HIGH"
    elif probability >= 0.4:
        risk = "MODERATE"
    elif probability >= 0.2:
        risk = "LOW"
    else:
        risk = "VERY LOW"
    
    print(f"Risk Level: {risk}")
    print("="*70)


def classify_batch(urls, classifier, extractor):
    """
    Classify multiple URLs
    
    Args:
        urls: List of URL strings
        classifier: Trained PhishingClassifier instance
        extractor: URLFeatureExtractor instance
    """
    print(f"\nClassifying {len(urls)} URLs...")
    print("="*70)
    
    results = []
    for i, url in enumerate(urls, 1):
        try:
            prediction, probability = classify_url(url, classifier, extractor)
            results.append({
                'url': url,
                'prediction': prediction,
                'probability': probability
            })
            
            status = "PHISHING" if prediction == 1 else "LEGITIMATE"
            print(f"{i}. {status:12} ({probability:.2%}) - {url[:50]}...")
            
        except Exception as e:
            print(f"{i}. ERROR - {url[:50]}... ({str(e)})")
    
    print("="*70)
    
    # Summary
    if results:
        phishing_count = sum(1 for r in results if r['prediction'] == 1)
        legitimate_count = len(results) - phishing_count
        
        print(f"\nSummary:")
        print(f"  Phishing: {phishing_count}/{len(results)} ({phishing_count/len(results)*100:.1f}%)")
        print(f"  Legitimate: {legitimate_count}/{len(results)} ({legitimate_count/len(results)*100:.1f}%)")
    
    return results


def main():
    """Main prediction function"""
    parser = argparse.ArgumentParser(
        description='Classify URLs as phishing or legitimate',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py --url "http://example.com"
  python predict.py --batch urls.txt
  python predict.py --test
        """
    )
    
    parser.add_argument('--url', type=str, help='Single URL to classify')
    parser.add_argument('--batch', type=str, help='File containing URLs (one per line)')
    parser.add_argument('--test', action='store_true', help='Run test with sample URLs')
    
    args = parser.parse_args()
    
    # Check if model exists
    try:
        print("Loading trained model...")
        classifier = PhishingClassifier()
        classifier.load_model()
        
        print("Initializing feature extractor...")
        extractor = URLFeatureExtractor()
        
        print("✓ Model and feature extractor loaded successfully!\n")
        
    except FileNotFoundError:
        print("\n❌ Error: Trained model not found!")
        print("Please run 'python train_model.py' first to train the model.")
        sys.exit(1)
    
    # Single URL classification
    if args.url:
        prediction, probability = classify_url(args.url, classifier, extractor)
        print_result(args.url, prediction, probability)
    
    # Batch classification
    elif args.batch:
        try:
            with open(args.batch, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
            
            if not urls:
                print(f"Error: No URLs found in {args.batch}")
                sys.exit(1)
            
            classify_batch(urls, classifier, extractor)
            
        except FileNotFoundError:
            print(f"Error: File not found: {args.batch}")
            sys.exit(1)
    
    # Test mode
    elif args.test:
        print("Running test with sample URLs...\n")
        
        test_urls = [
            "https://www.google.com",
            "https://www.github.com",
            "http://phishing-example-123-login.com/secure/account",
            "https://www.paypal.com-verify-account.suspicious-domain.com",
            "https://www.amazon.com",
            "http://192.168.1.1/admin/login.php",
            "https://secure.bank-login-verify.tk/account/update"
        ]
        
        classify_batch(test_urls, classifier, extractor)
    
    # No arguments provided
    else:
        parser.print_help()
        print("\n💡 Tip: Use --url to classify a single URL or --test to run sample tests")


if __name__ == "__main__":
    main()
