"""
Training Script for Phishing URL Classifier

Trains a Random Forest classifier using the phishing detection dataset,
evaluates performance, generates visualizations, and saves the model.
"""

import sys
import os
from phishing_classifier import PhishingClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_confusion_matrix(cm, save_path='confusion_matrix.png'):
    """Plot and save confusion matrix heatmap"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legitimate', 'Phishing'],
                yticklabels=['Legitimate', 'Phishing'])
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {save_path}")
    plt.close()


def plot_roc_curve(y_test, y_pred_proba, roc_auc, save_path='roc_curve.png'):
    """Plot and save ROC curve"""
    from sklearn.metrics import roc_curve
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to: {save_path}")
    plt.close()


def plot_feature_importance(importance_df, top_n=20, save_path='feature_importance.png'):
    """Plot and save top N most important features"""
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n} Most Important Features', fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Feature importance plot saved to: {save_path}")
    plt.close()


def main():
    """Main training function"""
    print("="*70)
    print("PHISHING URL CLASSIFIER - TRAINING")
    print("="*70)
    
    # Dataset path
    dataset_path = "Phishing Detection Dataset/Phishing Detection Dataset/Dataset.csv"
    
    if not os.path.exists(dataset_path):
        print(f"\nError: Dataset not found at {dataset_path}")
        print("Please ensure the dataset is in the correct location.")
        sys.exit(1)
    
    # Initialize classifier
    print("\n1. Initializing Random Forest Classifier...")
    classifier = PhishingClassifier(n_estimators=100, random_state=42)
    
    # Load data
    print("\n2. Loading dataset...")
    X, y = classifier.load_data(dataset_path)
    
    # Split data
    print("\n3. Splitting data into train/test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Preprocess data
    print("\n4. Preprocessing features (standardization)...")
    X_train_scaled, X_test_scaled = classifier.preprocess_data(X_train, X_test)
    
    # Train model
    print("\n5. Training Random Forest model...")
    print("This may take a few minutes...")
    classifier.train(X_train_scaled, y_train)
    
    # Evaluate model
    print("\n6. Evaluating model performance...")
    metrics = classifier.evaluate(X_test_scaled, y_test)
    
    # Print metrics
    classifier.print_metrics(metrics)
    
    # Generate visualizations
    print("\n7. Generating performance visualizations...")
    
    # Confusion Matrix
    plot_confusion_matrix(metrics['confusion_matrix'])
    
    # ROC Curve
    plot_roc_curve(metrics['y_test'], metrics['y_pred_proba'], metrics['roc_auc'])
    
    # Feature Importance
    importance_df = classifier.get_feature_importance()
    plot_feature_importance(importance_df, top_n=20)
    
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))
    
    # Save model
    print("\n8. Saving trained model...")
    classifier.save_model()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nModel files created:")
    print("  - phishing_model.pkl")
    print("  - scaler.pkl")
    print("  - feature_names.pkl")
    print("\nVisualization files created:")
    print("  - confusion_matrix.png")
    print("  - roc_curve.png")
    print("  - feature_importance.png")
    print("\nYou can now use predict.py to classify URLs!")
    print("="*70)


if __name__ == "__main__":
    main()
