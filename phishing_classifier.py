"""
Phishing URL Classifier using Random Forest

Core module for training and using a Random Forest classifier to detect phishing URLs.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import joblib
import os


class PhishingClassifier:
    """Random Forest classifier for phishing URL detection"""
    
    def __init__(self, n_estimators=100, random_state=42):
        """
        Initialize the classifier
        
        Args:
            n_estimators: Number of trees in the forest
            random_state: Random seed for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = None
        
    def load_data(self, csv_path):
        """
        Load dataset from CSV file
        
        Args:
            csv_path: Path to the dataset CSV file
            
        Returns:
            X: Feature matrix
            y: Target labels
        """
        print(f"Loading dataset from: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Separate features and target
        X = df.drop('Type', axis=1)
        y = df['Type']
        
        self.feature_names = X.columns.tolist()
        
        print(f"Dataset loaded: {len(df)} samples, {len(X.columns)} features")
        print(f"Phishing URLs: {sum(y == 1)} ({sum(y == 1)/len(y)*100:.1f}%)")
        print(f"Legitimate URLs: {sum(y == 0)} ({sum(y == 0)/len(y)*100:.1f}%)")
        
        return X, y
    
    def preprocess_data(self, X_train, X_test):
        """
        Preprocess features using standardization
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Returns:
            X_train_scaled: Scaled training features
            X_test_scaled: Scaled test features
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def train(self, X_train, y_train):
        """
        Train the Random Forest classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        print("\nTraining Random Forest classifier...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("Training completed!")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'y_test': y_test
        }
        
        return metrics
    
    def get_feature_importance(self):
        """
        Get feature importance scores
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        })
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
    
    def predict(self, X):
        """
        Predict phishing URLs
        
        Args:
            X: Feature matrix (can be a single sample or batch)
            
        Returns:
            predictions: Array of predictions (0=legitimate, 1=phishing)
            probabilities: Array of phishing probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Ensure X is 2D
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        return predictions, probabilities
    
    def save_model(self, model_path='phishing_model.pkl', scaler_path='scaler.pkl'):
        """
        Save trained model and scaler to disk
        
        Args:
            model_path: Path to save the model
            scaler_path: Path to save the scaler
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.feature_names, 'feature_names.pkl')
        
        print(f"\nModel saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")
        print(f"Feature names saved to: feature_names.pkl")
    
    def load_model(self, model_path='phishing_model.pkl', scaler_path='scaler.pkl'):
        """
        Load trained model and scaler from disk
        
        Args:
            model_path: Path to the saved model
            scaler_path: Path to the saved scaler
        """
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError("Model or scaler file not found")
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        if os.path.exists('feature_names.pkl'):
            self.feature_names = joblib.load('feature_names.pkl')
        
        self.is_trained = True
        
        print(f"Model loaded from: {model_path}")
        print(f"Scaler loaded from: {scaler_path}")
    
    def print_metrics(self, metrics):
        """
        Print evaluation metrics in a formatted way
        
        Args:
            metrics: Dictionary of evaluation metrics
        """
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        
        print(f"\nAccuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        
        print("\nClassification Report:")
        print(metrics['classification_report'])
        
        print("="*50)


# Example usage
if __name__ == "__main__":
    # Initialize classifier
    classifier = PhishingClassifier(n_estimators=100)
    
    # Example of how to use the classifier
    print("Phishing URL Classifier initialized")
    print("Use train_model.py to train the classifier")
