"""
ROC Curve Verification Script

This script allows you to verify the ROC curve by:
1. Recalculating the ROC curve from model predictions
2. Displaying detailed metrics
3. Comparing with the saved curve
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, auc
from phishing_classifier import PhishingClassifier
from sklearn.model_selection import train_test_split
import pandas as pd


def verify_roc_curve():
    """Verify the ROC curve by recalculating from the model"""
    
    print("="*70)
    print("ROC CURVE VERIFICATION")
    print("="*70)
    
    # Load the trained model
    print("\n1. Loading trained model...")
    classifier = PhishingClassifier()
    classifier.load_model()
    
    # Load the dataset
    print("2. Loading dataset...")
    dataset_path = "Phishing Detection Dataset/Phishing Detection Dataset/Dataset.csv"
    X, y = classifier.load_data(dataset_path)
    
    # Split data (same as training)
    print("3. Splitting data (same 80/20 split as training)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Preprocess
    print("4. Preprocessing features...")
    X_train_scaled, X_test_scaled = classifier.preprocess_data(X_train, X_test)
    
    # Get predictions
    print("5. Getting model predictions on test set...")
    y_pred_proba = classifier.model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate ROC curve
    print("6. Calculating ROC curve...")
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Alternative AUC calculation
    auc_alternative = auc(fpr, tpr)
    
    print("\n" + "="*70)
    print("ROC CURVE METRICS")
    print("="*70)
    
    print(f"\n📊 ROC-AUC Score: {roc_auc:.6f}")
    print(f"📊 AUC (alternative calculation): {auc_alternative:.6f}")
    print(f"\n📈 Number of threshold points: {len(thresholds)}")
    print(f"📈 FPR range: [{fpr.min():.6f}, {fpr.max():.6f}]")
    print(f"📈 TPR range: [{tpr.min():.6f}, {tpr.max():.6f}]")
    
    # Show key statistics
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    if roc_auc >= 0.95:
        performance = "EXCELLENT"
        emoji = "🌟"
    elif roc_auc >= 0.90:
        performance = "VERY GOOD"
        emoji = "✨"
    elif roc_auc >= 0.80:
        performance = "GOOD"
        emoji = "👍"
    elif roc_auc >= 0.70:
        performance = "FAIR"
        emoji = "👌"
    else:
        performance = "NEEDS IMPROVEMENT"
        emoji = "⚠️"
    
    print(f"\n{emoji} Model Performance: {performance}")
    print(f"\nAn AUC of {roc_auc:.4f} means:")
    print(f"  • The model has {roc_auc*100:.2f}% probability of correctly ranking")
    print(f"    a random phishing URL higher than a random legitimate URL")
    print(f"  • {(roc_auc - 0.5) / 0.5 * 100:.1f}% better than random guessing (50%)")
    
    # Show some threshold examples
    print("\n" + "="*70)
    print("SAMPLE THRESHOLD POINTS")
    print("="*70)
    
    # Find specific operating points
    indices = [0, len(fpr)//4, len(fpr)//2, 3*len(fpr)//4, -1]
    
    print(f"\n{'Threshold':<12} {'FPR':<12} {'TPR':<12} {'Description'}")
    print("-" * 70)
    
    for i in indices:
        desc = ""
        if i == 0:
            desc = "(Most lenient)"
        elif i == -1:
            desc = "(Most strict)"
        elif i == len(fpr)//2:
            desc = "(Balanced)"
        
        print(f"{thresholds[i]:<12.4f} {fpr[i]:<12.4f} {tpr[i]:<12.4f} {desc}")
    
    # Find optimal threshold (closest to top-left corner)
    optimal_idx = np.argmax(tpr - fpr)
    print(f"\n🎯 Optimal threshold: {thresholds[optimal_idx]:.4f}")
    print(f"   FPR: {fpr[optimal_idx]:.4f} | TPR: {tpr[optimal_idx]:.4f}")
    
    # Plot verification curve
    print("\n7. Generating verification plot...")
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve
    plt.subplot(2, 1, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier (AUC = 0.50)')
    plt.scatter([fpr[optimal_idx]], [tpr[optimal_idx]], color='red', s=100, 
                zorder=5, label=f'Optimal point')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Verification', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    # Plot threshold distribution
    plt.subplot(2, 1, 2)
    plt.hist(y_pred_proba[y_test == 0], bins=50, alpha=0.5, 
             label='Legitimate URLs', color='green')
    plt.hist(y_pred_proba[y_test == 1], bins=50, alpha=0.5, 
             label='Phishing URLs', color='red')
    plt.axvline(thresholds[optimal_idx], color='black', linestyle='--', 
                label=f'Optimal threshold ({thresholds[optimal_idx]:.3f})')
    plt.xlabel('Predicted Probability (Phishing)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Prediction Distribution by Class', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('roc_curve_verification.png', dpi=300, bbox_inches='tight')
    print("✅ Verification plot saved: roc_curve_verification.png")
    
    # Additional validation checks
    print("\n" + "="*70)
    print("VALIDATION CHECKS")
    print("="*70)
    
    checks_passed = 0
    total_checks = 5
    
    # Check 1: AUC should be between 0.5 and 1.0
    if 0.5 <= roc_auc <= 1.0:
        print("✅ Check 1: AUC is in valid range [0.5, 1.0]")
        checks_passed += 1
    else:
        print("❌ Check 1: FAILED - AUC out of range")
    
    # Check 2: ROC curve should start at (0,0)
    if abs(fpr[0]) < 0.01 and abs(tpr[0]) < 0.01:
        print("✅ Check 2: ROC curve starts at origin (0,0)")
        checks_passed += 1
    else:
        print("❌ Check 2: FAILED - ROC curve doesn't start at origin")
    
    # Check 3: ROC curve should end at (1,1)
    if abs(fpr[-1] - 1.0) < 0.01 and abs(tpr[-1] - 1.0) < 0.01:
        print("✅ Check 3: ROC curve ends at (1,1)")
        checks_passed += 1
    else:
        print("❌ Check 3: FAILED - ROC curve doesn't end at (1,1)")
    
    # Check 4: TPR should be monotonically increasing
    if np.all(np.diff(tpr) >= -1e-10):  # Allow tiny numerical errors
        print("✅ Check 4: TPR is monotonically increasing")
        checks_passed += 1
    else:
        print("❌ Check 4: FAILED - TPR not monotonically increasing")
    
    # Check 5: Curve should be above diagonal
    above_diagonal = np.mean(tpr - fpr > 0)
    if above_diagonal > 0.9:
        print(f"✅ Check 5: Curve is {above_diagonal*100:.1f}% above random baseline")
        checks_passed += 1
    else:
        print(f"❌ Check 5: FAILED - Only {above_diagonal*100:.1f}% above baseline")
    
    print(f"\n{'='*70}")
    print(f"VALIDATION SUMMARY: {checks_passed}/{total_checks} checks passed")
    print("="*70)
    
    if checks_passed == total_checks:
        print("\n🎉 All validation checks passed! The ROC curve is VALID.")
    else:
        print(f"\n⚠️  {total_checks - checks_passed} check(s) failed. Please review.")
    
    return roc_auc, fpr, tpr, thresholds


if __name__ == "__main__":
    verify_roc_curve()
