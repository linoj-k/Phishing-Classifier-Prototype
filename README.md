# Phishing URL Classifier

A Random Forest-based machine learning classifier that detects phishing URLs with high accuracy. The system automatically extracts 41 URL features and classifies raw URLs as either legitimate or phishing.

## 🎯 Features

- **Raw URL Classification**: Accepts raw URLs like `http://example.com/path` directly
- **Automatic Feature Extraction**: Extracts 41 features from URLs including:
  - URL structure analysis (length, components, special characters)
  - Domain and subdomain analysis
  - Digit patterns and repetition detection
  - Shannon entropy calculations
- **High Accuracy**: Trained on 247,951 URL samples
- **Performance Visualizations**: Generates confusion matrix, ROC curve, and feature importance plots
- **Batch Processing**: Classify multiple URLs at once
- **Command-line Interface**: Easy-to-use prediction interface

## 📦 Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Training the Model

Train the Random Forest classifier on the dataset:

```bash
python train_model.py
```

This will:
- Load and preprocess the dataset
- Train a Random Forest model with 100 trees
- Evaluate performance on test set
- Generate visualization plots
- Save the trained model

**Output files:**
- `phishing_model.pkl` - Trained Random Forest model
- `scaler.pkl` - Feature scaler
- `feature_names.pkl` - Feature name mappings
- `confusion_matrix.png` - Confusion matrix heatmap
- `roc_curve.png` - ROC curve plot
- `feature_importance.png` - Top 20 important features

### Making Predictions

#### Single URL Classification

```bash
python predict.py --url "http://example.com/path"
```

#### Batch Classification

Create a text file with URLs (one per line), then:

```bash
python predict.py --batch urls.txt
```

#### Test Mode

Run with sample URLs:

```bash
python predict.py --test
```

## 📊 Model Performance

The classifier achieves excellent performance on the test set:
- **High accuracy** for distinguishing phishing from legitimate URLs
- **ROC-AUC score** measuring classification quality
- **Precision and Recall** balanced for real-world usage

See generated visualization files for detailed performance metrics.

## 🔍 How It Works

### 1. Feature Extraction (`url_features.py`)

The `URLFeatureExtractor` class extracts 41 features from raw URLs:

```python
from url_features import URLFeatureExtractor

extractor = URLFeatureExtractor()
features = extractor.extract_features("http://example.com")
```

### 2. Classification (`phishing_classifier.py`)

The `PhishingClassifier` class provides:
- Data loading and preprocessing
- Random Forest training
- Model evaluation
- Prediction interface

### 3. Training Pipeline (`train_model.py`)

Automated training workflow:
1. Load dataset (247K+ samples)
2. Split into train/test (80/20)
3. Standardize features
4. Train Random Forest
5. Evaluate and visualize
6. Save model

### 4. Prediction Interface (`predict.py`)

User-friendly CLI for:
- Single URL predictions
- Batch processing
- Risk assessment

## 📂 Project Structure

```
Classifier Prototype/
├── url_features.py          # Feature extraction from URLs
├── phishing_classifier.py   # Random Forest classifier
├── train_model.py          # Training script
├── predict.py              # Prediction interface
├── requirements.txt        # Python dependencies
├── Phishing Detection Dataset/
│   └── Dataset.csv         # Training dataset (247K samples)
└── Output Files (after training):
    ├── phishing_model.pkl
    ├── scaler.pkl
    ├── feature_names.pkl
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── feature_importance.png
```

## 🛠️ Technical Details

### Dataset Features (41 total)

| Category | Features |
|----------|----------|
| **URL Structure** | Length, dots, digits, special characters |
| **Domain Analysis** | Length, subdomain count, character patterns |
| **Path/Query** | Presence, length, components |
| **Entropy** | Shannon entropy of URL and domain |

### Model Configuration

- **Algorithm**: Random Forest Classifier
- **Trees**: 100 estimators
- **Max Depth**: 20
- **Feature Scaling**: StandardScaler
- **Train/Test Split**: 80/20

## 💡 Usage Examples

### Example 1: Check a suspicious URL

```bash
python predict.py --url "http://paypal-secure-login.tk/verify"
```

Output:
```
======================================================================
URL: http://paypal-secure-login.tk/verify
----------------------------------------------------------------------
Classification: ⚠️  PHISHING
Phishing Probability: 94.32%
Risk Level: VERY HIGH
======================================================================
```

### Example 2: Batch classification

Create `urls.txt`:
```
https://www.google.com
http://suspicious-login-page.com
https://www.github.com
```

Run:
```bash
python predict.py --batch urls.txt
```

## 🎓 Dataset

The model is trained on a comprehensive phishing detection dataset containing:
- **247,951 URL samples**
- **Balanced classes**: Phishing and legitimate URLs
- **41 pre-extracted features**

Located at: `Phishing Detection Dataset/Phishing Detection Dataset/Dataset.csv`

## 📈 Visualizations

After training, check the generated PNG files:

1. **confusion_matrix.png** - Shows true/false positives and negatives
2. **roc_curve.png** - Displays the ROC curve with AUC score
3. **feature_importance.png** - Top 20 most important features

## ⚙️ Dependencies

- `scikit-learn` - Machine learning
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `matplotlib` - Plotting
- `seaborn` - Statistical visualizations
- `joblib` - Model persistence

## 📝 Notes

- The model requires training before making predictions
- Feature extraction is fully automated for raw URLs
- Predictions include confidence scores (0-100%)
- Risk levels: VERY LOW, LOW, MODERATE, HIGH, VERY HIGH

## 🔒 Security Considerations

This classifier is a machine learning tool and should be used as part of a defense-in-depth strategy. Always verify suspicious URLs through multiple methods and never rely solely on automated classification.

---

**Built with scikit-learn Random Forest Classifier**
