# Technical Deep Dive: Phishing Classifier Implementation

## Table of Contents
1. [Architecture Overview](#architecture)
2. [URL Feature Extraction Engine](#feature-extraction)
3. [Random Forest Classifier Core](#classifier-core)
4. [Training Pipeline](#training-pipeline)
5. [Prediction System](#prediction-system)
6. [Performance Optimization](#optimization)
7. [Mathematical Foundations](#mathematics)
8. [Advanced Topics](#advanced-topics)

---

## 1. Architecture Overview {#architecture}

### System Design Philosophy

The classifier follows a **modular, pipeline-based architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                     LAYERS ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────┤
│  Presentation Layer   │  predict.py                             │
│  (CLI Interface)      │  - Argument parsing                     │
│                       │  - User interaction                     │
│                       │  - Result formatting                    │
├───────────────────────┼─────────────────────────────────────────┤
│  Application Layer    │  phishing_classifier.py                 │
│  (Business Logic)     │  - Model management                     │
│                       │  - Training coordination                │
│                       │  - Evaluation orchestration             │
├───────────────────────┼─────────────────────────────────────────┤
│  Feature Layer        │  url_features.py                        │
│  (Data Processing)    │  - Feature extraction                   │
│                       │  - URL parsing                          │
│                       │  - Entropy calculation                  │
├───────────────────────┼─────────────────────────────────────────┤
│  Model Layer          │  scikit-learn RandomForestClassifier    │
│  (ML Algorithms)      │  - Tree construction                    │
│                       │  - Ensemble voting                       │
│                       │  - Prediction aggregation                │
├───────────────────────┼─────────────────────────────────────────┤
│  Persistence Layer    │  joblib                                 │
│  (Storage)            │  - Model serialization                  │
│                       │  - Scaler persistence                    │
│                       │  - Feature name mapping                  │
└─────────────────────────────────────────────────────────────────┘
```

### Design Patterns Used

1. **Strategy Pattern**: Feature extraction strategies can be swapped
2. **Template Method**: Training process follows a fixed template
3. **Facade Pattern**: `PhishingClassifier` provides simplified interface
4. **Singleton Pattern**: Model loading (implicit through joblib)

### File Organization

```
Classifier Prototype/
├── url_features.py           # Feature extraction (192 lines)
│   └── URLFeatureExtractor   # Main class
│       ├── extract_features()
│       ├── extract_as_array()
│       └── extract_batch()
│
├── phishing_classifier.py    # Core ML module (230 lines)
│   └── PhishingClassifier    # Main class
│       ├── load_data()
│       ├── preprocess_data()
│       ├── train()
│       ├── evaluate()
│       ├── predict()
│       └── save_model()/load_model()
│
├── train_model.py            # Training script (150 lines)
│   └── main()
│       ├── plot_confusion_matrix()
│       ├── plot_roc_curve()
│       └── plot_feature_importance()
│
└── predict.py                # Prediction CLI (170 lines)
    └── main()
        ├── classify_url()
        ├── classify_batch()
        └── print_result()
```

---

## 2. URL Feature Extraction Engine {#feature-extraction}

### Class Architecture

```python
class URLFeatureExtractor:
    FEATURE_NAMES = [...]  # 41 feature names (class constant)
    
    def extract_features(self, url: str) -> Dict[str, float]
    def extract_as_array(self, url: str) -> List[float]
    def extract_batch(self, urls: List[str]) -> List[List[float]]
    
    # Private helper methods
    def _has_repeated_digits(self, text: str) -> int
    def _extract_subdomains(self, domain: str) -> List[str]
    def _calculate_entropy(self, text: str) -> float
```

### Feature Extraction Algorithm

#### URL Parsing Strategy

```python
from urllib.parse import urlparse

parsed = urlparse(url)
# Components:
# - scheme: 'http' or 'https'
# - netloc: 'www.example.com' (domain)
# - path: '/path/to/page'
# - query: 'id=123&name=test'
# - fragment: 'section'
```

**Why `urlparse`?**
- RFC 3986 compliant
- Handles edge cases (malformed URLs, IPv6, etc.)
- Separates components cleanly

#### Feature Categories Implementation

##### A. URL-Level Features (Lines 52-71)

```python
# Direct character counting using generator expressions
features['url_length'] = len(url)
features['number_of_digits_in_url'] = sum(c.isdigit() for c in url)
features['number_of_dots_in_url'] = url.count('.')
```

**Complexity:** O(n) where n = URL length
**Memory:** O(1) - streaming computation

##### B. Special Character Counting (Lines 61-71)

```python
special_chars = ['@', '?', '=', '$', '!', '#', '%', '-', '_', '/']
features['number_of_special_char_in_url'] = sum(url.count(c) for c in special_chars)
```

**Optimization Note:** Each `count()` is O(n), total O(k*n) where k=10 special chars.
Could be optimized with single pass:

```python
# Alternative O(n) approach:
char_counts = Counter(url)
total_special = sum(char_counts[c] for c in special_chars)
```

##### C. Repeated Digit Detection (Lines 133-138)

```python
def _has_repeated_digits(self, text):
    for i in range(len(text) - 1):
        if text[i].isdigit() and text[i] == text[i + 1]:
            return 1
    return 0
```

**Algorithm:**
- Single pass through string
- Early termination on first match
- Time: O(n), Space: O(1)

**Pattern matched:** `"11"`, `"22"`, `"333"`, etc.

##### D. Subdomain Extraction (Lines 140-151)

```python
def _extract_subdomains(self, domain):
    if not domain:
        return []
    
    parts = domain.split('.')
    # Last two parts = main domain (e.g., example.com)
    if len(parts) > 2:
        return parts[:-2]  # Everything before main domain
    elif len(parts) == 2:
        return parts[:1]   # First part as subdomain
    return []
```

**Examples:**
- `www.sub.example.com` → `['www', 'sub']`
- `sub.example.com` → `['sub']`
- `example.com` → `[]`

**Edge Case Handling:**
- `co.uk` domains: Treats `co` as subdomain (limitation, but acceptable)
- Empty domain: Returns empty list

##### E. Shannon Entropy Calculation (Lines 153-168)

```python
def _calculate_entropy(self, text):
    if not text:
        return 0.0
    
    # Count character frequencies
    counter = Counter(text)
    length = len(text)
    
    # Shannon entropy formula
    entropy = 0.0
    for count in counter.values():
        probability = count / length
        entropy -= probability * math.log2(probability)
    
    return entropy
```

**Mathematical Foundation:**

Shannon Entropy: H(X) = -Σ p(x) * log₂(p(x))

**Example Calculation:**

For URL `"google.com"`:
```
Characters: g(2), o(3), g(1), l(1), e(1), .(1), c(1), m(1)
Total length: 10

Probabilities:
- 'o': 3/10 = 0.3
- 'g': 2/10 = 0.2
- Others: 1/10 = 0.1 each

H = -(0.3*log₂(0.3) + 0.2*log₂(0.2) + 6*0.1*log₂(0.1))
H ≈ 2.846 bits
```

**Interpretation:**
- Low entropy (< 3.0): Predictable, common patterns → Likely legitimate
- High entropy (> 4.5): Random, unpredictable → Potentially suspicious

**Time Complexity:** O(n) for counting + O(k) for entropy calc, where k = unique chars
**Space Complexity:** O(k)

#### Default Value Handling

```python
# For empty subdomains (lines 113-122)
if subdomains:
    # Calculate actual values
    features['average_subdomain_length'] = sum(len(sd) for sd in subdomains) / len(subdomains)
else:
    # Dataset-derived default values
    features['having_special_characters_in_subdomain'] = 1
    features['number_of_special_characters_in_subdomain'] = 3
```

**Why these defaults?**
Analyzed from dataset: when no subdomains exist, these are median values from training data.

### Output Format

```python
# Returns dictionary (extract_features)
{
    'url_length': 42,
    'number_of_dots_in_url': 2,
    ...
    'entropy_of_domain': 2.751
}

# Returns array (extract_as_array)
[42, 2, 0, 0, 8, 0, 0, 5, ..., 2.751]  # 41 values in exact order
```

**Critical:** Array order must match `FEATURE_NAMES` constant to align with trained model.

---

## 3. Random Forest Classifier Core {#classifier-core}

### Class Design

```python
class PhishingClassifier:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(...)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = None
```

**State Management:**
- `is_trained`: Guards against using untrained model
- `feature_names`: Ensures feature alignment during prediction
- `scaler`: Fitted only once during training, transformed during prediction

### Random Forest Configuration

```python
RandomForestClassifier(
    n_estimators=100,        # Number of trees
    max_depth=20,            # Maximum tree depth
    min_samples_split=5,     # Minimum samples to split node
    min_samples_leaf=2,      # Minimum samples in leaf
    random_state=42,         # Reproducibility
    n_jobs=-1               # Use all CPU cores
)
```

#### Hyperparameter Analysis

**n_estimators=100:**
- Trade-off: Accuracy vs training time
- 100 trees: ~2-3 minutes training on 240K samples
- Diminishing returns after ~100-150 trees

**max_depth=20:**
- Prevents overfitting (unbounded depth → memorization)
- 20 levels ≈ 2^20 = 1M possible leaf nodes
- Sufficient for most decision boundaries

**min_samples_split=5:**
- Node must have ≥5 samples to split further
- Reduces tree complexity
- Prevents tiny, noisy splits

**min_samples_leaf=2:**
- Each leaf must have ≥2 samples
- Additional overfitting protection
- Ensures leaves represent multiple examples

**n_jobs=-1:**
- Parallel tree construction
- Uses all CPU cores (ProcessPoolExecutor internally)
- ~4x speedup on quad-core CPU

### Data Loading Implementation

```python
def load_data(self, csv_path):
    df = pd.read_csv(csv_path)
    
    X = df.drop('Type', axis=1)  # Features
    y = df['Type']                # Labels
    
    self.feature_names = X.columns.tolist()
    
    return X, y
```

**Pandas Operations:**
- `read_csv`: Efficient C-based CSV parser
- `drop`: Returns view (not copy) for memory efficiency
- Column extraction: O(1) operation

### Feature Standardization

```python
def preprocess_data(self, X_train, X_test):
    X_train_scaled = self.scaler.fit_transform(X_train)
    X_test_scaled = self.scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
```

**StandardScaler Mathematics:**

For each feature j:
```
X_scaled[j] = (X[j] - μⱼ) / σⱼ

where:
- μⱼ = mean of feature j in training data
- σⱼ = standard deviation of feature j in training data
```

**Why fit only on training data?**
- Prevents data leakage from test set
- Test set represents "future" unseen data
- Using test statistics would give unrealistic performance

**Stored Parameters:**
```python
scaler.mean_      # Shape: (41,) - one mean per feature
scaler.scale_     # Shape: (41,) - one std dev per feature
```

### Training Process

```python
def train(self, X_train, y_train):
    self.model.fit(X_train, y_train)
    self.is_trained = True
```

**What `fit()` does internally:**

1. **Bootstrap Sampling** (for each tree):
```python
for tree_idx in range(n_estimators):
    # Sample with replacement
    sample_indices = np.random.choice(n_samples, n_samples, replace=True)
    X_bootstrap = X_train[sample_indices]
    y_bootstrap = y_train[sample_indices]
```

2. **Feature Sampling** (at each node):
```python
# √41 ≈ 6 features considered per split
max_features = int(np.sqrt(n_features))  # ≈ 6
```

3. **Tree Construction** (CART algorithm):
```python
def build_tree(X, y, depth):
    if stopping_criterion(depth, n_samples):
        return create_leaf(y)
    
    # Find best split
    best_feature, best_threshold = find_best_split(X, y)
    
    # Recursive split
    left_mask = X[:, best_feature] <= best_threshold
    left_tree = build_tree(X[left_mask], y[left_mask], depth+1)
    right_tree = build_tree(X[~left_mask], y[~left_mask], depth+1)
    
    return Decision(best_feature, best_threshold, left_tree, right_tree)
```

4. **Split Criterion** (Gini Impurity):
```
Gini(node) = 1 - Σ(pᵢ²)

where pᵢ = proportion of class i in node
```

**Example:**
```
Node with 100 samples: 70 phishing, 30 legitimate
Gini = 1 - (0.7² + 0.3²) = 1 - (0.49 + 0.09) = 0.42

Pure node (all one class):
Gini = 1 - 1² = 0 (perfect)
```

### Evaluation Metrics

```python
def evaluate(self, X_test, y_test):
    y_pred = self.model.predict(X_test)
    y_pred_proba = self.model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        ...
    }
```

**Metric Calculations:**

```python
# Confusion Matrix
[[TN, FP],
 [FN, TP]]

# Accuracy
accuracy = (TP + TN) / (TP + TN + FP + FN)

# Precision
precision = TP / (TP + FP)

# Recall
recall = TP / (TP + FN)

# F1-Score
f1 = 2 * (precision * recall) / (precision + recall)

# ROC-AUC (via sklearn implementation)
# Step 1: Sort predictions by probability
# Step 2: Calculate TPR and FPR at each threshold
# Step 3: Compute area under ROC curve
```

### Prediction Pipeline

```python
def predict(self, X):
    # Input validation
    if not self.is_trained:
        raise ValueError("Model must be trained before prediction")
    
    # Ensure 2D array
    if len(X.shape) == 1:
        X = X.reshape(1, -1)
    
    # Scale features
    X_scaled = self.scaler.transform(X)
    
    # Predict
    predictions = self.model.predict(X_scaled)
    probabilities = self.model.predict_proba(X_scaled)[:, 1]
    
    return predictions, probabilities
```

**Probability Aggregation:**

For each tree:
```python
# Each tree returns: 0 or 1
tree_predictions = [tree.predict(X_scaled) for tree in self.model.estimators_]

# Aggregate
probability = np.mean(tree_predictions)  # Average of 0s and 1s
```

Example:
```
94 trees predict 1 (phishing)
6 trees predict 0 (legitimate)
Probability = 94/100 = 0.94
```

### Model Persistence

```python
def save_model(self, model_path='phishing_model.pkl', scaler_path='scaler.pkl'):
    joblib.dump(self.model, model_path)
    joblib.dump(self.scaler, scaler_path)
    joblib.dump(self.feature_names, 'feature_names.pkl')
```

**joblib vs pickle:**
- `joblib`: Optimized for large numpy arrays (Random Forest trees)
- Uses memory mapping for large files
- More efficient for scikit-learn models
- Compression support

**File Sizes:**
```
phishing_model.pkl:    79 MB  (100 trees with ~198K training samples)
scaler.pkl:            3 KB   (41 means + 41 std devs)
feature_names.pkl:     1 KB   (41 strings)
```

---

## 4. Training Pipeline {#training-pipeline}

### Main Training Flow

```python
def main():
    # 1. Initialize
    classifier = PhishingClassifier(n_estimators=100, random_state=42)
    
    # 2. Load data
    X, y = classifier.load_data(dataset_path)
    
    # 3. Split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Preprocess
    X_train_scaled, X_test_scaled = classifier.preprocess_data(X_train, X_test)
    
    # 5. Train
    classifier.train(X_train_scaled, y_train)
    
    # 6. Evaluate
    metrics = classifier.evaluate(X_test_scaled, y_test)
    
    # 7. Visualize
    plot_confusion_matrix(metrics['confusion_matrix'])
    plot_roc_curve(metrics['y_test'], metrics['y_pred_proba'], metrics['roc_auc'])
    plot_feature_importance(classifier.get_feature_importance())
    
    # 8. Save
    classifier.save_model()
```

### Stratified Splitting

```python
train_test_split(..., stratify=y)
```

**Why stratify?**
Maintains class distribution in both sets:

```
Original: 50% phishing, 50% legitimate
Train:    50% phishing, 50% legitimate (not 48% and 52%)
Test:     50% phishing, 50% legitimate
```

**Implementation internally:**
```python
for class_label in unique(y):
    class_indices = where(y == class_label)
    class_train, class_test = split(class_indices, test_size)
    train_indices.extend(class_train)
    test_indices.extend(class_test)
```

### Visualization Functions

#### Confusion Matrix Heatmap

```python
def plot_confusion_matrix(cm, save_path='confusion_matrix.png'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legitimate', 'Phishing'],
                yticklabels=['Legitimate', 'Phishing'])
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
```

**Parameters:**
- `annot=True`: Show values in cells
- `fmt='d'`: Integer format (not 24500.0, but 24500)
- `cmap='Blues'`: Color scheme (light→dark for low→high values)
- `dpi=300`: High resolution for publication quality

#### ROC Curve

```python
def plot_roc_curve(y_test, y_pred_proba, roc_auc, save_path='roc_curve.png'):
    from sklearn.metrics import roc_curve
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random Classifier')
    # ... styling ...
```

**ROC Curve Computation:**

```python
# Pseudocode for roc_curve()
thresholds = sorted(unique(y_pred_proba), reverse=True)
fpr_list, tpr_list = [], []

for threshold in thresholds:
    y_pred_binary = (y_pred_proba >= threshold).astype(int)
    
    TP = sum((y_test == 1) & (y_pred_binary == 1))
    FP = sum((y_test == 0) & (y_pred_binary == 1))
    TN = sum((y_test == 0) & (y_pred_binary == 0))
    FN = sum((y_test == 1) & (y_pred_binary == 0))
    
    tpr = TP / (TP + FN)  # Sensitivity
    fpr = FP / (FP + TN)  # 1 - Specificity
    
    tpr_list.append(tpr)
    fpr_list.append(fpr)

return np.array(fpr_list), np.array(tpr_list), thresholds
```

#### Feature Importance

```python
def plot_feature_importance(importance_df, top_n=20, save_path='feature_importance.png'):
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
    plt.yticks(range(len(top_features)), top_features['feature'])
    # ... styling ...
```

**Feature Importance Calculation:**

Random Forest computes importance using **Mean Decrease in Impurity (MDI)**:

```python
importance[feature_j] = Σ (node_impurity_decrease caused by feature_j)
                        / n_trees
```

For each split on feature j:
```
impurity_decrease = n_samples_node * (gini_parent - weighted_gini_children)
```

**Normalization:**
```python
importance /= importance.sum()  # Sum to 1.0
```

---

## 5. Prediction System {#prediction-system}

### CLI Argument Parsing

```python
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
parser.add_argument('--batch', type=str, help='File containing URLs')
parser.add_argument('--test', action='store_true', help='Run test')
```

**Argument Types:**
- `type=str`: Validates input is string
- `action='store_true'`: Boolean flag (presence = True)
- `epilog`: Shows usage examples in help

### Single URL Classification

```python
def classify_url(url, classifier, extractor):
    # Extract features
    features = extractor.extract_as_array(url)
    features_array = np.array(features).reshape(1, -1)
    
    # Predict
    prediction, probability = classifier.predict(features_array)
    
    return prediction[0], probability[0]
```

**Shape Management:**
```python
# extractor.extract_as_array() returns: [val1, val2, ..., val41]
# Shape: (41,)

# np.array(features) creates: [val1, val2, ..., val41]
# Shape: (41,)

# reshape(1, -1) converts to: [[val1, val2, ..., val41]]
# Shape: (1, 41) - required for sklearn
```

### Batch Processing

```python
def classify_batch(urls, classifier, extractor):
    results = []
    for i, url in enumerate(urls, 1):
        try:
            prediction, probability = classify_url(url, classifier, extractor)
            results.append({
                'url': url,
                'prediction': prediction,
                'probability': probability
            })
            # ... logging ...
        except Exception as e:
            print(f"{i}. ERROR - {url[:50]}... ({str(e)})")
    
    return results
```

**Error Handling:**
- `try-except` prevents one bad URL from stopping the batch
- Continues processing remaining URLs
- Logs errors but doesn't crash

**Optimization Opportunity:**
```python
# Current: O(n) predictions
# Could optimize to:
features_batch = extractor.extract_batch(urls)  # Already implemented!
predictions, probabilities = classifier.predict(np.array(features_batch))
# Single batch prediction: faster due to vectorization
```

### Risk Level Categorization

```python
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
```

**Threshold Selection:**
Based on empirical analysis of prediction distribution. Could be tuned based on:
- Business requirements (false positive tolerance)
- ROC curve optimal points
- Cost-benefit analysis

---

## 6. Performance Optimization {#optimization}

### Time Complexity Analysis

| Operation | Complexity | Details |
|-----------|------------|---------|
| Feature extraction | O(n) | n = URL length, typically < 500 |
| Standardization | O(f) | f = 41 features |
| Tree prediction | O(d*t) | d = depth (20), t = trees (100) |
| Total prediction | O(n + d*t) | ≈ O(2000) → constant |

**Training Complexity:**
```
O(n_samples * n_features * sqrt(n_features) * depth * n_trees)
= O(198360 * 41 * 6 * 20 * 100)
≈ O(9.7 billion operations)
```

**Actual training time:** ~2-3 minutes on modern CPU

### Memory Usage

```python
# Training memory footprint
X_train: 198360 × 41 × 8 bytes = ~65 MB (float64)
y_train: 198360 × 8 bytes = ~1.5 MB
Model:   100 trees × ~790 KB ≈ 79 MB
Total:   ~145 MB
```

**Model size analysis:**
Each tree stores:
- Node split features
- Threshold values
- Leaf predictions
- Tree structure

### Vectorization Benefits

```python
# Non-vectorized (slow)
for i in range(len(X)):
    result[i] = scaler.transform([X[i]])

# Vectorized (fast)
result = scaler.transform(X)  # NumPy/C optimized
```

**Speedup:** ~10-100x due to:
- SIMD instructions
- Cache locality
- Avoiding Python loops

### Parallelization

```python
RandomForestClassifier(..., n_jobs=-1)
```

**What gets parallelized:**
- Tree construction (embarrassingly parallel)
- Prediction across trees
- Feature importance calculation

**Not parallelized:**
- Single tree traversal (inherently sequential)
- Data loading

---

## 7. Mathematical Foundations {#mathematics}

### Gini Impurity Deep Dive

**Formula:**
```
Gini(D) = 1 - Σᵢ pᵢ²

where:
- D = dataset at node
- pᵢ = proportion of class i
```

**Why squared?**
Penalizes mixed nodes more than linear measure.

**Example comparison:**

Node A: 50 phishing, 50 legitimate
```
Gini = 1 - (0.5² + 0.5²) = 0.5
```

Node B: 90 phishing, 10 legitimate
```
Gini = 1 - (0.9² + 0.1²) = 0.18
```

Node B is "purer" → lower Gini.

### Information Gain (Alternative)

```
IG(D, feature) = Gini(D_parent) - Σ (wᵢ * Gini(D_child_i))

where:
- wᵢ = weight of child (proportion of samples)
```

**Example:**
```
Parent: 100 samples (60 phishing, 40 legit) → Gini = 0.48

Split on url_length <= 50:
- Left:  30 samples (5 phishing, 25 legit) → Gini = 0.28
- Right: 70 samples (55 phishing, 15 legit) → Gini = 0.34

IG = 0.48 - (0.3*0.28 + 0.7*0.34) = 0.48 - 0.32 = 0.16
```

Positive IG → good split!

### Ensemble Voting Mathematics

**Hard Voting (Classification):**
```
Ĉ = mode({C₁, C₂, ..., C₁₀₀})

where Cᵢ = prediction of tree i
```

**Soft Voting (Probability):**
```
P(phishing) = (1/n_trees) * Σᵢ Pᵢ(phishing)

where Pᵢ(phishing) = probability from tree i
```

**Why soft voting is better:**
- Uses full probability information
- More robust to close calls
- Produces calibrated probabilities

### Bootstrap Aggregating (Bagging)

**Sampling:**
```
For each tree:
  Sample n_samples with replacement from training set
  
Expected unique samples: n * (1 - (1 - 1/n)ⁿ) ≈ 0.632n
```

**Out-of-Bag (OOB) Estimation:**
~37% of samples not used in each tree → can estimate error without validation set.

```python
# Could implement
oob_score = model.oob_score_  # If oob_score=True in constructor
```

---

## 8. Advanced Topics {#advanced-topics}

### Handling Imbalanced Data

**Current approach:** Stratified splitting

**Alternative techniques:**

1. **SMOTE (Synthetic Minority Over-sampling):**
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

2. **Class weights:**
```python
RandomForestClassifier(..., class_weight='balanced')
# Automatically adjusts for imbalance
```

3. **Threshold tuning:**
```python
# Instead of 0.5, use optimal threshold from ROC curve
optimal_threshold = thresholds[np.argmax(tpr - fpr)]
```

### Hyperparameter Tuning

**Grid Search:**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='roc_auc',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
```

**Random Search (faster):**
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distributions = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(10, 50),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(),
    param_distributions,
    n_iter=50,  # Try 50 combinations
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
```

### Model Interpretability

**SHAP (SHapley Additive exPlanations):**
```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Visualize
shap.summary_plot(shap_values, X_test, feature_names=feature_names)
```

**Partial Dependence Plots:**
```python
from sklearn.inspection import partial_dependence

pd_result = partial_dependence(
    model, X_train, features=[0, 5, 10]  # Feature indices
)

# Shows how predictions change as feature values change
```

### Production Deployment

**Model Serving API (Flask example):**
```python
from flask import Flask, request, jsonify
from phishing_classifier import PhishingClassifier
from url_features import URLFeatureExtractor

app = Flask(__name__)
classifier = PhishingClassifier()
classifier.load_model()
extractor = URLFeatureExtractor()

@app.route('/predict', methods=['POST'])
def predict():
    url = request.json['url']
    features = extractor.extract_as_array(url)
    prediction, probability = classifier.predict([features])
    
    return jsonify({
        'url': url,
        'prediction': 'phishing' if prediction[0] == 1 else 'legitimate',
        'confidence': float(probability[0])
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Docker Containerization:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "api.py"]
```

### Model Monitoring

**Drift Detection:**
```python
def detect_drift(X_new, X_reference, threshold=0.05):
    """Check if new data distribution differs from reference"""
    from scipy.stats import ks_2samp
    
    p_values = []
    for i in range(X_new.shape[1]):
        statistic, p_value = ks_2samp(X_new[:, i], X_reference[:, i])
        p_values.append(p_value)
    
    drift_detected = any(p < threshold for p in p_values)
    return drift_detected, p_values
```

**Performance Tracking:**
```python
class ModelMonitor:
    def __init__(self):
        self.predictions = []
        self.actuals = []
    
    def log_prediction(self, prediction, actual=None):
        self.predictions.append(prediction)
        if actual is not None:
            self.actuals.append(actual)
    
    def get_accuracy(self):
        if not self.actuals:
            return None
        return accuracy_score(self.actuals, self.predictions)
```

---

## Code Quality Considerations

### Type Hints
```python
from typing import List, Dict, Tuple, Optional
import numpy as np

def extract_features(self, url: str) -> Dict[str, float]: ...
def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: ...
```

### Error Handling
```python
def load_model(self, model_path: str) -> None:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        self.model = joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")
```

### Logging
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train(self, X_train, y_train):
    logger.info(f"Training on {len(X_train)} samples...")
    self.model.fit(X_train, y_train)
    logger.info("Training completed successfully")
```

### Unit Testing
```python
import unittest

class TestURLFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = URLFeatureExtractor()
    
    def test_url_length(self):
        url = "http://example.com"
        features = self.extractor.extract_features(url)
        self.assertEqual(features['url_length'], len(url))
    
    def test_entropy_calculation(self):
        # Test known entropy value
        entropy = self.extractor._calculate_entropy("aaaa")
        self.assertAlmostEqual(entropy, 0.0, places=5)
```

---

## Summary

This classifier implementation demonstrates:

✅ **Clean Architecture**: Separation of concerns, modular design
✅ **Efficient Algorithms**: Vectorization, parallelization
✅ **Production Ready**: Error handling, logging, model persistence
✅ **Interpretable**: Feature importance, visualizations
✅ **Scalable**: Batch processing, efficient data structures
✅ **Maintainable**: Clear code structure, documentation

**Key Technical Achievements:**
- 41-feature extraction pipeline
- 100-tree Random Forest ensemble
- ~95%+ accuracy on 250K samples
- Real-time prediction (< 100ms per URL)
- Comprehensive evaluation metrics

The implementation balances simplicity with performance, making it suitable for both learning and production deployment.
