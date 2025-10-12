# ============================================================
# improved_text_model.py
# Enhanced Text-Based Model with Advanced Feature Engineering
# ============================================================

import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# Enhanced Feature Engineering
# ------------------------------

def extract_ipq(text):
    """Extract Item Pack Quantity from text"""
    if pd.isna(text):
        return 1
    
    # Look for patterns like "Pack of 12", "12-Pack", "Count: 24", etc.
    patterns = [
        r'pack of (\d+)',
        r'(\d+)-pack',
        r'(\d+) pack',
        r'count[:\s]+(\d+)',
        r'quantity[:\s]+(\d+)',
        r'\((\d+)\s*count\)',
    ]
    
    text_lower = text.lower()
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            qty = int(match.group(1))
            if 1 <= qty <= 1000:  # Sanity check
                return qty
    
    return 1  # Default to 1 if not found


def extract_numeric_features(text):
    """Extract various numeric features from text"""
    if pd.isna(text):
        return {
            'num_count': 0,
            'max_number': 0,
            'has_dimensions': 0,
            'has_weight': 0,
            'has_volume': 0
        }
    
    text_lower = text.lower()
    
    # Find all numbers
    numbers = re.findall(r'\d+\.?\d*', text)
    numbers = [float(n) for n in numbers if float(n) < 10000]  # Filter outliers
    
    features = {
        'num_count': len(numbers),
        'max_number': max(numbers) if numbers else 0,
        'has_dimensions': 1 if re.search(r'\d+\s*x\s*\d+', text_lower) else 0,
        'has_weight': 1 if re.search(r'\d+\s*(oz|lb|kg|g|gram|pound)', text_lower) else 0,
        'has_volume': 1 if re.search(r'\d+\s*(ml|l|oz|gallon|fl)', text_lower) else 0,
    }
    
    return features


def extract_text_features(text):
    """Extract text-based features"""
    if pd.isna(text):
        return {
            'text_length': 0,
            'word_count': 0,
            'avg_word_length': 0,
            'uppercase_ratio': 0,
            'digit_ratio': 0,
            'special_char_ratio': 0
        }
    
    features = {
        'text_length': len(text),
        'word_count': len(text.split()),
        'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
        'digit_ratio': sum(1 for c in text if c.isdigit()) / len(text) if text else 0,
        'special_char_ratio': sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text) if text else 0
    }
    
    return features


def extract_brand_keywords(text):
    """Check for presence of common brand-related keywords"""
    if pd.isna(text):
        return 0
    
    brand_indicators = [
        'brand', 'trademark', 'original', 'authentic', 'genuine',
        'premium', 'professional', 'certified', 'authorized'
    ]
    
    text_lower = text.lower()
    return sum(1 for keyword in brand_indicators if keyword in text_lower)


def extract_quality_keywords(text):
    """Check for quality-related keywords"""
    if pd.isna(text):
        return 0
    
    quality_words = [
        'premium', 'luxury', 'deluxe', 'professional', 'high-quality',
        'best', 'top', 'superior', 'excellent', 'finest', 'elite'
    ]
    
    text_lower = text.lower()
    return sum(1 for keyword in quality_words if keyword in text_lower)


def clean_text(text):
    """Clean text while preserving important information"""
    if pd.isna(text):
        return ""
    
    text = text.lower()
    
    # Keep numbers but remove URLs and HTML
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    
    # Keep alphanumeric and some special chars
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.mean(diff) * 100


# ------------------------------
# Feature Preparation Pipeline
# ------------------------------

def create_feature_dataframe(df):
    """Create comprehensive feature set from catalog_content"""
    print("🔧 Extracting features...")
    
    features_list = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        text = row['catalog_content']
        
        # Extract all features
        features = {
            'ipq': extract_ipq(text),
            'brand_keywords': extract_brand_keywords(text),
            'quality_keywords': extract_quality_keywords(text),
        }
        
        # Add numeric features
        features.update(extract_numeric_features(text))
        
        # Add text features
        features.update(extract_text_features(text))
        
        features_list.append(features)
    
    return pd.DataFrame(features_list)


def prepare_enhanced_features(train_df, test_df, test_size=0.1, random_state=42):
    """
    Prepare enhanced features with both TF-IDF and manual features
    """
    # Make copies
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    # 1. Clean text for TF-IDF
    print("🧹 Cleaning text data...")
    tqdm.pandas(desc="Cleaning train")
    train_df["clean_text"] = train_df["catalog_content"].progress_apply(clean_text)
    
    tqdm.pandas(desc="Cleaning test")
    test_df["clean_text"] = test_df["catalog_content"].progress_apply(clean_text)
    
    # 2. Extract manual features
    print("\n📊 Creating feature dataframes...")
    train_features = create_feature_dataframe(train_df)
    test_features = create_feature_dataframe(test_df)
    
    # 3. Create TF-IDF features
    print("\n🔤 Creating TF-IDF features...")
    tfidf = TfidfVectorizer(
        max_features=5000,  # Reduced to save memory
        ngram_range=(1, 3),  # Include trigrams
        min_df=3,
        max_df=0.85,
        stop_words='english',
        sublinear_tf=True  # Use log scaling
    )
    
    # Fit on all data (train + test) for better coverage
    all_text = pd.concat([train_df["clean_text"], test_df["clean_text"]])
    tfidf.fit(all_text)
    
    # Transform
    train_tfidf = tfidf.transform(train_df["clean_text"])
    test_tfidf = tfidf.transform(test_df["clean_text"])
    
    # 4. Scale manual features
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)
    
    # 5. Combine features
    from scipy.sparse import hstack, csr_matrix
    
    X_train_combined = hstack([train_tfidf, csr_matrix(train_features_scaled)])
    X_test_combined = hstack([test_tfidf, csr_matrix(test_features_scaled)])
    
    # 6. Split for validation
    train_df["log_price"] = np.log1p(train_df["price"])
    
    train_indices, valid_indices = train_test_split(
        range(len(train_df)), 
        test_size=test_size, 
        random_state=random_state
    )
    
    X_train = X_train_combined[train_indices]
    X_valid = X_train_combined[valid_indices]
    y_train = train_df["log_price"].iloc[train_indices]
    y_valid = train_df["log_price"].iloc[valid_indices]
    X_test = X_test_combined
    
    # Store preprocessing objects
    preprocessing = {
        'tfidf': tfidf,
        'scaler': scaler
    }
    
    return X_train, y_train, X_valid, y_valid, X_test, preprocessing


def train_enhanced_model(X_train, y_train, X_valid, y_valid, model_type='ridge'):
    """
    Train enhanced model with better hyperparameters
    """
    print(f"\n🚀 Training {model_type.upper()} model...")
    
    if model_type == 'ridge':
        model = Ridge(alpha=10.0, random_state=42)  # Increased regularization
    elif model_type == 'rf':
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'gbm':
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)
    
    # Evaluate
    print("🔍 Evaluating on validation set...")
    val_preds = model.predict(X_valid)
    val_preds_exp = np.expm1(val_preds)
    y_valid_exp = np.expm1(y_valid)
    
    mae = mean_absolute_error(y_valid_exp, val_preds_exp)
    smape_val = smape(y_valid_exp, val_preds_exp)
    
    print(f"📊 Validation MAE: {mae:.3f}")
    print(f"📊 Validation SMAPE: {smape_val:.2f}%")
    
    return model, smape_val


def predict_enhanced(model, X_test, test_df):
    """Generate predictions"""
    print("\n🎯 Generating predictions...")
    
    test_preds_log = model.predict(X_test)
    test_preds = np.expm1(test_preds_log)
    test_preds = np.maximum(test_preds, 0.01)
    
    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': test_preds
    })
    
    return submission


# ------------------------------
# Main Execution
# ------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 ENHANCED TEXT-BASED MODEL")
    print("=" * 70)
    
    # Load data
    train_df = pd.read_csv("dataset/train.csv")
    test_df = pd.read_csv("dataset/test.csv")
    
    print(f"\n✅ Train shape: {train_df.shape}")
    print(f"✅ Test shape: {test_df.shape}")
    
    # Clean data
    train_df = train_df.dropna(subset=["catalog_content", "price"]).reset_index(drop=True)
    
    # Prepare features
    X_train, y_train, X_valid, y_valid, X_test, preprocessing = prepare_enhanced_features(
        train_df, test_df
    )
    
    print(f"\n✅ Feature shapes:")
    print(f"   Train: {X_train.shape}")
    print(f"   Valid: {X_valid.shape}")
    print(f"   Test: {X_test.shape}")
    
    # Train model
    model, smape_val = train_enhanced_model(X_train, y_train, X_valid, y_valid, model_type='ridge')
    
    # Generate predictions
    submission = predict_enhanced(model, X_test, test_df)
    submission.to_csv("submission_enhanced.csv", index=False)
    
    print("\n" + "=" * 70)
    print("✅ ENHANCED MODEL COMPLETE!")
    print(f"📊 Validation SMAPE: {smape_val:.2f}%")
    print(f"📁 Saved: submission_enhanced.csv")
    print("=" * 70)