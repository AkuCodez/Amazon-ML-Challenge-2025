# ============================================================
# quick_ipq_baseline.py
# Quick improvement: Add IPQ feature to existing baseline
# Expected: 48-52% SMAPE (vs 57% baseline)
# ============================================================

import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
import joblib
from tqdm import tqdm

# ------------------------------
# IPQ Extraction (CRITICAL!)
# ------------------------------

def extract_ipq(text):
    """
    Extract Item Pack Quantity - THE most important feature!
    This alone can reduce SMAPE by ~10%
    """
    if pd.isna(text):
        return 1
    
    text_lower = text.lower()
    
    # Common patterns for pack quantities
    patterns = [
        r'pack of (\d+)',           # "Pack of 12"
        r'(\d+)-pack',              # "12-Pack"
        r'(\d+)\s*pack',            # "12 pack"
        r'count[:\s]+(\d+)',        # "Count: 24"
        r'quantity[:\s]+(\d+)',     # "Quantity: 6"
        r'\((\d+)\s*count\)',       # "(12 count)"
        r'set of (\d+)',            # "Set of 4"
        r'(\d+)\s*ct',              # "24 ct"
        r'(\d+)\s*piece',           # "6 piece"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            qty = int(match.group(1))
            # Sanity check: reasonable pack sizes
            if 1 <= qty <= 1000:
                return qty
    
    return 1  # Default single item


def extract_basic_numeric_features(text):
    """Extract simple numeric features"""
    if pd.isna(text):
        return {
            'has_weight': 0,
            'has_volume': 0,
            'has_dimensions': 0,
            'number_count': 0
        }
    
    text_lower = text.lower()
    
    return {
        'has_weight': 1 if re.search(r'\d+\s*(oz|lb|kg|g|gram|pound|ounce)', text_lower) else 0,
        'has_volume': 1 if re.search(r'\d+\s*(ml|l|liter|fl|fluid|gallon)', text_lower) else 0,
        'has_dimensions': 1 if re.search(r'\d+\s*x\s*\d+', text_lower) else 0,
        'number_count': len(re.findall(r'\d+', text))
    }


def clean_text(text):
    """Clean text but KEEP numbers (they matter!)"""
    if pd.isna(text):
        return ""
    
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    
    # Keep alphanumeric characters
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
# Main Pipeline with IPQ
# ------------------------------

def prepare_features_with_ipq(train_df, test_df, test_size=0.1, random_state=42):
    """Prepare features with IPQ and basic numeric features"""
    
    print("🔍 Extracting IPQ and numeric features...")
    
    # Extract IPQ
    tqdm.pandas(desc="Train IPQ")
    train_df["ipq"] = train_df["catalog_content"].progress_apply(extract_ipq)
    
    tqdm.pandas(desc="Test IPQ")
    test_df["ipq"] = test_df["catalog_content"].progress_apply(extract_ipq)
    
    # Extract basic numeric features
    print("\n📊 Extracting numeric features...")
    train_numeric = pd.DataFrame([
        extract_basic_numeric_features(text) 
        for text in tqdm(train_df["catalog_content"], desc="Train numeric")
    ])
    
    test_numeric = pd.DataFrame([
        extract_basic_numeric_features(text) 
        for text in tqdm(test_df["catalog_content"], desc="Test numeric")
    ])
    
    # Combine with IPQ
    train_numeric['ipq'] = train_df['ipq'].values
    test_numeric['ipq'] = test_df['ipq'].values
    
    print(f"\n✅ IPQ Statistics (Train):")
    print(f"   Mean: {train_df['ipq'].mean():.2f}")
    print(f"   Median: {train_df['ipq'].median():.0f}")
    print(f"   Max: {train_df['ipq'].max():.0f}")
    print(f"   Single items (IPQ=1): {(train_df['ipq']==1).sum()} ({(train_df['ipq']==1).mean()*100:.1f}%)")
    
    # Clean text for TF-IDF
    print("\n🧹 Cleaning text...")
    tqdm.pandas(desc="Clean train")
    train_df["clean_text"] = train_df["catalog_content"].progress_apply(clean_text)
    
    tqdm.pandas(desc="Clean test")
    test_df["clean_text"] = test_df["catalog_content"].progress_apply(clean_text)
    
    # TF-IDF
    print("\n🔤 Creating TF-IDF features...")
    tfidf = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        stop_words='english'
    )
    
    train_tfidf = tfidf.fit_transform(train_df["clean_text"])
    test_tfidf = tfidf.transform(test_df["clean_text"])
    
    # Scale numeric features
    scaler = StandardScaler()
    train_numeric_scaled = scaler.fit_transform(train_numeric)
    test_numeric_scaled = scaler.transform(test_numeric)
    
    # Combine TF-IDF + numeric features
    X_train_combined = hstack([train_tfidf, csr_matrix(train_numeric_scaled)])
    X_test_combined = hstack([test_tfidf, csr_matrix(test_numeric_scaled)])
    
    # Split for validation
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
    
    model_components = {
        'tfidf': tfidf,
        'scaler': scaler
    }
    
    return X_train, y_train, X_valid, y_valid, X_test, model_components


def train_model(X_train, y_train, X_valid, y_valid):
    """Train Ridge model"""
    
    print("\n🚀 Training Ridge regression...")
    model = Ridge(alpha=5.0, random_state=42)