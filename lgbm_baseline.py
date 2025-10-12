# ============================================================
# lgbm_baseline.py
# Stronger baseline using LightGBM instead of Ridge
# ============================================================

import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
import joblib
from tqdm import tqdm
import lightgbm as lgb  # <-- IMPORT LIGHTGBM

# ------------------------------
# Helper Functions (Metrics, etc.)
# ------------------------------

def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator) * 100

def clean_text(text):
    """Basic text cleaning"""
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s.,]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ------------------------------
# Feature Engineering (Same as your improved_baseline_v2.py)
# ------------------------------

def extract_ipq(text):
    """Extract Item Pack Quantity"""
    if pd.isna(text): return 1
    text_lower = text.lower()
    patterns = [r'pack of (\d+)', r'(\d+)-pack', r'(\d+)\s*pack', r'(\d+)\s*count']
    for p in patterns:
        match = re.search(p, text_lower)
        if match:
            qty = int(match.group(1))
            if 1 <= qty <= 1000: return qty
    return 1

def extract_weight_oz(text):
    """Extract weight in ounces"""
    if pd.isna(text): return 0
    text = text.lower()
    patterns = [r'(\d+\.?\d*)\s*(?:oz|ounce|ounces)', r'(\d+\.?\d*)\s*lbs']
    for i, p in enumerate(patterns):
        match = re.search(p, text)
        if match:
            val = float(match.group(1))
            return val * 16 if i == 1 else val
    return 0

def extract_volume_ml(text):
    """Extract volume in ml"""
    if pd.isna(text): return 0
    text = text.lower()
    patterns = [r'(\d+\.?\d*)\s*(?:fl oz|fluid ounce)', r'(\d+\.?\d*)\s*liters']
    for i, p in enumerate(patterns):
        match = re.search(p, text)
        if match:
            val = float(match.group(1))
            return val * 29.5735 if i == 0 else val * 1000
    return 0
    
def get_quality_score(text):
    """Simple brand/quality scoring"""
    if pd.isna(text): return 0
    score = 0
    keywords = {'organic': 2, 'natural': 1, 'gourmet': 2, 'premium': 2, 'select': 1}
    for k, v in keywords.items():
        if k in text.lower():
            score += v
    return score

def extract_comprehensive_features(df):
    """Main feature extraction function"""
    tqdm.pandas(desc="Processing")
    df['clean_text'] = df['catalog_content'].progress_apply(clean_text)
    df['ipq'] = df['catalog_content'].progress_apply(extract_ipq)
    df['weight_oz'] = df['catalog_content'].progress_apply(extract_weight_oz)
    df['volume_ml'] = df['catalog_content'].progress_apply(extract_volume_ml)
    df['quality_score'] = df['catalog_content'].progress_apply(get_quality_score)
    df['has_weight'] = (df['weight_oz'] > 0).astype(int)
    df['has_volume'] = (df['volume_ml'] > 0).astype(int)
    df['log_ipq'] = np.log1p(df['ipq'])
    df['log_weight'] = np.log1p(df['weight_oz'])
    df['log_volume'] = np.log1p(df['volume_ml'])
    df['text_length'] = df['clean_text'].apply(len)
    df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))
    return df

# ------------------------------
# Data Preparation and Training
# ------------------------------

def prepare_and_train_lgbm(train_df, test_df):
    """Prepares data and trains the LightGBM model"""
    
    print("🧹 Cleaning and extracting features...")
    train_df = extract_comprehensive_features(train_df.copy())
    test_df = extract_comprehensive_features(test_df.copy())

    # TF-IDF Features
    print("🔤 Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3), min_df=3, max_df=0.9)
    train_text_features = vectorizer.fit_transform(train_df['clean_text'])
    test_text_features = vectorizer.transform(test_df['clean_text'])

    # Numeric Features
    numeric_cols = [
        'ipq', 'weight_oz', 'volume_ml', 'quality_score', 'has_weight', 
        'has_volume', 'log_ipq', 'log_weight', 'log_volume', 
        'text_length', 'word_count'
    ]
    scaler = StandardScaler()
    train_numeric_features = scaler.fit_transform(train_df[numeric_cols])
    test_numeric_features = scaler.transform(test_df[numeric_cols])

    # Combine all features
    X = hstack([train_text_features, csr_matrix(train_numeric_features)]).tocsr()
    X_test = hstack([test_text_features, csr_matrix(test_numeric_features)]).tocsr()
    y = np.log1p(train_df['price']) # Use log-transform on price for better performance

    # Train/Validation Split
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)

    print(f"\n✅ Feature shapes:\n   Train: {X_train.shape}\n   Valid: {X_valid.shape}\n   Test: {X_test.shape}")

    # ============================================================
    # MODEL CHANGE: From Ridge to LightGBM
    # ============================================================
    print("\n🚀 Training LightGBM model...")
    
    lgbm = lgb.LGBMRegressor(
        objective='regression_l1',  # MAE, good for this kind of data
        metric='mae',
        n_estimators=2000,          # More trees
        learning_rate=0.02,         # Slower learning
        feature_fraction=0.8,       # Feature subsampling
        bagging_fraction=0.8,       # Data subsampling
        bagging_freq=1,
        lambda_l1=0.1,              # L1 regularization
        lambda_l2=0.1,              # L2 regularization
        num_leaves=31,              # Number of leaves in one tree
        verbose=-1,
        n_jobs=-1,
        seed=42,
        boosting_type='gbdt',
    )

    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric='mae',
        callbacks=[lgb.early_stopping(100, verbose=False)] # Stop if validation score doesn't improve
    )
    # ============================================================

    print("🔍 Evaluating on validation set...")
    valid_preds_log = lgbm.predict(X_valid)
    valid_preds = np.expm1(valid_preds_log) # Inverse log-transform
    y_valid_orig = np.expm1(y_valid)
    
    val_mae = mean_absolute_error(y_valid_orig, valid_preds)
    val_smape = smape(y_valid_orig, valid_preds)
    
    print(f"📊 Validation MAE: {val_mae:.3f}")
    print(f"📊 Validation SMAPE: {val_smape:.2f}%")
    
    return lgbm, vectorizer, scaler, X_test, val_smape


# ------------------------------
# Main Execution
# ------------------------------
if __name__ == "__main__":
    print("="*70)
    print("🚀 LightGBM BASELINE")
    print("="*70)

    # Load data
    print("\n📥 Loading data...")
    train_df = pd.read_csv("dataset/train.csv")
    test_df = pd.read_csv("dataset/test.csv")
    train_df = train_df.dropna(subset=["catalog_content", "price"]).reset_index(drop=True)

    # Train model
    model, vectorizer, scaler, X_test, smape_score = prepare_and_train_lgbm(train_df, test_df)

    # Generate predictions
    print("\n🎯 Generating predictions...")
    test_preds_log = model.predict(X_test)
    test_preds = np.expm1(test_preds_log)
    test_preds = np.maximum(0.01, test_preds) # Ensure prices are positive

    submission_df = pd.DataFrame({'sample_id': test_df['sample_id'], 'price': test_preds})
    submission_file = "submission_lgbm.csv"
    submission_df.to_csv(submission_file, index=False)
    print(f"✅ Saved: {submission_file}")
    
    # Save model
    joblib.dump(model, 'lgbm_model.pkl')
    print("✅ Model saved!")
    
    print("\n" + "="*70)
    print("✅ COMPLETE!")
    print("="*70)