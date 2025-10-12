# ============================================================
# lgbm_v2.py
# Added Brand Extraction and Hyperparameter Tuning
# Goal: Break into the 40s SMAPE
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
import lightgbm as lgb

# ------------------------------
# Helper Functions
# ------------------------------

def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator) * 100

def clean_text(text):
    if pd.isna(text): return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s.,]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ------------------------------
# Feature Engineering v2
# ------------------------------

def extract_ipq(text):
    if pd.isna(text): return 1
    text_lower = text.lower()
    patterns = [r'pack of (\d+)', r'(\d+)-pack', r'(\d+)\s*pack', r'(\d+)\s*count']
    for p in patterns:
        match = re.search(p, text_lower)
        if match:
            qty = int(match.group(1))
            if 1 <= qty <= 1000: return qty
    return 1

# NEW: Brand Extraction Function
def extract_brand(text):
    """Extracts a potential brand name from the start of the text."""
    if pd.isna(text):
        return "unknown"
    words = text.split()
    # Common English stop words and conjunctions to ignore
    stop_words = {'a', 'an', 'the', 'for', 'with', 'and', 'or', 'of', 'in', 'to', 'by', 'on'}
    for word in words[:5]: # Check first 5 words
        # A brand is often a capitalized word (but not all-caps like 'USDA')
        if word.istitle() and word.lower() not in stop_words and len(word) > 1:
            cleaned_word = re.sub(r'[^\w\s-]', '', word) # Allow hyphens in brands
            return cleaned_word.lower()
    return "unknown"

def extract_comprehensive_features(df):
    """Main feature extraction function, now with brand extraction."""
    tqdm.pandas(desc="Processing")
    # Text cleaning is now part of feature extraction
    df['clean_text'] = df['catalog_content'].progress_apply(clean_text)
    
    # NEW: Extract Brand
    print("✨ Extracting brands...")
    df['brand'] = df['catalog_content'].progress_apply(extract_brand)
    # Convert brand to a categorical type for LightGBM
    df['brand_cat'] = df['brand'].astype('category').cat.codes
    
    print("🔧 Extracting other features...")
    df['ipq'] = df['catalog_content'].progress_apply(extract_ipq)
    df['text_length'] = df['clean_text'].apply(len)
    df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))
    df['log_ipq'] = np.log1p(df['ipq'])
    
    return df

# ------------------------------
# Data Preparation and Training
# ------------------------------

def prepare_and_train_lgbm_v2(train_df, test_df):
    """Prepares data and trains the tuned LightGBM model"""
    
    print("🧹 Extracting features for train and test...")
    train_df = extract_comprehensive_features(train_df.copy())
    test_df = extract_comprehensive_features(test_df.copy())

    # TF-IDF Features
    print("🔤 Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=7000, ngram_range=(1, 3), min_df=3)
    train_text_features = vectorizer.fit_transform(train_df['clean_text'])
    test_text_features = vectorizer.transform(test_df['clean_text'])

    # Numeric & Categorical Features
    # Note: We now include our new brand category feature
    numeric_cols = ['ipq', 'text_length', 'word_count', 'log_ipq', 'brand_cat']
    
    scaler = StandardScaler()
    train_numeric_features = scaler.fit_transform(train_df[numeric_cols])
    test_numeric_features = scaler.transform(test_df[numeric_cols])

    # Combine all features
    X = hstack([train_text_features, csr_matrix(train_numeric_features)]).tocsr()
    X_test = hstack([test_text_features, csr_matrix(test_numeric_features)]).tocsr()
    y = np.log1p(train_df['price'])

    # Train/Validation Split
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)

    print(f"\n✅ Feature shapes:\n   Train: {X_train.shape}\n   Valid: {X_valid.shape}\n   Test: {X_test.shape}")

    # ============================================================
    # UPDATED: Tuned LightGBM Hyperparameters
    # ============================================================
    print("\n🚀 Training tuned LightGBM model...")
    
    lgbm = lgb.LGBMRegressor(
        objective='regression_l1',
        metric='mae',
        n_estimators=3000,          # More trees
        learning_rate=0.01,         # Much smaller learning rate
        num_leaves=64,              # More complex trees
        max_depth=10,               # Limit tree depth
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l1=0.1,
        lambda_l2=0.1,
        colsample_bytree=0.7,       # Added feature subsampling per tree
        verbose=-1,
        n_jobs=-1,
        seed=42,
    )

    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric='mae',
        callbacks=[lgb.early_stopping(150, verbose=True)] # More patience for early stopping
    )
    # ============================================================

    print("🔍 Evaluating on validation set...")
    valid_preds_log = lgbm.predict(X_valid)
    valid_preds = np.expm1(valid_preds_log)
    y_valid_orig = np.expm1(y_valid)
    
    val_mae = mean_absolute_error(y_valid_orig, valid_preds)
    val_smape = smape(y_valid_orig, valid_preds)
    
    print(f"📊 Validation MAE: {val_mae:.3f}")
    print(f"📊 Validation SMAPE: {val_smape:.2f}%")
    
    return lgbm, vectorizer, scaler, X_test

# ------------------------------
# Main Execution
# ------------------------------
if __name__ == "__main__":
    print("="*70)
    print("🚀 LightGBM V2 - Brand Features & Tuning")
    print("="*70)

    # Load data
    train_df = pd.read_csv("dataset/train.csv")
    test_df = pd.read_csv("dataset/test.csv")
    train_df = train_df.dropna(subset=["catalog_content", "price"]).reset_index(drop=True)

    # Train model
    model, vectorizer, scaler, X_test = prepare_and_train_lgbm_v2(train_df, test_df)

    # Generate predictions
    print("\n🎯 Generating predictions...")
    test_preds_log = model.predict(X_test)
    test_preds = np.expm1(test_preds_log)
    test_preds = np.maximum(0.01, test_preds)

    submission_df = pd.DataFrame({'sample_id': test_df['sample_id'], 'price': test_preds})
    submission_file = "submission_lgbm_v2.csv"
    submission_df.to_csv(submission_file, index=False)
    print(f"✅ Saved: {submission_file}")
    
    joblib.dump(model, 'lgbm_model_v2.pkl')
    print("✅ Model saved!")
    print("\n" + "="*70 + "\n✅ COMPLETE!\n" + "="*70)