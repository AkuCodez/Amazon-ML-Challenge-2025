# ============================================================
# create_submission.py
# ------------------------------------------------------------
# This script trains our best model (v2: Brands + Specs) on
# the FULL training dataset and generates a submission file
# for the test dataset.
# ============================================================

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb
from tqdm import tqdm
import re
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
TRAIN_CSV = "dataset/train.csv"
TEST_CSV = "dataset/test.csv" # Load the test data
SUBMISSION_FILE = "submission.csv"

# --- All Feature Engineering & Helper functions (Unchanged) ---
def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator) * 100

def clean_text(text):
    if pd.isna(text): return ""
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9\s.,]', '', text.lower())).strip()

def create_brand_list(df):
    cv = CountVectorizer(stop_words='english', max_features=200)
    cv.fit(df['clean_text'])
    initial_brands = [
        'sony', 'samsung', 'apple', 'microsoft', 'google', 'amazon', 'procter & gamble',
        'nike', 'adidas', 'lego', 'disney', 'marvel', 'star wars', 'hp', 'dell'
    ]
    return list(set(initial_brands + cv.get_feature_names_out().tolist()))

def extract_brand(text, brand_list):
    if pd.isna(text): return "unknown"
    text_lower = text.lower()
    for brand in brand_list:
        if brand in text_lower:
            return brand
    return "unknown"

def extract_specs(text):
    if pd.isna(text): return 0
    match = re.search(r'(\d+\.?\d*)\s*(oz|lb|gb|mg|ml|in|ft|yd|mm|cm|m|count)', text.lower())
    if match: return float(match.group(1))
    return 0

def extract_ipq(text):
    if pd.isna(text): return 1
    text_lower = text.lower() if isinstance(text, str) else ''
    patterns = [r'pack of (\d+)', r'(\d+)-pack', r'(\d+)\s*pack', r'(\d+)\s*count']
    for p in patterns:
        match = re.search(p, text_lower)
        if match: return int(match.group(1))
    return 1

def engineer_features(df):
    # --- FIX: Initialize tqdm for pandas ---
    tqdm.pandas(desc="Engineering features")
    
    df['clean_text'] = df['catalog_content'].fillna('').progress_apply(clean_text)
    brand_list = create_brand_list(df)
    df['brand'] = df['catalog_content'].progress_apply(lambda x: extract_brand(x, brand_list))
    df['specs'] = df['catalog_content'].progress_apply(extract_specs)
    df['log_specs'] = np.log1p(df['specs'])
    df['ipq'] = df['catalog_content'].progress_apply(extract_ipq)
    df['log_ipq'] = np.log1p(df['ipq'])
    df['text_length'] = df['clean_text'].apply(len)
    df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))
    return df

# --- Main Submission Generation ---
def main():
    print("=" * 70)
    print("🚀 GENERATING SUBMISSION FILE")
    print("=" * 70)

    # 1. Load Data
    print("📥 Loading train and test data...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    # Keep track of the test sample_ids for the final file
    test_ids = test_df['sample_id']
    
    # --- ✨ Combine train and test for consistent feature engineering ---
    combined_df = pd.concat([train_df.drop('price', axis=1), test_df], ignore_index=True)

    # 2. Engineer Features on the combined dataframe
    print("🛠️  Engineering features for all data...")
    combined_df = engineer_features(combined_df)

    # 3. Create Feature Matrices
    print("✨ Creating feature matrices...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    text_tfidf = vectorizer.fit_transform(combined_df['clean_text'])
    
    numeric_cols = ['ipq', 'log_ipq', 'text_length', 'word_count', 'specs', 'log_specs']
    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(combined_df[numeric_cols])
    
    le = LabelEncoder()
    brand_encoded = le.fit_transform(combined_df['brand']).reshape(-1, 1)

    X_full = hstack([text_tfidf, csr_matrix(numeric_scaled), csr_matrix(brand_encoded)]).tocsr()
    
    # --- ✨ Split back into train and test sets ---
    X_train_full = X_full[:len(train_df)]
    X_test = X_full[len(train_df):]
    y_train_full = np.log1p(train_df['price'])
    
    print(f"Train shape: {X_train_full.shape}, Test shape: {X_test.shape}")

    # 4. Train model on 100% of the training data
    print("🚀 Training final model on ALL training data...")
    lgbm = lgb.LGBMRegressor(
        objective='regression_l1', metric='mae', learning_rate=0.05,
        n_estimators=1500, # Use a high number, as there's no early stopping
        num_leaves=50, reg_alpha=0.1, n_jobs=-1, seed=42
    )
    lgbm.fit(X_train_full, y_train_full)

    # 5. Predict on the test set
    print("🎯 Predicting prices for the test set...")
    test_preds_log = lgbm.predict(X_test)
    test_preds = np.expm1(test_preds_log)

    # 6. Create submission file
    print(f"💾 Saving submission file to {SUBMISSION_FILE}...")
    submission_df = pd.DataFrame({'sample_id': test_ids, 'price': test_preds})
    submission_df.to_csv(SUBMISSION_FILE, index=False)
    
    print("\n" + "="*70)
    print("✅ Submission file generated successfully!")
    print("="*70)

if __name__ == "__main__":
    main()