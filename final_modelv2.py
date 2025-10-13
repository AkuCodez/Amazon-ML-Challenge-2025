# ============================================================
# final_model_advanced_v2.py
# ------------------------------------------------------------
# This version adds a more comprehensive brand list and
# extracts numerical specifications (e.g., 500mg, 12-inch).
# ============================================================

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb
from tqdm import tqdm
import re
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
TRAIN_CSV = "dataset/train.csv"
PREVIOUS_BEST_SMAPE = 52.53 # From our v1 advanced model

# --- Feature Engineering ---

def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator) * 100

def clean_text(text):
    if pd.isna(text): return ""
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9\s.,]', '', text.lower())).strip()

def extract_ipq(text):
    """Extracts item-per-unit quantity (e.g., 'pack of 12')."""
    if pd.isna(text): return 1
    # This function should use the raw text to catch all variations
    text_lower = text.lower() if isinstance(text, str) else ''
    patterns = [r'pack of (\d+)', r'(\d+)-pack', r'(\d+)\s*pack', r'(\d+)\s*count']
    for p in patterns:
        match = re.search(p, text_lower)
        if match:
            return int(match.group(1))
    return 1
# --- ✨ NEW: Data-Driven Brand List ---
def create_brand_list(df):
    """Analyzes text to create a more comprehensive list of brands."""
    print("Building data-driven brand list...")
    # Use CountVectorizer to find the most common single words (potential brands)
    cv = CountVectorizer(stop_words='english', max_features=200)
    cv.fit(df['clean_text'])
    # Add our original list for good measure
    initial_brands = [
        'sony', 'samsung', 'apple', 'microsoft', 'google', 'amazon', 'procter & gamble',
        'nike', 'adidas', 'lego', 'disney', 'marvel', 'star wars', 'hp', 'dell',
        'kitchenaid', 'cuisinart', 'oxo', 'brita', 'purina', 'pedigree', 'tide',
        'bounty', 'charmin', 'crest', 'oral-b', 'gillette', 'duracell', 'neutrogena'
    ]
    # Combine and remove duplicates
    potential_brands = list(set(initial_brands + cv.get_feature_names_out().tolist()))
    print(f"Found {len(potential_brands)} potential brand names.")
    return potential_brands

def extract_brand(text, brand_list):
    if pd.isna(text): return "unknown"
    text_lower = text.lower()
    for brand in brand_list:
        if brand in text_lower:
            return brand
    return "unknown"

# --- ✨ NEW: Numerical Specification Extractor ---
def extract_specs(text):
    """Extracts numerical values like weight, size, etc."""
    if pd.isna(text): return 0
    # This regex looks for a number followed by a common unit (oz, lb, gb, mg, ml, etc.)
    # It's a simple start and can be expanded.
    match = re.search(r'(\d+\.?\d*)\s*(oz|lb|gb|mg|ml|in|ft|yd|mm|cm|m|count)', text.lower())
    if match:
        return float(match.group(1))
    return 0

def engineer_features(df):
    tqdm.pandas(desc="Cleaning text")
    df['clean_text'] = df['catalog_content'].progress_apply(clean_text)
    
    # --- Adding the NEW features ---
    brand_list = create_brand_list(df)
    tqdm.pandas(desc="Extracting brand names")
    df['brand'] = df['catalog_content'].progress_apply(lambda x: extract_brand(x, brand_list))
    
    tqdm.pandas(desc="Extracting numerical specs")
    df['specs'] = df['catalog_content'].progress_apply(extract_specs)
    df['log_specs'] = np.log1p(df['specs'])

    # --- Existing features ---
    # Extract IPQ from the original content to be safe
    tqdm.pandas(desc="Extracting pack quantity (IPQ)")
    df['ipq'] = df['catalog_content'].progress_apply(extract_ipq)
    df['log_ipq'] = np.log1p(df['ipq'])
    df['text_length'] = df['clean_text'].apply(len)
    df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))
    return df

# --- Main Training ---
def main():
    print("=" * 70)
    print("🚀 TRAINING ADVANCED MODEL (v2: Brands + Specs)")
    print("=" * 70)

    train_df = pd.read_csv(TRAIN_CSV)
    train_df = engineer_features(train_df)

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    train_text_tfidf = vectorizer.fit_transform(train_df['clean_text'])
    
    # --- ✨ Add the NEW numerical spec features ---
    numeric_cols = ['ipq', 'log_ipq', 'text_length', 'word_count', 'specs', 'log_specs']
    scaler = StandardScaler()
    train_numeric_scaled = scaler.fit_transform(train_df[numeric_cols])
    
    le = LabelEncoder()
    train_brand_encoded = le.fit_transform(train_df['brand']).reshape(-1, 1)

    print("✨ Combining all features...")
    X = hstack([
        train_text_tfidf,
        csr_matrix(train_numeric_scaled),
        csr_matrix(train_brand_encoded)
    ]).tocsr()
    y = np.log1p(train_df['price'])
    print(f"✅ Combined feature matrix shape: {X.shape}")

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)

    lgbm = lgb.LGBMRegressor(
        objective='regression_l1', metric='mae', learning_rate=0.05,
        n_estimators=1500, num_leaves=50, reg_alpha=0.1, n_jobs=-1, seed=42
    )

    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric='mae',
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )

    valid_preds_log = lgbm.predict(X_valid)
    valid_preds = np.expm1(valid_preds_log)
    y_valid_orig = np.expm1(y_valid)
    val_smape = smape(y_valid_orig, valid_preds)

    print("\n" + "=" * 70)
    print("🏆 ADVANCED MODEL RESULTS (v2)")
    print("=" * 70)
    print(f"   - Previous Best SMAPE:  {PREVIOUS_BEST_SMAPE:.2f}%")
    print(f"   - NEW SMAPE (Brands+Specs):   {val_smape:.2f}%")
    print("-" * 70)
    improvement = PREVIOUS_BEST_SMAPE - val_smape
    print(f"🚀 Improvement from v1: {improvement:.2f}%")

if __name__ == "__main__":
    main()