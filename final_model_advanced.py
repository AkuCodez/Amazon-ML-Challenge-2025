# ============================================================
# final_model_advanced.py
# ------------------------------------------------------------
# The first step to a leaderboard score. This model adds
# advanced feature engineering, starting with BRAND extraction.
# ============================================================

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
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
BEST_TEXT_ONLY_SMAPE = 52.72

# --- Feature Engineering ---

# A sample list of brands. In a real competition, this list would be much longer.
# We can build this list by looking at the most frequent words in the text.
BRAND_LIST = [
    'sony', 'samsung', 'apple', 'microsoft', 'google', 'amazon', 'procter & gamble',
    'nike', 'adidas', 'lego', 'disney', 'marvel', 'star wars', 'hp', 'dell',
    'kitchenaid', 'cuisinart', 'oxo', 'brita', 'purina', 'pedigree', 'tide',
    'bounty', 'charmin', 'crest', 'oral-b', 'gillette', 'duracell', 'neutrogena'
]

def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator) * 100

def clean_text(text):
    if pd.isna(text): return ""
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9\s.,]', '', text.lower())).strip()

def extract_ipq(text):
    if pd.isna(text): return 1
    patterns = [r'pack of (\d+)', r'(\d+)-pack', r'(\d+)\s*pack', r'(\d+)\s*count']
    for p in patterns:
        match = re.search(p.lower(), text.lower())
        if match: return int(match.group(1))
    return 1

# --- ✨ NEW: Advanced Feature Function ---
def extract_brand(text):
    """Searches for a brand in the text."""
    if pd.isna(text): return "unknown"
    text_lower = text.lower()
    for brand in BRAND_LIST:
        if brand in text_lower:
            return brand
    return "unknown"

def engineer_features(df):
    tqdm.pandas(desc="Engineering text features")
    df['clean_text'] = df['catalog_content'].progress_apply(clean_text)
    df['ipq'] = df['catalog_content'].progress_apply(extract_ipq)
    df['log_ipq'] = np.log1p(df['ipq'])
    df['text_length'] = df['clean_text'].apply(len)
    df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))
    
    # --- ✨ ADDING THE NEW BRAND FEATURE ---
    print("Extracting brand names...")
    df['brand'] = df['catalog_content'].progress_apply(extract_brand)
    return df

# --- Main Training ---
def main():
    print("=" * 70)
    print("🚀 TRAINING ADVANCED MODEL (v1: With Brand Feature)")
    print("=" * 70)

    # 1. Load data
    train_df = pd.read_csv(TRAIN_CSV)

    # 2. Engineer features (now includes brand)
    train_df = engineer_features(train_df)

    # 3. Create feature matrices
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    train_text_tfidf = vectorizer.fit_transform(train_df['clean_text'])
    
    numeric_cols = ['ipq', 'log_ipq', 'text_length', 'word_count']
    scaler = StandardScaler()
    train_numeric_scaled = scaler.fit_transform(train_df[numeric_cols])
    
    # --- ✨ Handle the new 'brand' categorical feature ---
    le = LabelEncoder()
    train_brand_encoded = le.fit_transform(train_df['brand'])
    # Reshape for hstack
    train_brand_encoded = train_brand_encoded.reshape(-1, 1)

    # 4. Combine all features
    print("✨ Combining all features (TF-IDF + Numeric + Brand)...")
    X = hstack([
        train_text_tfidf,
        csr_matrix(train_numeric_scaled),
        csr_matrix(train_brand_encoded)
    ]).tocsr()
    y = np.log1p(train_df['price'])
    print(f"✅ Combined feature matrix shape: {X.shape}")

    # 5. Split and Train
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)

    lgbm = lgb.LGBMRegressor(
        objective='regression_l1', metric='mae',
        learning_rate=0.05, n_estimators=1500, num_leaves=50, reg_alpha=0.1,
        n_jobs=-1, seed=42
    )

    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric='mae',
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )

    # 6. Evaluate
    valid_preds_log = lgbm.predict(X_valid)
    valid_preds = np.expm1(valid_preds_log)
    y_valid_orig = np.expm1(y_valid)
    val_smape = smape(y_valid_orig, valid_preds)

    print("\n" + "=" * 70)
    print("🏆 ADVANCED MODEL RESULTS (v1)")
    print("=" * 70)
    print(f"   - Previous Best SMAPE:  {BEST_TEXT_ONLY_SMAPE:.2f}%")
    print(f"   - NEW SMAPE with Brand:   {val_smape:.2f}%")
    print("-" * 70)
    improvement = BEST_TEXT_ONLY_SMAPE - val_smape
    print(f"🚀 Improvement: {improvement:.2f}%")

if __name__ == "__main__":
    main()