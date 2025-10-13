# ============================================================
# multimodal_model.py (FULL VERSION)
# ------------------------------------------------------------
# This is the final model. It combines the best text features
# with the newly extracted image features for the full 75k dataset.
# ============================================================

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb
from tqdm import tqdm
import re
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
# --- KEY CHANGE: Use the FULL dataset files ---
TRAIN_CSV = "dataset/train.csv"
IMG_FEATURES = "dataset/train_image_features.npy"
BEST_TEXT_ONLY_SMAPE = 52.72 # Our official baseline score
# -----------------------------------------------

# --- Helper Functions & Feature Engineering (Identical to before) ---
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

def engineer_features(df):
    tqdm.pandas(desc="Engineering text features")
    df['clean_text'] = df['catalog_content'].progress_apply(clean_text)
    df['ipq'] = df['catalog_content'].progress_apply(extract_ipq)
    df['log_ipq'] = np.log1p(df['ipq'])
    df['text_length'] = df['clean_text'].apply(len)
    df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))
    return df

# --- Main Training ---
def main():
    print("=" * 70)
    print("🚀 TRAINING FINAL MULTIMODAL MODEL (FULL 75k DATASET)")
    print("=" * 70)

    # 1. Load data
    print("📥 Loading full text data and image features...")
    train_df = pd.read_csv(TRAIN_CSV)
    train_img_features = np.load(IMG_FEATURES)
    print(f"✅ Loaded {len(train_df)} samples.")

    # 2. Engineer text features
    train_df = engineer_features(train_df)

    # 3. Create text feature matrix
    # Use more features for the full dataset
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2)) 
    train_text_tfidf = vectorizer.fit_transform(train_df['clean_text'])
    numeric_cols = ['ipq', 'log_ipq', 'text_length', 'word_count']
    scaler = StandardScaler()
    train_numeric_scaled = scaler.fit_transform(train_df[numeric_cols])

    # 4. ✨ Combine text and image features ✨
    print("✨ Combining text and image features...")
    X = hstack([
        train_text_tfidf,
        csr_matrix(train_numeric_scaled),
        csr_matrix(train_img_features)
    ]).tocsr()
    y = np.log1p(train_df['price'])
    print(f"✅ Combined feature matrix shape: {X.shape}")

    # 5. Split and Train
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)

    print("🚀 Training with tuned hyperparameters...")
    lgbm = lgb.LGBMRegressor(
        objective='regression_l1', metric='mae',
        learning_rate=0.05, n_estimators=1500, num_leaves=50, reg_alpha=0.1,
        n_jobs=-1, seed=42
    )

    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric='mae',
        callbacks=[lgb.early_stopping(100, verbose=True)] # See the progress this time
    )

    # 6. Evaluate and Compare
    valid_preds_log = lgbm.predict(X_valid)
    valid_preds = np.expm1(valid_preds_log)
    y_valid_orig = np.expm1(y_valid)
    val_smape = smape(y_valid_orig, valid_preds)

    print("\n" + "=" * 70)
    print("🏆 FINAL RESULTS")
    print("=" * 70)
    print(f"   - Best Text-Only SMAPE:     {BEST_TEXT_ONLY_SMAPE:.2f}%")
    print(f"   - FINAL MULTIMODAL SMAPE:   {val_smape:.2f}%")
    print("-" * 70)
    
    improvement = BEST_TEXT_ONLY_SMAPE - val_smape
    if improvement > 0.1: # Require a meaningful improvement
        print(f"🎉🎉🎉 SUCCESS! The image features improved the score by {improvement:.2f}%. 🎉🎉🎉")
        print("This confirms that our multimodal strategy was a winner.")
    else:
        print("🤔 The score did not significantly improve. While the pipeline worked,")
        print("the visual features didn't add as much value as hoped for this dataset.")

if __name__ == "__main__":
    main()