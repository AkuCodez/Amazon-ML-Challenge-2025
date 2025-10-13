# ============================================================
# final_sbert_refined.py
# ------------------------------------------------------------
# The final refined model. This script uses a fast SBERT
# model, denoises its features with TruncatedSVD, combines
# them with our proven TF-IDF & handcrafted features, and
# trains a LightGBM model optimized directly for SMAPE.
# ============================================================

import pandas as pd
import numpy as np
import re
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# --- Configuration ---
LEADERBOARD_BEST = 50.72

# --- Helper & Feature Engineering Functions ---

def smape(y_true, y_pred):
    """Calculates SMAPE for evaluation."""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator) * 100

def engineer_features(df):
    """A single function to perform all our proven feature engineering."""
    tqdm.pandas(desc="Engineering features")
    
    # 1. Clean Text
    df['clean_text'] = df['catalog_content'].fillna('').progress_apply(
        lambda x: re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9\s.,]', '', x.lower())).strip()
    )
    
    # 2. Extract Brand
    cv = CountVectorizer(stop_words='english', max_features=200)
    cv.fit(df['clean_text'])
    initial_brands = ['sony', 'samsung', 'apple', 'microsoft', 'google', 'amazon', 'nike', 'adidas']
    brand_list = list(set(initial_brands + cv.get_feature_names_out().tolist()))
    df['brand'] = df['catalog_content'].progress_apply(
        lambda x: next((b for b in brand_list if b in x.lower()), "unknown") if pd.notna(x) else "unknown"
    )

    # 3. Extract Specs and IPQ
    df['specs'] = df['catalog_content'].progress_apply(
        lambda x: float(re.search(r'(\d+\.?\d*)\s*(oz|lb|gb|mg|ml|in|count)', x.lower()).group(1)) if pd.notna(x) and re.search(r'(\d+\.?\d*)\s*(oz|lb|gb|mg|ml|in|count)', x.lower()) else 0
    )
    df['ipq'] = df['catalog_content'].progress_apply(
        lambda x: int(next(filter(None, re.search(r'pack of (\d+)|(\d+)-pack|(\d+)\s*pack', x.lower()).groups()), 1)) if pd.notna(x) and re.search(r'pack of (\d+)|(\d+)-pack|(\d+)\s*pack', x.lower()) else 1
    )
    return df

# --- Main Model Function ---

def create_refined_sbert_model():
    """
    Main function to build, train, and evaluate the refined model.
    """
    print("=" * 70)
    print("🚀 TRAINING THE FINAL REFINED SBERT MODEL")
    print("=" * 70)

    # 1. Load data
    train_df = pd.read_csv("dataset/train.csv")
    
    # 2. Perform all handcrafted feature engineering
    train_df = engineer_features(train_df)
    
    # 3. Generate SBERT embeddings and denoise with SVD
    print("\nStep 1/4: Generating and denoising SBERT embeddings...")
    sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = sbert_model.encode(train_df['clean_text'].tolist(), 
                                    batch_size=256, 
                                    show_progress_bar=True)
    svd = TruncatedSVD(n_components=50, random_state=42)
    embeddings_reduced = svd.fit_transform(embeddings)
    print(f"✅ SBERT features denoised to {embeddings_reduced.shape[1]} dimensions.")

    # 4. Generate TF-IDF features
    print("\nStep 2/4: Generating TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    tfidf_features = vectorizer.fit_transform(train_df['clean_text'])
    print(f"✅ TF-IDF features created with shape: {tfidf_features.shape}")

    # 5. Process numeric and categorical handcrafted features
    print("\nStep 3/4: Processing handcrafted features...")
    scaler = StandardScaler()
    numeric_features = scaler.fit_transform(train_df[['specs', 'ipq']])
    
    le = LabelEncoder()
    brand_features = le.fit_transform(train_df['brand']).reshape(-1, 1)
    
    # 6. Combine all features with strategic weights
    print("\nStep 4/4: Combining all features with strategic weights...")
    X_combined = hstack([
        tfidf_features * 0.6,
        csr_matrix(embeddings_reduced) * 0.2,
        csr_matrix(numeric_features) * 0.1,
        csr_matrix(brand_features) * 0.1
    ]).tocsr()
    print(f"✅ Final feature matrix created with shape: {X_combined.shape}")

    y = np.log1p(train_df['price'])
    X_train, X_valid, y_train, y_valid = train_test_split(X_combined, y, test_size=0.1, random_state=42)

    # 7. Define custom SMAPE evaluation metric for LightGBM
    def lgbm_smape_metric(y_true_log, y_pred_log):
        y_true = np.expm1(y_true_log)
        y_pred = np.expm1(y_pred_log)
        val_smape = smape(y_true, y_pred)
        return 'smape', val_smape, False # False means lower is better

    # 8. Train the final model
    print("\n🚀 Training final LightGBM model...")
    lgb_params = {
        'objective': 'regression_l1', # Use MAE for stable training
        'metric': 'None', # We use a custom eval metric
        'n_estimators': 2000,
        'learning_rate': 0.03,
        'num_leaves': 40,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'n_jobs': -1,
        'seed': 42
    }
    
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X_train, y_train,
              eval_set=[(X_valid, y_valid)],
              eval_metric=lgbm_smape_metric,
              callbacks=[lgb.early_stopping(100, verbose=True)])
    
    # Final Evaluation
    best_score = model.best_score_['valid_0']['smape']
    print("\n" + "=" * 70)
    print("🏆 FINAL REFINED MODEL RESULTS")
    print("=" * 70)
    print(f"   - Leaderboard Best SMAPE:   {LEADERBOARD_BEST:.2f}%")
    print(f"   - NEW SMAPE (Refined SBERT): {best_score:.2f}%")
    print("-" * 70)
    improvement = LEADERBOARD_BEST - best_score
    print(f"🚀 Improvement from Leaderboard Best: {improvement:.2f}%")
    if improvement > 0.1:
        print("🎉🎉🎉🎉🎉 VICTORY! THIS IS OUR WINNING MODEL! 🎉🎉🎉🎉🎉")

if __name__ == "__main__":
    create_refined_sbert_model()