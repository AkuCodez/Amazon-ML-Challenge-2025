# ============================================================
#sberttuned.py
# ------------------------------------------------------------
# The final model. This script combines our best handcrafted
# features with SBERT embeddings and uses Optuna to find the
# optimal hyperparameters for LightGBM.
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import lightgbm as lgb
from sentence_transformers import SentenceTransformer
import optuna # Import Optuna
import warnings
import re
from tqdm import tqdm

warnings.filterwarnings('ignore')

# --- Configuration ---
TRAIN_CSV = "dataset/train.csv"
LEADERBOARD_BEST = 50.72

# --- All Helper & Feature Engineering Functions ---
def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator) * 100

def clean_text(text):
    if pd.isna(text): return ""
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9\s.,]', '', text.lower())).strip()

def create_brand_list(df):
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(stop_words='english', max_features=200)
    cv.fit(df['clean_text'])
    initial_brands = ['sony', 'samsung', 'apple', 'microsoft', 'google', 'amazon', 'nike', 'adidas', 'lego', 'disney']
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
    match = re.search(r'(\d+\.?\d*)\s*(oz|lb|gb|mg|ml|in|count)', text.lower())
    if match: return float(match.group(1))
    return 0

def extract_ipq(text):
    if pd.isna(text): return 1
    text_lower = text.lower() if isinstance(text, str) else ''
    patterns = [r'pack of (\d+)', r'(\d+)-pack', r'(\d+)\s*pack']
    for p in patterns:
        match = re.search(p, text_lower)
        if match: return int(match.group(1))
    return 1

def engineer_features(df):
    tqdm.pandas(desc="Engineering handcrafted features")
    df['clean_text'] = df['catalog_content'].fillna('').progress_apply(clean_text)
    brand_list = create_brand_list(df)
    df['brand'] = df['catalog_content'].progress_apply(lambda x: extract_brand(x, brand_list))
    df['specs'] = df['catalog_content'].progress_apply(extract_specs)
    df['ipq'] = df['catalog_content'].progress_apply(extract_ipq)
    return df

# --- Main Training ---
def main():
    print("=" * 70)
    print("🚀 TRAINING FINAL TUNED SBERT MODEL (v5)")
    print("=" * 70)

    train_df = pd.read_csv(TRAIN_CSV)
    
    print("Loading SBERT model and generating embeddings...")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    sentences = train_df['catalog_content'].fillna('').tolist()
    sbert_embeddings = sbert_model.encode(sentences, show_progress_bar=True)

    train_df = engineer_features(train_df)

    print("Combining SBERT embeddings and handcrafted features...")
    scaler = StandardScaler()
    numeric_cols = ['specs', 'ipq']
    train_numeric_scaled = scaler.fit_transform(train_df[numeric_cols])
    
    le = LabelEncoder()
    train_brand_encoded = le.fit_transform(train_df['brand']).reshape(-1, 1)

    X = np.hstack([sbert_embeddings, train_numeric_scaled, train_brand_encoded])
    y = np.log1p(train_df['price'])
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)

    # --- ✨ STEP 1: Define the Hyperparameter Optimization Function ---
    def objective(trial):
        params = {
            'objective': 'regression_l1',
            'metric': 'mae',
            'n_estimators': 2000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', -1, 20),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'n_jobs': -1,
            'seed': 42,
            'boosting_type': 'gbdt',
        }
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_valid, y_valid)],
                  eval_metric='mae',
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        
        preds = model.predict(X_valid)
        mae = np.mean(np.abs(preds - y_valid))
        return mae

    # --- ✨ STEP 2: Run the Optimization ---
    print("\n" + "="*70)
    print("🤖 Starting Hyperparameter Optimization with Optuna...")
    print("="*70)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50) # Run 50 trials to find the best params

    print(f"\n✅ Optimization complete! Best MAE: {study.best_value:.4f}")
    print("Best parameters found:")
    print(study.best_params)

    # --- ✨ STEP 3: Train the Final Model with the Best Parameters ---
    print("\n" + "="*70)
    print("🚀 Training final model with BEST hyperparameters...")
    print("="*70)
    best_params = study.best_params
    best_params['n_estimators'] = 2500 # Increase estimators for final training
    best_params['objective'] = 'regression_l1'
    best_params['metric'] = 'mae'
    best_params['n_jobs'] = -1
    best_params['seed'] = 42

    final_model = lgb.LGBMRegressor(**best_params)
    final_model.fit(X_train, y_train,
                    eval_set=[(X_valid, y_valid)],
                    eval_metric='mae',
                    callbacks=[lgb.early_stopping(100, verbose=True)])

    # --- Final Evaluation ---
    valid_preds_log = final_model.predict(X_valid)
    valid_preds = np.expm1(valid_preds_log)
    y_valid_orig = np.expm1(y_valid)
    val_smape = smape(y_valid_orig, valid_preds)

    print("\n" + "=" * 70)
    print("🏆 FINAL TUNED SBERT MODEL RESULTS (v5)")
    print("=" * 70)
    print(f"   - Leaderboard Best SMAPE:   {LEADERBOARD_BEST:.2f}%")
    print(f"   - NEW SMAPE (Tuned SBERT):  {val_smape:.2f}%")
    print("-" * 70)
    improvement = LEADERBOARD_BEST - val_smape
    print(f"🚀 Improvement from Leaderboard Best: {improvement:.2f}%")
    if improvement > 0.1:
        print("🎉🎉🎉🎉🎉 VICTORY! We have achieved a new best score! 🎉🎉🎉🎉🎉")

if __name__ == "__main__":
    main()