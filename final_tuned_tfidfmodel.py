# ============================================================
# final_tuned_tfidf_model.py
# ------------------------------------------------------------
# The final, most reliable model. This script takes our best
# feature engineering (TF-IDF + Handcrafted) and uses Optuna
# to find its absolute best hyperparameters.
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb
import optuna
import warnings
import re
from tqdm import tqdm

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING) # Keep the output clean

# --- Configuration ---
TRAIN_CSV = "dataset/train.csv"
LEADERBOARD_BEST = 50.72

# --- Feature Engineering & Helper Functions (from our best model) ---
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
    initial_brands = ['sony', 'samsung', 'apple', 'microsoft', 'google', 'amazon', 'nike', 'adidas']
    return list(set(initial_brands + cv.get_feature_names_out().tolist()))

def extract_brand(text, brand_list):
    if pd.isna(text): return "unknown"
    for brand in brand_list:
        if brand in text.lower():
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

# --- Main Training ---
def main():
    print("=" * 70)
    print("🚀 OPTIMIZING OUR CHAMPION TF-IDF MODEL")
    print("=" * 70)

    # 1. Load and prepare data
    train_df = pd.read_csv(TRAIN_CSV)
    train_df['clean_text'] = train_df['catalog_content'].fillna('').apply(clean_text)
    
    brand_list = create_brand_list(train_df)
    train_df['brand'] = train_df['catalog_content'].apply(lambda x: extract_brand(x, brand_list))
    train_df['specs'] = train_df['catalog_content'].apply(extract_specs)
    train_df['ipq'] = train_df['catalog_content'].apply(extract_ipq)

    # 2. Create feature matrix
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    text_tfidf = vectorizer.fit_transform(train_df['clean_text'])
    
    numeric_cols = ['specs', 'ipq']
    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(train_df[numeric_cols])
    
    le = LabelEncoder()
    brand_encoded = le.fit_transform(train_df['brand']).reshape(-1, 1)

    X = hstack([text_tfidf, csr_matrix(numeric_scaled), csr_matrix(brand_encoded)]).tocsr()
    y = np.log1p(train_df['price'])
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)

    # 3. Define the Optuna objective function
    def objective(trial):
        params = {
            'objective': 'regression_l1', 'metric': 'mae', 'n_estimators': 1500,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', -1, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 80),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 0.5),
            'n_jobs': -1, 'seed': 42,
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], callbacks=[lgb.early_stopping(100, verbose=False)])
        preds = model.predict(X_valid)
        y_valid_orig = np.expm1(y_valid)
        preds_orig = np.expm1(preds)
        return smape(y_valid_orig, preds_orig) # Optimize directly for SMAPE!

    # 4. Run the optimization
    print("\n" + "="*70)
    print("🤖 Starting Hyperparameter Optimization (30 Trials)...")
    print("="*70)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30) # A solid search that respects our time limit

    # 5. Final Evaluation with best parameters
    best_smape = study.best_value
    print("\n" + "=" * 70)
    print("🏆 FINAL TUNED TF-IDF MODEL RESULTS")
    print("=" * 70)
    print(f"   - Leaderboard Best SMAPE:   {LEADERBOARD_BEST:.2f}%")
    print(f"   - NEW Best SMAPE Found:     {best_smape:.2f}%")
    print("-" * 70)
    improvement = LEADERBOARD_BEST - best_smape
    print(f"🚀 Improvement from Leaderboard Best: {improvement:.2f}%")
    if improvement > 0.1:
        print("🎉🎉🎉🎉🎉 VICTORY! We found a better model! 🎉🎉🎉🎉🎉")
        print("\nBest parameters found:")
        print(study.best_params)
    else:
        print("🤔 No significant improvement. Our original settings were already very strong.")


if __name__ == "__main__":
    main()