#!/usr/bin/env python3
"""
THE ULTIMATE SPRINT MODEL - FINAL VERSION
Combines advanced feature engineering with pre-computed image features.
Target: Sub-45 SMAPE. Designed to finish in under 90 minutes.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb
from tqdm import tqdm
import re
import warnings
import os
import gc

warnings.filterwarnings('ignore')

# ================== CONFIGURATION ==================
TRAIN_CSV = "dataset/train.csv"
TEST_CSV = "dataset/test.csv"
# !! CRITICAL: USE YOUR PRE-COMPUTED IMAGE FEATURES !!
TRAIN_IMG_FEATURES_NPY = "dataset/train_image_features.npy"
TEST_IMG_FEATURES_NPY = "dataset/test_image_features.npy" # Make sure you have this file

IMAGE_FEATURE_DIM = 512 # The dimension of your pre-computed features

# ================== ENHANCED FEATURE ENGINEERING ==================
# (Using the superior functions from the Emergency script)

def engineer_all_features(df):
    """Comprehensive feature engineering"""
    print("🔧 Engineering advanced features...")
    tqdm.pandas(desc="Cleaning text")
    df['clean_text'] = df['catalog_content'].progress_apply(
        lambda x: re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9\s.,]', '', str(x).lower())).strip()
    )
    
    def extract_brand(text):
        match = re.search(r'^([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)\s+', str(text))
        return match.group(1).lower() if match else "unknown"
    tqdm.pandas(desc="Extracting brands")
    df['brand'] = df['catalog_content'].progress_apply(extract_brand)

    def extract_advanced_features(text):
        text_lower = str(text).lower()
        features = {'ipq': 1, 'specs': 0, 'size': 0, 'weight': 0, 'volume': 0}
        ipq_match = re.search(r'(\d+)\s*(?:pack|count|pcs|ct|pk)', text_lower)
        if ipq_match: features['ipq'] = min(int(ipq_match.group(1)), 100)
        weight_match = re.search(r'(\d+\.?\d*)\s*(oz|lb|kg|g)', text_lower)
        if weight_match:
            val, unit = float(weight_match.group(1)), weight_match.group(2)
            if unit == 'oz': features['weight'] = val * 28.35
            elif unit == 'lb': features['weight'] = val * 453.6
            elif unit == 'kg': features['weight'] = val * 1000
            else: features['weight'] = val
        return features

    tqdm.pandas(desc="Extracting numerical features")
    extracted = df['catalog_content'].progress_apply(extract_advanced_features)
    df = pd.concat([df, extracted.apply(pd.Series)], axis=1)
    
    df['log_ipq'] = np.log1p(df['ipq'])
    df['log_weight'] = np.log1p(df['weight'])
    df['text_length'] = df['clean_text'].apply(len)
    df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))
    df['ipq_weight'] = df['ipq'] * df['weight']
    
    return df

# ================== MAIN TRAINING PIPELINE ==================

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(np.abs(y_pred - y_true) / (denominator + 1e-8)) * 100

def train_ultimate_model():
    print("="*80)
    print("🚀 THE ULTIMATE SPRINT - TARGET: SUB-45 SMAPE")
    print("="*80)
    
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    
    train_df = engineer_all_features(train_df)
    test_df = engineer_all_features(test_df)
    
    print("\n🖼️ Loading PRE-COMPUTED image features...")
    if os.path.exists(TRAIN_IMG_FEATURES_NPY) and os.path.exists(TEST_IMG_FEATURES_NPY):
        train_img_features = np.load(TRAIN_IMG_FEATURES_NPY)
        test_img_features = np.load(TEST_IMG_FEATURES_NPY)
        # Ensure correct shape
        if train_img_features.shape[0] != len(train_df):
            raise ValueError("Train image features have wrong number of rows!")
        global IMAGE_FEATURE_DIM
        IMAGE_FEATURE_DIM = train_img_features.shape[1] # Dynamically set dimension
        print(f"✅ Loaded image features with dimension: {IMAGE_FEATURE_DIM}")
    else:
        print("⚠️ Image feature files not found! Proceeding without image data.")
        train_img_features = np.zeros((len(train_df), 1))
        test_img_features = np.zeros((len(test_df), 1))
        IMAGE_FEATURE_DIM = 1


    print("\n📝 Creating TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 3), min_df=3, sublinear_tf=True)
    train_tfidf = tfidf.fit_transform(train_df['clean_text'])
    test_tfidf = tfidf.transform(test_df['clean_text'])
    
    print("\n🏷️ Encoding brands...")
    brand_encoder = LabelEncoder()
    all_brands = pd.concat([train_df['brand'], test_df['brand']]).unique()
    brand_encoder.fit(all_brands)
    train_brand_encoded = brand_encoder.transform(train_df['brand']).reshape(-1, 1)
    test_brand_encoded = brand_encoder.transform(test_df['brand']).reshape(-1, 1)

    print("\n🔢 Processing numerical features...")
    numeric_cols = ['ipq', 'log_ipq', 'weight', 'log_weight', 'text_length', 'word_count', 'ipq_weight']
    scaler = StandardScaler()
    train_numeric = scaler.fit_transform(train_df[numeric_cols])
    test_numeric = scaler.transform(test_df[numeric_cols])

    print("\n🔗 Combining all features...")
    X_train = hstack([train_tfidf, csr_matrix(train_numeric), csr_matrix(train_brand_encoded), csr_matrix(train_img_features)]).tocsr()
    X_test = hstack([test_tfidf, csr_matrix(test_numeric), csr_matrix(test_brand_encoded), csr_matrix(test_img_features)]).tocsr()
    y_train = np.log1p(train_df['price'])
    print(f"✅ Final feature matrix shape: {X_train.shape}")
    
    print("\n🎯 Training ensemble of models with Cross-Validation...")
    params = {
        'objective': 'quantile', 'alpha': 0.5, 'metric': 'quantile',
        'boosting_type': 'gbdt', 'num_leaves': 127, 'learning_rate': 0.02,
        'feature_fraction': 0.7, 'bagging_fraction': 0.8, 'bagging_freq': 5,
        'n_estimators': 3000, 'random_state': 42, 'n_jobs': -1, 'verbose': -1
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    test_preds_ensemble = []
    oof_preds = np.zeros(len(train_df))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        print(f"\n📊 Training Fold {fold}/5...")
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(150, verbose=False)])
        
        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred
        test_preds_ensemble.append(model.predict(X_test))
        
        fold_smape = smape(np.expm1(y_val), np.expm1(val_pred))
        print(f"  Fold {fold} SMAPE: {fold_smape:.2f}%")
        
    test_preds_log = np.mean(test_preds_ensemble, axis=0)
    test_preds = np.expm1(test_preds_log)
    
    overall_smape = smape(np.expm1(y_train), np.expm1(oof_preds))
    print(f"\n🏁 Overall CV SMAPE: {overall_smape:.2f}%")

    print("\n📈 Generating final submission...")
    submission = pd.DataFrame({'sample_id': test_df['sample_id'], 'price': test_preds})
    submission['price'] = submission['price'].clip(lower=0.01)
    submission.to_csv('submission_ultimate_sprint.csv', index=False)
    print(f"✅ Submission saved to 'submission_ultimate_sprint.csv'")
    
if __name__ == "__main__":
    train_ultimate_model()