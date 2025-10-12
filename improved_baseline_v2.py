# ============================================================
# improved_baseline_v2.py
# Better approach: Focus on features that actually correlate with price
# ============================================================

import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
import joblib
from tqdm import tqdm

# ------------------------------
# Enhanced Feature Engineering
# ------------------------------

def extract_ipq(text):
    """Extract Item Pack Quantity"""
    if pd.isna(text):
        return 1
    
    text_lower = text.lower()
    patterns = [
        r'pack of (\d+)', r'(\d+)-pack', r'(\d+)\s*pack',
        r'count[:\s]+(\d+)', r'quantity[:\s]+(\d+)',
        r'\((\d+)\s*count\)', r'set of (\d+)',
        r'(\d+)\s*ct', r'(\d+)\s*piece',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            qty = int(match.group(1))
            if 1 <= qty <= 1000:
                return qty
    return 1


def extract_weight_oz(text):
    """Extract weight in ounces (normalized)"""
    if pd.isna(text):
        return 0
    
    text_lower = text.lower()
    
    # Extract ounces
    oz_match = re.search(r'(\d+\.?\d*)\s*(?:oz|ounce)', text_lower)
    if oz_match:
        return float(oz_match.group(1))
    
    # Extract pounds (convert to oz)
    lb_match = re.search(r'(\d+\.?\d*)\s*(?:lb|pound)', text_lower)
    if lb_match:
        return float(lb_match.group(1)) * 16
    
    # Extract grams (convert to oz)
    g_match = re.search(r'(\d+\.?\d*)\s*(?:g|gram)(?!\w)', text_lower)
    if g_match:
        return float(g_match.group(1)) * 0.035274
    
    # Extract kg (convert to oz)
    kg_match = re.search(r'(\d+\.?\d*)\s*kg', text_lower)
    if kg_match:
        return float(kg_match.group(1)) * 35.274
    
    return 0


def extract_volume_ml(text):
    """Extract volume in milliliters (normalized)"""
    if pd.isna(text):
        return 0
    
    text_lower = text.lower()
    
    # Extract ml
    ml_match = re.search(r'(\d+\.?\d*)\s*ml', text_lower)
    if ml_match:
        return float(ml_match.group(1))
    
    # Extract liters
    l_match = re.search(r'(\d+\.?\d*)\s*(?:l|liter)(?!\w)', text_lower)
    if l_match:
        return float(l_match.group(1)) * 1000
    
    # Extract fl oz
    floz_match = re.search(r'(\d+\.?\d*)\s*(?:fl|fluid)\s*oz', text_lower)
    if floz_match:
        return float(floz_match.group(1)) * 29.5735
    
    # Extract gallons
    gal_match = re.search(r'(\d+\.?\d*)\s*gallon', text_lower)
    if gal_match:
        return float(gal_match.group(1)) * 3785.41
    
    return 0


def extract_brand_quality_score(text):
    """Score based on premium/quality keywords"""
    if pd.isna(text):
        return 0
    
    text_lower = text.lower()
    
    premium_words = {
        'premium': 3, 'luxury': 3, 'deluxe': 2, 'professional': 2,
        'organic': 2, 'natural': 1, 'gourmet': 2, 'artisan': 2,
        'authentic': 1, 'original': 1, 'certified': 1, 'best': 1,
        'top': 1, 'superior': 2, 'excellent': 1, 'finest': 2,
        'elite': 2, 'exclusive': 2, 'signature': 1
    }
    
    score = sum(weight for word, weight in premium_words.items() if word in text_lower)
    return min(score, 10)  # Cap at 10


def extract_text_features(text):
    """Extract text-based features"""
    if pd.isna(text):
        return {
            'text_length': 0,
            'word_count': 0,
            'avg_word_length': 0,
            'uppercase_ratio': 0,
            'number_count': 0,
            'has_bullet_points': 0
        }
    
    words = text.split()
    
    return {
        'text_length': len(text),
        'word_count': len(words),
        'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
        'number_count': len(re.findall(r'\d+', text)),
        'has_bullet_points': 1 if 'bullet point' in text.lower() else 0
    }


def create_comprehensive_features(df):
    """Create all features from catalog_content"""
    print("🔧 Extracting comprehensive features...")
    
    features = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        text = row['catalog_content']
        
        # Extract all features
        ipq = extract_ipq(text)
        weight = extract_weight_oz(text)
        volume = extract_volume_ml(text)
        quality_score = extract_brand_quality_score(text)
        
        text_feats = extract_text_features(text)
        
        # Combine features
        feat_dict = {
            'ipq': ipq,
            'weight_oz': weight,
            'volume_ml': volume,
            'quality_score': quality_score,
            'has_weight': 1 if weight > 0 else 0,
            'has_volume': 1 if volume > 0 else 0,
            'log_ipq': np.log1p(ipq),
            'log_weight': np.log1p(weight),
            'log_volume': np.log1p(volume),
        }
        
        # Add text features
        feat_dict.update(text_feats)
        
        features.append(feat_dict)
    
    return pd.DataFrame(features)


def clean_text_keep_numbers(text):
    """Clean text but KEEP numbers"""
    if pd.isna(text):
        return ""
    
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.mean(diff) * 100


# ------------------------------
# Main Pipeline
# ------------------------------

def prepare_improved_features(train_df, test_df, test_size=0.1, random_state=42):
    """Prepare improved feature set"""
    
    # Clean text
    print("\n🧹 Cleaning text...")
    tqdm.pandas(desc="Clean train")
    train_df["clean_text"] = train_df["catalog_content"].progress_apply(clean_text_keep_numbers)
    
    tqdm.pandas(desc="Clean test")
    test_df["clean_text"] = test_df["catalog_content"].progress_apply(clean_text_keep_numbers)
    
    # Extract manual features
    print("\n📊 Extracting manual features...")
    train_features = create_comprehensive_features(train_df)
    test_features = create_comprehensive_features(test_df)
    
    # Show feature correlations
    print("\n📈 Feature Correlations with Price:")
    print("=" * 50)
    for col in train_features.columns:
        corr = train_features[col].corr(train_df['price'])
        print(f"  {col:20s}: {corr:>7.3f}")
    
    # TF-IDF with character n-grams
    print("\n🔤 Creating TF-IDF features...")
    
    # Word-level TF-IDF
    word_tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.85,
        stop_words='english',
        sublinear_tf=True
    )
    
    # Character-level TF-IDF
    char_tfidf = TfidfVectorizer(
        analyzer='char',
        ngram_range=(3, 5),
        max_features=2000,
        min_df=5
    )
    
    # Fit on all text
    all_text = pd.concat([train_df["clean_text"], test_df["clean_text"]])
    word_tfidf.fit(all_text)
    char_tfidf.fit(all_text)
    
    # Transform
    train_word_tfidf = word_tfidf.transform(train_df["clean_text"])
    train_char_tfidf = char_tfidf.transform(train_df["clean_text"])
    test_word_tfidf = word_tfidf.transform(test_df["clean_text"])
    test_char_tfidf = char_tfidf.transform(test_df["clean_text"])
    
    # Scale manual features
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)
    
    # Combine all features
    X_train_combined = hstack([
        train_word_tfidf, 
        train_char_tfidf,
        csr_matrix(train_features_scaled)
    ])
    
    X_test_combined = hstack([
        test_word_tfidf,
        test_char_tfidf,
        csr_matrix(test_features_scaled)
    ])
    
    # Split for validation
    train_df["log_price"] = np.log1p(train_df["price"])
    
    train_indices, valid_indices = train_test_split(
        range(len(train_df)), 
        test_size=test_size, 
        random_state=random_state
    )
    
    X_train = X_train_combined[train_indices]
    X_valid = X_train_combined[valid_indices]
    y_train = train_df["log_price"].iloc[train_indices]
    y_valid = train_df["log_price"].iloc[valid_indices]
    X_test = X_test_combined
    
    preprocessing = {
        'word_tfidf': word_tfidf,
        'char_tfidf': char_tfidf,
        'scaler': scaler
    }
    
    return X_train, y_train, X_valid, y_valid, X_test, preprocessing


def train_improved_model(X_train, y_train, X_valid, y_valid):
    """Train Ridge model with better regularization"""
    
    print("\n🚀 Training Ridge regression...")
    model = Ridge(alpha=10.0, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    print("🔍 Evaluating on validation set...")
    val_preds = model.predict(X_valid)
    val_preds_exp = np.expm1(val_preds)
    y_valid_exp = np.expm1(y_valid)
    
    mae = mean_absolute_error(y_valid_exp, val_preds_exp)
    smape_val = smape(y_valid_exp, val_preds_exp)
    
    print(f"\n📊 Validation MAE: {mae:.3f}")
    print(f"📊 Validation SMAPE: {smape_val:.2f}%")
    
    return model, smape_val


def generate_submission(model, X_test, test_df, filename="submission_improved_v2.csv"):
    """Generate predictions"""
    
    print("\n🎯 Generating predictions...")
    test_preds_log = model.predict(X_test)
    test_preds = np.expm1(test_preds_log)
    test_preds = np.maximum(test_preds, 0.01)
    
    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': test_preds
    })
    
    submission.to_csv(filename, index=False)
    print(f"✅ Saved: {filename}")
    
    return submission


# ------------------------------
# Main Execution
# ------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 IMPROVED BASELINE V2 - BETTER FEATURE ENGINEERING")
    print("=" * 70)
    print("\nKey improvements:")
    print("  ✅ Extract weight & volume (normalized)")
    print("  ✅ Quality/brand scoring")
    print("  ✅ Character-level n-grams")
    print("  ✅ Log-transformed features")
    print("  ✅ Feature correlation analysis")
    print("=" * 70)
    
    # Load data
    print("\n📥 Loading data...")
    train_df = pd.read_csv("dataset/train.csv")
    test_df = pd.read_csv("dataset/test.csv")
    
    print(f"✅ Train: {train_df.shape}")
    print(f"✅ Test: {test_df.shape}")
    
    # Clean
    train_df = train_df.dropna(subset=["catalog_content", "price"]).reset_index(drop=True)
    
    # Prepare features
    X_train, y_train, X_valid, y_valid, X_test, preprocessing = prepare_improved_features(
        train_df, test_df
    )
    
    print(f"\n✅ Feature shapes:")
    print(f"   Train: {X_train.shape}")
    print(f"   Valid: {X_valid.shape}")
    print(f"   Test: {X_test.shape}")
    
    # Train
    model, smape_val = train_improved_model(X_train, y_train, X_valid, y_valid)
    
    # Compare with baseline
    baseline_smape = 57.15
    improvement = baseline_smape - smape_val
    improvement_pct = (improvement / baseline_smape) * 100
    
    print("\n" + "=" * 70)
    print("📈 RESULTS COMPARISON")
    print("=" * 70)
    print(f"Baseline SMAPE:    {baseline_smape:.2f}%")
    print(f"Improved SMAPE:    {smape_val:.2f}%")
    print(f"Improvement:       {improvement:.2f}% ({improvement_pct:.1f}% relative)")
    print("=" * 70)
    
    # Generate submission
    submission = generate_submission(model, X_test, test_df)
    
    # Save model
    print("\n💾 Saving model...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/improved_v2_model.pkl")
    joblib.dump(preprocessing, "models/improved_v2_preprocessing.pkl")
    print("✅ Model saved!")
    
    print("\n" + "=" * 70)
    print("✅ COMPLETE!")
    print("=" * 70)