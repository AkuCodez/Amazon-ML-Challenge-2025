"""
SAMPLE CODE — BASELINE EXECUTION PIPELINE
-----------------------------------------
This script demonstrates how to:
1. Load train/test datasets
2. Preprocess data using the same transformations as baseline_model.py
3. Train a baseline model (LightGBM)
4. Generate predictions
5. Save submission file (ready for leaderboard upload)
"""

import pandas as pd
import os
from baseline_model import (
    prepare_features,
    train_and_evaluate,
    predict_and_prepare_submission
)

# ----------------------------
# Step 1: Path Configuration
# ----------------------------
DATA_PATH = "data/"  # Folder containing train.csv, test.csv
TRAIN_FILE = os.path.join(DATA_PATH, "train.csv")
TEST_FILE = os.path.join(DATA_PATH, "test.csv")
SUBMISSION_FILE = "submission.csv"

# ----------------------------
# Step 2: Load Data
# ----------------------------
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print("✅ Data Loaded:")
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# ----------------------------
# Step 3: Prepare Features
# ----------------------------
# The prepare_features() handles feature extraction, encoding, scaling, etc.
X_train, y_train, X_valid, y_valid, X_test = prepare_features(train_df, test_df)

print(f"Prepared Feature Shapes - Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")

# ----------------------------
# Step 4: Train and Evaluate
# ----------------------------
model, valid_smape = train_and_evaluate(X_train, y_train, X_valid, y_valid)

print(f"📊 Validation SMAPE: {valid_smape:.4f}")

# ----------------------------
# Step 5: Predict on Test Set
# ----------------------------
submission = predict_and_prepare_submission(model, X_test, test_df)
submission.to_csv(SUBMISSION_FILE, index=False)

print(f"✅ Submission file saved: {SUBMISSION_FILE}")
print(submission.head())