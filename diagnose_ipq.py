# ============================================================
# diagnose_ipq.py
# Check if IPQ extraction is working properly
# ============================================================

import pandas as pd
import re
from tqdm import tqdm

def extract_ipq(text):
    """Extract Item Pack Quantity"""
    if pd.isna(text):
        return 1
    
    text_lower = text.lower()
    
    patterns = [
        r'pack of (\d+)',
        r'(\d+)-pack',
        r'(\d+)\s*pack',
        r'count[:\s]+(\d+)',
        r'quantity[:\s]+(\d+)',
        r'\((\d+)\s*count\)',
        r'set of (\d+)',
        r'(\d+)\s*ct',
        r'(\d+)\s*piece',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            qty = int(match.group(1))
            if 1 <= qty <= 1000:
                return qty
    
    return 1


print("=" * 70)
print("🔍 IPQ EXTRACTION DIAGNOSTIC")
print("=" * 70)

# Load data
print("\n📥 Loading training data...")
train_df = pd.read_csv("dataset/train.csv")
print(f"✅ Loaded {len(train_df)} rows")

# Extract IPQ
print("\n🔍 Extracting IPQ from catalog_content...")
tqdm.pandas(desc="Extracting IPQ")
train_df["ipq"] = train_df["catalog_content"].progress_apply(extract_ipq)

# Statistics
print("\n" + "=" * 70)
print("📊 IPQ STATISTICS")
print("=" * 70)

print(f"\nBasic Stats:")
print(f"  Mean IPQ: {train_df['ipq'].mean():.2f}")
print(f"  Median IPQ: {train_df['ipq'].median():.0f}")
print(f"  Min IPQ: {train_df['ipq'].min():.0f}")
print(f"  Max IPQ: {train_df['ipq'].max():.0f}")
print(f"  Std Dev: {train_df['ipq'].std():.2f}")

print(f"\nDistribution:")
print(f"  IPQ = 1 (single items): {(train_df['ipq']==1).sum():,} ({(train_df['ipq']==1).mean()*100:.1f}%)")
print(f"  IPQ > 1 (multi-packs): {(train_df['ipq']>1).sum():,} ({(train_df['ipq']>1).mean()*100:.1f}%)")
print(f"  IPQ >= 5: {(train_df['ipq']>=5).sum():,} ({(train_df['ipq']>=5).mean()*100:.1f}%)")
print(f"  IPQ >= 10: {(train_df['ipq']>=10).sum():,} ({(train_df['ipq']>=10).mean()*100:.1f}%)")

print(f"\nTop 10 Most Common IPQ Values:")
print(train_df['ipq'].value_counts().head(10))

# Correlation with price
correlation = train_df['ipq'].corr(train_df['price'])
print(f"\n📈 Correlation with price: {correlation:.3f}")

if correlation > 0.1:
    print("✅ GOOD: Positive correlation detected!")
elif correlation < -0.1:
    print("⚠️  WARNING: Negative correlation - unexpected!")
else:
    print("❌ BAD: No correlation - IPQ may not be useful or extracted wrong!")

# Show examples
print("\n" + "=" * 70)
print("📋 SAMPLE EXTRACTIONS (IPQ > 1)")
print("=" * 70)

multi_pack = train_df[train_df['ipq'] > 1].head(20)
for idx, row in multi_pack.iterrows():
    text = row['catalog_content'][:150]
    print(f"\nIPQ={row['ipq']:>3} | Price=${row['price']:>6.2f} | {text}...")

# Price analysis by IPQ
print("\n" + "=" * 70)
print("💰 PRICE ANALYSIS BY IPQ")
print("=" * 70)

ipq_groups = train_df.groupby('ipq')['price'].agg(['count', 'mean', 'median', 'std'])
print(ipq_groups.head(15))

# Check if IPQ=1 dominates
if (train_df['ipq']==1).mean() > 0.9:
    print("\n" + "=" * 70)
    print("❌ CRITICAL ISSUE DETECTED!")
    print("=" * 70)
    print(f"⚠️  {(train_df['ipq']==1).mean()*100:.1f}% of products have IPQ=1")
    print("⚠️  IPQ extraction is likely FAILING!")
    print("\nPossible reasons:")
    print("  1. Pack quantities not mentioned in catalog_content")
    print("  2. Different format than expected (e.g., 'Quantity:12' vs 'pack of 12')")
    print("  3. Need more regex patterns")
    
    print("\n🔍 Let's check some random catalog_content samples:")
    print("=" * 70)
    for idx, row in train_df.sample(10).iterrows():
        print(f"\n{row['catalog_content'][:200]}...")
        print(f"  → Extracted IPQ: {row['ipq']}")
        
elif (train_df['ipq']>1).mean() > 0.05:
    print("\n" + "=" * 70)
    print("✅ IPQ EXTRACTION SEEMS TO BE WORKING!")
    print("=" * 70)
    print(f"✅ {(train_df['ipq']>1).mean()*100:.1f}% of products have IPQ>1")
    print(f"✅ Correlation with price: {correlation:.3f}")
    print("\nYour model should improve with IPQ feature!")
    
else:
    print("\n" + "=" * 70)
    print("⚠️  MARGINAL IPQ EXTRACTION")
    print("=" * 70)
    print(f"⚠️  Only {(train_df['ipq']>1).mean()*100:.1f}% of products have IPQ>1")
    print("⚠️  IPQ feature may have limited impact")

print("\n" + "=" * 70)
print("✅ DIAGNOSTIC COMPLETE")
print("=" * 70)