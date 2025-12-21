"""Quick validation script for the generated dataset"""
import pandas as pd

print("=" * 60)
print("Dataset Validation Report")
print("=" * 60)

# Load main dataset
df = pd.read_csv('ecommerce_churn_dataset.csv')

print(f"\n[OK] Total Records: {len(df)}")
print(f"  Requirement: >= 1000 records")
print(f"  Status: {'PASS' if len(df) >= 1000 else 'FAIL'}")

print(f"\n[OK] Churn Distribution:")
churn_counts = df['churn'].value_counts()
print(f"  No Churn (0): {churn_counts.get(0, 0)}")
print(f"  Churn (1): {churn_counts.get(1, 0)}")
print(f"  Churn Rate: {df['churn'].mean():.2%}")

print(f"\n[OK] Text Features:")
print(f"  Chat logs: {df['chat_log'].notna().sum()} ({df['chat_log'].notna().sum()/len(df)*100:.1f}%)")
print(f"  Reviews: {df['review_text'].notna().sum()} ({df['review_text'].notna().sum()/len(df)*100:.1f}%)")

print(f"\n[OK] Sentiment Features:")
print(f"  Chat sentiment score range: {df['chat_sentiment_score'].min():.3f} - {df['chat_sentiment_score'].max():.3f}")
print(f"  Review sentiment score range: {df['review_sentiment_score'].min():.3f} - {df['review_sentiment_score'].max():.3f}")
print(f"  Average sentiment: {df['sentiment_score_avg'].mean():.3f}")

print(f"\n[OK] Frustration Index:")
print(f"  Range: {df['frustration_index'].min():.3f} - {df['frustration_index'].max():.3f}")
print(f"  Mean: {df['frustration_index'].mean():.3f}")
print(f"  Std: {df['frustration_index'].std():.3f}")
print(f"\n  By Churn Status:")
print(f"    No Churn: {df[df['churn']==0]['frustration_index'].mean():.3f}")
print(f"    Churn: {df[df['churn']==1]['frustration_index'].mean():.3f}")

print(f"\n[OK] Transaction Features:")
print(f"  Average frequency: {df['frequency'].mean():.2f}")
print(f"  Average recency: {df['recency'].mean():.2f} days")
print(f"  Average monetary value: ${df['monetary_value'].mean():.2f}")

print(f"\n[OK] Missing Values Check:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("  No missing values found - PASS")
else:
    print("  Missing values found:")
    print(missing[missing > 0])

# Check transaction file
transactions = pd.read_csv('ecommerce_transactions.csv')
print(f"\n[OK] Transaction History:")
print(f"  Total transactions: {len(transactions)}")
print(f"  Unique customers: {transactions['customer_id'].nunique()}")
print(f"  Date range: {transactions['transaction_date'].min()} to {transactions['transaction_date'].max()}")

print("\n" + "=" * 60)
print("Validation Complete!")
print("=" * 60)

