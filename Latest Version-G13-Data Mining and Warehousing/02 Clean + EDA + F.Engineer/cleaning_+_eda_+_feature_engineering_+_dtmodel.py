"""# DATA CLEANING"""

import pandas as pd
import numpy as np
import re

df = pd.read_csv("ecommerce_churn_llm_slm_dataset.csv")
print("Shape:", df.shape)
df.head()

df.isna().sum()

# standardize column names (remove leading/trailing spaces, replace with _)
df.columns = (
    df.columns
      .str.strip()
      .str.replace(r"\s+", "_", regex=True)
)

# remove duplicate rows
df = df.drop_duplicates()

# If customer_id exists, keep unique customer rows (= one row one cust)
if "customer_id" in df.columns:
    df = df.drop_duplicates(subset=["customer_id"], keep="first")

# identify common column groups (diff dtypes need diff cleaning strategy)
TEXT_COLS = ["customer_feedback", "support_chat_excerpt", "reason_for_low_activity"]
CAT_COLS  = ["gender", "state", "income_band", "segment"]
NUMERIC_COLS = [
    "age","tenure_days","orders_90d","orders_30d","avg_order_value","total_spend_90d",
    "days_since_last_order","browse_sessions_30d","cart_abandon_rate","return_rate",
    "refund_count_90d","discount_ratio","coupon_used_rate","delivery_delay_rate",
    "late_delivery_count","failed_delivery_count","tickets_90d",
    "avg_first_response_hours","email_open_rate","push_open_rate",
    "loyalty_points_balance","app_rating"
]

# coerce types safely
# 1) churn must numeric (0/1)
if "churn" in df.columns:
    df["churn"] = pd.to_numeric(df["churn"], errors="coerce")

# 2) numeric cols : convert numbers stored as text --> numeric
for c in NUMERIC_COLS:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# 3) categories
for c in CAT_COLS:
    if c in df.columns:
        df[c] = df[c].astype("string")

# 4) text
for c in TEXT_COLS:
    if c in df.columns:
        df[c] = df[c].astype("string")

# clean text fields (remove extra space, normalize formatting, convert empty strings --> missing)
def clean_text(x):
    if pd.isna(x):
        return pd.NA
    s = str(x).replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return pd.NA if s == "" else s

for c in TEXT_COLS:
    if c in df.columns:
        df[c] = df[c].apply(clean_text)

# handle missing values
# 1) fill missing values with empty string (=nlp wont crash)
for c in TEXT_COLS:
    if c in df.columns:
        df[c] = df[c].fillna("")

# 2) categorical : fill missing with "unknown"
for c in CAT_COLS:
    if c in df.columns:
        df[c] = df[c].fillna("unknown").astype("category")

# 3) numeric : median imputation
for c in NUMERIC_COLS:
    if c in df.columns:
        med = df[c].median()
        df[c] = df[c].fillna(med)

# 4) churn : drop rows if missing, enforce 0/1 int
if "churn" in df.columns:
    df = df[df["churn"].notna()].copy()
    # If churn contains weird values, clip to [0,1] then round
    df["churn"] = df["churn"].clip(0, 1).round().astype(int)

# logical consistency + realistic constraints
def clip_col(col, lo=None, hi=None):
    if col in df.columns:
        if lo is not None:
            df[col] = df[col].clip(lower=lo)
        if hi is not None:
            df[col] = df[col].clip(upper=hi)

# Common realism rules (age & rating limits)
clip_col("age", 18, 80)
clip_col("app_rating", 1, 5)

# Rate/proportion columns should be within [0,1] (probabilities)
for rate_col in [
    "cart_abandon_rate","return_rate","discount_ratio","coupon_used_rate",
    "delivery_delay_rate","email_open_rate","push_open_rate"
]:
    clip_col(rate_col, 0, 1)

# Orders logic: orders_30d cannot exceed orders_90d
if "orders_90d" in df.columns and "orders_30d" in df.columns:
    df["orders_30d"] = np.minimum(df["orders_30d"], df["orders_90d"])

# days_since_last_order cannot be negative
clip_col("days_since_last_order", 0, None)

# counts cannot be negative
for count_col in ["refund_count_90d","late_delivery_count","failed_delivery_count","tickets_90d",
                  "orders_90d","orders_30d","browse_sessions_30d","loyalty_points_balance","tenure_days"]:
    clip_col(count_col, 0, None)

# final quick report
print("\n=== CLEANING REPORT ===")
print("Final shape:", df.shape)

missing_top = df.isna().sum().sort_values(ascending=False).head(15)
print("\nTop missing counts (should be near 0 for most):")
print(missing_top)

if "churn" in df.columns:
    print("\nChurn distribution:")
    print(df["churn"].value_counts())

# Preview some feedback content to verify it looks fine
if "customer_feedback" in df.columns:
    non_empty = df[df["customer_feedback"].str.strip() != ""]
    print("\nNon-empty customer_feedback rows:", len(non_empty))
    display(non_empty[["customer_id","customer_feedback"]].head(8) if "customer_id" in df.columns
            else non_empty[["customer_feedback"]].head(8))

df.to_csv("ecommerce_churn_cleaned.csv")

"""#  EDA"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# post cleaning sanity check
print("Shape:", df.shape)
display(df.head())
display(df.describe(include="all").T)

# churn distribution
plt.figure()
df["churn"].value_counts().plot(kind="bar")
plt.title("Churn Distribution")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")
plt.show()

print(df["churn"].value_counts(normalize=True))

# demographic analysis vs churn
# 1) age vs churn
plt.figure()
df.boxplot(column="age", by="churn")
plt.title("Age Distribution by Churn")
plt.suptitle("")
plt.xlabel("Churn")
plt.ylabel("Age")
plt.show()

# 2) gender vs churn
if "gender" in df.columns:
    pd.crosstab(df["gender"], df["churn"], normalize="index").plot(kind="bar")
    plt.title("Churn Rate by Gender")
    plt.ylabel("Proportion")
    plt.show()

# behavioural features vs churn
# 1) orders & activity
behaviour_cols = ["orders_90d","orders_30d","browse_sessions_30d","days_since_last_order"]

for col in behaviour_cols:
    if col in df.columns:
        plt.figure()
        df.boxplot(column=col, by="churn")
        plt.title(f"{col} by Churn")
        plt.suptitle("")
        plt.xlabel("Churn")
        plt.ylabel(col)
        plt.show()

# monetary features vs churn
monetary_cols = ["avg_order_value","total_spend_90d","loyalty_points_balance"]

for col in monetary_cols:
    if col in df.columns:
        plt.figure()
        df.boxplot(column=col, by="churn")
        plt.title(f"{col} by Churn")
        plt.suptitle("")
        plt.xlabel("Churn")
        plt.ylabel(col)
        plt.show()

# service quality & friction indicators
service_cols = [
    "cart_abandon_rate","return_rate","delivery_delay_rate",
    "late_delivery_count","failed_delivery_count","tickets_90d"
]

for col in service_cols:
    if col in df.columns:
        plt.figure()
        df.boxplot(column=col, by="churn")
        plt.title(f"{col} by Churn")
        plt.suptitle("")
        plt.xlabel("Churn")
        plt.ylabel(col)
        plt.show()

# engagement channels vs churn
engagement_cols = ["email_open_rate","push_open_rate","coupon_used_rate"]

for col in engagement_cols:
    if col in df.columns:
        plt.figure()
        df.boxplot(column=col, by="churn")
        plt.title(f"{col} by Churn")
        plt.suptitle("")
        plt.xlabel("Churn")
        plt.ylabel(col)
        plt.show()

# text data completeness
text_cols = ["customer_feedback","support_chat_excerpt","reason_for_low_activity"]

for col in text_cols:
    if col in df.columns:
        non_empty = (df[col].str.strip() != "").sum()
        print(f"{col}: {non_empty} non-empty rows ({non_empty/len(df):.1%})")

# correlation analysis (numeric only)
numeric_df = df.select_dtypes(include=[np.number])

corr = numeric_df.corr()

plt.figure(figsize=(10,8))
plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Heatmap (Numeric Features)")
plt.tight_layout()
plt.show()

"""Overall EDA Summary :

The exploratory data analysis reveals that the dataset is well-balanced in terms of churn classes and exhibits realistic customer behaviour patterns. Demographic variables such as age and gender show limited influence on churn, while behavioural, engagement, and service-quality features display clearer separation between churned and non-churned customers. Variables including days since last order, order frequency, browsing activity, delivery performance, cart abandonment, customer support interactions, and engagement rates demonstrate meaningful relationships with churn. Correlation analysis further indicates low multicollinearity among numeric features, suggesting that the dataset is suitable for baseline modelling using interpretable classifiers such as a decision tree.

# FEATURE ENGINEERING
"""

#imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from transformers import pipeline as hf_pipeline

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression

RANDOM_SEED = 42

# build combined text field (sentiment model need 1 input text = combine feedback + chat)
df = df.copy()
for c in ["customer_feedback", "support_chat_excerpt"]:
    if c in df.columns:
        df[c] = df[c].fillna("").astype(str)

df["combined_text"] = (df["customer_feedback"] + " " + df["support_chat_excerpt"]).str.strip()

# SLM Sentiment Feature (DistilBERT fine tuned on SST-2 sentiment classification, output = pos & neg)
#load model
slm_sentiment = hf_pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# sentiment labeling func (empty : neutral, otherwise pass to sentiment model : pos/neg)
def slm_sentiment_label(text):
    if not isinstance(text, str) or text.strip() == "":
        return "neutral"
    out = slm_sentiment(text[:512])[0]  # truncate to 512 tokens worth of chars (simple safety)
    return "positive" if out["label"].upper() == "POSITIVE" else "negative"

# batch inference instead of .apply (faster)
texts = df["combined_text"].tolist()
labels = []
BATCH = 32
for i in range(0, len(texts), BATCH):
    batch = texts[i:i+BATCH]
    # replace empty with placeholder to avoid pipeline complaining
    batch_safe = [t if t.strip() != "" else "[EMPTY]" for t in batch]
    outs = slm_sentiment(batch_safe)
    for t, o in zip(batch, outs):
        if t.strip() == "":
            labels.append("neutral")
        else:
            labels.append("positive" if o["label"].upper() == "POSITIVE" else "negative")

df["slm_sentiment"] = labels

# simple text-derived numeric features (keep info from text w/o put raw text into onehotencoder)
df["has_feedback"] = (df["customer_feedback"].str.strip() != "").astype(int)
df["has_support_chat"] = (df["support_chat_excerpt"].str.strip() != "").astype(int)

df["feedback_len"] = df["customer_feedback"].str.len()
df["support_len"] = df["support_chat_excerpt"].str.len()
df["combined_len"] = df["combined_text"].str.len()

# -------------------------------
# 4) Optional business-sense ratio features (orders/sessions & spend/order ratio)
# Why: ratios can be more informative than raw counts
# -------------------------------
if "orders_90d" in df.columns and "browse_sessions_30d" in df.columns:
    df["orders_per_session_approx"] = df["orders_90d"] / (df["browse_sessions_30d"] + 1e-6)

if "total_spend_90d" in df.columns and "orders_90d" in df.columns:
    df["spend_per_order_approx"] = df["total_spend_90d"] / (df["orders_90d"] + 1e-6)
