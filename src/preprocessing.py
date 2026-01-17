import pandas as pd
import numpy as np
import re
from transformers import pipeline

# Global cache for sentiment pipeline to avoid multiple loads
_SENTIMENT_PIPELINE = None

def get_sentiment_pipeline():
    """Loads and caches the DistilBERT sentiment analysis pipeline."""
    global _SENTIMENT_PIPELINE
    if _SENTIMENT_PIPELINE is None:
        _SENTIMENT_PIPELINE = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
    return _SENTIMENT_PIPELINE

def clean_data(df_raw):
    """
    Standardizes and cleans the raw e-commerce churn dataset.
    
    Args:
        df_raw (pd.DataFrame): The raw input dataframe.
        
    Returns:
        pd.DataFrame: A cleaned and standardized dataframe.
    """
    df = df_raw.copy()
    
    # Standardize column names
    df.columns = df.columns.str.strip().str.replace(r"\s+", "_", regex=True)
    
    # Drop duplicates
    df = df.drop_duplicates()
    if "customer_id" in df.columns:
        df = df.drop_duplicates(subset=["customer_id"], keep="first")
    
    # Define Column Groups
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
    
    # Type Coercion
    for c in NUMERIC_COLS:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CAT_COLS:
        if c in df.columns: df[c] = df[c].astype("string")
    for c in TEXT_COLS:
        if c in df.columns: df[c] = df[c].astype("string")
            
    # Clean Text Fields
    def _clean_text_str(x):
        if pd.isna(x): return ""
        s = str(x).replace("\u00a0", " ")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    for c in TEXT_COLS:
        if c in df.columns:
            df[c] = df[c].apply(_clean_text_str).fillna("")
    
    # Handling Missing Values
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())
    
    for c in CAT_COLS:
        if c in df.columns:
            df[c] = df[c].fillna("unknown")
            
    # Churn Target Cleaning
    if "churn" in df.columns:
        df = df[df["churn"].notna()].copy()
        df["churn"] = df["churn"].clip(0, 1).round().astype(int)
        
    # Logical Constraints
    def _clip_col(df, col, lo=None, hi=None):
        if col in df.columns:
            df[col] = df[col].clip(lower=lo, upper=hi)

    _clip_col(df, "age", 18, 80)
    _clip_col(df, "app_rating", 1, 5)
    for rate_col in ["cart_abandon_rate","return_rate","discount_ratio","coupon_used_rate",
                     "delivery_delay_rate","email_open_rate","push_open_rate"]:
        _clip_col(df, rate_col, 0, 1)
        
    # Consistency Checks
    if "orders_90d" in df.columns and "orders_30d" in df.columns:
        df["orders_30d"] = np.minimum(df["orders_30d"], df["orders_90d"])

    return df

def get_sentiment(text):
    """
    Predicts POSITIVE or NEGATIVE sentiment for a given text.
    
    Args:
        text (str): Input text.
        
    Returns:
        tuple: (label, score) - e.g., ('POSITIVE', 0.99)
    """
    if not text or len(str(text).strip()) < 3:
        return "NEUTRAL", 0.0
    
    pipe = get_sentiment_pipeline()
    res = pipe(str(text)[:512], truncation=True)[0]
    return res['label'], res['score']

def add_sentiment_features(df):
    """
    Combines text fields and adds sentiment analysis features.
    
    Args:
        df (pd.DataFrame): Dataframe with text columns.
        
    Returns:
        pd.DataFrame: Dataframe with 'sentiment_label' and 'sentiment_score'.
    """
    df = df.copy()
    # Combine feedback and chat
    df["combined_text"] = (
        df["customer_feedback"].fillna("") + " " + 
        df["support_chat_excerpt"].fillna("")
    ).str.strip()
    
    # Apply sentiment
    sentiments = df["combined_text"].apply(get_sentiment)
    df["sentiment_label"] = sentiments.apply(lambda x: x[0])
    df["sentiment_score"] = sentiments.apply(lambda x: x[1])
    
    return df
