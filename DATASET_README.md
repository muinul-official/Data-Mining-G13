# E-Commerce Churn & Burn Analysis Dataset

## Overview
This dataset contains **1,200 customer records** for predicting customer churn in an e-commerce platform. The dataset includes transaction data, customer support chat logs, and product reviews with sentiment analysis features.

## Dataset Files

### 1. `ecommerce_churn_dataset.csv` (Main Dataset)
The primary dataset containing all features for churn prediction.

**Key Features:**
- **Customer Demographics**: age, gender, location, membership_tier
- **RFM Features**: recency, frequency, monetary_value, avg_order_value
- **Behavioral Features**: days_since_last_login, cart_abandonment_rate, email_open_rate
- **Text Features**: chat_log, review_text (for sentiment analysis)
- **Sentiment Scores**: chat_sentiment_score, review_sentiment_score, sentiment_score_avg
- **Frustration Index**: Calculated from multiple signals (sentiment, keywords, ratings, tickets)
- **Target Variable**: churn (0 = No churn, 1 = Churn)

**Statistics:**
- Total Records: 1,200
- Churn Rate: 22.92% (275 churned, 925 retained)
- Average Frustration Index: 0.456
- Average Recency: 61.9 days
- Average Frequency: 5.3 orders
- Average Monetary Value: $371.58

### 2. `ecommerce_transactions.csv`
Detailed transaction history for all customers.

**Columns:**
- customer_id
- transaction_date
- amount
- product_category (Electronics, Clothing, Home & Garden, etc.)
- quantity
- discount_applied (True/False)
- payment_method (Credit Card, Debit Card, PayPal, Bank Transfer)
- shipping_method (Standard, Express, Overnight)

**Total Transactions:** ~6,400 records

### 3. `ecommerce_chat_logs.csv`
Customer support chat logs with sentiment analysis.

**Columns:**
- customer_id
- chat_log (text content)
- chat_sentiment_score (0.0 = negative, 1.0 = positive)
- chat_length (character count)
- has_negative_keywords (True/False)

### 4. `ecommerce_reviews.csv`
Product reviews with ratings and sentiment scores.

**Columns:**
- customer_id
- review_text (text content)
- review_rating (1-5 stars)
- review_sentiment_score (0.0 = negative, 1.0 = positive)
- review_length (character count)

## Key Features for Analysis

### Frustration Index
The **frustration_index** is a composite feature (0-1 scale) calculated from:
- Chat sentiment score (40% weight)
- Review sentiment score (30% weight)
- Presence of negative keywords (20% weight)
- Review rating (10% weight)
- Number of support tickets (10% weight)

**Higher frustration index = Higher churn probability**

### Relationship Between Features and Churn
- **Churned customers** have higher frustration index (mean: 0.693) vs retained (mean: 0.386)
- **Recency**: Longer time since last purchase increases churn risk
- **Frequency**: Fewer orders increase churn risk
- **Monetary Value**: Lower spending increases churn risk
- **Sentiment**: Negative chat/review sentiment correlates with churn

## Usage for Project

### For Feature Engineering (SLM/LLM Analysis):
1. Use `chat_log` and `review_text` columns for sentiment analysis
2. Extract additional features using SLMs/LLMs:
   - Sentiment classification
   - Topic extraction
   - Emotion detection
   - Intent classification
3. Enhance frustration_index with AI-extracted features

### For Predictive Modeling:
- **Target Variable**: `churn` (binary classification)
- **Features**: All numerical and categorical features
- **Text Features**: Use sentiment scores and derived features from chat/review text

### For Model Evaluation:
- Use metrics: Accuracy, F1-score, ROC-AUC
- Compare models: Random Forest, Logistic Regression, XGBoost, Neural Networks
- Interpret feature importance, especially frustration_index and sentiment features

## Data Quality Notes

- All customer IDs are unique
- No missing values in main dataset
- Realistic relationships between features and churn
- Text data includes varied sentiment patterns
- Transaction dates span from 2023-01-01 to 2024-12-31

## Next Steps

1. **Exploratory Data Analysis (EDA)**
   - Visualize distributions of key features
   - Analyze correlation between features and churn
   - Examine text sentiment patterns

2. **Feature Engineering**
   - Use SLMs/LLMs to extract additional features from text
   - Create interaction features
   - Engineer time-based features

3. **Model Development**
   - Split data into train/test sets
   - Build and compare multiple models
   - Tune hyperparameters

4. **Model Interpretation**
   - Analyze feature importance
   - Use LLMs to generate insights and summaries
   - Create business recommendations

## Dataset Generation

The dataset was generated using `generate_ecommerce_churn_dataset.py` with realistic patterns:
- Transaction amounts follow log-normal distribution
- Churn probability based on RFM features and frustration index
- Text sentiment aligned with customer behavior patterns
- Realistic customer demographics and behaviors

