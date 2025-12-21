"""
E-Commerce Churn & Burn Analysis Dataset Generator
Generates realistic dataset with transaction data, chat logs, and reviews
for predicting customer churn with sentiment-based features.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Configuration
N_CUSTOMERS = 1200  # Generate more than 1000 to allow for filtering
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 12, 31)

# Product categories
PRODUCT_CATEGORIES = [
    'Electronics', 'Clothing', 'Home & Garden', 'Books', 'Sports',
    'Beauty', 'Toys', 'Food & Beverages', 'Automotive', 'Health'
]

# Sentiment patterns for chat logs (frustrated customers)
FRUSTRATED_CHAT_PATTERNS = [
    "I've been waiting for my order for 3 weeks! This is unacceptable.",
    "Your customer service is terrible. No one responds to my emails.",
    "The product I received was completely different from what I ordered.",
    "I want a refund immediately. This is the worst experience ever.",
    "Why is my order still pending? I placed it 2 weeks ago!",
    "The product quality is terrible. I'm very disappointed.",
    "I've called multiple times but no one helps me.",
    "This is ridiculous. I'm never shopping here again.",
    "My package was damaged and you're refusing to help.",
    "I've been a customer for years and this is how you treat me?",
    "The website is broken and I can't complete my purchase.",
    "Your return policy is confusing and unfair.",
    "I paid for express shipping but it's been delayed.",
    "The customer support chat keeps disconnecting.",
    "I'm extremely frustrated with this service."
]

# Neutral chat patterns
NEUTRAL_CHAT_PATTERNS = [
    "Hi, I'd like to check the status of my order.",
    "Can you help me with product recommendations?",
    "I have a question about shipping options.",
    "What's the return policy for electronics?",
    "I need to update my delivery address.",
    "Can you tell me more about this product?",
    "I'm looking for a gift for my friend.",
    "What payment methods do you accept?",
    "I'd like to know about your loyalty program.",
    "Can I change my order before it ships?",
    "I need help with my account settings.",
    "What are your business hours?",
    "I want to subscribe to your newsletter.",
    "Can you explain your warranty policy?",
    "I'm interested in bulk ordering."
]

# Positive chat patterns
POSITIVE_CHAT_PATTERNS = [
    "Thank you so much! The product exceeded my expectations.",
    "I love your service! Keep up the great work.",
    "The delivery was super fast. I'm very happy!",
    "Your customer service team is amazing!",
    "I'm so satisfied with my purchase. Thank you!",
    "This is the best online shopping experience I've had.",
    "I'll definitely shop here again. Great job!",
    "The product quality is excellent. Highly recommend!",
    "Your website is easy to use and checkout was smooth.",
    "I appreciate the quick response to my inquiry.",
    "The packaging was perfect and the product arrived safely.",
    "I'm impressed with your customer service.",
    "Thank you for resolving my issue so quickly.",
    "I love the variety of products you offer.",
    "Great prices and fast shipping. Very satisfied!"
]

# Review templates with varying sentiment
NEGATIVE_REVIEWS = [
    "Terrible product quality. Broke after one week. Waste of money.",
    "Very disappointed. Not as described. Poor customer service.",
    "The worst purchase I've made. Don't buy this product.",
    "Cheaply made and overpriced. Regret buying this.",
    "Arrived damaged and seller refused to help. Avoid!",
    "Poor quality materials. Fell apart quickly.",
    "Not worth the price. Very disappointed.",
    "Customer service was unhelpful and rude.",
    "Product doesn't work as advertised. Very frustrating.",
    "Low quality and expensive. Would not recommend."
]

NEUTRAL_REVIEWS = [
    "Product is okay. Does what it's supposed to do.",
    "Average quality. Nothing special but works fine.",
    "It's decent for the price. Could be better.",
    "Meets basic expectations. No complaints.",
    "Standard product. Gets the job done.",
    "Fair quality. Worth the money.",
    "It's fine. Nothing exceptional.",
    "Decent product. Would consider buying again.",
    "Average purchase. No major issues.",
    "Okay product. Satisfied but not thrilled."
]

POSITIVE_REVIEWS = [
    "Excellent product! Exceeded my expectations. Highly recommend!",
    "Amazing quality and fast shipping. Very satisfied!",
    "Best purchase I've made. Great value for money.",
    "Outstanding product and customer service. Will buy again!",
    "Perfect! Exactly as described. Love it!",
    "High quality and great price. Very happy!",
    "Fantastic product. Exceeded expectations. 5 stars!",
    "Great value and excellent quality. Highly satisfied!",
    "Wonderful product. Fast delivery. Highly recommend!",
    "Excellent purchase. Great quality and service!"
]

def generate_customer_id(index):
    """Generate unique customer ID"""
    return f"CUST{str(index+1).zfill(6)}"

def generate_transaction_history(customer_id, n_transactions, start_date, end_date):
    """Generate transaction history for a customer"""
    transactions = []
    dates = pd.date_range(start_date, end_date, periods=n_transactions)
    
    for date in dates:
        transaction = {
            'customer_id': customer_id,
            'transaction_date': date,
            'amount': round(np.random.lognormal(3.5, 1.2), 2),
            'product_category': random.choice(PRODUCT_CATEGORIES),
            'quantity': random.randint(1, 5),
            'discount_applied': random.choice([True, False]),
            'payment_method': random.choice(['Credit Card', 'Debit Card', 'PayPal', 'Bank Transfer']),
            'shipping_method': random.choice(['Standard', 'Express', 'Overnight'])
        }
        transactions.append(transaction)
    
    return transactions

def calculate_rfm_features(transactions_df, current_date):
    """Calculate RFM (Recency, Frequency, Monetary) features"""
    customer_transactions = transactions_df.groupby('customer_id').agg({
        'transaction_date': ['max', 'count'],
        'amount': 'sum'
    }).reset_index()
    
    customer_transactions.columns = ['customer_id', 'last_transaction_date', 'frequency', 'monetary_value']
    customer_transactions['recency'] = (current_date - customer_transactions['last_transaction_date']).dt.days
    
    return customer_transactions

def generate_chat_log(customer_id, frustration_level):
    """Generate customer support chat log based on frustration level"""
    if frustration_level == 'high':
        chat_text = random.choice(FRUSTRATED_CHAT_PATTERNS)
        sentiment_score = np.random.uniform(0.0, 0.3)  # Negative sentiment
    elif frustration_level == 'medium':
        chat_text = random.choice(NEUTRAL_CHAT_PATTERNS)
        sentiment_score = np.random.uniform(0.4, 0.6)  # Neutral sentiment
    else:
        chat_text = random.choice(POSITIVE_CHAT_PATTERNS)
        sentiment_score = np.random.uniform(0.7, 1.0)  # Positive sentiment
    
    # Add some variation
    chat_text = chat_text + " " + random.choice([
        "Please help.", "Thank you.", "I need assistance.",
        "Can someone respond?", "Looking forward to your reply."
    ])
    
    return {
        'customer_id': customer_id,
        'chat_log': chat_text,
        'chat_sentiment_score': round(sentiment_score, 3),
        'chat_length': len(chat_text),
        'has_negative_keywords': any(word in chat_text.lower() for word in 
                                    ['terrible', 'worst', 'unacceptable', 'disappointed', 
                                     'frustrated', 'refund', 'broken', 'damaged'])
    }

def generate_product_review(customer_id, frustration_level):
    """Generate product review based on frustration level"""
    if frustration_level == 'high':
        review_text = random.choice(NEGATIVE_REVIEWS)
        review_rating = random.randint(1, 2)
        sentiment_score = np.random.uniform(0.0, 0.3)
    elif frustration_level == 'medium':
        review_text = random.choice(NEUTRAL_REVIEWS)
        review_rating = random.randint(3, 3)
        sentiment_score = np.random.uniform(0.4, 0.6)
    else:
        review_text = random.choice(POSITIVE_REVIEWS)
        review_rating = random.randint(4, 5)
        sentiment_score = np.random.uniform(0.7, 1.0)
    
    return {
        'customer_id': customer_id,
        'review_text': review_text,
        'review_rating': review_rating,
        'review_sentiment_score': round(sentiment_score, 3),
        'review_length': len(review_text)
    }

def calculate_frustration_index(chat_sentiment, review_sentiment, chat_negative_keywords, 
                                review_rating, support_tickets):
    """Calculate frustration index from multiple signals"""
    # Normalize sentiment scores (lower = more frustrated)
    sentiment_component = (1 - chat_sentiment) * 0.4 + (1 - review_sentiment) * 0.3
    
    # Negative keywords indicator
    keyword_component = 0.2 if chat_negative_keywords else 0.0
    
    # Review rating component (lower rating = more frustrated)
    rating_component = (5 - review_rating) / 5 * 0.1
    
    # Support tickets component
    tickets_component = min(support_tickets / 5, 1.0) * 0.1
    
    frustration_index = sentiment_component + keyword_component + rating_component + tickets_component
    return min(1.0, max(0.0, frustration_index))  # Clamp between 0 and 1

def determine_churn(rfm_features, frustration_index, customer_age_days, avg_order_value):
    """Determine if customer will churn based on features"""
    # Higher recency (longer since last purchase) increases churn probability
    recency_factor = min(rfm_features['recency'] / 180, 1.0) * 0.3
    
    # Lower frequency increases churn probability
    frequency_factor = (1 - min(rfm_features['frequency'] / 20, 1.0)) * 0.2
    
    # Higher frustration index increases churn probability
    frustration_factor = frustration_index * 0.3
    
    # Lower monetary value increases churn probability
    monetary_factor = (1 - min(rfm_features['monetary_value'] / 5000, 1.0)) * 0.1
    
    # Older customers less likely to churn (loyalty)
    age_factor = (1 - min(customer_age_days / 730, 1.0)) * 0.1
    
    churn_probability = (recency_factor + frequency_factor + frustration_factor + 
                        monetary_factor - age_factor)
    
    # Add some randomness
    churn_probability += np.random.normal(0, 0.1)
    churn_probability = max(0, min(1, churn_probability))
    
    return 1 if churn_probability > 0.5 else 0, churn_probability

print("Generating E-Commerce Churn Dataset...")
print("=" * 60)

# Generate customer base
customers = []
all_transactions = []
chat_logs = []
reviews = []

for i in range(N_CUSTOMERS):
    customer_id = generate_customer_id(i)
    
    # Customer demographics
    age = random.randint(18, 75)
    gender = random.choice(['Male', 'Female', 'Other'])
    location = random.choice(['Urban', 'Suburban', 'Rural'])
    membership_tier = random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'])
    
    # Customer account age
    account_start_date = START_DATE + timedelta(days=random.randint(0, 730))
    customer_age_days = (END_DATE - account_start_date).days
    
    # Number of transactions (customers with fewer transactions more likely to churn)
    n_transactions = random.choices(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30],
        weights=[10, 8, 7, 6, 5, 4, 3, 3, 2, 2, 1, 1, 1, 1]
    )[0]
    
    # Generate transaction history
    transactions = generate_transaction_history(
        customer_id, n_transactions, account_start_date, END_DATE
    )
    all_transactions.extend(transactions)
    
    # Calculate RFM features
    if len(transactions) > 0:
        trans_df = pd.DataFrame(transactions)
        rfm = calculate_rfm_features(trans_df, END_DATE)
        recency = rfm['recency'].values[0]
        frequency = rfm['frequency'].values[0]
        monetary_value = rfm['monetary_value'].values[0]
    else:
        recency = 999
        frequency = 0
        monetary_value = 0
    
    # Determine frustration level based on RFM (customers with poor RFM more frustrated)
    if recency > 90 or frequency < 3 or monetary_value < 200:
        frustration_level = random.choices(
            ['high', 'medium', 'low'],
            weights=[0.6, 0.3, 0.1]
        )[0]
    else:
        frustration_level = random.choices(
            ['high', 'medium', 'low'],
            weights=[0.1, 0.3, 0.6]
        )[0]
    
    # Generate chat log
    chat_log = generate_chat_log(customer_id, frustration_level)
    chat_logs.append(chat_log)
    
    # Generate product review
    review = generate_product_review(customer_id, frustration_level)
    reviews.append(review)
    
    # Calculate frustration index
    support_tickets = random.randint(0, 5) if frustration_level == 'high' else random.randint(0, 2)
    frustration_index = calculate_frustration_index(
        chat_log['chat_sentiment_score'],
        review['review_sentiment_score'],
        chat_log['has_negative_keywords'],
        review['review_rating'],
        support_tickets
    )
    
    # Calculate average order value
    if len(transactions) > 0:
        avg_order_value = sum(t['amount'] for t in transactions) / len(transactions)
    else:
        avg_order_value = 0
    
    # Determine churn
    rfm_features = {
        'recency': recency,
        'frequency': frequency,
        'monetary_value': monetary_value
    }
    churn, churn_probability = determine_churn(
        rfm_features, frustration_index, customer_age_days, avg_order_value
    )
    
    # Additional features
    days_since_last_login = random.randint(0, min(recency + 10, 180))
    cart_abandonment_rate = random.uniform(0, 0.8) if frustration_level == 'high' else random.uniform(0, 0.4)
    email_open_rate = random.uniform(0.1, 0.9) if churn == 0 else random.uniform(0.05, 0.3)
    
    customer = {
        'customer_id': customer_id,
        'age': age,
        'gender': gender,
        'location': location,
        'membership_tier': membership_tier,
        'account_age_days': customer_age_days,
        'recency': recency,
        'frequency': frequency,
        'monetary_value': round(monetary_value, 2),
        'avg_order_value': round(avg_order_value, 2),
        'total_orders': frequency,
        'days_since_last_login': days_since_last_login,
        'cart_abandonment_rate': round(cart_abandonment_rate, 3),
        'email_open_rate': round(email_open_rate, 3),
        'support_tickets': support_tickets,
        'frustration_index': round(frustration_index, 3),
        'churn': churn,
        'churn_probability': round(churn_probability, 3)
    }
    
    customers.append(customer)

# Create DataFrames
customers_df = pd.DataFrame(customers)
transactions_df = pd.DataFrame(all_transactions)
chat_logs_df = pd.DataFrame(chat_logs)
reviews_df = pd.DataFrame(reviews)

# Merge text data with customer data
final_df = customers_df.merge(chat_logs_df[['customer_id', 'chat_log', 'chat_sentiment_score', 
                                             'chat_length', 'has_negative_keywords']], 
                              on='customer_id', how='left')
final_df = final_df.merge(reviews_df[['customer_id', 'review_text', 'review_rating', 
                                      'review_sentiment_score', 'review_length']], 
                          on='customer_id', how='left')

# Add some additional derived features
final_df['total_spent_per_day'] = final_df['monetary_value'] / (final_df['account_age_days'] + 1)
final_df['order_frequency_per_month'] = final_df['frequency'] / ((final_df['account_age_days'] / 30) + 1)
final_df['sentiment_score_avg'] = (final_df['chat_sentiment_score'] + final_df['review_sentiment_score']) / 2
final_df['text_engagement_score'] = (final_df['chat_length'] + final_df['review_length']) / 200

# Reorder columns for better readability
column_order = [
    'customer_id', 'age', 'gender', 'location', 'membership_tier',
    'account_age_days', 'recency', 'frequency', 'monetary_value', 'avg_order_value',
    'total_orders', 'days_since_last_login', 'cart_abandonment_rate', 'email_open_rate',
    'support_tickets', 'frustration_index', 'chat_log', 'chat_sentiment_score', 
    'chat_length', 'has_negative_keywords', 'review_text', 'review_rating',
    'review_sentiment_score', 'review_length', 'sentiment_score_avg', 
    'text_engagement_score', 'total_spent_per_day', 'order_frequency_per_month', 'churn', 'churn_probability'
]

final_df = final_df[column_order]

# Save datasets
print(f"\nGenerated {len(final_df)} customer records")
print(f"Churn rate: {final_df['churn'].mean():.2%}")
print(f"\nDataset Statistics:")
print(f"  - Average frustration index: {final_df['frustration_index'].mean():.3f}")
print(f"  - Average recency: {final_df['recency'].mean():.1f} days")
print(f"  - Average frequency: {final_df['frequency'].mean():.1f} orders")
print(f"  - Average monetary value: ${final_df['monetary_value'].mean():.2f}")

# Save main dataset
final_df.to_csv('ecommerce_churn_dataset.csv', index=False)
print(f"\n[OK] Saved: ecommerce_churn_dataset.csv")

# Save transactions separately
transactions_df.to_csv('ecommerce_transactions.csv', index=False)
print(f"[OK] Saved: ecommerce_transactions.csv")

# Save chat logs separately
chat_logs_df.to_csv('ecommerce_chat_logs.csv', index=False)
print(f"[OK] Saved: ecommerce_chat_logs.csv")

# Save reviews separately
reviews_df.to_csv('ecommerce_reviews.csv', index=False)
print(f"[OK] Saved: ecommerce_reviews.csv")

# Display sample
print("\n" + "=" * 60)
print("Sample Data (First 5 rows):")
print("=" * 60)
print(final_df.head().to_string())

print("\n" + "=" * 60)
print("Dataset Summary:")
print("=" * 60)
print(final_df.describe())

print("\n" + "=" * 60)
print("Churn Distribution:")
print("=" * 60)
print(final_df['churn'].value_counts())
print(f"\nChurn Rate: {final_df['churn'].mean():.2%}")

print("\n" + "=" * 60)
print("Frustration Index by Churn Status:")
print("=" * 60)
print(final_df.groupby('churn')['frustration_index'].describe())

print("\n[OK] Dataset generation complete!")
print("\nFiles created:")
print("  1. ecommerce_churn_dataset.csv - Main dataset with all features")
print("  2. ecommerce_transactions.csv - Detailed transaction history")
print("  3. ecommerce_chat_logs.csv - Customer support chat logs")
print("  4. ecommerce_reviews.csv - Product reviews")

