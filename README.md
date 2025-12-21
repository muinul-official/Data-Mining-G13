# E-Commerce Churn & Burn Analysis Dataset Generator

A comprehensive Python tool for generating realistic e-commerce datasets with transaction data, customer support chat logs, and product reviews. This dataset is designed for predicting customer churn using sentiment-based features and RFM (Recency, Frequency, Monetary) analysis.

## 📋 Overview

This project generates synthetic e-commerce customer data that includes:
- **Customer demographics** (age, gender, location, membership tier)
- **Transaction history** with detailed purchase records
- **Customer support chat logs** with sentiment analysis
- **Product reviews** with ratings and sentiment scores
- **Churn prediction labels** based on multiple behavioral indicators

The dataset is ideal for:
- Customer churn prediction models
- Sentiment analysis research
- RFM analysis studies
- E-commerce analytics projects
- Machine learning model training

## ✨ Features

- **Realistic Data Generation**: Creates synthetic but realistic customer behavior patterns
- **Sentiment Analysis**: Includes sentiment scores for chat logs and reviews
- **RFM Analysis**: Calculates Recency, Frequency, and Monetary value features
- **Frustration Index**: Composite metric combining multiple customer satisfaction signals
- **Multiple Data Sources**: Generates separate datasets for transactions, chats, and reviews
- **Validation Script**: Includes a validation script to verify dataset quality
- **Reproducible**: Uses fixed random seeds for consistent results

## 🛠️ Requirements

- Python 3.7 or higher
- pandas
- numpy

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/muinul-official/Data-Mining-G13.git
cd "Data mining and warehouse"
```

2. Install required packages:
```bash
pip install pandas numpy
```

## 🚀 Usage

### Generate Dataset

Run the main script to generate the dataset:

```bash
python generate_ecommerce_churn_dataset.py
```

This will create four CSV files:
- `ecommerce_churn_dataset.csv` - Main dataset with all features
- `ecommerce_transactions.csv` - Detailed transaction history
- `ecommerce_chat_logs.csv` - Customer support chat logs
- `ecommerce_reviews.csv` - Product reviews

### Validate Dataset

Run the validation script to check dataset quality:

```bash
python validate_dataset.py
```

The validation script checks:
- Total number of records (≥1000)
- Churn distribution
- Text features completeness
- Sentiment score ranges
- Missing values
- Transaction history integrity

## 📊 Dataset Structure

### Main Dataset (`ecommerce_churn_dataset.csv`)

The main dataset contains the following features:

#### Customer Demographics
- `customer_id` - Unique customer identifier
- `age` - Customer age
- `gender` - Customer gender
- `location` - Customer location type (Urban/Suburban/Rural)
- `membership_tier` - Membership level (Bronze/Silver/Gold/Platinum)

#### RFM Features
- `recency` - Days since last transaction
- `frequency` - Total number of orders
- `monetary_value` - Total amount spent
- `avg_order_value` - Average order value
- `total_orders` - Total number of orders

#### Behavioral Features
- `account_age_days` - Days since account creation
- `days_since_last_login` - Days since last login
- `cart_abandonment_rate` - Rate of abandoned shopping carts
- `email_open_rate` - Email engagement rate
- `support_tickets` - Number of support tickets

#### Sentiment Features
- `chat_log` - Customer support chat log text
- `chat_sentiment_score` - Sentiment score for chat (0-1)
- `chat_length` - Length of chat log
- `has_negative_keywords` - Boolean flag for negative keywords
- `review_text` - Product review text
- `review_rating` - Product review rating (1-5)
- `review_sentiment_score` - Sentiment score for review (0-1)
- `review_length` - Length of review text
- `sentiment_score_avg` - Average sentiment score

#### Derived Features
- `frustration_index` - Composite frustration metric (0-1)
- `total_spent_per_day` - Spending rate
- `order_frequency_per_month` - Order frequency rate
- `text_engagement_score` - Text engagement metric

#### Target Variable
- `churn` - Binary churn indicator (0 = No churn, 1 = Churn)
- `churn_probability` - Predicted churn probability

### Transaction Dataset (`ecommerce_transactions.csv`)

Contains detailed transaction records:
- `customer_id` - Customer identifier
- `transaction_date` - Date of transaction
- `amount` - Transaction amount
- `product_category` - Product category
- `quantity` - Quantity purchased
- `discount_applied` - Whether discount was applied
- `payment_method` - Payment method used
- `shipping_method` - Shipping method selected

### Chat Logs Dataset (`ecommerce_chat_logs.csv`)

Contains customer support interactions:
- `customer_id` - Customer identifier
- `chat_log` - Chat conversation text
- `chat_sentiment_score` - Sentiment score (0-1)
- `chat_length` - Length of chat log
- `has_negative_keywords` - Boolean flag

### Reviews Dataset (`ecommerce_reviews.csv`)

Contains product reviews:
- `customer_id` - Customer identifier
- `review_text` - Review text content
- `review_rating` - Rating (1-5 stars)
- `review_sentiment_score` - Sentiment score (0-1)
- `review_length` - Length of review

## 📁 Project Structure

```
Data mining and warehouse/
│
├── generate_ecommerce_churn_dataset.py  # Main dataset generator
├── validate_dataset.py                  # Dataset validation script
├── README.md                            # Project documentation
├── .gitignore                           # Git ignore rules
│
└── Generated Files (not in repo):
    ├── ecommerce_churn_dataset.csv      # Main dataset
    ├── ecommerce_transactions.csv       # Transaction history
    ├── ecommerce_chat_logs.csv         # Chat logs
    └── ecommerce_reviews.csv           # Product reviews
```

## 🔧 Configuration

You can modify the following parameters in `generate_ecommerce_churn_dataset.py`:

- `N_CUSTOMERS` - Number of customers to generate (default: 1200)
- `START_DATE` - Start date for transactions (default: 2023-01-01)
- `END_DATE` - End date for transactions (default: 2024-12-31)
- `PRODUCT_CATEGORIES` - List of product categories

## 📈 Dataset Statistics

The generated dataset typically includes:
- **1,200+ customer records**
- **Churn rate**: ~30-40% (varies based on generation)
- **Date range**: 2023-01-01 to 2024-12-31
- **Product categories**: 10 categories
- **Sentiment scores**: Range from 0.0 to 1.0

## 🎯 Use Cases

This dataset can be used for:
1. **Churn Prediction**: Build ML models to predict customer churn
2. **Sentiment Analysis**: Analyze customer sentiment from text data
3. **RFM Segmentation**: Segment customers based on RFM values
4. **Customer Lifetime Value**: Calculate CLV using transaction data
5. **Text Mining**: Extract insights from chat logs and reviews
6. **Feature Engineering**: Create new features from existing data

## 📝 Notes

- The dataset uses fixed random seeds (42) for reproducibility
- CSV and ZIP files are excluded from the repository (see `.gitignore`)
- The frustration index is calculated from multiple signals including sentiment, keywords, ratings, and support tickets
- Churn is determined based on RFM features, frustration index, customer age, and average order value

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is part of a Data Mining and Warehouse course project.

## 👤 Author

**muinul-official**

- GitHub: [@muinul-official](https://github.com/muinul-official)

## 🙏 Acknowledgments

- University of Malaya (UM) - Course project
- WIE3007 - Data Mining and Warehouse

---

**Note**: This is a synthetic dataset generator. The data is generated for educational and research purposes only.

