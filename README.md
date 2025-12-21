# E-Commerce Churn & Burn Analysis

## Project Overview
This project predicts customer churn in an e-commerce platform using transaction data, customer support chat logs, and product reviews. The analysis incorporates sentiment analysis from text data to create a "Frustration Index" feature that links customer behavior (numbers) with customer sentiment (text).

## Dataset
- **Total Records**: 1,200 customers
- **Churn Rate**: 22.92%
- **Features**: Transaction history, chat logs, product reviews, sentiment scores, frustration index

### Files
- `ecommerce_churn_dataset.csv` - Main dataset with all features
- `ecommerce_transactions.csv` - Detailed transaction history
- `ecommerce_chat_logs.csv` - Customer support chat logs
- `ecommerce_reviews.csv` - Product reviews with ratings

See `DATASET_README.md` for detailed dataset documentation.

## Project Structure
```
.
├── generate_ecommerce_churn_dataset.py  # Dataset generation script
├── validate_dataset.py                  # Dataset validation script
├── DATASET_README.md                   # Dataset documentation
├── README.md                           # Project README
└── *.csv                               # Dataset files
```

## Requirements
- Python 3.7+
- pandas
- numpy

## Usage

### Generate Dataset
```bash
python generate_ecommerce_churn_dataset.py
```

### Validate Dataset
```bash
python validate_dataset.py
```

## Next Steps
1. Exploratory Data Analysis (EDA)
2. Feature Engineering using SLMs/LLMs
3. Predictive Model Development
4. Model Evaluation & Interpretation

## Assignment Details
**Course**: WIE 3007 – Data Mining (2025/2026 – Semester 1)  
**Institution**: Universiti Malaya  
**Topic**: E-Commerce "Churn & Burn" Analysis

