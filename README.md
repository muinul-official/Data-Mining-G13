# WIE3007 Data Mining and Warehousing - Group G13 Project

## 📋 Project Overview

This repository contains the complete Group G13 project for **WIE3007 Data Mining and Warehousing** at the University of Malaya (UM). The project focuses on **E-Commerce Customer Churn Prediction** using advanced machine learning techniques and AI-enhanced analytics with LLM-generated customer feedback.

## 🎯 Project Objectives

1. **Predict Customer Churn**: Build predictive models to identify customers likely to leave the platform
2. **Leverage AI/LLM**: Utilize Google's Gemini (gemma-3-4b-it) to generate realistic customer feedback, support chat excerpts, and activity reasons
3. **Comprehensive Analysis**: Perform exploratory data analysis, feature engineering, and model evaluation
4. **Multi-Model Comparison**: Implement and compare multiple machine learning algorithms

## 📂 Repository Structure

```
Data-Mining-G13/
│
├── Latest Version-G13-Data Mining and Warehousing/
│   ├── Data_Mining_Group_Project.ipynb          # Main Jupyter notebook with complete analysis
│   ├── data_mining_group_project.py             # Python script version
│   ├── ecommerce_churn_llm_final.csv           # Final dataset (1,800 rows) with LLM features
│   ├── llm_text_generation_input.jsonl         # LLM input prompts (500 samples)
│   └── llm_text_generation_output.jsonl        # LLM generated text outputs
│
├── Previous version-G13-Data Mining and Warehousing-Group Project/
│   ├── EDA_&_Feature_Engineering.ipynb
│   ├── PredictiveModelDevelopment_(LogisticRegression).ipynb
│   ├── PredictionModelXGBoost.ipynb
│   ├── Initial datasets and scripts
│   └── WIE3007_GroupProject_UM_Formatted.pdf
│
├── README.md
└── .gitignore
```

## ✨ Key Features

### 1. **Advanced Dataset Generation**
- **1,800+ customer records** with realistic behavioral patterns
- **Malaysian context**: States, income bands (B40/M40/T20), local demographics
- **27+ behavioral features** including:
  - Transaction metrics (orders, spending, recency)
  - Engagement metrics (browsing, cart abandonment, email/push open rates)
  - Service quality metrics (delivery delays, refunds, support tickets)
  - Customer satisfaction metrics (app ratings, loyalty points)

### 2. **AI-Enhanced Analytics**
- **LLM Integration**: Google Gemini API (gemma-3-4b-it model)
- **500 customers with LLM-generated text**:
  - `customer_feedback`: Realistic feedback in casual English with Malaysian expressions
  - `support_chat_excerpt`: Customer support chat messages
  - `reason_for_low_activity`: Brief explanations for reduced engagement
- **Smart generation**: Feedback adapts to customer behavior (e.g., delivery complaints for high delay rates)

### 3. **Comprehensive Machine Learning Pipeline**
- **Data Preprocessing**: Handling missing values, feature scaling, encoding
- **Feature Engineering**: RFM analysis, derived metrics, behavioral indicators
- **Multiple Models**:
  - Decision Tree Classifier
  - Logistic Regression
  - Random Forest Classifier
  - XGBoost Classifier
  - Neural Network (MLP)
- **Model Evaluation**: Accuracy, F1-score, ROC-AUC, confusion matrices
- **Feature Importance**: SHAP values and permutation importance analysis

## 🛠️ Technologies & Libraries

### Core ML Stack
```python
- pandas, numpy          # Data manipulation
- scikit-learn          # Machine learning models & preprocessing
- xgboost               # Gradient boosting
- matplotlib            # Visualization
```

### AI/LLM Integration
```python
- google-genai          # Google Gemini API
- transformers, torch   # NLP capabilities
```

### Advanced Analytics
```python
- SHAP                  # Model explainability
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Google Colab (recommended) or local Jupyter environment
- Google API Key for Gemini (if regenerating LLM features)

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/muinul-official/Data-Mining-G13.git
cd Data-Mining-G13
```

2. **Install dependencies**:
```bash
pip install -q pandas numpy scikit-learn matplotlib transformers torch xgboost shap
pip install -q -U google-genai
```

3. **Set up Google API** (optional, only if regenerating LLM features):
```python
import os
os.environ["GOOGLE_API_KEY"] = "your_api_key_here"
```

### Running the Analysis

#### Option 1: Jupyter Notebook (Recommended)
```bash
cd "Latest Version-G13-Data Mining and Warehousing"
jupyter notebook Data_Mining_Group_Project.ipynb
```

#### Option 2: Python Script
```bash
cd "Latest Version-G13-Data Mining and Warehousing"
python data_mining_group_project.py
```

## 📊 Dataset Details

### Main Dataset: `ecommerce_churn_llm_final.csv`

**Total Records**: 1,800 customers  
**Churn Rate**: ~40-45% (realistic imbalanced dataset)

#### Feature Categories

**1. Demographics** (7 features)
- `customer_id`, `age`, `gender`, `state`, `income_band`, `tenure_days`, `segment`

**2. Transaction Behavior** (8 features)
- `orders_90d`, `orders_30d`, `avg_order_value`, `total_spend_90d`
- `days_since_last_order`, `browse_sessions_30d`, `cart_abandon_rate`, `return_rate`

**3. Service Quality** (9 features)
- `refund_count_90d`, `delivery_delay_rate`, `late_delivery_count`, `failed_delivery_count`
- `tickets_90d`, `avg_first_response_hours`, `discount_ratio`, `coupon_used_rate`

**4. Engagement** (4 features)
- `email_open_rate`, `push_open_rate`, `loyalty_points_balance`, `app_rating`

**5. LLM-Generated Text** (3 features)
- `customer_feedback`: Realistic customer comments (Malaysian English style)
- `support_chat_excerpt`: Customer support conversation snippets
- `reason_for_low_activity`: Brief activity explanations

**6. Target Variable**
- `churn`: Binary indicator (0 = Active, 1 = Churned)

### LLM Text Generation Files

**Input**: `llm_text_generation_input.jsonl`
- 500 prompts with customer profile summaries
- Structured JSON format for API consumption

**Output**: `llm_text_generation_output.jsonl`
- 500 LLM-generated responses
- Rate-limited generation (~27 requests/min to avoid quota issues)

## 🔬 Methodology

### 1. Dataset Generation
- **Realistic simulation** using statistical distributions (Poisson, Beta, Lognormal)
- **Correlated features** ensuring behavioral consistency
- **Churn labels** derived from risk scoring (sigmoid function)
- **Missing values** injected realistically (~2-4% for select features)

### 2. LLM Integration
```python
Model: gemma-3-4b-it (Google Gemini)
Rate Limiting: 2.2s per request (~27 req/min)
Prompts: Context-aware based on customer behavior
Error Handling: Retry logic with exponential backoff
```

### 3. Machine Learning Pipeline
```
Data Loading → Missing Value Handling → Feature Engineering
    ↓
Train/Test Split (80/20) → Preprocessing (Scaling, Encoding)
    ↓
Model Training (5 algorithms) → Hyperparameter Tuning
    ↓
Evaluation (Accuracy, F1, ROC-AUC) → Feature Importance Analysis
```

## 📈 Results & Insights

The project demonstrates:
- ✅ **Effective churn prediction** using multiple ML algorithms
- ✅ **AI-enhanced dataset** with realistic customer feedback
- ✅ **Feature importance analysis** revealing key churn indicators
- ✅ **Model comparison** to identify best-performing algorithms
- ✅ **Complete documentation** for reproducibility

## 🎓 Academic Context

**Course**: WIE3007 - Data Mining and Warehousing  
**Institution**: University of Malaya (UM)  
**Group**: G13  
**Project Type**: Group Assignment - Comprehensive Data Mining Project

### Project Requirements Met
✅ Dataset size ≥ 1,000 records (1,800 provided)  
✅ Multiple predictive models implemented (5 algorithms)  
✅ AI/LLM integration (Google Gemini for text generation)  
✅ Comprehensive documentation and code quality  
✅ Feature engineering and analysis  
✅ Model evaluation and comparison  

## 📝 Configuration & Customization

### Dataset Generation Parameters
```python
RANDOM_SEED = 42              # Reproducibility
N = 1500                       # Base customer count (1,800 after LLM merge)
N_TEXT_ROWS = 500              # Customers with LLM-generated text
```

### LLM Settings
```python
model = "models/gemma-3-4b-it"
base_sleep_s = 2.2             # Rate limiting (27 req/min)
max_retries_per_row = 5        # Error handling
```

### Malaysian Demographics
```python
states = ["Selangor", "KL", "Johor", "Penang", "Perak", "Sabah", "Sarawak", "Pahang", "Kelangor", "Terengganu"]
income_bands = ["B40", "M40", "T20"]
segments = ["new", "regular", "loyal", "premium"]
```

## 🤝 Contributing

This is an academic project for WIE3007. For any questions or suggestions:
- Open an issue on GitHub
- Contact the project maintainer

## 📄 License

This project is part of an academic course assignment at the University of Malaya.

## 👥 Group Members

**Group G13**  
WIE3007 - Data Mining and Warehousing  
University of Malaya

## 🙏 Acknowledgments

- **University of Malaya** - WIE3007 Course Instructors
- **Google Gemini API** - AI-powered text generation
- **Open Source Community** - scikit-learn, XGBoost, SHAP

---

## 📌 Important Notes

1. **API Keys**: The Google API key in the code is for demonstration purposes. Use your own key for production
2. **Rate Limits**: LLM generation respects API quota limits (~27 req/min)
3. **Reproducibility**: Fixed random seed (42) ensures consistent results
4. **Data Privacy**: All data is synthetically generated for educational purposes
5. **Latest Version**: Focus on files in `Latest Version-G13-Data Mining and Warehousing/` folder

---

**Last Updated**: Based on feedback improvements - January 2026  
**Repository**: https://github.com/muinul-official/Data-Mining-G13  
**Maintained by**: [@muinul-official](https://github.com/muinul-official)
