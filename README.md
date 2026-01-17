# E-Commerce Customer Churn Prediction - Group G13

This project implements a comprehensive Data Mining pipeline to predict customer churn in a Malaysian e-commerce context. It utilizes synthetic behavioral data enhanced with AI-generated feedback and compares multiple machine learning models with advanced explainability.

## 🚀 Key Features
- **Synthetic Data Generation**: Realistic customer behavior models with Malaysian demographics.
- **AI-Enhanced Feedback**: Customer feedback and support logs generated via Gemini/Gemma models.
- **Advanced Sentiment Analysis**: DistilBERT-based SLM (Small Language Model) sentiment feature engineering.
- **Comprehensive Benchmarking**: Comparative analysis of 5 models (Logistic Regression, Decision Tree, Random Forest, XGBoost, MLP).
- **Explainable AI**: SHAP values for understanding key churn drivers.

## 📂 Project Structure
```text
.
├── 01_Dataset_Generation.ipynb         # Statistical simulation & LLM feedback
├── 02_EDA_Cleaning_Baseline_DT.ipynb   # Exploratory data analysis & Benchmark
├── 03_Model_LR.ipynb                   # Logistic Regression implementation
├── 04_Model_RF.ipynb                   # Random Forest implementation
├── 05_Model_XGB.ipynb                  # XGBoost implementation
├── 06_Model_MLP.ipynb                  # Neural Network implementation
├── 09_model_comparison.ipynb           # GridSearch, Cross-validation & Comparisons
├── Final Deliverable
├── src/                                # Modularized Python Source Code
│   ├── __init__.py
│   ├── data_preparation.py             # Simulation logic
│   ├── preprocessing.py                # Cleaning & Sentiment engineering
│   └── eda.py                          # Reusable visualization functions
├── api_key_loader.py                   # Secure credential loading
├── requirements.txt                    # Project dependencies
└── README.md                           # This file
```

## 🛠️ Setup Instructions
1. **Clone the repository**:
   ```bash
   git clone https://github.com/muinul-official/Data-Mining-G13.git
   cd Data-Mining-G13
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure API Key**:
   Create a `.env` file in the root directory and add your Google API key:
   ```env
   GOOGLE_API_KEY=your_actual_key_here
   ```
4. **Run Notebooks**: Start with `01_Dataset_Generation.ipynb` to prepare the data.

## 📊 Models Evaluated
- **Logistic Regression**: Baseline linear model.
- **Decision Tree**: Rule-based interpretability.
- **Random Forest**: Ensemble stability.
- **XGBoost**: High-performance gradient boosting.
- **MLP Neural Network**: Non-linear pattern recognition.

---
**Course**: Data Mining and Warehousing (WIE3007)
**Group**: G13
