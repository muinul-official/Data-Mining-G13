import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_churn_distribution(df):
    """Plots the distribution of the churn target variable."""
    plt.figure(figsize=(6, 4))
    sns.countplot(x='churn', data=df, palette='viridis')
    plt.title("Churn Distribution")
    plt.xlabel("Churn (0=No, 1=Yes)")
    plt.ylabel("Count")
    plt.show()

def plot_feature_importance(importances, feature_names, title="Feature Importance"):
    """Plots a bar chart for feature importance."""
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df):
    """Plots a heatmap for numeric feature correlations."""
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm')
    plt.title("Numeric Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()

def plot_distribution_by_churn(df, column):
    """Plots the distribution of a numeric column segmented by churn."""
    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=df, x=column, hue='churn', fill=True, palette='coolwarm')
    plt.title(f"Distribution of {column} by Churn")
    plt.show()
