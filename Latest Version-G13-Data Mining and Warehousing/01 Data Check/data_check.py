"""# DATA CHECK


"""

#Import seperately to prevent calling LLM again (40 minutes)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    learning_curve
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance

from xgboost import XGBClassifier

from transformers import pipeline as hf_pipeline


RANDOM_SEED = 42

df = pd.read_csv("ecommerce_churn_llm_final.csv")
df.head()

def keep_base_column(df, base_col):
    cols = [c for c in df.columns if c == base_col or c.startswith(base_col + "_")]
    if len(cols) <= 1:
        return df  # nothing to do

    # keep the clean base name
    for c in cols:
        if c != base_col:
            df.drop(columns=c, inplace=True)
    return df

for col in ["customer_feedback", "support_chat_excerpt", "reason_for_low_activity"]:
    df = keep_base_column(df, col)

[c for c in df.columns if "feedback" in c or "support_chat" in c or "reason_for" in c]

df.to_csv("ecommerce_churn_llm_slm_dataset.csv", index=False)
print("Final clean dataset saved.")

df[
    df["customer_feedback"].astype(str).str.strip() != ""
][
    ["customer_id", "customer_feedback", "support_chat_excerpt", "reason_for_low_activity"]
].sample(5, random_state=42)

df.groupby("churn")["customer_feedback"].apply(lambda x: x.sample(3, random_state=0))

df[
    ["customer_id", "customer_feedback", "support_chat_excerpt", "reason_for_low_activity"]
].to_csv("feedback_preview.csv", index=False)
