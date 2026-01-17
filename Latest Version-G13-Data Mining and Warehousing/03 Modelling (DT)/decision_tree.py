# define ML Inputs
target = "churn"
drop_cols = ["customer_id"] if "customer_id" in df.columns else []

RAW_TEXT_TO_EXCLUDE = ["customer_feedback", "support_chat_excerpt", "reason_for_low_activity", "combined_text"]
RAW_TEXT_TO_EXCLUDE = [c for c in RAW_TEXT_TO_EXCLUDE if c in df.columns]

X = df.drop(columns=[target] + drop_cols + RAW_TEXT_TO_EXCLUDE)
y = df[target]

# split cols into numeric vs categorical
# numeric : scaling, categorical : onehotencoding
numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]

# preprocessing pipeline (standardize numeric, one-hot encode categorical)
preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("scaler", StandardScaler())]), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

# Train/test split + CV
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

# quick check : show engineered cols
print("Engineered columns added:",
      [c for c in ["slm_sentiment","has_feedback","has_support_chat","feedback_len","support_len","combined_len",
                   "orders_per_session_approx","spend_per_order_approx"] if c in df.columns])

print("\nX shape (after excluding raw text):", X.shape)
print("Numeric cols:", len(numeric_cols), "| Categorical cols:", len(categorical_cols))
display(X.head())

# save features & target separately
X.to_csv("X_features.csv", index=False)
y.to_csv("y_target.csv", index=False)

print("Saved X_features.csv and y_target.csv")

"""# DECISION TREE (baseline)"""

# =======================================================
# BASELINE + TUNED DECISION TREE (with feature importance)
# =======================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# -----------------------------
# 1) Evaluation function
# -----------------------------
def evaluate(model, X_train, y_train, X_test, y_test, name="Model"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    else:
        auc = None

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n=== {name} ===")
    print("Accuracy :", round(acc, 4))
    print("F1-score :", round(f1, 4))
    print("ROC-AUC  :", round(auc, 4) if auc is not None else "N/A")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return {"model": name, "accuracy": acc, "f1": f1, "roc_auc": auc}


# -----------------------------
# 2) Feature importance (Decision Tree in pipeline)
# -----------------------------
def get_feature_names_from_preprocess(fitted_preprocess, numeric_cols, categorical_cols):
    # numeric names
    num_features = list(numeric_cols)

    # one-hot encoded category names
    ohe = fitted_preprocess.named_transformers_["cat"]
    cat_features = list(ohe.get_feature_names_out(categorical_cols))

    return num_features + cat_features


def plot_dt_feature_importance(trained_pipe, numeric_cols, categorical_cols, top_n=15, title="DT Feature Importance"):
    prep = trained_pipe.named_steps["prep"]
    clf = trained_pipe.named_steps["clf"]

    if not hasattr(clf, "feature_importances_"):
        print("This model does not provide feature_importances_.")
        return

    feature_names = get_feature_names_from_preprocess(prep, numeric_cols, categorical_cols)
    importances = clf.feature_importances_

    s = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(8, 6))
    plt.barh(s.index[::-1], s.values[::-1])
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.grid(True, axis="x")
    plt.show()


# -----------------------------
# 3) Baseline Decision Tree
# -----------------------------
results = []

dt_base = Pipeline([
    ("prep", preprocess),
    ("clf", DecisionTreeClassifier(random_state=RANDOM_SEED, class_weight="balanced"))
])

results.append(evaluate(dt_base, X_train, y_train, X_test, y_test, name="DecisionTree (baseline)"))

# Feature importance for baseline
plot_dt_feature_importance(
    trained_pipe=dt_base,
    numeric_cols=numeric_cols,
    categorical_cols=categorical_cols,
    top_n=15,
    title="DecisionTree (baseline) - Feature Importance"
)

# -----------------------------
# 4) Tuned Decision Tree (GridSearchCV)
# -----------------------------
dt_grid = {
    "clf__max_depth": [2, 3, 5, 8, 12, None],
    "clf__min_samples_split": [2, 10, 30],
    "clf__min_samples_leaf": [1, 5, 15]
}

dt_search = GridSearchCV(
    estimator=dt_base,
    param_grid=dt_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1
)

dt_search.fit(X_train, y_train)
dt_best = dt_search.best_estimator_
print("\nDT best params:", dt_search.best_params_)

results.append(evaluate(dt_best, X_train, y_train, X_test, y_test, name="DecisionTree (tuned)"))

# Feature importance for tuned DT
plot_dt_feature_importance(
    trained_pipe=dt_best,
    numeric_cols=numeric_cols,
    categorical_cols=categorical_cols,
    top_n=15,
    title="DecisionTree (tuned) - Feature Importance"
)

# -----------------------------
# 5) Results summary table
# -----------------------------
results_df = pd.DataFrame(results)
display(results_df)
