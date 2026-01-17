# =======================================================
# 3) XGBoost (tuned) + imbalance handling
# =======================================================
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = (neg / pos) if pos > 0 else 1.0

xgb_pipe = Pipeline([
    ("prep", preprocess),
    ("clf", XGBClassifier(
        random_state=RANDOM_SEED,
        n_estimators=300,
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight
    ))
])

xgb_grid = {
    "clf__max_depth": [3, 5, 7],
    "clf__learning_rate": [0.03, 0.05, 0.1],
    "clf__subsample": [0.8, 1.0],
    "clf__colsample_bytree": [0.8, 1.0],
    "clf__min_child_weight": [1, 5, 10]
}

xgb_search = GridSearchCV(
    estimator=xgb_pipe,
    param_grid=xgb_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1
)

xgb_search.fit(X_train, y_train)
xgb_best = xgb_search.best_estimator_
print("\nXGB best params:", xgb_search.best_params_)

results.append(evaluate(xgb_best, X_train, y_train, X_test, y_test, name="XGBoost (tuned)"))

plot_feature_importance_generic(
    trained_pipe=xgb_best,
    X_train=X_train,
    y_train=y_train,
    numeric_cols=numeric_cols,
    categorical_cols=categorical_cols,
    model_name="XGBoost (tuned)",
    top_n=15
)


# get predictions
y_pred = xgb_best.predict(X_test)

# plot confusion matrix
disp = ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["No Churn", "Churn"],
    cmap="Blues",
    values_format="d"
)

plt.title("XGBoost (Tuned) – Confusion Matrix")
plt.show()
