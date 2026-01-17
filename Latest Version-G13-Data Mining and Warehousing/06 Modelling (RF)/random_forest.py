# =======================================================
# 2) Random Forest (tuned)
# =======================================================
rf_pipe = Pipeline([
    ("prep", preprocess),
    ("clf", RandomForestClassifier(
        random_state=RANDOM_SEED,
        class_weight="balanced",
        n_estimators=300,
        n_jobs=-1
    ))
])

rf_grid = {
    "clf__max_depth": [6, 10, None],
    "clf__min_samples_leaf": [1, 3, 8],
    "clf__max_features": ["sqrt", "log2", 0.5]
}

rf_search = GridSearchCV(
    estimator=rf_pipe,
    param_grid=rf_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1
)

rf_search.fit(X_train, y_train)
rf_best = rf_search.best_estimator_
print("\nRF best params:", rf_search.best_params_)

results.append(evaluate(rf_best, X_train, y_train, X_test, y_test, name="RandomForest (tuned)"))

plot_feature_importance_generic(
    trained_pipe=rf_best,
    X_train=X_train,
    y_train=y_train,
    numeric_cols=numeric_cols,
    categorical_cols=categorical_cols,
    model_name="RandomForest (tuned)",
    top_n=15
)



# get predictions
y_pred = rf_best.predict(X_test)

# plot confusion matrix
disp = ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["No Churn", "Churn"],
    cmap="Blues",
    values_format="d"
)

plt.title("Random Forest (Tuned) – Confusion Matrix")
plt.show()
