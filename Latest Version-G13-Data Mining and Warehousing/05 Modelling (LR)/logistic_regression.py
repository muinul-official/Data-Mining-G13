"""# COMPARABLE MODELLING


1.   Logistic Regression
2.   Random Forest
3.   XGBoost
4.   MLP



"""

# =======================================================
# 1) Logistic Regression (tuned)
# =======================================================
lr_pipe = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="liblinear"
    ))
])

lr_grid = {"clf__C": [0.1, 0.5, 1, 2, 5]}

lr_search = GridSearchCV(
    estimator=lr_pipe,
    param_grid=lr_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1
)

lr_search.fit(X_train, y_train)
lr_best = lr_search.best_estimator_
print("\nLR best params:", lr_search.best_params_)

results.append(evaluate(lr_best, X_train, y_train, X_test, y_test, name="LogReg (tuned)"))

plot_feature_importance_generic(
    trained_pipe=lr_best,
    X_train=X_train,
    y_train=y_train,
    numeric_cols=numeric_cols,
    categorical_cols=categorical_cols,
    model_name="LogReg (tuned)",
    top_n=15
)
