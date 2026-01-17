# =======================================================
# 4) MLP (tuned) + permutation importance
# =======================================================
mlp_pipe = Pipeline([
    ("prep", preprocess),
    ("clf", MLPClassifier(
        random_state=RANDOM_SEED,
        max_iter=600,
        early_stopping=True,
        validation_fraction=0.15
    ))
])

mlp_grid = {
    "clf__hidden_layer_sizes": [(32,), (64,), (64, 32)],
    "clf__alpha": [1e-4, 1e-3, 1e-2],
    "clf__learning_rate_init": [1e-3, 5e-4]
}

mlp_search = GridSearchCV(
    estimator=mlp_pipe,
    param_grid=mlp_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1
)

mlp_search.fit(X_train, y_train)
mlp_best = mlp_search.best_estimator_
print("\nMLP best params:", mlp_search.best_params_)

results.append(evaluate(mlp_best, X_train, y_train, X_test, y_test, name="MLP (tuned)"))

plot_feature_importance_generic(
    trained_pipe=mlp_best,
    X_train=X_train,
    y_train=y_train,
    numeric_cols=numeric_cols,
    categorical_cols=categorical_cols,
    model_name="MLP (tuned)",
    top_n=15,
    n_repeats_perm=3
)

# =======================================================
# 5) Results summary table (all models)
# =======================================================
results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
display(results_df)
