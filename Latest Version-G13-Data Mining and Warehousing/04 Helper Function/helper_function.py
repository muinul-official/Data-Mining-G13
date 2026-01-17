"""# Helper Function for Plotting Feature Importance"""

# Generic feature importance (works for more models)

def plot_feature_importance_generic(
    trained_pipe,
    X_train,
    y_train,
    numeric_cols,
    categorical_cols,
    model_name,
    top_n=15,
    n_repeats_perm=5
):
    prep = trained_pipe.named_steps["prep"]
    clf = trained_pipe.named_steps["clf"]

    # 1) Tree-based (DT/RF/XGB)
    if hasattr(clf, "feature_importances_"):
        feature_names = get_feature_names_from_preprocess(prep, numeric_cols, categorical_cols)
        importances = clf.feature_importances_
        s = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(top_n)

        plt.figure(figsize=(8, 6))
        plt.barh(s.index[::-1], s.values[::-1])
        plt.title(f"{model_name} - Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.grid(True, axis="x")
        plt.show()
        return

    # 2) Logistic Regression coefficients
    if isinstance(clf, LogisticRegression) and hasattr(clf, "coef_"):
        feature_names = get_feature_names_from_preprocess(prep, numeric_cols, categorical_cols)
        importances = np.abs(clf.coef_.ravel())
        s = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(top_n)

        plt.figure(figsize=(8, 6))
        plt.barh(s.index[::-1], s.values[::-1])
        plt.title(f"{model_name} - |Coefficient| Importance")
        plt.xlabel("|Coefficient|")
        plt.ylabel("Feature")
        plt.grid(True, axis="x")
        plt.show()
        return

    # 3) Permutation importance fallback (MLP, etc.)
    perm = permutation_importance(
        trained_pipe,
        X_train,
        y_train,
        n_repeats=n_repeats_perm,
        random_state=RANDOM_SEED,
        scoring="f1",
        n_jobs=-1
    )

    # NOTE: permutation_importance returns importances per original columns (before OHE)
    # Since we are using a Pipeline with preprocess, we keep it simple and show original feature columns.

    s = pd.Series(perm.importances_mean, index=X_train.columns).sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(8, 6))
    plt.barh(s.index[::-1], s.values[::-1])
    plt.title(f"{model_name} - Permutation Importance (F1)")
    plt.xlabel("Mean Importance")
    plt.ylabel("Feature")
    plt.grid(True, axis="x")
    plt.show()
