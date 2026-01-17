"""# LEARNING CURVES"""

# =======================================================
# DT Learning Curves (Underfit / Overfit / Good Fit)
# =======================================================
def plot_learning_curve_f1(estimator, X, y, title):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator,
        X, y,
        cv=5,
        scoring="f1",
        train_sizes=np.linspace(0.1, 1.0, 6),
        n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    plt.figure(figsize=(7, 5))
    plt.plot(train_sizes, train_mean, marker="o", label="Training F1")
    plt.plot(train_sizes, val_mean, marker="s", label="Validation F1")
    plt.title(title)
    plt.xlabel("Training Samples")
    plt.ylabel("F1 Score")
    plt.grid(True)
    plt.legend()
    plt.show()


# Underfit example: shallow DT
dt_underfit = Pipeline([
    ("prep", preprocess),
    ("clf", DecisionTreeClassifier(
        max_depth=2,
        random_state=RANDOM_SEED,
        class_weight="balanced"
    ))
])
plot_learning_curve_f1(dt_underfit, X, y, "Underfitting: Decision Tree (max_depth=2)")


# Overfit example: deep DT
dt_overfit = Pipeline([
    ("prep", preprocess),
    ("clf", DecisionTreeClassifier(
        max_depth=None,
        random_state=RANDOM_SEED,
        class_weight="balanced"
    ))
])
plot_learning_curve_f1(dt_overfit, X, y, "Overfitting: Decision Tree (no depth limit)")


# Good fit: tuned DT from your friend's GridSearchCV
plot_learning_curve_f1(dt_best, X, y, "Good Fit: Tuned Decision Tree")

# =======================================================
# LR Learning Curves (Underfit / Overfit / Good Fit)
# =======================================================

# Underfit LR: very strong regularisation (tiny C)
lr_underfit = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(
        C=0.001,
        max_iter=3000,
        class_weight="balanced",
        solver="liblinear"
    ))
])
plot_learning_curve_f1(lr_underfit, X, y, "Underfitting: Logistic Regression (C=0.001)")


# Overfit-ish LR: very weak regularisation (huge C)
lr_overfit = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(
        C=1000,
        max_iter=3000,
        class_weight="balanced",
        solver="liblinear"
    ))
])
plot_learning_curve_f1(lr_overfit, X, y, "Overfitting-ish: Logistic Regression (C=1000)")


# Good fit LR: tuned LR from GridSearchCV
plot_learning_curve_f1(lr_best, X, y, "Good Fit: Tuned Logistic Regression")
