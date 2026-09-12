
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# LOAD DATA
# ============================================================

tennis_df = pd.read_csv("data/tennis_matches.csv")

print("Dataset loaded successfully.")
print("Records:", len(tennis_df))


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

X_logistic = tennis_df[["ranking_difference"]]
y_logistic = tennis_df["player_a_wins"]

X_logistic_train, X_logistic_test, y_logistic_train, y_logistic_test = train_test_split(
    X_logistic,
    y_logistic,
    test_size=0.2,
    random_state=42,
    stratify=y_logistic
)

logistic_model = LogisticRegression()

logistic_model.fit(
    X_logistic_train,
    y_logistic_train
)

y_logistic_pred = logistic_model.predict(
    X_logistic_test
)


# ============================================================
# DECISION TREE
# ============================================================

X_tree = tennis_df[
    [
        "player_a_ranking",
        "player_b_ranking",
        "player_a_recent_win_rate",
        "player_b_recent_win_rate"
    ]
]

y_tree = tennis_df["player_a_wins"]

X_tree_train, X_tree_test, y_tree_train, y_tree_test = train_test_split(
    X_tree,
    y_tree,
    test_size=0.2,
    random_state=42,
    stratify=y_tree
)

tree_model = DecisionTreeClassifier(
    random_state=42
)

tree_model.fit(
    X_tree_train,
    y_tree_train
)

y_tree_pred = tree_model.predict(
    X_tree_test
)


# ============================================================
# EVALUATE LOGISTIC REGRESSION
# ============================================================

logistic_cm = confusion_matrix(
    y_logistic_test,
    y_logistic_pred
)

logistic_accuracy = accuracy_score(
    y_logistic_test,
    y_logistic_pred
)

logistic_precision = precision_score(
    y_logistic_test,
    y_logistic_pred
)

logistic_recall = recall_score(
    y_logistic_test,
    y_logistic_pred
)

logistic_f1 = f1_score(
    y_logistic_test,
    y_logistic_pred
)


# ============================================================
# EVALUATE DECISION TREE
# ============================================================

tree_cm = confusion_matrix(
    y_tree_test,
    y_tree_pred
)

tree_accuracy = accuracy_score(
    y_tree_test,
    y_tree_pred
)

tree_precision = precision_score(
    y_tree_test,
    y_tree_pred
)

tree_recall = recall_score(
    y_tree_test,
    y_tree_pred
)

tree_f1 = f1_score(
    y_tree_test,
    y_tree_pred
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("LOGISTIC REGRESSION")
print("========================================")

print("\nConfusion Matrix:")
print(logistic_cm)

print("\nAccuracy:", round(logistic_accuracy * 100, 2), "%")
print("Precision:", round(logistic_precision * 100, 2), "%")
print("Recall:", round(logistic_recall * 100, 2), "%")
print("F1-score:", round(logistic_f1 * 100, 2), "%")


print("\n========================================")
print("DECISION TREE")
print("========================================")

print("\nConfusion Matrix:")
print(tree_cm)

print("\nAccuracy:", round(tree_accuracy * 100, 2), "%")
print("Precision:", round(tree_precision * 100, 2), "%")
print("Recall:", round(tree_recall * 100, 2), "%")
print("F1-score:", round(tree_f1 * 100, 2), "%")


# ============================================================
# COMPARISON
# ============================================================

print("\n========================================")
print("MODEL COMPARISON")
print("========================================")

print(
    f"\nLogistic Regression Accuracy: "
    f"{logistic_accuracy * 100:.2f}%"
)

print(
    f"Decision Tree Accuracy: "
    f"{tree_accuracy * 100:.2f}%"
)

print("\nEvaluation completed successfully.")
