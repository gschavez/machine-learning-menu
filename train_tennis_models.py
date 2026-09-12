import pandas as pd
from sklearn.model_selection import train_test_split

# Load the prepared tennis dataset
tennis_df = pd.read_csv("data/tennis_matches.csv")

print("Dataset loaded successfully.")
print("Records:", len(tennis_df))
print("Columns:", tennis_df.columns.tolist())

# ============================================================
# LOGISTIC REGRESSION
# ============================================================

# Logistic Regression uses one independent variable:
# ranking_difference
#
# Positive value:
# Player A has a better ranking than Player B.
#
# Negative value:
# Player B has a better ranking than Player A.

X_logistic = tennis_df[[
    "ranking_difference"
]]

y_logistic = tennis_df["player_a_wins"]

# Split the data into 80% training and 20% testing
X_logistic_train, X_logistic_test, y_logistic_train, y_logistic_test = train_test_split(
    X_logistic,
    y_logistic,
    test_size=0.2,
    random_state=42,
    stratify=y_logistic
)

print("\nLogistic Regression data:")
print("Training records:", len(X_logistic_train))
print("Testing records:", len(X_logistic_test))

print("\nTraining class distribution:")
print(y_logistic_train.value_counts())

print("\nTesting class distribution:")
print(y_logistic_test.value_counts())


# ============================================================
# DECISION TREE
# ============================================================

# Decision Tree uses four independent variables.

X_tree = tennis_df[[
    "player_a_ranking",
    "player_b_ranking",
    "player_a_recent_win_rate",
    "player_b_recent_win_rate"
]]

y_tree = tennis_df["player_a_wins"]

# Split the data into 80% training and 20% testing
X_tree_train, X_tree_test, y_tree_train, y_tree_test = train_test_split(
    X_tree,
    y_tree,
    test_size=0.2,
    random_state=42,
    stratify=y_tree
)

print("\nDecision Tree data:")
print("Training records:", len(X_tree_train))
print("Testing records:", len(X_tree_test))

print("\nTraining class distribution:")
print(y_tree_train.value_counts())

print("\nTesting class distribution:")
print(y_tree_test.value_counts())

# ============================================================
# TRAIN THE MODELS
# ============================================================

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Train Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(X_logistic_train, y_logistic_train)

# Train Decision Tree
tree_model = DecisionTreeClassifier(
    random_state=42
)
tree_model.fit(X_tree_train, y_tree_train)

print("\nModels trained successfully.")


# ============================================================
# MAKE PREDICTIONS
# ============================================================

# Logistic Regression predictions
y_logistic_pred = logistic_model.predict(X_logistic_test)

# Decision Tree predictions
y_tree_pred = tree_model.predict(X_tree_test)

print("\nPredictions generated successfully.")

print("\nLogistic Regression predictions:")
print(y_logistic_pred[:10])

print("\nDecision Tree predictions:")
print(y_tree_pred[:10])

# ============================================================
# MODEL EVALUATION
# ============================================================

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ------------------------------------------------------------
# Logistic Regression evaluation
# ------------------------------------------------------------

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


print("\n========================================")
print("LOGISTIC REGRESSION EVALUATION")
print("========================================")

print("\nConfusion Matrix:")
print(logistic_cm)

print("\nAccuracy:", round(logistic_accuracy, 4))
print("Precision:", round(logistic_precision, 4))
print("Recall:", round(logistic_recall, 4))
print("F1-score:", round(logistic_f1, 4))


# ------------------------------------------------------------
# Decision Tree evaluation
# ------------------------------------------------------------

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


print("\n========================================")
print("DECISION TREE EVALUATION")
print("========================================")

print("\nConfusion Matrix:")
print(tree_cm)

print("\nAccuracy:", round(tree_accuracy, 4))
print("Precision:", round(tree_precision, 4))
print("Recall:", round(tree_recall, 4))
print("F1-score:", round(tree_f1, 4))

# ============================================================
# VISUALIZATIONS
# ============================================================

import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. CLASS DISTRIBUTION
# ------------------------------------------------------------

class_counts = tennis_df["player_a_wins"].value_counts().sort_index()

plt.figure(figsize=(7, 5))

plt.bar(
    ["Player A loses", "Player A wins"],
    class_counts.values
)

plt.title("Tennis Match Classification Dataset")
plt.xlabel("Match outcome")
plt.ylabel("Number of matches")

plt.tight_layout()
plt.savefig("class_distribution.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# 2. RANKING DIFFERENCE AND MATCH OUTCOME
# ------------------------------------------------------------

wins = tennis_df[tennis_df["player_a_wins"] == 1]
losses = tennis_df[tennis_df["player_a_wins"] == 0]

plt.figure(figsize=(8, 5))

plt.scatter(
    losses["ranking_difference"],
    losses["player_a_wins"],
    alpha=0.5,
    label="Player A loses"
)

plt.scatter(
    wins["ranking_difference"],
    wins["player_a_wins"],
    alpha=0.5,
    label="Player A wins"
)

plt.yticks(
    [0, 1],
    ["Player A loses", "Player A wins"]
)

plt.axvline(0, linestyle="--")

plt.title("Ranking Difference and Match Outcome")
plt.xlabel(
    "Ranking difference (Player B ranking - Player A ranking)"
)
plt.ylabel("Match outcome")
plt.legend()

plt.tight_layout()
plt.savefig("ranking_difference.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# 3. LOGISTIC REGRESSION CONFUSION MATRIX
# ------------------------------------------------------------

plt.figure(figsize=(6, 5))

plt.imshow(logistic_cm)

plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks(
    [0, 1],
    ["Player A loses", "Player A wins"]
)

plt.yticks(
    [0, 1],
    ["Player A loses", "Player A wins"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            logistic_cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()
plt.savefig("logistic_confusion_matrix.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# 4. DECISION TREE CONFUSION MATRIX
# ------------------------------------------------------------

plt.figure(figsize=(6, 5))

plt.imshow(tree_cm)

plt.title("Decision Tree - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks(
    [0, 1],
    ["Player A loses", "Player A wins"]
)

plt.yticks(
    [0, 1],
    ["Player A loses", "Player A wins"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            tree_cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()
plt.savefig("tree_confusion_matrix.png", dpi=300)
plt.show()


print("\nVisualizations created successfully.")