import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix


# ============================================================
# LOAD DATA
# ============================================================

tennis_df = pd.read_csv("data/tennis_matches.csv")

print("Dataset loaded successfully.")
print("Records:", len(tennis_df))


# ============================================================
# 1. CLASS DISTRIBUTION
# ============================================================

class_counts = tennis_df["player_a_wins"].value_counts().sort_index()

labels = ["Player A loses", "Player A wins"]

plt.figure(figsize=(7, 5))

plt.bar(
    labels,
    class_counts.values
)

plt.title("Tennis Match Classification Dataset")
plt.xlabel("Match outcome")
plt.ylabel("Number of matches")

plt.tight_layout()

plt.savefig(
    "class_distribution.png",
    dpi=300
)

plt.show()


# ============================================================
# 2. RANKING DIFFERENCE BY CLASS
# ============================================================

plt.figure(figsize=(8, 5))

loses = tennis_df[tennis_df["player_a_wins"] == 0]
wins = tennis_df[tennis_df["player_a_wins"] == 1]

plt.scatter(
    loses["ranking_difference"],
    loses["player_a_wins"],
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

plt.title("Ranking Difference and Match Outcome")

plt.xlabel(
    "Ranking difference (Player B ranking - Player A ranking)"
)

plt.ylabel("Match outcome")

plt.legend()

plt.tight_layout()

plt.savefig(
    "static/ranking_difference.png",
    dpi=300
)

plt.show()

# ============================================================
# PREPARE DATA FOR LOGISTIC REGRESSION
# ============================================================

X_logistic = tennis_df[
    ["ranking_difference"]
]

y_logistic = tennis_df[
    "player_a_wins"
]


X_logistic_train, X_logistic_test, y_logistic_train, y_logistic_test = train_test_split(
    X_logistic,
    y_logistic,
    test_size=0.2,
    random_state=42,
    stratify=y_logistic
)


# ============================================================
# TRAIN LOGISTIC REGRESSION
# ============================================================

logistic_model = LogisticRegression()

logistic_model.fit(
    X_logistic_train,
    y_logistic_train
)

y_logistic_pred = logistic_model.predict(
    X_logistic_test
)


# ============================================================
# PREPARE DATA FOR DECISION TREE
# ============================================================

X_tree = tennis_df[
    [
        "player_a_ranking",
        "player_b_ranking",
        "player_a_recent_win_rate",
        "player_b_recent_win_rate"
    ]
]

y_tree = tennis_df[
    "player_a_wins"
]


X_tree_train, X_tree_test, y_tree_train, y_tree_test = train_test_split(
    X_tree,
    y_tree,
    test_size=0.2,
    random_state=42,
    stratify=y_tree
)


# ============================================================
# TRAIN DECISION TREE
# ============================================================

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
# 3. LOGISTIC REGRESSION CONFUSION MATRIX
# ============================================================

logistic_cm = confusion_matrix(
    y_logistic_test,
    y_logistic_pred
)

plt.figure(figsize=(6, 5))

plt.imshow(
    logistic_cm
)

plt.xticks(
    [0, 1],
    ["Predicted Lose", "Predicted Win"]
)

plt.yticks(
    [0, 1],
    ["Actual Lose", "Actual Win"]
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

plt.title(
    "Logistic Regression - Confusion Matrix"
)

plt.xlabel("Prediction")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "logistic_confusion_matrix.png",
    dpi=300
)

plt.show()


# ============================================================
# 4. DECISION TREE CONFUSION MATRIX
# ============================================================

tree_cm = confusion_matrix(
    y_tree_test,
    y_tree_pred
)

plt.figure(figsize=(6, 5))

plt.imshow(
    tree_cm
)

plt.xticks(
    [0, 1],
    ["Predicted Lose", "Predicted Win"]
)

plt.yticks(
    [0, 1],
    ["Actual Lose", "Actual Win"]
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

plt.title(
    "Decision Tree - Confusion Matrix"
)

plt.xlabel("Prediction")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "tree_confusion_matrix.png",
    dpi=300
)

plt.show()


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\nVisualizations created successfully.")

print("\nGenerated files:")
print("- class_distribution.png")
print("- ranking_difference.png")
print("- logistic_confusion_matrix.png")
print("- tree_confusion_matrix.png")
