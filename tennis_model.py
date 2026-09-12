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


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

X_logistic = tennis_df[[
    "ranking_difference"
]]

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

X_tree = tennis_df[[
    "player_a_ranking",
    "player_b_ranking",
    "player_a_recent_win_rate",
    "player_b_recent_win_rate"
]]

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
# LOGISTIC REGRESSION EVALUATION
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
# DECISION TREE EVALUATION
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
# GENERAL DATA FUNCTIONS
# ============================================================

def get_num_records():
    """Return the total number of records in the dataset."""
    return len(tennis_df)


def get_training_records():
    """Return the number of training records."""
    return len(X_logistic_train)


def get_testing_records():
    """Return the number of testing records."""
    return len(X_logistic_test)


# ============================================================
# LOGISTIC REGRESSION PREDICTION
# ============================================================

def predict_logistic(ranking_difference):
    input_data = pd.DataFrame(
        [[ranking_difference]],
        columns=["ranking_difference"]
    )

    prediction = logistic_model.predict(input_data)
    return int(prediction[0])


# ============================================================
# DECISION TREE PREDICTION
# ============================================================

def predict_tree(
    player_a_ranking,
    player_b_ranking,
    player_a_recent_win_rate,
    player_b_recent_win_rate
):
    input_data = pd.DataFrame(
        [[
            player_a_ranking,
            player_b_ranking,
            player_a_recent_win_rate,
            player_b_recent_win_rate
        ]],
        columns=[
            "player_a_ranking",
            "player_b_ranking",
            "player_a_recent_win_rate",
            "player_b_recent_win_rate"
        ]
    )

    prediction = tree_model.predict(input_data)
    return int(prediction[0])

# ============================================================
# LOGISTIC REGRESSION METRICS
# ============================================================

def get_logistic_metrics():
    """Return Logistic Regression evaluation metrics."""

    return {
        "accuracy": logistic_accuracy,
        "precision": logistic_precision,
        "recall": logistic_recall,
        "f1": logistic_f1
    }


def get_logistic_confusion_matrix():
    """Return Logistic Regression confusion matrix."""

    return logistic_cm


# ============================================================
# DECISION TREE METRICS
# ============================================================

def get_tree_metrics():
    """Return Decision Tree evaluation metrics."""

    return {
        "accuracy": tree_accuracy,
        "precision": tree_precision,
        "recall": tree_recall,
        "f1": tree_f1
    }


def get_tree_confusion_matrix():
    """Return Decision Tree confusion matrix."""

    return tree_cm