import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

import tennis_model

# 1. DATA VISUALIZATION
# Player A ranking vs Player B ranking, colored by outcome

df = tennis_model.tennis_df

wins = df[df["player_a_wins"] == 1]
losses = df[df["player_a_wins"] == 0]

plt.figure(figsize= (8, 6))

plt.scatter(
    losses["player_a_ranking"],
    losses["player_b_ranking"],
    alpha=0.5,
    label="Player A loses",
    color="#f87171"
)

plt.scatter(
    wins["player_a_ranking"],
    wins["player_b_ranking"],
    alpha=0.5,
    label="Player A wins",
    color="#38bdf8"
)

plt.title("Decision Tree Dataset - Player Rankings by Match Outcome")
plt.xlabel("Player A ranking")
plt.ylabel("Player B ranking")
plt.legend()

plt.tight_layout()
plt.savefig("static/decision_tree_scatter.png", dpi=200)
plt.close()

print("Saved static/decision_tree_scatter.png")

# 2. TREE STRUCTURE VISUALIZATION
# max_depth=3 is only for the drawing (readability).
# The model used for predictions/metrics is the full tree.

plt.figure(figsize=(20, 10))

plot_tree(
    tennis_model.tree_model,
    max_depth=3,
    feature_names=[
        "player_a_ranking",
        "player_b_ranking",
        "player_a_recent_win_rate",
        "player_b_recent_win_rate"
    ],
    class_names=["Player A loses", "Player A wins"],
    filled=True,
    rounded=True,
    fontsize=9
)

plt.title("Decision Tree Structure (top 3 levels shown for readability)")
plt.tight_layout()
plt.savefig("static/decision_tree_structure.png", dpi=200)
plt.close()

print("Saved static/decision_tree_structure.png")
print("\nReal trained-model metrics for reference:")
print(tennis_model.get_tree_metrics())
