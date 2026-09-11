import matplotlib.pyplot as plt
import numpy as np


# Model results
metrics = ["Accuracy", "Precision", "Recall", "F1-score"]

logistic_scores = [66.67, 66.95, 65.83, 66.39]
tree_scores = [53.33, 53.45, 51.67, 52.54]


# Positions for the bars
x = np.arange(len(metrics))
width = 0.35


plt.figure(figsize=(8, 5))

plt.bar(
    x - width / 2,
    logistic_scores,
    width,
    label="Logistic Regression"
)

plt.bar(
    x + width / 2,
    tree_scores,
    width,
    label="Decision Tree"
)


plt.title("Classification Model Evaluation")
plt.xlabel("Evaluation metric")
plt.ylabel("Score (%)")

plt.xticks(x, metrics)
plt.ylim(0, 100)

plt.legend()

plt.tight_layout()

plt.savefig(
    "model_comparison.png",
    dpi=300
)

plt.show()


print("Comparison graph created successfully.")
print("File: model_comparison.png")
