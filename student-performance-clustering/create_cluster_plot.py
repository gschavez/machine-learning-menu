import sys
import matplotlib.pyplot as plt

sys.path.append("student-performance-clustering/models")

from student_clustering import train_kmeans

df, kmeans, centroids_df = train_kmeans()

plt.figure(figsize=(10, 7))

for cluster in sorted(df["cluster"].unique()):
    cluster_data = df[df["cluster"] == cluster]

    plt.scatter(
        cluster_data["math score"],
        cluster_data["reading score"],
        label=f"Cluster {cluster}",
        alpha=0.6
    )

plt.scatter(
    centroids_df["math score"],
    centroids_df["reading score"],
    marker="X",
    s=250,
    label="Centroids"
)

plt.xlabel("Math Score")
plt.ylabel("Reading Score")
plt.title("Student Performance Clusters")
plt.legend()
plt.grid(True, alpha=0.2)
plt.tight_layout()

plt.savefig(
    "student-performance-clustering/static/student_clusters.png",
    dpi=150
)

plt.close()

print("Scatter plot created successfully.")
