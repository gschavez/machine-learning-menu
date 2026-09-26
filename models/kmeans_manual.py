import io
import base64

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Dataset and context
# ---------------------------------------------------------------------

DATASET_PATH = "data/kmeans-manual-exercise-student-performance.csv"

DATASET_CONTEXT = {
    "title": "Student Performance Segmentation",
    "description": (
        "This dataset represents 100 students from a university course. "
        "Each student is described by two numerical variables used to "
        "understand study habits and academic outcomes. The 100 records "
        "were synthetically generated for this exercise so that the "
        "manual K-Means iterations converge to three clearly "
        "interpretable groups."
    ),
    "variables": [
        (
            "Study_Hours_Weekly",
            "The average number of hours the student spends studying "
            "per week (self-reported, outside of class time)."
        ),
        (
            "Final_Grade",
            "The student's final grade in the course, on a 0 to 100 "
            "scale."
        ),
    ],
    "goal": (
        "Apply K-Means with K = 3 to group these 100 students into "
        "three performance segments based on study hours and final "
        "grade, performing the calculations by hand (with formulas) "
        "across three iterations, and interpret what each resulting "
        "segment represents for the course."
    ),

}


INITIAL_CENTROIDS = np.array([
    [5.0, 20.0],    # Cluster 1
    [15.0, 50.0],   # Cluster 2
    [30.0, 85.0],   # Cluster 3
])

CLUSTER_COLORS = ["#38bdf8", "#a78bfa", "#f472b6"]
CLUSTER_LABELS = [
    "Cluster 1 - At-risk students",
    "Cluster 2 - Average students",
    "Cluster 3 - High-achieving students",
]

INTERPRETATION_TEXT = (
    "After 3 iterations, K-Means converges to three student segments "
    "based on weekly study hours and final grade.\n\n"
    "Cluster 1 - At-risk students: fewer weekly study hours combined "
    "with a lower final grade. These students may benefit from "
    "tutoring, more frequent check-ins, or study skills support.\n\n"
    "Cluster 2 - Average students: a moderate number of study hours "
    "and a moderate final grade. This is the most 'typical' segment "
    "of the course.\n\n"
    "Cluster 3 - High-achieving students: more weekly study hours "
    "combined with a higher final grade. This segment could be given "
    "more advanced or enrichment material.\n\n"
    "Comparing Iteration 2 and Iteration 3, most students keep the "
    "same cluster, but a small number of students near the boundary "
    "between groups still move, and the total within-cluster variance "
    "keeps decreasing (178.37 to 106.63 to 94.73). This shows the "
    "algorithm is still converging: the solution is improving with "
    "each iteration, but it has not fully stabilized within only 3 "
    "iterations."
)


# ---------------------------------------------------------------------
# Main K-Means calculation (manual style)
# ---------------------------------------------------------------------

def _load_dataset():

    with open(DATASET_PATH, "r", encoding="utf-8-sig") as f:
        first_line = f.readline()

    sep = ";" if first_line.count(";") >= first_line.count(",") else ","
  
    decimal = "," if sep == ";" else "."

    df = pd.read_csv(
        DATASET_PATH,
        sep=sep,
        decimal=decimal,
        encoding="utf-8-sig" 
    )


    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]

    required = {"ID", "Study_Hours_Weekly", "Final_Grade"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas {missing} en '{DATASET_PATH}'. "
            f"Columnas detectadas: {df.columns.tolist()}. "
            f"Separador usado: '{sep}'. Revisa el archivo CSV: probablemente "
            f"tiene un encabezado, separador o codificación distinta a la esperada."
        )

    return df


def _run_manual_kmeans(df, initial_centroids, n_iterations=3):

    X = df[["Study_Hours_Weekly", "Final_Grade"]].to_numpy(dtype=float)
    ids = df["ID"].to_numpy()

    centroids = initial_centroids.copy()
    iterations = []

    for step in range(n_iterations):
        centroids_in = centroids.copy()

        # Distance from each point to each centroid
        diffs = X[:, None, :] - centroids_in[None, :, :]
        distances = np.sqrt((diffs ** 2).sum(axis=2))

        clusters = distances.argmin(axis=1)
        nearest_dist = distances[np.arange(len(X)), clusters]
        sq_dist = nearest_dist ** 2

        # New centroids
        new_centroids = np.zeros_like(centroids_in)
        counts = np.zeros(len(centroids_in), dtype=int)
        cluster_variance = np.zeros(len(centroids_in))

        for k in range(len(centroids_in)):
            mask = clusters == k
            counts[k] = mask.sum()
            if counts[k] > 0:
                new_centroids[k] = X[mask].mean(axis=0)
                cluster_variance[k] = sq_dist[mask].mean()
            else:
                new_centroids[k] = centroids_in[k]
                cluster_variance[k] = 0.0

        total_variance = sq_dist.mean()

        records = pd.DataFrame({
            "ID": ids,
            "Hours": X[:, 0],
            "Grade": X[:, 1],
            "Dist_C1": distances[:, 0],
            "Dist_C2": distances[:, 1],
            "Dist_C3": distances[:, 2],
            "Cluster": clusters + 1,
            "SqDist": sq_dist,
        })

        iterations.append({
            "step": step + 1,
            "centroids_in": centroids_in,
            "records": records,
            "clusters": clusters,
            "new_centroids": new_centroids,
            "counts": counts,
            "cluster_variance": cluster_variance,
            "total_variance": total_variance,
        })

        centroids = new_centroids

    return iterations


_df = _load_dataset()
_iterations = _run_manual_kmeans(_df, INITIAL_CENTROIDS, n_iterations=3)


# ---------------------------------------------------------------------
# Public accessors
# ---------------------------------------------------------------------

def get_context():
    return DATASET_CONTEXT


def get_dataset_preview(n=10):
    return _df.head(n).to_dict(orient="records")


def get_num_records():
    return len(_df)


def get_initial_centroids():
    return INITIAL_CENTROIDS.tolist()


def get_iteration(step):
    """step es 1, 2 o 3"""
    data = _iterations[step - 1]
    table = data["records"].copy()
    table["Dist_C1"] = table["Dist_C1"].round(2)
    table["Dist_C2"] = table["Dist_C2"].round(2)
    table["Dist_C3"] = table["Dist_C3"].round(2)
    table["SqDist"] = table["SqDist"].round(2)

    return {
        "step": data["step"],
        "centroids_in": data["centroids_in"].round(2).tolist(),
        "new_centroids": data["new_centroids"].round(4).tolist(),
        "counts": data["counts"].tolist(),
        "cluster_variance": [round(v, 2) for v in data["cluster_variance"]],
        "total_variance": round(data["total_variance"], 2),
        "records": table.to_dict(orient="records"),
    }


def get_variance_comparison():
    return [
        {"iteration": i + 1, "total_variance": round(it["total_variance"], 2)}
        for i, it in enumerate(_iterations)
    ]


def get_interpretation():
    return INTERPRETATION_TEXT


def get_cluster_labels():
    return CLUSTER_LABELS


# ---------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------

def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def generate_initial_plot():
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(
        _df["Study_Hours_Weekly"], _df["Final_Grade"],
        alpha=0.6, color="#94a3b8", label="Students"
    )
    ax.scatter(
        INITIAL_CENTROIDS[:, 0], INITIAL_CENTROIDS[:, 1],
        marker="X", s=220, color="#f87171",
        edgecolor="black", linewidth=1, label="Initial centroids"
    )
    ax.set_title("Initial Data and Initial Centroids")
    ax.set_xlabel("Weekly Study Hours")
    ax.set_ylabel("Final Grade")
    ax.legend()
    ax.grid(alpha=0.2)
    return _fig_to_base64(fig)


def generate_iteration_plot(step):
    data = _iterations[step - 1]
    clusters = data["clusters"]
    new_centroids = data["new_centroids"]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for k in range(3):
        mask = clusters == k
        ax.scatter(
            _df["Study_Hours_Weekly"][mask], _df["Final_Grade"][mask],
            alpha=0.7, color=CLUSTER_COLORS[k], label=CLUSTER_LABELS[k]
        )

    ax.scatter(
        new_centroids[:, 0], new_centroids[:, 1],
        marker="X", s=220, color="black",
        edgecolor="white", linewidth=1, label="Updated centroids"
    )
    ax.set_title(f"Iteration {step} - Cluster Assignment and New Centroids")
    ax.set_xlabel("Weekly Study Hours")
    ax.set_ylabel("Final Grade")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.2)
    return _fig_to_base64(fig)


def generate_variance_plot():
    comparison = get_variance_comparison()
    xs = [c["iteration"] for c in comparison]
    ys = [c["total_variance"] for c in comparison]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, marker="o", color="#38bdf8", linewidth=2)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    ax.set_title("Within-Cluster Variance per Iteration")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total within-cluster variance")
    ax.set_xticks(xs)
    ax.grid(alpha=0.2)
    return _fig_to_base64(fig)