import os
import sys

from flask import Flask, render_template

# Add the models directory to Python's import path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

sys.path.append(MODELS_DIR)

from student_clustering import train_kmeans


app = Flask(__name__)


@app.route("/")
def index():
    # Train the K-Means model and obtain the results
    df, kmeans, centroids_df = train_kmeans()

    # Number of students in each cluster
    cluster_counts = (
        df["cluster"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    # Convert centroid information to a format that Jinja can use
    centroids = centroids_df.round(2).to_dict(orient="records")

    # Convert all student records to dictionaries
    students = df.to_dict(orient="records")

    return render_template(
        "student-clustering.html",
        students=students,
        cluster_counts=cluster_counts,
        centroids=centroids,
        features=[
            "Math Score",
            "Reading Score",
            "Writing Score"
        ],
        n_clusters=kmeans.n_clusters,
        inertia=round(kmeans.inertia_, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)