import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

DATA_PATH = "student-performance-clustering/data/StudentsPerformance.csv"

FEATURES = [
    "math score",
    "reading score",
    "writing score"
]

def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df, X_scaled, scaler


def train_kmeans():
    df, X_scaled, scaler = load_and_prepare_data()

    kmeans = KMeans(
        n_clusters=2,
        init="k-means++",
        n_init=10,
        random_state=42
    )

    df["cluster"] = kmeans.fit_predict(X_scaled)

    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    centroids_df = pd.DataFrame(centroids, columns=FEATURES)

    return df, kmeans, centroids_df


def get_cluster_counts():
    df, _, _ = train_kmeans()
    return df["cluster"].value_counts().sort_index()


def get_cluster_data():
    df, _, _ = train_kmeans()
    return df


def get_centroids():
    _, _, centroids_df = train_kmeans()
    return centroids_df


if __name__ == "__main__":
    df, kmeans, centroids_df = train_kmeans()

    print("Dataset shape:", df.shape)

    print("\nCluster counts:")
    print(df["cluster"].value_counts().sort_index())

    print("\nCluster centroids:")
    print(centroids_df.round(2))

    print("\nInertia:", round(kmeans.inertia_, 2))
