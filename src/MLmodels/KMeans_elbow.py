import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)


from src.data.load_data import load_data, get_project_root


def load_clustering_data(use_preprocessed=False):
    """
    Loads dataset and prepares features for K-Means clustering.
    Can load selected numerical indicators or the full preprocessed feature set.
    """
    if use_preprocessed:
        train_path = os.path.join(project_root, "data", "preprocessed_train.csv")
        if os.path.exists(train_path):
            df = pd.read_csv(train_path)
            drop_cols = ["PlacementStatus", "Salary Package", "StudentID", "IsAnomaly"]
            feature_cols = [c for c in df.columns if c not in drop_cols]
            X = df[feature_cols].copy()
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            return X_scaled, X, feature_cols, scaler

    # Default: Use key student performance and skill features
    df = load_data()
    feature_cols = [
        "CGPA",
        "CodingTestScore",
        "AptitudeTestScore",
        "SoftSkillsRating",
        "Internships",
        "Projects"
    ]
    # Filter available columns
    available_cols = [c for c in feature_cols if c in df.columns]
    X_raw = df[available_cols].dropna().reset_index(drop=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    return X_scaled, X_raw, available_cols, scaler


def find_optimal_k(X, k_range=range(1, 11), save_chart=True, filename="kmeans_elbow.png"):
    """
    Computes WCSS (Within-Cluster Sum of Squares / Inertia) for a range of k values
    and plots the Elbow Method curve to identify optimal clusters.
    """
    wcss = []
    print("\nCalculating WCSS for Elbow Method (k=1 to 10)...")
    for k in k_range:
        km = KMeans(
            n_clusters=k,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=42
        )
        km.fit(X)
        wcss.append(km.inertia_)

    print("\nWCSS Values across Clusters:")
    for k, val in zip(k_range, wcss):
        print(f"  K = {k:2d}  |  WCSS = {val:,.2f}")

    if save_chart:
        charts_dir = os.path.join(project_root, "app", "static", "charts")
        os.makedirs(charts_dir, exist_ok=True)
        chart_path = os.path.join(charts_dir, filename)

        plt.figure(figsize=(8, 5))
        plt.plot(list(k_range), wcss, marker='o', color='#2563eb', linewidth=2.5, markersize=7)
        plt.title("Elbow Method for Optimal K (Student Clustering)", fontsize=13, fontweight='bold')
        plt.xlabel("Number of Clusters (K)", fontsize=11)
        plt.ylabel("Within-Cluster Sum of Squares (WCSS / Inertia)", fontsize=11)
        plt.xticks(list(k_range))
        plt.grid(True, linestyle="--", alpha=0.6)
        
        # Annotate suggested elbow point at K=3
        if len(wcss) >= 3:
            plt.annotate(
                'Optimal Elbow Point (K=3)',
                xy=(3, wcss[2]),
                xytext=(4, wcss[2] * 1.15),
                arrowprops=dict(facecolor='#ef4444', shrink=0.05, width=1.5, headwidth=8),
                fontsize=10,
                fontweight='semibold',
                color='#ef4444'
            )

        plt.tight_layout()
        plt.savefig(chart_path, dpi=300)
        plt.close()
        print(f"[INFO] Elbow curve chart saved to: {chart_path}")

    # Optimal K heuristic (typically 3 for student performance: Low, Medium, High)
    optimal_k = 3
    return wcss, optimal_k


def create_model(k=3):
    """Initializes a KMeans clustering model with specified k."""
    return KMeans(
        n_clusters=k,
        init='k-means++',
        n_init=10,
        max_iter=300,
        random_state=42
    )


def train_model(model, X):
    """Fits the KMeans model on input data and returns model with cluster labels."""
    print(f"[INFO] Training K-Means model with K={model.n_clusters} clusters...", flush=True)
    labels = model.fit_predict(X)
    return model, labels


def create_and_tune_model(X, k=3):
    """Convenience pipeline function to instantiate and train KMeans."""
    model = create_model(k=k)
    return train_model(model, X)


def evaluate_model(model, X, labels):
    """
    Evaluates clustering performance using:
    1. Inertia (WCSS)
    2. Convergence iterations
    3. Silhouette Score
    """
    inertia = model.inertia_
    n_iter = model.n_iter_
    
    # Calculate silhouette score (using representative sample if large dataset for rapid evaluation)
    if len(X) > 10000:
        silhouette = silhouette_score(X, labels, sample_size=10000, random_state=42)
    else:
        silhouette = silhouette_score(X, labels)

    print("\n" + "=" * 50)
    print("        K-MEANS CLUSTERING EVALUATION METRICS       ")
    print("=" * 50)
    print(f" Number of Clusters (K):  {model.n_clusters}")
    print(f" Inertia (WCSS):          {inertia:,.2f}")
    print(f" Iterations to Converge:  {n_iter}")
    print(f" Silhouette Score:        {silhouette:.4f}")
    print("=" * 50)

    return {
        "clusters": model.n_clusters,
        "inertia": inertia,
        "iterations": n_iter,
        "silhouette_score": silhouette
    }


def display_clusters(X_raw, labels, model, scaler, feature_x="CGPA", feature_y="CodingTestScore", filename="kmeans_elbow_clusters.png"):
    """
    Generates a 2D scatter plot visualization of the student clusters with centroids.
    Saves the plot to app/static/charts/.
    """
    charts_dir = os.path.join(project_root, "app", "static", "charts")
    os.makedirs(charts_dir, exist_ok=True)
    plot_path = os.path.join(charts_dir, filename)

    plt.figure(figsize=(9, 6))

    # Scatter plot data points colored by cluster
    scatter = plt.scatter(
        X_raw[feature_x],
        X_raw[feature_y],
        c=labels,
        cmap='viridis',
        alpha=0.6,
        edgecolors='none',
        s=35
    )

    # Inverse transform centroids to original scale for plotting
    centers_scaled = model.cluster_centers_
    centers_original = scaler.inverse_transform(centers_scaled)

    cols = list(X_raw.columns)
    idx_x = cols.index(feature_x)
    idx_y = cols.index(feature_y)

    plt.scatter(
        centers_original[:, idx_x],
        centers_original[:, idx_y],
        marker='X',
        s=220,
        c='red',
        edgecolors='black',
        linewidths=1.5,
        label='Cluster Centroids'
    )

    plt.title(f"Student Segmentation: {feature_x} vs {feature_y} (K={model.n_clusters})", fontsize=13, fontweight='bold')
    plt.xlabel(feature_x, fontsize=11)
    plt.ylabel(feature_y, fontsize=11)
    plt.colorbar(scatter, label="Cluster Group")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"[INFO] Cluster scatter plot saved to: {plot_path}")


def save_model(model, scaler=None, feature_cols=None):
    """Saves trained KMeans model and metadata to models/ directory."""
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    artifact = {
        "model": model,
        "scaler": scaler,
        "features": feature_cols,
        "n_clusters": model.n_clusters
    }

    save_path_elbow = os.path.join(models_dir, "kmeans_elbow.pkl")
    save_path_general = os.path.join(models_dir, "kmeans_model.pkl")

    joblib.dump(artifact, save_path_elbow)
    joblib.dump(artifact, save_path_general)
    print(f"[SUCCESS] KMeans model artifact saved to: {save_path_elbow}")


def main():
    print("==================================================")
    print("   K-MEANS CLUSTERING WITH ELBOW METHOD PIPELINE  ")
    print("==================================================")

    # 1. Load Data
    print("\n[1/5] Loading and preparing student data...")
    X_scaled, X_raw, feature_cols, scaler = load_clustering_data()
    print(f"      Features used ({len(feature_cols)}): {feature_cols}")
    print(f"      Dataset Shape: {X_scaled.shape}")

    # 2. Elbow Method for Optimal K
    print("\n[2/5] Running Elbow Method to find optimal K...")
    wcss_list, optimal_k = find_optimal_k(X_scaled, k_range=range(1, 11))
    print(f"      Optimal K determined: {optimal_k}")

    # 3. Train KMeans Model with Optimal K
    print(f"\n[3/5] Training final K-Means model with K={optimal_k}...")
    model = create_model(k=optimal_k)
    model, labels = train_model(model, X_scaled)

    # 4. Evaluate Clustering Performance
    print("\n[4/5] Evaluating clustering metrics...")
    metrics = evaluate_model(model, X_scaled, labels)

    # 5. Visualize & Save Artifacts
    print("\n[5/5] Generating cluster visualization and saving model artifact...")
    display_clusters(X_raw, labels, model, scaler, feature_x="CGPA", feature_y="CodingTestScore")
    save_model(model, scaler, feature_cols)

    # Calculate cluster profiles for interpretability
    X_raw_analyzed = X_raw.copy()
    X_raw_analyzed["Cluster"] = labels
    cluster_means = X_raw_analyzed.groupby("Cluster").mean()
    print("\nCluster Profiles (Average Metrics per Group):")
    print(cluster_means.round(2))

    print("\n==================================================")
    print("   K-MEANS ELBOW PIPELINE COMPLETED SUCCESSFULLY! ")
    print("==================================================")


if __name__ == "__main__":
    main()