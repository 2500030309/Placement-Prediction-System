import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Load Dataset (works with relative path or fallback)
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
csv_path = os.path.join(base_dir, "data", "placement_data.csv")
if not os.path.exists(csv_path):
    csv_path = r"C:\Users\LUCKY\PycharmProjects\Placements Prediction System\data\placement_data.csv"

df = pd.read_csv(csv_path)

# 2. Select numerical features for clustering
features = [
    "CGPA",
    "Internships",
    "Projects",
    "CodingTestScore"
]

X = df[features].dropna()

print("Selected Features:")
print("\nNull Values:", X.isnull().sum())
print(X.head())

# 3. Standardization (Scaling values so all features have mean=0 and variance=1)
scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)

# 4. Train K-Means Clustering Model
# Dividing students into 3 performance clusters (e.g., Low, Medium, High placement readiness)
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init='auto'
)

clusters = kmeans.fit_predict(x_scaled)

# 5. Add cluster assignments back to DataFrame
X["cluster"] = clusters

print("\nCluster Assignment:")
print(X.head(10))

# 6. Calculate Real Cluster Centers (un-scaled values for easy interpretation)
centers_scaled = kmeans.cluster_centers_
centers = scaler.inverse_transform(centers_scaled)

centers_df = pd.DataFrame(
    centers,
    columns=features
)

print("\nCluster Centers (Average per group):")
print(centers_df)

# 7. Visualize and Save Clusters Plot
charts_dir = os.path.join(base_dir, "app", "static", "charts")
os.makedirs(charts_dir, exist_ok=True)
plot_path = os.path.join(charts_dir, "kmeans_clusters.png")

plt.figure(figsize=(8, 5))
scatter = plt.scatter(X["CGPA"], X["CodingTestScore"], c=X["cluster"], cmap="viridis", alpha=0.5)
plt.title("Student Clusters: CGPA vs Coding Test Score", fontsize=14, fontweight='bold')
plt.xlabel("CGPA", fontsize=12)
plt.ylabel("Coding Test Score", fontsize=12)
plt.colorbar(scatter, label="Cluster ID")
plt.tight_layout()
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"\n[INFO] K-Means cluster plot saved to: {plot_path}")