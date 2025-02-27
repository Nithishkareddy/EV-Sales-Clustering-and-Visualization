import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load CSV files
df_2w = pd.read_csv("2wheel.csv")
df_3w = pd.read_csv("3Wheel.csv")
df_4w = pd.read_csv("4wheel.csv")

# Print column names to verify
print("\nColumns in 2-Wheelers Dataset:", df_2w.columns)
print("\nColumns in 3-Wheelers Dataset:", df_3w.columns)
print("\nColumns in 4-Wheelers Dataset:", df_4w.columns)

# Selecting relevant feature for clustering (Total Sales)
X_2w = df_2w[["Total"]]
X_3w = df_3w[["Total"]]
X_4w = df_4w[["Total"]]

# Apply K-Means clustering
def apply_kmeans(X, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    X["Cluster"] = kmeans.fit_predict(X)
    return X, kmeans

X_2w, kmeans_2w = apply_kmeans(X_2w)
X_3w, kmeans_3w = apply_kmeans(X_3w)
X_4w, kmeans_4w = apply_kmeans(X_4w)

# Print cluster distribution
print("\n2-Wheelers Clusters:\n", X_2w["Cluster"].value_counts())
print("\n3-Wheelers Clusters:\n", X_3w["Cluster"].value_counts())
print("\n4-Wheelers Clusters:\n", X_4w["Cluster"].value_counts())

# Visualizing Clusters
def plot_clusters(X, df, title, manufacturer_column):
    plt.figure(figsize=(14, 7))
    
    # Merge clusters back to original data for labeling
    df = df.copy()
    df["Cluster"] = X["Cluster"]
    
    # Sort data by total sales
    df = df.sort_values(by="Total", ascending=False)
    
    # Create bar plot
    ax = sns.barplot(data=df, x="Total", y=manufacturer_column, hue="Cluster", palette="viridis", dodge=False)

    plt.xlabel("Total EV Sales (Units)")  # Clear label for X-axis
    plt.ylabel("Manufacturer")  # Y-axis label
    plt.title(title)  # Chart title
    plt.legend(title="Cluster")  # Legend title
    
    # Display exact values on the bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", label_type="edge", fontsize=10, padding=3)

    plt.xticks(rotation=45)  # Rotate X-axis labels if needed
    plt.grid(axis="x", linestyle="--", alpha=0.7)  # Light grid for clarity
    plt.show()

# Call the function with correct manufacturer column names
plot_clusters(X_2w, df_2w, "2-Wheelers EV Sales Clustering", "EV Manufacturer")
plot_clusters(X_3w, df_3w, "3-Wheelers EV Sales Clustering", "E3W Manufacturer")
plot_clusters(X_4w, df_4w, "4-Wheelers EV Sales Clustering", "EV OEM")
