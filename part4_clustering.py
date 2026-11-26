from pathlib import Path

import pandas as pd

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

data_path = Path(__file__).with_name("geneva_listings.csv")
df = pd.read_csv(data_path)

# Convert price to a number so it can be compared with other listing features.
df["price"] = (df["price"]
               .str.replace("$", "", regex=False)
               .str.replace(",", "", regex=False)
               .astype(float))

# Check missing values and ranges.
candidate_cols = ["accommodates", "bedrooms", "beds", "price"]
print("Missing values in candidate features")
print(df[candidate_cols].isna().sum())

print("\nSummary statistics")
print(df[candidate_cols].describe().round(2))

# Keep listings with a recorded bed count and retain price in CHF for describing the clusters later.
cluster_df = df.dropna(subset=["beds"]).copy()
cluster_df["log_price"] = np.log(cluster_df["price"])

cluster_features = ["accommodates", "beds", "log_price"]
X_scaled = StandardScaler().fit_transform(cluster_df[cluster_features])

# Compare how well listings fit their own cluster versus nearby clusters.
k_results = []
for k in range(2, 7):
    model = KMeans(n_clusters=k, 
                   n_init=20, 
                   random_state=42)
    labels = model.fit_predict(X_scaled)
    k_results.append({"k": k,
                      "Silhouette score": silhouette_score(X_scaled, 
                                                           labels)})

print("\nListings used for clustering:", len(cluster_df))
print(pd.DataFrame(k_results).round(4).to_string(index=False))

# Fit the selected two-cluster model and attach each listing's label.
final_kmeans = KMeans(n_clusters=2, n_init=20, random_state=42)
cluster_df["cluster"] = final_kmeans.fit_predict(X_scaled)

# Summarize the clusters in original units so they are easy to interpret.
# Use median prices because a few very expensive listings can raise the mean.
cluster_summary = cluster_df.groupby("cluster").agg(listings=("price", "size"),
                                                    median_guests=("accommodates", "median"),
                                                    median_beds=("beds", "median"),
                                                    median_price_chf=("price", "median"))

cluster_summary["share_pct"] = (100 * cluster_summary["listings"] / len(cluster_df))

print("\nCluster profiles")
print(cluster_summary.round(2).to_string())

cluster_names = {0: "Larger, higher-priced",
                 1: "Smaller, lower-priced"}
cluster_df["cluster_name"] = cluster_df["cluster"].map(cluster_names)
cluster_order = list(cluster_names.values())

fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

# Compare the number of listings in each cluster.
counts = cluster_df["cluster_name"].value_counts().reindex(cluster_order)
counts.plot.barh(ax=axes[0], color=["#4477AA", "#228833"])
axes[0].set_title("Listings by cluster")
axes[0].set_xlabel("Number of listings")
axes[0].set_ylabel("")

# Compare guest capacity within each cluster using percentages.
capacity_pct = pd.crosstab(cluster_df["accommodates"],
                           cluster_df["cluster_name"],
                           normalize="columns").reindex(columns=cluster_order) * 100

capacity_pct.plot.bar(ax=axes[1], color=["#4477AA", "#228833"])

axes[1].set_title("Guest capacity by cluster")
axes[1].set_xlabel("Guests accommodated")
axes[1].set_ylabel("Share within cluster (%)")
axes[1].tick_params(axis="x", labelrotation=0)
axes[1].legend(fontsize=8)

# Show prices in CHF and use a log scale to make the spread readable.
prices_by_cluster = [cluster_df.loc[cluster_df["cluster_name"] == name, 
                                    "price"]
                     for name in cluster_order]
axes[2].boxplot(prices_by_cluster)
axes[2].set_xticks([1, 2])
axes[2].set_xticklabels(["Larger", "Smaller"])
axes[2].set_yscale("log")
axes[2].set_title("Price distribution by cluster")
axes[2].set_ylabel("Price (CHF, log scale)")

figure_path = Path(__file__).with_name("part4_cluster_profiles.png")
fig.savefig(figure_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print("\nSaved cluster figure:", figure_path)