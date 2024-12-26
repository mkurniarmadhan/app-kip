import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Baca data
data = pd.read_csv("data.csv")  # Ganti dengan lokasi file Anda
features = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11", "P12"]

# Normalisasi data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[features])

# Hierarchical clustering dengan Average Linkage
# average atau ward
linkage_matrix = linkage(data_scaled, method="ward", metric="euclidean")

# Visualisasi dendrogram
plt.figure(figsize=(10, 7))
dendrogram(
    linkage_matrix, labels=data["Responden"].values, leaf_rotation=90, leaf_font_size=10
)
plt.title("Dendrogram untuk Data Mahasiswa")
plt.xlabel("Responden")
plt.ylabel("Jarak")
plt.show()


num_clusters = 3
clusters = fcluster(linkage_matrix, num_clusters, criterion="maxclust")

# Tambahkan hasil clustering ke dalam DataFrame
data["Cluster"] = clusters
print("\nJumlah anggota dalam setiap cluster:")
print(data["Cluster"].value_counts())


# Tampilkan anggota tiap cluster
for cluster_id in range(1, num_clusters + 1):
    print(f"\nCluster {cluster_id}: ")
    print(data[data["Cluster"] == cluster_id][["Responden", "P9", "P10", "P11", "P12"]])
