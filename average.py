import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Membaca data
data = pd.read_csv("data-responden.csv")

dataset = data.copy()
# Normalisasi Data dengan MinMaxScaler
scaler = MinMaxScaler()

normalisasi = [
    "P1",
    "P2",
    "P3",
    "P4",
    "P5",
    "P6",
    "P7",
    "P8",
    "P9",
    "P10",
    "P11",
    "P12",
]
# normatisi
data_scaled = scaler.fit_transform(data[normalisasi])
data_scaled_df = pd.DataFrame(data_scaled, columns=normalisasi)
data_scaled_df.to_csv("data_normalized.csv")


# Melakukan linkage untuk clustering dengan metode Ward
Z = linkage(data_scaled, method="ward", metric="euclidean")

# Visualisasi Dendrogram
plt.figure(figsize=(10, 7))
dendrogram(Z, labels=data["Responden"].tolist(), leaf_rotation=90, leaf_font_size=10)
plt.xlabel("Responden")
plt.ylabel("Jarak (Distance)")
plt.tight_layout()
plt.show()

# Tentukan range klaster yang akan diuji
cluster_range = range(2, 6)
silhouette_scores = []

# Hitung Silhouette Score untuk setiap jumlah klaster
for n_clusters in cluster_range:
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters, linkage="ward", metric="euclidean"
    )
    cluster_labels = clustering.fit_predict(data_scaled)
    score = silhouette_score(data_scaled, cluster_labels)
    print(f"n_clusters :{n_clusters} niali: {score}")
    silhouette_scores.append(score)


# Plot Silhouette Scores
plt.figure(figsize=(8, 5))
plt.plot(cluster_range, silhouette_scores, marker="o")
plt.title("Silhouette Scores untuk Berbagai Jumlah Klaster")
plt.xlabel("Jumlah Klaster")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()

# Tentukan jumlah klaster optimal
optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
print(f"Jumlah klaster optimal berdasarkan Silhouette Score: {optimal_clusters}")

# Clustering dengan jumlah klaster optimal
clustering = AgglomerativeClustering(
    n_clusters=optimal_clusters, linkage="ward", metric="euclidean"
)
data["Cluster"] = clustering.fit_predict(data_scaled)


cluster_mean_ipk = {}
# Menampilkan Rata-rata dan Interpretasi untuk Setiap Klaster
for cluster in sorted(data["Cluster"].unique()):

    cluster_data = data[data["Cluster"] == cluster]
    mean_ipk_per_semester = cluster_data[
        ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]
    ].mean()
    cluster_mean_ipk[cluster] = mean_ipk_per_semester
    jumlah_data = cluster_data.shape[0]
    rata_rata_ipk = cluster_data["P9"].mean()
    mbkm_percent = (cluster_data["P10"].mean()) * 100
    bekerja_percent = (cluster_data["P11"].mean()) * 100
    organisasi_percent = (cluster_data["P12"].mean()) * 100
    anggota = cluster_data["Responden"].tolist()

    print(f"\nKlaster {cluster}:")
    print(f"Jumlah Mahasiswa: {jumlah_data}")
    print(f"Rata-rata IPK Kumulatif (P9): {rata_rata_ipk:.2f}")
    print(f"Persentase Keikutsertaan MBKM: {mbkm_percent:.2f}%")
    print(f"Persentase Bekerja: {bekerja_percent:.2f}%")
    print(f"Persentase Keterlibatan Organisasi: {organisasi_percent:.2f}%")
    print(f"Anggota Klaster: {', '.join(anggota)}")

    # Interpretasi Pola
    if rata_rata_ipk >= 3.5:
        print("Pola IPK: Stabil atau meningkat dengan IPK Kumulatif yang baik.")
    else:
        print("Pola IPK: Mungkin ada penurunan atau ketidakstabilan dalam IPK.")

    if mbkm_percent > 50:
        print(
            "Faktor MBKM: Mahasiswa dalam klaster ini memiliki banyak yang ikut MBKM."
        )
    if bekerja_percent > 50:
        print(
            "Faktor Pekerjaan: Mahasiswa dalam klaster ini banyak yang bekerja sambil kuliah."
        )
    if organisasi_percent > 50:
        print("Faktor Organisasi: Mahasiswa dalam klaster ini aktif dalam organisasi.")


# Visualisasi perubahan IPK rata-rata per klaster dalam satu grafik
plt.figure(figsize=(10, 6))

# Plotkan pola perubahan IPK untuk setiap klaster
for cluster, mean_ipk in cluster_mean_ipk.items():
    plt.plot(
        [
            "Semester 1",
            "Semester 2",
            "Semester 3",
            "Semester 4",
            "Semester 5",
            "Semester 6",
            "Semester 7",
            "Semester 8",
        ],
        mean_ipk.values,
        marker="o",
        label=f"Klaster {cluster}",
    )

plt.title("Pola Perubahan IPK Rata-rata per Klaster")
plt.xlabel("Semester")
plt.ylabel("Rata-rata IPK")
plt.legend(title="Klaster")
plt.grid(True)
plt.tight_layout()
plt.show()

# Simpan hasil ke file CSV
data.to_csv("data_with_clusters.csv", index=False)
print("\nHasil clustering disimpan ke 'data_with_clusters.csv'")
