import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from sklearn.cluster import AgglomerativeClustering

# Membaca data
data = pd.read_csv("data-responden.csv")
features = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11", "P12"]

# Normalisasi data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[features])

# Hierarchical clustering dengan Average Linkage
linkage_matrix = linkage(data_scaled, method="ward", metric="euclidean")

# Menentukan jumlah klaster
num_clusters = 9
clusters = AgglomerativeClustering(
    n_clusters=num_clusters, linkage="ward", metric="euclidean"
)
# Menambahkan kolom cluster ke data
data["Cluster"] = clusters.fit_predict(data_scaled)

# Menampilkan jumlah anggota dalam setiap cluster
print("\nJumlah anggota dalam setiap cluster:")
print(data["Cluster"].value_counts())

# Interpretasi klaster
interpretasi = []
for cluster in sorted(data["Cluster"].unique()):
    cluster_data = data[data["Cluster"] == cluster]

    # Informasi dasar cluster
    jumlah_anggota = cluster_data.shape[0]
    anggota = cluster_data["Responden"].tolist()

    # Menghitung pola rata-rata
    avg_ipk = cluster_data[[f"P{i}" for i in range(1, 9)]].mean(axis=1).mean()
    mbkm_percent = cluster_data["P10"].mean() * 100
    bekerja_percent = cluster_data["P11"].mean() * 100
    organisasi_percent = cluster_data["P12"].mean() * 100

    # Interpretasi pola IPK
    if avg_ipk >= 3.5:
        pola_ipk = "Stabil atau meningkat dengan IPK kumulatif yang baik."
    else:
        pola_ipk = "Mungkin ada penurunan atau ketidakstabilan dalam IPK."

    # Interpretasi faktor pendukung
    faktor = []
    if mbkm_percent > 50:
        faktor.append("MBKM memberikan pengaruh besar.")
    if bekerja_percent > 50:
        faktor.append("Bekerja mungkin menjadi faktor signifikan.")
    if organisasi_percent > 50:
        faktor.append("Organisasi memiliki dampak signifikan.")

    # Menyusun interpretasi untuk file
    interpretasi_hasil = f"""
Cluster {cluster}:
Jumlah anggota: {jumlah_anggota}
Anggota: {', '.join(map(str, anggota))}
Rata-rata IPK: {avg_ipk:.2f}
Pola IPK: {pola_ipk}
"""
    if faktor:
        interpretasi_hasil += "Faktor Pendukung:\n- " + "\n- ".join(faktor) + "\n"
    else:
        interpretasi_hasil += (
            "Tidak ada faktor pendukung signifikan yang teridentifikasi.\n"
        )

    # Menyimpan interpretasi ke daftar
    interpretasi.append(interpretasi_hasil)
    print(interpretasi_hasil)

# Menyimpan interpretasi ke file "catatan.txt"
with open("catatan.txt", "w") as file:
    file.write("\n".join(interpretasi))

print("\nInterpretasi klaster telah disimpan ke file 'catatan.txt'.")
