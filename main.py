import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score

from scipy.stats import zscore

# Langkah 1: Import Data dan Persiapan
data = pd.read_csv('data-responden.csv')

# Ganti nama kolom
data.columns = ['Timestamp', 'Email Address', 'KIP', 'Responden', 'NIM', 'Jenis kelamin', 'Fakultas', 
                'IP S1', 'IP S2', 'IP S3', 'IP S4', 'IP S5', 'IP S6', 'IP S7', 'IP S8', 'IPK', 
                'MBKM', 'Organisasi', 'Organisasi Detail', 'Aktivitas Luar Kampus', 'Bekerja', 'Faktor Kerja']

data['Responden'] = [f'R{i+1}' for i in range(len(data))]

# Menghapus kolom yang tidak diperlukan
data.drop(columns=['Timestamp', 'Email Address', 'NIM', 'Jenis kelamin', 'Fakultas', 'KIP', 
                   'Organisasi Detail', 'Aktivitas Luar Kampus', 'Faktor Kerja'], inplace=True)

# Ganti nama kolom

# Mengubah nilai biner menjadi angka
data['MBKM'] = data['MBKM'].map({'Ya': 1, 'Tidak': 0})
data['Organisasi'] = data['Organisasi'].map({'Ya': 1, 'Tidak': 0})
data['Bekerja'] = data['Bekerja'].map({'Ya': 1, 'Tidak': 0})

data.columns = ['Responden'] + [f'P{i}' for i in range(1, len(data.columns))]

# Menyimpan DataFrame dengan kolom yang telah diganti ke file baru
data.to_csv('data-responden-updated.csv', index=False)


# ambil data respodnen
data = pd.read_csv('data-responden-updated.csv')

# Menghapus kolom non-numerik sebelum normalisasi
data_numeric = data.drop(columns=['Responden'])

# Melakukan normalisasi Z-score
data_normalized = data_numeric.apply(zscore).round(4)

# Menyertakan kolom 'Responden' kembali
data_normalized['Responden'] = data['Responden']

# Menyimpan DataFrame yang telah dinormalisasi ke file baru
data_normalized.to_csv('data-responden-normalized.csv', index=False)




# HITUNG JARAK
from scipy.spatial.distance import pdist, squareform

data = pd.read_csv('data-responden-normalized.csv')

# Menghapus kolom non-numerik sebelum menghitung jarak
data_numeric = data_normalized.drop(columns=['Responden'])

# Menghitung jarak Euclidean antar semua pasangan titik
distance_matrix = pdist(data_numeric, metric='euclidean')

# Mengubah matriks jarak menjadi bentuk matriks simetris
distance_matrix_symmetric = squareform(distance_matrix)

# Mengubah matriks jarak menjadi DataFrame untuk visualisasi
distance_matrix_df = pd.DataFrame(distance_matrix_symmetric, index=data_normalized['Responden'], columns=data_normalized['Responden'])

# Menyimpan matriks jarak ke file CSV
distance_matrix_df.to_csv('distance_matrix.csv')


import scipy.cluster.hierarchy as sch
# Melakukan klasterisasi hierarkis
linkage_matrix = sch.linkage(distance_matrix, method='average')

# Membuat dendrogram
plt.figure(figsize=(10, 7))
sch.dendrogram(linkage_matrix, labels=data_normalized['Responden'].values, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Dendrogram')
plt.xlabel('Responden')
plt.ylabel('Jarak Euclidean')
plt.xticks(rotation=90)
plt.tight_layout()
# plt.show()




# Menentukan rentang jumlah klaster yang akan diuji
range_n_clusters = list(range(2, 11))  # Uji dari 2 hingga 10 klaster

silhouette_scores = []

for n_clusters in range_n_clusters:
    # Melakukan linkage untuk klasterisasi hierarkis
    linkage_matrix = linkage(distance_matrix, method='average')
    
    # Menentukan label klaster untuk jumlah klaster yang ditentukan
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    
    # Hitung Silhouette Score
    score = silhouette_score(data_numeric, cluster_labels)
    silhouette_scores.append(score)
    print(f'Jumlah Klaster: {n_clusters}, Silhouette Score: {score:.4f}')


# Plot Silhouette Score untuk setiap jumlah klaster
plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, silhouette_scores, marker='o')
plt.title('Silhouette Score untuk Jumlah Klaster yang Berbeda (Hierarchical Clustering)')
plt.xlabel('Jumlah Klaster')
plt.ylabel('Silhouette Score')
plt.xticks(range_n_clusters)
plt.grid(True)

# plt.show()

# # Tentukan jumlah klaster optimal dari silhouette_scores
klaster_optimal = silhouette_scores.index(max(silhouette_scores)) + 2
print(f'Jumlah klaster optimal: {klaster_optimal}')



# Langkah 7: Membuat Klaster dengan Jumlah Klaster Optimal
cluster_labels = fcluster(linkage_matrix, klaster_optimal, criterion='maxclust')
data['Cluster'] = cluster_labels



# Interpretasi hasil klasterisasi dan menambahkan ke dalam tabel terpisah
interpretasi = {
    "Cluster": [],
    "Rata-rata IP S1": [],
    "Rata-rata IPK": [],
    "Persentase MBKM": [],
    "Persentase Organisasi": [],
    "Persentase Bekerja": [],
    "Anggota": []
}





# Interpretasi hasil klasterisasi
print("\nInterpretasi Hasil Klasterisasi:")

for cluster in range(1, klaster_optimal + 1):
    cluster_data = data[data['Cluster'] == cluster]


    interpretasi["Cluster"].append(cluster)
    interpretasi["Rata-rata IP S1"].append(cluster_data['P1'].mean())
    interpretasi["Rata-rata IPK"].append(cluster_data['P9'].mean())
    interpretasi["Persentase MBKM"].append(cluster_data['P10'].mean() * 100)
    interpretasi["Persentase Organisasi"].append(cluster_data['P11'].mean() * 100)
    interpretasi["Persentase Bekerja"].append(cluster_data['P12'].mean() * 100)
    interpretasi["Anggota"].append(', '.join(cluster_data['Responden'].tolist()))





interpretasi_df = pd.DataFrame(interpretasi)
print(interpretasi_df)



# Visualisasi pola perubahan IPK
plt.figure(figsize=(12, 6))
semesters = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8','P9','P10','P11','P12']
for cluster in data['Cluster'].unique():
    cluster_data = data[data['Cluster'] == cluster]
    mean_ips = cluster_data[semesters].mean()
    sns.lineplot(x=semesters, y=mean_ips, label=f'Cluster {cluster}')

# Visualisasi IPK keseluruhan untuk setiap klaster
mean_ipk = data.groupby('Cluster')['P9'].mean().reset_index()
for i, row in mean_ipk.iterrows():
    plt.scatter(semesters[-1], row['P9'], s=100, label=f'IPK Keseluruhan Cluster {row["Cluster"]}', color='red', marker='o')

plt.title('Pola Perubahan IPK Mahasiswa dalam Setiap Klaster')
plt.xlabel('Semester')
plt.ylabel('IPK Rata-Rata')
plt.legend()
plt.show()