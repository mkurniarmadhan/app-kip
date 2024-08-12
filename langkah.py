import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as dist
import matplotlib.pyplot as plt

# 1. Persiapkan Data
# Contoh data; Anda harus mengganti ini dengan data Anda sendiri.
data = {
    'IPK S1': [3.2, 3.5, 3.1, 3.9, 3.6],
    'IPK S2': [3.4, 3.6, 3.2, 4.0, 3.7],
    'IPK S3': [3.5, 3.7, 3.3, 4.1, 3.8],
    'IPK S4': [3.6, 3.8, 3.4, 4.2, 3.9],
    'IPK S5': [3.7, 3.9, 3.5, 4.3, 4.0],
    'IPK S6': [3.8, 4.0, 3.6, 4.4, 4.1]
}
df = pd.DataFrame(data)


print(data)

# 2. Hitung Matriks Jarak
# Hitung jarak Euclidean antara objek
dist_matrix = dist.pdist(df, metric='euclidean')


print(dist_matrix)
dist_matrix_square = dist.squareform(dist_matrix)

# 3. Membentuk Hierarchical Clustering dengan Average Linkage
# Buat dendogram
linkage_matrix = sch.linkage(dist_matrix, method='average')

# 4. Visualisasi Dendogram
plt.figure(figsize=(10, 7))
sch.dendrogram(linkage_matrix, labels=df.index)
plt.title('Dendogram untuk Hierarchical Clustering dengan Average Linkage')
plt.xlabel('Index')
plt.ylabel('Jarak')
plt.show()

# 5. Menentukan Jumlah Klaster
# Tentukan jumlah klaster dengan memotong dendogram pada level tertentu
num_clusters = 3  # Anda dapat memilih jumlah klaster yang diinginkan
clusters = sch.fcluster(linkage_matrix, num_clusters, criterion='maxclust')

# 6. Interpretasi Hasil
df['Cluster'] = clusters
print(df)



# sebelum di edit

# # Langkah 2: Normalisasi Data
# fitur = ['IP S1', 'IP S2', 'IP S3', 'IP S4', 'IP S5', 'IP S6', 'IP S7', 'IP S8', 'IPK', 'MBKM', 'Organisasi', 'Bekerja']
# X = data[fitur]

# X = X.apply(pd.to_numeric, errors='coerce')


# scaler = StandardScaler()
# X_scaler = scaler.fit_transform(X)


# print(data[X_scaler].head())




# Langkah 3: Perhitungan Jarak
# matrix_jarak = pdist(X_scaler, metric='euclidean')




# # Langkah 4: Hierarchical Clustering
# Z = linkage(matrix_jarak, method='average')

# # Langkah 5: Visualisasi Dendrogram
# plt.figure(figsize=(10, 7))
# dendrogram(Z, labels=data['Responden'].values)
# plt.title('Dendrogram untuk Hierarchical Clustering')
# plt.xlabel('Indeks Mahasiswa')
# plt.ylabel('Jarak Euclidean')
# plt.show()

# # Langkah 6: Menentukan Jumlah Klaster Optimal
# max_klaster = 4
# silhouette_scores = []
# for k in range(2, max_klaster + 1):
#     cluster_labels = fcluster(Z, k, criterion='maxclust')
#     silhouette_avg = silhouette_score(X_scaler, cluster_labels)
#     silhouette_scores.append(silhouette_avg)

# plt.figure(figsize=(10, 7))
# plt.plot(range(2, max_klaster + 1), silhouette_scores, marker='o')
# plt.title('Silhouette Scores untuk Menentukan Jumlah Klaster Optimal')
# plt.xlabel('Jumlah Klaster')
# plt.ylabel('Silhouette Score')
# plt.show()

# # Tentukan jumlah klaster optimal dari silhouette_scores
# klaster_optimal = silhouette_scores.index(max(silhouette_scores)) + 2
# print(f'Jumlah klaster optimal: {klaster_optimal}')

# # Langkah 7: Membuat Klaster dengan Jumlah Klaster Optimal
# cluster_labels = fcluster(Z, klaster_optimal, criterion='maxclust')
# data['Cluster'] = cluster_labels



# # Langkah 8: Visualisasi Pola Perubahan IPK
# plt.figure(figsize=(12, 6))
# semesters = ['IP S1', 'IP S2', 'IP S3', 'IP S4', 'IP S5', 'IP S6', 'IP S7', 'IP S8']

# for cluster in data['Cluster'].unique():
#     cluster_data = data[data['Cluster'] == cluster]
#     mean_ips = cluster_data[semesters].mean()
#     sns.lineplot(x=semesters, y=mean_ips, label=f'Cluster {cluster}')

# mean_ipk = data.groupby('Cluster')['IPK'].mean().reset_index()
# for i, row in mean_ipk.iterrows():
#     plt.scatter(semesters[-1], row['IPK'], s=100, label=f'IPK Keseluruhan Cluster {row["Cluster"]}', color='red', marker='o')

# plt.title('Pola Perubahan IPK Mahasiswa dalam Setiap Klaster')
# plt.xlabel('Semester')
# plt.ylabel('IPK Rata-Rata')
# plt.legend()
# plt.show()

# # Langkah 9: Interpretasi Hasil
# cluster_groups = data.groupby('Cluster').mean(numeric_only=True)
# interpretasi = "\nInterpretasi Hasil:\n"
# for cluster_id, group in cluster_groups.iterrows():
#     interpretasi += f"\nCluster {cluster_id}:\n"
#     interpretasi += f" - IPK Semester 1: {group['IP S1']:.2f}\n"
#     interpretasi += f" - IPK Semester 2: {group['IP S2']:.2f}\n"
#     interpretasi += f" - IPK Semester 3: {group['IP S3']:.2f}\n"
#     interpretasi += f" - IPK Semester 4: {group['IP S4']:.2f}\n"
#     interpretasi += f" - IPK Semester 5: {group['IP S5']:.2f}\n"
#     interpretasi += f" - IPK Semester 6: {group['IP S6']:.2f}\n"
#     interpretasi += f" - IPK Semester 7: {group['IP S7']:.2f}\n"
#     interpretasi += f" - IPK Semester 8: {group['IP S8']:.2f}\n"
#     interpretasi += f" - IPK Saat Ini: {group['IPK']:.2f}\n"
#     interpretasi += f" - Persentase MBKM: {group['MBKM']*100:.0f}%\n"
#     interpretasi += f" - Persentase Organisasi: {group['Organisasi']*100:.0f}%\n"
#     interpretasi += f" - Persentase Bekerja sambil Kuliah: {group['Bekerja']*100:.0f}%\n"

#     if group['MBKM'] == 0 and group['Organisasi'] == 0 and group['Bekerja'] == 0:
#         interpretasi += "   Interpretasi: Mahasiswa dalam klaster ini cenderung fokus pada studi akademik tanpa terlibat dalam kegiatan ekstrakurikuler.\n"
#     elif group['MBKM'] == 1 and group['Organisasi'] == 1 and group['Bekerja'] == 1:
#         interpretasi += "   Interpretasi: Mahasiswa dalam klaster ini berhasil menjaga performa akademik yang sangat baik meskipun terlibat dalam banyak kegiatan non-akademik.\n"
#     else:
#         interpretasi += "   Interpretasi: Mahasiswa dalam klaster ini menunjukkan performa akademik yang lebih rendah, meskipun terlibat dalam MBKM dan organisasi.\n"

# print(interpretasi)
