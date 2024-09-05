import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
from scipy.stats import zscore

app = Flask(__name__)

# Folder untuk menyimpan file statis hasil proses
UPLOAD_FOLDER = 'static/hasil'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Fungsi untuk melakukan klasterisasi dan menghasilkan visualisasi
def process_file(filepath):
    # Load data
    data = pd.read_csv(filepath)

    # Ganti nama kolom untuk kemudahan pemrosesan
    data.columns = ['Timestamp', 'Email Address', 'KIP', 'Responden', 'NIM', 'Jenis kelamin', 'Fakultas', 
                    'IP S1', 'IP S2', 'IP S3', 'IP S4', 'IP S5', 'IP S6', 'IP S7', 'IP S8', 'IPK', 
                    'MBKM', 'Organisasi', 'Organisasi Detail', 'Aktivitas Luar Kampus', 'Bekerja', 'Faktor Kerja']

    # Hapus kolom yang tidak diperlukan
    data.drop(columns=['Timestamp', 'Email Address', 'NIM', 'Jenis kelamin', 'Fakultas', 'KIP', 
                       'Organisasi Detail', 'Aktivitas Luar Kampus', 'Faktor Kerja'], inplace=True)

    # Konversi nilai biner 'Ya'/'Tidak' ke angka
    data['MBKM'] = data['MBKM'].map({'Ya': 1, 'Tidak': 0})
    data['Organisasi'] = data['Organisasi'].map({'Ya': 1, 'Tidak': 0})
    data['Bekerja'] = data['Bekerja'].map({'Ya': 1, 'Tidak': 0})

    # Normalisasi Z-score
    data_numeric = data.drop(columns=['Responden'])
    data_normalized = data_numeric.apply(zscore)

    # Hitung matriks jarak Euclidean
    distance_matrix = pdist(data_normalized, metric='euclidean')

    # Hierarchical Clustering dan Dendrogram
    linkage_matrix = linkage(distance_matrix, method='average')
    
    # Simpan dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix, labels=data['Responden'].values, orientation='top', distance_sort='descending', show_leaf_counts=True)
    dendrogram_path = os.path.join(app.config['UPLOAD_FOLDER'], 'dendrogram.png')
    plt.savefig(dendrogram_path)
    plt.close()

    # Silhouette Score untuk berbagai klaster
    range_n_clusters = list(range(2, 11))
    silhouette_scores = []
    for n_clusters in range_n_clusters:
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        score = silhouette_score(data_numeric, cluster_labels)
        silhouette_scores.append(score)

    # Plot Silhouette Score
    plt.figure(figsize=(10, 6))
    plt.plot(range_n_clusters, silhouette_scores, marker='o')
    plt.title('Silhouette Score untuk Berbagai Jumlah Klaster')
    plt.xlabel('Jumlah Klaster')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    silhouette_path = os.path.join(app.config['UPLOAD_FOLDER'], 'silhouette_scores.png')
    plt.savefig(silhouette_path)
    plt.close()

    # Tentukan jumlah klaster optimal
    klaster_optimal = silhouette_scores.index(max(silhouette_scores)) + 2

    # Buat klaster dengan jumlah klaster optimal
    cluster_labels = fcluster(linkage_matrix, klaster_optimal, criterion='maxclust')
    data['Cluster'] = cluster_labels

    # Visualisasi pola perubahan IPK setiap klaster
    plt.figure(figsize=(12, 6))
    semesters = ['IP S1', 'IP S2', 'IP S3', 'IP S4', 'IP S5', 'IP S6', 'IP S7', 'IP S8']
    for cluster in data['Cluster'].unique():
        cluster_data = data[data['Cluster'] == cluster]
        mean_ips = cluster_data[semesters].mean()
        sns.lineplot(x=semesters, y=mean_ips, label=f'Cluster {cluster}')
    
    plt.title('Pola Perubahan IPK Mahasiswa dalam Setiap Klaster')
    plt.xlabel('Semester')
    plt.ylabel('IPK Rata-Rata')
    perubahan_ipk_path = os.path.join(app.config['UPLOAD_FOLDER'], 'perubahan_ipk.png')
    plt.savefig(perubahan_ipk_path)
    plt.close()

    # Interpretasi Hasil Klasterisasi
    interpretasi = []
    for cluster in sorted(data['Cluster'].unique()):
        cluster_data = data[data['Cluster'] == cluster]
        jumlah_data = cluster_data.shape[0]
        rata_rata_ipk = cluster_data['IPK'].mean()
        mbkm_percent = (cluster_data['MBKM'].mean()) * 100
        bekerja_percent = (cluster_data['Bekerja'].mean()) * 100
        organisasi_percent = (cluster_data['Organisasi'].mean()) * 100
        
        interpretasi.append({
            'cluster': cluster,
            'jumlah_data': jumlah_data,
            'rata_rata_ipk': round(rata_rata_ipk, 2),
            'mbkm_percent': round(mbkm_percent, 2),
            'bekerja_percent': round(bekerja_percent, 2),
            'organisasi_percent': round(organisasi_percent, 2)
        })

    return {
        'dendrogram': dendrogram_path,
        'silhouette': silhouette_path,
        'perubahan_ipk': perubahan_ipk_path,
        'cluster_count': klaster_optimal,
        'interpretasi': interpretasi
    }

# Route untuk halaman utama
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Proses file dan dapatkan hasil
            hasil = process_file(filepath)
            return render_template('hasil.html', hasil=hasil)
    return render_template('index.html')

# Jalankan aplikasi
if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
