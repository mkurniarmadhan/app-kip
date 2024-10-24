import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
from scipy.stats import zscore

app = Flask(__name__)

# Set secret key for flash messages
app.secret_key = 'appkip'

# Folder untuk menyimpan file statis hasil proses
UPLOAD_FOLDER = 'static/hasil'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Fungsi untuk melakukan klasterisasi dan menghasilkan visualisasi
def process_file(filepath):
    try:

        # Load data
        data = pd.read_csv(filepath)

        # Ganti nama kolom untuk kemudahan pemrosesan
        data.columns = ['Timestamp', 'Email Address', 'KIP', 'Responden', 'NIM', 'Jenis kelamin', 'Fakultas', 
                        'IP S1', 'IP S2', 'IP S3', 'IP S4', 'IP S5', 'IP S6', 'IP S7', 'IP S8', 'IPK', 
                        'MBKM', 'Organisasi', 'Organisasi Detail', 'Aktivitas Luar Kampus', 'Bekerja', 'Faktor Kerja']

        data['Responden'] = [f'R{i+1}' for i in range(len(data))]

        # Hapus kolom yang tidak diperlukan
        data.drop(columns=['Timestamp', 'Email Address', 'NIM', 'Jenis kelamin', 'Fakultas', 'KIP', 
                           'Organisasi Detail', 'Aktivitas Luar Kampus', 'Faktor Kerja'], inplace=True)


        # Konversi nilai biner 'Ya'/'Tidak' ke angka
        data['MBKM'] = data['MBKM'].map({'Ya': 1, 'Tidak': 0})
        data['Organisasi'] = data['Organisasi'].map({'Ya': 1, 'Tidak': 0})
        data['Bekerja'] = data['Bekerja'].map({'Ya': 1, 'Tidak': 0})
    

        dataset = data.copy()

        # Normalisasi Z-score
        data_numeric = data.drop(columns=['Responden'])
        data_normalized = data_numeric.apply(zscore).round(4)
        
        data_normalized.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'data_normalized.csv'))


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
        range_n_clusters = list(range(3, 11))
        silhouette_scores = []
        for n_clusters in range_n_clusters:
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            score = silhouette_score(data_normalized, cluster_labels)
            silhouette_scores.append(score.round(4))

        silhouette_df = pd.DataFrame({
            'Jumlah Klaster': range_n_clusters,
            'Nilai Silhouette Score': silhouette_scores
        })

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
        klaster_index = silhouette_scores.index(max(silhouette_scores))
        klaster_optimal = range_n_clusters[klaster_index]

        # Tampilkan klaster optimal
        print(f"Jumlah klaster optimal: {klaster_optimal}")
        print(f"Silhouette Score tertinggi: {max(silhouette_scores).round(4)}")

        # silhouette_df.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'silhouette_scores.csv'), index=False)

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
            anggota = cluster_data['Responden'].tolist()

            if mbkm_percent == 0 and organisasi_percent ==0 and bekerja_percent ==0:
                interpretasi_hasil =     "Mahasiswa dalam klaster ini cenderung fokus pada studi akademik tanpa terlibat dalam kegiatan ekstrakurikuler."
            elif mbkm_percent == 1 and organisasi_percent ==1 and bekerja_percent ==1:
                interpretasi_hasil = "Mahasiswa dalam klaster ini berhasil menjaga performa akademik yang sangat baik meskipun terlibat dalam banyak kegiatan non-akademik."
            else:
                interpretasi_hasil = "Mahasiswa dalam klaster ini menunjukkan performa akademik yang lebih rendah, meskipun terlibat dalam MBKM dan organisasi."

            interpretasi.append({
                'cluster': cluster,
                'jumlah_data': jumlah_data,
                'rata_rata_ipk': round(rata_rata_ipk, 2),
                'mbkm_percent': round(mbkm_percent, 2),
                'bekerja_percent': round(bekerja_percent, 2),
                'organisasi_percent': round(organisasi_percent, 2),
                'anggota': anggota,
                'interpretasi':interpretasi_hasil
            })

      
        return {
            'dendrogram': dendrogram_path,
            'silhouette': silhouette_path,
            'perubahan_ipk': perubahan_ipk_path,
            'cluster_count': klaster_optimal,
            'interpretasi': interpretasi,
            'dataset':dataset.to_html(classes='table table-striped',),        
            'data':data.to_html(classes='table table-striped',),        
            'data_normalized':data_normalized.to_html(classes='table table-striped',),        
            'silhouette_df':silhouette_df.to_html(classes='table table-striped') 
        }

    except Exception as e:
        # Tangkap semua error dan kembalikan pesan error
        return {'error': str(e)}

# Route untuk halaman utama
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('Tidak ada file yang di upload','danger')
            return redirect(request.url)
        if file and file.filename.endswith('.csv'):  # Pastikan file CSV
            # filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'dataset.csv')
            file.save(filepath)
            # Proses file dan dapatkan hasil
            hasil = process_file(filepath)
            if 'error' in hasil:
                flash("Data tidak valid",'danger')
                flash(f"Error: {hasil['error']}",'warning')
                return redirect(request.url)
            return render_template('hasil.html', hasil=hasil)
        else:
            flash('Hanya menerima upload csv','danger')
            return redirect(request.url)
    return render_template('index.html')

# Jalankan aplikasi
if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
