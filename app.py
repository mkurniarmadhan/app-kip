from flask import Flask, render_template, request, url_for
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
from scipy.stats import zscore
from sklearn.metrics import silhouette_score
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    if file and file.filename.endswith('.csv'):
        # Load and preprocess data
        data = pd.read_csv(file)
        data.columns = ['Timestamp', 'Email Address', 'KIP', 'Responden', 'NIM', 'Jenis kelamin', 'Fakultas', 
                        'IP S1', 'IP S2', 'IP S3', 'IP S4', 'IP S5', 'IP S6', 'IP S7', 'IP S8', 'IPK', 
                        'MBKM', 'Organisasi', 'Organisasi Detail', 'Aktivitas Luar Kampus', 'Bekerja', 'Faktor Kerja']

        data['Responden'] = [f'R{i+1}' for i in range(len(data))]
        data.drop(columns=['Timestamp', 'Email Address', 'NIM', 'Jenis kelamin', 'Fakultas', 'KIP', 
                           'Organisasi Detail', 'Aktivitas Luar Kampus', 'Faktor Kerja'], inplace=True)

        data['MBKM'] = data['MBKM'].map({'Ya': 1, 'Tidak': 0})
        data['Organisasi'] = data['Organisasi'].map({'Ya': 1, 'Tidak': 0})
        data['Bekerja'] = data['Bekerja'].map({'Ya': 1, 'Tidak': 0})
        data.columns = ['Responden'] + [f'P{i}' for i in range(1, len(data.columns))]

        # Normalisasi
        data_numeric = data.drop(columns=['Responden'])
        data_normalized = data_numeric.apply(zscore).round(4)
        data_normalized['Responden'] = data['Responden']

        # Menghitung jarak
        distance_matrix = pdist(data_numeric, metric='euclidean')

        # Hierarchical Clustering
        linkage_matrix = sch.linkage(distance_matrix, method='average')

        # Silhouette Score
        range_n_clusters = list(range(2, 11))
        silhouette_scores = []
        for n_clusters in range_n_clusters:
            cluster_labels = sch.fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            score = silhouette_score(data_numeric, cluster_labels)
            silhouette_scores.append(score)

        klaster_optimal = silhouette_scores.index(max(silhouette_scores)) + 2
        cluster_labels = sch.fcluster(linkage_matrix, klaster_optimal, criterion='maxclust')
        data['Cluster'] = cluster_labels

        # Path untuk menyimpan gambar
        static_path = os.path.join('static', 'images')
        if not os.path.exists(static_path):
            os.makedirs(static_path)

        # Create dendrogram plot
        dendrogram_file = os.path.join(static_path, 'dendrogram.png')
        plt.figure(figsize=(10, 7))
        sch.dendrogram(linkage_matrix, labels=data['Responden'].values, orientation='top', distance_sort='descending', show_leaf_counts=True)
        plt.title('Dendrogram')
        plt.xlabel('Responden')
        plt.ylabel('Jarak Euclidean')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(dendrogram_file)
        plt.close()

        # Save silhouette scores plot
        silhouette_file = os.path.join(static_path, 'silhouette_scores.png')
        plt.figure(figsize=(10, 6))
        plt.plot(range_n_clusters, silhouette_scores, marker='o')
        plt.title('Silhouette Score untuk Jumlah Klaster yang Berbeda')
        plt.xlabel('Jumlah Klaster')
        plt.ylabel('Silhouette Score')
        plt.xticks(range_n_clusters)
        plt.grid(True)
        plt.savefig(silhouette_file)
        plt.close()

        # Generate URLs for images
        dendrogram_img_url = url_for('static', filename='images/dendrogram.png')
        silhouette_score_img_url = url_for('static', filename='images/silhouette_scores.png')

        return render_template('result.html', dendrogram_img=dendrogram_img_url, silhouette_score_img=silhouette_score_img_url, klaster_optimal=klaster_optimal, data=data.to_html(classes='table table-striped'))

    return 'Invalid file type'

if __name__ == '__main__':
    app.run(debug=True)
