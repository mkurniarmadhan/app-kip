import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.stats import zscore

app = Flask(__name__)

app.secret_key = "appkip"

# Folder untuk menyimpan file statis hasil proses
UPLOAD_FOLDER = "static/hasil"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# Fungsi untuk melakukan klasterisasi dan menghasilkan visualisasi
def process_file(filepath):
    try:
        data = pd.read_csv(filepath)

        features = [
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

        dataset = data.copy()

        # Normalisasi Z-score
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data[features])

        data_scaled_df = pd.DataFrame(data_scaled, columns=features)
        data_scaled_df.to_csv(
            os.path.join(app.config["UPLOAD_FOLDER"], "data_normalized.csv")
        )

        # Hierarchical Clustering dan Dendrogram
        linkage_matrix = linkage(data_scaled, method="ward", metric="euclidean")

        # Simpan dendrogram
        plt.figure(figsize=(10, 7))
        dendrogram(
            linkage_matrix,
            labels=data["Responden"].values,
            orientation="top",
            distance_sort="descending",
            show_leaf_counts=True,
        )
        dendrogram_path = os.path.join(app.config["UPLOAD_FOLDER"], "dendrogram.png")
        plt.savefig(dendrogram_path)
        plt.close()

        #  jumlah klaster tetap (3 klaster)
        num_clusters = 3

        # Buat klaster dengan jumlah klaster tetap
        cluster_labels = fcluster(linkage_matrix, num_clusters, criterion="maxclust")
        data["Cluster"] = cluster_labels

        # Hitung Silhouette Score
        silhouette_avg = silhouette_score(data_scaled, cluster_labels)

        # Hitung Silhouette Score untuk setiap responden
        silhouette_values = silhouette_samples(data_scaled, cluster_labels)

        # Visualisasi Plot Silhouette
        plt.figure(figsize=(10, 7))
        sns.histplot(silhouette_values, kde=True, color="blue")
        plt.axvline(
            x=silhouette_avg,
            color="red",
            linestyle="--",
            label=f"Average Silhouette Score: {silhouette_avg:.2f}",
        )
        plt.title("Distribusi Silhouette Score per Sampel")
        plt.xlabel("Silhouette Score")
        plt.ylabel("Frekuensi")
        silhouette_plot_path = os.path.join(
            app.config["UPLOAD_FOLDER"], "silhouette_score_plot.png"
        )
        plt.savefig(silhouette_plot_path)
        plt.close()

        # Visualisasi pola perubahan IPK setiap klaster
        plt.figure(figsize=(12, 6))
        semesters = [
            "P1",
            "P2",
            "P3",
            "P4",
            "P5",
            "P6",
            "P7",
            "P8",
        ]
        for cluster in data["Cluster"].unique():
            cluster_data = data[data["Cluster"] == cluster]
            mean_ips = cluster_data[semesters].mean()
            sns.lineplot(x=semesters, y=mean_ips, label=f"Cluster {cluster}")

        plt.title("Pola Perubahan IPK Mahasiswa dalam Setiap Klaster")
        plt.xlabel("Semester")
        plt.ylabel("IPK Rata-Rata")
        perubahan_ipk_path = os.path.join(
            app.config["UPLOAD_FOLDER"], "perubahan_ipk.png"
        )
        plt.savefig(perubahan_ipk_path)
        plt.close()

        # Interpretasi Hasil Klasterisasi
        interpretasi = []
        for cluster in sorted(data["Cluster"].unique()):
            cluster_data = data[data["Cluster"] == cluster]
            jumlah_data = cluster_data.shape[0]
            rata_rata_ipk = cluster_data["P9"].mean()
            mbkm_percent = (cluster_data["P10"].mean()) * 100
            bekerja_percent = (cluster_data["P11"].mean()) * 100
            organisasi_percent = (cluster_data["P12"].mean()) * 100
            anggota = cluster_data["Responden"].tolist()

            if mbkm_percent == 0 and organisasi_percent == 0 and bekerja_percent == 0:
                interpretasi_hasil = "Mahasiswa dalam klaster ini cenderung fokus pada studi akademik tanpa terlibat dalam kegiatan ekstrakurikuler."
            elif mbkm_percent == 1 and organisasi_percent == 1 and bekerja_percent == 1:
                interpretasi_hasil = "Mahasiswa dalam klaster ini berhasil menjaga performa akademik yang sangat baik meskipun terlibat dalam banyak kegiatan non-akademik."
            else:
                interpretasi_hasil = "Mahasiswa dalam klaster ini menunjukkan performa akademik yang lebih rendah, meskipun terlibat dalam MBKM dan organisasi."

            interpretasi.append(
                {
                    "cluster": cluster,
                    "jumlah_data": jumlah_data,
                    "rata_rata_ipk": round(rata_rata_ipk, 2),
                    "mbkm_percent": round(mbkm_percent, 2),
                    "bekerja_percent": round(bekerja_percent, 2),
                    "organisasi_percent": round(organisasi_percent, 2),
                    "anggota": anggota,
                    "interpretasi": interpretasi_hasil,
                }
            )

        return {
            "dendrogram": dendrogram_path,
            "perubahan_ipk": perubahan_ipk_path,
            "silhouette_score_plot": silhouette_plot_path,
            "silhouette_score": round(silhouette_avg, 2),
            "interpretasi": interpretasi,
            "dataset": dataset.to_html(classes="table table-striped"),
            "data": data.to_html(classes="table table-striped"),
            "data_normalized": data_scaled_df.to_html(classes="table table-striped"),
        }

    except Exception as e:
        return {"error": str(e)}


# Route untuk halaman utama
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("Tidak ada file yang di upload", "danger")
            return redirect(request.url)
        if file and file.filename.endswith(".csv"):  # Pastikan file CSV
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], "dataset.csv")
            file.save(filepath)
            # Proses aplikasi
            hasil = process_file(filepath)
            if "error" in hasil:
                flash("Data tidak valid", "danger")
                flash(f"Error: {hasil['error']}", "warning")
                return redirect(request.url)
            return render_template("hasil.html", hasil=hasil)
        else:
            flash("Hanya menerima upload csv", "danger")
            return redirect(request.url)
    return render_template("index.html")


if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
