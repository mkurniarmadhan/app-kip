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
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.stats import zscore

app = Flask(__name__)

app.secret_key = "appkip"

UPLOAD_FOLDER = "static/hasil"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def process_file(filepath):
    try:

        # baca data yang di updaload
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

        # Normalisasi Z
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data[features])

        data_scaled_df = pd.DataFrame(data_scaled, columns=features)
        data_scaled_df.to_csv(
            os.path.join(app.config["UPLOAD_FOLDER"], "data_normalized.csv")
        )

        # Hierarchical Clustering dan Dendrogram
        linkage_matrix = linkage(data_scaled, method="ward", metric="euclidean")

        # Visualisasi Dendrogram
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
            silhouette_scores.append(score)
        silhouette_df = pd.DataFrame(
            {
                "Jumlah Klaster": cluster_range,
                "Nilai Silhouette Score": silhouette_scores,
            }
        )

        # Plot Silhouette Scores
        plt.figure(figsize=(8, 5))
        plt.plot(cluster_range, silhouette_scores, marker="o")
        plt.title("Silhouette Scores untuk Berbagai Jumlah Klaster")
        plt.xlabel("Jumlah Klaster")
        plt.ylabel("Silhouette Score")
        plt.grid(True)
        silhouette_path = os.path.join(
            app.config["UPLOAD_FOLDER"], "silhouette_score.png"
        )
        plt.savefig(silhouette_path)
        plt.close()

        klaster_optimal = cluster_range[np.argmax(silhouette_scores)]

        # Clustering dengan jumlah klaster optimal
        clustering = AgglomerativeClustering(
            n_clusters=klaster_optimal, linkage="ward", metric="euclidean"
        )

        data["Cluster"] = clustering.fit_predict(data_scaled)

        cluster_mean_ipk = {}

        # Interpretasi Hasil Klasterisasi
        interpretasi = []
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

            interpretasi_hasil = ""
            if rata_rata_ipk >= 3.5:
                interpretasi_hasil += (
                    "Pola IPK: Stabil atau meningkat dengan IPK Kumulatif yang baik.\n"
                )
            else:
                interpretasi_hasil += (
                    "Pola IPK: Mungkin ada penurunan atau ketidakstabilan dalam IPK.\n"
                )

            if mbkm_percent > 50:
                interpretasi_hasil += "Faktor MBKM: Mahasiswa dalam klaster ini memiliki banyak yang ikut MBKM.\n"

            if bekerja_percent > 50:
                interpretasi_hasil += "Faktor Pekerjaan: Mahasiswa dalam klaster ini banyak yang bekerja sambil kuliah.\n"

            if organisasi_percent > 50:
                interpretasi_hasil += "Faktor Organisasi: Mahasiswa dalam klaster ini aktif dalam organisasi.\n"

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
            plt.xlabel("Semester")
            plt.ylabel("Rata-rata IPK")
            plt.legend(title="Klaster")
            plt.grid(True)
            plt.tight_layout()
            perubahan_ipk_path = os.path.join(
                app.config["UPLOAD_FOLDER"], "perubahan_ipk.png"
            )
            plt.savefig(perubahan_ipk_path)
            plt.close()

        return {
            "dendrogram": dendrogram_path,
            "perubahan_ipk": perubahan_ipk_path,
            "silhouette_score_plot": silhouette_path,
            "klaster_optimal": klaster_optimal,
            "interpretasi": interpretasi,
            "dataset": dataset.to_html(classes="table table-striped"),
            "data": data.to_html(classes="table table-striped"),
            "data_normalized": data_scaled_df.to_html(classes="table table-striped"),
            "silhouette_df": silhouette_df.to_html(classes="table table-striped"),
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
        if file and file.filename.endswith(".csv"):
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
