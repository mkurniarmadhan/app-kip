import numpy as np

# Matriks jarak awal
distance_matrix = np.array(
    [
        [0, 2.363, 2.808, 1.962, 1.486],
        [2.363, 0, 2.040, 2.060, 1.887],
        [2.808, 2.040, 0, 1.933, 2.428],
        [1.962, 2.060, 1.933, 0, 1.395],
        [1.486, 1.887, 2.428, 1.395, 0],
    ]
)


# Fungsi untuk memperbarui matriks jarak menggunakan metode Ward
def update_distance_matrix(dist_matrix, cluster1, cluster2):
    n = len(dist_matrix)
    new_matrix = np.zeros((n - 1, n - 1))

    # Copy jarak yang tidak terkait dengan cluster1 dan cluster2
    index_map = [i for i in range(n) if i not in [cluster1, cluster2]]
    for i, idx1 in enumerate(index_map):
        for j, idx2 in enumerate(index_map):
            new_matrix[i, j] = dist_matrix[idx1, idx2]

    # Tambahkan jarak untuk klaster baru
    for i, idx in enumerate(index_map):
        new_distance = (dist_matrix[cluster1, idx] + dist_matrix[cluster2, idx]) / 2
        new_matrix[i, len(index_map)] = new_distance
        new_matrix[len(index_map), i] = new_distance

    # Tambahkan elemen diagonal (nol untuk klaster baru)
    new_matrix[len(index_map), len(index_map)] = 0
    return new_matrix


# Proses penggabungan
while len(distance_matrix) > 1:
    n = len(distance_matrix)

    # Tampilkan matriks sebelum penggabungan
    print("\nMatriks Sebelum Penggabungan:")
    print(distance_matrix)

    # Cari dua klaster dengan jarak terkecil
    min_val = np.inf
    cluster1, cluster2 = -1, -1
    for i in range(n):
        for j in range(i + 1, n):
            if distance_matrix[i, j] < min_val:
                min_val = distance_matrix[i, j]
                cluster1, cluster2 = i, j

    print(f"Gabungkan klaster {cluster1} dan {cluster2} (jarak: {min_val})")

    # Update matriks jarak
    distance_matrix = update_distance_matrix(distance_matrix, cluster1, cluster2)

    # Tampilkan matriks setelah penggabungan
    print("Matriks Setelah Penggabungan:")
    print(distance_matrix)
