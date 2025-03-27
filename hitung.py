import numpy as np
import pandas as pd


# Data untuk setiap responden
data = {
    "R1": np.array(
        [0.67, 1.00, 0.67, 0.75, 0.75, 1.00, 0.80, 1.00, 1.00, 0.00, 1.00, 0.00]
    ),
    "R2": np.array(
        [0.00, 0.99, 1.00, 1.00, 0.25, 0.00, 0.00, 0.00, 0.73, 1.00, 0.00, 0.00]
    ),
    "R3": np.array(
        [0.07, 0.00, 0.00, 0.00, 0.50, 0.00, 0.00, 0.25, 0.50, 1.00, 0.00, 1.00]
    ),
    "R4": np.array(
        [0.67, 0.81, 0.67, 0.42, 0.75, 1.00, 1.00, 0.25, 0.68, 1.00, 0.00, 1.00]
    ),
    "R5": np.array(
        [1.00, 0.99, 0.80, 0.67, 0.67, 0.75, 0.80, 1.00, 0.91, 1.00, 0.00, 0.00]
    ),
}


# Nama responden
respondents = list(data.keys())

# Menghitung jarak Euclidean untuk setiap pasangan responden
distance_matrix = np.zeros((len(respondents), len(respondents)))

for i, r1 in enumerate(respondents):
    for j, r2 in enumerate(respondents):
        distance_matrix[i, j] = np.sqrt(np.sum((data[r1] - data[r2]) ** 2))

# Membulatkan hasil ke 3 angka desimal
distance_matrix = np.round(distance_matrix, 3)

# Membuat tabel dengan pandas
distance_table = pd.DataFrame(distance_matrix, index=respondents, columns=respondents)

# Menampilkan tabel
print("Matriks Jarak (Euclidean):")
print(distance_table)
