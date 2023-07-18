from __future__ import annotations
import h5py
import numpy as np
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

class Transformation:
    def __init__(self, baseline_data):
        shape1 = baseline_data.shape[1] * baseline_data.shape[2] * baseline_data.shape[3]
        baseline_data = np.reshape(baseline_data, (baseline_data.shape[0], shape1))
        self.ss = StandardScaler()
        self.ss.fit(baseline_data)
        self.pca = PCA(n_components=50)
        self.pca.fit(self.ss.transform(baseline_data))

    def transform(self, data):
        shape1 = data.shape[1] * data.shape[2] * data.shape[3]
        data = np.reshape(data, (data.shape[0], shape1))
        return self.pca.transform(self.ss.transform(data))

class Embeddings:
    def __init__(self, data, transform: Transformation):
        reduced_data = transform.transform(data)
        self.cov = np.cov(reduced_data.T)
        self.mean = np.mean(reduced_data, axis=0)



domains = ["baseline", "clear", "daytime", "night",
           "partly_cloudy", "residential", "city_street",
           "dawn_dusk", "highway", "overcast", "rainy", "snowy"]

# Create baseline
with h5py.File(f"embeddings/baseline/embeddings.h5", "r") as hf:
    transform = Transformation(np.array(hf['data']))

# Extract Means and Covariances
for d in tqdm(domains):
    with h5py.File(f"embeddings/{d}/embeddings.h5", "r") as hf:
        print(hf['data'].shape)
        embedding = Embeddings(np.array(hf['data']), transform=transform)
        np.savetxt(f"embeddings/{d}/mean.csv", embedding.mean)
        np.savetxt(f"embeddings/{d}/cov.csv", embedding.cov)

