from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.linalg import sqrtm

def fid_score(mean_a, cov_a, mean_b, cov_b):
    m = sum((mean_a - mean_b) ** 2)
    n = cov_a + cov_b - 2 * sqrtm(cov_a @ cov_b)
    c = m + np.trace(n)
    return c

domains = ["baseline", "clear", "daytime", "night",
           "partly_cloudy", "residential", "city_street",
           "dawn_dusk", "highway", "overcast", "rainy", "snowy"]

embeddings = {}
df = pd.DataFrame(columns=domains, index=domains)

for d in domains:
    mean = np.loadtxt(f"embeddings/{d}/mean.csv")
    cov = np.loadtxt(f"embeddings/{d}/cov.csv")
    embeddings[d] = (mean, cov)

for data1 in domains:
    series = {}
    for data2 in domains:
        mean1, cov1 = embeddings[data1]
        mean2, cov2 = embeddings[data2]
        distance = fid_score(mean1, cov1, mean2, cov2)
        series[data2] = 1/ (distance/1e4 + 0.1)

    df.loc[data1] = pd.Series(series)


df = df.style.format(decimal='.', precision=3)
print(df.to_latex())