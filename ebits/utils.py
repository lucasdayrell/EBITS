import numpy as np
from sklearn.cluster import KMeans

def cluster_patterns(P: np.ndarray, n_clusters: int = 5):
    km = KMeans(n_clusters=n_clusters, n_init='auto').fit(P)
    centers = km.cluster_centers_
    labels = km.labels_
    return centers, labels
