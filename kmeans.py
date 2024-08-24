from joblib import Parallel, delayed
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from numba import njit, prange
sns.set_theme(style="whitegrid")

def generate_random_points(n, dim):
    return np.random.rand(n, dim).astype(np.float32)

@njit(parallel=True)
def compute_label_numba(point, centroids):
    k = centroids.shape[0]
    distances = np.empty(k)
    for i in prange(k):
        distances[i] = np.linalg.norm(point - centroids[i])
    return np.argmin(distances)

def compute_label_joblib(point, centroids):
    k = centroids.shape[0]
    distances = np.empty(k)
    for i in range(k):
        distances[i] = np.linalg.norm(point - centroids[i])
    return np.argmin(distances)

class KMeans:
    def __init__(self, points, k, use_numba=True):
        self.points = points
        self.k = k
        self.centroids = np.random.rand(k, np.shape(points)[1]).astype(np.float32)
        self.labels = np.zeros(points.shape[0], dtype=np.int32)
        self.use_numba = use_numba

    def fit(self, n_jobs = 1):
        for _ in range(100):
            if self.use_numba:
                new_labels = np.empty(self.points.shape[0], dtype=np.int32)
                for i in range(self.points.shape[0]):
                    new_labels[i] = compute_label_numba(self.points[i], self.centroids)
            else:
                new_labels = Parallel(n_jobs=n_jobs)(delayed(compute_label_joblib)(self.points[i], self.centroids) for i in range(self.points.shape[0]))
                new_labels = np.array(new_labels)

            if np.array_equal(self.labels, new_labels):
                break

            self.labels = new_labels

            for i in range(self.k):
                points_in_cluster = self.points[self.labels == i]
                if len(points_in_cluster) > 0:
                    self.centroids[i] = np.mean(points_in_cluster, axis=0)

if __name__ == "__main__":
    np.random.seed(111)
    sizes = [5000, 7500, 10000] 
    n_jobs = [1, 2, 4, -1]
    k = 1000
    for n in sizes:
        print(f"====={n}====")
        points = generate_random_points(n, 100)
        
        kmeans_numba = KMeans(points, k=k, use_numba=True)
        start_time = time.time()
        kmeans_numba.fit()
        end_time = time.time()
        print(f"Numba: {n} points fitted in: {end_time - start_time:.4f} seconds")

        for n_job in n_jobs:
            kmeans_joblib = KMeans(points, k=k, use_numba=False)
            start_time = time.time()
            kmeans_joblib.fit(n_jobs=n_job)
            end_time = time.time()
            print(f"Joblib_{n_job}: {n} points fitted in: {end_time - start_time:.4f} seconds")

