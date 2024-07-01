import numpy as np
import matplotlib.pyplot as plt
import typing as t
plt.ion()


def euclidean_distance(x1: np.ndarray[t.Any, np.dtype[t.Any]], x2):
   #return np.linalg.norm(x1 - x2)
    return np.sqrt(np.sum((x1 - x2)**2))



class KMeans:
    def __init__(self, n_clusters, * ,max_iter = 100, plot_steps = False) -> None:
        self.n_clusters=n_clusters
        self.max_iter=max_iter
        self.plot_steps=plot_steps

        #Lists to store indices of sample for each cluster.
        self.clusters = [[] for _ in range(self.n_clusters)]

        #List to store centers of each cluster.
        self.centroids = []

    def predict(self , X: np.ndarray[t.Any, np.dtype[t.Any]]):
        self.X = X
        assert isinstance(X, np.ndarray), "only supporting numpy array"

        self.n_samples , self.n_features=X.shape

        #initialize the centers
        random_samples_idx = np.random.choice(self.n_samples, self.n_clusters, replace=False)
        self.centroids = [self.X[idx] for idx in random_samples_idx]

        # Optimize clusters
        for _ in range(self.max_iter):

            #assign samples to the closet centroids (create clusters)
            self.clusters = self._create_clusters(self.centroids)

            if self.plot_steps:
                self.plot()

            #calculate the centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            if self._is_converged(centroids_old, self.centroids):
                break
            
            if self.plot_steps:
                self.plot()

        #classify samples as the index of their clusters
        return self._get_cluster_labels(self.clusters)
    
    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels
    
    def _create_clusters(self, centroids):

        clusters = [[] for _ in range(self.n_clusters)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample,centroids)
            clusters[centroid_idx].append(idx)

        return clusters

    def _closest_centroid(self, sample, centroids):
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)

        return closest_idx
    
    def _get_centroids(self, clusters):

        # assign mean values of the clusters to centroids
        centroids = np.zeros((self.n_clusters, self.n_features))

        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean

        return centroids        

    def _is_converged(self, centroids_old, centroids):
        
        # Check distances between old and new centroids
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.n_clusters)]
        return sum(distances) == 0    

    def plot(self):
        
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index  in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="red", linewidth=2)
        
        plt.show()


    
# Testing

if __name__ == '__main__':  
    np.random.seed(42)
    from sklearn.datasets import make_blobs

    X,y = make_blobs(centers=5, n_samples=100, n_features=2, shuffle=True, random_state=40)

    print(X.shape)

    clusters = len(np.unique(y))
    print(clusters)

    model = KMeans(n_clusters=clusters, max_iter=150, plot_steps=True)
    y_predict = model.predict(X)

    model.plot()


