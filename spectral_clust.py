import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

def gaussian_weighing(v1, v2, bw):
    return np.exp(-np.sum((v1-v2)**2)/(2*bw**2))

def gaussian_kernel(data, sigma):
    n = data.shape[0]
    kernel_matrix = np.identity(n)
    for i in range(n):
        for j in range(i+1,n):
            kernel_matrix[i,j] = gaussian_weighing(data[i,:], data[j,:], bw = sigma)
    return (kernel_matrix + kernel_matrix.T) - np.identity(n)

class spectral_clust:
    def __init__(self, data, n_clust):
        self.data = data
        self.n_clust = n_clust
        
    def create_W(self, method):
        n = self.data.shape[0]
        W = np.empty((n,n))
        
        if(method == 'knn'):
            k = int(input('Enter k: '))
            nn = NearestNeighbors(n_neighbors = k+1, radius = None)
            nn.fit(self.data)
            dist, idx = nn.kneighbors(self.data)
            for i in range(n):
                W[i,idx[i,1:]] = 1
            self.W = W
            self.method = 'knn'
            return self
        
        elif(method == 'epsilon ball'):
            epsilon = float(input('Enter epsilon: '))
            nn = NearestNeighbors(radius = epsilon)
            nn.fit(self.data)
            dist, idx = nn.radius_neighbors(self.data)
            for i in range(n):
                idx[i] = idx[i][np.argsort(dist[i])]
                W[i,idx[i]] = 1
            self.W = W
            self.method = 'epsilon neighborhood'
            return self
        
        elif(method == 'fully connected'):
            sigma = float(input('Enter sigma: '))
            self.W = gaussian_kernel(self.data, sigma)
            self.method = 'fully connected'

        else:
            method = str(input('Incorrect method, please enter again: '))
            self.create_W(method)
        
    def laplacian(self):
        self.L = np.diag(np.sum(self.W, axis = 1)) - self.W
        return self
        
    def cluster(self):
        self.vals, self.vecs = np.linalg.eig(self.L)
        eigenvals_sorted_indices = np.argsort(self.vals)
        self.eigenvals_sorted = self.vals.real[eigenvals_sorted_indices]
        self.eigenvecs_sorted = self.vecs.real[:,eigenvals_sorted_indices]
        kmeans = KMeans(n_clusters = self.n_clust)
        kmeans.fit(self.eigenvecs_sorted[:,:self.n_clust])
        self.partition = kmeans.labels_
        return self.partition

