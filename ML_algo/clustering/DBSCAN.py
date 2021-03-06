'''
@author: Yuanhang Tang
@file: ML_algo/clustering/DBSCAN.py
@github: https://github.com/yuanhangtangle/ML_codes/tree/main/ML_algo/clustering
'''

import numpy as np
import matplotlib.pyplot as plt
import ML_algo.utils as utils
import seaborn as sns
class DBSCAN:
    def __init__(self, X, eps = 1, minPts = 4, ord = 2, paired = False):
        '''
        Density-Based Spatial Cluster of Application with Noise.
        '''

        self.train_x = np.asarray(X, dtype=np.float)
        self.num_samples = self.train_x.shape[0]
        self.num_features = self.train_x.shape[1]
        self.num_groups = 0
        self.groups = [-1] * self.num_samples
        self.reachable = [[] for i in range(self.num_samples)]
        self.ord = ord
        self.minPts = minPts
        self.eps = eps
        self.pairs = []
        self.core_samples = []
        self.non_core_samples = []
        self.train(paired)

    def train(self, paired):
        dist = utils.Minkowski_distance(self.ord)
        # compute distance matirx
        dist_mat = [[dist(self.train_x[i], self.train_x[j]) for j in range(self.num_samples)] for i in range(self.num_samples)]
        # construct adjacent list and core samples
        adj_list = [[] for i in range(self.num_samples)]
        for i in range(self.num_samples):
            for j in range(self.num_samples):
                if dist_mat[i][j] < self.eps:
                    adj_list[i].append(j)
            if (len(adj_list[i]) - 1) >= self.minPts: # NOT including itself
                self.core_samples.append(i)
            else:
                self.non_core_samples.append(i)
        self.BFS(adj_list, paired)
        self.groups = np.asarray(self.groups)
        self.dist_mat = dist_mat

    def BFS(self, adj_list, paired):
        # bfs
        q = []
        visited = [False] * self.num_samples
        num_comp = 0
        # search until core_samples are all visited
        for idx in self.core_samples:
            if visited[idx]: continue
            q.append(idx) # choose a core sample
            visited[idx] = True
            while q:
                i = q.pop(0) # get first item
                self.groups[i] = num_comp # mark it and
                if i in self.core_samples: # if it is a core sample
                    for j in adj_list[i]: # search its neighbors
                        if not visited[j]: # if it has not been visited
                            q.append(j)
                            visited[j] = True  # visited
                            if paired and i != j:
                                self.pairs.append([i,j])
            num_comp += 1 # increase connected component marker

if __name__ == '__main__':
    size = 20
    eps = 0.8
    minPts = 3
    x1 = np.random.multivariate_normal(mean=(-2, 0), cov=np.diag((1, 0.8)), size=size)
    x2 = np.random.multivariate_normal(mean=(2, 0), cov=np.diag((1, 1.5)), size=size)
    x3 = np.random.multivariate_normal(mean=(0, -2), cov=np.diag((0.9, 1.0)), size=size)
    x4 = np.random.multivariate_normal(mean=(0, 2), cov=np.diag((0.9, 0.9)), size=size)
    origin = np.array([[1] * size + [2] * size + [3] * size + [4] * size]).T
    X = np.r_[x1, x2, x3, x4]
    #X = np.c_[X, origin]
    xmean = [-2, 2, 0, 0]
    ymean = [0, 0, -2, 2]

    dbscan = DBSCAN(X, eps, minPts, paired=True)
    for i in dbscan.non_core_samples:
        print(i, X[i, 0:2])

    plt.subplot(121)
    # inliers
    plt.scatter(
        X[:, 0][dbscan.groups != -1],
        X[:, 1][dbscan.groups != -1],
        c=dbscan.groups[dbscan.groups != -1] / 4,
        marker='.'
    )
    # outliers
    outliers_x = X[:, 0][dbscan.groups == -1]
    outliers_y = X[:, 1][dbscan.groups == -1]
    plt.scatter(
        outliers_x,
        outliers_y,
        marker = '*'
    )
    '''for (i,j) in zip(outliers_x, outliers_y):
        plt.text(i,j,'[%.3f,%.3f]'%(i,j))'''
    # draw a circle for each outlier
    '''for (x,y) in zip(outliers_x, outliers_y):
        utils.circle(x,y,eps)'''

    # draw connected component
    for (i,j) in dbscan.pairs:
        plt.plot([X[i,0],X[j,0]], [X[i,1], X[j,1]], color = 'r', linewidth = 1)
    plt.title('DBSCAN')

    plt.subplot(122)
    plt.scatter(X[:, 0], X[:, 1], c=origin / 4, marker='.')
    plt.title('origin')
    plt.show()

    '''plt.figure()
    sns.heatmap(np.asarray(dbscan.dist_mat) < eps, cmap = 'rainbow')
    plt.show()'''






