'''
@author: Yuanhang Tang
@file: ML_algo/clustering/agnes.py
@github: https://github.com/yuanhangtangle/ML_codes/tree/main/ML_algo/clustering
'''

import numpy as np
import ML_algo.utils as utils
import matplotlib.pyplot as plt

class AGNES_cluster:
    def __init__(self, X, groups, ord = 2, dist = 'avg', store = []):
        '''
        Implement the classic AGNES clustering method.
        params:
        X: training sample
        groups: number of clusters
        ord: order of distance meature; the 'p' in Minkowski distance
        dist: method to compute the distance between two sets, choose from [min, max, avg]
        store: to store the cluster output if the number of groups is in ``store``
        '''
        self.train_x = np.asarray(X, dtype=np.float)
        self.num_samples = self.train_x.shape[0]
        self.num_features = self.train_x.shape[1]
        self.num_groups = self.num_samples #current number of groups
        self.groups = np.arange(self.train_x.shape[0]) # groups
        self.ord = ord
        self.set_dist = utils._set_dist(ord, dist)
        self.store_groups = dict() # number of groups -> cluster output
        self.train(groups, store)

    def train(self, groups, store):
        min1 = -1
        min2 = -1
        n = self.num_samples - groups
        for i in range(n): # merge n pairs
            minDist = 0x7fffffff # C++ style ???
            for g1 in self.groups: # closest pair
                for g2 in range(g1 + 1, self.num_groups):
                    group1 = self.train_x[self.groups == g1]
                    group2 = self.train_x[self.groups == g2]
                    d = self.set_dist(group1, group2)
                    if(d < minDist):
                        minG1 = g1
                        minG2 = g2
                        minDist = d
                        
            for i in range(self.num_samples): # merge groups
                if self.groups[i] == minG2:
                    self.groups[i] = minG1
                elif self.groups[i] == self.num_groups - 1:
                    self.groups[i] = minG2
            self.num_groups -= 1

            if type(store) == bool and store:
                self.store_groups[self.num_groups] = self.groups.copy()
            elif type(store) == list and self.num_groups in store:
                self.store_groups[self.num_groups] = self.groups.copy()
                    
if __name__ == '__main__':
    size = 10
    x1 = np.random.multivariate_normal(mean=(-2, 0), cov=np.diag((1, 0.8)), size=size)
    x2 = np.random.multivariate_normal(mean=(2, 0), cov=np.diag((1, 1.5)), size=size)
    x3 = np.random.multivariate_normal(mean=(0, -2), cov=np.diag((0.9, 1.0)), size=size)
    x4 = np.random.multivariate_normal(mean=(0, 2), cov=np.diag((0.9, 0.9)), size=size)
    origin = np.array([[1] * size + [2] * size + [3] * size + [4] * size]).T
    X = np.r_[x1, x2, x3, x4]
    X = np.c_[X, origin]
    xmean = [-2, 2, 0, 0]
    ymean = [0, 0, -2, 2]


    agnes = AGNES_cluster(X, 1, dist='avg', store=[2,4,6])
    for i in [4]:
        plt.subplot(121)
        plt.scatter(X[:, 0], X[:, 1], c =agnes.store_groups[i] / 4)
        plt.title('AGNES')

        plt.subplot(122)
        plt.scatter(X[:, 0], X[:, 1], c=origin / 4)
        plt.title('origin')
        plt.show()

