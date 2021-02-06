'''
@@author: Yuanhang Tang
@@datetime: 2020/10/22
@@description: this module implements the classic k means algorithm
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as  plt
class k_means_cluster:
    def __init__(self , X, k, eps = 0.001, max_step = 100, times = 15): #(m,n) matrix, with m samples and n attributes
        self.X = np.asarray(X)
        self.group_num = k # number of cluster
        mean_idx = np.random.choice(range(0,self.X.shape[0]),k)
        self.group = np.zeros([self.X.shape[0], 1])
        self.centers = self.X[mean_idx]
        self.dist = np.zeros([X.shape[0], 1])#square of Euclidean distance
        self.convergence = False
        self.loss = 0
        self.optimize(max_step, eps)

    def optimize(self, max_step, eps):
        '''
        coordinate descend, optimization
        :param max_step: maximum steps used in the optimization
        :param eps: tolerance; iteration stops if (new loss - old loss)/old loss < eps
        '''
        self.assign_samples()
        self.compute_loss()
        old_loss = self.loss
        for i in range(max_step):
            self.compute_means()
            self.assign_samples()
            self.compute_loss()
            '''
                        if (old_loss - self.loss)/old_loss < eps:
                self.convergence = True
                break
            else:
            '''
            old_loss = self.loss

    def assign_samples(self):
        '''
        assign sample to the nearest center
        '''
        for i in range(X.shape[0]):
            idx, min = self._assign_single_sample(i)
            (self.group[i, 0], self.dist[i, 0]) = (idx,min)

    def _assign_single_sample(self, i):
        '''
        assign a sample to its nearest center
        :param i: index of the sample
        :return: idx of center, square of the Euclidean distance to this center
        '''
        min = (self.X[i] - self.centers[0])@(self.X[i] - self.centers[0]).T
        idx = 0
        for j in range(1, self.group_num):
            t = (self.X[i] - self.centers[j])@(self.X[i] - self.centers[j]).T
            if t < min:
                (idx, min) = (j, t)
        return idx,min

    def compute_means(self):
        '''
        compute the center for every cluster
        '''
        for i in range(self.group_num):
            t = np.array((self.group == i), dtype=np.int32)
            c = t*(self.X)
            self.centers[i] = np.sum(t*(self.X), axis = 0)/np.sum(t)

    def compute_loss(self):
        self.loss = np.sum(self.dist)/self.group_num

    def pred(self, x):
        min = (x - self.centers[0]) @ (x - self.centers[0]).T
        idx = 0
        for j in range(1, self.group_num):
            t = (x - self.centers[j]) @ (x - self.centers[j])
            if t < min:
                (idx, min) = (j, t)
        return idx
        
if __name__ == "__main__":
    print('testing  message')
    x1 = np.random.multivariate_normal(mean = (-2,0), cov = np.diag((1,0.8)),size = 20)
    x2 = np.random.multivariate_normal(mean = (2,0), cov = np.diag((1,1.5)),size = 20)
    x3 = np.random.multivariate_normal(mean = (0,-2), cov = np.diag((0.9,1.0)),size = 20)
    x4 = np.random.multivariate_normal(mean = (0,2), cov = np.diag((0.9,0.9)),size = 20)
    origin = np.array([[1]*20+[2]*20+[3]*20+[4]*20]).T
    X = np.r_[x1, x2, x3, x4]
    X = np.c_[X,origin]
    xmean = [-2,2,0,0]
    ymean = [0,0,-2,2]

    kmeans = k_means_cluster(X, 4, eps = 0.01)
    X = np.c_[X,kmeans.group]
    plt.scatter(X[:,0],X[:,1],c = X[:,2]/4)
    plt.scatter(xmean, ymean, marker='x', s = 120, label = 'original means')
    plt.scatter(kmeans.centers[:,0], kmeans.centers[:,1], marker='*', s = 120, label = 'k means centers')
    plt.legend()
    plt.show()
