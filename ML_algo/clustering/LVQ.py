'''
@author: Yuanhang Tang
@file: ML_algo/clustering/LVQ.py
@github: https://github.com/yuanhangtangle/ML_codes/tree/main/ML_algo/clustering
'''

import numpy as np
import random
import matplotlib.pyplot as plt
import ML_algo.utils as utils

class LVQ:
    def __init__(self, X, Y, max_step, groups, ord = 2, learning_rate = 1e-3):
        self.train_x = np.asarray(X, dtype=np.float)
        self.train_y = np.asarray(Y, dtype=np.int).reshape(-1)
        self.num_samples = self.train_x.shape[0]
        self.num_features = self.train_x.shape[1]
        self.learning_rate = learning_rate
        self.ord = ord
        self.num_groups = 0
        self.centers = np.array([])
        self.groups = []

        self.train(max_step, groups, ord)

    def train(self, max_step, groups, ord):
        for key in groups:
            self.num_groups += groups[key]
        self.centers = np.zeros([self.num_groups, self.num_features]) # set aside room for prototypes
        # sampling prototypes
        idx = 0
        for key in groups:
            c = np.argwhere(self.train_y == key).reshape(-1)
            ran_idx = random.sample(list(np.argwhere(self.train_y == key).reshape(-1)), groups[key])
            #self.centers.extend(list(self.train_x[idx]))
            self.centers[idx:idx+groups[key]] = self.train_x[ran_idx, :]
            self.groups.extend([key] * groups[key])
            idx = idx + groups[key]
        self.groups = np.asarray(self.groups)
        # udpate prototypes
        dist = utils.Minkowski_distance(ord) # set distance
        random_idx = range(0,self.num_samples)
        for i in range(max_step):
            idx = np.random.choice(random_idx, 1)[-1]# pick one sample
            minDist = 0x7fffffff # C++ style again :)
            for j in range(0, self.num_groups):
                d = dist(self.train_x[idx], self.centers[j])
                if d < minDist: # closest prototype
                    minIdx = j
                    minDist = d
            a = self.learning_rate * (self.train_x[idx] - self.centers[minIdx]).reshape(-1)
            d = self.groups[minIdx]
            if(self.train_y[idx] == self.groups[minIdx]): #same group
                self.centers[minIdx] += a
                # print(idx, minIdx, "same: {} == {}".format(self.train_y[idx], self.groups[minIdx]))
            else: # different group
                self.centers[minIdx] -= a
                # print(idx, minIdx, 'diff: {} == {}'.format(self.train_y[idx], self.groups[minIdx]))


        # assign samples
        #self._assign_samples(

    def _assign_samples(self):
        pass

if __name__ == '__main__':
    size = 10
    max_step = 10000
    groups = {
        1:2,
        2:2,
        3:2,
        4:2
    }
    lr = 1e-2
    x1 = np.random.multivariate_normal(mean=(-2, 0), cov=np.diag((1, 0.8)), size=size)
    x2 = np.random.multivariate_normal(mean=(2, 0), cov=np.diag((1, 1.5)), size=size)
    x3 = np.random.multivariate_normal(mean=(0, -2), cov=np.diag((0.9, 1.0)), size=size)
    x4 = np.random.multivariate_normal(mean=(0, 2), cov=np.diag((0.9, 0.9)), size=size)
    origin = np.array([[1] * size + [2] * size + [3] * size + [4] * size]).T
    X = np.r_[x1, x2, x3, x4]
    X = np.c_[X, origin]
    xmean = [-2, 2, 0, 0]
    ymean = [0, 0, -2, 2]

    lvq = LVQ(X, origin.reshape(4*size, 1), max_step = max_step, groups=groups, learning_rate=lr)
    plt.scatter(X[:, 0], X[:, 1], c=origin / 4)
    plt.scatter(xmean, ymean, marker='x', s=120, label='original means')
    plt.scatter(lvq.centers[:, 0], lvq.centers[:, 1], c = lvq.groups/4, marker='*', s=120, label='LVQ centers')
    plt.legend()
    plt.show()
