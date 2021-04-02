import numpy as np
import ML_algo.utils as utils
class MDS:
    def __init__(self, X, num_comp):
        self.train_x = np.asarray(X)
        self.num_samples = self.train_x.shape[0]
        self.num_comp = num_comp
        self.transform_x = []
        self.train()

    def train(self):
        minkowski_sum = utils.Minkowski_sum(2)
        dist_mat_square = np.array([
            [ minkowski_sum(self.train_x[i] - self.train_x[j]) for i in range(self.num_samples) ]
            for j in range(self.num_samples)
        ])
        col_sum = np.sum(dist_mat_square, axis = 1).squeeze()/self.num_samples
        row_sum = np.sum(dist_mat_square, axis = 0).squeeze()/self.num_samples
        total_sum = np.sum(dist_mat_square)/self.num_samples/self.num_samples

        prod_mat = np.array(
            [-0.5*(dist_mat_square[i,j] - col_sum[i] - row_sum[j] + total_sum) for i in range(self.num_samples)]
            for j in self.num_samples
        )
