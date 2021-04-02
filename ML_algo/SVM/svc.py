import numpy as np
import pandas as pd
import ML_algo.utils as utils
import  matplotlib.pyplot as plt
class SVC:
    available_kernels = ['LINEAR', 'RBF', 'POLY']
    def __init__(self, C, kernel = 'LINEAR', gamma = None, p = None):
        self.C = C
        self.alpha = None
        self.w = 0
        self.b = 0
        self.is_fitted = False
        self.is_trained = False
        self.kernel = kernel
        self.gamma = gamma
        self.p = p

        assert kernel in SVC.available_kernels, '`kernel` must be selected from `SVC.available_kernels`'

        #3 set kernel
        if kernel == 'LINEAR':
            self.kernel_map = utils.linear_kernel()
        elif kernel == 'POLY':
            p = float(p)
            # assert isinstance(p, float), 'A float object must be passed to `p` when polynomial kernel is chosen'
            self.kernel_map = utils.poly_kernel(p)

        self.hyperparams = {
            'C': self.C,
            'alpha': self.alpha,
            'kernel': self.kernel,
            'gamma': self.gamma,
            'p': self.p
        }

    def fit(self, X,  Y):
        self.train_x = np.asarray(X)
        assert len(self.train_x.shape) == 2, "X must be 2 dimension"
        self.num_features = self.train_x.shape[1]
        self.num_samples = len(Y)
        assert self.train_x.shape[0] == self.num_samples, "The number of samples in X and y must be the same"
        self.train_y = np.asarray(Y).reshape((self.num_samples,-1))

        # set alpha according to number of features
        self.alpha = np.zeros([self.num_samples,1])

        # set rbf kernel
        if self.kernel == 'RBF':
            if self.gamma is None:
                self.gamma = 1 / self.num_features
                self.hyperparams['gamma'] = self.gamma
            self.kernel_map = utils.rbf_kernel(self.gamma)
        # build gram matrix
        self._build_gram()
        self.is_fitted = True

    def train(
            self,
            max_step = 100000,
            eps = 1e-3,
            print_step = None
    ):
        self.eps = eps
        self.E = np.zeros(self.num_samples)
        self._update_E()
        step = 0
        self.loss_store = []

        while step < max_step:
            '''# check support vectors
            start_index = np.random.randint(0, self.num_samples)
            # loop for the first idx where the sample violates the KKT condition
            for inc_all in range(self.num_samples):
                i_all = (start_index + inc_all) % self.num_samples
                if not self._check_KKT(i):
                    break
            if i_all == start_index: # all satisfied
                break

            # if not satisfied, search support vector
            i = i_all
            while i != start_index:
                if self._is_support_vector(i) and not self._check_KKT(i):
                    break
                else:
                    i = (i + 1) % self.num_samples

            if i == start_index: # all support vector satisfy KKT condition
                i = i_all # just work with the first

            # now we search for the second one'''
            start_index = np.random.randint(0, self.num_samples)
            for inc in range(self.num_samples):
                i = (start_index + inc) % self.num_samples
                if self._check_KKT(i):
                    continue
                j = int(np.argmax(np.abs(self.E - self.E[i])))
                if self.train_y[i] == self.train_y[j]:
                    L = max(0, self.alpha[j, 0] + self.alpha[i, 0] - self.C)
                    H = min(self.C, self.alpha[j, 0] + self.alpha[i, 0])
                else:
                    L = max(0, self.alpha[j, 0] - self.alpha[i, 0])
                    H = min(self.C, self.C + self.alpha[j, 0] - self.alpha[i, 0])
                self._update_params(i, j, L, H)
                self.loss_store.append(self.loss())


            if isinstance(print_step, int) and step % print_step == 0:
                print(self.loss_store[-1])
            step += 1

        #del self.gram, self.E, self.eps

    def predict(self, x: np.ndarray = None, idx : int = None) -> np.ndarray:
        if idx is not None:
            return np.sum(self.w * self.gram[:, idx]) + self.b
        else:
            return np.sum(self.w * self.kernel_map(self.train_x, x).reshape(-1,1)) + self.b

    def loss(self):
        return self.w.T @ self.gram @ self.w / 2 - np.sum(self.alpha)

    def model_info(self):
        print('=' * 30, 'MODEL INFO', '=' * 30)
        name = []
        value = []
        name.extend(['is_fitted', 'is_trained','number of features','number of samples'])
        value.extend([self.is_fitted, self.is_trained, self.num_features, self.num_samples])
        for key in self.hyperparams:
            name.append(key)
            value.append(self.hyperparams[key])
        utils.pair_print(name, value)
        print('=' * 66, end = '\n\n')

    def _check_KKT(self, i : int) -> bool:
        # gradient of w and b is always 0;
        # constraint on alpha is always satisfied;
        # original constraint can always be satisfied be adjusting xi;
        # it remains to check complementary slackness;

        t = self.train_y[i] * self.predict(idx = i)
        if np.allclose(self.alpha[i], 0, atol = self.eps):
            return t >= 1
        elif np.allclose(self.alpha[i], self.C, atol = self.eps):
            return t <= 1
        else:
            return np.allclose(t, 1, atol = self.eps)

    def _set_w_b(self) -> bool:
        self.w = self.alpha * self.train_y # the kernel part is neglected
        self.b = np.zeros([1,1])
        cnt = 0
        for i in range(self.num_samples):
            if self._is_support_vector(i):
                self.b += self.train_y[i] - np.sum(self.w * self.gram[:, i])
                cnt += 1
        if cnt > 0:
            self.b /= cnt

    def _build_gram(self):
        self.gram = np.zeros([self.num_samples, self.num_samples])
        for i in range(0, self.num_samples):
                self.gram[:,i] = self.kernel_map(self.train_x, self.train_x[i,:])

    def _update_E(self):
        for idx in range(self.num_samples):
            self.E[idx] = float(self.predict(idx = idx) - self.train_y[idx])

    def _update_params(self, i:int, j:int, L, H):
        alpha_old = self.alpha[j]
        self.alpha[j] = np.clip(
            self.alpha[j] + \
            self.train_y[j] * (self.E[i] - self.E[j]) / (self.gram[i, i] + self.gram[j, j] - 2 * self.gram[i, j]),
            L,
            H
        )
        self.alpha[i] = self.alpha[i] + self.train_y[i] * self.train_y[j] * (alpha_old - self.alpha[j])
        self._set_w_b()
        self._update_E()

    def _w(self): # for LINEAR kernel only
        return np.sum(self.w * self.train_x, axis = 0)

    def _is_support_vector(self, idx:int) -> bool:
        return (self.eps < self.alpha[idx,0]) and (self.alpha[idx, 0] < self.C - self.eps)

if __name__ == '__main__':
    svc = SVC(C = 1, kernel='LINEAR')
    x = np.array([
        [0, 2],
        [1, 3],
        [0, 3],
        [1, 4],
        [-1, 2],

        [1, 1],
        [2, 2],
        [1, 0],
        [2, 1],
        [3, 2]
    ])

    y = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1]).reshape([10,1])
    svc.fit(x,y)
    svc.model_info()
    svc.train(max_step=1000, print_step=10)
    w = svc._w()
    b = svc.b
    print("alpha: ", svc.alpha.T)
    print("KKT: ", end = '')
    for i in range(10):
        print(svc._check_KKT(i), end = ' ')

    plt.figure()
    for i in range(10):
        plt.scatter(x[i][0], x[i][1], marker = 'x' if i in [0,1,5,6] else 'o',c = 'r' if y[i][0] == 1 else 'b')

    xs = np.linspace(-0,1,3)
    ys = -(w[0] * xs + b)/w[1]
    plt.plot(xs, ys.reshape(-1))
    plt.show()
