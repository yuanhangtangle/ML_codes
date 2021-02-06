import numpy as np
import scipy.stats as stats
from ML_algo.gradient_optimizer.optimizer import optimizer as GradOpt
import ML_algo.utils as utils
GradOpt = GradOpt()
available_penalty = [None, 'L2', 'L1']
available_optimizers = GradOpt.available_optimizers

class Model():
    def __init__(
            self,
            C=0,
            penalty=None,
    ):
        '''
        penalty: str or dict. If a str is given, all parameters will be penalized by the same penalty; if a dict of
            form ``param : penalty`` is given, those not in the dict will not be penalized.

        '''
        if penalty is not None and penalty not in available_penalty:
            raise Exception('UNKNOWN PENALTY METHOD. CHOOSE FROM {}'.format(available_penalty))

        self.hyperparams = {
            'C' : C
        }
        self.penalty = penalty
        self.params = dict()
        self.is_fitted = False
        self.is_trained = False

    def grad_wrapper(func):
        def _func(self,learning_rate, x=None, y=None):
            grad = func(self, learning_rate, self.train_x, self.train_y)
            if self.penalty in available_penalty:
                if self.penalty == 'L1':
                    for param in grad:
                        grad[param] += self.hyperparams['C'] * np.sign(self.params[param])
                elif self.penalty == 'L2':
                    for param in grad:
                        grad[param] += 2 * self.hyperparams['C'] * self.params[param]
            else:
                for param in grad:
                    if self.penalty[param] == 'L1':
                        grad[param] += self.hyperparams['C'] * np.sign(self.params[param])
                    elif self.penalty == 'L2':
                        grad[param] += 2 * self.hyperparams['C'] * self.params[param]
            return grad
        return _func

    @grad_wrapper
    def gradient(self, learning_rate, x=None, y=None):
        pass

    def fit(self, x, y, weights = None):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.num_features = x.shape[1]
        self.num_samples = x.shape[0]

        if weights is not None:
            try:
                weights = np.asarray(weights).reshape(self.num_samples, 1)
            except:
                raise Exception("Lenght Of ``weights`` Must Be The Same As The Number Of Training Samples")
            if np.sum(self.hyperparams['weights'] <= 0) > 0:
                raise Exception("Weights Should Not Contain Negative Elements Or Zeros")
        else:
            weights = np.asarray([1 / self.num_samples for i in range(self.num_samples)]).reshape(self.num_samples, 1)
        self.hyperparams['weights'] = weights
        self.train_x = np.c_[np.ones([self.num_samples, 1]), x]
        self.train_y = y.reshape([self.num_samples, 1])
        self.is_fitted = True

class Linear_Regression(Model):
    def __init__(
            self,
            C=0,
            penalty=None,
    ):
        super().__init__(C,penalty)
        self.params = {'beta':None}

    def fit(self, x, y, weights = None):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.num_features = x.shape[1]
        self.num_samples = x.shape[0]

        if weights is not None:
            try:
                weights = np.asarray(weights).reshape(self.num_samples, 1)
            except:
                raise Exception("Lenght Of ``weights`` Must Be The Same As The Number Of Training Samples")
            if np.sum(self.hyperparams['weights'] <= 0) > 0:
                raise Exception("Weights Should Not Contain Negative Elements Or Zeros")
        else:
            self.hyperparams['weights'] = np.asarray([1 / self.num_samples for i in range(self.num_samples)]).reshape(
                self.num_samples, 1)

        self.train_x = np.c_[np.ones([self.num_samples, 1]), x]
        self.train_y = y.reshape([self.num_samples, 1])
        self.is_fitted = True
        self.params['beta'] = np.asarray([
            [1],[1],[1],[1]
        ])

    @Model.grad_wrapper
    def gradient(self, learning_rate, x, y):
        return {'beta':self.train_x.T @ ((x @ self.params['beta']  - self.train_y) * self.hyperparams['weights'])}


if __name__ == '__main__':
    x = [
        [1,0,0],
        [0,2,3],
        [3,4,5]
    ]
    y = [
        [2],
        [3],
        [4]
    ]
    for p in available_penalty:
        lr = Linear_Regression(C = 10, penalty=p)
        lr.fit(x,y)
        print(lr.gradient(1))
