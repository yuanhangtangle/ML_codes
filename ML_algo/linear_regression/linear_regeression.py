import numpy as np
import scipy.stats as stats
from ML_algo.gradient_optimizer.optimizer import optimizer as GradOpt
import ML_algo.utils as utils
GradOpt = GradOpt()

class linear_regression():
    available_penalty = [None, 'L2', 'L1']
    available_optimizers = GradOpt.available_optimizers

    def __init__(self,
                 C=0,
                 penalty = None,
                 ):
        if penalty is not None and penalty not in linear_regression.available_penalty:
            raise Exception('UNKNOWN PENALTY METHOD. CHOOSE FROM {}'.format(linear_regression.available_penalty))
        self.hyperparams = {
            'C' : C,
            #'weights': weights,
            #'penalty': penalty
        }
        self.penalty = penalty
        self.params = {
            'beta': None
        }
        self.is_fitted = False
        self.is_trained = False
        

    def gradient(self, leanring_rate, x = None, y = None):
        if x is None:
            x = self.train_x
            y = self.train_y
        elif x is not None and y is None:
            raise Exception('Y MUST BE PROVIDED WHEN X IS PROVIDED!')

        weights = self.hyperparams['weights']
        dic = {'beta': 0}
        # gradient of the main part
        grad = x.T @ ((x @ self.params['beta']  - y) * weights)

        if self.hyperparams['penalty'] is None:
            dic['beta'] = grad
        elif self.hyperparams['penalty'] == 'L2':
            dic['beta'] = grad + self.hyperparams['C'] * self.params['beta']
        elif self.hyperparams['penalty'] == 'L1':
            dic['beta'] = grad + self.hyperparams['C'] * np.sign(self.params['beta'])

        return dic

    def prox_gradient(self, learning_rate, x = None, y = None, weights = None):
        if x is None:
            x = self.train_x
            y = self.train_y
        elif x is not None and y is None:
            raise Exception('Y MUST BE PROVIDED WHEN X IS PROVIDED!')
        if weights is None:
            weights = self.hyperparams['weights']
        if learning_rate is None:
            learning_rate = self.hyperparams['learning_rate']

        dic = {'beta': 0}
        # gradient of the main part
        st = utils.soft_thresholding(self.hyperparams['C'] * learning_rate)
        p = self.params['beta'] - learning_rate * x.T @ ((x @ self.params['beta'] - y) * weights)
        dic['beta'] = st(p)
        return dic

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
            self.hyperparams['weights'] = np.asarray([1 / self.num_samples for i in range(self.num_samples)]).reshape(self.num_samples, 1)

        self.train_x = np.c_[np.ones([self.num_samples, 1]), x]
        self.train_y = y.reshape([self.num_samples, 1])
        self.is_fitted = True

    def train(
            self,
            param = None,
            optimizer='SGD',
            batch_size=None,
            max_epoch=None,
            learning_rate=0.001,
            beta_momen = 0.9,
            beta_RMS = 0.999,
            eps_RMS = 1e-8,
            print_step=False,
    ):
        if self.hyperparams['penalty'] == 'L1' and optimizer is None:
            raise Exception("CLOSED FORM SOLUTION IS UNAVAILAIBLE. PLEASE SELECT FROM ", linear_regression.available_optimizers)
        if self.hyperparams['penalty'] != 'L1' and optimizer in ['SSGD', 'PGD']:
            raise Exception("SSGD OR PGD ARE RECOMMENDEDED FOR L1 PENALTY ONLY! NOT AVAILABLE FOR OTHER PENALTY!")

        self.optimizer = optimizer
        if optimizer in linear_regression.available_optimizers:
            self.hyperparams['learning_rate'] = learning_rate
            self.hyperparams['batch_size'] = batch_size
            self.hyperparams['max_epoch'] = max_epoch
            self.hyperparams['beta_momen'] = beta_momen
            self.hyperparams['beta_RMS'] = beta_RMS
            self.hyperparams['eps_RMS'] = eps_RMS
        if param is None:
            self.params['beta'] = np.random.randn(self.num_features + 1, 1)
        else:
            self.params['beta'] = param
        if self.hyperparams['penalty'] == 'L1' and self.optimizer is None:
            raise Exception('Not Closed Form Solution For General L1 Regularity Linear Regression!')
        loss_store = []
        # try closed form solution
        # MAY ENCOUNTER NUMERICAL PROBLEM!
        if self.optimizer == None:
            if self.hyperparams['penalty'] is None:
                self.params['beta'] = np.linalg.inv((self.hyperparams['weights'].T * self.train_x.T) @ self.train_x) @\
                            self.train_x.T @ (self.train_y * self.hyperparams['weights'])
            elif self.hyperparams['penalty'] == 'L2':
                self.params['beta'] = np.linalg.inv((self.hyperparams['weights'].T * self.train_x.T) @ \
                            self.train_x + self.hyperparams['C'] * np.eye(self.num_features + 1)) \
                            @ self.train_x.T @ (self.train_y * self.hyperparams['weights'])
            else:
                raise Exception('No Closed Form Solution For Penalty {}'.format(self.penalty))
        # gradient method
        elif  self.optimizer in linear_regression.available_optimizers:
            #print(self.optimizer)
            info_print = utils.info_print(self, ['loss'])
            loss_store = GradOpt.GD(
                self,
                ['train_x', 'train_y'],
                'gradient',
                self.optimizer,
                print_step=print_step,
                info_print=info_print)

        return loss_store

    def predict(self, x, add_const=True):
        if add_const:
            x = np.c_[np.ones([self.num_samples, 1]), x]

        return (x @ self.params['beta'])

    def loss(self, x=None, y=None):
        if x is None:
            x = self.train_x
            y = self.train_y
        else:
            x = np.asarray(x, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32)
            x = np.c_[np.ones([self.num_samples, 1]), x]

        return np.sum(((y - self.predict(x, add_const=False)) ** 2 * self.hyperparams['weights'])) / 2

    def penalty_loss(self, ):
        loss = self.loss()
        if self.hyperparams['penalty'] == 'L2':
            return loss + self.hyperparams['C'] * np.sum(self.params['beta']**2)
        elif self.hyperparams['penalty'] == 'L1':
            return loss + self.hyperparams['C'] * np.sum(np.abs(self.params['beta']))
        elif self.hyperparams['penalty'] is None:
            return loss

    def statistic(self, alpha=0.05):
        x = self.train_x[:, 1]
        y = self.train_y
        y_pred = self.predict(self.train_x, add_const=False)

        k = self.num_features
        n = self.num_samples

        SSE = np.sum((y - y_pred) ** 2)
        SST = np.sum((y - y.mean()) * y)
        SSR = np.sum((y_pred - y_pred.mean()) ** 2)
        sigma_e = SSE / (n - k - 1)
        F_test = SSR * (n - k - 1) / SSE / k

        F_q = stats.f(k, n - k - 1).ppf(1 - alpha)
        test_result = 'NO SIGNIFICANT LINEAR DEPENDENCE!' if F_test < F_q else 'SIGNIFICANT LINEAR DEPENDENCE!'
        print_list = [
            ('k', k),
            ('n', n),
            ('SSE', SSE),
            ('SSR', SSR),
            ('SST', SST),
            ('sigma_e', sigma_e)
        ]

        if k == 1:
            Lxx = np.sum((x - x.mean()) * x)
            Lyy = SST
            Lxy = np.sum((y - y.mean()) * x)
            print_list.extend([
                ('Lxx', Lxx),
                ('Lxy', Lxy),
                ('Lyy', Lyy)
            ])

        print_list.extend([
            ('F_test', F_test),
            ('F_q', F_q),
            ('test_result', test_result)
        ])

        print('=' * 30, 'statistics', '=' * 30)
        utils.pair_print(print_list)
        print('=' * (len('statistics') + 62), end='\n\n')

    def model_info(self, ):
        print('=' * 30, 'MODEL INFO', '=' * 30)
        name = ('penalty', 'optimizer', 'C', 'params', 'samples', 'features')
        if not self.is_fitted or not self.is_trained:
            print('THIS MODEL HAS NOT BEEN FITTED OR TRAINED!')
        else:
            value = (
                self.hyperparams['penalty'],
                self.optimizer,
                self.hyperparams['C'],
                self.params['beta'].reshape(-1),
                self.num_samples,
                self.num_features
            )

            if self.optimizer in GradOpt.available_optimizers:
                name += ['learning_rate', 'beta_momen', 'beta_RMS', 'eps_RMS']
                value += [self.hyperparams[key] for key in name]
            utils.pair_print(name, value)
        print('=' * 66, end = '\n\n')