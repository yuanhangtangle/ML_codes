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
        def _func(self, x=None, y=None):
            grad = func(self, self.train_x, self.train_y)
            if self.penalty in available_penalty:# same penalty
                if self.penalty == 'L1':
                    for param in grad:
                        grad[param] += self.hyperparams['C'] * np.sign(self.params[param])
                elif self.penalty == 'L2':
                    for param in grad:
                        grad[param] += 2 * self.hyperparams['C'] * self.params[param]
            else: # different parameters with different penalty
                for param in grad:
                    if self.penalty[param] == 'L1':
                        grad[param] += self.hyperparams['C'] * np.sign(self.params[param])
                    elif self.penalty == 'L2':
                        grad[param] += 2 * self.hyperparams['C'] * self.params[param]
            return grad
        return _func

    @grad_wrapper
    def gradient(self, x=None, y=None):
        pass

    def fit(self, x, y, weights = None):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.num_features = x.shape[1] # Excluding constant
        self.num_samples = x.shape[0]
        self.num_classes = len(np.unique(y))

        if weights is not None:
            try:
                weights = np.asarray(weights, dtype=np.float32).reshape(self.num_samples, 1)
            except:
                raise Exception("Lenght Of ``weights`` Must Be The Same As The Number Of Training Samples")
            if np.sum(self.hyperparams['weights'] <= 0) > 0:
                raise Exception("Weights Should Not Contain Negative Elements Or Zeros")
        else:
            weights = np.asarray([1 / self.num_samples for i in range(self.num_samples)], dtype=np.float32).reshape(self.num_samples, 1)
        self.hyperparams['weights'] = weights
        self.train_x = np.c_[np.ones([self.num_samples, 1]), x]
        self.train_y = y.reshape([self.num_samples, 1])
        self.is_fitted = True

    def parameter_initializer(self, param=None,):
        '''
        Initialize parameters.
        '''
        pass

    def gradient_initializer(
            self,
            max_epoch=5000,
            min_epoch=10,
            optimizer='SGD',
            batch_size=None,
            learning_rate=0.001,
            beta_momen=0.9,
            beta_RMS=0.999,
            eps_RMS=1e-8,
            tol = None
    ):
        '''
        set hyper-parameters for gradient optimizer
        '''
        if self.penalty == 'L1' and optimizer not in ['SSGD','PGD']:
            optimizer = 'PGD'
            print("OPTIMIZER SWITCHED TO PROXIMAL GRADIENT DESCENT SINCE L1 LOSS IS CHOSEN")
        if batch_size is None:
            batch_size = 64 if self.num_samples > 64 else self.num_samples
        self.optimizer = optimizer
        self.hyperparams['learning_rate'] = learning_rate
        self.hyperparams['batch_size'] = batch_size
        self.hyperparams['max_epoch'] = max_epoch
        self.hyperparams['beta_momen'] = beta_momen
        self.hyperparams['beta_RMS'] = beta_RMS
        self.hyperparams['eps_RMS'] = eps_RMS
        self.hyperparams['min_epoch'] = min_epoch
        self.hyperparams['tol'] = tol

    def train(
            self,
            print_step = False,
    ):
        info_print = utils.info_print(self, ['loss'])
        loss_store = GradOpt.GD(
            self,
            ['train_x', 'train_y'],
            'gradient',
            self.optimizer,
            print_step=print_step,
            info_print=info_print
        )

        self.is_trained = True
        return loss_store

    def predict(self, x, add_constant = True):
        pass

    def loss_wrapper(func):
        def _func(self, include_penalty = True):
            loss = func(self, include_penalty)
            if include_penalty:
                if self.penalty in available_penalty:# same penalty
                    if self.penalty == 'L1':
                        for param in self.params:
                            loss += np.sum(np.abs(self.params[param])) * self.hyperparams['C']
                    elif self.penalty == 'L2':
                        for param in self.params:
                            loss += np.sum(self.params[param] * self.params[param]) * self.hyperparams['C']
                else: # different parameters with different penalty
                    for param in self.params:
                        if self.penalty[param] == 'L1':
                            loss += np.sum(np.abs(self.params[param])) * self.hyperparams['C']
                        elif self.penalty == 'L2':
                            loss += np.sum(self.params[param] * self.params[param]) * self.hyperparams['C']
            return loss

        return _func

    def loss(self, include_penalty = True):
        pass

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

