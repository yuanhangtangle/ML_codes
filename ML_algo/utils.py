import numpy as np

def soft_thresholding(lamda):
    '''
    The classic soft thresholding operator for scalar.

    params:
    ``lamda``: threshold;
    ``x``: scalar.

    return:
    a function implementing the soft thresholding operator with the given threshold.

    '''
    if lamda <= 0:
        raise Exception('The Threshold Must Be Positive')

    def _st(x):
        return x - lamda if x > lamda else (x + lamda if x < -lamda else 0)
    return np.vectorize(_st)

def update_momentum(**kwargs):
    if 'beta_momen' not in kwargs:
        raise Exception('``beta_momen`` Must Be Given!')

    beta = kwargs['beta_momen']
    if  beta <= 0:
        raise Exception('The Beta Parameter Must Be Positive')
    eta = 1 - beta

    def _up(old, new, **kwargs):
        learning_rate = kwargs['learning_rate']
        for key in old:
            old[key] = beta * old[key] + eta * new[key]
            new[key] = learning_rate * old[key]
        return old, old
    return _up

def update_RMS(**kwargs):
    if 'beta_RMS' not in kwargs:
        raise Exception('``beta_RMS`` Must Be Given!')
    if 'eps' not in kwargs:
        raise Exception('``eps`` Must Be Given!')

    beta = kwargs['beta_RMS']
    if beta <= 0:
        raise Exception('The Beta Parameter Must Be Positive')
    eta = 1 - beta
    eps = kwargs['eps']
    def _up(old, new, **kwargs):
        learning_rate = kwargs['learning_rate']
        for key in old:
            old[key] = beta * old[key] + eta * (new[key] ** 2)
            new[key] = learning_rate * new[key] / (np.sqrt(old[key]) + eps)
        return old, new
    return _up

def update_adam(**kwargs):
    if 'beta_momen' not in kwargs:
        raise Exception('``beta_momen`` Must Be Given!')
    if 'beta_RMS' not in kwargs:
        raise Exception('``beta_RMS`` Must Be Given!')
    if 'eps' not in kwargs:
        raise Exception('``eps`` Must Be Given!')

    beta_momen = kwargs['beta_momen']
    beta_RMS = kwargs['beta_RMS']
    eps = kwargs['eps']
    eta_momen = 1 - beta_momen
    eta_RMS = 1 - beta_RMS
    #beta_momen_cum = beta_momen

    def _up(old, new, **kwargs):
        learning_rate = kwargs['learning_rate']
        for key in old:
            old[key][0] = beta_momen * old[key][0] + eta_momen * new[key] # update momen
            old[key][1] = beta_RMS * old[key][1] + eta_RMS * (new[key] ** 2) # update rms
            new[key] = learning_rate * old[key][0] / ( np.sqrt(old[key][1]) + eps )
        return old, new

    return _up

def update_gradient(**kwargs):
    def _up(old, new, **kwargs):
        learning_rate = kwargs['learning_rate']
        for key in new:
            new[key] = learning_rate * new[key]
        return None, new

    return _up

def update_subgradient(**kwargs):
    def _up(old, new, **kwargs):
        learning_rate = kwargs['learning_rate']#/kwargs['update_step']
        for key in new:
            new[key] = learning_rate * new[key]
        return None, new
    return _up

def update_proxgradient(**kwargs):
    lr = kwargs['learning_rate']
    st = soft_thresholding(lr * kwargs['C'])
    def _up(old, new, **kwargs):
        for key in old:
            new[key] = st(old[key] - lr * new[key])
        return new, new
    return _up

def pair_print(name, value):
    for (a, b) in zip(name, value):
        print(str(a).ljust(20, ' '), b)

def info_print(obj, funcs = [], vars = []):
    def _info_print():
        value = []
        for key in funcs:
            value.append(obj.__getattribute__(key)())
        for key in vars:
            value.append(obj.__getattribute__(key))
        pair_print(funcs + vars, value)
    return _info_print