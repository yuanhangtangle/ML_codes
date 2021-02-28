import numpy as np
import matplotlib.pyplot as plt

def jump(thresh):
    '''
    jump function. Returns 1 if x > thresh else 0.
    '''
    def _jp(x):
        return 1 if x > thresh else 0

    return np.vectorize(_jp)

def logistic(x):
    return 1/(1 + np.exp(-x))

def softmax(x, all = False, byCol = True):
    t = np.exp(x)
    p = t/(1 + np.sum(t))
    if all:
        return np.c_[p, 1 - np.sum(p, axis = 1, keepdims=True)] if byCol \
            else np.r_[p, 1 - np.sum(p, axis = 0, keepdims=True)]
    else:
        return p

def accuracy_score(y_true,y_pred):
    n = len(y_true)
    return np.sum(y_true == y_pred)/n

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

def Minkowski_distance(ord = '2'):
    def _minkow(x, y):
        return np.linalg.norm(x-y, ord=ord)
    return _minkow

def min_set_dist(X, Y, ord = '2'):
    '''
    compute the minimal distance of two sets.
    parem X, Y: two sets whose rows represent the elements in that set
    dist: measure, Minkowski distance. Specially, 2 for Euclidean distance, 1 for Manhattan distance.
    '''
    X = np.asarray(X, dtype=np.float)
    Y = np.asarray(Y, dtype=np.float)
    m = X.shape[0], n = Y.shape[0]
    dist = Minkowski_distance(ord)
    minDist = 0x7fffffff # well... C++ style?
    for xrow in X:
        for yrow in Y:
            minDist = min(dist(xrow,yrow), minDist)
    return minDist

def max_set_dist(X, Y, ord = '2'):
    '''
    compute the maximal distance of two sets.
    parem X, Y: two sets whose rows represent the elements in that set
    dist: measure, Minkowski distance. Specially, 2 for Euclidean distance, 1 for Manhattan distance.
    '''
    X = np.asarray(X, dtype=np.float)
    Y = np.asarray(Y, dtype=np.float)
    m = X.shape[0], n = Y.shape[0]
    dist = Minkowski_distance(ord)
    minDist = 0
    for xrow in X:
        for yrow in Y:
            minDist = max(dist(xrow,yrow), minDist)
    return minDist

def avg_set_dist(X, Y, ord = '2'):
    '''
    compute the average distance of two sets.
    parem X, Y: two sets whose rows represent the elements in that set
    dist: measure, Minkowski distance. Specially, 2 for Euclidean distance, 1 for Manhattan distance.
    '''
    X = np.asarray(X, dtype=np.float)
    Y = np.asarray(Y, dtype=np.float)
    m = X.shape[0]
    n = Y.shape[0]
    dist = Minkowski_distance(ord)
    minDist = 0.
    for xrow in X:
        for yrow in Y:
            minDist += dist(xrow,yrow)
    return minDist/m/n

def _min_set_dist(ord):
    def _func(x, y):
        return min_set_dist(x,y,ord)
    return _func

def _max_set_dist(ord):
    def _func(x, y):
        return max_set_dist(x,y,ord)
    return _func

def _avg_set_dist(ord):
    def _func(x, y):
        return avg_set_dist(x,y,ord)
    return _func

def _set_dist(ord, dist):
    if dist == 'min':
        return _min_set_dist(ord)
    elif dist == 'max':
        return _max_set_dist(ord)
    elif dist == 'avg':
        return _avg_set_dist(ord)
    else:
        raise Exception("Unknown Set Distance Measure. Choose From [min, max, avg]!")

def circle(x,y,r, c = 'b'):
    theta = np.linspace(-np.pi, np.pi, 100)
    xs = x + np.cos(theta)
    ys = y + np.sin(theta)
    plt.plot(xs, ys, color = c)
