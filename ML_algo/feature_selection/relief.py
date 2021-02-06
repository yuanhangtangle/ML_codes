import numpy as np

def relief(X,Y, m = None):
    '''
    implement the Rellief algorithm for feature selection
    
    params:
    X: training instances, of shape (num_instances, num_features)
    Y: training labels, of shape (num_instances,)
    m: sampling size, default: number of training instances
    '''
    X = np.asarray(X)
    num_features = X.shape[1]
    num_instances = X.shape[0]
    if m is None:
        m = num_instances
        
    dist_matrix = np.linalg.norm(X[0] - X, ord = 1, axis = 1, keepdims=True)
    dist_matrix[0,0] = num_features + 1
    for i in range(1, num_instances):
        dist_matrix = np.c_[dist_matrix, np.linalg.norm(X[i] - X, ord = 1, axis = 1)]
        dist_matrix[i,i] = num_features + 1
        
    indices = list(range(0,num_instances))
    W = np.zeros(num_features)
    for _ in range(m):
        i = np.random.choice(indices)
        argmin_posi = np.argmin(dist_matrix[i] + (1-Y)*(num_instances + 1))
        w_posi = np.abs(X[i] - X[argmin_posi])**2
        argmin_nega = np.argmin(dist_matrix[i] + (Y)*(num_instances + 1))
        w_mega = np.abs(X[i] - X[argmin_nega])**2
        if Y[i]:
            W += w_mega - w_posi
        else:
            W -= w_mega - w_posi
    W = list(np.argsort(W))
    W.reverse()
    return W
