import numpy as np
def SFS(X, Y, tester, num_attr = None, print_step = True):
    '''
    implement a sequential forward search algorithm
    
    params:
    X: training instances, of shape (num_instances, num_features)
    Y: training labels, of shape (num_instances,)
    num_attr: number of features needed
    print_step: whehter to print current subsets and scores during search. Default True.
    '''
    X = np.asarray(X)
    if num_attr is None:
        num_attr = X.shape[1]
    all_attr = list(range(X.shape[1]))
    sub_attr = []
    
    current_best_score = 0
    for num in range(1, num_attr+1): # 迭代次数
        current_best_attr = None
        for attr in all_attr: # 遍历剩下的特征
            _X = X[:,sub_attr + [attr]]
            score = tester(_X,Y) # 得到结果
            if score > current_best_score: # 如果比现有的好
                current_best_score = score
                current_best_attr = attr
        # 如果没有更好的结果, 就放弃吧
        if current_best_attr is None:
            return sub_attr
        else:
            sub_attr.append(current_best_attr)
            all_attr.remove(current_best_attr)
            if print_step:
                print('num_attr = {}, best_score = {}, best_sub_attr = {}'.format(num, current_best_score, sub_attr))

    return sub_attr

def SBS(X, Y, tester, num_attr, print_step = True):
    '''
    implement a sequential backward search algorithm
    
    params:
    X: training instances, of shape (num_instances, num_features)
    Y: training labels, of shape (num_instances,)
    num_attr: number of features needed
    print_step: whehter to print current subsets and scores during search. Default True.
    '''
    X = np.asarray(X)
    all_attr = list(range(X.shape[1]))
    
    current_best_score = 0
    for num in range(0, X.shape[1]-num_attr): # 迭代次数
        current_best_score = 0
        sub_attr = all_attr[:]
        for attr in all_attr: # 遍历剩下的特征
            sub_attr.remove(attr)
            _X = X[:,sub_attr]
            score = tester(_X,Y) # 得到结果
            if score > current_best_score: # 如果比现有的好
                current_best_score = score
                current_best_attr = attr
            sub_attr.append(attr)
            
        all_attr.remove(current_best_attr)
        if print_step:
            print('num_attr = {}, best_score = {}, best_sub_attr = {}'.format(X.shape[1]-num-1, current_best_score, all_attr))
    return all_attr