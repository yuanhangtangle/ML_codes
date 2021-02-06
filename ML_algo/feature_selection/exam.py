from relief import relief
from heuristic_search import SBS, SFS
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np


def svc_tester(X,Y):
    svc = SVC(gamma = 'auto')
    paramGrid = {
        'class_weight':[None, 'balanced'],
        'kernel': ['linear']
    }

    gs = GridSearchCV(
        cv = 10,
        estimator=svc,
        param_grid=paramGrid,
        n_jobs = -1
    )
    _ = gs.fit(X,Y)
    return gs.best_score_

if __name__ == '__main__':
    wine56X = np.asarray(pd.read_csv('wine56X.csv'))
    wine56Y = np.asarray(pd.read_csv('wine56Y.csv')).reshape(-1)
    
    relief_attrs = relief(wine56X,wine56Y)
    best_sub_attr = []
    for i in range(9):
        best_sub_attr.append(relief_attrs[i])
        print('num_attrs = {}, score = {}, best_sub_attrs = {}'.\
            format(i+1, svc_tester(wine56X[:,best_sub_attr], wine56Y), best_sub_attr))
    SFS(wine56X, wine56Y, tester = svc_tester)






