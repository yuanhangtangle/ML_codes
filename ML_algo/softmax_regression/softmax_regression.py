import ML_algo.gradient_model
from sklearn import datasets
from sklearn.preprocessing import  MinMaxScaler
import ML_algo.utils as utils
import numpy as np
from ML_algo.gradient_model import Model

class Softmax_Regression(Model):
    def __init__(
            self,
            C=0,
            penalty=None,
    ):
        super().__init__(C,penalty)
        self.params = {'beta': None}

    def parameter_initializer(self, beta = None):
        if not self.is_fitted:
            raise Exception("THE MODEL NEEDS TO BE FITTED BEFORE ITS PARAMETERSARE INITIALIZED!")
        if beta is None:
            self.params['beta'] = np.random.randn(self.num_features + 1, self.num_classes-1)
        else:
            self.params['beta'] = np.asarray(beta, dtype = np.float)


    @Model.grad_wrapper
    def gradient(self, x, y):
        beta = 0
        num = x.shape[0]
        for i in range(num):
            beta += x[i].reshape(self.num_features+1,1) @ utils.softmax(x[i].reshape(1,self.num_features+1) @ self.params['beta'])
            if y[i,0] < self.num_classes - 1:
                beta[:, int(y[i,0])] -= x[i]
        return {'beta': beta}

    '''
        @Model.grad_wrapper
        def gradient(self, x, y):
            beta = x.T @ utils.softmax(x @ self.params['beta'])
            num = x.shape[0]
            for i in range(num):
                if y[i, 0] < self.num_classes - 1:
                    beta[:, int(y[i, 0])] -= x[i]
            return {'beta': beta}
    '''

    @Model.loss_wrapper
    def loss(self, include_penalty = True):
        l = 0
        for i in range(self.num_samples):
            l += np.log(
                utils.softmax(
                    self.train_x[i,:].reshape(1,self.num_features+1) @ self.params['beta'],
                    all=True
                )[0, int(self.train_y[i,0])]
            )

        return -l

    def predict(self, x, add_constant = True, thresh = 0):
        if add_constant:
            x = np.c_[np.ones([self.num_samples, 1]), x]
        p = utils.softmax(x @ self.params['beta'], all=True)
        return np.argmax(p, axis=1)

if __name__ == '__main__':
    iris = datasets.load_iris()
    x = iris.data
    x = MinMaxScaler().fit_transform(x)
    y = iris.target.reshape(150,1)

    lr = Softmax_Regression()
    lr.fit(x,y)
    lr.parameter_initializer()
    lr.gradient_initializer(tol = False, max_epoch = 500,optimizer='Momentum')
    lr.train(print_step=100)


    y_pred = lr.predict(x)
    print(lr.params['beta'])
    #print(y)
    print(utils.accuracy_score(y_true=y,y_pred=y_pred))