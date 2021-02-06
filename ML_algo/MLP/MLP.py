import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit  #the sigmoid function

def sigmoid_deriv(x):
    '''
    derivative of the sigmoid function
    '''
    y = expit(x)
    return y*(1-y)

def sigmoid_reverse(x):
    return -np.log(1/x - 1)

available_activation_functions = {'tanh': np.tanh,
                                   'relu': np.vectorize(lambda x: max(x,0)),
                                   'linear': (lambda x: x),
                                  'sigmoid':expit}

available_activation_gradient = {'tanh':lambda x: 1- np.tanh(x)**2,
                                  'relu':np.vectorize(lambda x: int(x>0)),
                                 'linear': (lambda x:1),
                                 'sigmoid': sigmoid_deriv}

class Layer():
    '''
    this class defines a layer in a MLP
    '''
    def __init__(self,n, activation = 'tanh', W = None, b = None, initializer = 'xavier'):
        self.W = W #weight matrix
        self.b = b #bias 
        self.activation = activation #activation function
        self.initializer = initializer #method to initiate the weight and the bias
        self.num_units = n # number of unit, must be the same as the number of columns of W

    def prop(self,A_previous_layer):
        '''
        compute the output of this layer and store some cache
        A_previous_layer : the ouput of last layer. See MLP.forward_propagation().
        '''
        self.Z = (self.W @ A_previous_layer + self.b).reshape(
            self.num_units,
            A_previous_layer.shape[1]
            )# reshape to keep the shape
        self.A = (available_activation_functions[self.activation](self.Z)).reshape(
            self.num_units,
            A_previous_layer.shape[1]
            )
        
    def back_prop(self, A_previous_layer, W_next_layer, dZ_next_layer):
        '''
        compute the gradient of the parameters and store some cache. dX represents the gradient 
        of the loss function with respect to X.
        '''
        self.dA = np.transpose(W_next_layer) @ dZ_next_layer
        self.dZ = self.dA * available_activation_gradient[self.activation](self.Z)
        self.dW = self.dZ @ np.transpose(A_previous_layer)
        self.db = np.sum(self.dZ, axis = 1, keepdims= True)

    def update_parameter(self, learning_rate = 0.001):
        '''
        gradient descent
        '''
        self.W = self.W - learning_rate*self.dW
        self.b = self.b - learning_rate*self.db

    def clear_cache(self):
        self.A = 0
        self.Z = 0
        self.dW = 0
        self.dZ = 0
        self.db = 0
        self.dA = 0

class MLP():
    '''
    Note that the first layer in this MLP class is the input layer whose number of units must be 
    consisten with the shape of the input
    '''
    def __init__(self, layer_unit_option = None, layers = None, loss_metric = 'mse'):
        '''
        layer_unit_option: list of tuples with shape (,2), with each tuple in form of (number of units, activation)
        '''
        if layers is None and layer_unit_option is None:
            raise Exception('structure of the MLP must be defined by ``layers`` or ``layer_units_option``')
        elif layers is not None: 
            #you may define each layer by yourself
            self.layers = layers
        else:
            #or simply put in the options
            self.layers = []
            for (n,activation) in layer_unit_option:
                self.layers.append(Layer(n, activation = activation))

        self.num_layers = len(self.layers)
        self.loss_metric = loss_metric
        self.fitted = False
        self.training_x = None #training instances
        self.training_y = None #labels
        self.num_instance = None #number of instance
        self.num_features = None

        for i in range(self.num_layers):
            if i == 0:
                n_previous = self.layers[i].num_units
            else:
                if self.layers[i].initializer == 'xavier':
                    self.layers[i].W = np.random.randn(self.layers[i].num_units, n_previous) / np.sqrt(n_previous)
                    self.layers[i].b = np.random.randn(self.layers[i].num_units, 1) / np.sqrt(n_previous)
                elif self.layers[i].initializer == 'He':
                    self.layers[i].W = np.random.randn(self.layers[i].num_units, n_previous) / np.sqrt(n_previous / 2)
                    self.layers[i].W = np.random.randn(self.layers[i].num_units, 1) / np.sqrt(n_previous / 2)
                n_previous = self.layers[i].num_units

    def fit(self,x,y):
        '''
        each column of X represents a training instance
        '''
        self.training_x = x
        self.training_y = y
        self.fitted = True
        self.num_instance = x.shape[1]
        self.num_features = x.shape[0]
        
    def forward_propagation(self, x):
        for i in range(self.num_layers):
            if i == 0: #the first layer is simply the input layer and does NOT perform any computation
                self.layers[i].A = x
            else:
                self.layers[i].prop(self.layers[i-1].A)

    def backward_propagation(self, y = None):
        '''
        compute the gradient of the loss function with respect to the parameters
        '''
        if y is None: y = self.training_y
        for i in range(self.num_layers - 1, 0, -1): #start from the last layer, i.e. the output layer
            if i == self.num_layers - 1: #the gradient 
                if self.loss_metric == 'mse':
                    self.layers[i].dA = (self.layers[i].A - y)
                elif self.loss_metric == 'cross_entropy':
                    pass  #remain to be writen

                self.layers[i].dZ = self.layers[i].dA * \
                                    available_activation_gradient[self.layers[i].activation](self.layers[i].Z)
                self.layers[i].dW = self.layers[i].dZ @ np.transpose(self.layers[i-1].A)
                self.layers[i].db = np.sum(self.layers[i].dZ, axis=1, keepdims = True)
            else:
                self.layers[i].back_prop(self.layers[i-1].A, self.layers[i+1].W, self.layers[i+1].dZ)

    def error(self, x=None):
        if x is None: #return the training error by default
            x = self.training_x
        self.forward_propagation(x)
        if self.loss_metric == 'mse':
            return np.sum((self.training_y - self.layers[-1].A) ** 2) / (2 * x.shape[1])
        elif self.loss_metric == 'mse_l2_regularization':
            pass #remain

    def predict(self,x,regression_transform = None):
        if regression_transform is not None: #experimental, transform the data by the sigmoid function
            x = expit(x)
        self.forward_propagation(x)
        if regression_transform is None:
            return self.layers[-1].A
        else:
            return sigmoid_reverse(self.layers[-1].A)

    def stachastic_gradient_descent(
            self, 
            learning_rate = 0.001, 
            steps = 1000, 
            print_loss = True,
            plot_loss = False, 
            learning_rate_decay = None, 
            seed = 1234, 
            regression_transform = None
        ):
        training_error = [self.error()]
        for s in range(steps):
            random_index = np.arange(0,self.num_instance,1)
            np.random.shuffle(random_index)
            for i in random_index:
                x = self.training_x[:,i].reshape(self.num_features,1)
                y = self.training_y[0,i]
                if regression_transform is not None: #try sigmoid first
                    x = expit(x)
                    y = expit(y)

                self.forward_propagation(x)
                self.backward_propagation(y)
                for layer in self.layers[1:]:
                    layer.update_parameter(learning_rate = learning_rate)
                print('\rstep:{}\t[{}%]'.format(s, i*100/self.num_instance), end = '')

            training_error.append(self.error())
            if s % 10 == 0:
                if print_loss:
                    print('step:{}\t training_error:{}'.format(s, training_error[-1]))
                if plot_loss:
                    plt.cla()
                    plt.semilogy(range(len(training_error)), training_error)
                    plt.pause(0.1)

    def mini_batch_gradient_descent(
            self, 
            learning_rate = 0.001, 
            steps = 1000, 
            print_loss = True, 
            batch_size = 16,
            plot_loss = False, 
            learning_rate_decay = None, 
            seed = 1234, 
            regression_transform = None
        ):
        training_error = []
        for s in range(steps):
            random_index = np.arange(0, self.num_instance, 1)
            np.random.shuffle(random_index)
            for i in range(0,self.num_instance,batch_size):
                size_current_batch = self.num_instance - i if i + batch_size >= self.num_instance else batch_size
                x = self.training_x[:, i:i+batch_size].reshape(self.num_features,size_current_batch)
                y = self.training_y[0, i:i + batch_size].reshape(1,size_current_batch)
                if regression_transform is not None: #try sigmoid first
                    x = expit(x)
                    y = expit(y)
                self.forward_propagation(x) ##shape of batch
                self.backward_propagation(y)
                for layer in self.layers[1:]:
                    layer.update_parameter(learning_rate=learning_rate)
                print('\rstep:{}\t[{}%]'.format(s, i * 100 / self.num_instance), end='')

            training_error.append(self.error())
            if s % 10 == 0:
                if print_loss:
                    print('training_error:{}'.format(training_error[-1]))
                if plot_loss:
                    plt.cla()
                    plt.semilogy(range(len(training_error)), training_error)
                    plt.pause(0.1)

    def info(self):
        '''
        print the structure of the MLP
        '''
        print('layer'.ljust(15, ' '), 'size'.ljust(15, ' '), 'activation'.ljust(15,' '), 'Weight matrix'.ljust(15,' '),
              'bias'.ljust(15,' '))
        print('-' * 79)
        for i in range(self.num_layers):
            print(str(i).ljust(15,' '), str(self.layers[i].num_units).ljust(15,' '), self.layers[i].activation.ljust(15,' '),
                  str(self.layers[i].W.shape).ljust(15,' ') if i else 'None'.ljust(15,' '),
                  str(self.layers[i].b.shape).ljust(15,' ') if i else 'None'.ljust(15,' '))
            print('-'*79)

if __name__ == '__main__':
    #we implement a MLP to fit the *sine* function
    myMLP = MLP(layer_unit_option = [(1, 'linear'),
                                     (10, 'relu'),
                                     (10, 'relu'),
                                     (10, 'tanh'),
                                     (10, 'tanh'),
                                     (1, 'linear')],
                loss_metric='mse')
    x = np.linspace(-5,5,50).reshape(1, 50)
    y = np.sin(x)
    epoch = 100
    #myMLP.info()
    myMLP.fit(x,y)
    y_pred = []
    plt.pause(5)
    for i in range(epoch):
        myMLP.mini_batch_gradient_descent(steps = 100, learning_rate= 0.001)
        y_pred.append(myMLP.predict(x).reshape(-1))
        #plt.cla()
        #plt.pause(0.1)

    plt.plot(x.reshape(-1),y.reshape(-1))
    plt.plot(x.reshape(-1), y_pred[-1])
    plt.savefig('./temp.jpg', dpi = 100)

