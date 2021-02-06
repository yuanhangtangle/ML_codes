import numpy as np
import ML_algo.utils as utils

class optimizer:
    """
    This class implements several classic gradient optimization method. See ``opimizer.available_optimizers`` for a list
    of available optimizers.

    params:
    ``SGD``:        Stochastic gradient descent;
    ``momentum``:   momentum optimizer;
    ``RMSprop``:    as it says;
    ``SSGD``:       subgradient stochstic gradient descent. This is not widely used due to slow convergence. The actual
                    learning rate for epoch k is ``leanring_rate/k``.
                    See https://www.stat.cmu.edu/~ryantibs/convexopt/lectures/sg-method.pdf for details.
    """
    available_optimizers = ['SGD', 'Momentum', 'RMSprop','Adam', 'SSGD','PGD']
    update_methods = {
        'SGD': utils.update_gradient,
        'SSGD': utils.update_subgradient,
        'Momentum': utils.update_momentum,
        'RMSprop': utils.update_RMS,
        'PGD': utils.update_proxgradient,
        'Adam':utils.update_adam
    }

    def __init__(self):
        pass

    def GD(
            self,
            obj,
            samples,
            gradient,
            method,
            print_step=False,
            info_print=None,
    ):
        """
        Implement stochastic gradient descent algorithm to optimize a object

        params:
        ``obj``: the object to optimize, must be a instance of a class in which variables given in ``params``,``samples``
                and ``gradient`` must be defined.
        ``samples``: list of str's, names of training samples including training x and training y.
        ``gradient``: str, name of the function defined in ``obj`` that provides the gradient when given training x and
                training y. It must return gradients in a dictionary in the form of ``param_names: gradient``.
                other parameters can be understood literally.
        ``print_step``: False or int. If int, training ``info`` must be given to print information.
        ``beta_momen``: weight for accumulated momentum, 1 - ``beta_momen`` is the weight for new gradient.
        ``beta_RMS``: weight for accumulated RMS, 1 - ``beta_RMS`` is the weight for new gradient.
        ``eps_RMS``: used to make sure of numerical stabilit when the RMS is divided, default 1e-8.

        NOTE: obj.__getattribute__(name) is used for all the names in ``params``,``samples`` and ``gradient``. MAKE SURE
                THEY ARE DEFINED IN ``obj``!
        """

        if type(print_step) == int and info_print is None:
            raise Exception('``INFO`` MUST BE GIVEN WHEN ``PRINT_STEP`` IS INT!')

        train_x = samples[0]
        train_y = samples[1]
        num_samples = obj.__getattribute__(train_x).shape[0]
        num_features = obj.__getattribute__(train_x).shape[1]
        loss = [obj.__getattribute__('penalty_loss')()] # current loss
        # ``old`` stores old information
        old = dict()

        if method == 'PGD':
            old = obj.params # use the current params
        elif method == 'Adam':
            for param in obj.params.keys():
                old[param] = [0,0] # [momen, rms]
        else: # use the first
            for param in obj.params.keys():
                old[param] = 0

        # generate the updater for accumulated information and gradient
        # learning rate may be adjusted in each epoch
        updater = optimizer.update_methods[method]( # a kwgs will take these arguments
            beta_momen=obj.hyperparams['beta_momen'],
            beta_RMS=obj.hyperparams['beta_RMS'],
            eps=obj.hyperparams['eps_RMS'],
            learning_rate = obj.hyperparams['learning_rate'],
            C = obj.hyperparams['C']
        )
        for step in range(1, obj.hyperparams['max_epoch']+1):
            random_index = np.arange(num_samples)
            np.random.shuffle(random_index)
            i = 0
            while i < num_samples:
                ranidx = random_index[i:i + obj.hyperparams['batch_size']]
                n = len(ranidx)
                batch_x = obj.__getattribute__(train_x)[ranidx].reshape(n, num_features)
                batch_y = obj.__getattribute__(train_y)[ranidx].reshape(n, 1)
                batch_weight = obj.hyperparams['weights'][ranidx].reshape(n, 1)
                # new stores the gradient at first
                new = obj.__getattribute__(gradient)(
                    obj.hyperparams['learning_rate'],
                    x=batch_x,
                    y=batch_y,
                    weights=batch_weight
                )
                old, new = updater(
                    old, new,
                    learning_rate = obj.hyperparams['learning_rate'],
                    update_step = step
                )
                # new stores the "modified gradient", e.g., momentum.
                if method == 'PGD':
                    obj.params = new
                else:
                    for param in new:
                        obj.params[param] -= new[param]
                i += obj.hyperparams['batch_size']
            loss.append(obj.__getattribute__('penalty_loss')()) # store the loss each epoch
            if type(print_step) == int and step % print_step == 0:
                print('step: ', step, end = '  ')
                info_print()
        return loss