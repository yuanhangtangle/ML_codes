Each class contains the following variable:
- ``hyperparams``: a dict containing the hyperparemeters of the learner. I
  may contains some or all of the following variables:
  - ``weights``: ``np.array``, weight of each training sample
  - ``C``: float, coefficient of the regularity term
  - ``penalty``: str, kind of penalty/regularity, e.g. L1, L2, ``None``
  - ``beta_momen``: as it says
  - ``beta_RMS``: as it says
  - ``eps``: used in RMSprop of Adam to avoid dividion by zero
  - ``learning_rate``: as it says
    
- ``params``: a dict of parameters of the learner
- ``train_x``: input training samples, with shape(num_samples, num_features);
- ``train_y``: output training samples, with shape(num_samples, 1);

and the following function:
- ``loss``: return the loss EXCLUDING the regularity term;
- ``penalty_loss``: return the loss INCLUDING the regularty term;
- ``predict``: as it says;


