from linear_regression.linear_regeression import linear_regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

lr = linear_regression()
auto = pd.read_csv('./auto_cleaned.csv')
x = np.asarray(auto.iloc[:, :-1])
y = np.asarray(auto.iloc[:,-1]).reshape(auto.shape[0],1)
max_epoch = 50
print_step = max_epoch // 10
beta = np.random.randn(x.shape[1] + 1, 1)
methods = ['Momentum','SGD','RMSprop','Adam']
x_ticks = range(max_epoch + 1)

lr = linear_regression(
    C=10,
    penalty='L2',
)
lr.fit(x, y)

for method in methods:
    loss = lr.train(
        param = beta.copy(),
        optimizer=method,
        batch_size=64,
        max_epoch=max_epoch,
        learning_rate=1e-3,
        print_step=False
    )
    plt.plot(x_ticks, loss, label = method)

plt.legend()
plt.show()