import os, pandas as pd, numpy as np
import matplotlib.pyplot as plt

from adalinegd import AdalineGD
from utility import plot_decision_regions

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

df = pd.read_csv(url, header=None, encoding='utf-8')

print(df.tail())

y = df.iloc[0:100, 4].values
y = np.where(y=='Iris-setosa', 0, 1)

X = df.iloc[0:100, [0,2]].values

plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='Setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='s', label='Versicolor')
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.show()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

ada1 = AdalineGD(lr=0.1, n_iter=15).fit(X, y)

ax[0].plot(range(1, len(ada1.losses_) + 1),
    np.log10(ada1.losses_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Mean squared error)')
ax[0].set_title('Adaline - Learning rate 0.1')


ada2 = AdalineGD(n_iter=15, lr=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.losses_) + 1),
    np.log10(ada2.losses_), marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('log(Mean squared error)')
ax[1].set_title('Adaline - Learning rate 0.1')
plt.show()