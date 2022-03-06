import os, pandas as pd, numpy as np
import matplotlib.pyplot as plt

from perceptron import Perceptron
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

ppn = Perceptron(lr=0.1, n_iter=10)
ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_)+1),
    ppn.errors_, marker='o'
)
plt.xlabel('Epoch')
plt.ylabel('No. of updates')
plt.show()



plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.show()


