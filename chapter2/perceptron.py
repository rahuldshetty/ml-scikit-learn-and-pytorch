import numpy as np

class Perceptron:

    def __init__(self, lr=0.01, n_iter=50, random_state=1):
        '''
        Perceptron classifier.

        Parameters
        ----------
        lr: float
            Learning Rate

        n_iter: int
            No. of passover the training dataset.

        random_state: int
            Random number generator seed for random weight
            initialization.

        Attributes
        ----------
        w_ : 1d-array
            Weights after fitting
        b_ : Scalar
            Bias unit after fitting

        errors_ : list
            No. of misclassification (updates) in each epoch
        '''

        self.lr = lr
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        '''
        X: array-like, shape = [n_examples, n_features]
            Training vector
        y: array-like, shape = [n_examples]
            Target values
        '''
        rgen = np.random.RandomState(
            self.random_state
        )
        self.w_ = rgen.normal(
            loc=0.0,
            scale=0.01,
            size=X.shape[1]
        )
        self.b_ = np.float_(0.)

        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for x, target in zip(X,y):
                update = self.lr * (target - self.predict(x))
                self.w_ += update * x
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)