import numpy as np

class AdalineGD:

    def __init__(self, lr=0.01, n_iter=50, random_state=1):
        '''
        ADAptive LInear NEuron classifier.

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

        losses_ : list
            MSE loss values
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

        self.losses_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            self.w_ += self.lr*2.0*X.T.dot(errors) / X.shape[0]
            self.b_ += self.lr*2.0*errors.mean()

            loss  = (errors**2).mean()
            self.losses_.append(loss)
            
        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def activation(self, X):
        return X
    
    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)