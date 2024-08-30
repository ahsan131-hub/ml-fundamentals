from os import error
import numpy as np


class Perceptron:
    """
        perceptron classsifier

        Parameters
        ----------
        eta : float
            learning rate (between 0.0 and 1.0)
        n_iter : int
            passes over the training dataset
        random_state : int
            random number generator seed for random weight initialization

    """

    w_ = []
    b_ = 0
    errors_ = []

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """
            Fit training data

            Parameters
            ----------
            X : {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number of samples and n_features is the number of features
            y : array-like, shape = [n_samples]
                Target values

            Returns
            -------
            self : object

        """

        rgen = np.RandomState(self.random_state)
        self.w_=rgen.normal(loc=0.0,scale=0.01,size=X.shape[1])
        self.b_=np.float_(0.)
        self.errors_=[]



        for _ in range(self.n_iter):
            errors=0
            for xi,target in zip(X,y):
                update = self.eta*(target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors  += int(update!=0.0)
            self.errors_.append(errors)

        return self
    

    def net_input(self,X):
        """ Calculate net input """
        return np.dot(X,self.w_) +self.b_
    
    def predict(self,X):
        """Return class label after unit step """
        return np.where(self.net_input(X)>=0.0,1,0)

    