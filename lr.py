import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from aicore.ml import data

class Lrfs:
    def __init__(self, n_features, epochs, lr, batch_count=128):
        """
        set random starting points for w and b
        and parameters for epochs and learning rate
        """
        self.W = np.random.randn(n_features)
        self.b = np.random.randn()
        self.lr = lr
        self.epochs = epochs
        self.min = None
        self.range = None
        self.batch_count = batch_count

    def plot_loss(self, losses):
        """Helper plotting loss vs epoch"""
        plt.figure()
        plt.ylabel('Cost')
        plt.xlabel('Epoch')
        plt.plot(losses)
        plt.show()

    def fit(self, X, y):
        """
        work through epochs, refining w and b values
        """
        # first, normalize
        if self.min == None:
            self.min = np.min(X, axis=0)
            self.range = np.max(X, axis=0) - self.min
        X_standardized = (X - self.min) / self.range
        X = X_standardized

        X_batches = np.array_split(X, self.batch_count)
        y_batches = np.array_split(y, self.batch_count)

        all_costs = []
        for epoch in np.arange(self.epochs):
            for batch in np.arange(self.batch_count):
                predictions = self.predict(X_batches[batch])
                new_W, new_b = self._step(self.W, self.b, X_batches[batch], predictions, y_batches[batch])
                self._update_params(new_W, new_b)
            cost = self.mse_loss(self.predict(X), y)
            all_costs.append(cost)
        
        self.plot_loss(all_costs)
        print('Final cost:',cost)
        print('Weight values:',self.W)
        print('Bias values:',self.b)

    def predict(self, X):
        """
        given features, calculated predicted target
        """
        y_pred = X @ self.W + self.b
        return y_pred

    def mse_loss(self, ypred, ytrue):
        """
        calculate loss function
        """
        errors = ypred - ytrue
        squared_errors = errors ** 2
        mean_squared_error = sum(squared_errors) / len(squared_errors)
        return mean_squared_error

    def _update_params(self, new_w, new_b):
        """
        update parameter values having re-calculated during an epoch
        """
        self.W = new_w
        self.b = new_b

    def _calc_deriv(self, X, ypred, ytrue):
        """
        the interesting bit: calculate dLdb and dLdw
        """
        m = len(ytrue)
        diffs = ypred - ytrue
        dLdw = 2 * np.sum(X.T * diffs).T / m
        dLdb = 2 * np.sum(diffs) / m
        return dLdw, dLdb

    def _step(self, W, b, X, ypred, ytrue):
        """
        make a step down the gradient towards the minimum point
        """
        dLdw, dLdb = self._calc_deriv(X, ypred, ytrue)
        new_W = W - dLdw * self.lr
        new_b = b - dLdb * self.lr
        return new_W, new_b

if __name__ == '__main__':
    X, y = datasets.fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.1)
    X_train, X_validation, X_test = data.standardize_multiple(X_train, X_validation, X_test)
    lr = Lrfs(n_features=X_train.shape[1], epochs=5000, lr=0.001)
    lr.fit(X_train, y_train)
