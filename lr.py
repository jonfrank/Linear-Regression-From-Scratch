from cgi import test
from unicodedata import name
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

class Lrfs:
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def mse_loss(self, ypred, ytrue):
        pass

    def _update_params(self, new_w, new_b):
        pass

    def _calc_deriv(self, X, ypred, ytrue):
        pass

    def _step(self, w, b, X, ypred, ytrue):
        pass

if __name__ == '__main__':
    lr = Lrfs()
    X, y = fetch_openml(name="house_prices", return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(y_test)
