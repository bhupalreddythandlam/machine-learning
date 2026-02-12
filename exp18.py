from sklearn.linear_model import Perceptron
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
model = Perceptron().fit(X, y)
print("Perceptron Accuracy:", model.score(X, y))