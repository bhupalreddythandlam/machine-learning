from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100, n_features=2, n_redundant=0)
model = LogisticRegression().fit(X, y)
print("Logistic Regression Score:", model.score(X, y))