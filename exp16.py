from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

X, y = load_iris(return_X_y=True)
print("LogReg Score:", LogisticRegression(max_iter=200).fit(X, y).score(X, y))
print("SVM Score:", SVC().fit(X, y).score(X, y))
print("Tree Score:", DecisionTreeClassifier().fit(X, y).score(X, y))