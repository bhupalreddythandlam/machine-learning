from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB

iris = load_iris()
model = GaussianNB().fit(iris.data, iris.target)
print("NB Iris Class:", iris.target_names[model.predict([[6.0, 3.0, 4.8, 1.8]])])