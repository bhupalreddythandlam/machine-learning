from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
model = KNeighborsClassifier(n_neighbors=5).fit(iris.data, iris.target)
print("Predicted Species:", iris.target_names[model.predict([[5.1, 3.5, 1.4, 0.2]])])