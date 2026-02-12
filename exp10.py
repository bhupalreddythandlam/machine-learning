from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.60)
model = GaussianMixture(n_components=3).fit(X)
print("Means found by EM:\n", model.means_)