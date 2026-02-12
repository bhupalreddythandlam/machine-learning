import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

X = np.array([1, 2, 3, 4]).reshape(-1, 1)
y = np.array([1, 4, 9, 16])

# Linear
lin = LinearRegression().fit(X, y)
# Polynomial (Degree 2)
poly_feat = PolynomialFeatures(degree=2)
X_poly = poly_feat.fit_transform(X)
poly = LinearRegression().fit(X_poly, y)

print("Linear R2:", lin.score(X, y))
print("Polynomial R2:", poly.score(X_poly, y))