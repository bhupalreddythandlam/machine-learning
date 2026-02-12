import numpy as np
from sklearn.linear_model import LinearRegression

# Days (1 to 5) -> Sales
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([100, 120, 150, 170, 200])
model = LinearRegression().fit(X, y)
print("Predicted Sales for Day 6:", model.predict([[6]]))