from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd

# Sqft, Rooms -> Price
df = pd.DataFrame({'Sqft': [1500, 2500, 1800], 'Rooms': [3, 4, 3], 'Price': [300, 500, 350]})
model = GradientBoostingRegressor().fit(df[['Sqft', 'Rooms']], df['Price'])
print("House Price Prediction:", model.predict([[2000, 3]]))