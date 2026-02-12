import pandas as pd
from sklearn.linear_model import LinearRegression

# Year, Mileage -> Price
df = pd.DataFrame({'Year': [2015, 2018, 2020], 'Miles': [50000, 20000, 5000], 'Price': [15000, 22000, 28000]})
model = LinearRegression().fit(df[['Year', 'Miles']], df['Price'])
print("Predicted Car Price:", model.predict([[2019, 10000]]))