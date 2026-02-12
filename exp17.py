from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# RAM, Battery, InternalMem -> PriceRange(0-3)
df = pd.DataFrame({'RAM': [2, 4, 8], 'Batt': [3000, 4000, 5000], 'Price': [0, 1, 3]})
model = RandomForestClassifier().fit(df[['RAM', 'Batt']], df['Price'])
print("Mobile Price Category:", model.predict([[6, 4500]]))