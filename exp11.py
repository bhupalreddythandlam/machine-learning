import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Mock data: Income, Age, Debt, Score(0=Bad, 1=Good)
df = pd.DataFrame({'Inc': [50, 20, 100], 'Age': [25, 45, 35], 'Label': [1, 0, 1]})
model = RandomForestClassifier().fit(df[['Inc', 'Age']], df['Label'])
print("Credit Prediction:", model.predict([[30, 40]]))