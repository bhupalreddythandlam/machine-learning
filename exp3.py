from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Mock dataset: Outlook, Temp, Humidity, Wind, Play
data = [['Sunny','Hot','High','Weak','No'], ['Overcast','Hot','High','Weak','Yes']]
df = pd.DataFrame(data, columns=['O','T','H','W','P'])
for col in df.columns: df[col] = LabelEncoder().fit_transform(df[col])

X, y = df.iloc[:, :-1], df.iloc[:, -1]
model = DecisionTreeClassifier(criterion='entropy').fit(X, y)
print("New Sample Prediction:", model.predict([[1, 0, 0, 1]])) # Predict for Overcast, Hot, High, Strong