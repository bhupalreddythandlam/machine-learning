import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB


data = {'Income': ['High', 'Medium', 'Low', 'Medium'],
        'Debt': ['Low', 'Medium', 'High', 'Low'],
        'PaymentHistory': ['Good', 'Average', 'Poor', 'Good'],
        'Score': ['High', 'Medium', 'Low', 'High']}
df = pd.DataFrame(data)
le = LabelEncoder()
df_encoded = df.apply(le.fit_transform)
X, y = df_encoded.drop('Score', axis=1), df_encoded['Score']


nb = CategoricalNB()
nb.fit(X, y)

new_data = [[1, 2, 0]]
prediction = nb.predict(new_data)
probs = nb.predict_proba(new_data)

print(f"Naive Bayes Prediction: {le.inverse_transform(prediction)[0]}")
print(f"Probabilities: {probs}")