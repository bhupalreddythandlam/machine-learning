import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression


data = {'Income': ['High', 'Medium', 'Low', 'Medium'],
        'Debt': ['Low', 'Medium', 'High', 'Low'],
        'PaymentHistory': ['Good', 'Average', 'Poor', 'Good'],
        'Score': ['High', 'Medium', 'Low', 'High']}
df = pd.DataFrame(data)
le = LabelEncoder()
df_encoded = df.apply(le.fit_transform)
X, y = df_encoded.drop('Score', axis=1), df_encoded['Score']


logreg = LogisticRegression()
logreg.fit(X, y)

new_data = [[1, 2, 0]]
prediction = logreg.predict(new_data)
probs = logreg.predict_proba(new_data)

print(f"Logistic Regression Prediction: {le.inverse_transform(prediction)[0]}")
print(f"Class Probabilities: {probs}")