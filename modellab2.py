import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier


data = {'Income': ['High', 'Medium', 'Low', 'Medium'],
        'Debt': ['Low', 'Medium', 'High', 'Low'],
        'PaymentHistory': ['Good', 'Average', 'Poor', 'Good'],
        'Score': ['High', 'Medium', 'Low', 'High']}
df = pd.DataFrame(data)
le = LabelEncoder()
df_encoded = df.apply(le.fit_transform)
X, y = df_encoded.drop('Score', axis=1), df_encoded['Score']


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

new_data = [[1, 2, 0]] 
prediction = knn.predict(new_data)
print(f"KNN Prediction: {le.inverse_transform(prediction)[0]}")